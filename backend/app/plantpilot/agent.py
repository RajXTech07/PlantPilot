"""Agent controller — orchestrates bounded investigation loops.

The agent loads an incident, constructs a prompt distinguishing facts from
hypotheses, invokes the configured model with registered tools, validates
and executes tool calls, and produces a structured investigation report.

All investigation state and audit events are persisted in SQLite.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI
from openai.types.chat import ChatCompletion

from .config import get_config
from .safety import default_safety_policy, SafetyPolicy
from .schemas import (
    ApprovalStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Incident,
    Investigation,
    InvestigationResult,
    InvestigationState,
    Recommendation,
    Status,
    ToolExecution,
)
from .storage import Database
from .tools import ToolRegistry, registered_tools, ToolError


class AgentError(RuntimeError):
    """Raised when the agent encounters an unrecoverable error."""


class AgentController:
    """Orchestrates AI investigations with safety controls."""

    def __init__(
        self,
        db: Database,
        tool_registry: ToolRegistry,
        safety_policy: SafetyPolicy,
    ) -> None:
        self.db = db
        self.tool_registry = tool_registry
        self.safety_policy = safety_policy
        self.config = get_config()

        self.client = OpenAI(
            base_url=self.config.nvidia_base_url,
            api_key=self.config.nvidia_api_key,
            timeout=self.config.api_timeout_seconds,
        )

    def investigate(
        self, investigation: Investigation, incident: Incident
    ) -> InvestigationResult:
        """Run a bounded investigation loop for the given incident.

        Enforces evidence-first workflow: the model must attempt diagnostic
        tools before concluding. If no tools are executed, the investigation
        is marked INCOMPLETE with an appropriate error message.
        """
        messages = self._build_initial_messages(incident)
        tools_openai = self.tool_registry.to_openai_tools()
        findings: List[str] = []
        evidence_ids: List[str] = []
        hypothesis_ids: List[str] = []
        recommendation_ids: List[str] = []
        errors: List[str] = []
        steps_taken = 0
        tool_call_count = 0
        tools_executed = False  # Track if any tool was successfully executed

        state = InvestigationState(
            investigation_id=investigation.id,
            incident_id=incident.id,
            target_machine_id=incident.machine_id,
            status=Status.IN_PROGRESS,
            initial_observations=[incident.description],
            budget={
                "max_steps": self.config.max_steps,
                "max_tool_calls": self.config.max_tool_calls,
                "steps_taken": 0,
                "tool_calls_taken": 0,
            },
        )
        executed_signatures: Set[str] = set()

        for step in range(self.config.max_steps):
            steps_taken = step + 1
            state.budget["steps_taken"] = steps_taken

            # Force tool call on first step to ensure evidence gathering
            tool_choice = "required" if step == 0 else "auto"

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_identifier,
                    messages=messages,  # type: ignore
                    tools=tools_openai,
                    tool_choice=tool_choice,
                    temperature=0.2,
                    max_tokens=self.config.max_output_tokens,
                )
            except Exception as exc:
                # Bounded retry: only for transient provider errors (500), not invalid args
                is_transient = "InternalServerError" in str(type(exc)) or "500" in str(exc) or "Failed to generate" in str(exc)
                if is_transient and steps_taken <= 2 and tool_call_count == 0:
                    # Retry once, then break on second failure
                    self.db.save_audit_event("provider_retry", f"Transient error at step {steps_taken}", {}, investigation.id)
                    continue  # retry loop once
                error_msg = f"API error: {type(exc).__name__}: {exc}"
                errors.append(error_msg)
                self.db.save_audit_event(
                    "api_error", error_msg, {"step": step}, investigation.id
                )
                break

            assistant_message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Preserve the assistant's response.
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": (
                        [tc.model_dump() for tc in assistant_message.tool_calls]
                        if assistant_message.tool_calls
                        else None
                    ),
                }
            )

            tool_calls = assistant_message.tool_calls or []

            # If no tool calls, the model has finished.
            if not tool_calls:
                if assistant_message.content:
                    findings.append(assistant_message.content[:self.config.max_message_chars])

                # Evidence-first: if no tools were executed, mark as INCOMPLETE
                if not tools_executed:
                    error_msg = "No diagnostic tools were executed. Investigation cannot complete without evidence."
                    errors.append(error_msg)
                    state.status = Status.INCOMPLETE
                    state.stopping_condition = "no_evidence_gathered"
                    self.db.save_audit_event(
                        "investigation_incomplete_no_evidence",
                        error_msg,
                        {"finish_reason": finish_reason, "steps": steps_taken},
                        investigation.id,
                    )
                    self.db.update_investigation(
                        investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        completed_at=datetime.now(timezone.utc),
                    )
                    return InvestigationResult(
                        investigation_id=investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        evidence_ids=evidence_ids,
                        hypothesis_ids=hypothesis_ids,
                        recommendation_ids=recommendation_ids,
                        errors=errors,
                        steps_taken=steps_taken,
                        completed=False,
                    )

                # Tools were executed - generate structured findings
                # Add a hypothesis based on evidence (model-generated claim)
                if findings and not any(findings):
                    pass  # findings already has content

                # Create a recommendation requiring human approval
                recommendation = Recommendation(
                    id=f"rec_{uuid.uuid4().hex[:12]}",
                    title="Review diagnostic results",
                    description="Investigate diagnostic results and determine root cause. Human approval required before any action.",
                    requires_approval=True,
                    approval_status=ApprovalStatus.PENDING_APPROVAL,
                    priority="high",
                )
                self.db.save_recommendation(investigation.id, recommendation)
                recommendation_ids.append(recommendation.id)
                state.recommendations.append(recommendation)

                # Add a model-generated hypothesis (explicitly unverified)
                if evidence_ids:
                    hypothesis = Hypothesis(
                        id=f"hyp_{uuid.uuid4().hex[:12]}",
                        title="Diagnostic evidence requires interpretation",
                        explanation="Evidence was obtained through diagnostic tools. Root cause determination requires further human review.",
                        confidence="low",
                        supporting_evidence_ids=evidence_ids,
                    )
                    self.db.save_hypothesis(investigation.id, hypothesis)
                    hypothesis_ids.append(hypothesis.id)
                    state.candidate_hypotheses.append(hypothesis)
                    state.supporting_evidence_map[hypothesis.id] = list(evidence_ids)

                    # Proof-or-Refuse Evidence Verification & Next-Test Ranking
                    try:
                        from .diagnostics import proof_or_refuse_diagnosis, rank_next_tests
                        loaded_inv = self.db.get_investigation(investigation.id)
                        if loaded_inv:
                            ev_dicts = [e.model_dump() for e in loaded_inv.evidence]
                            hyp_dicts = [h.model_dump() for h in loaded_inv.hypotheses]
                            proof_res = proof_or_refuse_diagnosis(
                                evidence=ev_dicts,
                                hypotheses=hyp_dicts,
                                machine_id=incident.machine_id,
                            )
                            diag_msg = f"Evidence Verification: {proof_res['diagnosis_status']} ({proof_res['verified_evidence_count']} verified items)"
                            findings.append(diag_msg)

                            if proof_res.get("is_refused"):
                                state.stopping_condition = "refused_insufficient_evidence"
                                state.limitations.extend(proof_res.get("refusal_reasons", []))
                                refusal_summary = "; ".join(proof_res.get("refusal_reasons", []))
                                findings.append(f"Diagnosis Refusal: {refusal_summary}")
                            else:
                                state.stopping_condition = "completed_supported_for_review"

                            candidate_tests = [
                                {"id": "test_vibration_fft", "name": "Spindle bearing vibration FFT scan", "diagnostic_value": 0.85, "cost": 0.2, "safety_risk": 0.05, "available": True, "permitted": True, "distinguishes_hypotheses": hypothesis_ids, "executable": False},
                                {"id": "test_oil_tribology", "name": "Lubrication oil tribology analysis", "diagnostic_value": 0.75, "cost": 0.3, "safety_risk": 0.0, "available": True, "permitted": True, "distinguishes_hypotheses": hypothesis_ids, "executable": False},
                                {"id": "test_thermal_imaging", "name": "Infrared thermographic hotspot scan", "diagnostic_value": 0.7, "cost": 0.1, "safety_risk": 0.0, "available": True, "permitted": True, "distinguishes_hypotheses": hypothesis_ids, "executable": False},
                            ]
                            performed_names = [t.get("tool_name", "") for t in state.diagnostic_tests_performed]
                            ranking = rank_next_tests(
                                hyp_dicts, candidate_tests, tests_already_performed=performed_names
                            )
                            if ranking.get("recommended_test"):
                                rec_test = ranking["recommended_test"]["name"]
                                findings.append(f"Recommended Next Inspection: {rec_test}")
                    except Exception as diag_err:
                        self.db.save_audit_event("diagnostic_assessment_error", str(diag_err), {}, investigation.id)

                # Normal completion
                state.status = Status.COMPLETED
                self.db.save_audit_event(
                    "investigation_complete",
                    f"Investigation finished at step {step + 1}",
                    {"finish_reason": finish_reason},
                    investigation.id,
                )
                self.db.update_investigation(
                    investigation.id,
                    status=Status.COMPLETED,
                    findings=findings,
                    completed_at=datetime.now(timezone.utc),
                )
                return InvestigationResult(
                    investigation_id=investigation.id,
                    status=Status.COMPLETED,
                    findings=findings,
                    evidence_ids=evidence_ids,
                    hypothesis_ids=hypothesis_ids,
                    recommendation_ids=recommendation_ids,
                    errors=errors,
                    steps_taken=steps_taken,
                    completed=True,
                )

            # Process tool calls.
            for tool_call in tool_calls:
                tool_call_count += 1
                state.budget["tool_calls_taken"] = tool_call_count
                if tool_call_count > self.config.max_tool_calls:
                    error_msg = f"Maximum tool calls ({self.config.max_tool_calls}) exceeded"
                    errors.append(error_msg)
                    state.status = Status.INCOMPLETE
                    state.stopping_condition = "max_tool_calls_exceeded"
                    self.db.save_audit_event(
                        "max_tool_calls_exceeded", error_msg, {}, investigation.id
                    )
                    self.db.update_investigation(
                        investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        completed_at=datetime.now(timezone.utc),
                    )
                    return InvestigationResult(
                        investigation_id=investigation.id,
                        status=Status.INCOMPLETE,
                        findings=findings,
                        evidence_ids=evidence_ids,
                        hypothesis_ids=hypothesis_ids,
                        recommendation_ids=recommendation_ids,
                        errors=errors,
                        steps_taken=steps_taken,
                        completed=False,
                    )

                tool_name = tool_call.function.name
                exec_id = f"exec_{uuid.uuid4().hex[:12]}"

                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    result_str = json.dumps({"error": "Invalid tool arguments"})
                    errors.append(f"Tool {tool_name}: invalid arguments")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        {},
                        result_str,
                        Status.FAILED,
                        f"Invalid JSON: {exc}",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )
                    continue

                # Deduplication check: avoid repeated calls that produce no new information
                call_sig = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                if call_sig in executed_signatures:
                    result_str = json.dumps({
                        "info": f"Tool '{tool_name}' was already executed with identical arguments. No new information gained."
                    })
                    state.diagnostic_tests_performed.append({
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "status": "DUPLICATE_SKIPPED",
                        "step_index": step,
                    })
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )
                    continue
                executed_signatures.add(call_sig)

                # Safety check.
                if not self.safety_policy.check_tool_call(tool_name, arguments):
                    result_str = json.dumps(
                        {"error": f"Tool {tool_name!r} not permitted"}
                    )
                    errors.append(f"Tool {tool_name}: safety check failed")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        "Safety policy rejected",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_str,
                        }
                    )
                    continue

                # Execute the tool.
                try:
                    result = self.tool_registry.execute(tool_name, arguments)
                    result_str = self._serialize_tool_result(result)
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.COMPLETED,
                        None,
                        step_index=step,
                    )
                    # Record evidence with exact provenance link.
                    # Machine-learning predictive calculations are tagged as DERIVED, while sensor readings are OBSERVED.
                    ev_type = (
                        EvidenceType.DERIVED
                        if tool_name == "predict_failure_risk"
                        else EvidenceType.OBSERVED
                    )
                    evidence = EvidenceItem(
                        id=f"ev_{uuid.uuid4().hex[:12]}",
                        evidence_type=ev_type,
                        description=f"Tool {tool_name} result",
                        source=tool_name,
                        value=result,
                        tool_execution_id=exec_id,
                        verified=True,
                    )
                    self.db.save_evidence(investigation.id, evidence)
                    evidence_ids.append(evidence.id)
                    tools_executed = True

                    state.collected_evidence.append(evidence)
                    state.verified_evidence.append(evidence)
                    state.diagnostic_tests_performed.append({
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "status": "COMPLETED",
                        "step_index": step,
                        "evidence_id": evidence.id,
                    })
                    self._revise_hypotheses_with_evidence(state, evidence, investigation.id)

                except ToolError as exc:
                    result_str = json.dumps({"error": str(exc)})
                    errors.append(f"Tool {tool_name}: {exc}")
                    # Do NOT set tools_executed = True for failed executions
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        str(exc),
                    )
                except Exception as exc:
                    result_str = json.dumps(
                        {"error": f"Tool execution failed: {type(exc).__name__}"}
                    )
                    errors.append(f"Tool {tool_name}: unexpected error")
                    self._record_tool_execution(
                        investigation.id,
                        exec_id,
                        tool_name,
                        arguments,
                        result_str,
                        Status.FAILED,
                        f"{type(exc).__name__}: {exc}",
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

        # Step limit reached without finishing.
        error_msg = f"Investigation did not complete within {self.config.max_steps} steps"
        errors.append(error_msg)
        self.db.save_audit_event(
            "max_steps_exceeded", error_msg, {}, investigation.id
        )
        self.db.update_investigation(
            investigation.id,
            status=Status.INCOMPLETE,
            findings=findings,
            completed_at=datetime.now(timezone.utc),
        )
        return InvestigationResult(
            investigation_id=investigation.id,
            status=Status.INCOMPLETE,
            findings=findings,
            evidence_ids=evidence_ids,
            hypothesis_ids=hypothesis_ids,
            recommendation_ids=recommendation_ids,
            errors=errors,
            steps_taken=steps_taken,
            completed=False,
        )

    def _build_initial_messages(self, incident: Incident) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are PlantPilot, a human-supervised industrial operations AI copilot. "
            "Investigate operational incidents using available tools. "
            "Never invent sensor readings or historical data. "
            "Distinguish verified observations from model-generated hypotheses. "
            "Explain uncertainty and recommend the next useful check. "
            "Do not claim a root cause is proven without sufficient evidence. "
            "Never initiate physical actions. All consequential recommendations "
            "require human approval."
        )
        user_prompt = (
            f"Incident: {incident.description}\n\n"
            f"Severity: {incident.severity}\n"
            f"Category: {incident.category or 'operational'}\n\n"
            "Investigate this incident. The dataset uses numeric UDI identifiers (e.g., 1, 2, 3). Retrieve the actual record by passing the numeric UDI to get_sensor_readings (e.g., station_id: 1). Then use diagnostic_math on the retrieved measurements. Do NOT pass the incident description text as a UDI. Always use a numeric identifier from AI4I 2020."
            "Identify plausible explanations and state what additional evidence is needed. "
            "Do not assume that an abnormal reading proves the root cause."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _serialize_tool_result(self, result: Any) -> str:
        """Serialize tool result to a size-limited JSON string."""
        try:
            serialized = json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(result)

        if len(serialized) > self.config.tool_result_max_chars:
            serialized = serialized[: self.config.tool_result_max_chars] + "...[truncated]"

        if len(serialized.encode("utf-8")) > self.config.tool_result_max_bytes:
            # Truncate by bytes.
            serialized_bytes = serialized.encode("utf-8")[
                : self.config.tool_result_max_bytes
            ]
            serialized = serialized_bytes.decode("utf-8", errors="ignore") + "...[truncated]"

        return serialized

    def _record_tool_execution(
        self,
        investigation_id: str,
        exec_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        status: Status,
        error_message: Optional[str],
        step_index: int = 0,
    ) -> None:
        """Persist a tool execution audit record."""
        execution = ToolExecution(
            id=exec_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            status=status,
            error_message=error_message,
            step_index=step_index,
            executed_at=datetime.now(timezone.utc),
            input_size_bytes=len(json.dumps(arguments).encode("utf-8")),
            output_size_bytes=len(result.encode("utf-8")),
        )
        self.db.save_tool_execution(investigation_id, execution)

    def _revise_hypotheses_with_evidence(
        self,
        state: InvestigationState,
        evidence: EvidenceItem,
        investigation_id: str,
    ) -> None:
        """Revise candidate hypotheses when new verified evidence arrives.

        Identifies whether new evidence supports or contradicts existing candidate
        hypotheses and updates confidence and evidence links in state and DB.
        """
        val = evidence.value
        if not isinstance(val, dict):
            return

        air_temp = val.get("air_temperature")
        proc_temp = val.get("process_temperature")
        wear = val.get("tool_wear")
        torque = val.get("torque")
        speed = val.get("rotational_speed")

        for hyp in state.candidate_hypotheses:
            title_lower = hyp.title.lower()

            # 1. Heat dissipation: if delta T >= 8.6 K (normal dissipation), contradicts heat dissipation failure
            if "heat" in title_lower and air_temp is not None and proc_temp is not None:
                try:
                    delta_t = float(proc_temp) - float(air_temp)
                    if delta_t >= 8.6:
                        if evidence.id not in hyp.contradicting_evidence_ids:
                            hyp.contradicting_evidence_ids.append(evidence.id)
                            state.contradicting_evidence_map.setdefault(hyp.id, []).append(evidence.id)
                            hyp.confidence = "low"
                            if self.db and getattr(self.db, "_conn", None) is not None:
                                try:
                                    self.db.save_hypothesis(investigation_id, hyp)
                                except Exception:
                                    pass
                except (ValueError, TypeError):
                    pass

            # 2. Tool wear: wear < 100 min contradicts tool wear failure; wear > 200 min supports it
            if "tool wear" in title_lower and wear is not None:
                try:
                    w = float(wear)
                    if w < 100.0:
                        if evidence.id not in hyp.contradicting_evidence_ids:
                            hyp.contradicting_evidence_ids.append(evidence.id)
                            state.contradicting_evidence_map.setdefault(hyp.id, []).append(evidence.id)
                            hyp.confidence = "low"
                            if self.db and getattr(self.db, "_conn", None) is not None:
                                try:
                                    self.db.save_hypothesis(investigation_id, hyp)
                                except Exception:
                                    pass
                    elif w > 200.0:
                        if evidence.id not in hyp.supporting_evidence_ids:
                            hyp.supporting_evidence_ids.append(evidence.id)
                            state.supporting_evidence_map.setdefault(hyp.id, []).append(evidence.id)
                            hyp.confidence = "high"
                            if self.db and getattr(self.db, "_conn", None) is not None:
                                try:
                                    self.db.save_hypothesis(investigation_id, hyp)
                                except Exception:
                                    pass
                except (ValueError, TypeError):
                    pass

            # 3. Power failure / overstrain: torque * speed bounds
            if "power" in title_lower and torque is not None and speed is not None:
                try:
                    power = float(torque) * (float(speed) * 2 * 3.14159 / 60)
                    if 3500 <= power <= 9000:
                        if evidence.id not in hyp.contradicting_evidence_ids:
                            hyp.contradicting_evidence_ids.append(evidence.id)
                            state.contradicting_evidence_map.setdefault(hyp.id, []).append(evidence.id)
                            hyp.confidence = "low"
                            if self.db and getattr(self.db, "_conn", None) is not None:
                                try:
                                    self.db.save_hypothesis(investigation_id, hyp)
                                except Exception:
                                    pass
                except (ValueError, TypeError):
                    pass


def create_agent_controller(db: Optional[Database] = None) -> AgentController:
    """Factory function to create an agent controller with defaults."""
    from .storage import get_database

    if db is None:
        db = get_database()
    return AgentController(
        db=db,
        tool_registry=registered_tools,
        safety_policy=default_safety_policy(),
    )
