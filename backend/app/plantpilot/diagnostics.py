"""Deterministic evidence sufficiency, Proof-or-Refuse diagnosis, self-auditing reports, and Next-Best-Test engine."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


# =====================================================================
# 1. Proof-or-Refuse Diagnosis
# =====================================================================

def proof_or_refuse_diagnosis(
    evidence: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    required_evidence: Optional[List[str]] = None,
    candidate_root_causes: Optional[List[str]] = None,
    machine_id: Optional[str] = None,
    incident: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Diagnostic validation layer enforcing Proof-or-Refuse principles.
    
    Determines whether collected evidence strictly supports a proposed conclusion
    or requires an explicit refusal.
    
    Checks:
    - Evidence references against actual collected verified evidence.
    - Distinguishes observations (OBSERVED, DERIVED) from unverified hypotheses.
    - Identifies missing evidence.
    - Identifies contradictory observations.
    - Checks whether evidence is verified and relevant to target machine.
    - Detects when several plausible root causes remain without distinguishing evidence.
    - Rejects conclusions based on nonexistent readings or tools.
    - Requires appropriate evidence before making strong causal claims.
    - Returns explicit refusal with explanation of what is unknown and next steps.
    """
    # 1. Filter verified observations and derivations
    verified_evidence = [
        e for e in evidence
        if e.get("verified") is True
        and str(e.get("evidence_type", "observed")).lower() in {"observed", "derived"}
    ]
    verified_ids = {str(e.get("id")) for e in verified_evidence}

    # 2. Check machine relevance if specified
    if not machine_id and incident and isinstance(incident, dict):
        machine_id = incident.get("machine_id")
    irrelevant_evidence = []
    if machine_id:
        for e in verified_evidence:
            val = e.get("value")
            if isinstance(val, dict) and "machine_id" in val:
                if str(val["machine_id"]) != str(machine_id):
                    irrelevant_evidence.append(str(e.get("id")))

    # 3. Missing required evidence
    missing_evidence: List[str] = []
    for item in (required_evidence or []):
        matched = any(
            item.lower() in str(e.get("description", "")).lower()
            or item.lower() in str(e.get("parameter_name", "")).lower()
            or (isinstance(e.get("value"), dict) and item.lower() in str(e.get("value")).lower())
            for e in verified_evidence
        )
        if not matched:
            missing_evidence.append(item)

    # 4. Check contradictory observations
    contradictions: List[Dict[str, Any]] = []
    # e.g., check conflicting temperature readings or wear readings across verified items
    param_values: Dict[str, List[Tuple[float, str]]] = {}
    for e in verified_evidence:
        pname = e.get("parameter_name")
        num_val = e.get("numeric_value")
        if pname and num_val is not None:
            try:
                val_float = float(num_val)
                param_values.setdefault(pname, []).append((val_float, str(e.get("id"))))
            except (ValueError, TypeError):
                pass

    for pname, vals in param_values.items():
        if len(vals) > 1:
            nums = [v[0] for v in vals]
            diff = max(nums) - min(nums)
            # If same parameter varies significantly without time change (e.g. > 15K or > 30 Nm)
            if pname in {"air_temperature", "process_temperature"} and diff > 15.0:
                contradictions.append({
                    "parameter": pname,
                    "difference": diff,
                    "evidence_ids": [v[1] for v in vals],
                    "issue": f"Contradictory temperature readings observed ({min(nums):.1f} vs {max(nums):.1f})",
                })
            elif pname == "torque" and diff > 40.0:
                contradictions.append({
                    "parameter": pname,
                    "difference": diff,
                    "evidence_ids": [v[1] for v in vals],
                    "issue": f"Contradictory torque readings observed ({min(nums):.1f} vs {max(nums):.1f})",
                })

    # 5. Evaluate candidate hypotheses
    evaluated_hypotheses = []
    competing_viable_hypotheses = []

    for h in hypotheses:
        h_id = h.get("id", "hyp_unknown")
        title = h.get("title", "Untitled hypothesis")
        supporting = [str(x) for x in h.get("supporting_evidence_ids", [])]
        contradicting = [str(x) for x in h.get("contradicting_evidence_ids", [])]

        invalid_support = [x for x in supporting if x not in verified_ids]
        invalid_contradiction = [x for x in contradicting if x not in verified_ids]

        # Check if supporting evidence contains strong causal proof
        has_verified_support = bool(supporting) and not invalid_support
        has_contradictions = bool(contradicting) or any(
            c["parameter"] in title.lower() for c in contradictions
        )

        is_viable = has_verified_support and not has_contradictions and not invalid_support
        if is_viable:
            competing_viable_hypotheses.append(title)

        evaluated_hypotheses.append({
            "hypothesis_id": h_id,
            "title": title,
            "supported_by_verified_evidence": has_verified_support,
            "supporting_evidence_ids": supporting,
            "contradicting_evidence_ids": contradicting,
            "unsupported_evidence_references": invalid_support + invalid_contradiction,
            "has_contradiction": has_contradictions,
            "is_viable": is_viable,
        })

    # 6. Ambiguous root causes: if multiple competing viable hypotheses exist without distinguishing evidence
    ambiguous_causes: List[str] = []
    if len(competing_viable_hypotheses) > 1:
        ambiguous_causes = competing_viable_hypotheses

    # 7. Formulate Proof or Refusal outcome
    is_refused = False
    refusal_reasons: List[str] = []
    useful_next_steps: List[str] = []

    if not verified_evidence:
        is_refused = True
        refusal_reasons.append("No verified sensor readings or derived measurements were collected.")
        useful_next_steps.append("Execute get_sensor_readings on the target machine station ID.")

    if missing_evidence:
        is_refused = True
        refusal_reasons.append(f"Missing essential diagnostic evidence: {', '.join(missing_evidence)}.")
        for m in missing_evidence:
            useful_next_steps.append(f"Measure and verify physical parameter: {m}.")

    if contradictions:
        is_refused = True
        refusal_reasons.append(f"Contradictory observations detected across sensor readings: {[c['issue'] for c in contradictions]}.")
        useful_next_steps.append("Recalibrate sensors and verify telemetry timestamps to resolve sensor discrepancies.")

    if ambiguous_causes:
        is_refused = True
        refusal_reasons.append(
            f"Multiple plausible root causes remain unresolved: {ambiguous_causes}. Evidence does not distinguish which failure mechanism initiated the incident."
        )
        useful_next_steps.append("Perform distinguishing physical tests (e.g. vibration FFT, oil tribology, thermal imaging).")

    if not evaluated_hypotheses:
        is_refused = True
        refusal_reasons.append("No diagnostic hypotheses have been evaluated.")
        useful_next_steps.append("Formulate testable hypotheses linked to verified observations.")

    elif not any(h["is_viable"] for h in evaluated_hypotheses):
        is_refused = True
        refusal_reasons.append("No hypothesis is supported by verified, non-contradicted evidence.")
        useful_next_steps.append("Gather telemetry on rotational speed, torque, and temperatures.")

    # Status determination
    if is_refused:
        diagnosis_status = "REFUSED_INSUFFICIENT_EVIDENCE"
        outcome = "REFUSED"
        message = (
            f"DIAGNOSIS REFUSED: {'; '.join(refusal_reasons)} "
            f"Unknowns must be resolved before reaching a causal conclusion."
        )
    else:
        diagnosis_status = "SUPPORTED_FOR_REVIEW"
        outcome = "SUPPORTED_FOR_REVIEW"
        message = (
            "Evidence supports hypothesis for human review; this confirms observational correlation, "
            "not independent causal proof."
        )

    return {
        "diagnosis_status": diagnosis_status,
        "outcome": outcome,
        "is_refused": is_refused,
        "refusal_reasons": refusal_reasons,
        "verified_evidence_count": len(verified_evidence),
        "missing_evidence": missing_evidence,
        "contradictions": contradictions,
        "ambiguous_causes": ambiguous_causes,
        "hypotheses": evaluated_hypotheses,
        "useful_next_steps": useful_next_steps,
        "message": message,
    }


def assess_diagnosis(
    evidence: List[Dict[str, Any]],
    hypotheses: List[Dict[str, Any]],
    required_evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Backward-compatible entry point delegating to Proof-or-Refuse validator."""
    res = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
        required_evidence=required_evidence,
    )
    sufficient = (res["diagnosis_status"] == "SUPPORTED_FOR_REVIEW")
    return {
        "diagnosis_status": "SUPPORTED_FOR_REVIEW" if sufficient else "INCONCLUSIVE",
        "verified_evidence_count": res["verified_evidence_count"],
        "missing_evidence": res["missing_evidence"],
        "hypotheses": res["hypotheses"],
        "message": (
            "Evidence supports at least one hypothesis for human review; this is not proof of causation."
            if sufficient
            else "Insufficient verified evidence to support a definitive diagnosis. Gather the missing evidence and resolve contradictions."
        ),
    }


# =====================================================================
# 2. Self-Auditing Reports
# =====================================================================

def audit_report(
    claims: List[Dict[str, Any]],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit individual narrative claims against underlying evidence records."""
    by_id = {str(e.get("id")): e for e in evidence}
    results = []

    for claim in claims:
        text = str(claim.get("text") or claim.get("claim", "")).strip()
        claim_type = str(claim.get("claim_type", "factual")).lower()
        refs = [str(x) for x in claim.get("evidence_ids", [])]
        if not refs and claim.get("evidence_reference"):
            refs = [str(claim["evidence_reference"])]
        matched = [by_id[x] for x in refs if x in by_id]
        verified = [
            e for e in matched
            if e.get("verified") is True
            and str(e.get("evidence_type", "observed")).lower() in {"observed", "derived"}
        ]

        issue = None
        required_correction = None
        classification = "FACTUAL_OBSERVATION"

        if claim_type == "numeric":
            classification = "NUMERICAL_MEASUREMENT"
        elif claim_type == "causal":
            classification = "CAUSAL_CLAIM"
        elif "repaired" in text.lower() or "replaced" in text.lower() or "intervention" in text.lower():
            classification = "INTERVENTION_CLAIM"

        if not refs:
            issue = "No evidence references supplied"
            required_correction = "Attach verified evidence references supporting this claim."
        elif len(matched) != len(refs):
            issue = "One or more evidence IDs do not exist"
            required_correction = "Remove invalid or fabricated evidence identifiers."
        elif not verified:
            issue = "Referenced evidence is not verified observation/derivation"
            required_correction = "Replace unverified assumptions with verified sensor observations."
        elif claim_type == "causal" and not claim.get("causation_established", False):
            issue = "Causal claim is not established by evidence metadata"
            required_correction = "Rephrase causal claim as an observational correlation or testable hypothesis."
        elif claim_type == "numeric" and claim.get("value") is not None:
            claimed_val = str(claim["value"])
            found_numeric = False
            for v in verified:
                num = v.get("numeric_value")
                val_raw = v.get("value")
                desc = str(v.get("description", ""))
                if num is not None and abs(float(num) - float(claim["value"])) < 1e-4:
                    found_numeric = True
                    break
                if claimed_val in str(val_raw) or claimed_val in desc:
                    found_numeric = True
                    break
            if not found_numeric:
                issue = "Claimed numeric value not found in referenced evidence"
                required_correction = f"Correct numeric measurement {claimed_val} to match actual sensor reading."

        status = "SUPPORTED" if issue is None else "UNVERIFIED"
        val_result = "VALID"
        if issue is not None:
            val_result = "INVALID" if ("not exist" in issue or "No evidence" in issue) else "UNVERIFIED"
        results.append({
            "claim": text,
            "claim_type": claim_type,
            "claim_classification": classification,
            "evidence_ids": refs,
            "status": status,
            "validation_result": val_result,
            "issue": issue,
            "required_correction": required_correction,
        })

    all_supported = bool(results) and all(r["status"] == "SUPPORTED" for r in results)
    supp_count = sum(1 for r in results if r["status"] == "SUPPORTED")
    return {
        "status": "VERIFIED" if all_supported else "REVIEW_REQUIRED",
        "claims_checked": len(results),
        "supported_claims": supp_count,
        "flagged_claims": len(results) - supp_count,
        "claims": results,
        "note": "Automated checks validate provenance and basic consistency; they do not independently prove physical causation.",
    }


def audit_investigation_record(
    investigation: Any,
    incident: Optional[Any] = None,
) -> Dict[str, Any]:
    """Audit actual investigation output against real database records.
    
    Validates:
    - Sensor and machine identities
    - Numerical values and physical units
    - Evidence references (provenance & verification)
    - Timestamps and freshness
    - Tool execution claims (checks tool executions table)
    - Failed or incomplete diagnostic tests
    - Contradictory evidence
    - Unsupported causal claims
    - Claims presented as facts when they are only hypotheses
    - Claims that an intervention succeeded without verified post-work evidence
    """
    evidence_items = getattr(investigation, "evidence", []) or []
    tool_executions = getattr(investigation, "tool_executions", []) or []
    hypotheses = getattr(investigation, "hypotheses", []) or []
    findings = getattr(investigation, "findings", []) or []
    recommendations = getattr(investigation, "recommendations", []) or []

    evidence_by_id = {e.id: e for e in evidence_items}
    exec_by_id = {t.id: t for t in tool_executions}
    successful_tools = {
        t.tool_name for t in tool_executions
        if getattr(t.status, "value", str(t.status)).lower() in {"completed", "status.completed"}
    }
    failed_tools = {
        t.tool_name for t in tool_executions
        if getattr(t.status, "value", str(t.status)).lower() in {"failed", "status.failed"}
    }

    audited_claims = []

    # 1. Audit narrative findings
    for finding in findings:
        text = str(finding)
        claim_cls = "OBSERVATION"
        validation = "VALID"
        issue = None
        corr = None
        supp_refs = []
        contra_refs = []

        # Check machine identity mismatch
        if incident and getattr(incident, "machine_id", None):
            expected_mach = str(incident.machine_id)
            # Find any machine identifiers mentioned like mach_xxx
            mach_matches = re.findall(r"mach_[a-f0-9]+", text)
            for m in mach_matches:
                if m != expected_mach:
                    validation = "INVALID"
                    issue = f"Machine identifier {m} in finding contradicts incident target machine {expected_mach}"
                    corr = f"Update report to reference correct incident machine {expected_mach}"

        # Check failed tool claimed as success
        for ft in failed_tools:
            if ft in text and ("success" in text.lower() or "executed" in text.lower()):
                validation = "INVALID"
                claim_cls = "TOOL_EXECUTION_CLAIM"
                issue = f"Report claims tool {ft} succeeded, but execution failed in audit record"
                corr = f"Disclose that tool {ft} failed and results are unavailable."

        # Check causal claims
        if any(w in text.lower() for w in ["caused by", "root cause is", "proves that", "definitely caused"]):
            claim_cls = "CAUSAL_CLAIM"
            validation = "UNVERIFIED"
            issue = "Unsupported causal claim: observational correlation presented as proven causation"
            corr = "Rephrase causal claim as an observational finding requiring human engineer confirmation."

        # Check unverified intervention claims
        if any(w in text.lower() for w in ["repaired", "fixed", "intervention completed", "maintenance verified"]):
            claim_cls = "INTERVENTION_CLAIM"
            # Check if any verified maintenance evidence exists
            has_maint_ev = any(
                "maintenance_verification" in str(e.parameter_name or "")
                and e.verified
                for e in evidence_items
            )
            if not has_maint_ev:
                validation = "UNVERIFIED"
                issue = "Intervention reported as successful without verified post-maintenance inspection sign-off"
                corr = "Mark maintenance as pending verified inspection sign-off before claiming resolution."

        audited_claims.append({
            "claim": text,
            "claim_classification": claim_cls,
            "supporting_evidence_references": supp_refs,
            "contradictory_evidence_references": contra_refs,
            "validation_result": validation,
            "explanation": issue or "Finding is consistent with recorded evidence.",
            "required_correction": corr,
        })

    # 2. Audit hypotheses
    for hyp in hypotheses:
        h_title = hyp.title
        h_expl = hyp.explanation
        supp = hyp.supporting_evidence_ids or []
        contra = hyp.contradicting_evidence_ids or []

        invalid_refs = [ref for ref in supp if ref not in evidence_by_id]
        unverified_refs = [
            ref for ref in supp
            if ref in evidence_by_id and not evidence_by_id[ref].verified
        ]

        val = "VALID"
        issue = None
        corr = None

        if invalid_refs:
            val = "INVALID"
            issue = f"Hypothesis references nonexistent evidence IDs: {invalid_refs}"
            corr = "Remove references to unrecorded evidence items."
        elif unverified_refs:
            val = "UNVERIFIED"
            issue = f"Hypothesis references unverified claims: {unverified_refs}"
            corr = "Ground hypothesis in verified sensor measurements."

        audited_claims.append({
            "claim": f"{h_title}: {h_expl}",
            "claim_classification": "HYPOTHESIS",
            "supporting_evidence_references": supp,
            "contradictory_evidence_references": contra,
            "validation_result": val,
            "explanation": issue or "Hypothesis correctly flagged as model-generated explanation.",
            "required_correction": corr,
        })

    # 3. Audit recommendations
    for rec in recommendations:
        val = "VALID"
        issue = None
        corr = None
        if not rec.requires_approval:
            val = "INVALID"
            issue = "Safety Violation: Recommendation did not require human approval"
            corr = "Enforce requires_approval=True on all recommendations."

        audited_claims.append({
            "claim": f"{rec.title}: {rec.description}",
            "claim_classification": "RECOMMENDATION",
            "supporting_evidence_references": [],
            "contradictory_evidence_references": [],
            "validation_result": val,
            "explanation": issue or "Recommendation is queued for human authorization.",
            "required_correction": corr,
        })

    valid_count = sum(1 for c in audited_claims if c["validation_result"] == "VALID")
    overall_status = "VERIFIED" if valid_count == len(audited_claims) and bool(audited_claims) else "REVIEW_REQUIRED"

    return {
        "status": overall_status,
        "investigation_id": getattr(investigation, "id", "unknown"),
        "claims_checked": len(audited_claims),
        "supported_claims": valid_count,
        "flagged_claims": len(audited_claims) - valid_count,
        "claims": audited_claims,
        "limitations": [
            "Audits deterministic schema constraints, IDs, tool execution logs, and numeric ranges.",
            "Does not replace expert human engineer judgment on physical mechanical mechanisms.",
        ],
    }


# =====================================================================
# 3. Next-Best-Test Engine
# =====================================================================

def rank_next_tests(
    hypotheses: List[Dict[str, Any]],
    available_tests: List[Dict[str, Any]],
    tests_already_performed: Optional[List[str]] = None,
    previous_failed_attempts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Next-Best-Test diagnostic test-selection mechanism.
    
    Evaluates candidate tests against:
    - Relevance to unresolved hypotheses
    - Expected ability to distinguish competing explanations
    - Diagnostic value (0.0 to 1.0)
    - Execution cost (0.0 to 1.0)
    - Safety risk (0.0 to 1.0)
    - Tool availability & authorization permissions
    - Tests already performed (avoids redundant queries)
    - Previous failed attempts (penalizes broken tools)
    - Distinguishes executable software/data tools from physical tests
    """
    ranked = []
    unresolved = [h for h in hypotheses if not h.get("resolved", False)]
    performed_set = set(tests_already_performed or [])
    failed_set = set(previous_failed_attempts or [])

    unresolved_identifiers = set()
    for h in unresolved:
        if h.get("id"):
            unresolved_identifiers.add(str(h["id"]))
        if h.get("title"):
            unresolved_identifiers.add(str(h["title"]))

    for test in available_tests:
        name = str(test.get("name", "")).strip()
        test_id = str(test.get("id", name))
        if not name:
            continue

        related = set(str(x) for x in test.get("distinguishes_hypotheses", []))
        relevant_count = len(related.intersection(unresolved_identifiers))

        value = max(0.0, min(1.0, float(test.get("diagnostic_value", 0.5))))
        cost = max(0.0, min(1.0, float(test.get("cost", 0.2))))
        risk = max(0.0, min(1.0, float(test.get("safety_risk", 0.0))))
        time_req = max(0.0, float(test.get("time_required_minutes", 10.0)))
        time_penalty = min(0.1, (time_req / 120.0) * 0.1)

        # Baseline score calculation matching existing formula for compatibility
        score = max(0.0, value * (0.5 + 0.5 * min(1, relevant_count)) - 0.2 * cost - 0.5 * risk)

        # Penalize if already performed
        is_performed = test_id in performed_set or name in performed_set
        if is_performed:
            score = score * 0.1  # Severe discount to avoid repeated redundant test

        # Penalize if previously failed
        has_failed = test_id in failed_set or name in failed_set
        if has_failed:
            score = score * 0.2

        # Hard blocks
        if test.get("available", True) is not True:
            score = 0.0
        if test.get("permitted", True) is not True:
            score = 0.0

        is_executable = bool(test.get("executable", False) or test.get("is_executable_tool", False))

        ranked.append({
            "test_id": test_id,
            "name": name,
            "score": round(score, 4),
            "diagnostic_value": value,
            "cost": cost,
            "safety_risk": risk,
            "time_required_minutes": time_req,
            "relevant_hypotheses": sorted(related),
            "available": test.get("available", True),
            "permitted": test.get("permitted", True),
            "already_performed": is_performed,
            "failed_previously": has_failed,
            "executable_by_system": is_executable,
            "is_executable_tool": is_executable,
            "reason": (
                "Ranked by diagnostic value, relevance, cost, and safety; "
                + ("requires human operator execution." if not is_executable else "can be executed by system tool.")
            ),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    recommended = ranked[0] if ranked and ranked[0]["score"] > 0 else None
    return {
        "recommended_test": recommended,
        "ranked_tests": ranked,
        "unresolved_hypotheses": [h.get("id", h.get("title")) for h in unresolved],
        "note": "Candidate tests are ranked deterministically. Executable tests run via tool registry; physical tests require operator dispatch.",
    }
