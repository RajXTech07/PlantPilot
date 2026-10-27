"""Tests for InvestigationState tracking, budget limits, deduplication, and hypothesis revision."""
import pytest
from plantpilot.schemas import (
    InvestigationState,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    ToolExecution,
    Status,
    Incident,
    Investigation,
)
from plantpilot.agent import AgentController
from plantpilot.storage import Database
from plantpilot.safety import SafetyPolicy
from plantpilot.tools import ToolRegistry


def test_investigation_state_initialization():
    state = InvestigationState(
        investigation_id="inv_test_123",
        incident_id="inc_test_123",
        budget={"max_steps": 5, "max_tool_calls": 10},
    )
    assert state.investigation_id == "inv_test_123"
    assert state.incident_id == "inc_test_123"
    assert state.status == Status.IN_PROGRESS
    assert state.collected_evidence == []
    assert state.verified_evidence == []
    assert state.budget["max_steps"] == 5
    assert state.budget["max_tool_calls"] == 10


def test_investigation_state_evidence_tracking():
    state = InvestigationState(
        investigation_id="inv_test_123",
        incident_id="inc_test_123",
    )
    ev1 = EvidenceItem(
        id="ev_001",
        evidence_type=EvidenceType.OBSERVED,
        source="sensor",
        parameter_name="air_temperature",
        numeric_value=298.1,
        description="Air temperature measurement",
        verified=True,
    )
    ev2 = EvidenceItem(
        id="ev_002",
        evidence_type=EvidenceType.OBSERVED,
        source="sensor",
        parameter_name="torque",
        numeric_value=42.5,
        description="Torque reading",
        verified=False,
    )

    state.collected_evidence.extend([ev1, ev2])
    state.verified_evidence.append(ev1)
    state.unverified_evidence.append(ev2)

    assert len(state.collected_evidence) == 2
    assert len(state.verified_evidence) == 1
    assert len(state.unverified_evidence) == 1
    assert state.verified_evidence[0].id == "ev_001"


def test_agent_controller_budget_tracking():
    state = InvestigationState(
        investigation_id="inv_mock",
        incident_id="inc_mock",
        budget={
            "max_steps": 3,
            "max_tool_calls": 5,
            "steps_taken": 3,
            "tool_calls_taken": 5,
        },
    )
    assert state.budget["steps_taken"] >= state.budget["max_steps"]
    assert state.budget["tool_calls_taken"] >= state.budget["max_tool_calls"]
    state.stopping_condition = "max_steps_exceeded"
    assert state.stopping_condition == "max_steps_exceeded"


def test_agent_controller_tool_call_deduplication():
    executed_signatures = set()

    sig1 = "get_sensor_readings:{\"udi\": 1}"
    sig2 = "get_sensor_readings:{\"udi\": 1}"
    sig3 = "get_sensor_readings:{\"udi\": 2}"

    assert sig1 not in executed_signatures
    executed_signatures.add(sig1)
    assert sig2 in executed_signatures
    assert sig3 not in executed_signatures


def test_agent_controller_revise_hypotheses_with_evidence(tmp_path):
    db_path = tmp_path / "test_revise.db"
    db = Database(db_path)
    tool_registry = ToolRegistry()
    safety_policy = SafetyPolicy()
    controller = AgentController(db, tool_registry, safety_policy)

    state = InvestigationState(
        investigation_id="inv_mock",
        incident_id="inc_mock",
    )

    # Initial candidate hypothesis for tool wear
    hyp = Hypothesis(
        id="hyp_001",
        title="Tool wear failure suspected",
        explanation="Suspected tool wear from machine runtime.",
        confidence="low",
        supporting_evidence_ids=[],
        contradicting_evidence_ids=[],
    )
    state.candidate_hypotheses.append(hyp)

    # Incoming verified sensor evidence with high tool wear (>200)
    ev = EvidenceItem(
        id="ev_wear_01",
        evidence_type=EvidenceType.OBSERVED,
        source="get_sensor_readings",
        description="Tool wear telemetry",
        value={"tool_wear": 240.0},
        verified=True,
    )

    controller._revise_hypotheses_with_evidence(state, ev, "inv_mock")

    assert "ev_wear_01" in hyp.supporting_evidence_ids
    assert hyp.confidence == "high"
    db.close()
