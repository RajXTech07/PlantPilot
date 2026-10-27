from plantpilot.diagnostics import audit_report, audit_investigation_record
from plantpilot.schemas import (
    Investigation,
    Incident,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Recommendation,
    ToolExecution,
    Status,
    ApprovalStatus,
)


def test_audit_report_claims_grounding():
    evidence = [
        {
            "id": "ev_01",
            "parameter_name": "rotational_speed",
            "numeric_value": 1500.0,
            "unit": "rpm",
            "verified": True,
        }
    ]

    # Valid claim matching verified evidence
    claims = [
        {
            "claim": "Rotational speed was observed at 1500.0 rpm.",
            "evidence_reference": "ev_01",
        }
    ]
    result = audit_report(claims=claims, evidence=evidence)
    assert result["status"] == "VERIFIED"
    assert result["supported_claims"] == 1
    assert result["flagged_claims"] == 0

    # Claim referencing non-existent evidence
    invalid_claims = [
        {
            "claim": "Torque exceeded 80 Nm.",
            "evidence_reference": "ev_nonexistent",
        }
    ]
    inv_result = audit_report(claims=invalid_claims, evidence=evidence)
    assert inv_result["status"] == "REVIEW_REQUIRED"
    assert inv_result["flagged_claims"] == 1
    assert inv_result["claims"][0]["validation_result"] == "INVALID"


def test_audit_investigation_record_catches_wrong_machine():
    incident = Incident(
        id="inc_001",
        machine_id="mach_alpha",
        description="Overheating reported",
        severity="high",
        category="mechanical",
        status=Status.PENDING,
        created_at="2026-10-27T10:00:00Z",
        updated_at="2026-10-27T10:00:00Z",
    )

    investigation = Investigation(
        id="inv_001",
        incident_id="inc_001",
        status=Status.COMPLETED,
        findings=[
            "Inspection on mach_beta revealed excessive thermal load.",  # mach_beta != mach_alpha
        ],
        evidence=[],
        hypotheses=[],
        recommendations=[],
        tool_executions=[],
        started_at="2026-10-27T10:01:00Z",
        completed_at="2026-10-27T10:05:00Z",
    )

    audit = audit_investigation_record(investigation=investigation, incident=incident)
    assert audit["status"] == "REVIEW_REQUIRED"
    assert audit["flagged_claims"] > 0
    flagged = [c for c in audit["claims"] if c["validation_result"] == "INVALID"]
    assert any("contradicts incident target machine mach_alpha" in f["explanation"] for f in flagged)


def test_audit_investigation_record_catches_failed_tool_claimed_as_success():
    incident = Incident(
        id="inc_002",
        machine_id="mach_01",
        description="Speed fluctuation",
        severity="medium",
        category="electrical",
        status=Status.PENDING,
        created_at="2026-10-27T10:00:00Z",
        updated_at="2026-10-27T10:00:00Z",
    )

    tool_exec = ToolExecution(
        id="exec_failed",
        tool_name="get_sensor_readings",
        arguments={"udi": 99999},
        result=None,
        status=Status.FAILED,
        error_message="UDI out of bounds",
        step_index=0,
    )

    investigation = Investigation(
        id="inv_002",
        incident_id="inc_002",
        status=Status.INCOMPLETE,
        findings=[
            "Executed get_sensor_readings with complete success and verified telemetry.",
        ],
        evidence=[],
        hypotheses=[],
        recommendations=[],
        tool_executions=[tool_exec],
        started_at="2026-10-27T10:01:00Z",
        completed_at="2026-10-27T10:02:00Z",
    )

    audit = audit_investigation_record(investigation=investigation, incident=incident)
    assert audit["status"] == "REVIEW_REQUIRED"
    tool_claims = [c for c in audit["claims"] if c["claim_classification"] == "TOOL_EXECUTION_CLAIM"]
    assert len(tool_claims) > 0
    assert tool_claims[0]["validation_result"] == "INVALID"
    assert "execution failed in audit record" in tool_claims[0]["explanation"]


def test_audit_investigation_record_flags_unsupported_causality():
    incident = Incident(
        id="inc_003",
        machine_id="mach_01",
        description="Vibration alert",
        severity="low",
        category="mechanical",
        status=Status.PENDING,
        created_at="2026-10-27T10:00:00Z",
        updated_at="2026-10-27T10:00:00Z",
    )

    investigation = Investigation(
        id="inv_003",
        incident_id="inc_003",
        status=Status.COMPLETED,
        findings=[
            "Vibration definitely caused by bearing spindle misalignment.",
        ],
        evidence=[],
        hypotheses=[],
        recommendations=[],
        tool_executions=[],
        started_at="2026-10-27T10:01:00Z",
        completed_at="2026-10-27T10:05:00Z",
    )

    audit = audit_investigation_record(investigation=investigation, incident=incident)
    causal_claims = [c for c in audit["claims"] if c["claim_classification"] == "CAUSAL_CLAIM"]
    assert len(causal_claims) > 0
    assert causal_claims[0]["validation_result"] == "UNVERIFIED"
    assert "Unsupported causal claim" in causal_claims[0]["explanation"]
