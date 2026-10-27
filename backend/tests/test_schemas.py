"""Tests for Pydantic data models."""

import pytest
from datetime import datetime, timezone

from plantpilot.schemas import (
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    Recommendation,
    ApprovalStatus,
    ToolExecution,
    Status,
    IncidentCreate,
    Incident,
    InvestigationCreate,
    Investigation,
    InvestigationResult,
)


def test_evidence_item_creation():
    """Test creating an evidence item."""
    evidence = EvidenceItem(
        id="ev_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Temperature reading: 88°C",
        source="sensor_temp_01",
        value=88.0,
        verified=True,
    )
    assert evidence.id == "ev_1"
    assert evidence.evidence_type == EvidenceType.OBSERVED
    assert evidence.verified is True


def test_evidence_item_defaults():
    """Test evidence item defaults."""
    evidence = EvidenceItem(
        id="ev_2",
        description="Something happened",
        source="system",
    )
    assert evidence.evidence_type == EvidenceType.OBSERVED
    assert evidence.verified is False
    assert evidence.value is None
    assert isinstance(evidence.timestamp, datetime)


def test_hypothesis_creation():
    """Test creating a hypothesis."""
    hypothesis = Hypothesis(
        id="hyp_1",
        title="Motor overheating due to bearing wear",
        explanation="Elevated temperature correlates with reduced lubrication",
        confidence="medium",
        supporting_evidence_ids=["ev_1", "ev_2"],
        contradicting_evidence_ids=[],
    )
    assert hypothesis.id == "hyp_1"
    assert hypothesis.confidence == "medium"
    assert len(hypothesis.supporting_evidence_ids) == 2


def test_recommendation_creation():
    """Test creating a recommendation."""
    recommendation = Recommendation(
        id="rec_1",
        title="Inspect motor bearings",
        description="Schedule maintenance check for motor bearings within 24 hours",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="high",
    )
    assert recommendation.requires_approval is True
    assert recommendation.approval_status == ApprovalStatus.PENDING_APPROVAL
    assert recommendation.priority == "high"


def test_tool_execution_creation():
    """Test creating a tool execution record."""
    execution = ToolExecution(
        id="exec_1",
        tool_name="diagnostic_math",
        arguments={"values": [1, 2, 3]},
        result='{"mean": 2.0}',
        status=Status.COMPLETED,
    )
    assert execution.tool_name == "diagnostic_math"
    assert execution.status == Status.COMPLETED
    assert execution.error_message is None


def test_incident_create_validation():
    """Test incident creation validates required fields."""
    with pytest.raises(Exception):  # Pydantic validation error
        IncidentCreate(description="")  # Too short


def test_incident_create_with_metadata():
    """Test incident creation with metadata."""
    create = IncidentCreate(
        description="Production line stopped",
        severity="high",
        category="operational",
        metadata={"station": "A", "line": 3},
    )
    assert create.metadata["station"] == "A"
    assert create.metadata["line"] == 3


def test_investigation_creation():
    """Test investigation creation."""
    investigation = Investigation(
        id="inv_1",
        incident_id="inc_1",
        status=Status.IN_PROGRESS,
        started_at=datetime.now(timezone.utc),
    )
    assert investigation.status == Status.IN_PROGRESS
    assert len(investigation.findings) == 0
    assert len(investigation.evidence) == 0
    assert len(investigation.tool_executions) == 0


def test_investigation_result_creation():
    """Test investigation result."""
    result = InvestigationResult(
        investigation_id="inv_1",
        status=Status.COMPLETED,
        findings=["Finding 1", "Finding 2"],
        evidence_ids=["ev_1", "ev_2"],
        hypothesis_ids=["hyp_1"],
        recommendation_ids=["rec_1"],
        errors=[],
        steps_taken=3,
        completed=True,
    )
    assert result.completed is True
    assert result.steps_taken == 3
    assert len(result.findings) == 2


def test_digital_twin_schemas():
    """Test DigitalTwin and TwinReading schemas."""
    from plantpilot.schemas import DigitalTwinCreate, DigitalTwin, TwinReadingItem, TwinReadingCreate

    twin_payload = DigitalTwinCreate(name="Milling Unit A", machine_type="M", metadata={"line": 2})
    assert twin_payload.name == "Milling Unit A"
    assert twin_payload.machine_type == "M"

    twin = DigitalTwin(
        id="twin_123",
        name="Milling Unit A",
        machine_type="M",
        metadata={"line": 2},
        created_at="2026-10-26T12:00:00Z",
        updated_at="2026-10-26T12:00:00Z",
        latest_observed_reading=None,
        observed_reading_count=0,
    )
    assert twin.id == "twin_123"
    assert twin.observed_reading_count == 0


def test_failure_risk_prediction_schema():
    """Test FailureRiskPrediction schema."""
    from plantpilot.schemas import FailureRiskPrediction, AnomalyEvidence, SecondaryModelResult

    pred = FailureRiskPrediction(
        model="HistGradientBoostingClassifier",
        failure_probability=0.85,
        threshold=0.42,
        failure_risk="elevated",
        predicted_failure=True,
        anomaly_evidence=AnomalyEvidence(
            flagged_parameters=["rotational_speed_rpm"],
            detection_methods={"rotational_speed_rpm": "iqr_lower"},
            explanation="Unusual rotational speed detected",
        ),
        secondary_model=SecondaryModelResult(
            model="RandomForestClassifier",
            failure_probability=0.81,
            threshold=0.40,
            predicted_failure=True,
        ),
        model_agreement=True,
        provenance="AI4I 2020 benchmark dataset",
    )
    assert pred.predicted_failure is True
    assert pred.model_agreement is True
    assert pred.failure_risk == "elevated"


def test_diagnostics_schemas():
    """Test diagnostics request and response schemas."""
    from plantpilot.schemas import (
        DiagnosisAssessRequest,
        DiagnosisAssessResponse,
        ReportAuditRequest,
        ReportAuditResponse,
        RankTestsRequest,
        RankTestsResponse,
    )

    assess_req = DiagnosisAssessRequest(
        evidence=[{"id": "ev_1", "verified": True}],
        hypotheses=[{"id": "hyp_1", "supporting_evidence_ids": ["ev_1"]}],
        required_evidence=["air_temp"],
    )
    assert len(assess_req.evidence) == 1
    assert assess_req.required_evidence == ["air_temp"]

    assess_res = DiagnosisAssessResponse(
        diagnosis_status="SUPPORTED_FOR_REVIEW",
        verified_evidence_count=1,
        missing_evidence=[],
        hypotheses=[],
        message="OK",
    )
    assert assess_res.diagnosis_status == "SUPPORTED_FOR_REVIEW"

    audit_req = ReportAuditRequest(
        claims=[{"text": "Claim", "claim_type": "factual", "evidence_ids": ["ev_1"]}],
        evidence=[{"id": "ev_1", "verified": True}],
    )
    assert len(audit_req.claims) == 1

    rank_req = RankTestsRequest(
        hypotheses=[{"id": "hyp_1"}],
        available_tests=[{"id": "test_1", "name": "Inspection", "diagnostic_value": 0.8}],
    )
    assert len(rank_req.available_tests) == 1


def test_workforce_schemas():
    """Test workforce worker and attendance schemas."""
    from plantpilot.schemas import (
        WorkerCreate,
        WorkerItem,
        ClockInRequest,
        ClockOutRequest,
        AttendanceRecordItem,
        ClearanceCheckRequest,
        ClearanceCheckResponse,
    )

    worker_payload = WorkerCreate(
        name="Elena Rostova",
        role="Diagnostics Lead",
        department="Maintenance",
        clearance_level="L3",
    )
    assert worker_payload.name == "Elena Rostova"
    assert worker_payload.clearance_level == "L3"
    assert worker_payload.status == "ACTIVE"

    worker = WorkerItem(
        worker_id="wrk_123",
        name="Elena Rostova",
        role="Diagnostics Lead",
        department="Maintenance",
        clearance_level="L3",
        status="ACTIVE",
        created_at="2026-10-27T00:00:00Z",
    )
    assert worker.worker_id == "wrk_123"

    clock_in = ClockInRequest(
        worker_id="wrk_123",
        timestamp="2026-10-27T08:00:00Z",
        shift_date="2026-10-27",
        break_minutes=30,
    )
    assert clock_in.break_minutes == 30

    check_res = ClearanceCheckResponse(
        worker_id="wrk_123",
        actual_clearance="L3",
        required_clearance="L2",
        authorized_by_clearance=True,
        status="AUTHORIZED",
        requires_human_review=False,
    )
    assert check_res.authorized_by_clearance is True


