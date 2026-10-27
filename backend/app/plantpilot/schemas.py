"""Pydantic data models for PlantPilot."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class EvidenceType(str, Enum):
    OBSERVED = "observed"           # verified measurement / reading
    DERIVED = "derived"             # calculation from observed values
    HYPOTHESIS = "hypothesis"       # model-generated claim (unverified)
    SIMULATED = "simulated"         # hypothetical / projected outcome


class ApprovalStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class Machine(BaseModel):
    """Manufacturing asset / equipment."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    product_id: str = Field(..., description="Matches AI4I 2020 dataset Product ID")
    type: str = Field(..., description="L, M, or H type")
    status: str = Field("active", description="active / maintenance / degraded / offline")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class MachineCreate(BaseModel):
    """Payload to register an industrial asset."""
    model_config = ConfigDict(populate_by_name=True)

    product_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, max_length=10)
    id: Optional[str] = None
    status: Optional[str] = "active"


class EvidenceItem(BaseModel):
    """A single piece of evidence with explicit provenance."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., description="Stable evidence identifier")
    evidence_type: EvidenceType = Field(
        EvidenceType.OBSERVED, description="Provenance category"
    )
    description: str = Field(
        ..., min_length=1, max_length=2000, description="What was observed or claimed"
    )
    source: str = Field(
        ..., min_length=1, max_length=500, description="Tool / sensor / source"
    )
    value: Optional[Any] = Field(None, description="Numeric or structured value")
    tool_execution_id: Optional[str] = Field(
        None, description="Exact FK to tool_executions.id that produced this observation"
    )
    parameter_name: Optional[str] = Field(
        None, description="Physical parameter (e.g. air_temperature, torque)"
    )
    numeric_value: Optional[float] = Field(
        None, description="Typed numeric measurement"
    )
    unit: Optional[str] = Field(
        None, description="Unit of measurement (K, Nm, rpm, min)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of observation or claim",
    )
    verified: bool = Field(
        False, description="True only when independently verified, not model-only"
    )


class Hypothesis(BaseModel):
    """A model-generated explanation — explicitly NOT verified evidence."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    explanation: str = Field(..., min_length=1, max_length=4000)
    confidence: Optional[str] = Field(
        None, description="e.g. low / medium / high — model's own assessment"
    )
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Recommendation(BaseModel):
    """A proposal for human review — never an executed action."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=4000)
    requires_approval: bool = Field(
        True, description="Always True for consequential recommendations"
    )
    approval_status: ApprovalStatus = Field(
        ApprovalStatus.PENDING_APPROVAL
    )
    priority: str = Field("medium", description="low / medium / high")
    reviewed_by: Optional[str] = Field(
        None, description="Operator or engineer who reviewed this recommendation"
    )
    review_notes: Optional[str] = Field(
        None, max_length=2000, description="Operator comments or rejection reasoning"
    )
    reviewed_at: Optional[datetime] = Field(
        None, description="UTC timestamp when the review decision occurred"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    executed_at: Optional[datetime] = Field(
        None, description="Only set when approved and executed (not here)"
    )


class ToolExecution(BaseModel):
    """Audit record of a registered tool execution."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = Field(
        None, max_length=4000, description="String representation, size-limited"
    )
    status: Status = Field(Status.COMPLETED)
    error_message: Optional[str] = Field(None, max_length=1000)
    step_index: int = Field(0, ge=0, description="Bounded loop step index (0-based)")
    executed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    input_size_bytes: int = Field(0, ge=0)
    output_size_bytes: int = Field(0, ge=0)


class IncidentCreate(BaseModel):
    """Payload to create a new incident."""
    model_config = ConfigDict(populate_by_name=True)

    description: str = Field(..., min_length=1, max_length=5000)
    severity: str = Field("medium", description="low / medium / high / critical")
    category: Optional[str] = Field("operational", max_length=200)
    machine_id: Optional[str] = Field(None, description="Optional asset/machine ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Incident(BaseModel):
    """Persisted incident."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    description: str
    severity: str
    category: Optional[str] = None
    machine_id: Optional[str] = None
    status: Status = Field(Status.PENDING)
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvestigationCreate(BaseModel):
    """Payload to start an investigation for an incident."""
    model_config = ConfigDict(populate_by_name=True)

    incident_id: str = Field(..., min_length=1)
    notes: Optional[str] = Field(None, max_length=2000)


class Investigation(BaseModel):
    """A completed or in-progress investigation."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    incident_id: str = Field(...)
    status: Status = Field(Status.IN_PROGRESS)
    findings: List[str] = Field(default_factory=list,
                                 description="Structured findings (text or JSON)")
    evidence: List[EvidenceItem] = Field(default_factory=list)
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    tool_executions: List[ToolExecution] = Field(default_factory=list,
                                                 description="Audit trail of all tool calls")
    recommendations: List[Recommendation] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list,
                               description="Explicit errors or failures")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    notes: Optional[str] = Field(None, max_length=2000)


class InvestigationResult(BaseModel):
    """Structured result returned by the agent controller."""
    model_config = ConfigDict(populate_by_name=True)

    investigation_id: str = Field(...)
    status: Status = Field(...)
    findings: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    hypothesis_ids: List[str] = Field(default_factory=list)
    recommendation_ids: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    steps_taken: int = Field(0, ge=0)
    completed: bool = Field(False)


class InvestigationState(BaseModel):
    """Structured in-memory investigation state tracking budget, evidence, and hypotheses."""
    model_config = ConfigDict(populate_by_name=True)

    investigation_id: str = Field(...)
    incident_id: str = Field(...)
    target_machine_id: Optional[str] = Field(None)
    status: Status = Field(Status.IN_PROGRESS)
    initial_observations: List[str] = Field(default_factory=list)
    collected_evidence: List[EvidenceItem] = Field(default_factory=list)
    verified_evidence: List[EvidenceItem] = Field(default_factory=list)
    unverified_evidence: List[EvidenceItem] = Field(default_factory=list)
    candidate_hypotheses: List[Hypothesis] = Field(default_factory=list)
    supporting_evidence_map: Dict[str, List[str]] = Field(default_factory=dict)
    contradicting_evidence_map: Dict[str, List[str]] = Field(default_factory=dict)
    unresolved_questions: List[str] = Field(default_factory=list)
    diagnostic_tests_performed: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[Recommendation] = Field(default_factory=list)
    stopping_condition: Optional[str] = Field(None)
    budget: Dict[str, int] = Field(default_factory=dict)
    final_conclusions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class DigitalTwinCreate(BaseModel):
    """Payload to register a software digital twin."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=200, description="Twin identifier name")
    machine_type: str = Field(..., min_length=1, max_length=50, description="Machine type (L, M, H, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata attributes")


class TwinReadingItem(BaseModel):
    """A recorded telemetry snapshot on a digital twin."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    twin_id: str = Field(...)
    reading: Dict[str, Any] = Field(...)
    assessment: Dict[str, Any] = Field(default_factory=dict)
    evidence_type: str = Field("observed", description="observed, simulated, or derived")
    created_at: str = Field(...)


class DigitalTwin(BaseModel):
    """Digital twin representation of a machine asset."""
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(...)
    name: str = Field(...)
    machine_type: str = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(...)
    updated_at: str = Field(...)
    latest_observed_reading: Optional[TwinReadingItem] = Field(None)
    observed_reading_count: int = Field(0, ge=0)


class TwinReadingCreate(BaseModel):
    """Payload to push telemetry onto a digital twin."""
    model_config = ConfigDict(populate_by_name=True)

    reading: Dict[str, Any] = Field(..., description="Measurement map (e.g. air_temperature_k, rotational_speed_rpm)")
    assessment: Optional[Dict[str, Any]] = Field(None, description="Optional pre-computed risk assessment")
    evidence_type: Optional[str] = Field("observed", description="observed vs simulated")


class TwinSimulationRequest(BaseModel):
    """Payload to run what-if simulation on a digital twin."""
    model_config = ConfigDict(populate_by_name=True)

    reading: Dict[str, Any] = Field(..., description="Hypothetical operational readings")
    include_secondary: bool = Field(True, description="Whether to include secondary Random Forest consensus")


class AnomalyEvidence(BaseModel):
    """Statistical anomaly detection signals."""
    model_config = ConfigDict(populate_by_name=True)

    flagged_parameters: List[str] = Field(default_factory=list)
    detection_methods: Dict[str, str] = Field(default_factory=dict)
    explanation: str = Field(...)


class SecondaryModelResult(BaseModel):
    """Secondary consensus model prediction."""
    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(...)
    failure_probability: float = Field(...)
    threshold: float = Field(...)
    predicted_failure: bool = Field(...)


class FailureRiskPrediction(BaseModel):
    """ML failure risk prediction and statistical anomaly assessment."""
    model_config = ConfigDict(populate_by_name=True)

    model: str = Field(...)
    failure_probability: float = Field(...)
    threshold: float = Field(...)
    failure_risk: str = Field(...)
    predicted_failure: bool = Field(...)
    anomaly_evidence: AnomalyEvidence = Field(...)
    secondary_model: Optional[SecondaryModelResult] = Field(None)
    model_agreement: Optional[bool] = Field(None)
    provenance: str = Field(...)


# ---------------------------------------------------------
# Diagnostics Schemas
# ---------------------------------------------------------

class DiagnosisAssessRequest(BaseModel):
    """Payload to assess evidence sufficiency for hypotheses."""
    model_config = ConfigDict(populate_by_name=True)

    evidence: List[Dict[str, Any]] = Field(..., description="Observed and derived evidence items")
    hypotheses: List[Dict[str, Any]] = Field(..., description="Hypotheses to evaluate against evidence")
    required_evidence: Optional[List[str]] = Field(None, description="Optional required evidence keywords")


class DiagnosisAssessResponse(BaseModel):
    """Result of deterministic evidence sufficiency assessment."""
    model_config = ConfigDict(populate_by_name=True)

    diagnosis_status: str = Field(..., description="SUPPORTED_FOR_REVIEW or INCONCLUSIVE")
    verified_evidence_count: int = Field(..., ge=0)
    missing_evidence: List[str] = Field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    message: str = Field(...)


class ReportAuditRequest(BaseModel):
    """Payload to audit investigation narrative claims against evidence."""
    model_config = ConfigDict(populate_by_name=True)

    claims: List[Dict[str, Any]] = Field(..., description="Claims to audit")
    evidence: List[Dict[str, Any]] = Field(..., description="Underlying verified evidence items")


class ReportAuditResponse(BaseModel):
    """Result of report claim auditing against evidence records."""
    model_config = ConfigDict(populate_by_name=True)

    status: str = Field(..., description="VERIFIED or REVIEW_REQUIRED")
    claims_checked: int = Field(..., ge=0)
    supported_claims: int = Field(..., ge=0)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    note: str = Field(...)


class RankTestsRequest(BaseModel):
    """Payload to rank next diagnostic or physical candidate tests."""
    model_config = ConfigDict(populate_by_name=True)

    hypotheses: List[Dict[str, Any]] = Field(..., description="Current unresolved hypotheses")
    available_tests: List[Dict[str, Any]] = Field(..., description="Candidate tests to evaluate")


class RankTestsResponse(BaseModel):
    """Result of multi-criteria diagnostic test ranking."""
    model_config = ConfigDict(populate_by_name=True)

    recommended_test: Optional[Dict[str, Any]] = Field(None)
    ranked_tests: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_hypotheses: List[Any] = Field(default_factory=list)
    note: str = Field(...)


# ---------------------------------------------------------
# Workforce & Clearance Schemas
# ---------------------------------------------------------

class WorkerCreate(BaseModel):
    """Payload to register an industrial worker or technician."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, max_length=200)
    role: str = Field(..., min_length=1, max_length=100)
    department: str = Field(..., min_length=1, max_length=100)
    clearance_level: str = Field(..., description="L1, L2, L3, or L4")
    worker_id: Optional[str] = Field(None, max_length=50)
    status: Optional[str] = Field("ACTIVE", description="ACTIVE or INACTIVE")


class WorkerItem(BaseModel):
    """Industrial personnel record."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str = Field(...)
    name: str = Field(...)
    role: str = Field(...)
    department: str = Field(...)
    clearance_level: str = Field(...)
    status: str = Field(...)
    created_at: str = Field(...)


class ClockInRequest(BaseModel):
    """Payload to record worker shift start."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str = Field(...)
    timestamp: str = Field(..., description="ISO-8601 clock-in timestamp")
    shift_date: str = Field(..., description="YYYY-MM-DD shift date")
    break_minutes: Optional[int] = Field(0, ge=0, le=1440)


class ClockOutRequest(BaseModel):
    """Payload to record worker shift end."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str = Field(...)
    timestamp: str = Field(..., description="ISO-8601 clock-out timestamp")


class AttendanceRecordItem(BaseModel):
    """Shift attendance record."""
    model_config = ConfigDict(populate_by_name=True)

    record_id: str = Field(...)
    worker_id: str = Field(...)
    shift_date: str = Field(...)
    clock_in: str = Field(...)
    clock_out: Optional[str] = Field(None)
    break_minutes: int = Field(0, ge=0)
    working_hours: Optional[float] = Field(None)
    attendance_status: str = Field(..., description="OPEN or COMPLETE")
    created_at: str = Field(...)


class ClearanceCheckRequest(BaseModel):
    """Payload to verify worker operational clearance level."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str = Field(...)
    required_level: str = Field(..., description="Required clearance: L1, L2, L3, or L4")


class ClearanceCheckResponse(BaseModel):
    """Result of clearance authorization check."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str = Field(...)
    actual_clearance: str = Field(...)
    required_clearance: str = Field(...)
    authorized_by_clearance: bool = Field(...)
    status: str = Field(...)
    requires_human_review: bool = Field(...)


class WorkforceAnomaliesRequest(BaseModel):
    """Payload to trigger deterministic workforce anomaly detection."""
    model_config = ConfigDict(populate_by_name=True)

    max_shift_hours: Optional[float] = Field(12.0, ge=1.0, le=24.0)


class WorkforceFindingsResponse(BaseModel):
    """Deterministic workforce review findings."""
    model_config = ConfigDict(populate_by_name=True)

    finding_count: int = Field(..., ge=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_mode: str = Field(...)
    note: str = Field(...)


class WorkforceReviewResponse(BaseModel):
    """Model-assisted workforce review response."""
    model_config = ConfigDict(populate_by_name=True)

    analysis_mode: str = Field(...)
    deterministic_findings: Dict[str, Any] = Field(...)
    model_review: Dict[str, Any] = Field(...)


class ProofOrRefuseRequest(BaseModel):
    """Payload to evaluate evidence sufficiency under Proof-or-Refuse rules."""
    model_config = ConfigDict(populate_by_name=True)

    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    incident: Optional[Dict[str, Any]] = None
    required_evidence: Optional[List[str]] = None


class ProofOrRefuseResponse(BaseModel):
    """Deterministic proof-or-refuse outcome and next steps."""
    model_config = ConfigDict(populate_by_name=True)

    diagnosis_status: str
    outcome: str
    is_refused: bool
    refusal_reasons: List[str] = Field(default_factory=list)
    verified_evidence_count: int = 0
    missing_evidence: List[str] = Field(default_factory=list)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    ambiguous_causes: List[str] = Field(default_factory=list)
    evaluated_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    useful_next_steps: List[str] = Field(default_factory=list)
    message: str


class InvestigationAuditResponse(BaseModel):
    """Report audit results for an existing investigation record."""
    model_config = ConfigDict(populate_by_name=True)

    status: str
    investigation_id: str
    claims_checked: int
    supported_claims: int
    flagged_claims: int
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ShiftScheduleCreate(BaseModel):
    """Payload to register a planned shift schedule."""
    model_config = ConfigDict(populate_by_name=True)

    shift_name: str
    shift_date: str
    start_time: str
    end_time: str
    machine_id: Optional[str] = None
    department: str
    required_roles: List[str] = Field(default_factory=list)
    min_headcount: int = 1
    min_clearance: str = "L1"
    assigned_worker_ids: List[str] = Field(default_factory=list)


class ShiftHandoverCreate(BaseModel):
    """Payload to log a shift-to-shift handover."""
    model_config = ConfigDict(populate_by_name=True)

    shift_date: str
    shift_name: str
    machine_id: Optional[str] = None
    outgoing_worker_id: str
    incoming_worker_id: str
    notes: str = ""
    verified: bool = False


class WorkOrderCreate(BaseModel):
    """Payload to create an industrial maintenance work order."""
    model_config = ConfigDict(populate_by_name=True)

    machine_id: str
    task_description: str
    assigned_worker_ids: List[str] = Field(default_factory=list)
    required_qualifications: List[str] = Field(default_factory=list)
    required_clearance: str = "L1"
    scheduled_start: str
    scheduled_end: str
    inspection_required: bool = True


class WorkOrderCompleteRequest(BaseModel):
    """Payload to mark maintenance complete with inspector verification."""
    model_config = ConfigDict(populate_by_name=True)

    actual_end: str
    inspector_id: Optional[str] = None
    inspection_evidence: Optional[Dict[str, Any]] = None


class WorkforceCorrelateRequest(BaseModel):
    """Payload to correlate workforce events with an operational incident."""
    model_config = ConfigDict(populate_by_name=True)

    incident_id: str
    machine_id: Optional[str] = None
    incident_time: Optional[str] = None
    window_hours: float = 48.0




