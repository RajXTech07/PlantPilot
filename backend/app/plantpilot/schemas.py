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

