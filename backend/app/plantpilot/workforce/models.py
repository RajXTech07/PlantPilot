"""Pydantic data models for Workforce Intelligence subsystem."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

LEVELS: Dict[str, int] = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}


class ClearanceLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"


class WorkerStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class AttendanceStatus(str, Enum):
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"


class WorkOrderStatus(str, Enum):
    DRAFT = "DRAFT"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class VerificationOutcome(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"


class FindingCategory(str, Enum):
    ATTENDANCE = "ATTENDANCE"
    QUALIFICATION = "QUALIFICATION"
    STAFFING = "STAFFING"
    MAINTENANCE = "MAINTENANCE"
    FATIGUE = "FATIGUE"
    COMPLIANCE = "COMPLIANCE"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CertificationRecord(BaseModel):
    """Worker qualification or safety certification record."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1)
    cert_id: Optional[str] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None
    valid: bool = True


class WorkerProfile(BaseModel):
    """Complete industrial worker profile."""
    model_config = ConfigDict(populate_by_name=True)

    worker_id: str
    name: str
    role: str
    department: str
    clearance_level: str
    status: str = "ACTIVE"
    skills: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    certifications: List[CertificationRecord] = Field(default_factory=list)
    assigned_machines: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AttendanceRecord(BaseModel):
    """Shift clock-in and clock-out record."""
    model_config = ConfigDict(populate_by_name=True)

    record_id: str
    worker_id: str
    shift_date: str
    clock_in: str
    clock_out: Optional[str] = None
    break_minutes: int = 0
    working_hours: Optional[float] = None
    attendance_status: str = "OPEN"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShiftSchedule(BaseModel):
    """Planned shift staffing requirements and worker assignments."""
    model_config = ConfigDict(populate_by_name=True)

    schedule_id: str
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
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShiftHandover(BaseModel):
    """Formal shift-to-shift operational handover record."""
    model_config = ConfigDict(populate_by_name=True)

    handover_id: str
    shift_date: str
    shift_name: str
    machine_id: Optional[str] = None
    outgoing_worker_id: str
    incoming_worker_id: str
    notes: str = ""
    verified: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkOrder(BaseModel):
    """Industrial maintenance work order and verification evidence."""
    model_config = ConfigDict(populate_by_name=True)

    order_id: str
    machine_id: str
    task_description: str
    assigned_worker_ids: List[str] = Field(default_factory=list)
    required_qualifications: List[str] = Field(default_factory=list)
    required_clearance: str = "L1"
    scheduled_start: str
    scheduled_end: str
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    status: str = "DRAFT"
    inspection_required: bool = True
    inspection_evidence: Optional[Dict[str, Any]] = None
    verification_outcome: str = "PENDING"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkforceFinding(BaseModel):
    """Deterministic, explainable operational finding or risk."""
    model_config = ConfigDict(populate_by_name=True)

    finding_id: str
    category: str
    severity: str
    worker_id: Optional[str] = None
    machine_id: Optional[str] = None
    work_order_id: Optional[str] = None
    shift_date: Optional[str] = None
    rule_name: str
    explanation: str
    details: Dict[str, Any] = Field(default_factory=dict)
    recommended_followup: str = ""
    resolution_status: str = "OPEN"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class WorkforceIncidentCorrelation(BaseModel):
    """Structured correlation of workforce events with a machine incident."""
    model_config = ConfigDict(populate_by_name=True)

    incident_id: str
    machine_id: Optional[str] = None
    incident_time: str
    recent_work_orders: List[WorkOrder] = Field(default_factory=list)
    unverified_maintenance_detected: bool = False
    missing_inspections: List[str] = Field(default_factory=list)
    active_workers_at_time: List[Dict[str, Any]] = Field(default_factory=list)
    qualification_issues: List[str] = Field(default_factory=list)
    shift_handovers: List[ShiftHandover] = Field(default_factory=list)
    temporal_correlations: List[str] = Field(default_factory=list)
    hypotheses: List[str] = Field(default_factory=list)
    supported_relationships: List[str] = Field(default_factory=list)
    causal_warnings: List[str] = Field(default_factory=list)
    suggested_evidence_items: List[Dict[str, Any]] = Field(default_factory=list)
