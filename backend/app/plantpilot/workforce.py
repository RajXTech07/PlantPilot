"""Public facade and compatibility layer for Workforce Intelligence subsystem."""
from __future__ import annotations

from .workforce.models import (
    LEVELS,
    ClearanceLevel,
    WorkerStatus,
    AttendanceStatus,
    WorkOrderStatus,
    VerificationOutcome,
    FindingCategory,
    FindingSeverity,
    CertificationRecord,
    WorkerProfile,
    AttendanceRecord,
    ShiftSchedule,
    ShiftHandover,
    WorkOrder,
    WorkforceFinding,
    WorkforceIncidentCorrelation,
)
from .workforce.registry import WorkerRegistry, parse_datetime
from .workforce.attendance import AttendanceTracker
from .workforce.scheduling import ShiftScheduler
from .workforce.maintenance import MaintenanceManager
from .workforce.analytics import WorkforceAnalyticsEngine
from .workforce.correlation import IncidentCorrelator
from .workforce.service import WorkforceService
from .workforce import (
    WorkforceStore,
    get_workforce_store,
    reset_workforce_store,
    model_assess_workforce,
)

__all__ = [
    "LEVELS",
    "ClearanceLevel",
    "WorkerStatus",
    "AttendanceStatus",
    "WorkOrderStatus",
    "VerificationOutcome",
    "FindingCategory",
    "FindingSeverity",
    "CertificationRecord",
    "WorkerProfile",
    "AttendanceRecord",
    "ShiftSchedule",
    "ShiftHandover",
    "WorkOrder",
    "WorkforceFinding",
    "WorkforceIncidentCorrelation",
    "WorkerRegistry",
    "AttendanceTracker",
    "ShiftScheduler",
    "MaintenanceManager",
    "WorkforceAnalyticsEngine",
    "IncidentCorrelator",
    "WorkforceService",
    "WorkforceStore",
    "get_workforce_store",
    "reset_workforce_store",
    "model_assess_workforce",
    "parse_datetime",
]
