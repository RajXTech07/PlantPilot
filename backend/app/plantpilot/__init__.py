"""PlantPilot — Backend Framework for Human-Supervised Industrial Operations AI Copilot."""

__version__ = "0.1.0"

# Relative imports work whether package is installed as "plantpilot" or
# imported from app/ as module "plantpilot" (pythonpath=["app"]).
from .config import get_config, ConfigError
from .schemas import (
    Incident,
    IncidentCreate,
    EvidenceItem,
    Hypothesis,
    ToolExecution,
    Recommendation,
    Investigation,
    InvestigationCreate,
    DigitalTwin,
    DigitalTwinCreate,
    TwinReadingCreate,
    TwinSimulationRequest,
    FailureRiskPrediction,
    DiagnosisAssessRequest,
    DiagnosisAssessResponse,
    ReportAuditRequest,
    ReportAuditResponse,
    RankTestsRequest,
    RankTestsResponse,
    WorkerCreate,
    WorkerItem,
    ClockInRequest,
    ClockOutRequest,
    AttendanceRecordItem,
    ClearanceCheckRequest,
    ClearanceCheckResponse,
    WorkforceAnomaliesRequest,
    WorkforceFindingsResponse,
    WorkforceReviewResponse,
)
from .safety import SafetyPolicy, check_safety
from .tools import ToolRegistry, registered_tools
from .storage import Database, get_database
from .agent import AgentController, InvestigationResult
from .digital_twin import TwinStore, get_twin_store, reset_twin_store
from .detection import predict as predict_failure_risk, normalize_reading
from .diagnostics import assess_diagnosis, audit_report, rank_next_tests
from .workforce import (
    WorkforceStore,
    get_workforce_store,
    reset_workforce_store,
    model_assess_workforce,
)

__all__ = [
    "get_config",
    "ConfigError",
    "Incident",
    "IncidentCreate",
    "EvidenceItem",
    "Hypothesis",
    "ToolExecution",
    "Recommendation",
    "Investigation",
    "InvestigationCreate",
    "DigitalTwin",
    "DigitalTwinCreate",
    "TwinReadingCreate",
    "TwinSimulationRequest",
    "FailureRiskPrediction",
    "DiagnosisAssessRequest",
    "DiagnosisAssessResponse",
    "ReportAuditRequest",
    "ReportAuditResponse",
    "RankTestsRequest",
    "RankTestsResponse",
    "WorkerCreate",
    "WorkerItem",
    "ClockInRequest",
    "ClockOutRequest",
    "AttendanceRecordItem",
    "ClearanceCheckRequest",
    "ClearanceCheckResponse",
    "WorkforceAnomaliesRequest",
    "WorkforceFindingsResponse",
    "WorkforceReviewResponse",
    "SafetyPolicy",
    "check_safety",
    "ToolRegistry",
    "registered_tools",
    "Database",
    "get_database",
    "AgentController",
    "InvestigationResult",
    "TwinStore",
    "get_twin_store",
    "reset_twin_store",
    "predict_failure_risk",
    "normalize_reading",
    "assess_diagnosis",
    "audit_report",
    "rank_next_tests",
    "WorkforceStore",
    "get_workforce_store",
    "reset_workforce_store",
    "model_assess_workforce",
]