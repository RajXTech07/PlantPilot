"""PlantPilot Prognostics Subsystem — Time to Breach and Physical Failure Estimation."""
from .models import (
    CrossingPolicy,
    ForecastRequest,
    ForecastResponse,
    ForecastStatus,
    ObservationItem,
    PhysicalFailureRequest,
    PhysicalFailureResponse,
    PredictionInterval,
    PredictionTarget,
    ThresholdDefinition,
)
from .service import PrognosticsService, get_prognostics_service, reset_prognostics_service
from .threshold import evaluate_threshold, is_breached, safety_margin
from .forecasting import forecast_trajectory
from .failure_model import evaluate_physical_failure
from .uncertainty import calculate_prediction_interval

__all__ = [
    "CrossingPolicy",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastStatus",
    "ObservationItem",
    "PhysicalFailureRequest",
    "PhysicalFailureResponse",
    "PredictionInterval",
    "PredictionTarget",
    "ThresholdDefinition",
    "PrognosticsService",
    "get_prognostics_service",
    "reset_prognostics_service",
    "evaluate_threshold",
    "is_breached",
    "safety_margin",
    "forecast_trajectory",
    "evaluate_physical_failure",
    "calculate_prediction_interval",
]
