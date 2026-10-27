"""Pydantic v2 data models for the PlantPilot Prognostics Subsystem."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionTarget(str, Enum):
    """Explicit prediction target distinction."""
    THRESHOLD_BREACH = "threshold_breach"
    PHYSICAL_FAILURE = "physical_failure"


class CrossingPolicy(str, Enum):
    """Policy defining when a threshold is considered breached."""
    FIRST_CROSSING = "first_crossing"
    SUSTAINED_CROSSING = "sustained_crossing"
    PERSISTENT_CROSSING = "persistent_crossing"


class ForecastStatus(str, Enum):
    """Explicit outcome status for a forecast evaluation."""
    BREACH_PREDICTED = "breach_predicted"
    NO_BREACH_WITHIN_HORIZON = "no_breach_within_horizon"
    UNSUPPORTED_TARGET = "unsupported_target"
    INADEQUATE_EVIDENCE = "inadequate_evidence"
    INVALID_INPUT = "invalid_input"


class ThresholdOperator(str, Enum):
    """Comparison operator for threshold function g(x)."""
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="


class ThresholdDefinition(BaseModel):
    """Registered mathematical threshold function g(x) >= 0."""
    model_config = ConfigDict(populate_by_name=True)

    parameter: str = Field(..., description="Telemetry parameter name (e.g. tool_wear, process_temperature, torque)")
    operator: ThresholdOperator = Field(ThresholdOperator.GTE, description="Comparison operator defining breach condition")
    value: float = Field(..., description="Numerical threshold limit")
    unit: str = Field(..., description="Expected physical unit (e.g. min, K, Nm, rpm)")
    persist_duration_seconds: float = Field(0.0, ge=0.0, description="Duration in seconds condition must hold for sustained crossing")
    persist_steps: int = Field(1, ge=1, description="Number of consecutive points condition must hold")
    critical_cumulative_time: float = Field(0.0, ge=0.0, description="Cumulative exceedance seconds for persistent crossing")


class ObservationItem(BaseModel):
    """Timestamped historical measurement point."""
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime = Field(..., description="UTC observation timestamp")
    values: Dict[str, float] = Field(..., description="Dictionary of parameter numerical values")
    units: Dict[str, str] = Field(default_factory=dict, description="Dictionary of physical units")

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class PredictionInterval(BaseModel):
    """Uncertainty quantification bounds for predicted breach time."""
    model_config = ConfigDict(populate_by_name=True)

    confidence_level: float = Field(0.95, ge=0.5, lt=1.0)
    lower_time: Optional[datetime] = None
    upper_time: Optional[datetime] = None
    lower_seconds_from_last_obs: Optional[float] = None
    upper_seconds_from_last_obs: Optional[float] = None
    method: str = Field("parameter_covariance", description="Method used to calculate interval")


class ForecastRequest(BaseModel):
    """Payload to request a deterministic or probabilistic threshold breach forecast."""
    model_config = ConfigDict(populate_by_name=True)

    asset_id: str = Field(..., min_length=1)
    observations: List[ObservationItem] = Field(..., min_length=2, description="Timestamped history (at least 2 points)")
    threshold: ThresholdDefinition = Field(...)
    forecast_horizon_seconds: float = Field(3600.0, gt=0.0, description="Maximum forecast lookahead horizon in seconds")
    prediction_target: PredictionTarget = Field(PredictionTarget.THRESHOLD_BREACH)
    crossing_policy: CrossingPolicy = Field(CrossingPolicy.FIRST_CROSSING)
    model_type: str = Field("linear_drift", description="linear_drift, polynomial_wear, or physics_thermal")
    operating_conditions: Dict[str, Any] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
    """Structured response for threshold breach forecasting."""
    model_config = ConfigDict(populate_by_name=True)

    forecast_id: str = Field(default_factory=lambda: f"fc_{uuid.uuid4().hex[:12]}")
    asset_id: str
    status: ForecastStatus
    prediction_target: PredictionTarget = PredictionTarget.THRESHOLD_BREACH
    predicted_time: Optional[datetime] = None
    predicted_time_seconds_from_last_obs: Optional[float] = None
    uncertainty: Optional[PredictionInterval] = None
    baseline_margin: float = Field(..., description="Current safety margin at last observation: -g(x0)")
    assumptions: List[str] = Field(default_factory=list)
    evidence_references: List[str] = Field(default_factory=list)
    model_version: str = "1.0.0"
    limitations: List[str] = Field(default_factory=list)
    trajectory_sample: List[Dict[str, Any]] = Field(default_factory=list)


class PhysicalFailureRequest(BaseModel):
    """Payload to estimate physical time-to-failure (strictly separated from threshold breach)."""
    model_config = ConfigDict(populate_by_name=True)

    asset_id: str = Field(..., min_length=1)
    observations: List[ObservationItem] = Field(..., min_length=2)
    failure_mode: str = Field("tool_wear_failure", description="Specific physical failure mode")
    degradation_model: str = Field("taylor_tool_life", description="Documented physical degradation model")
    operating_speed_rpm: Optional[float] = None
    calibrated_parameters: Dict[str, float] = Field(default_factory=dict)
    censored_observation: bool = Field(True, description="Whether observations are right-censored prior to failure")
    forecast_horizon_seconds: float = Field(36000.0, gt=0.0)


class PhysicalFailureResponse(BaseModel):
    """Result of physical time-to-failure analysis."""
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(default_factory=lambda: f"fail_{uuid.uuid4().hex[:12]}")
    asset_id: str
    status: ForecastStatus
    failure_mode: str
    degradation_model: str
    estimated_failure_time: Optional[datetime] = None
    estimated_rul_seconds: Optional[float] = None
    uncertainty: Optional[PredictionInterval] = None
    censored: bool = True
    validated_assumptions: List[str] = Field(default_factory=list)
    refusal_reasons: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
