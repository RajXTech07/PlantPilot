"""Pydantic v2 schemas and data contracts for Fragility Search."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ..prognostics.models import ThresholdDefinition


class OptimalityStatus(str, Enum):
    """Explicit mathematical optimality status."""
    VERIFIED_OPTIMUM = "verified_optimum"
    FEASIBLE_CANDIDATE_FOUND = "feasible_candidate_found"
    BASELINE_ALREADY_BREACHED = "baseline_already_breached"
    INFEASIBLE = "infeasible"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INVALID_REQUEST = "invalid_request"


class DisturbanceVariable(BaseModel):
    """Specification of a disturbance dimension, allowed domain, and cost normalization."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1, description="Operating variable name (e.g. rotational_speed, torque)")
    min_val: float = Field(..., description="Absolute lower feasibility bound x_min")
    max_val: float = Field(..., description="Absolute upper feasibility bound x_max")
    nominal_scale: float = Field(1.0, gt=0.0, description="Characteristic scale s_i for normalization")
    weight: float = Field(1.0, gt=0.0, description="Cost penalty weight w_i")
    step_size: Optional[float] = Field(None, gt=0.0, description="Optional discretization step size")

    @field_validator("max_val")
    @classmethod
    def check_bounds(cls, v: float, info) -> float:
        min_v = info.data.get("min_val")
        if min_v is not None and v < min_v:
            raise ValueError(f"max_val ({v}) must be greater than or equal to min_val ({min_v})")
        return v


class SearchBudget(BaseModel):
    """Budget constraints to guarantee deterministic termination."""
    model_config = ConfigDict(populate_by_name=True)

    max_evaluations: int = Field(500, ge=10, le=10000)
    tolerance: float = Field(1e-4, gt=0.0)
    max_iterations: int = Field(100, ge=5, le=2000)


class PhysicalConstraint(BaseModel):
    """Physical coupling or domain constraint c(x) <= 0."""
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(...)
    expression: str = Field(..., description="Description or formula, e.g. 'process_temp >= air_temp' or 'power_limit'")
    max_power_watts: Optional[float] = None


class FragilitySearchRequest(BaseModel):
    """Payload to find smallest feasible disturbance causing threshold breach."""
    model_config = ConfigDict(populate_by_name=True)

    asset_id: str = Field(..., min_length=1)
    baseline_state: Dict[str, float] = Field(..., description="Current baseline operational state x_0")
    threshold: ThresholdDefinition = Field(..., description="Threshold condition g(x) >= 0 to trigger")
    variables: List[DisturbanceVariable] = Field(..., min_length=1, description="Variables subject to disturbance")
    cost_norm: int = Field(2, description="L1 (1) or L2 (2) norm for disturbance cost")
    constraints: List[PhysicalConstraint] = Field(default_factory=list)
    registered_model: str = Field("ai4i_linear_margin", description="Registered model name for evaluation")
    search_budget: SearchBudget = Field(default_factory=SearchBudget)


class FragilitySearchResponse(BaseModel):
    """Result of smallest feasible disturbance search."""
    model_config = ConfigDict(populate_by_name=True)

    search_id: str = Field(default_factory=lambda: f"frag_{uuid.uuid4().hex[:12]}")
    asset_id: str
    status: OptimalityStatus
    baseline_margin: float = Field(..., description="Safety margin at baseline -g(x_0)")
    optimal_disturbance: Dict[str, float] = Field(default_factory=dict, description="Delta x minimizing cost")
    perturbed_state: Dict[str, float] = Field(default_factory=dict, description="x_0 + Delta x")
    threshold_evaluation: float = Field(..., description="g(x_0 + Delta x)")
    disturbance_cost: float = Field(..., description="Total weighted cost C(Delta x)")
    evaluations_used: int = Field(..., ge=0)
    search_method: str = Field(...)
    optimality_status: OptimalityStatus
    model_version: str = "1.0.0"
    evidence_references: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class SensitivityRankItem(BaseModel):
    """Sensitivity ranking for a single operating variable."""
    model_config = ConfigDict(populate_by_name=True)

    variable: str
    min_disturbance_needed: float
    normalized_cost: float
    rank: int
    direction: str
    breach_achieved: bool


class SensitivityAnalysisRequest(BaseModel):
    """Request to compute single-parameter sensitivity rankings."""
    model_config = ConfigDict(populate_by_name=True)

    asset_id: str = Field(..., min_length=1)
    baseline_state: Dict[str, float] = Field(...)
    threshold: ThresholdDefinition = Field(...)
    variables: List[DisturbanceVariable] = Field(..., min_length=1)


class SensitivityAnalysisResponse(BaseModel):
    """Ranking of variables by fragility (easiest to breach)."""
    model_config = ConfigDict(populate_by_name=True)

    asset_id: str
    baseline_margin: float
    ranked_variables: List[SensitivityRankItem]
    most_fragile_variable: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
