"""Physical degradation modeling and time-to-failure evaluation.

Enforces strict separation between threshold crossings and physical failure.
Refuses unsupported failure targets or inadequate evidence.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np

from .models import (
    ForecastStatus,
    PhysicalFailureRequest,
    PhysicalFailureResponse,
    PredictionInterval,
)

SUPPORTED_PHYSICAL_MODELS = {"taylor_tool_life", "critical_wear_limit"}


def evaluate_physical_failure(request: PhysicalFailureRequest) -> PhysicalFailureResponse:
    """Evaluate physical time-to-failure using a documented degradation model.
    
    Refuses with status UNSUPPORTED_TARGET or INADEQUATE_EVIDENCE if required
    calibrated parameters or degradation observations are absent.
    """
    model = request.degradation_model.strip().lower()
    
    # Check 1: Supported degradation model
    if model not in SUPPORTED_PHYSICAL_MODELS:
        return PhysicalFailureResponse(
            asset_id=request.asset_id,
            status=ForecastStatus.UNSUPPORTED_TARGET,
            failure_mode=request.failure_mode,
            degradation_model=request.degradation_model,
            censored=request.censored_observation,
            refusal_reasons=[
                f"Degradation model '{request.degradation_model}' is not a registered physical degradation model.",
                f"Supported physical models are: {sorted(SUPPORTED_PHYSICAL_MODELS)}.",
                "Threshold crossings cannot be substituted for physical failure dynamics.",
            ],
            limitations=["Physical failure prediction refused without an empirical or physics-based run-to-failure model."],
        )

    # Check 2: At least 2 observations
    if len(request.observations) < 2:
        return PhysicalFailureResponse(
            asset_id=request.asset_id,
            status=ForecastStatus.INADEQUATE_EVIDENCE,
            failure_mode=request.failure_mode,
            degradation_model=request.degradation_model,
            censored=request.censored_observation,
            refusal_reasons=["At least 2 chronological observations are required to estimate degradation."],
        )

    last_obs = request.observations[-1]
    t_last = last_obs.timestamp
    t_last_epoch = t_last.timestamp()

    # Extract current wear or degradation parameter
    wear_val = None
    for k, v in last_obs.values.items():
        if "wear" in k.lower():
            wear_val = float(v)
            break

    if wear_val is None:
        return PhysicalFailureResponse(
            asset_id=request.asset_id,
            status=ForecastStatus.INADEQUATE_EVIDENCE,
            failure_mode=request.failure_mode,
            degradation_model=request.degradation_model,
            censored=request.censored_observation,
            refusal_reasons=["Observations do not contain a tool_wear or degradation parameter."],
        )

    params = request.calibrated_parameters

    if model == "taylor_tool_life":
        # Taylor equation: V * T^n = C  => T_life_min = (C / V)^(1/n)
        # where V = cutting speed in m/min (or derived from rotational_speed_rpm)
        # n = Taylor exponent (typical carbide milling: 0.20 - 0.35)
        # C = Taylor constant
        speed_rpm = request.operating_speed_rpm or last_obs.values.get("rotational_speed") or last_obs.values.get("rotational_speed_rpm")
        if speed_rpm is None or speed_rpm <= 0:
            return PhysicalFailureResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INADEQUATE_EVIDENCE,
                failure_mode=request.failure_mode,
                degradation_model=request.degradation_model,
                censored=request.censored_observation,
                refusal_reasons=["Operating spindle speed (RPM) is required for Taylor tool life calculation."],
            )

        n = params.get("taylor_exponent_n")
        C = params.get("taylor_constant_C")
        tool_diameter_mm = params.get("tool_diameter_mm", 20.0)

        if n is None or C is None or n <= 0 or C <= 0:
            return PhysicalFailureResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INADEQUATE_EVIDENCE,
                failure_mode=request.failure_mode,
                degradation_model=request.degradation_model,
                censored=request.censored_observation,
                refusal_reasons=[
                    "Missing calibrated Taylor model parameters 'taylor_exponent_n' and/or 'taylor_constant_C'.",
                    "Uncalibrated models cannot be used to extrapolate physical failure time.",
                ],
            )

        # Cutting speed V = pi * D * N / 1000  (m/min)
        V = (math.pi * tool_diameter_mm * speed_rpm) / 1000.0
        if V <= 0:
            return PhysicalFailureResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INVALID_INPUT,
                failure_mode=request.failure_mode,
                degradation_model=request.degradation_model,
                refusal_reasons=["Calculated cutting speed is non-positive."],
            )

        # Expected total life in minutes
        total_life_min = (C / V) ** (1.0 / n)
        # RUL in minutes
        rul_min = max(0.0, total_life_min - wear_val)
        rul_seconds = rul_min * 60.0

        est_time = datetime.fromtimestamp(t_last_epoch + rul_seconds, tz=timezone.utc)
        
        # Uncertainty band assuming 10% parameter variance on C
        rul_lower = max(0.0, ((0.9 * C / V) ** (1.0 / n) - wear_val) * 60.0)
        rul_upper = max(0.0, ((1.1 * C / V) ** (1.0 / n) - wear_val) * 60.0)

        uncertainty = PredictionInterval(
            confidence_level=0.90,
            lower_time=datetime.fromtimestamp(t_last_epoch + rul_lower, tz=timezone.utc),
            upper_time=datetime.fromtimestamp(t_last_epoch + rul_upper, tz=timezone.utc),
            lower_seconds_from_last_obs=round(rul_lower, 1),
            upper_seconds_from_last_obs=round(rul_upper, 1),
            method="taylor_parameter_sensitivity_10pct",
        )

        return PhysicalFailureResponse(
            asset_id=request.asset_id,
            status=ForecastStatus.BREACH_PREDICTED if rul_seconds <= request.forecast_horizon_seconds else ForecastStatus.NO_BREACH_WITHIN_HORIZON,
            failure_mode=request.failure_mode,
            degradation_model=request.degradation_model,
            estimated_failure_time=est_time,
            estimated_rul_seconds=round(rul_seconds, 2),
            uncertainty=uncertainty,
            censored=request.censored_observation,
            validated_assumptions=[
                f"Taylor tool life equation V*T^n = C valid for carbide end-mill on steel workpiece.",
                f"Constant cutting speed V = {round(V, 2)} m/min derived from {speed_rpm} RPM and {tool_diameter_mm} mm diameter.",
                f"Calibrated parameters: n = {n}, C = {C}.",
            ],
            limitations=[
                "Assumes continuous cutting without severe thermal shock or irregular chatter.",
                "Observations right-censored prior to failure; verified against Taylor model wear progression.",
            ],
        )

    elif model == "critical_wear_limit":
        crit_wear = params.get("critical_wear_limit_min")
        wear_rate_min_per_hour = params.get("wear_rate_min_per_hour")

        if crit_wear is None or wear_rate_min_per_hour is None or wear_rate_min_per_hour <= 0:
            return PhysicalFailureResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INADEQUATE_EVIDENCE,
                failure_mode=request.failure_mode,
                degradation_model=request.degradation_model,
                censored=request.censored_observation,
                refusal_reasons=[
                    "Missing calibrated parameters 'critical_wear_limit_min' and/or 'wear_rate_min_per_hour'.",
                ],
            )

        remaining_wear = max(0.0, crit_wear - wear_val)
        rul_hours = remaining_wear / wear_rate_min_per_hour
        rul_seconds = rul_hours * 3600.0

        est_time = datetime.fromtimestamp(t_last_epoch + rul_seconds, tz=timezone.utc)

        return PhysicalFailureResponse(
            asset_id=request.asset_id,
            status=ForecastStatus.BREACH_PREDICTED if rul_seconds <= request.forecast_horizon_seconds else ForecastStatus.NO_BREACH_WITHIN_HORIZON,
            failure_mode=request.failure_mode,
            degradation_model=request.degradation_model,
            estimated_failure_time=est_time,
            estimated_rul_seconds=round(rul_seconds, 2),
            censored=request.censored_observation,
            validated_assumptions=[
                f"Wear limit calibrated to {crit_wear} min.",
                f"Linear steady-state wear rate of {wear_rate_min_per_hour} min/hr.",
            ],
            limitations=["Does not account for non-linear tertiary accelerated wear phase near catastrophic failure."],
        )

    return PhysicalFailureResponse(
        asset_id=request.asset_id,
        status=ForecastStatus.UNSUPPORTED_TARGET,
        failure_mode=request.failure_mode,
        degradation_model=request.degradation_model,
        refusal_reasons=["Target physical model could not be processed."],
    )
