"""Analytical uncertainty propagation and prediction interval estimation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import stats

from .models import PredictionInterval, ThresholdDefinition, ThresholdOperator


def calculate_prediction_interval(
    t_breach_sec: Optional[float],
    t_last_epoch: float,
    metadata: Dict[str, Any],
    threshold: ThresholdDefinition,
    confidence_level: float = 0.95,
) -> Optional[PredictionInterval]:
    """Calculate prediction interval [t_lower, t_upper] for threshold breach time using analytical error propagation.
    
    If no crossing is predicted or covariance is unavailable, returns None.
    """
    if t_breach_sec is None:
        return None

    model_type = metadata.get("model_type")
    sigma = metadata.get("sigma_residual", 0.0)

    if sigma <= 1e-9:
        # Zero variance deterministic fit
        breach_dt = datetime.fromtimestamp(t_last_epoch + t_breach_sec, tz=timezone.utc)
        return PredictionInterval(
            confidence_level=confidence_level,
            lower_time=breach_dt,
            upper_time=breach_dt,
            lower_seconds_from_last_obs=t_breach_sec,
            upper_seconds_from_last_obs=t_breach_sec,
            method="deterministic_zero_residual",
        )

    slope = metadata.get("slope_b")
    if slope is None or abs(slope) < 1e-9:
        return None

    # Calculate t-distribution multiplier
    n_points = metadata.get("history_points", 5)
    df = max(1, n_points - 2)
    t_crit = float(stats.t.ppf(1.0 - (1.0 - confidence_level) / 2.0, df))

    # Margin on the state value at the breach time
    # delta_y = t_crit * sigma
    state_uncertainty = t_crit * sigma

    # First-order delta method propagation to crossing time: delta_t = delta_y / |slope|
    dt_uncertainty = state_uncertainty / abs(slope)

    lower_sec = max(0.0, t_breach_sec - dt_uncertainty)
    upper_sec = t_breach_sec + dt_uncertainty

    lower_dt = datetime.fromtimestamp(t_last_epoch + lower_sec, tz=timezone.utc)
    upper_dt = datetime.fromtimestamp(t_last_epoch + upper_sec, tz=timezone.utc)

    return PredictionInterval(
        confidence_level=confidence_level,
        lower_time=lower_dt,
        upper_time=upper_dt,
        lower_seconds_from_last_obs=round(lower_sec, 2),
        upper_seconds_from_last_obs=round(upper_sec, 2),
        method="linear_covariance_delta_method",
    )
