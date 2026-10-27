"""Trajectory forecasting models and observational input validation."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np

from .models import ForecastRequest, ObservationItem, ThresholdDefinition


VALID_UNITS: Dict[str, set[str]] = {
    "air_temperature": {"K", "k", "degK", "kelvin", "degC", "C", "celsius"},
    "process_temperature": {"K", "k", "degK", "kelvin", "degC", "C", "celsius"},
    "rotational_speed": {"rpm", "RPM", "rad/s", "hz", "Hz"},
    "torque": {"Nm", "nm", "N.m", "N*m"},
    "tool_wear": {"min", "minutes", "m", "s", "sec"},
}


class ForecastValidationError(ValueError):
    """Raised when forecast inputs violate physical units, domain constraints, or history requirements."""


def validate_observations(
    observations: List[ObservationItem],
    threshold: ThresholdDefinition,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Validate timestamp monotonicity, physical units, and numeric validity.
    
    Returns:
        (t_rel_seconds, values, t_last_epoch)
    """
    param = threshold.parameter
    expected_unit = threshold.unit.strip().lower()

    if len(observations) < 2:
        raise ForecastValidationError("At least 2 timestamped observations are required for forecasting.")

    # Check units if declared
    param_key = param.lower().replace(" ", "_")
    matched_group = next((k for k in VALID_UNITS if k in param_key), None)
    if matched_group:
        valid_set = {u.lower() for u in VALID_UNITS[matched_group]}
        if expected_unit not in valid_set:
            raise ForecastValidationError(
                f"Invalid unit '{threshold.unit}' for parameter '{param}'. Expected one of {sorted(valid_set)}."
            )

    times_epoch: List[float] = []
    vals: List[float] = []

    for idx, obs in enumerate(observations):
        t_epoch = obs.timestamp.timestamp()
        if idx > 0 and t_epoch <= times_epoch[-1]:
            raise ForecastValidationError(
                f"Observations must be strictly increasing in time. Timestamp at index {idx} ({obs.timestamp}) "
                f"is not after {observations[idx - 1].timestamp}."
            )
        times_epoch.append(t_epoch)

        # Extract parameter value
        val = None
        for k, v in obs.values.items():
            if k.lower().replace(" ", "_") == param_key or k.lower() == param.lower():
                val = float(v)
                break
        if val is None:
            raise ForecastValidationError(f"Observation at index {idx} is missing parameter '{param}'.")

        if not math.isfinite(val):
            raise ForecastValidationError(f"Observation at index {idx} contains non-finite value for '{param}'.")

        # Physical domain checks
        if "temperature" in param_key and expected_unit in {"k", "kelvin"} and val < 0.0:
            raise ForecastValidationError(f"Physical constraint violation: Temperature cannot be negative in Kelvin ({val} K).")
        if "wear" in param_key and val < 0.0:
            raise ForecastValidationError(f"Physical constraint violation: Tool wear cannot be negative ({val}).")

        # Check declared unit on observation if present
        obs_unit = obs.units.get(param) or obs.units.get(param_key)
        if obs_unit and matched_group:
            if obs_unit.strip().lower() not in valid_set:
                raise ForecastValidationError(
                    f"Observation at index {idx} has incompatible unit '{obs_unit}' for '{param}'."
                )

        vals.append(val)

    t_last = times_epoch[-1]
    t_rel = np.array([t - t_last for t in times_epoch], dtype=float)
    y_arr = np.array(vals, dtype=float)
    return t_rel, y_arr, t_last


def fit_forecast_model(
    t_rel: np.ndarray,
    y: np.ndarray,
    model_type: str,
    horizon_seconds: float,
    num_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit degradation model on relative time and project over the horizon [0, horizon_seconds].
    
    Returns:
        (forecast_t_rel, forecast_y, model_fit_metadata)
    """
    forecast_t = np.linspace(0.0, horizon_seconds, num_steps)
    metadata: Dict[str, Any] = {"model_type": model_type, "history_points": len(t_rel)}

    if model_type == "polynomial_wear" and len(t_rel) >= 3:
        # y(t) = a + b t + c t^2 with c >= 0
        poly_coeffs = np.polyfit(t_rel, y, 2)
        c, b, a = poly_coeffs
        # If wear acceleration is unphysically negative due to noise, clamp to linear
        if c < 0:
            poly_coeffs = [0.0, *np.polyfit(t_rel, y, 1)]
            c, b, a = poly_coeffs

        forecast_y = np.polyval(poly_coeffs, forecast_t)
        residuals = y - np.polyval(poly_coeffs, t_rel)
        sigma = float(np.std(residuals, ddof=2)) if len(t_rel) > 2 else 0.0
        metadata.update({
            "coefficients": [float(c), float(b), float(a)],
            "sigma_residual": sigma,
            "degree": 2,
        })

    elif model_type == "physics_thermal":
        # First-order thermal dissipation towards equilibrium T_inf
        # y(t) = T_inf + (y0 - T_inf) * exp(-t / tau)
        t_span = t_rel[-1] - t_rel[0]
        dy = y[-1] - y[0]
        drift_rate = dy / t_span if abs(t_span) > 1e-6 else 0.0
        # If rising, equilibrium is above current; if falling, below
        tau = max(100.0, horizon_seconds / 2.0)
        t_inf = y[-1] + drift_rate * tau
        forecast_y = t_inf + (y[-1] - t_inf) * np.exp(-forecast_t / tau)
        residuals = y - (t_inf + (y[0] - t_inf) * np.exp(-(t_rel - t_rel[0]) / tau))
        sigma = float(np.std(residuals))
        metadata.update({
            "equilibrium": float(t_inf),
            "time_constant_tau": float(tau),
            "sigma_residual": sigma,
        })

    else:
        # Default: linear drift y(t) = a + b * t
        b, a = np.polyfit(t_rel, y, 1)
        forecast_y = a + b * forecast_t
        residuals = y - (a + b * t_rel)
        sigma = float(np.std(residuals, ddof=1)) if len(t_rel) > 1 else 0.0
        # Estimate parameter covariance for uncertainty propagation
        X = np.vstack([np.ones_like(t_rel), t_rel]).T
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            param_cov = (sigma ** 2) * XtX_inv
        except np.linalg.LinAlgError:
            param_cov = np.eye(2) * (sigma ** 2)

        metadata.update({
            "slope_b": float(b),
            "intercept_a": float(a),
            "sigma_residual": sigma,
            "param_covariance": param_cov.tolist(),
        })

    return forecast_t, forecast_y, metadata


def forecast_trajectory(request: ForecastRequest) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Validate observations and compute forecast trajectory.
    
    Returns:
        (forecast_t_rel, forecast_y, t_last_epoch, metadata)
    """
    t_rel, y_history, t_last_epoch = validate_observations(request.observations, request.threshold)
    forecast_t, forecast_y, metadata = fit_forecast_model(
        t_rel,
        y_history,
        request.model_type,
        request.forecast_horizon_seconds,
    )
    return forecast_t, forecast_y, t_last_epoch, metadata
