"""Threshold evaluation functions and crossing policy enforcement."""
from __future__ import annotations

from typing import List, Tuple
from .models import CrossingPolicy, ThresholdDefinition, ThresholdOperator


def evaluate_threshold(value: float, threshold: ThresholdDefinition) -> float:
    """Evaluate threshold function g(x).
    
    A breach occurs when g(x) >= 0.
    Positive value indicates exceedance magnitude; negative value indicates safety margin.
    """
    op = threshold.operator
    limit = threshold.value
    
    if op in (ThresholdOperator.GT, ThresholdOperator.GTE):
        # Breach when value >= limit
        return value - limit
    elif op in (ThresholdOperator.LT, ThresholdOperator.LTE):
        # Breach when value <= limit
        return limit - value
    else:
        raise ValueError(f"Unsupported threshold operator: {op}")


def is_breached(value: float, threshold: ThresholdDefinition) -> bool:
    """Return True if the single value satisfies the breach condition g(x) >= 0."""
    g_val = evaluate_threshold(value, threshold)
    if threshold.operator in (ThresholdOperator.GT, ThresholdOperator.LT):
        return g_val > 0.0
    return g_val >= 0.0


def safety_margin(value: float, threshold: ThresholdDefinition) -> float:
    """Calculate the safety margin m(x) = -g(x).
    
    Positive indicates safe distance to threshold; zero or negative indicates breach.
    """
    return -evaluate_threshold(value, threshold)


def find_crossing_index(
    times: List[float],
    values: List[float],
    threshold: ThresholdDefinition,
    policy: CrossingPolicy = CrossingPolicy.FIRST_CROSSING,
) -> Tuple[int | None, float | None]:
    """Find the index and exact interpolated time of threshold crossing under the specified policy.
    
    Returns:
        (crossing_index, exact_crossing_time) or (None, None) if no crossing occurs.
    """
    if not times or not values or len(times) != len(values):
        return None, None

    n = len(values)

    if policy == CrossingPolicy.FIRST_CROSSING:
        for i in range(n):
            if is_breached(values[i], threshold):
                # Interpolate if previous point exists
                if i > 0 and not is_breached(values[i - 1], threshold):
                    g_prev = evaluate_threshold(values[i - 1], threshold)
                    g_curr = evaluate_threshold(values[i], threshold)
                    denom = g_curr - g_prev
                    if abs(denom) > 1e-12:
                        frac = -g_prev / denom
                        frac = max(0.0, min(1.0, frac))
                        interp_time = times[i - 1] + frac * (times[i] - times[i - 1])
                        return i, interp_time
                return i, times[i]
        return None, None

    elif policy == CrossingPolicy.SUSTAINED_CROSSING:
        req_steps = max(1, threshold.persist_steps)
        req_dur = threshold.persist_duration_seconds

        consec_start = None
        for i in range(n):
            if is_breached(values[i], threshold):
                if consec_start is None:
                    consec_start = i
                
                cur_steps = i - consec_start + 1
                cur_dur = times[i] - times[consec_start]
                if cur_steps >= req_steps and cur_dur >= req_dur:
                    return consec_start, times[consec_start]
            else:
                consec_start = None
        return None, None

    elif policy == CrossingPolicy.PERSISTENT_CROSSING:
        cum_time = 0.0
        crit_time = threshold.critical_cumulative_time
        for i in range(1, n):
            dt = times[i] - times[i - 1]
            if is_breached(values[i], threshold):
                cum_time += dt
                if cum_time >= crit_time:
                    return i, times[i]
        return None, None

    return None, None
