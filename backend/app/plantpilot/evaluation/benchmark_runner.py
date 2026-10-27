"""Deterministic synthetic benchmark runners for prognostics and fragility search."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import numpy as np

from ..prognostics.models import (
    CrossingPolicy,
    ForecastRequest,
    ObservationItem,
    ThresholdDefinition,
    ThresholdOperator,
)
from ..prognostics.service import PrognosticsService
from ..fragility.models import DisturbanceVariable, FragilitySearchRequest
from ..fragility.service import FragilityService
from .prognostics_metrics import compute_breach_metrics
from .fragility_metrics import compute_fragility_metrics


def run_prognostics_synthetic_benchmark(num_cases: int = 30) -> Dict[str, Any]:
    """Run deterministic synthetic benchmark across multiple degradation trajectories with known analytical breach times."""
    service = PrognosticsService()
    predictions = []
    ground_truth = []

    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    rng = np.random.RandomState(42)  # Fixed deterministic seed

    for i in range(num_cases):
        y0 = float(rng.uniform(50.0, 100.0))
        drift = float(rng.uniform(0.05, 0.50))  # units / second
        limit = 150.0

        # In 20% of cases, drift is zero or negative (no breach)
        will_breach = (i % 5 != 0)
        if not will_breach:
            drift = -0.01

        # Generate 5 observation points (1 every 60s)
        obs = []
        for step in range(5):
            t_cur = t0 + timedelta(seconds=step * 60)
            noise = float(rng.normal(0.0, 0.1))
            val = y0 + drift * (step * 60) + noise
            obs.append(
                ObservationItem(
                    timestamp=t_cur,
                    values={"tool_wear": round(val, 2)},
                    units={"tool_wear": "min"},
                )
            )

        thresh = ThresholdDefinition(
            parameter="tool_wear",
            operator=ThresholdOperator.GTE,
            value=limit,
            unit="min",
        )
        t_last_sec = 4 * 60

        if will_breach:
            true_breach_from_last = max(0.0, (limit - (y0 + drift * t_last_sec)) / drift)
            gt_entry = {
                "breached": True,
                "breach_time_seconds": true_breach_from_last,
            }
        else:
            gt_entry = {
                "breached": False,
                "breach_time_seconds": None,
            }

        req = ForecastRequest(
            asset_id=f"mach_bench_{i:03d}",
            observations=obs,
            threshold=thresh,
            forecast_horizon_seconds=3600.0,
            crossing_policy=CrossingPolicy.FIRST_CROSSING,
        )
        resp = service.forecast_breach(req)

        predictions.append({
            "status": resp.status.value,
            "predicted_time_seconds": resp.predicted_time_seconds_from_last_obs,
            "uncertainty": resp.uncertainty.model_dump() if resp.uncertainty else None,
        })
        ground_truth.append(gt_entry)

    metrics = compute_breach_metrics(predictions, ground_truth)
    return {
        "benchmark_type": "synthetic_prognostics_linear_drift",
        "sample_size": num_cases,
        "seed": 42,
        "metrics": metrics,
    }


def run_fragility_synthetic_benchmark(num_cases: int = 20) -> Dict[str, Any]:
    """Run deterministic synthetic benchmark across fragility search cases with known analytical optima."""
    service = FragilityService()
    search_results = []
    analytical_optima = []

    rng = np.random.RandomState(42)

    for i in range(num_cases):
        x0_torque = float(rng.uniform(20.0, 45.0))
        target_limit = float(rng.uniform(55.0, 75.0))
        scale = 10.0
        weight = 1.0

        thresh = ThresholdDefinition(
            parameter="torque",
            operator=ThresholdOperator.GTE,
            value=target_limit,
            unit="Nm",
        )
        var = DisturbanceVariable(
            name="torque",
            min_val=10.0,
            max_val=85.0,
            nominal_scale=scale,
            weight=weight,
        )

        true_dx = target_limit - x0_torque
        true_cost = true_dx / scale

        req = FragilitySearchRequest(
            asset_id=f"mach_frag_{i:03d}",
            baseline_state={"torque": x0_torque},
            threshold=thresh,
            variables=[var],
            cost_norm=2,
        )

        resp = service.search_fragility(req)
        search_results.append({
            "status": resp.status.value,
            "disturbance_cost": resp.disturbance_cost,
            "evaluations_used": resp.evaluations_used,
        })
        analytical_optima.append({
            "true_optimal_dx": true_dx,
            "true_optimal_cost": true_cost,
        })

    metrics = compute_fragility_metrics(search_results, analytical_optima)
    return {
        "benchmark_type": "synthetic_fragility_1d_analytical",
        "sample_size": num_cases,
        "seed": 42,
        "metrics": metrics,
    }
