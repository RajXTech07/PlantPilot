"""Unit tests for the Prognostics Subsystem (Time to Breach and Physical Failure)."""
from datetime import datetime, timedelta, timezone
import pytest

from plantpilot.prognostics.models import (
    CrossingPolicy,
    ForecastRequest,
    ForecastStatus,
    ObservationItem,
    PhysicalFailureRequest,
    PredictionTarget,
    ThresholdDefinition,
    ThresholdOperator,
)
from plantpilot.prognostics.service import PrognosticsService
from plantpilot.prognostics.threshold import evaluate_threshold, is_breached, safety_margin, find_crossing_index


@pytest.fixture
def service(tmp_path):
    db_file = tmp_path / "prognostics_test.db"
    return PrognosticsService(db_path=db_file)


def test_threshold_math_and_margin():
    """Verify threshold function g(x) and safety margin m(x) = -g(x)."""
    thresh_high = ThresholdDefinition(
        parameter="process_temperature",
        operator=ThresholdOperator.GTE,
        value=310.0,
        unit="K",
    )
    # Safe state (305 K): g(305) = -5, margin = 5 > 0
    assert evaluate_threshold(305.0, thresh_high) == -5.0
    assert not is_breached(305.0, thresh_high)
    assert safety_margin(305.0, thresh_high) == 5.0

    # Breach state (312 K): g(312) = 2 >= 0, margin = -2 <= 0
    assert evaluate_threshold(312.0, thresh_high) == 2.0
    assert is_breached(312.0, thresh_high)
    assert safety_margin(312.0, thresh_high) == -2.0

    # Lower bound threshold (e.g. speed dropping below 1000 rpm)
    thresh_low = ThresholdDefinition(
        parameter="rotational_speed",
        operator=ThresholdOperator.LTE,
        value=1000.0,
        unit="rpm",
    )
    assert evaluate_threshold(1200.0, thresh_low) == -200.0
    assert not is_breached(1200.0, thresh_low)
    assert evaluate_threshold(950.0, thresh_low) == 50.0
    assert is_breached(950.0, thresh_low)


def test_analytically_known_linear_breach_time(service):
    """Verify exact analytical threshold breach time for linear degradation.
    
    Given: Wear starts at 100 min at t=0, increases by 10 min every 60 seconds (slope = 1/6 min/sec).
    Threshold is 150 min.
    Delta wear needed = 50 min.
    Expected time = 50 / (10/60) = 300 seconds.
    """
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        ObservationItem(
            timestamp=t0,
            values={"tool_wear": 100.0},
            units={"tool_wear": "min"},
        ),
        ObservationItem(
            timestamp=t0 + timedelta(seconds=60),
            values={"tool_wear": 110.0},
            units={"tool_wear": "min"},
        ),
        ObservationItem(
            timestamp=t0 + timedelta(seconds=120),
            values={"tool_wear": 120.0},
            units={"tool_wear": "min"},
        ),
    ]
    threshold = ThresholdDefinition(
        parameter="tool_wear",
        operator=ThresholdOperator.GTE,
        value=150.0,
        unit="min",
    )
    req = ForecastRequest(
        asset_id="mach_cnc_01",
        observations=obs,
        threshold=threshold,
        forecast_horizon_seconds=600.0,
        crossing_policy=CrossingPolicy.FIRST_CROSSING,
        model_type="linear_drift",
    )

    resp = service.forecast_breach(req)
    assert resp.status == ForecastStatus.BREACH_PREDICTED
    assert resp.predicted_time is not None
    # 180s from last observation (which was at t=120s, total 300s from t0)
    assert abs(resp.predicted_time_seconds_from_last_obs - 180.0) < 1.0
    assert resp.uncertainty is not None
    assert resp.uncertainty.lower_seconds_from_last_obs is not None


def test_no_breach_within_horizon_and_safety_disclaimer(service):
    """Verify that when no crossing occurs in the horizon, an explicit status is returned
    and the system never reports this as proof that the asset is safe.
    """
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        ObservationItem(
            timestamp=t0,
            values={"tool_wear": 10.0},
            units={"tool_wear": "min"},
        ),
        ObservationItem(
            timestamp=t0 + timedelta(seconds=60),
            values={"tool_wear": 11.0},
            units={"tool_wear": "min"},
        ),
    ]
    # Limit is 200 min, slope is 1 min/min = 1/60 min/sec. In 300 seconds, wear reaches only 16 min.
    threshold = ThresholdDefinition(
        parameter="tool_wear",
        operator=ThresholdOperator.GTE,
        value=200.0,
        unit="min",
    )
    req = ForecastRequest(
        asset_id="mach_cnc_01",
        observations=obs,
        threshold=threshold,
        forecast_horizon_seconds=300.0,
    )
    resp = service.forecast_breach(req)
    assert resp.status == ForecastStatus.NO_BREACH_WITHIN_HORIZON
    assert resp.predicted_time is None
    # Crucial safety invariant check: limitations must explicitly warn against false proof of safety
    assert any("does NOT constitute proof that the asset is safe" in lim for lim in resp.limitations)


def test_input_validation_rejections(service):
    """Verify strict rejection of invalid units, timestamps, and missing parameters."""
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    # 1. Invalid units: torque declared with Kelvin
    bad_unit_thresh = ThresholdDefinition(
        parameter="torque",
        operator=ThresholdOperator.GTE,
        value=50.0,
        unit="K",  # Invalid unit for torque!
    )
    obs = [
        ObservationItem(timestamp=t0, values={"torque": 30.0}, units={"torque": "K"}),
        ObservationItem(timestamp=t0 + timedelta(seconds=60), values={"torque": 35.0}, units={"torque": "K"}),
    ]
    resp = service.forecast_breach(
        ForecastRequest(asset_id="m1", observations=obs, threshold=bad_unit_thresh)
    )
    assert resp.status == ForecastStatus.INVALID_INPUT
    assert any("Invalid unit" in lim for lim in resp.limitations)

    # 2. Non-monotonic timestamps (out of chronological order)
    good_thresh = ThresholdDefinition(parameter="torque", operator=ThresholdOperator.GTE, value=50.0, unit="Nm")
    bad_time_obs = [
        ObservationItem(timestamp=t0 + timedelta(seconds=60), values={"torque": 30.0}, units={"torque": "Nm"}),
        ObservationItem(timestamp=t0, values={"torque": 35.0}, units={"torque": "Nm"}),
    ]
    resp2 = service.forecast_breach(
        ForecastRequest(asset_id="m1", observations=bad_time_obs, threshold=good_thresh)
    )
    assert resp2.status == ForecastStatus.INVALID_INPUT
    assert any("strictly increasing" in lim for lim in resp2.limitations)


def test_crossing_policies_sustained_and_persistent():
    """Verify sustained crossing policy requires consecutive persistence."""
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    # Exceedance occurs only for 1 point at t=10, then drops below, then stays above from t=30 to t=50
    values = [290.0, 315.0, 295.0, 312.0, 314.0, 316.0]
    thresh = ThresholdDefinition(
        parameter="process_temperature",
        operator=ThresholdOperator.GTE,
        value=310.0,
        unit="K",
        persist_steps=3,
        persist_duration_seconds=20.0,
    )
    # First crossing would be at t=10 (or interpolated between 0 and 10)
    idx_first, t_first = find_crossing_index(times, values, thresh, CrossingPolicy.FIRST_CROSSING)
    assert idx_first is not None and t_first < 20.0

    # Sustained crossing should reject the single spike at t=10 and identify t=30 as the start
    idx_sustained, t_sustained = find_crossing_index(times, values, thresh, CrossingPolicy.SUSTAINED_CROSSING)
    assert idx_sustained == 3
    assert t_sustained == 30.0


def test_physical_failure_separate_from_threshold(service):
    """Verify that physical time-to-failure is treated as a distinct target
    and refuses ungrounded requests without physical degradation models.
    """
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = [
        ObservationItem(timestamp=t0, values={"tool_wear": 50.0, "rotational_speed_rpm": 1500.0}),
        ObservationItem(timestamp=t0 + timedelta(seconds=60), values={"tool_wear": 55.0, "rotational_speed_rpm": 1500.0}),
    ]

    # 1. Unsupported degradation model refusal
    bad_req = PhysicalFailureRequest(
        asset_id="mach_01",
        observations=obs,
        degradation_model="unsupported_neural_blackbox",
    )
    resp_unsupported = service.evaluate_failure(bad_req)
    assert resp_unsupported.status == ForecastStatus.UNSUPPORTED_TARGET
    assert any("not a registered physical degradation model" in r for r in resp_unsupported.refusal_reasons)

    # 2. Inadequate evidence (missing Taylor constants n and C)
    incomplete_taylor = PhysicalFailureRequest(
        asset_id="mach_01",
        observations=obs,
        degradation_model="taylor_tool_life",
        calibrated_parameters={},  # Missing n and C!
    )
    resp_inadequate = service.evaluate_failure(incomplete_taylor)
    assert resp_inadequate.status == ForecastStatus.INADEQUATE_EVIDENCE
    assert any("taylor_exponent_n" in r for r in resp_inadequate.refusal_reasons)

    # 3. Successful evaluation with validated Taylor model parameters
    valid_taylor = PhysicalFailureRequest(
        asset_id="mach_01",
        observations=obs,
        degradation_model="taylor_tool_life",
        operating_speed_rpm=1500.0,
        calibrated_parameters={
            "taylor_exponent_n": 0.25,
            "taylor_constant_C": 400.0,
            "tool_diameter_mm": 20.0,
        },
        forecast_horizon_seconds=50000.0,
    )
    resp_valid = service.evaluate_failure(valid_taylor)
    assert resp_valid.status in (ForecastStatus.BREACH_PREDICTED, ForecastStatus.NO_BREACH_WITHIN_HORIZON)
    assert resp_valid.estimated_rul_seconds is not None
    assert resp_valid.estimated_rul_seconds > 0.0
    assert len(resp_valid.validated_assumptions) > 0
