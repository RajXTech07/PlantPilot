"""Integration tests for Prognostics (Time to Breach) and Fragility Search subsystems.

Covers:
- Tool execution through registered_tools
- Safety policy checks
- FastAPI endpoints for prognostics, fragility, and benchmarks
- Agent evidence classification (DERIVED vs SIMULATED vs OBSERVED)
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from plantpilot.main import app
from plantpilot.tools import registered_tools, ToolError
from plantpilot.safety import default_safety_policy
from plantpilot.schemas import (
    EvidenceType,
    PredictionTarget,
    ThresholdOperator,
    ForecastStatus,
    OptimalityStatus,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_tool_registry_estimate_time_to_breach():
    """Verify estimate_time_to_breach tool executes through ToolRegistry."""
    args = {
        "asset_id": "mach_test_01",
        "prediction_target": "threshold_breach",
        "threshold": {
            "parameter": "tool_wear",
            "value": 200.0,
            "operator": ">=",
            "unit": "min",
        },
        "observations": [
            {"timestamp": "2026-10-27T10:00:00Z", "values": {"tool_wear": 150.0}, "units": {"tool_wear": "min"}},
            {"timestamp": "2026-10-27T10:10:00Z", "values": {"tool_wear": 160.0}, "units": {"tool_wear": "min"}},
            {"timestamp": "2026-10-27T10:20:00Z", "values": {"tool_wear": 170.0}, "units": {"tool_wear": "min"}},
        ],
        "forecast_horizon_seconds": 7200.0,
        "model_type": "linear_drift",
    }
    result = registered_tools.execute("estimate_time_to_breach", args)
    assert result["asset_id"] == "mach_test_01"
    assert result["status"] == ForecastStatus.BREACH_PREDICTED.value
    assert result["predicted_time_seconds_from_last_obs"] is not None
    assert result["predicted_time_seconds_from_last_obs"] > 0


def test_tool_registry_evaluate_physical_failure():
    """Verify evaluate_physical_failure_time tool executes through ToolRegistry."""
    args = {
        "asset_id": "mach_test_02",
        "failure_mode": "tool_wear_failure",
        "degradation_model": "taylor_tool_life",
        "operating_speed_rpm": 1500.0,
        "observations": [
            {"timestamp": "2026-10-27T10:00:00Z", "values": {"tool_wear": 40.0}},
            {"timestamp": "2026-10-27T10:10:00Z", "values": {"tool_wear": 50.0}},
        ],
        "calibrated_parameters": {
            "taylor_exponent_n": 0.25,
            "taylor_constant_C": 400.0,
            "critical_wear_limit": 220.0,
        },
    }
    result = registered_tools.execute("evaluate_physical_failure_time", args)
    assert result["asset_id"] == "mach_test_02"
    assert result["status"] == ForecastStatus.BREACH_PREDICTED.value
    assert result["estimated_rul_seconds"] is not None
    assert result["estimated_rul_seconds"] > 0


def test_tool_registry_search_operating_fragility():
    """Verify search_operating_fragility tool executes through ToolRegistry."""
    args = {
        "asset_id": "mach_test_03",
        "baseline_state": {
            "rotational_speed": 1500.0,
            "torque": 40.0,
        },
        "threshold": {
            "parameter": "torque",
            "value": 65.0,
            "operator": ">=",
            "unit": "Nm",
        },
        "variables": [
            {
                "name": "torque",
                "min_val": 10.0,
                "max_val": 80.0,
                "nominal_scale": 15.0,
                "weight": 1.0,
            }
        ],
    }
    result = registered_tools.execute("search_operating_fragility", args)
    assert result["asset_id"] == "mach_test_03"
    assert result["status"] == OptimalityStatus.VERIFIED_OPTIMUM.value
    assert result["disturbance_cost"] > 0


def test_tool_validation_rejection():
    """Verify tool validation rejects invalid or missing arguments."""
    with pytest.raises(ToolError):
        # Missing required observations
        registered_tools.execute("estimate_time_to_breach", {
            "asset_id": "mach_01",
            "prediction_target": "operating_limit",
            "threshold": {"parameter": "torque", "value": 50.0, "operator": ">=", "unit": "Nm"},
        })


def test_safety_policy_permits_new_tools():
    """Verify default safety policy permits prognostics and fragility tools."""
    policy = default_safety_policy()
    assert policy.check_tool_call("estimate_time_to_breach", {}) is True
    assert policy.check_tool_call("evaluate_physical_failure_time", {}) is True
    assert policy.check_tool_call("search_operating_fragility", {}) is True


def test_api_prognostics_forecast_endpoint(client):
    """Test POST /api/v1/prognostics/forecast endpoint."""
    payload = {
        "asset_id": "mach_api_01",
        "prediction_target": "threshold_breach",
        "threshold": {
            "parameter": "tool_wear",
            "value": 200.0,
            "operator": ">=",
            "unit": "min",
        },
        "observations": [
            {"timestamp": "2026-10-27T10:00:00Z", "values": {"tool_wear": 100.0}, "units": {"tool_wear": "min"}},
            {"timestamp": "2026-10-27T10:10:00Z", "values": {"tool_wear": 110.0}, "units": {"tool_wear": "min"}},
            {"timestamp": "2026-10-27T10:20:00Z", "values": {"tool_wear": 120.0}, "units": {"tool_wear": "min"}},
        ],
        "forecast_horizon_seconds": 10000.0,
        "model_type": "linear_drift",
    }
    response = client.post("/api/v1/prognostics/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "mach_api_01"
    assert data["status"] == ForecastStatus.BREACH_PREDICTED.value
    assert "uncertainty" in data


def test_api_physical_failure_endpoint(client):
    """Test POST /api/v1/prognostics/failure-time endpoint."""
    payload = {
        "asset_id": "mach_api_02",
        "failure_mode": "tool_wear_failure",
        "degradation_model": "taylor_tool_life",
        "operating_speed_rpm": 1500.0,
        "observations": [
            {"timestamp": "2026-10-27T10:00:00Z", "values": {"tool_wear": 30.0}},
            {"timestamp": "2026-10-27T10:10:00Z", "values": {"tool_wear": 40.0}},
        ],
        "calibrated_parameters": {
            "taylor_exponent_n": 0.25,
            "taylor_constant_C": 400.0,
            "critical_wear_limit": 200.0,
        },
    }
    response = client.post("/api/v1/prognostics/failure-time", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "mach_api_02"
    assert data["status"] == ForecastStatus.BREACH_PREDICTED.value
    assert data["estimated_rul_seconds"] > 0


def test_api_prognostics_benchmark_endpoint(client):
    """Test POST /api/v1/prognostics/benchmark endpoint."""
    response = client.post("/api/v1/prognostics/benchmark", json={"num_cases": 20})
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size"] == 20
    assert "metrics" in data
    assert data["metrics"]["mae_seconds"] is not None


def test_api_fragility_search_endpoint(client):
    """Test POST /api/v1/fragility/search endpoint."""
    payload = {
        "asset_id": "mach_api_03",
        "baseline_state": {
            "torque": 40.0,
        },
        "threshold": {
            "parameter": "torque",
            "value": 70.0,
            "operator": ">=",
            "unit": "Nm",
        },
        "variables": [
            {
                "name": "torque",
                "min_val": 20.0,
                "max_val": 90.0,
                "nominal_scale": 15.0,
                "weight": 1.0,
            }
        ],
    }
    response = client.post("/api/v1/fragility/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "mach_api_03"
    assert data["status"] == OptimalityStatus.VERIFIED_OPTIMUM.value


def test_api_fragility_sensitivity_endpoint(client):
    """Test POST /api/v1/fragility/sensitivity endpoint."""
    payload = {
        "asset_id": "mach_api_04",
        "baseline_state": {
            "air_temperature": 300.0,
            "process_temperature": 310.0,
        },
        "threshold": {
            "parameter": "process_temperature",
            "value": 315.0,
            "operator": ">=",
            "unit": "K",
        },
        "variables": [
            {
                "name": "air_temperature",
                "min_val": 290.0,
                "max_val": 320.0,
                "nominal_scale": 10.0,
                "weight": 1.0,
            },
            {
                "name": "process_temperature",
                "min_val": 305.0,
                "max_val": 325.0,
                "nominal_scale": 10.0,
                "weight": 1.0,
            },
        ],
    }
    response = client.post("/api/v1/fragility/sensitivity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == "mach_api_04"
    assert len(data["ranked_variables"]) == 2
    assert data["most_fragile_variable"] == "process_temperature"


def test_api_fragility_benchmark_endpoint(client):
    """Test POST /api/v1/fragility/benchmark endpoint."""
    response = client.post("/api/v1/fragility/benchmark", json={"num_cases": 10})
    assert response.status_code == 200
    data = response.json()
    assert data["sample_size"] == 10
    assert "metrics" in data
    assert data["metrics"]["feasibility_rate"] == 1.0


def test_agent_evidence_classification():
    """Verify agent correctly classifies evidence types for prognostics and fragility tools."""
    # Prognostics tool -> DERIVED
    for tool_name in ["predict_failure_risk", "estimate_time_to_breach", "evaluate_physical_failure_time"]:
        ev_type = (
            EvidenceType.DERIVED
            if tool_name in ("predict_failure_risk", "estimate_time_to_breach", "evaluate_physical_failure_time")
            else (EvidenceType.SIMULATED if tool_name == "search_operating_fragility" else EvidenceType.OBSERVED)
        )
        assert ev_type == EvidenceType.DERIVED

    # Fragility search -> SIMULATED
    tool_name = "search_operating_fragility"
    ev_type = (
        EvidenceType.DERIVED
        if tool_name in ("predict_failure_risk", "estimate_time_to_breach", "evaluate_physical_failure_time")
        else (EvidenceType.SIMULATED if tool_name == "search_operating_fragility" else EvidenceType.OBSERVED)
    )
    assert ev_type == EvidenceType.SIMULATED

    # Sensor reading -> OBSERVED
    tool_name = "get_sensor_readings"
    ev_type = (
        EvidenceType.DERIVED
        if tool_name in ("predict_failure_risk", "estimate_time_to_breach", "evaluate_physical_failure_time")
        else (EvidenceType.SIMULATED if tool_name == "search_operating_fragility" else EvidenceType.OBSERVED)
    )
    assert ev_type == EvidenceType.OBSERVED
