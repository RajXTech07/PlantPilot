"""Tests for ML failure-risk inference and anomaly evidence (detection.py)."""

import math
import pytest
from plantpilot.detection import normalize_reading, predict, _models, _iqr_references


def test_normalize_reading_valid():
    """Test normalising valid readings with alias mappings."""
    reading = {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
    }
    vector, values = normalize_reading(reading)
    assert len(vector) == 5
    assert vector == [298.1, 308.6, 1551.0, 42.8, 0.0]
    assert values["air_temperature_k"] == 298.1


def test_normalize_reading_raw_column_aliases():
    """Test normalize_reading handles raw AI4I 2020 column names."""
    reading = {
        "Air temperature [K]": "300.5",
        "Process temperature [K]": "310.2",
        "Rotational speed [rpm]": "1420",
        "Torque [Nm]": "49.1",
        "Tool wear [min]": "15",
    }
    vector, values = normalize_reading(reading)
    assert vector == [300.5, 310.2, 1420.0, 49.1, 15.0]


def test_normalize_reading_missing_field():
    """Test normalize_reading raises ValueError on missing features."""
    with pytest.raises(ValueError, match="Missing required measurement"):
        normalize_reading({"air_temperature_k": 298.0})


def test_normalize_reading_invalid_number():
    """Test normalize_reading rejects non-numeric or non-finite inputs."""
    with pytest.raises(ValueError, match="must be numeric"):
        normalize_reading({
            "air_temperature_k": "not-a-number",
            "process_temperature_k": 308.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40.0,
            "tool_wear_min": 10,
        })

    with pytest.raises(ValueError, match="must be finite"):
        normalize_reading({
            "air_temperature_k": float("nan"),
            "process_temperature_k": 308.0,
            "rotational_speed_rpm": 1500,
            "torque_nm": 40.0,
            "tool_wear_min": 10,
        })


def test_models_lazy_initialization():
    """Test that models are trained, threshold optimized, and cached via lru_cache."""
    hgb, rf, hgb_thresh, rf_thresh = _models()
    assert hgb is not None
    assert rf is not None
    assert 0.0 < hgb_thresh < 1.0
    assert 0.0 < rf_thresh < 1.0

    # Test lru_cache returns same instances
    hgb2, rf2, hgb_thresh2, rf_thresh2 = _models()
    assert hgb is hgb2
    assert rf is rf2
    assert hgb_thresh == hgb_thresh2


def test_predict_single_model():
    """Test predict() with primary model only."""
    reading = {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551.0,
        "torque_nm": 42.8,
        "tool_wear_min": 0.0,
    }
    result = predict(reading, include_secondary=False)
    assert result["model"] == "HistGradientBoostingClassifier"
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["failure_risk"] in ["elevated", "not_elevated"]
    assert isinstance(result["predicted_failure"], bool)
    assert "anomaly_evidence" in result
    assert result["secondary_model"] is None
    assert "model_agreement" not in result


def test_predict_with_secondary_consensus():
    """Test predict() with secondary RandomForest consensus."""
    reading = {
        "air_temperature_k": 304.5,
        "process_temperature_k": 313.5,
        "rotational_speed_rpm": 1200.0,
        "torque_nm": 70.0,
        "tool_wear_min": 220.0,
    }
    result = predict(reading, include_secondary=True)
    assert result["secondary_model"] is not None
    assert result["secondary_model"]["model"] == "RandomForestClassifier"
    assert "model_agreement" in result
    assert isinstance(result["model_agreement"], bool)
    assert "flagged_parameters" in result["anomaly_evidence"]


def test_iqr_references_cached():
    """Test _iqr_references returns fitted reference statistics."""
    refs = _iqr_references()
    assert "air_temperature_k" in refs
    assert "lower" in refs["air_temperature_k"]
    assert "upper" in refs["air_temperature_k"]
