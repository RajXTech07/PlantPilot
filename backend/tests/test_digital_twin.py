"""Tests for persistent digital-twin records and what-if simulation (digital_twin.py)."""

import pytest
from plantpilot.digital_twin import TwinStore, get_twin_store, reset_twin_store


def test_twin_store_crud():
    """Test TwinStore creation, retrieval, and listing."""
    store = get_twin_store()
    twin = store.create("Test Milling Unit", "M", {"bay": 4})
    assert twin is not None
    assert twin["name"] == "Test Milling Unit"
    assert twin["machine_type"] == "M"
    assert twin["metadata"] == {"bay": 4}
    assert twin["observed_reading_count"] == 0
    assert twin["latest_observed_reading"] is None

    # Get by ID
    loaded = store.get(twin["id"])
    assert loaded["id"] == twin["id"]
    assert loaded["name"] == twin["name"]

    # List twins
    all_twins = store.list(limit=10)
    assert any(t["id"] == twin["id"] for t in all_twins)


def test_twin_reading_isolation_observed_vs_simulated():
    """Test that observed readings update state while simulated readings remain strictly isolated."""
    store = get_twin_store()
    twin = store.create("Stamping Press 01", "H", {"line": 1})
    initial_updated_at = twin["updated_at"]

    # 1. Add simulated reading
    sim_reading = {
        "air_temperature_k": 305.0,
        "torque_nm": 80.0,
    }
    sim_assessment = {"failure_risk": "elevated", "failure_probability": 0.88}
    sim_rec = store.add_reading(
        twin["id"],
        sim_reading,
        sim_assessment,
        evidence_type="simulated",
    )
    assert sim_rec["evidence_type"] == "simulated"

    # Twin observed status must remain completely unchanged
    twin_after_sim = store.get(twin["id"])
    assert twin_after_sim["observed_reading_count"] == 0
    assert twin_after_sim["latest_observed_reading"] is None
    assert twin_after_sim["updated_at"] == initial_updated_at

    # 2. Add observed reading
    obs_reading = {
        "air_temperature_k": 298.0,
        "torque_nm": 42.0,
    }
    obs_assessment = {"failure_risk": "not_elevated", "failure_probability": 0.02}
    obs_rec = store.add_reading(
        twin["id"],
        obs_reading,
        obs_assessment,
        evidence_type="observed",
    )
    assert obs_rec["evidence_type"] == "observed"

    # Twin observed status is updated
    twin_after_obs = store.get(twin["id"])
    assert twin_after_obs["observed_reading_count"] == 1
    assert twin_after_obs["latest_observed_reading"] is not None
    assert twin_after_obs["latest_observed_reading"]["id"] == obs_rec["id"]


def test_twin_history_retrieval_and_limits():
    """Test history querying, ordering, and limits."""
    store = get_twin_store()
    twin = store.create("CNC Lathe 05", "L", {})

    for i in range(5):
        store.add_reading(
            twin["id"],
            {"reading_index": i},
            {"status": "ok"},
            evidence_type="observed",
        )

    history = store.history(twin["id"], limit=3)
    assert len(history) == 3
    # Ordered descending by created_at
    assert history[0]["reading"]["reading_index"] == 4
    assert history[1]["reading"]["reading_index"] == 3


def test_twin_not_found_handling():
    """Test that querying or adding readings for unknown twin IDs returns None."""
    store = get_twin_store()
    assert store.get("twin_nonexistent") is None
    assert store.history("twin_nonexistent") is None
    assert store.add_reading("twin_nonexistent", {}, {}) is None


def test_twin_store_singleton_and_reset():
    """Test get_twin_store singleton behavior and reset."""
    s1 = get_twin_store()
    s2 = get_twin_store()
    assert s1 is s2

    reset_twin_store()
    s3 = get_twin_store()
    assert s3 is not s1
