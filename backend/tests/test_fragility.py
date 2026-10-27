"""Unit tests for the Fragility Search Subsystem (Smallest Feasible Disturbance)."""
import pytest

from plantpilot.fragility.models import (
    DisturbanceVariable,
    FragilitySearchRequest,
    OptimalityStatus,
    PhysicalConstraint,
    SearchBudget,
    SensitivityAnalysisRequest,
)
from plantpilot.fragility.service import FragilityService
from plantpilot.prognostics.models import ThresholdDefinition, ThresholdOperator


@pytest.fixture
def service(tmp_path):
    db_file = tmp_path / "fragility_test.db"
    return FragilityService(db_path=db_file)


def test_analytical_optimum_1d_fragility(service):
    """Verify that a 1D monotonic threshold search finds the exact verified analytical optimum."""
    # Baseline torque is 40.0 Nm. Threshold is >= 60.0 Nm.
    # Disturbance variable: torque in [10.0, 80.0], nominal_scale = 10.0, weight = 1.0.
    # Expected disturbance: +20.0 Nm.
    # Expected cost: 20.0 / 10.0 = 2.0.
    thresh = ThresholdDefinition(
        parameter="torque",
        operator=ThresholdOperator.GTE,
        value=60.0,
        unit="Nm",
    )
    var = DisturbanceVariable(
        name="torque",
        min_val=10.0,
        max_val=80.0,
        nominal_scale=10.0,
        weight=1.0,
    )
    req = FragilitySearchRequest(
        asset_id="mach_cnc_01",
        baseline_state={"torque": 40.0, "rotational_speed": 1500.0},
        threshold=thresh,
        variables=[var],
        cost_norm=2,
    )

    resp = service.search_fragility(req)
    assert resp.status == OptimalityStatus.VERIFIED_OPTIMUM
    assert resp.optimality_status == OptimalityStatus.VERIFIED_OPTIMUM
    assert abs(resp.optimal_disturbance["torque"] - 20.0) < 1e-4
    assert abs(resp.perturbed_state["torque"] - 60.0) < 1e-4
    assert abs(resp.disturbance_cost - 2.0) < 1e-4
    assert resp.threshold_evaluation >= 0.0
    assert any("Derived from registered software model" in lim for lim in resp.limitations)


def test_baseline_already_breached(service):
    """Verify that if the baseline already exceeds the threshold,
    status is BASELINE_ALREADY_BREACHED with zero disturbance and cost.
    """
    thresh = ThresholdDefinition(
        parameter="process_temperature",
        operator=ThresholdOperator.GTE,
        value=310.0,
        unit="K",
    )
    var = DisturbanceVariable(
        name="process_temperature",
        min_val=290.0,
        max_val=330.0,
        nominal_scale=5.0,
    )
    req = FragilitySearchRequest(
        asset_id="mach_cnc_01",
        baseline_state={"process_temperature": 315.0},  # Already breached!
        threshold=thresh,
        variables=[var],
    )

    resp = service.search_fragility(req)
    assert resp.status == OptimalityStatus.BASELINE_ALREADY_BREACHED
    assert resp.disturbance_cost == 0.0
    assert resp.optimal_disturbance["process_temperature"] == 0.0
    assert resp.baseline_margin < 0.0


def test_infeasible_fragility_search(service):
    """Verify that if the threshold cannot be reached within physical bounds,
    the system honestly reports INFEASIBLE.
    """
    thresh = ThresholdDefinition(
        parameter="torque",
        operator=ThresholdOperator.GTE,
        value=100.0,  # Far above upper bound 70.0 Nm!
        unit="Nm",
    )
    var = DisturbanceVariable(
        name="torque",
        min_val=10.0,
        max_val=70.0,  # Max allowable is 70 Nm
        nominal_scale=10.0,
    )
    req = FragilitySearchRequest(
        asset_id="mach_cnc_01",
        baseline_state={"torque": 40.0},
        threshold=thresh,
        variables=[var],
    )

    resp = service.search_fragility(req)
    assert resp.status == OptimalityStatus.INFEASIBLE
    assert resp.disturbance_cost == float("inf")


def test_coupled_physical_constraint_enforcement(service):
    """Verify that physical coupling constraints (e.g. process_temp >= air_temp)
    are respected during disturbance verification.
    """
    thresh = ThresholdDefinition(
        parameter="air_temperature",
        operator=ThresholdOperator.GTE,
        value=315.0,
        unit="K",
    )
    var = DisturbanceVariable(
        name="air_temperature",
        min_val=295.0,
        max_val=330.0,
        nominal_scale=5.0,
    )
    # Air temperature pushed to 315 K while process temperature remains at 305 K
    # violates physical coupling constraint (process temp cannot be < air temp)
    req = FragilitySearchRequest(
        asset_id="mach_cnc_01",
        baseline_state={"air_temperature": 300.0, "process_temperature": 305.0},
        threshold=thresh,
        variables=[var],
    )

    resp = service.search_fragility(req)
    # Target of 315 K air temp with 305 K process temp violates thermal feasibility
    assert resp.status == OptimalityStatus.INFEASIBLE


def test_budget_exhausted_enforcement(service):
    """Verify that deterministic evaluation budget limits prevent runaway execution."""
    thresh = ThresholdDefinition(
        parameter="torque",
        operator=ThresholdOperator.GTE,
        value=75.0,
        unit="Nm",
    )
    v1 = DisturbanceVariable(name="rotational_speed", min_val=1000.0, max_val=2500.0, nominal_scale=200.0)
    v2 = DisturbanceVariable(name="air_temperature", min_val=295.0, max_val=315.0, nominal_scale=5.0)

    # Threshold is on torque, but neither variable is torque and budget is set to tiny value (2)
    req = FragilitySearchRequest(
        asset_id="mach_cnc_01",
        baseline_state={"torque": 40.0, "rotational_speed": 1500.0, "air_temperature": 300.0},
        threshold=thresh,
        variables=[v1, v2],
        search_budget=SearchBudget(max_evaluations=10),
    )

    resp = service.search_fragility(req)
    assert resp.status in (OptimalityStatus.BUDGET_EXHAUSTED, OptimalityStatus.INFEASIBLE)
    assert resp.evaluations_used <= 15


def test_parameter_fragility_sensitivity_ranking(service):
    """Verify single-parameter sensitivity ranking identifies the most fragile parameter."""
    thresh = ThresholdDefinition(
        parameter="torque",
        operator=ThresholdOperator.GTE,
        value=50.0,
        unit="Nm",
    )
    v_torque = DisturbanceVariable(name="torque", min_val=10.0, max_val=80.0, nominal_scale=10.0)
    v_speed = DisturbanceVariable(name="rotational_speed", min_val=1000.0, max_val=2500.0, nominal_scale=200.0)

    req = SensitivityAnalysisRequest(
        asset_id="mach_01",
        baseline_state={"torque": 45.0, "rotational_speed": 1500.0},
        threshold=thresh,
        variables=[v_torque, v_speed],
    )

    resp = service.evaluate_sensitivity(req)
    assert len(resp.ranked_variables) == 2
    # Torque needs only +5.0 Nm (normalized cost 5/10 = 0.5) to breach, whereas speed cannot breach torque
    assert resp.most_fragile_variable == "torque"
    assert resp.ranked_variables[0].variable == "torque"
    assert resp.ranked_variables[0].rank == 1
    assert abs(resp.ranked_variables[0].min_disturbance_needed - 5.0) < 1e-4
