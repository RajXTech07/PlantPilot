"""Unit tests for prognostics and fragility benchmark evaluation runners and metrics."""
from plantpilot.evaluation.prognostics_metrics import compute_breach_metrics
from plantpilot.evaluation.fragility_metrics import compute_fragility_metrics
from plantpilot.evaluation.benchmark_runner import (
    run_prognostics_synthetic_benchmark,
    run_fragility_synthetic_benchmark,
)


def test_compute_breach_metrics():
    """Verify statistical metrics calculation for breach forecasting."""
    predictions = [
        {"status": "breach_predicted", "predicted_time_seconds": 105.0, "uncertainty": {"lower_seconds": 90.0, "upper_seconds": 120.0}},
        {"status": "breach_predicted", "predicted_time_seconds": 210.0, "uncertainty": {"lower_seconds": 200.0, "upper_seconds": 220.0}},
        {"status": "no_breach_within_horizon", "predicted_time_seconds": None, "uncertainty": None},
    ]
    ground_truth = [
        {"breached": True, "breach_time_seconds": 100.0},
        {"breached": True, "breach_time_seconds": 200.0},
        {"breached": False, "breach_time_seconds": None},
    ]

    metrics = compute_breach_metrics(predictions, ground_truth)
    assert metrics["mae_seconds"] == 7.5  # (5 + 10) / 2
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["prediction_interval_coverage_probability"] == 1.0


def test_compute_fragility_metrics():
    """Verify statistical metrics calculation for fragility search."""
    search_results = [
        {"status": "verified_optimum", "disturbance_cost": 2.0, "evaluations_used": 2},
        {"status": "verified_optimum", "disturbance_cost": 4.02, "evaluations_used": 4},
    ]
    analytical_optima = [
        {"true_optimal_cost": 2.0},
        {"true_optimal_cost": 4.0},
    ]

    metrics = compute_fragility_metrics(search_results, analytical_optima)
    assert metrics["feasibility_rate"] == 1.0
    assert metrics["verification_pass_rate"] == 1.0
    assert abs(metrics["mean_disturbance_cost"] - 3.01) < 1e-4
    assert abs(metrics["mean_evaluations_used"] - 3.0) < 1e-4
    assert metrics["mean_optimality_gap"] < 0.01


def test_run_prognostics_synthetic_benchmark():
    """Verify end-to-end deterministic prognostics benchmark."""
    res = run_prognostics_synthetic_benchmark(num_cases=15)
    assert res["sample_size"] == 15
    metrics = res["metrics"]
    assert metrics["evaluated_cases"] == 15
    assert metrics["mae_seconds"] < 25.0  # High accuracy on linear synthetic trajectories
    assert metrics["false_positive_rate"] == 0.0


def test_run_fragility_synthetic_benchmark():
    """Verify end-to-end deterministic fragility benchmark."""
    res = run_fragility_synthetic_benchmark(num_cases=10)
    assert res["sample_size"] == 10
    metrics = res["metrics"]
    assert metrics["feasibility_rate"] == 1.0
    assert metrics["mean_optimality_gap"] < 1e-5  # Exact analytical match on 1D linear cases
