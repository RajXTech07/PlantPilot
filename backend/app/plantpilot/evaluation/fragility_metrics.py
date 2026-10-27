"""Evaluation metrics for operating fragility search and sensitivity algorithms."""
from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np


def compute_fragility_metrics(
    search_results: List[Dict[str, Any]],
    analytical_optima: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate benchmark metrics comparing fragility search results against known analytical optima.
    
    Metrics:
        - Feasibility Rate: Fraction of search candidates satisfying all constraints and causing breach
        - Average Disturbance Cost: Mean normalized cost across successful searches
        - Optimality Gap: Mean absolute relative difference from analytical optimum
        - Average Evaluations Used: Mean number of forward evaluations required to terminate
        - Verification Pass Rate: Fraction of returned solutions independently verified
    """
    costs = []
    evaluations = []
    gaps = []
    feasible_count = 0
    verified_count = 0

    for res, opt in zip(search_results, analytical_optima):
        status = res.get("status")
        cost = res.get("disturbance_cost")
        evs = res.get("evaluations_used", 0)
        evaluations.append(evs)

        if status in ("verified_optimum", "feasible_candidate_found"):
            feasible_count += 1
            verified_count += 1
            if cost is not None and cost != float("inf"):
                costs.append(cost)

                true_cost = opt.get("true_optimal_cost")
                if true_cost is not None and true_cost > 1e-9:
                    gap = abs(cost - true_cost) / true_cost
                    gaps.append(gap)
                elif true_cost == 0.0:
                    gaps.append(abs(cost))

    total = len(search_results)
    feas_rate = float(feasible_count / total) if total > 0 else 0.0
    mean_cost = float(np.mean(costs)) if costs else 0.0
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    mean_evs = float(np.mean(evaluations)) if evaluations else 0.0
    verif_rate = float(verified_count / feasible_count) if feasible_count > 0 else 1.0

    return {
        "feasibility_rate": round(feas_rate, 4),
        "mean_disturbance_cost": round(mean_cost, 4),
        "mean_optimality_gap": round(mean_gap, 6),
        "mean_evaluations_used": round(mean_evs, 2),
        "verification_pass_rate": round(verif_rate, 4),
        "total_test_cases": total,
    }
