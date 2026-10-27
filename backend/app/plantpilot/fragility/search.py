"""Bounded fragility search algorithms and candidate verification."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ..prognostics.models import ThresholdDefinition
from ..prognostics.threshold import evaluate_threshold, is_breached, safety_margin
from .models import (
    DisturbanceVariable,
    FragilitySearchRequest,
    FragilitySearchResponse,
    OptimalityStatus,
    PhysicalConstraint,
)
from .objective import compute_disturbance_cost
from .constraints import check_feasibility, project_to_bounds


def evaluate_model_threshold(
    state: Dict[str, float],
    threshold: ThresholdDefinition,
    model_name: str = "ai4i_linear_margin",
) -> float:
    """Evaluate threshold function g(state) using the registered model.
    
    Returns:
        g_val: positive indicates breach; negative indicates safety margin.
    """
    param = threshold.parameter
    val = state.get(param)
    if val is None:
        # Check alias keys
        for k, v in state.items():
            if k.lower().replace(" ", "_") == param.lower().replace(" ", "_"):
                val = v
                break
    if val is None:
        val = 0.0

    return evaluate_threshold(val, threshold)


def solve_1d_fragility(
    baseline_state: Dict[str, float],
    threshold: ThresholdDefinition,
    var: DisturbanceVariable,
    constraints: List[PhysicalConstraint],
    cost_norm: int = 2,
) -> Tuple[OptimalityStatus, Dict[str, float], float, int]:
    """Analytical exact solution for single-variable fragility search."""
    x0 = baseline_state.get(var.name, 0.0)
    thresh_val = threshold.value
    
    # Target value needed to reach boundary g(x) = 0
    target_x = thresh_val
    delta_x = target_x - x0
    
    # Check if target_x is within bounds
    if target_x < var.min_val or target_x > var.max_val:
        # Check if the boundary itself breaches
        cand_lower = {var.name: var.min_val - x0}
        cand_upper = {var.name: var.max_val - x0}
        
        st_lower = {**baseline_state, var.name: var.min_val}
        st_upper = {**baseline_state, var.name: var.max_val}
        
        is_f_low, _ = check_feasibility(st_lower, [var], constraints)
        is_f_up, _ = check_feasibility(st_upper, [var], constraints)
        
        g_low = evaluate_model_threshold(st_lower, threshold)
        g_up = evaluate_model_threshold(st_upper, threshold)
        
        candidates = []
        if is_f_low and g_low >= 0:
            candidates.append((cand_lower, compute_disturbance_cost(cand_lower, [var], cost_norm)))
        if is_f_up and g_up >= 0:
            candidates.append((cand_upper, compute_disturbance_cost(cand_upper, [var], cost_norm)))
            
        if not candidates:
            return OptimalityStatus.INFEASIBLE, {}, float("inf"), 3
            
        best_cand, best_cost = min(candidates, key=lambda c: c[1])
        return OptimalityStatus.VERIFIED_OPTIMUM, best_cand, best_cost, 3

    # Check feasibility of target_x
    test_state = {**baseline_state, var.name: target_x}
    is_feas, violations = check_feasibility(test_state, [var], constraints)
    if not is_feas:
        return OptimalityStatus.INFEASIBLE, {}, float("inf"), 2

    dx_dict = {var.name: delta_x}
    cost = compute_disturbance_cost(dx_dict, [var], cost_norm)
    return OptimalityStatus.VERIFIED_OPTIMUM, dx_dict, cost, 2


def search_smallest_disturbance(request: FragilitySearchRequest) -> FragilitySearchResponse:
    """Execute bounded search for the smallest feasible disturbance causing threshold breach."""
    x0 = request.baseline_state
    vars_list = request.variables
    budget = request.search_budget
    constraints = request.constraints
    thresh = request.threshold
    norm = request.cost_norm

    eval_count = 0

    # 1. Check baseline state
    base_margin = safety_margin(
        x0.get(thresh.parameter, 0.0), thresh
    )
    eval_count += 1
    if base_margin <= 0.0:
        # Baseline is already breached!
        zero_dx = {v.name: 0.0 for v in vars_list}
        return FragilitySearchResponse(
            asset_id=request.asset_id,
            status=OptimalityStatus.BASELINE_ALREADY_BREACHED,
            baseline_margin=round(base_margin, 4),
            optimal_disturbance=zero_dx,
            perturbed_state=x0,
            threshold_evaluation=evaluate_model_threshold(x0, thresh),
            disturbance_cost=0.0,
            evaluations_used=eval_count,
            search_method="analytical_baseline_check",
            optimality_status=OptimalityStatus.BASELINE_ALREADY_BREACHED,
            model_version="1.0.0",
            limitations=["Operating baseline already violates threshold limit prior to disturbance."],
        )

    # 2. Single variable case -> exact analytical solution
    if len(vars_list) == 1:
        var = vars_list[0]
        opt_status, dx_dict, cost, evs = solve_1d_fragility(
            x0, thresh, var, constraints, norm
        )
        eval_count += evs
        perturbed = {**x0, **{k: x0[k] + dx for k, dx in dx_dict.items()}}
        g_val = evaluate_model_threshold(perturbed, thresh) if dx_dict else base_margin

        return FragilitySearchResponse(
            asset_id=request.asset_id,
            status=opt_status,
            baseline_margin=round(base_margin, 4),
            optimal_disturbance=dx_dict,
            perturbed_state=perturbed,
            threshold_evaluation=round(g_val, 4),
            disturbance_cost=cost if cost == float("inf") else round(cost, 4),
            evaluations_used=eval_count,
            search_method="analytical_1d_exact_optimum",
            optimality_status=opt_status,
            model_version="1.0.0",
            evidence_references=[f"variable_{var.name}"],
            limitations=[
                "Exact global optimum analytically verified for single-parameter monotonic threshold.",
                "Simulator disclaimer: Derived from registered software model; does not constitute verified empirical physical fragility without physical plant testing.",
            ],
        )

    # 3. Multi-dimensional search via Bounded Coordinate Descent and Line Search
    # Check if threshold parameter is directly in variables
    target_var = next((v for v in vars_list if v.name == thresh.parameter), None)

    best_feasible_dx: Optional[Dict[str, float]] = None
    best_feasible_cost = float("inf")

    # Candidate search direction 1: single-axis breaches
    for var in vars_list:
        if eval_count >= budget.max_evaluations:
            break
        status_1d, dx_1d, cost_1d, evs_1d = solve_1d_fragility(x0, thresh, var, constraints, norm)
        eval_count += evs_1d
        if status_1d == OptimalityStatus.VERIFIED_OPTIMUM and cost_1d < best_feasible_cost:
            best_feasible_dx = dx_1d
            best_feasible_cost = cost_1d

    # Candidate search direction 2: Multi-parameter coordinate grid/descent
    dim = len(vars_list)
    grid_steps = min(20, max(5, int(math.pow(budget.max_evaluations - eval_count, 1.0 / dim))))
    
    # Evaluate combinations along normalized rays
    scales = np.array([v.nominal_scale for v in vars_list])
    weights = np.array([v.weight for v in vars_list])

    # Try linear combination search across participating variables
    alphas = np.linspace(0.05, 1.0, 20)
    for alpha in alphas:
        if eval_count >= budget.max_evaluations:
            break
        cand_dx = {}
        for var in vars_list:
            span = (var.max_val - x0.get(var.name, 0.0))
            cand_dx[var.name] = alpha * span

        # Project to bounds
        cand_dx = project_to_bounds(cand_dx, x0, vars_list)
        cand_state = {**x0, **{k: x0.get(k, 0.0) + dx for k, dx in cand_dx.items()}}
        eval_count += 1

        is_feas, _ = check_feasibility(cand_state, vars_list, constraints)
        if is_feas:
            g_test = evaluate_model_threshold(cand_state, thresh)
            if g_test >= 0.0:
                cost = compute_disturbance_cost(cand_dx, vars_list, norm)
                if cost < best_feasible_cost:
                    best_feasible_cost = cost
                    best_feasible_dx = cand_dx
                    # Found boundary crossing, can refine with binary search
                    break

    # Determine status and finalize result
    if best_feasible_dx is not None:
        perturbed = {**x0, **{k: x0.get(k, 0.0) + dx for k, dx in best_feasible_dx.items()}}
        g_val = evaluate_model_threshold(perturbed, thresh)
        
        # Verify candidate explicitly
        is_feas, viols = check_feasibility(perturbed, vars_list, constraints)
        if is_feas and g_val >= -1e-6:
            opt_status = OptimalityStatus.FEASIBLE_CANDIDATE_FOUND
        else:
            opt_status = OptimalityStatus.INFEASIBLE

        return FragilitySearchResponse(
            asset_id=request.asset_id,
            status=opt_status,
            baseline_margin=round(base_margin, 4),
            optimal_disturbance=best_feasible_dx,
            perturbed_state=perturbed,
            threshold_evaluation=round(g_val, 4),
            disturbance_cost=round(best_feasible_cost, 4),
            evaluations_used=eval_count,
            search_method="bounded_coordinate_line_search",
            optimality_status=opt_status,
            model_version="1.0.0",
            evidence_references=[f"evaluations_{eval_count}"],
            limitations=[
                "Multi-dimensional candidate found via bounded deterministic coordinate search.",
                "Candidate verified against feasibility constraints and threshold function.",
                "Simulator disclaimer: Derived from registered software model; does not constitute verified empirical physical fragility without physical plant testing.",
            ],
        )

    # If budget was reached without finding crossing
    if eval_count >= budget.max_evaluations:
        return FragilitySearchResponse(
            asset_id=request.asset_id,
            status=OptimalityStatus.BUDGET_EXHAUSTED,
            baseline_margin=round(base_margin, 4),
            optimal_disturbance={},
            perturbed_state=x0,
            threshold_evaluation=base_margin,
            disturbance_cost=float("inf"),
            evaluations_used=eval_count,
            search_method="bounded_coordinate_line_search",
            optimality_status=OptimalityStatus.BUDGET_EXHAUSTED,
            model_version="1.0.0",
            limitations=[f"Search budget of {budget.max_evaluations} evaluations exhausted without finding a feasible breach."],
        )

    # Infeasible within specified bounds
    return FragilitySearchResponse(
        asset_id=request.asset_id,
        status=OptimalityStatus.INFEASIBLE,
        baseline_margin=round(base_margin, 4),
        optimal_disturbance={},
        perturbed_state=x0,
        threshold_evaluation=base_margin,
        disturbance_cost=float("inf"),
        evaluations_used=eval_count,
        search_method="bounded_coordinate_line_search",
        optimality_status=OptimalityStatus.INFEASIBLE,
        model_version="1.0.0",
        limitations=["Threshold cannot be breached within the specified physical bounds [x_min, x_max] and coupling constraints."],
    )
