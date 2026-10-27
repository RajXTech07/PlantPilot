"""Parameter fragility ranking and single-variable sensitivity analysis."""
from __future__ import annotations

from typing import List, Optional
from ..prognostics.threshold import safety_margin
from .models import (
    OptimalityStatus,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
    SensitivityRankItem,
)
from .search import solve_1d_fragility


def analyze_parameter_sensitivities(request: SensitivityAnalysisRequest) -> SensitivityAnalysisResponse:
    """Evaluate and rank operating variables by their individual fragility (smallest disturbance needed to breach)."""
    x0 = request.baseline_state
    thresh = request.threshold
    vars_list = request.variables

    base_margin = safety_margin(x0.get(thresh.parameter, 0.0), thresh)

    ranked_items: List[SensitivityRankItem] = []

    for var in vars_list:
        status, dx_dict, cost, _ = solve_1d_fragility(
            baseline_state=x0,
            threshold=thresh,
            var=var,
            constraints=[],
            cost_norm=1,
        )
        dx_val = dx_dict.get(var.name, 0.0)
        breach_possible = status == OptimalityStatus.VERIFIED_OPTIMUM

        ranked_items.append(
            SensitivityRankItem(
                variable=var.name,
                min_disturbance_needed=round(dx_val, 4),
                normalized_cost=round(cost, 4) if breach_possible else float("inf"),
                rank=0,  # Will assign after sorting
                direction="increase" if dx_val > 0 else "decrease",
                breach_achieved=breach_possible,
            )
        )

    # Sort items by normalized cost ascending (lowest cost = most fragile = rank 1)
    ranked_items.sort(key=lambda item: item.normalized_cost)
    for idx, item in enumerate(ranked_items):
        item.rank = idx + 1

    most_fragile = ranked_items[0].variable if ranked_items and ranked_items[0].breach_achieved else None

    return SensitivityAnalysisResponse(
        asset_id=request.asset_id,
        baseline_margin=round(base_margin, 4),
        ranked_variables=ranked_items,
        most_fragile_variable=most_fragile,
        limitations=[
            "Single-variable sensitivity evaluates each dimension independently without multivariate coupling interactions.",
            "Rank 1 identifies the operating parameter with the smallest normalized distance to the threshold boundary.",
        ],
    )
