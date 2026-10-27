"""Weighted disturbance cost functions and gradient calculations."""
from __future__ import annotations

import math
from typing import Dict, List
import numpy as np

from .models import DisturbanceVariable


def compute_disturbance_cost(
    delta_x: Dict[str, float],
    variables: List[DisturbanceVariable],
    cost_norm: int = 2,
) -> float:
    """Compute the weighted normalized disturbance cost C(Delta x).
    
    C(Delta x) = sum_i w_i * |Delta x_i / s_i|^p  (or L2 Euclidean norm if p=2).
    """
    total = 0.0
    var_map = {v.name: v for v in variables}

    for name, dx in delta_x.items():
        var = var_map.get(name)
        if var is None:
            continue
        w = var.weight
        s = var.nominal_scale
        ratio = abs(dx) / s

        if cost_norm == 1:
            total += w * ratio
        else:
            total += w * (ratio ** 2)

    if cost_norm == 2:
        return float(math.sqrt(total))
    return float(total)


def compute_cost_gradient(
    delta_x: np.ndarray,
    weights: np.ndarray,
    scales: np.ndarray,
    cost_norm: int = 2,
) -> np.ndarray:
    """Compute analytical gradient of disturbance cost with respect to delta_x."""
    scaled = delta_x / scales
    if cost_norm == 1:
        return weights * np.sign(scaled) / scales
    else:
        # For L2 norm sqrt(sum w * (dx/s)^2): grad = (w * dx / s^2) / norm
        norm_sq = np.sum(weights * (scaled ** 2))
        norm = math.sqrt(norm_sq) if norm_sq > 1e-12 else 1e-6
        return (weights * delta_x / (scales ** 2)) / norm
