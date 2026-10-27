"""Feasibility bounds and physical coupling constraints for disturbance search."""
from __future__ import annotations

import math
from typing import Dict, List, Tuple
from .models import DisturbanceVariable, PhysicalConstraint


def project_to_bounds(
    delta_x: Dict[str, float],
    baseline_state: Dict[str, float],
    variables: List[DisturbanceVariable],
) -> Dict[str, float]:
    """Project delta_x so that x_0 + delta_x strictly respects [x_min, x_max]."""
    projected = {}
    var_map = {v.name: v for v in variables}

    for name, dx in delta_x.items():
        var = var_map.get(name)
        if var is None:
            projected[name] = dx
            continue
        x0 = baseline_state.get(name, 0.0)
        clamped_x = max(var.min_val, min(var.max_val, x0 + dx))
        projected[name] = clamped_x - x0

    return projected


def check_feasibility(
    state: Dict[str, float],
    variables: List[DisturbanceVariable],
    constraints: List[PhysicalConstraint],
) -> Tuple[bool, List[str]]:
    """Verify whether a candidate perturbed state satisfies all bounds and coupling constraints.
    
    Returns:
        (is_feasible, list_of_violations)
    """
    violations: List[str] = []

    # 1. Variable bounds
    for var in variables:
        val = state.get(var.name)
        if val is not None:
            if val < var.min_val - 1e-6:
                violations.append(f"Variable '{var.name}' ({val}) below lower bound {var.min_val}")
            elif val > var.max_val + 1e-6:
                violations.append(f"Variable '{var.name}' ({val}) above upper bound {var.max_val}")

    # 2. Coupled physics constraints
    air_t = state.get("air_temperature") or state.get("air_temperature_k")
    proc_t = state.get("process_temperature") or state.get("process_temperature_k")
    if air_t is not None and proc_t is not None:
        if proc_t < air_t - 0.5:
            violations.append(f"Thermal violation: Process temperature ({proc_t} K) cannot be significantly lower than ambient air temperature ({air_t} K).")

    speed = state.get("rotational_speed") or state.get("rotational_speed_rpm")
    torque = state.get("torque") or state.get("torque_nm")
    if speed is not None and torque is not None:
        if speed < 0:
            violations.append(f"Kinematic violation: Rotational speed cannot be negative ({speed} RPM).")
        if torque < 0:
            violations.append(f"Dynamic violation: Torque cannot be negative ({torque} Nm).")

        # Milling spindle electrical power constraint P = 2 * pi * N * T / 60
        power_watts = (2.0 * math.pi * speed * torque) / 60.0
        # Check explicit user power constraints if specified
        for c in constraints:
            if c.max_power_watts and power_watts > c.max_power_watts:
                violations.append(f"Mechanical power violation: Spindle power {round(power_watts, 1)} W exceeds limit {c.max_power_watts} W.")

    return len(violations) == 0, violations
