"""Tool registry — provider-independent.

Each tool is a registered, validated function with explicit permission and
risk metadata. Only tools registered by the backend may execute. The
registry is independent of the model provider and exposes the OpenAI
compatible tool-definition format for the NVIDIA endpoint.

Future sensor-data adapters should implement get_sensor_readings() and
register with a name like "get_station_readings" or "fetch_telemetry".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .safety import RiskLevel, ToolRisk
AGENT_PARAMS = {"air_temperature": "Air temperature [K]", "process_temperature": "Process temperature [K]", "rotational_speed": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]"}


from .data_adapter import get_sensor_readings, DataAdapterError


class ToolError(ValueError):
    """Raised when a tool call is rejected or fails."""


@dataclass
class Tool:
    """A registered tool."""
    name: str
    description: str
    risk_level: RiskLevel
    requires_approval: bool
    handler: Callable[[Dict[str, Any]], Any]
    argument_schema: Dict[str, Any]
    max_input_bytes: int = 4000

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to the OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.argument_schema,
            },
        }

    def to_risk(self) -> ToolRisk:
        return ToolRisk(
            name=self.name,
            risk_level=self.risk_level,
            requires_approval=self.requires_approval,
            description=self.description,
        )


class ToolRegistry:
    """Holds all registered tools and validates/executes calls."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Return all tools in OpenAI-compatible format."""
        return [t.to_openai_format() for t in self._tools.values()]

    def validate_arguments(self, tool_name: str, arguments: Any) -> Dict[str, Any]:
        """Validate and normalize tool arguments.

        Rejects unknown tools, non-dict arguments, and oversized inputs.
        Validates against the tool's declared JSON schema.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool: {tool_name!r}")
        if not isinstance(arguments, dict):
            raise ToolError(
                f"Tool {tool_name!r} arguments must be a JSON object, got {type(arguments).__name__}"
            )
        serialized = json.dumps(arguments, default=str, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > tool.max_input_bytes:
            raise ToolError(
                f"Tool {tool_name!r} input too large: {len(serialized.encode('utf-8'))} bytes"
            )
        # Enforce the declared JSON schema (basic checks).
        schema = tool.argument_schema
        required = schema.get("required", [])
        for key in required:
            if key not in arguments:
                raise ToolError(
                    f"Tool {tool_name!r} missing required argument: {key!r}"
                )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in arguments:
                if key not in properties:
                    raise ToolError(
                        f"Tool {tool_name!r} has unexpected argument: {key!r}"
                    )
        # Type-check declared properties.
        for key, value in arguments.items():
            prop = properties.get(key)
            if prop is None:
                continue
            # Check array minItems constraint
            if prop.get("type") == "array" and "minItems" in prop:
                if not isinstance(value, list) or len(value) < prop["minItems"]:
                    raise ToolError(
                        f"Tool {tool_name!r} argument {key!r} must have at least {prop['minItems']} items"
                    )
            prop_type = prop.get("type")
            if prop_type == "number" and not isinstance(value, (int, float)):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a number"
                )
            if prop_type == "integer" and not isinstance(value, int):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be an integer"
                )
            if prop_type == "string" and not isinstance(value, str):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a string"
                )
            if prop_type == "boolean" and not isinstance(value, bool):
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be a boolean"
                )
            if "enum" in prop and value not in prop["enum"]:
                raise ToolError(
                    f"Tool {tool_name!r} argument {key!r} must be one of {prop['enum']}"
                )
        return arguments

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Validate and execute a tool call."""
        validated = self.validate_arguments(tool_name, arguments)
        tool = self._tools[tool_name]
        return tool.handler(validated)


def _sensor_data_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    udi_raw = arguments.get("udi")
    sensor_type = arguments.get("sensor_type", "all")
    
    if udi_raw is None:
        raise ToolError("udi is required. Provide a numeric UDI from AI4I 2020 dataset (e.g., 1). Do not pass incident description.")
    if isinstance(udi_raw, str) and not udi_raw.isdigit():
        raise ToolError(f"Invalid UDI: {udi_raw!r}. Must be positive integer, not text.")
    try:
        udi_int = int(udi_raw)
    except (ValueError, TypeError):
        raise ToolError(f"Invalid UDI: {udi_raw!r}. Must be positive integer.")
    if udi_int <= 0:
        raise ToolError(f"UDI must be positive: {udi_int}.")
    
    try:
        param_map = {
            "all": None,
            "air_temperature": "air_temperature",
            "process_temperature": "process_temperature",
            "rotational_speed": "rotational_speed",
            "torque": "torque",
            "tool_wear": "tool_wear",
        }
        mapped = param_map.get(str(sensor_type).lower(), str(sensor_type).lower())
        if mapped is not None and mapped not in AGENT_PARAMS:
            raise ToolError(f"Unsupported parameter: {sensor_type!r}. Supported: {list(AGENT_PARAMS.keys())}")
        result = get_sensor_readings(udi_int, parameter=mapped if mapped else None)
        return result
    except DataAdapterError as exc:
        raise ToolError(str(exc))


def _diagnostic_math_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Harmless deterministic math on caller-provided numbers.

    Computes sum, mean, min, max, and standard deviation. This tool
    requires VALID numeric input - it cannot compute statistics on empty data.
    For operational evidence, first retrieve sensor data using a data-adapter tool.
    """
    values = arguments["values"]
    if not isinstance(values, list) or not values:
        raise ToolError(
            "Cannot compute statistics on empty data. "
            "Provide non-empty values array, or retrieve sensor data first."
        )
    numeric = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ToolError("all values must be numeric")
        numeric.append(float(v))
    n = len(numeric)
    total = sum(numeric)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in numeric) / n
    std = math.sqrt(variance)
    return {
        "count": n,
        "sum": total,
        "mean": mean,
        "min": min(numeric),
        "max": max(numeric),
        "std_dev": std,
        "variance": variance,
    }


def _predict_failure_risk_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .detection import predict
    reading = {
        "air_temperature_k": arguments["air_temperature"],
        "process_temperature_k": arguments["process_temperature"],
        "rotational_speed_rpm": arguments["rotational_speed"],
        "torque_nm": arguments["torque"],
        "tool_wear_min": arguments["tool_wear"],
    }
    include_secondary = bool(arguments.get("include_secondary", True))
    try:
        return predict(reading, include_secondary=include_secondary)
    except Exception as exc:
        raise ToolError(f"Prediction inference failed: {exc}")


def _twin_telemetry_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .digital_twin import get_twin_store
    twin_id = str(arguments["twin_id"]).strip()
    if not twin_id:
        raise ToolError("twin_id cannot be empty")
    limit = int(arguments.get("limit", 10))
    if limit <= 0:
        raise ToolError("limit must be a positive integer")
    store = get_twin_store()
    twin = store.get(twin_id)
    if not twin:
        raise ToolError(f"Digital twin {twin_id!r} not found")
    history = store.history(twin_id, limit=limit) or []
    return {
        "twin_id": twin["id"],
        "name": twin["name"],
        "machine_type": twin["machine_type"],
        "metadata": twin["metadata"],
        "created_at": twin["created_at"],
        "updated_at": twin["updated_at"],
        "observed_reading_count": twin["observed_reading_count"],
        "latest_observed_reading": twin["latest_observed_reading"],
        "recent_history": history,
    }


def _assess_evidence_sufficiency_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .diagnostics import assess_diagnosis
    evidence = arguments.get("evidence", [])
    hypotheses = arguments.get("hypotheses", [])
    required = arguments.get("required_evidence")
    try:
        return assess_diagnosis(evidence=evidence, hypotheses=hypotheses, required_evidence=required)
    except Exception as exc:
        raise ToolError(f"Evidence sufficiency assessment failed: {exc}")


def _rank_diagnostic_tests_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .diagnostics import rank_next_tests
    hypotheses = arguments.get("hypotheses", [])
    available_tests = arguments.get("available_tests", [])
    try:
        return rank_next_tests(hypotheses=hypotheses, available_tests=available_tests)
    except Exception as exc:
        raise ToolError(f"Diagnostic test ranking failed: {exc}")


def _verify_operator_clearance_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .workforce import get_workforce_store
    worker_id = str(arguments["worker_id"]).strip()
    required_level = str(arguments["required_level"]).strip().upper()
    try:
        store = get_workforce_store()
        return store.clearance_check(worker_id=worker_id, required_level=required_level)
    except LookupError as exc:
        raise ToolError(str(exc))
    except ValueError as exc:
        raise ToolError(str(exc))
def _correlate_workforce_incident_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .workforce import get_workforce_store
    incident_id = str(arguments.get("incident_id", "")).strip()
    if not incident_id:
        raise ToolError("incident_id is required")
    machine_id = arguments.get("machine_id")
    incident_time = arguments.get("incident_time")
    window_hours = float(arguments.get("window_hours", 48.0))
    try:
        store = get_workforce_store()
        corr = store.correlate_incident(
            incident_id=incident_id,
            machine_id=machine_id,
            incident_time=incident_time,
            window_hours=window_hours,
        )
        return corr.model_dump()
    except Exception as exc:
        raise ToolError(f"Workforce correlation failed: {exc}")


def _get_work_orders_handler(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from .workforce import get_workforce_store
    machine_id = str(arguments.get("machine_id", "")).strip()
    if not machine_id:
        raise ToolError("machine_id is required")
    limit = int(arguments.get("limit", 10))
    try:
        store = get_workforce_store()
        orders = store.list_work_orders(machine_id=machine_id, limit=limit)
        return {"machine_id": machine_id, "work_order_count": len(orders), "work_orders": orders}
    except Exception as exc:
        raise ToolError(f"Failed to retrieve work orders: {exc}")


def _build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()

    # Stub: Sensor data retrieval (future implementation)
    registry.register(
        Tool(
            name="get_sensor_readings",
            description=(
                "Retrieve a historical observation from the AI4I 2020 predictive-maintenance dataset. "
                "Requires a numeric UDI (record identifier) - NOT incident description, station, or timestamp. "
                "Returns measurements with units/provenance. Supported: air_temperature, process_temperature, rotational_speed, torque, tool_wear, all."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_sensor_data_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "udi": {
                        "type": "integer",
                        "description": "Numeric UDI (record identifier) from AI4I 2020 dataset. Must be integer, not text.",
                    },
                    "sensor_type": {
                        "type": "string",
                        "enum": ["all", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"],
                        "description": "Type of sensor to query.",
                    },
                },
                "required": ["udi"],
                "additionalProperties": False,
            },
        )
    )

    # Diagnostic math tool (requires non-empty input)
    registry.register(
        Tool(
            name="diagnostic_math",
            description=(
                "Compute deterministic summary statistics (count, sum, mean, "
                "min, max, standard deviation) on a list of numeric values "
                "provided by the caller. Read-only, no side effects. "
                "Requires non-empty values array."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_diagnostic_math_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 1,
                        "description": "List of numeric values to summarize (must be non-empty).",
                    }
                },
                "required": ["values"],
                "additionalProperties": False,
            },
            max_input_bytes=4000,
        )
    )

    # Supervised ML failure-risk prediction + statistical anomaly evidence
    registry.register(
        Tool(
            name="predict_failure_risk",
            description=(
                "Predict machine failure risk and evaluate multi-signal statistical anomaly evidence "
                "using supervised machine learning (HistGradientBoosting + RandomForest) trained on the AI4I 2020 dataset. "
                "Requires numeric parameters: air_temperature (K), process_temperature (K), rotational_speed (rpm), torque (Nm), tool_wear (min)."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_predict_failure_risk_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "air_temperature": {
                        "type": "number",
                        "description": "Air temperature in Kelvin [K] (e.g. 298.1).",
                    },
                    "process_temperature": {
                        "type": "number",
                        "description": "Process temperature in Kelvin [K] (e.g. 308.6).",
                    },
                    "rotational_speed": {
                        "type": "number",
                        "description": "Rotational speed in RPM [rpm] (e.g. 1551).",
                    },
                    "torque": {
                        "type": "number",
                        "description": "Torque in Newton-meters [Nm] (e.g. 42.8).",
                    },
                    "tool_wear": {
                        "type": "number",
                        "description": "Tool wear in minutes [min] (e.g. 0).",
                    },
                    "include_secondary": {
                        "type": "boolean",
                        "description": "Whether to run secondary Random Forest classifier for cross-model consensus (default true).",
                    },
                },
                "required": [
                    "air_temperature",
                    "process_temperature",
                    "rotational_speed",
                    "torque",
                    "tool_wear",
                ],
                "additionalProperties": False,
            },
            max_input_bytes=4000,
        )
    )

    # Digital twin telemetry retrieval
    registry.register(
        Tool(
            name="get_twin_telemetry",
            description=(
                "Retrieve current operational status, latest observed telemetry reading, "
                "and recent telemetry history for a machine digital twin by twin_id."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_twin_telemetry_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "twin_id": {
                        "type": "string",
                        "description": "Unique identifier of the digital twin (e.g. 'twin_abc123').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent telemetry history records to return (default 10).",
                    },
                },
                "required": ["twin_id"],
                "additionalProperties": False,
            },
            max_input_bytes=2000,
        )
    )

    # Deterministic evidence sufficiency assessment
    registry.register(
        Tool(
            name="assess_evidence_sufficiency",
            description=(
                "Deterministically evaluate whether gathered evidence sufficiently supports "
                "hypotheses and satisfies required evidence criteria before completing an investigation."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_assess_evidence_sufficiency_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of evidence records to verify against hypotheses.",
                    },
                    "hypotheses": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of hypotheses to evaluate.",
                    },
                    "required_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of required evidence keywords or parameters.",
                    },
                },
                "required": ["evidence", "hypotheses"],
                "additionalProperties": False,
            },
            max_input_bytes=4000,
        )
    )

    # Multi-criteria candidate diagnostic test ranking
    registry.register(
        Tool(
            name="rank_diagnostic_tests",
            description=(
                "Rank candidate diagnostic or physical inspection tests according to diagnostic value, "
                "relevance to unresolved hypotheses, operational cost, and safety risk."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_rank_diagnostic_tests_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Current unresolved or evaluated hypotheses.",
                    },
                    "available_tests": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of candidate tests with diagnostic_value, cost, and safety_risk.",
                    },
                },
                "required": ["hypotheses", "available_tests"],
                "additionalProperties": False,
            },
            max_input_bytes=4000,
        )
    )

    # Worker clearance level verification
    registry.register(
        Tool(
            name="verify_operator_clearance",
            description=(
                "Verify whether an industrial worker or technician holds the necessary clearance level "
                "(L1, L2, L3, L4) to review or execute maintenance operations."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_verify_operator_clearance_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "worker_id": {
                        "type": "string",
                        "description": "Unique identifier of the worker or technician.",
                    },
                    "required_level": {
                        "type": "string",
                        "enum": ["L1", "L2", "L3", "L4"],
                        "description": "Minimum required clearance level (L1, L2, L3, or L4).",
                    },
                },
                "required": ["worker_id", "required_level"],
                "additionalProperties": False,
            },
            max_input_bytes=2000,
        )
    )

    # Workforce timeline and maintenance correlation
    registry.register(
        Tool(
            name="correlate_workforce_incident",
            description=(
                "Correlate plant workforce records, recent maintenance work orders, technician clearance, "
                "and shift handovers around an operational incident timeline."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_correlate_workforce_incident_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "incident_id": {
                        "type": "string",
                        "description": "Unique identifier of the operational incident to correlate.",
                    },
                    "machine_id": {
                        "type": "string",
                        "description": "Optional machine identifier (e.g. mach_xxx or product_id).",
                    },
                    "incident_time": {
                        "type": "string",
                        "description": "Optional ISO-8601 incident timestamp (defaults to incident creation time).",
                    },
                    "window_hours": {
                        "type": "number",
                        "description": "Lookback time window in hours prior to the incident (default 48.0).",
                    },
                },
                "required": ["incident_id"],
                "additionalProperties": False,
            },
            max_input_bytes=2000,
        )
    )

    # Machine maintenance work order inspection history
    registry.register(
        Tool(
            name="get_work_orders_for_machine",
            description=(
                "Retrieve recent maintenance work orders, inspection evidence, and verification outcomes "
                "for a specific machine asset."
            ),
            risk_level=RiskLevel.NONE,
            requires_approval=False,
            handler=_get_work_orders_handler,
            argument_schema={
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string",
                        "description": "Machine asset identifier (e.g. mach_xxx).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of recent work orders to return (default 10).",
                    },
                },
                "required": ["machine_id"],
                "additionalProperties": False,
            },
            max_input_bytes=2000,
        )
    )
    return registry


# Module-level default registry used by the app and tests.
registered_tools = _build_default_registry()

