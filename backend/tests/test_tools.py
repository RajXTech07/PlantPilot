"""Tests for the tool registry and built-in tools."""

import pytest
import json

from plantpilot.tools import (
    Tool,
    ToolRegistry,
    ToolError,
    registered_tools,
)
from plantpilot.safety import RiskLevel


def test_tool_registry_register():
    """Test registering a tool."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {"result": "ok"}

    tool = Tool(
        name="dummy",
        description="A dummy tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    registry.register(tool)
    assert registry.get("dummy") is not None
    assert "dummy" in registry.names()


def test_tool_registry_duplicate_registration():
    """Test that duplicate registration raises an error."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {}

    tool = Tool(
        name="dummy",
        description="A dummy tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={"type": "object"},
    )

    registry.register(tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_tool_to_openai_format():
    """Test tool conversion to OpenAI format."""
    def dummy_handler(args):
        return {}

    tool = Tool(
        name="test_tool",
        description="Test description",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
            },
            "required": ["value"],
        },
    )

    openai_format = tool.to_openai_format()
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == "test_tool"
    assert openai_format["function"]["description"] == "Test description"
    assert "properties" in openai_format["function"]["parameters"]


def test_diagnostic_math_tool():
    """Test the built-in diagnostic_math tool."""
    tool = registered_tools.get("diagnostic_math")
    assert tool is not None
    assert tool.risk_level == RiskLevel.NONE

    result = tool.handler({"values": [1, 2, 3, 4, 5]})
    assert result["count"] == 5
    assert result["sum"] == 15
    assert result["mean"] == 3.0
    assert result["min"] == 1
    assert result["max"] == 5
    assert result["std_dev"] > 0


def test_diagnostic_math_validation():
    """Test diagnostic_math validates input."""
    tool = registered_tools.get("diagnostic_math")

    with pytest.raises(ToolError):
        tool.handler({"values": []})  # Empty list

    with pytest.raises(ToolError):
        tool.handler({"values": [1, "two", 3]})  # Non-numeric


def test_tool_registry_validate_arguments():
    """Test argument validation."""
    registry = ToolRegistry()

    def dummy_handler(args):
        return {}

    tool = Tool(
        name="test",
        description="Test",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=dummy_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "flag": {"type": "boolean"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    registry.register(tool)

    # Valid arguments
    validated = registry.validate_arguments("test", {"value": 42})
    assert validated["value"] == 42

    # Missing required argument
    with pytest.raises(ToolError, match="missing required argument"):
        registry.validate_arguments("test", {})

    # Unexpected argument
    with pytest.raises(ToolError, match="unexpected argument"):
        registry.validate_arguments("test", {"value": 42, "extra": "data"})

    # Wrong type
    with pytest.raises(ToolError, match="must be a number"):
        registry.validate_arguments("test", {"value": "not a number"})


def test_tool_registry_execute():
    """Test tool execution."""
    registry = ToolRegistry()

    def echo_handler(args):
        return {"echo": args.get("message", "")}

    tool = Tool(
        name="echo",
        description="Echo tool",
        risk_level=RiskLevel.NONE,
        requires_approval=False,
        handler=echo_handler,
        argument_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
        },
    )
    registry.register(tool)

    result = registry.execute("echo", {"message": "hello"})
    assert result["echo"] == "hello"


def test_tool_registry_execute_unknown_tool():
    """Test executing an unknown tool raises an error."""
    registry = ToolRegistry()

    with pytest.raises(ToolError, match="Unknown tool"):
        registry.execute("nonexistent", {})


def test_tool_registry_to_openai_tools():
    """Test conversion of all tools to OpenAI format."""
    tools = registered_tools.to_openai_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(t["type"] == "function" for t in tools)
    names = [t["function"]["name"] for t in tools]
    assert "predict_failure_risk" in names
    assert "get_twin_telemetry" in names


def test_predict_failure_risk_tool():
    """Test the predict_failure_risk registered tool."""
    tool = registered_tools.get("predict_failure_risk")
    assert tool is not None
    assert tool.risk_level == RiskLevel.NONE

    args = {
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551.0,
        "torque": 42.8,
        "tool_wear": 0.0,
        "include_secondary": True,
    }
    result = registered_tools.execute("predict_failure_risk", args)
    assert "failure_probability" in result
    assert "threshold" in result
    assert "failure_risk" in result
    assert "anomaly_evidence" in result
    assert "secondary_model" in result


def test_predict_failure_risk_validation():
    """Test argument validation for predict_failure_risk."""
    with pytest.raises(ToolError, match="missing required argument"):
        registered_tools.execute("predict_failure_risk", {"air_temperature": 298.1})

    with pytest.raises(ToolError, match="must be a number"):
        registered_tools.execute(
            "predict_failure_risk",
            {
                "air_temperature": "hot",
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
            },
        )


def test_twin_telemetry_tool():
    """Test get_twin_telemetry registered tool."""
    from plantpilot.digital_twin import get_twin_store

    store = get_twin_store()
    twin = store.create("Pump Station 4", "H", {"location": "Bay 1"})
    store.add_reading(
        twin["id"],
        {"air_temperature_k": 301.0, "torque_nm": 45.0},
        {"risk": "low"},
        evidence_type="observed",
    )

    result = registered_tools.execute("get_twin_telemetry", {"twin_id": twin["id"]})
    assert result["twin_id"] == twin["id"]
    assert result["name"] == "Pump Station 4"
    assert result["observed_reading_count"] == 1
    assert result["latest_observed_reading"] is not None
    assert len(result["recent_history"]) == 1


def test_twin_telemetry_not_found():
    """Test get_twin_telemetry fails honestly for unknown twin."""
    with pytest.raises(ToolError, match="not found"):
        registered_tools.execute("get_twin_telemetry", {"twin_id": "twin_nonexistent"})


def test_assess_evidence_sufficiency_tool():
    """Test assess_evidence_sufficiency registered tool."""
    evidence = [
        {"id": "ev_1", "verified": True, "evidence_type": "observed", "description": "Temp 300K"}
    ]
    hypotheses = [
        {"id": "hyp_1", "title": "Overheating", "supporting_evidence_ids": ["ev_1"]}
    ]
    res = registered_tools.execute(
        "assess_evidence_sufficiency",
        {"evidence": evidence, "hypotheses": hypotheses},
    )
    assert res["diagnosis_status"] == "SUPPORTED_FOR_REVIEW"
    assert res["verified_evidence_count"] == 1


def test_rank_diagnostic_tests_tool():
    """Test rank_diagnostic_tests registered tool."""
    hypotheses = [{"id": "hyp_1", "title": "Wear", "resolved": False}]
    tests = [
        {
            "id": "t1",
            "name": "Optical Scan",
            "diagnostic_value": 0.9,
            "cost": 0.1,
            "safety_risk": 0.05,
            "available": True,
            "permitted": True,
        }
    ]
    res = registered_tools.execute(
        "rank_diagnostic_tests",
        {"hypotheses": hypotheses, "available_tests": tests},
    )
    assert res["recommended_test"] is not None
    assert res["recommended_test"]["name"] == "Optical Scan"


def test_verify_operator_clearance_tool():
    """Test verify_operator_clearance registered tool."""
    from plantpilot.workforce import get_workforce_store

    store = get_workforce_store()
    worker = store.create_worker(
        {
            "name": "Alex Vance",
            "role": "Robotics Specialist",
            "department": "Automation",
            "clearance_level": "L3",
        }
    )
    res = registered_tools.execute(
        "verify_operator_clearance",
        {"worker_id": worker["worker_id"], "required_level": "L2"},
    )
    assert res["authorized_by_clearance"] is True
    assert res["status"] == "AUTHORIZED"

    res_l4 = registered_tools.execute(
        "verify_operator_clearance",
        {"worker_id": worker["worker_id"], "required_level": "L4"},
    )
    assert res_l4["authorized_by_clearance"] is False
    assert res_l4["status"] == "FLAGGED"


