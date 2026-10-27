from plantpilot.diagnostics import rank_next_tests


def test_rank_next_tests_prioritizes_distinguishing_tests():
    hypotheses = [
        {"id": "hyp_overheat", "title": "Overheating failure", "resolved": False},
        {"id": "hyp_strain", "title": "Mechanical strain", "resolved": False},
    ]

    available_tests = [
        {
            "id": "test_speed",
            "name": "Check rotational speed",
            "diagnostic_value": 0.5,
            "cost": 0.1,
            "safety_risk": 0.0,
            "distinguishes_hypotheses": ["hyp_overheat"],
            "is_executable_tool": True,
        },
        {
            "id": "test_thermal_imaging",
            "name": "Thermal FLIR imaging inspection",
            "diagnostic_value": 0.95,
            "cost": 0.2,
            "safety_risk": 0.05,
            "distinguishes_hypotheses": ["hyp_overheat", "hyp_strain"],
            "is_executable_tool": False,
        },
    ]

    result = rank_next_tests(
        hypotheses=hypotheses,
        available_tests=available_tests,
    )

    ranked = result["ranked_tests"]
    assert len(ranked) == 2
    # Thermal FLIR should be ranked higher due to higher diagnostic value and distinguishing both
    assert ranked[0]["name"] == "Thermal FLIR imaging inspection"
    assert ranked[0]["score"] > ranked[1]["score"]
    assert ranked[0]["is_executable_tool"] is False
    assert ranked[1]["is_executable_tool"] is True


def test_rank_next_tests_penalizes_already_performed():
    hypotheses = [
        {"id": "hyp_01", "title": "Sensor drift", "resolved": False},
    ]

    available_tests = [
        {
            "id": "get_sensor_readings",
            "name": "get_sensor_readings",
            "diagnostic_value": 0.8,
            "cost": 0.1,
            "safety_risk": 0.0,
            "distinguishes_hypotheses": ["hyp_01"],
            "is_executable_tool": True,
        },
        {
            "id": "diagnostic_math",
            "name": "diagnostic_math",
            "diagnostic_value": 0.6,
            "cost": 0.1,
            "safety_risk": 0.0,
            "distinguishes_hypotheses": ["hyp_01"],
            "is_executable_tool": True,
        },
    ]

    # Without penalty, get_sensor_readings ranks #1
    unpenalized = rank_next_tests(hypotheses=hypotheses, available_tests=available_tests)
    assert unpenalized["ranked_tests"][0]["name"] == "get_sensor_readings"

    # With get_sensor_readings already performed, it is penalized severely
    penalized = rank_next_tests(
        hypotheses=hypotheses,
        available_tests=available_tests,
        tests_already_performed=["get_sensor_readings"],
    )
    assert penalized["ranked_tests"][0]["name"] == "diagnostic_math"
    executed_item = [t for t in penalized["ranked_tests"] if t["name"] == "get_sensor_readings"][0]
    assert executed_item["already_performed"] is True


def test_rank_next_tests_penalizes_failed_attempts():
    hypotheses = [{"id": "hyp_01", "title": "Tool wear", "resolved": False}]
    available_tests = [
        {
            "id": "broken_sensor_query",
            "name": "broken_sensor_query",
            "diagnostic_value": 0.9,
            "cost": 0.1,
            "safety_risk": 0.0,
            "distinguishes_hypotheses": ["hyp_01"],
        },
        {
            "id": "alternative_inspection",
            "name": "alternative_inspection",
            "diagnostic_value": 0.7,
            "cost": 0.2,
            "safety_risk": 0.0,
            "distinguishes_hypotheses": ["hyp_01"],
        },
    ]

    result = rank_next_tests(
        hypotheses=hypotheses,
        available_tests=available_tests,
        previous_failed_attempts=["broken_sensor_query"],
    )
    assert result["ranked_tests"][0]["name"] == "alternative_inspection"
    failed_test = [t for t in result["ranked_tests"] if t["name"] == "broken_sensor_query"][0]
    assert failed_test["failed_previously"] is True
