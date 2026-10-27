from plantpilot.diagnostics import proof_or_refuse_diagnosis


def test_proof_or_refuse_sufficient_evidence():
    evidence = [
        {
            "id": "ev_01",
            "parameter_name": "torque",
            "numeric_value": 68.5,
            "description": "High torque reading observed",
            "verified": True,
        },
        {
            "id": "ev_02",
            "parameter_name": "rotational_speed",
            "numeric_value": 1280.0,
            "description": "Reduced rotational speed observed",
            "verified": True,
        },
    ]
    hypotheses = [
        {
            "id": "hyp_01",
            "title": "Power failure / overload mechanism",
            "explanation": "High torque coupled with reduced RPM indicates mechanical strain.",
            "supporting_evidence_ids": ["ev_01", "ev_02"],
            "contradicting_evidence_ids": [],
        }
    ]

    result = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
        required_evidence=["torque"],
    )

    assert result["is_refused"] is False
    assert result["outcome"] == "SUPPORTED_FOR_REVIEW"
    assert result["diagnosis_status"] == "SUPPORTED_FOR_REVIEW"
    assert result["verified_evidence_count"] == 2
    assert len(result["refusal_reasons"]) == 0


def test_proof_or_refuse_missing_evidence():
    evidence = []
    hypotheses = [
        {
            "id": "hyp_01",
            "title": "Heat dissipation failure",
            "supporting_evidence_ids": [],
            "contradicting_evidence_ids": [],
        }
    ]

    result = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
        required_evidence=["air_temperature"],
    )

    assert result["is_refused"] is True
    assert result["outcome"] == "REFUSED"
    assert result["diagnosis_status"] == "REFUSED_INSUFFICIENT_EVIDENCE"
    assert any("No verified sensor readings" in r for r in result["refusal_reasons"])
    assert any("air_temperature" in m for m in result["missing_evidence"])
    assert len(result["useful_next_steps"]) > 0


def test_proof_or_refuse_unverified_evidence_rejected():
    evidence = [
        {
            "id": "ev_unverified",
            "parameter_name": "tool_wear",
            "numeric_value": 240.0,
            "description": "Unverified claim of high tool wear",
            "verified": False,
        }
    ]
    hypotheses = [
        {
            "id": "hyp_01",
            "title": "Tool wear failure",
            "supporting_evidence_ids": ["ev_unverified"],
            "contradicting_evidence_ids": [],
        }
    ]

    result = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
    )

    assert result["is_refused"] is True
    assert result["verified_evidence_count"] == 0
    assert any("No verified sensor readings" in r for r in result["refusal_reasons"])


def test_proof_or_refuse_contradictory_evidence():
    evidence = [
        {
            "id": "ev_temp_1",
            "parameter_name": "air_temperature",
            "numeric_value": 298.0,
            "description": "Normal ambient temperature",
            "verified": True,
        },
        {
            "id": "ev_temp_2",
            "parameter_name": "air_temperature",
            "numeric_value": 325.0,  # 27K difference -> severe contradiction
            "description": "Extreme ambient temperature",
            "verified": True,
        },
    ]
    hypotheses = [
        {
            "id": "hyp_01",
            "title": "Air temperature thermal shock",
            "supporting_evidence_ids": ["ev_temp_1"],
            "contradicting_evidence_ids": [],
        }
    ]

    result = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
    )

    assert result["is_refused"] is True
    assert len(result["contradictions"]) > 0
    assert any("Contradictory observations detected" in r for r in result["refusal_reasons"])
    assert any("Recalibrate sensors" in s for s in result["useful_next_steps"])


def test_proof_or_refuse_ambiguous_root_causes():
    evidence = [
        {
            "id": "ev_01",
            "parameter_name": "process_temperature",
            "numeric_value": 310.0,
            "description": "Elevated process temperature",
            "verified": True,
        }
    ]
    # Two competing hypotheses supported by the same evidence without distinguishing data
    hypotheses = [
        {
            "id": "hyp_01",
            "title": "Coolant system failure",
            "supporting_evidence_ids": ["ev_01"],
            "contradicting_evidence_ids": [],
        },
        {
            "id": "hyp_02",
            "title": "Bearing friction thermal rise",
            "supporting_evidence_ids": ["ev_01"],
            "contradicting_evidence_ids": [],
        },
    ]

    result = proof_or_refuse_diagnosis(
        evidence=evidence,
        hypotheses=hypotheses,
    )

    assert result["is_refused"] is True
    assert len(result["ambiguous_causes"]) == 2
    assert any("Multiple plausible root causes remain unresolved" in r for r in result["refusal_reasons"])
    assert any("distinguishing physical tests" in s for s in result["useful_next_steps"])
