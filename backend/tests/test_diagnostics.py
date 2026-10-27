"""Tests for diagnostics.py (evidence sufficiency, report audit, and test ranking)."""

import pytest
from plantpilot.diagnostics import assess_diagnosis, audit_report, rank_next_tests


def test_assess_diagnosis_supported():
    evidence = [
        {
            "id": "ev_1",
            "verified": True,
            "evidence_type": "observed",
            "description": "Air temperature 300K",
            "value": 300.0,
        },
        {
            "id": "ev_2",
            "verified": True,
            "evidence_type": "derived",
            "description": "Temperature difference 10K",
            "value": 10.0,
        },
    ]
    hypotheses = [
        {
            "id": "hyp_1",
            "title": "Heat dissipation failure",
            "supporting_evidence_ids": ["ev_1", "ev_2"],
            "contradicting_evidence_ids": [],
        }
    ]
    result = assess_diagnosis(evidence, hypotheses, required_evidence=["air temperature"])
    assert result["diagnosis_status"] == "SUPPORTED_FOR_REVIEW"
    assert result["verified_evidence_count"] == 2
    assert len(result["missing_evidence"]) == 0
    assert result["hypotheses"][0]["supported_by_verified_evidence"] is True


def test_assess_diagnosis_inconclusive_missing_evidence():
    evidence = [
        {
            "id": "ev_1",
            "verified": True,
            "evidence_type": "observed",
            "description": "Rotational speed 1500 rpm",
        }
    ]
    hypotheses = [
        {
            "id": "hyp_1",
            "title": "Tool wear failure",
            "supporting_evidence_ids": ["ev_1"],
            "contradicting_evidence_ids": [],
        }
    ]
    # Required evidence "torque" is not present
    result = assess_diagnosis(evidence, hypotheses, required_evidence=["torque"])
    assert result["diagnosis_status"] == "INCONCLUSIVE"
    assert "torque" in result["missing_evidence"]


def test_assess_diagnosis_unverified_or_invalid_reference():
    evidence = [
        {
            "id": "ev_1",
            "verified": False,  # unverified
            "evidence_type": "hypothesis",
            "description": "Unverified speculation",
        }
    ]
    hypotheses = [
        {
            "id": "hyp_1",
            "title": "Overstrain failure",
            "supporting_evidence_ids": ["ev_1", "ev_nonexistent"],
            "contradicting_evidence_ids": [],
        }
    ]
    result = assess_diagnosis(evidence, hypotheses)
    assert result["diagnosis_status"] == "INCONCLUSIVE"
    assert result["verified_evidence_count"] == 0
    assert "ev_nonexistent" in result["hypotheses"][0]["unsupported_evidence_references"]


def test_audit_report_all_verified():
    evidence = [
        {
            "id": "ev_1",
            "verified": True,
            "evidence_type": "observed",
            "description": "Torque measured at 55 Nm",
            "value": 55.0,
        }
    ]
    claims = [
        {
            "text": "Torque reached 55 Nm",
            "claim_type": "numeric",
            "value": 55.0,
            "evidence_ids": ["ev_1"],
        }
    ]
    audit = audit_report(claims, evidence)
    assert audit["status"] == "VERIFIED"
    assert audit["claims_checked"] == 1
    assert audit["supported_claims"] == 1
    assert audit["claims"][0]["status"] == "SUPPORTED"


def test_audit_report_flags_issues():
    evidence = [
        {
            "id": "ev_1",
            "verified": True,
            "evidence_type": "observed",
            "description": "Tool wear is 210 min",
            "value": 210,
        }
    ]
    claims = [
        # Missing evidence refs
        {"text": "Line failed due to fatigue", "claim_type": "factual", "evidence_ids": []},
        # Causal claim without causation_established
        {
            "text": "High wear caused the stoppage",
            "claim_type": "causal",
            "causation_established": False,
            "evidence_ids": ["ev_1"],
        },
        # Numeric mismatch
        {
            "text": "Rotational speed was 9999",
            "claim_type": "numeric",
            "value": 9999,
            "evidence_ids": ["ev_1"],
        },
    ]
    audit = audit_report(claims, evidence)
    assert audit["status"] == "REVIEW_REQUIRED"
    assert audit["claims_checked"] == 3
    assert audit["supported_claims"] == 0
    assert audit["claims"][0]["issue"] == "No evidence references supplied"
    assert "Causal claim is not established" in audit["claims"][1]["issue"]
    assert "Claimed numeric value not found" in audit["claims"][2]["issue"]


def test_rank_next_tests():
    hypotheses = [
        {"id": "hyp_1", "title": "Tool wear failure", "resolved": False},
        {"id": "hyp_2", "title": "Overstrain failure", "resolved": False},
    ]
    available_tests = [
        {
            "id": "test_1",
            "name": "Optical micrometer tool edge scan",
            "distinguishes_hypotheses": ["hyp_1"],
            "diagnostic_value": 0.9,
            "cost": 0.1,
            "safety_risk": 0.05,
            "available": True,
            "permitted": True,
        },
        {
            "id": "test_2",
            "name": "Destructive spindle teardown",
            "distinguishes_hypotheses": ["hyp_1", "hyp_2"],
            "diagnostic_value": 0.95,
            "cost": 0.9,
            "safety_risk": 0.8,
            "available": True,
            "permitted": True,
        },
        {
            "id": "test_3",
            "name": "Unauthorized high-voltage test",
            "distinguishes_hypotheses": ["hyp_2"],
            "diagnostic_value": 0.8,
            "cost": 0.2,
            "safety_risk": 0.2,
            "available": True,
            "permitted": False,  # Not permitted
        },
    ]
    ranked = rank_next_tests(hypotheses, available_tests)
    assert ranked["recommended_test"] is not None
    assert ranked["recommended_test"]["name"] == "Optical micrometer tool edge scan"
    # test_3 should be zeroed out because permitted is False
    test_3_result = next(t for t in ranked["ranked_tests"] if t["test_id"] == "test_3")
    assert test_3_result["score"] == 0.0
