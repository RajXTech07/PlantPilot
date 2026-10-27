"""Deterministic evidence sufficiency, report audit, and next-test ranking."""
from __future__ import annotations
from typing import Any


def assess_diagnosis(evidence: list[dict[str, Any]], hypotheses: list[dict[str, Any]], required_evidence: list[str] | None = None) -> dict[str, Any]:
    verified = [e for e in evidence if e.get("verified") is True and str(e.get("evidence_type", "observed")).lower() in {"observed", "derived"}]
    ids = {str(e.get("id")) for e in verified}
    missing = [item for item in (required_evidence or []) if not any(item.lower() in str(e.get("description", "")).lower() for e in verified)]
    evaluated = []
    for h in hypotheses:
        supporting = [str(x) for x in h.get("supporting_evidence_ids", [])]
        contradicting = [str(x) for x in h.get("contradicting_evidence_ids", [])]
        invalid_support = [x for x in supporting if x not in ids]
        invalid_contradiction = [x for x in contradicting if x not in ids]
        evaluated.append({"hypothesis_id": h.get("id"), "title": h.get("title", "Untitled hypothesis"), "supported_by_verified_evidence": bool(supporting) and not invalid_support, "supporting_evidence_ids": supporting, "contradicting_evidence_ids": contradicting, "unsupported_evidence_references": invalid_support + invalid_contradiction, "has_contradiction": bool(contradicting)})
    sufficient = bool(verified) and not missing and bool(evaluated) and any(h["supported_by_verified_evidence"] and not h["has_contradiction"] and not h["unsupported_evidence_references"] for h in evaluated)
    return {"diagnosis_status": "SUPPORTED_FOR_REVIEW" if sufficient else "INCONCLUSIVE", "verified_evidence_count": len(verified), "missing_evidence": missing, "hypotheses": evaluated, "message": "Evidence supports at least one hypothesis for human review; this is not proof of causation." if sufficient else "Insufficient verified evidence to support a definitive diagnosis. Gather the missing evidence and resolve contradictions."}


def audit_report(claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(e.get("id")): e for e in evidence}
    results = []
    for claim in claims:
        refs = [str(x) for x in claim.get("evidence_ids", [])]
        matched = [by_id[x] for x in refs if x in by_id]
        verified = [e for e in matched if e.get("verified") is True and str(e.get("evidence_type", "observed")).lower() in {"observed", "derived"}]
        issue = None
        if not refs: issue = "No evidence references supplied"
        elif len(matched) != len(refs): issue = "One or more evidence IDs do not exist"
        elif not verified: issue = "Referenced evidence is not verified observation/derivation"
        elif claim.get("claim_type") == "causal" and not claim.get("causation_established", False): issue = "Causal claim is not established by evidence metadata"
        elif claim.get("claim_type") == "numeric" and claim.get("value") is not None and not any(str(claim["value"]) in str(e.get("value", e.get("description", ""))) for e in verified): issue = "Claimed numeric value not found in referenced evidence"
        results.append({"claim": claim.get("text", ""), "claim_type": claim.get("claim_type", "factual"), "evidence_ids": refs, "status": "SUPPORTED" if issue is None else "UNVERIFIED", "issue": issue})
    return {"status": "VERIFIED" if results and all(r["status"] == "SUPPORTED" for r in results) else "REVIEW_REQUIRED", "claims_checked": len(results), "supported_claims": sum(r["status"] == "SUPPORTED" for r in results), "claims": results, "note": "Automated checks validate provenance and basic consistency; they do not independently prove physical causation."}


def rank_next_tests(hypotheses: list[dict[str, Any]], available_tests: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = []
    unresolved = [h for h in hypotheses if not h.get("resolved", False)]
    for test in available_tests:
        name = str(test.get("name", "")).strip()
        if not name: continue
        related = set(str(x) for x in test.get("distinguishes_hypotheses", []))
        relevant = len(related.intersection(str(h.get("id", h.get("title", ""))) for h in unresolved))
        value = max(0.0, min(1.0, float(test.get("diagnostic_value", 0.5))))
        cost = max(0.0, min(1.0, float(test.get("cost", 0.2))))
        risk = max(0.0, min(1.0, float(test.get("safety_risk", 0.0))))
        score = max(0.0, value * (0.5 + 0.5 * min(1, relevant)) - 0.2 * cost - 0.5 * risk)
        if test.get("available", True) is not True: score = 0.0
        if test.get("permitted", True) is not True: score = 0.0
        ranked.append({"test_id": test.get("id"), "name": name, "score": round(score, 4), "diagnostic_value": value, "cost": cost, "safety_risk": risk, "relevant_hypotheses": sorted(related), "available": test.get("available", True), "permitted": test.get("permitted", True), "reason": "Ranked by diagnostic value, relevance, cost, and safety; confirm with an authorized operator."})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {"recommended_test": ranked[0] if ranked and ranked[0]["score"] > 0 else None, "ranked_tests": ranked, "unresolved_hypotheses": [h.get("id", h.get("title")) for h in unresolved], "note": "Only supplied candidate tests are ranked; this endpoint does not execute physical tests."}
