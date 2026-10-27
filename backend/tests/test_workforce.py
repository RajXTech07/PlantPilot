"""Unit tests for workforce store, attendance tracking, clearance gating, and fatigue detection."""

import pytest
from unittest.mock import MagicMock, patch
from plantpilot.workforce import WorkforceStore, get_workforce_store, reset_workforce_store, model_assess_workforce


def test_create_and_get_worker(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    worker = store.create_worker(
        {
            "name": "Marcus Aurelius",
            "role": "Lead Systems Engineer",
            "department": "Plant Reliability",
            "clearance_level": "L3",
            "status": "ACTIVE",
        }
    )
    assert worker["worker_id"].startswith("wrk_")
    assert worker["name"] == "Marcus Aurelius"
    assert worker["clearance_level"] == "L3"
    assert worker["status"] == "ACTIVE"

    fetched = store.get_worker(worker["worker_id"])
    assert fetched is not None
    assert fetched["name"] == "Marcus Aurelius"
    store.close()


def test_create_worker_invalid_clearance_or_status(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    with pytest.raises(ValueError, match="clearance_level must be"):
        store.create_worker(
            {
                "name": "Invalid Level",
                "role": "Tech",
                "department": "Ops",
                "clearance_level": "L5",
            }
        )

    with pytest.raises(ValueError, match="status must be"):
        store.create_worker(
            {
                "name": "Invalid Status",
                "role": "Tech",
                "department": "Ops",
                "clearance_level": "L2",
                "status": "SUSPENDED",
            }
        )
    store.close()


def test_clock_in_and_clock_out(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    worker = store.create_worker(
        {
            "name": "Jane Doe",
            "role": "Operator",
            "department": "Assembly",
            "clearance_level": "L2",
        }
    )
    wid = worker["worker_id"]

    rec = store.clock_in(
        worker_id=wid,
        timestamp="2026-10-27T08:00:00Z",
        shift_date="2026-10-27",
        break_minutes=30,
    )
    assert rec["record_id"].startswith("att_")
    assert rec["attendance_status"] == "OPEN"
    assert rec["working_hours"] is None

    # Duplicate open clock-in should fail
    with pytest.raises(ValueError, match="already has an open attendance record"):
        store.clock_in(wid, "2026-10-27T09:00:00Z", "2026-10-27")

    # Clock out
    closed = store.clock_out(wid, "2026-10-27T16:30:00Z")
    assert closed["attendance_status"] == "COMPLETE"
    # Total duration = 8.5 hours - 0.5 hours break = 8.0 hours
    assert closed["working_hours"] == 8.0

    history = store.attendance(wid)
    assert len(history) == 1
    assert history[0]["record_id"] == rec["record_id"]
    store.close()


def test_clock_out_before_clock_in(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    worker = store.create_worker(
        {
            "name": "Bob Smith",
            "role": "Machinist",
            "department": "Milling",
            "clearance_level": "L1",
        }
    )
    wid = worker["worker_id"]
    store.clock_in(wid, "2026-10-27T12:00:00Z", "2026-10-27")

    with pytest.raises(ValueError, match="clock-out must be after clock-in"):
        store.clock_out(wid, "2026-10-27T10:00:00Z")
    store.close()


def test_clearance_check(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    w_l2 = store.create_worker(
        {"name": "L2 Tech", "role": "Junior", "department": "Ops", "clearance_level": "L2"}
    )
    w_l4 = store.create_worker(
        {"name": "L4 Lead", "role": "Principal", "department": "Safety", "clearance_level": "L4"}
    )
    w_inactive = store.create_worker(
        {
            "name": "Inactive Tech",
            "role": "Contractor",
            "department": "Ops",
            "clearance_level": "L4",
            "status": "INACTIVE",
        }
    )

    # L2 worker requesting L2 -> Authorized
    res = store.clearance_check(w_l2["worker_id"], "L2")
    assert res["authorized_by_clearance"] is True
    assert res["status"] == "AUTHORIZED"

    # L2 worker requesting L3 -> Flagged
    res_high = store.clearance_check(w_l2["worker_id"], "L3")
    assert res_high["authorized_by_clearance"] is False
    assert res_high["status"] == "FLAGGED"
    assert res_high["requires_human_review"] is True

    # L4 worker requesting L1 -> Authorized
    assert store.clearance_check(w_l4["worker_id"], "L1")["authorized_by_clearance"] is True

    # Inactive worker even with L4 -> Flagged
    assert store.clearance_check(w_inactive["worker_id"], "L1")["authorized_by_clearance"] is False
    store.close()


def test_detect_anomalies(tmp_path):
    store = WorkforceStore(db_path=tmp_path / "test_workforce.db")
    w1 = store.create_worker({"name": "Overtime Worker", "role": "Tech", "department": "M", "clearance_level": "L2"})
    w2 = store.create_worker({"name": "Missing Clockout", "role": "Tech", "department": "M", "clearance_level": "L2"})

    # Excessive shift (> 12 hours, e.g. 14 hours)
    store.clock_in(w1["worker_id"], "2026-10-27T06:00:00Z", "2026-10-27", break_minutes=0)
    store.clock_out(w1["worker_id"], "2026-10-27T20:30:00Z")  # 14.5 hours

    # Open clock out
    store.clock_in(w2["worker_id"], "2026-10-27T07:00:00Z", "2026-10-27")

    detection = store.detect(max_shift_hours=12.0)
    assert detection["finding_count"] >= 2
    types = [f["type"] for f in detection["findings"]]
    assert "EXCESSIVE_RECORDED_HOURS" in types
    assert "MISSING_CLOCK_OUT" in types
    store.close()


def test_model_assess_workforce_mocked():
    findings = {
        "finding_count": 1,
        "findings": [
            {
                "finding_id": "wfn_123",
                "type": "EXCESSIVE_RECORDED_HOURS",
                "severity": "high",
                "worker_id": "wrk_1",
                "working_hours": 14.5,
            }
        ],
    }

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message = MagicMock()
    mock_choice.message.content = '{"summary": "Fatigue risk identified for shift exceeding 14 hours.", "prioritized_findings": [{"finding_id": "wfn_123", "suggested_human_followup": "Verify operator relief"}], "uncertainties": []}'
    mock_resp.choices = [mock_choice]


    with patch("plantpilot.workforce.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create.return_value = mock_resp

        result = model_assess_workforce(findings)
        assert result["analysis_mode"] == "model_assisted_review"
        assert result["deterministic_findings"] == findings
        assert "Fatigue risk identified" in result["model_review"]["summary"]
