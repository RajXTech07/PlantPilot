import pytest
from plantpilot.workforce.service import WorkforceService


@pytest.fixture
def wf(tmp_path):
    db_file = tmp_path / "test_workforce_modular.db"
    service = WorkforceService(db_path=db_file)
    yield service
    service.close()


# -------------------------------------------------------------
# Worker Registry & Clearance Tests
# -------------------------------------------------------------

def test_worker_registration_and_clearance(wf):
    worker = wf.create_worker({
        "worker_id": "W-001",
        "name": "Alex Mercer",
        "role": "Senior Technician",
        "department": "Mechanical Maintenance",
        "clearance_level": "L3",
        "status": "ACTIVE",
        "skills": ["vibration_analysis", "bearing_replacement"],
        "certifications": [
            {
                "name": "ISO Vibration Cat II",
                "cert_id": "CERT-VIB-01",
                "issued_at": "2025-01-01T00:00:00Z",
                "expires_at": "2027-01-01T00:00:00Z",
                "valid": True,
            }
        ],
    })
    assert worker["worker_id"] == "W-001"
    assert worker["clearance_level"] == "L3"

    # L3 clearance allows L1, L2, L3 operations
    chk_l1 = wf.clearance_check("W-001", "L1")
    assert chk_l1["authorized_by_clearance"] is True

    chk_l3 = wf.clearance_check("W-001", "L3")
    assert chk_l3["authorized_by_clearance"] is True

    # L3 does NOT allow L4 operations
    chk_l4 = wf.clearance_check("W-001", "L4")
    assert chk_l4["authorized_by_clearance"] is False
    assert chk_l4["status"] in {"FLAGGED", "DENIED"}


def test_expired_certification_revokes_clearance(wf):
    wf.create_worker({
        "worker_id": "W-002",
        "name": "Jordan Bell",
        "role": "Electrical Tech",
        "department": "Electrical",
        "clearance_level": "L2",
        "status": "ACTIVE",
        "certifications": [
            {
                "name": "High Voltage Safety",
                "cert_id": "HV-EXPIRED",
                "issued_at": "2023-01-01T00:00:00Z",
                "expires_at": "2024-01-01T00:00:00Z",  # Expired
                "valid": False,
            }
        ],
    })

    chk = wf.clearance_check("W-002", "L2")
    assert chk["authorized_by_clearance"] is False
    assert chk["requires_human_review"] is True
    assert any("expired" in r.lower() for r in chk.get("unmet_requirements", []))


# -------------------------------------------------------------
# Attendance & Overnight Shifts Tests
# -------------------------------------------------------------

def test_attendance_and_break_deduction(wf):
    wf.create_worker({
        "worker_id": "W-003",
        "name": "Elena Vance",
        "role": "Machinist",
        "department": "Production",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    # Clock in at 08:00 with 30 min scheduled break
    record = wf.clock_in("W-003", "2026-10-27T08:00:00Z", "2026-10-27", break_minutes=30)
    assert record["worker_id"] == "W-003"
    assert record["break_minutes"] == 30
    assert record["attendance_status"] == "OPEN"

    # Clock out at 16:30 (8.5 hours elapsed - 0.5 break = 8.0 working hours)
    out = wf.clock_out("W-003", "2026-10-27T16:30:00Z")
    assert out["attendance_status"] == "COMPLETE"
    assert pytest.approx(out["working_hours"], 0.05) == 8.0


def test_overnight_shift_hours_calculation(wf):
    wf.create_worker({
        "worker_id": "W-004",
        "name": "Marcus Night",
        "role": "Night Supervisor",
        "department": "Operations",
        "clearance_level": "L2",
        "status": "ACTIVE",
    })

    # Starts 22:00 on Day 1, ends 06:00 on Day 2 with 30 min break
    wf.clock_in("W-004", "2026-10-27T22:00:00Z", "2026-10-27", break_minutes=30)
    out = wf.clock_out("W-004", "2026-10-28T06:00:00Z")

    # 8 hours elapsed - 0.5 break = 7.5 working hours
    assert pytest.approx(out["working_hours"], 0.05) == 7.5
    assert out["attendance_status"] == "COMPLETE"


# -------------------------------------------------------------
# Scheduling & Shift Coverage Tests
# -------------------------------------------------------------

def test_shift_scheduling_and_coverage(wf):
    wf.create_worker({
        "worker_id": "W-OPERATOR",
        "name": "Sam Operator",
        "role": "Operator",
        "department": "Machining",
        "clearance_level": "L2",
        "status": "ACTIVE",
    })

    # Create shift schedule requiring 2 headcount with clearance L2
    sched = wf.create_schedule({
        "shift_name": "Day Shift",
        "shift_date": "2026-10-27",
        "start_time": "08:00",
        "end_time": "16:00",
        "machine_id": "mach_cnc_01",
        "department": "Machining",
        "required_roles": ["Operator", "Inspector"],
        "min_headcount": 2,
        "min_clearance": "L2",
        "assigned_worker_ids": ["W-OPERATOR"],
    })
    sched_id = sched["schedule_id"]

    # Verify coverage before anyone clocks in -> understaffed
    cov1 = wf.verify_shift_coverage(sched_id)
    assert cov1["is_fully_covered"] is False
    assert cov1["active_worker_count"] == 0

    # Worker clocks in
    wf.clock_in("W-OPERATOR", "2026-10-27T08:00:00Z", "2026-10-27")
    cov2 = wf.verify_shift_coverage(sched_id)
    # Headcount is 1 out of 2 required -> still flags understaffed
    assert cov2["is_fully_covered"] is False
    assert cov2["active_worker_count"] == 1
    assert any("headcount" in g.lower() for g in cov2["coverage_gaps"])


def test_shift_handover_logging(wf):
    wf.create_worker({
        "worker_id": "W-OUT",
        "name": "Outgoing Worker",
        "role": "Operator",
        "department": "Machining",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })
    wf.create_worker({
        "worker_id": "W-IN",
        "name": "Incoming Worker",
        "role": "Operator",
        "department": "Machining",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    handover = wf.record_handover({
        "shift_date": "2026-10-27",
        "shift_name": "Evening Handover",
        "machine_id": "mach_lathe_02",
        "outgoing_worker_id": "W-OUT",
        "incoming_worker_id": "W-IN",
        "notes": "Tool wear slightly high on insert #3. Replaced coolant filter.",
        "verified": True,
    })
    assert handover["handover_id"].startswith("hnd_") or handover["handover_id"].startswith("handover_")
    assert handover["verified"] is True

    handovers = wf.list_handovers(shift_date="2026-10-27")
    assert len(handovers) == 1
    assert handovers[0]["notes"] == "Tool wear slightly high on insert #3. Replaced coolant filter."


# -------------------------------------------------------------
# Maintenance Work Orders & Inspection Verification Tests
# -------------------------------------------------------------

def test_maintenance_work_order_verification(wf):
    wf.create_worker({
        "worker_id": "W-TECH",
        "name": "Tech Mike",
        "role": "Technician",
        "department": "Maintenance",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })
    wf.create_worker({
        "worker_id": "W-INSPECT",
        "name": "Inspector Sarah",
        "role": "Quality Inspector",
        "department": "Maintenance",
        "clearance_level": "L3",
        "status": "ACTIVE",
    })

    order = wf.create_work_order({
        "machine_id": "mach_milling_04",
        "task_description": "Replace spindle bearings and check runout",
        "assigned_worker_ids": ["W-TECH"],
        "required_clearance": "L1",
        "scheduled_start": "2026-10-27T09:00:00Z",
        "scheduled_end": "2026-10-27T12:00:00Z",
        "inspection_required": True,
    })
    order_id = order["order_id"]

    # If completed without inspector or evidence, verification outcome is UNVERIFIED
    comp1 = wf.record_completion(
        order_id=order_id,
        actual_end="2026-10-27T11:45:00Z",
        inspection_evidence=None,
        inspector_id=None,
    )
    assert comp1["status"] == "COMPLETED"
    assert comp1["verification_outcome"] == "UNVERIFIED"

    # Now verify with an authorized L3 inspector and dial indicator evidence
    comp2 = wf.record_completion(
        order_id=order_id,
        actual_end="2026-10-27T12:00:00Z",
        inspection_evidence={"dial_indicator_runout_mm": 0.002, "noise_dba": 68},
        inspector_id="W-INSPECT",
    )
    assert comp2["verification_outcome"] == "VERIFIED"


# -------------------------------------------------------------
# Analytics & Deterministic Anomaly Detection Tests
# -------------------------------------------------------------

def test_workforce_analytics_detects_anomalies(wf):
    wf.create_worker({
        "worker_id": "W-TIRED",
        "name": "Overworked Operator",
        "role": "Operator",
        "department": "Production",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    # Clock in 16 hours ago without clock out -> excessive hours & missing clock out
    wf.clock_in("W-TIRED", "2026-10-27T04:00:00Z", "2026-10-27")
    wf.clock_out("W-TIRED", "2026-10-27T20:00:00Z")  # 16 hours!

    findings_res = wf.detect(max_shift_hours=12.0)
    assert findings_res["finding_count"] > 0
    types = [f["type"] for f in findings_res["findings"]]
    assert "EXCESSIVE_RECORDED_HOURS" in types


# -------------------------------------------------------------
# Incident Correlation Tests
# -------------------------------------------------------------

def test_workforce_incident_correlation(wf):
    wf.create_worker({
        "worker_id": "W-CORR-1",
        "name": "On-Duty Tech",
        "role": "Maintenance Tech",
        "department": "Maintenance",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    # Create unverified work order on target machine completed just before incident
    wo = wf.create_work_order({
        "machine_id": "mach_press_09",
        "task_description": "Lubrication and seal replacement",
        "assigned_worker_ids": ["W-CORR-1"],
        "scheduled_start": "2026-10-27T08:00:00Z",
        "scheduled_end": "2026-10-27T10:00:00Z",
        "inspection_required": True,
    })
    wf.record_completion(wo["order_id"], actual_end="2026-10-27T09:30:00Z", inspector_id=None)

    # Correlate with an incident happening at 10:00
    corr = wf.correlate_incident(
        incident_id="inc_press_fail_1",
        machine_id="mach_press_09",
        incident_time="2026-10-27T10:00:00Z",
        window_hours=24.0,
    )

    assert corr.machine_id == "mach_press_09"
    assert corr.unverified_maintenance_detected is True
    assert len(corr.recent_work_orders) == 1
    assert any("unverified maintenance" in h.lower() for h in corr.hypotheses)
    assert len(corr.suggested_evidence_items) > 0
