"""Integration tests for new FastAPI endpoints for Workforce, Diagnostics, and Audit."""
import pytest
from fastapi.testclient import TestClient

from plantpilot.main import app
from plantpilot.config import reset_config
from plantpilot.storage import reset_database
from plantpilot.workforce import reset_workforce_store


@pytest.fixture
def client():
    reset_config()
    reset_database()
    reset_workforce_store()
    return TestClient(app)


def test_shift_schedule_endpoints(client):
    # 1. Create shift schedule
    payload = {
        "shift_name": "Morning Operations",
        "shift_date": "2026-10-27",
        "start_time": "06:00",
        "end_time": "14:00",
        "department": "CNC Machining",
        "required_roles": ["Operator"],
        "min_headcount": 1,
        "min_clearance": "L1",
        "assigned_worker_ids": [],
    }
    resp = client.post("/api/v1/workforce/shifts/schedules", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["shift_name"] == "Morning Operations"
    sched_id = data["schedule_id"]

    # 2. List shift schedules
    resp_list = client.get("/api/v1/workforce/shifts/schedules?shift_date=2026-10-27")
    assert resp_list.status_code == 200
    assert len(resp_list.json()) >= 1

    # 3. Check coverage
    resp_cov = client.get(f"/api/v1/workforce/shifts/schedules/{sched_id}/coverage")
    assert resp_cov.status_code == 200
    assert "is_fully_covered" in resp_cov.json()


def test_shift_handover_endpoints(client):
    # Register workers first
    client.post("/api/v1/workforce/workers", json={
        "worker_id": "W-HO-1",
        "name": "Operator 1",
        "role": "Operator",
        "department": "Stamping",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })
    client.post("/api/v1/workforce/workers", json={
        "worker_id": "W-HO-2",
        "name": "Operator 2",
        "role": "Operator",
        "department": "Stamping",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    payload = {
        "shift_date": "2026-10-27",
        "shift_name": "Afternoon Shift",
        "outgoing_worker_id": "W-HO-1",
        "incoming_worker_id": "W-HO-2",
        "notes": "Machine 3 ran smoothly with no hydraulic pressure drops.",
        "verified": True,
    }
    resp = client.post("/api/v1/workforce/shifts/handovers", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["handover_id"].startswith("hnd_") or data["handover_id"].startswith("handover_")
    assert data["notes"] == "Machine 3 ran smoothly with no hydraulic pressure drops."

    # List handovers
    resp_list = client.get("/api/v1/workforce/shifts/handovers?shift_date=2026-10-27")
    assert resp_list.status_code == 200
    assert len(resp_list.json()) >= 1


def test_work_order_lifecycle_endpoints(client):
    # 1. Create work order
    wo_payload = {
        "machine_id": "mach_press_01",
        "task_description": "Clean hydraulic valves and check fluid levels",
        "scheduled_start": "2026-10-27T08:00:00Z",
        "scheduled_end": "2026-10-27T10:00:00Z",
        "required_clearance": "L1",
        "inspection_required": True,
    }
    resp = client.post("/api/v1/workforce/work-orders", json=wo_payload)
    assert resp.status_code == 201
    wo = resp.json()
    order_id = wo["order_id"]
    assert wo["status"] == "ASSIGNED"

    # 2. Get work order
    resp_get = client.get(f"/api/v1/workforce/work-orders/{order_id}")
    assert resp_get.status_code == 200
    assert resp_get.json()["task_description"] == "Clean hydraulic valves and check fluid levels"

    # 3. List work orders
    resp_list = client.get("/api/v1/workforce/work-orders?machine_id=mach_press_01")
    assert resp_list.status_code == 200
    assert len(resp_list.json()) >= 1

    # 4. Complete work order
    comp_payload = {
        "actual_end": "2026-10-27T09:45:00Z",
        "inspector_id": None,
        "inspection_evidence": None,
    }
    resp_comp = client.post(f"/api/v1/workforce/work-orders/{order_id}/complete", json=comp_payload)
    assert resp_comp.status_code == 200
    assert resp_comp.json()["status"] == "COMPLETED"
    assert resp_comp.json()["verification_outcome"] == "UNVERIFIED"


def test_workforce_correlate_endpoint(client):
    payload = {
        "incident_id": "inc_hydraulic_drop",
        "machine_id": "mach_press_01",
        "incident_time": "2026-10-27T10:00:00Z",
        "window_hours": 24.0,
    }
    resp = client.post("/api/v1/workforce/correlate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["incident_id"] == "inc_hydraulic_drop"
    assert "hypotheses" in data
    assert "temporal_correlations" in data


def test_proof_or_refuse_endpoint(client):
    req = {
        "evidence": [
            {
                "id": "ev_01",
                "parameter_name": "torque",
                "numeric_value": 75.0,
                "verified": True,
            }
        ],
        "hypotheses": [
            {
                "id": "hyp_strain",
                "title": "Torque overload",
                "supporting_evidence_ids": ["ev_01"],
            }
        ],
        "required_evidence": ["torque"],
    }
    resp = client.post("/api/v1/diagnostics/proof-or-refuse", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome"] == "SUPPORTED_FOR_REVIEW"
    assert data["is_refused"] is False


def test_investigation_audit_endpoint(client):
    # Register machine & create incident
    mach_resp = client.post("/api/v1/machines", json={
        "product_id": "M-PROD-01",
        "type": "M",
    })
    mach_id = mach_resp.json()["id"]

    inc_resp = client.post("/api/v1/incidents", json={
        "machine_id": mach_id,
        "description": "Bearing vibration spike",
        "severity": "high",
        "category": "mechanical",
    })
    inc_id = inc_resp.json()["id"]

    # Trigger investigation
    inv_resp = client.post(f"/api/v1/incidents/{inc_id}/investigations", json={
        "incident_id": inc_id,
        "notes": "Audit verification test",
    })
    inv_id = inv_resp.json()["investigation_id"]

    # Audit the investigation record
    audit_resp = client.post(f"/api/v1/investigations/{inv_id}/audit")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert "status" in audit_data
    assert "claims_checked" in audit_data
    assert "claims" in audit_data
    assert audit_data["investigation_id"] == inv_id
