"""Tests for the FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from plantpilot.main import app
from plantpilot.schemas import IncidentCreate, InvestigationCreate
from plantpilot.config import reset_config
from plantpilot.storage import reset_database


@pytest.fixture
def client():
    """Create a test client."""
    reset_config()
    reset_database()
    return TestClient(app)


def test_root_endpoint(client):
    """Test GET / returns application info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "PlantPilot"
    assert data["version"] == "0.1.0"
    assert data["status"] == "running"


def test_health_endpoint(client):
    """Test GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "ok"
    assert "provider" not in data  # Not checked by default


def test_health_endpoint_with_provider_check(client):
    """Test GET /health?check_provider=true."""
    response = client.get("/health?check_provider=true")
    assert response.status_code == 200
    data = response.json()
    assert "provider" in data
    # Provider check may succeed or fail depending on API key validity


def test_create_incident(client):
    """Test POST /api/v1/incidents creates an incident."""
    payload = {
        "description": "Production line stopped unexpectedly",
        "severity": "high",
        "category": "operational",
        "metadata": {"station": "A"},
    }
    response = client.post("/api/v1/incidents", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["description"] == payload["description"]
    assert data["severity"] == payload["severity"]
    assert data["category"] == payload["category"]
    assert data["status"] == "pending"
    assert "id" in data
    assert data["id"].startswith("inc_")


def test_create_incident_validation(client):
    """Test POST /api/v1/incidents validates input."""
    payload = {
        "description": "",  # Too short
        "severity": "high",
    }
    response = client.post("/api/v1/incidents", json=payload)
    assert response.status_code == 422  # Validation error


def test_get_incident(client):
    """Test GET /api/v1/incidents/{id} retrieves an incident."""
    # Create an incident first
    payload = {
        "description": "Temperature sensor anomaly",
        "severity": "medium",
    }
    create_response = client.post("/api/v1/incidents", json=payload)
    incident_id = create_response.json()["id"]

    # Retrieve it
    response = client.get(f"/api/v1/incidents/{incident_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == incident_id
    assert data["description"] == payload["description"]


def test_get_incident_not_found(client):
    """Test GET /api/v1/incidents/{id} returns 404 for unknown ID."""
    response = client.get("/api/v1/incidents/inc_nonexistent")
    assert response.status_code == 404


def test_list_incidents(client):
    """Test GET /api/v1/incidents lists incidents."""
    import uuid
    unique_suffix = uuid.uuid4().hex[:8]
    # Create 3 incidents with unique descriptions
    for i in range(3):
        payload = {
            "description": f"Test incident {unique_suffix} #{i}",
            "severity": "low",
        }
        client.post("/api/v1/incidents", json=payload)

    # List them
    response = client.get("/api/v1/incidents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # At least 3 incidents with our suffix should be present
    matching = [d for d in data if f"Test incident {unique_suffix}" in d["description"]]
    assert len(matching) >= 3  # Each created incident should be in the list


def test_list_incidents_pagination(client):
    """Test GET /api/v1/incidents supports pagination."""
    # Create 5 incidents
    for i in range(5):
        payload = {
            "description": f"Incident {i}",
            "severity": "low",
        }
        client.post("/api/v1/incidents", json=payload)

    # Get first page
    response = client.get("/api/v1/incidents?limit=2&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    # Get second page
    response = client.get("/api/v1/incidents?limit=2&offset=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_list_incidents_validation(client):
    """Test GET /api/v1/incidents validates pagination parameters."""
    response = client.get("/api/v1/incidents?limit=0")
    assert response.status_code == 400

    response = client.get("/api/v1/incidents?limit=300")
    assert response.status_code == 400

    response = client.get("/api/v1/incidents?offset=-1")
    assert response.status_code == 400


def test_machines_api_endpoints(client):
    """Test machine registration and retrieval via API."""
    # Create machine
    payload = {"product_id": "L47181", "type": "L", "status": "active"}
    res = client.post("/api/v1/machines", json=payload)
    assert res.status_code == 201
    machine = res.json()
    assert machine["product_id"] == "L47181"
    assert machine["type"] == "L"
    machine_id = machine["id"]

    # Get single machine
    res_get = client.get(f"/api/v1/machines/{machine_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == machine_id

    # List machines
    res_list = client.get("/api/v1/machines")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1


def test_recommendation_review_api_endpoint(client):
    """Test operator reviewing a recommendation via API."""
    import uuid
    from plantpilot.storage import get_database
    from plantpilot.schemas import IncidentCreate, InvestigationCreate, Recommendation, ApprovalStatus

    db = get_database()
    incident = db.create_incident(IncidentCreate(description="Review Test", severity="high"))
    inv = db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)
    rec_id = f"rec_api_{uuid.uuid4().hex[:8]}"
    rec = Recommendation(
        id=rec_id,
        title="Stop Spindle",
        description="Emergency stop spindle",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="critical",
    )
    db.save_recommendation(inv.id, rec)

    # Operator review call
    response = client.post(
        f"/api/v1/recommendations/{rec_id}/review",
        json={
            "status_choice": "approved",
            "reviewed_by": "operator_alice",
            "notes": "Emergency stop confirmed",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["approval_status"] == "approved"
    assert data["reviewed_by"] == "operator_alice"
    assert data["review_notes"] == "Emergency stop confirmed"
    assert data["reviewed_at"] is not None


def test_create_and_list_digital_twins(client):
    """Test POST /api/v1/twins and GET /api/v1/twins."""
    payload = {
        "name": "CNC Milling Station 01",
        "machine_type": "M",
        "metadata": {"line": 3, "vendor": "DMG MORI"},
    }
    response = client.post("/api/v1/twins", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == payload["name"]
    assert data["machine_type"] == payload["machine_type"]
    assert data["id"].startswith("twin_")
    assert data["observed_reading_count"] == 0

    twin_id = data["id"]

    # Test list twins
    list_resp = client.get("/api/v1/twins")
    assert list_resp.status_code == 200
    twins = list_resp.json()
    assert any(t["id"] == twin_id for t in twins)


def test_get_digital_twin_and_history(client):
    """Test GET /api/v1/twins/{twin_id} and history."""
    # Create twin
    create_resp = client.post(
        "/api/v1/twins",
        json={"name": "Lathe 02", "machine_type": "L", "metadata": {}},
    )
    twin_id = create_resp.json()["id"]

    # Get single twin
    get_resp = client.get(f"/api/v1/twins/{twin_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == twin_id

    # Get history (initially empty)
    hist_resp = client.get(f"/api/v1/twins/{twin_id}/history")
    assert hist_resp.status_code == 200
    assert hist_resp.json() == []


def test_twin_reading_ingestion_with_auto_risk(client):
    """Test POST /api/v1/twins/{id}/readings computes ML risk assessment automatically."""
    create_resp = client.post(
        "/api/v1/twins",
        json={"name": "Press Unit 03", "machine_type": "H", "metadata": {}},
    )
    twin_id = create_resp.json()["id"]

    reading_payload = {
        "reading": {
            "air_temperature_k": 298.2,
            "process_temperature_k": 308.7,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 10.0,
        }
    }
    resp = client.post(f"/api/v1/twins/{twin_id}/readings", json=reading_payload)
    assert resp.status_code == 201
    rec = resp.json()
    assert rec["twin_id"] == twin_id
    assert rec["evidence_type"] == "observed"
    assert "failure_probability" in rec["assessment"]
    assert "threshold" in rec["assessment"]
    assert "failure_risk" in rec["assessment"]

    # Verify twin updated_at and count updated
    twin_resp = client.get(f"/api/v1/twins/{twin_id}")
    twin_data = twin_resp.json()
    assert twin_data["observed_reading_count"] == 1
    assert twin_data["latest_observed_reading"] is not None


def test_twin_simulation_isolated(client):
    """Test POST /api/v1/twins/{id}/simulate does not alter observed count."""
    create_resp = client.post(
        "/api/v1/twins",
        json={"name": "Grinder 04", "machine_type": "M", "metadata": {}},
    )
    twin_id = create_resp.json()["id"]

    sim_payload = {
        "reading": {
            "air_temperature_k": 303.0,
            "process_temperature_k": 313.0,
            "rotational_speed_rpm": 1250.0,
            "torque_nm": 65.0,
            "tool_wear_min": 210.0,
        },
        "include_secondary": True,
    }
    resp = client.post(f"/api/v1/twins/{twin_id}/simulate", json=sim_payload)
    assert resp.status_code == 200
    sim_rec = resp.json()
    assert sim_rec["evidence_type"] == "simulated"
    assert "failure_probability" in sim_rec["assessment"]
    assert "secondary_model" in sim_rec["assessment"]

    # Twin observed count must remain 0
    twin_resp = client.get(f"/api/v1/twins/{twin_id}")
    twin_data = twin_resp.json()
    assert twin_data["observed_reading_count"] == 0
    assert twin_data["latest_observed_reading"] is None

    # History contains the simulated record
    hist_resp = client.get(f"/api/v1/twins/{twin_id}/history")
    history = hist_resp.json()
    assert len(history) == 1
    assert history[0]["evidence_type"] == "simulated"


def test_predict_risk_endpoint(client):
    """Test POST /api/v1/predict-risk standalone prediction."""
    payload = {
        "reading": {
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1551.0,
            "torque_nm": 42.8,
            "tool_wear_min": 0.0,
        },
        "include_secondary": True,
    }
    response = client.post("/api/v1/predict-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "HistGradientBoostingClassifier"
    assert "failure_probability" in data
    assert "threshold" in data
    assert data["failure_risk"] in ["elevated", "not_elevated"]
    assert "anomaly_evidence" in data
    assert "secondary_model" in data
    assert data["secondary_model"]["model"] == "RandomForestClassifier"


def test_twin_not_found_endpoints(client):
    """Test 404 behavior for unknown twin IDs."""
    assert client.get("/api/v1/twins/twin_unknown").status_code == 404
    assert client.get("/api/v1/twins/twin_unknown/history").status_code == 404
    assert client.post("/api/v1/twins/twin_unknown/readings", json={"reading": {}}).status_code == 404
    assert client.post("/api/v1/twins/twin_unknown/simulate", json={"reading": {}}).status_code == 404


def test_diagnostics_endpoints(client):
    """Test diagnostics API endpoints."""
    # Assess diagnosis
    assess_resp = client.post(
        "/api/v1/diagnostics/assess",
        json={
            "evidence": [{"id": "ev_1", "verified": True, "evidence_type": "observed", "description": "Temp 300K"}],
            "hypotheses": [{"id": "hyp_1", "title": "Overheating", "supporting_evidence_ids": ["ev_1"]}],
            "required_evidence": ["temp"],
        },
    )
    assert assess_resp.status_code == 200
    assert assess_resp.json()["diagnosis_status"] == "SUPPORTED_FOR_REVIEW"

    # Audit report
    audit_resp = client.post(
        "/api/v1/diagnostics/audit-report",
        json={
            "claims": [{"text": "Pressure ok", "claim_type": "factual", "evidence_ids": ["ev_1"]}],
            "evidence": [{"id": "ev_1", "verified": True, "evidence_type": "observed"}],
        },
    )
    assert audit_resp.status_code == 200
    assert audit_resp.json()["status"] == "VERIFIED"

    # Rank tests
    rank_resp = client.post(
        "/api/v1/diagnostics/rank-tests",
        json={
            "hypotheses": [{"id": "hyp_1", "title": "Wear", "resolved": False}],
            "available_tests": [
                {
                    "id": "t1",
                    "name": "Optical Measurement",
                    "diagnostic_value": 0.8,
                    "cost": 0.2,
                    "safety_risk": 0.1,
                    "available": True,
                    "permitted": True,
                }
            ],
        },
    )
    assert rank_resp.status_code == 200
    assert rank_resp.json()["recommended_test"]["name"] == "Optical Measurement"


def test_workforce_endpoints(client):
    """Test workforce worker, shift attendance, and clearance API endpoints."""
    # 1. Create worker
    create_resp = client.post(
        "/api/v1/workforce/workers",
        json={
            "name": "Sarah Connor",
            "role": "Robotics Supervisor",
            "department": "Security & Maintenance",
            "clearance_level": "L3",
        },
    )
    assert create_resp.status_code == 201
    worker = create_resp.json()
    assert worker["worker_id"].startswith("wrk_")
    assert worker["clearance_level"] == "L3"

    wid = worker["worker_id"]

    # 2. Get and list workers
    get_resp = client.get(f"/api/v1/workforce/workers/{wid}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Sarah Connor"

    list_resp = client.get("/api/v1/workforce/workers")
    assert list_resp.status_code == 200
    assert any(w["worker_id"] == wid for w in list_resp.json())

    # 3. Clock in
    clock_in_resp = client.post(
        "/api/v1/workforce/clock-in",
        json={
            "worker_id": wid,
            "timestamp": "2026-10-27T07:00:00Z",
            "shift_date": "2026-10-27",
            "break_minutes": 45,
        },
    )
    assert clock_in_resp.status_code == 201
    record = clock_in_resp.json()
    assert record["attendance_status"] == "OPEN"

    # 4. Clock out
    clock_out_resp = client.post(
        "/api/v1/workforce/clock-out",
        json={
            "worker_id": wid,
            "timestamp": "2026-10-27T15:45:00Z",
        },
    )
    assert clock_out_resp.status_code == 200
    closed_rec = clock_out_resp.json()
    assert closed_rec["attendance_status"] == "COMPLETE"
    assert closed_rec["working_hours"] == 8.0

    # 5. Get attendance
    att_resp = client.get(f"/api/v1/workforce/workers/{wid}/attendance")
    assert att_resp.status_code == 200
    assert len(att_resp.json()) == 1

    # 6. Clearance check
    check_ok = client.post(
        "/api/v1/workforce/clearance-check",
        json={"worker_id": wid, "required_level": "L2"},
    )
    assert check_ok.status_code == 200
    assert check_ok.json()["authorized_by_clearance"] is True

    check_denied = client.post(
        "/api/v1/workforce/clearance-check",
        json={"worker_id": wid, "required_level": "L4"},
    )
    assert check_denied.status_code == 200
    assert check_denied.json()["authorized_by_clearance"] is False

    # 7. Anomaly detection
    anomaly_resp = client.post(
        "/api/v1/workforce/detect-anomalies",
        json={"max_shift_hours": 12.0},
    )
    assert anomaly_resp.status_code == 200
    assert "finding_count" in anomaly_resp.json()


def test_recommendation_review_clearance_gating(client):
    """Test clearance gating on recommendation approval."""
    import uuid
    from plantpilot.schemas import Recommendation, ApprovalStatus
    from plantpilot.storage import get_database
    from plantpilot.workforce import get_workforce_store

    db = get_database()
    store = get_workforce_store()

    # Register L1 junior worker and L3 senior worker
    w_junior = store.create_worker(
        {"name": "Junior Tech", "role": "Junior", "department": "Maintenance", "clearance_level": "L1"}
    )
    w_senior = store.create_worker(
        {"name": "Lead Engineer", "role": "Lead", "department": "Maintenance", "clearance_level": "L3"}
    )

    # Create investigation and critical recommendation
    inv_resp = client.post("/api/v1/incidents", json={"description": "Spindle failure", "severity": "critical"})
    inc_id = inv_resp.json()["id"]
    incident = db.get_incident(inc_id)
    from plantpilot.schemas import InvestigationCreate
    inv = db.create_investigation(InvestigationCreate(incident_id=inc_id), incident=incident)


    rec_id = f"rec_clearance_{uuid.uuid4().hex[:8]}"
    rec = Recommendation(
        id=rec_id,
        title="Emergency Spindle Teardown",
        description="Immediate disassembly of high-voltage spindle",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="critical",  # Requires L3
    )
    db.save_recommendation(inv.id, rec)

    # 1. Junior L1 worker attempts to APPROVE critical recommendation -> 403 Forbidden
    resp_forbidden = client.post(
        f"/api/v1/recommendations/{rec_id}/review",
        json={
            "status_choice": "approved",
            "reviewed_by": w_junior["worker_id"],
            "notes": "Looks fine to me",
        },
    )
    assert resp_forbidden.status_code == 403
    assert "not authorized" in resp_forbidden.json()["detail"].lower() or "required" in resp_forbidden.json()["detail"].lower()

    # 2. Senior L3 worker approves -> 200 OK
    resp_approved = client.post(
        f"/api/v1/recommendations/{rec_id}/review",
        json={
            "status_choice": "approved",
            "reviewed_by": w_senior["worker_id"],
            "notes": "Approved by senior engineer",
        },
    )
    assert resp_approved.status_code == 200
    assert resp_approved.json()["approval_status"] == "approved"
    assert resp_approved.json()["reviewed_by"] == w_senior["worker_id"]




