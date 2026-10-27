"""Integration tests between Workforce Intelligence and Investigation Engine."""
import pytest
from plantpilot.schemas import (
    Incident,
    IncidentCreate,
    Investigation,
    MachineCreate,
    Status,
    ApprovalStatus,
    Recommendation,
)
from plantpilot.agent import AgentController
from plantpilot.tools import registered_tools
from plantpilot.safety import default_safety_policy
from plantpilot.storage import Database
from plantpilot.workforce import get_workforce_store, reset_workforce_store


@pytest.fixture
def clean_db(tmp_path):
    reset_workforce_store()
    db_file = tmp_path / "test_wf_integration.db"
    db = Database(db_file)
    db.connect()
    yield db
    db.close()
    reset_workforce_store()


def test_workforce_incident_correlation_tool(clean_db):
    store = get_workforce_store(clean_db.db_path)
    
    # 1. Register machine
    mach = clean_db.create_machine(MachineCreate(product_id="M-TEST-01", type="M"))
    mach_id = mach.id
    
    # 2. Register worker
    store.create_worker({
        "worker_id": "W-MAINT-1",
        "name": "Jane Technician",
        "role": "Maintenance",
        "department": "Mechanical",
        "clearance_level": "L2",
        "status": "ACTIVE",
    })
    
    # 3. Create unverified maintenance work order
    wo = store.create_work_order({
        "machine_id": mach_id,
        "task_description": "Spindle bearing check and oil flush",
        "assigned_worker_ids": ["W-MAINT-1"],
        "scheduled_start": "2026-10-27T08:00:00Z",
        "scheduled_end": "2026-10-27T09:00:00Z",
        "inspection_required": True,
    })
    store.record_completion(wo["order_id"], actual_end="2026-10-27T08:45:00Z", inspector_id=None)
    
    # 4. Create incident
    inc = clean_db.create_incident(IncidentCreate(
        machine_id=mach_id,
        description="Spindle overheating 15 min after restart",
        severity="high",
        category="mechanical",
    ))
    
    # 5. Execute workforce correlation tool
    tool = registered_tools.get("correlate_workforce_incident")
    assert tool is not None
    
    result = registered_tools.execute("correlate_workforce_incident", {
        "incident_id": inc.id,
        "machine_id": mach_id,
        "incident_time": "2026-10-27T09:00:00Z",
        "window_hours": 24.0,
    })
    
    assert result["unverified_maintenance_detected"] is True
    assert len(result["recent_work_orders"]) == 1
    assert any("unverified maintenance" in h.lower() for h in result["hypotheses"])
    assert len(result["suggested_evidence_items"]) > 0


def test_clearance_enforcement_on_critical_recommendation(clean_db):
    store = get_workforce_store(clean_db.db_path)

    # Register technician with clearance L1
    store.create_worker({
        "worker_id": "W-JUNIOR",
        "name": "Junior Operator",
        "role": "Operator",
        "department": "Production",
        "clearance_level": "L1",
        "status": "ACTIVE",
    })

    # Register supervisor with clearance L3
    store.create_worker({
        "worker_id": "W-SUPERVISOR",
        "name": "Lead Supervisor",
        "role": "Plant Supervisor",
        "department": "Operations",
        "clearance_level": "L3",
        "status": "ACTIVE",
    })

    # Create recommendation for CRITICAL machine replacement
    rec = Recommendation(
        id="rec_critical_01",
        title="Immediate emergency shutdown of boiler #2",
        description="Hazardous temperature rise detected.",
        priority="critical",
        requires_approval=True,
    )

    # Reviewing with L1 operator must fail clearance check (requires L3)
    chk_junior = store.clearance_check("W-JUNIOR", "L3")
    assert chk_junior["authorized_by_clearance"] is False
    assert chk_junior["status"] == "FLAGGED"

    # Reviewing with L3 supervisor must succeed
    chk_super = store.clearance_check("W-SUPERVISOR", "L3")
    assert chk_super["authorized_by_clearance"] is True
    assert chk_super["status"] == "AUTHORIZED"
