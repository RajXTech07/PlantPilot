"""Tests for SQLite persistence layer."""

import pytest
from pathlib import Path
from datetime import datetime, timezone

from plantpilot.storage import Database
from plantpilot.schemas import (
    ApprovalStatus,
    EvidenceItem,
    EvidenceType,
    Hypothesis,
    IncidentCreate,
    InvestigationCreate,
    Machine,
    MachineCreate,
    Recommendation,
    Status,
    ToolExecution,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    db.connect()
    yield db
    db.close()


def test_database_initialization(test_db):
    """Test database schema is created."""
    # Should not raise
    assert test_db._conn is not None


def test_create_incident(test_db):
    """Test creating an incident."""
    create = IncidentCreate(
        description="Test incident",
        severity="high",
        category="operational",
        metadata={"key": "value"},
    )

    incident = test_db.create_incident(create)

    assert incident.id.startswith("inc_")
    assert incident.description == "Test incident"
    assert incident.severity == "high"
    assert incident.status == Status.PENDING
    assert incident.metadata["key"] == "value"


def test_get_incident(test_db):
    """Test retrieving an incident."""
    create = IncidentCreate(
        description="Test incident",
        severity="medium",
    )

    created = test_db.create_incident(create)
    retrieved = test_db.get_incident(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.description == created.description


def test_get_incident_not_found(test_db):
    """Test retrieving a non-existent incident returns None."""
    result = test_db.get_incident("inc_nonexistent")
    assert result is None


def test_list_incidents(test_db):
    """Test listing incidents."""
    for i in range(3):
        create = IncidentCreate(
            description=f"Incident {i}",
            severity="low",
        )
        test_db.create_incident(create)

    incidents = test_db.list_incidents()
    assert len(incidents) == 3
    # Newest first
    assert "Incident 2" in incidents[0].description


def test_list_incidents_pagination(test_db):
    """Test incident pagination."""
    for i in range(10):
        create = IncidentCreate(
            description=f"Incident {i}",
            severity="low",
        )
        test_db.create_incident(create)

    page1 = test_db.list_incidents(limit=5, offset=0)
    assert len(page1) == 5

    page2 = test_db.list_incidents(limit=5, offset=5)
    assert len(page2) == 5

    # Different incidents
    assert page1[0].id != page2[0].id


def test_create_investigation(test_db):
    """Test creating an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )

    create = InvestigationCreate(
        incident_id=incident.id,
        notes="Starting investigation",
    )

    investigation = test_db.create_investigation(create, incident)

    assert investigation.id.startswith("inv_")
    assert investigation.incident_id == incident.id
    assert investigation.status == Status.IN_PROGRESS
    assert investigation.notes == "Starting investigation"


def test_get_investigation(test_db):
    """Test retrieving an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)

    created = test_db.create_investigation(create, incident)
    retrieved = test_db.get_investigation(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.incident_id == incident.id


def test_get_investigation_not_found(test_db):
    """Test retrieving a non-existent investigation returns None."""
    result = test_db.get_investigation("inv_nonexistent")
    assert result is None


def test_update_investigation(test_db):
    """Test updating an investigation."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    test_db.update_investigation(
        investigation.id,
        status=Status.COMPLETED,
        findings=["Finding 1", "Finding 2"],
        completed_at=datetime.now(timezone.utc),
    )

    retrieved = test_db.get_investigation(investigation.id)
    assert retrieved.status == Status.COMPLETED
    assert len(retrieved.findings) == 2
    assert retrieved.completed_at is not None


def test_save_and_load_evidence(test_db):
    """Test saving and loading evidence."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    evidence = EvidenceItem(
        id="ev_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Temperature reading",
        source="sensor_01",
        value=88.0,
        verified=True,
    )

    test_db.save_evidence(investigation.id, evidence)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.evidence) == 1
    assert retrieved.evidence[0].id == "ev_1"
    assert retrieved.evidence[0].value == 88.0


def test_save_and_load_hypothesis(test_db):
    """Test saving and loading hypotheses."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    hypothesis = Hypothesis(
        id="hyp_1",
        title="Root cause hypothesis",
        explanation="Detailed explanation",
        confidence="medium",
        supporting_evidence_ids=["ev_1"],
    )

    test_db.save_hypothesis(investigation.id, hypothesis)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.hypotheses) == 1
    assert retrieved.hypotheses[0].title == "Root cause hypothesis"


def test_save_and_load_recommendation(test_db):
    """Test saving and loading recommendations."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    recommendation = Recommendation(
        id="rec_1",
        title="Take action",
        description="Detailed recommendation",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="high",
    )

    test_db.save_recommendation(investigation.id, recommendation)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.recommendations) == 1
    assert retrieved.recommendations[0].title == "Take action"


def test_save_and_load_tool_execution(test_db):
    """Test saving and loading tool executions."""
    incident = test_db.create_incident(
        IncidentCreate(description="Test", severity="high")
    )
    create = InvestigationCreate(incident_id=incident.id)
    investigation = test_db.create_investigation(create, incident)

    execution = ToolExecution(
        id="exec_1",
        tool_name="diagnostic_math",
        arguments={"values": [1, 2, 3]},
        result='{"mean": 2.0}',
        status=Status.COMPLETED,
    )

    test_db.save_tool_execution(investigation.id, execution)

    retrieved = test_db.get_investigation(investigation.id)
    assert len(retrieved.tool_executions) == 1
    assert retrieved.tool_executions[0].tool_name == "diagnostic_math"


def test_save_audit_event(test_db):
    """Test saving an audit event."""
    # Should not raise
    test_db.save_audit_event(
        event_type="test_event",
        message="Test message",
        details={"key": "value"},
    )


def test_database_persistence(tmp_path):
    """Test data persists across connections."""
    db_path = tmp_path / "test.db"

    # Create and save data
    db1 = Database(db_path)
    db1.connect()
    create = IncidentCreate(description="Test", severity="high")
    incident = db1.create_incident(create)
    incident_id = incident.id
    db1.close()

    # Reopen and verify
    db2 = Database(db_path)
    db2.connect()
    retrieved = db2.get_incident(incident_id)
    assert retrieved is not None
    assert retrieved.id == incident_id
    db2.close()


def test_machine_registry_crud(test_db):
    """Test creating, getting, and listing machines."""
    create = MachineCreate(product_id="M14860", type="M", status="active")
    machine = test_db.create_machine(create)

    assert machine.id.startswith("mach_")
    assert machine.product_id == "M14860"
    assert machine.type == "M"

    retrieved = test_db.get_machine(machine.id)
    assert retrieved is not None
    assert retrieved.product_id == "M14860"

    machines = test_db.list_machines()
    assert len(machines) == 1
    assert machines[0].product_id == "M14860"


def test_hypothesis_evidence_junction_table(test_db):
    """Test many-to-many junction table between hypotheses and evidence."""
    incident = test_db.create_incident(IncidentCreate(description="Test", severity="high"))
    inv = test_db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)

    ev1 = EvidenceItem(
        id="ev_supp_1",
        evidence_type=EvidenceType.OBSERVED,
        description="High torque",
        source="sensor_torque",
        value=65.2,
        verified=True,
    )
    ev2 = EvidenceItem(
        id="ev_contra_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Normal temperature",
        source="sensor_temp",
        value=298.1,
        verified=True,
    )
    test_db.save_evidence(inv.id, ev1)
    test_db.save_evidence(inv.id, ev2)

    hyp = Hypothesis(
        id="hyp_overload",
        title="Torque Overload",
        explanation="Torque exceeded normal range",
        confidence="high",
        supporting_evidence_ids=["ev_supp_1"],
        contradicting_evidence_ids=["ev_contra_1"],
    )
    test_db.save_hypothesis(inv.id, hyp)

    retrieved_inv = test_db.get_investigation(inv.id)
    assert len(retrieved_inv.hypotheses) == 1
    loaded_hyp = retrieved_inv.hypotheses[0]
    assert loaded_hyp.supporting_evidence_ids == ["ev_supp_1"]
    assert loaded_hyp.contradicting_evidence_ids == ["ev_contra_1"]


def test_recommendation_human_review_audit(test_db):
    """Test operator sign-off workflow on recommendations."""
    incident = test_db.create_incident(IncidentCreate(description="Test", severity="medium"))
    inv = test_db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)

    rec = Recommendation(
        id="rec_inspect_01",
        title="Inspect Milling Tool",
        description="Check tool wear",
        requires_approval=True,
        approval_status=ApprovalStatus.PENDING_APPROVAL,
        priority="high",
    )
    test_db.save_recommendation(inv.id, rec)

    # Operator approves recommendation
    reviewed = test_db.review_recommendation(
        recommendation_id="rec_inspect_01",
        approval_status=ApprovalStatus.APPROVED,
        reviewed_by="operator_john",
        review_notes="Tool wear physically verified > 200 min",
    )

    assert reviewed is not None
    assert reviewed.approval_status == ApprovalStatus.APPROVED
    assert reviewed.reviewed_by == "operator_john"
    assert reviewed.review_notes == "Tool wear physically verified > 200 min"
    assert reviewed.reviewed_at is not None

    # Verify persistence
    retrieved_inv = test_db.get_investigation(inv.id)
    assert retrieved_inv.recommendations[0].approval_status == ApprovalStatus.APPROVED
    assert retrieved_inv.recommendations[0].reviewed_by == "operator_john"


def test_foreign_key_cascade_deletion(test_db):
    """Test that deleting an incident cascades to investigation and child tables."""
    incident = test_db.create_incident(IncidentCreate(description="Test", severity="high"))
    inv = test_db.create_investigation(InvestigationCreate(incident_id=incident.id), incident)

    ev = EvidenceItem(
        id="ev_cascade_1",
        evidence_type=EvidenceType.OBSERVED,
        description="Reading",
        source="sensor",
        value=10.0,
    )
    test_db.save_evidence(inv.id, ev)

    # Verify records exist
    assert test_db.get_investigation(inv.id) is not None

    # Delete incident
    test_db._conn.execute("DELETE FROM incidents WHERE id = ?", (incident.id,))
    test_db._conn.commit()

    # Investigation must be cascade deleted
    assert test_db.get_investigation(inv.id) is None
    # Evidence must also be cascade deleted
    rows = test_db._conn.execute("SELECT * FROM evidence WHERE id = ?", ("ev_cascade_1",)).fetchall()
    assert len(rows) == 0

