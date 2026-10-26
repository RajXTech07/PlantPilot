"""SQLite persistence layer for PlantPilot.

Uses Python's built-in sqlite3 module with parameterised queries. The
database schema supports machines, incidents, investigations, evidence,
hypotheses, recommendations, tool executions, and audit events with
foreign key enforcement and cascade integrity.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from .config import get_config
from .schemas import (
    ApprovalStatus,
    EvidenceItem,
    Hypothesis,
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    Machine,
    MachineCreate,
    Recommendation,
    Status,
    ToolExecution,
)


class DatabaseError(RuntimeError):
    """Raised on database operation failures."""


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Open connection and initialize schema if needed."""
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enforce foreign key constraints in SQLite
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._initialize_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _initialize_schema(self) -> None:
        """Create tables if they don't exist and run automated column migrations."""
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS machines (
                id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                machine_id TEXT,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (machine_id) REFERENCES machines(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                incident_id TEXT NOT NULL,
                status TEXT NOT NULL,
                findings_json TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                tool_execution_id TEXT,
                evidence_type TEXT NOT NULL,
                parameter_name TEXT,
                numeric_value REAL,
                unit TEXT,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                value_json TEXT,
                timestamp TEXT NOT NULL,
                verified INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE,
                FOREIGN KEY (tool_execution_id) REFERENCES tool_executions(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                explanation TEXT NOT NULL,
                confidence TEXT,
                supporting_evidence_json TEXT,
                contradicting_evidence_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS hypothesis_evidence (
                hypothesis_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                PRIMARY KEY (hypothesis_id, evidence_id, relationship_type),
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id) ON DELETE CASCADE,
                FOREIGN KEY (evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                requires_approval INTEGER NOT NULL DEFAULT 1,
                approval_status TEXT NOT NULL,
                priority TEXT NOT NULL,
                reviewed_by TEXT,
                review_notes TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                executed_at TEXT,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tool_executions (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                executed_at TEXT NOT NULL,
                input_size_bytes INTEGER NOT NULL DEFAULT 0,
                output_size_bytes INTEGER NOT NULL DEFAULT 0,
                step_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                id TEXT PRIMARY KEY,
                investigation_id TEXT,
                event_type TEXT NOT NULL,
                message TEXT NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL
            );
        """)
        self._migrate_columns()

        # Indexes are safely created after all tables and columns are guaranteed to exist
        self._conn.executescript("""
            CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
            CREATE INDEX IF NOT EXISTS idx_incidents_machine ON incidents(machine_id);
            CREATE INDEX IF NOT EXISTS idx_investigations_incident ON investigations(incident_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_investigation ON evidence(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_tool ON evidence(tool_execution_id);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_investigation ON hypotheses(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_hyp_ev_hyp ON hypothesis_evidence(hypothesis_id);
            CREATE INDEX IF NOT EXISTS idx_hyp_ev_ev ON hypothesis_evidence(evidence_id);
            CREATE INDEX IF NOT EXISTS idx_recommendations_investigation ON recommendations(investigation_id);
            CREATE INDEX IF NOT EXISTS idx_tool_executions_investigation ON tool_executions(investigation_id);
        """)
        self._conn.commit()

    def _migrate_columns(self) -> None:
        """Safely ensure any newly added columns exist in older SQLite databases."""
        if self._conn is None:
            return

        migrations = [
            ("incidents", "machine_id", "TEXT"),
            ("evidence", "tool_execution_id", "TEXT"),
            ("evidence", "parameter_name", "TEXT"),
            ("evidence", "numeric_value", "REAL"),
            ("evidence", "unit", "TEXT"),
            ("recommendations", "reviewed_by", "TEXT"),
            ("recommendations", "review_notes", "TEXT"),
            ("recommendations", "reviewed_at", "TEXT"),
            ("tool_executions", "step_index", "INTEGER NOT NULL DEFAULT 0"),
        ]

        for table, col, col_type in migrations:
            try:
                cursor = self._conn.execute(f"PRAGMA table_info({table})")
                existing = [row["name"] for row in cursor.fetchall()]
                if col not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass
        self._conn.commit()

    # --- Machines / Assets ---

    def create_machine(self, create: MachineCreate) -> Machine:
        if self._conn is None:
            raise DatabaseError("Not connected")

        existing = self._conn.execute(
            "SELECT * FROM machines WHERE product_id = ?", (create.product_id,)
        ).fetchone()
        if existing is not None:
            return Machine(
                id=existing["id"],
                product_id=existing["product_id"],
                type=existing["type"],
                status=create.status or existing["status"],
                created_at=datetime.fromisoformat(existing["created_at"]),
            )

        machine_id = create.id or f"mach_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO machines (id, product_id, type, status, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (machine_id, create.product_id, create.type, create.status or "active", now),
        )
        self._conn.commit()

        return Machine(
            id=machine_id,
            product_id=create.product_id,
            type=create.type,
            status=create.status or "active",
            created_at=datetime.fromisoformat(now),
        )

    def get_machine(self, machine_id: str) -> Optional[Machine]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM machines WHERE id = ?", (machine_id,)
        ).fetchone()

        if row is None:
            return None

        return Machine(
            id=row["id"],
            product_id=row["product_id"],
            type=row["type"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def list_machines(self, limit: int = 50, offset: int = 0) -> List[Machine]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        rows = self._conn.execute(
            "SELECT * FROM machines ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [
            Machine(
                id=row["id"],
                product_id=row["product_id"],
                type=row["type"],
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    # --- Incidents ---

    def create_incident(self, create: IncidentCreate) -> Incident:
        if self._conn is None:
            raise DatabaseError("Not connected")

        incident_id = f"inc_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO incidents
               (id, machine_id, description, severity, category, status, created_at, updated_at, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                incident_id,
                create.machine_id,
                create.description,
                create.severity,
                create.category,
                Status.PENDING.value,
                now,
                now,
                json.dumps(create.metadata),
            ),
        )
        self._conn.commit()

        return Incident(
            id=incident_id,
            machine_id=create.machine_id,
            description=create.description,
            severity=create.severity,
            category=create.category,
            status=Status.PENDING,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            metadata=create.metadata,
        )

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM incidents WHERE id = ?", (incident_id,)
        ).fetchone()

        if row is None:
            return None

        return Incident(
            id=row["id"],
            machine_id=row["machine_id"] if "machine_id" in row.keys() else None,
            description=row["description"],
            severity=row["severity"],
            category=row["category"],
            status=Status(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def list_incidents(self, limit: int = 50, offset: int = 0) -> List[Incident]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        rows = self._conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        return [
            Incident(
                id=row["id"],
                machine_id=row["machine_id"] if "machine_id" in row.keys() else None,
                description=row["description"],
                severity=row["severity"],
                category=row["category"],
                status=Status(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            for row in rows
        ]

    # --- Investigations ---

    def create_investigation(
        self, create: InvestigationCreate, incident: Incident
    ) -> Investigation:
        if self._conn is None:
            raise DatabaseError("Not connected")

        investigation_id = f"inv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO investigations
               (id, incident_id, status, findings_json, started_at, created_at, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                investigation_id,
                create.incident_id,
                Status.IN_PROGRESS.value,
                json.dumps([]),
                now,
                now,
                create.notes,
            ),
        )
        self._conn.commit()

        return Investigation(
            id=investigation_id,
            incident_id=create.incident_id,
            status=Status.IN_PROGRESS,
            findings=[],
            evidence=[],
            hypotheses=[],
            tool_executions=[],
            recommendations=[],
            errors=[],
            started_at=datetime.fromisoformat(now),
            completed_at=None,
            created_at=datetime.fromisoformat(now),
            notes=create.notes,
        )

    def get_investigation(self, investigation_id: str) -> Optional[Investigation]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
        ).fetchone()

        if row is None:
            return None

        # Load related records
        evidence = self._load_evidence(investigation_id)
        hypotheses = self._load_hypotheses(investigation_id)
        recommendations = self._load_recommendations(investigation_id)
        tool_executions = self._load_tool_executions(investigation_id)

        return Investigation(
            id=row["id"],
            incident_id=row["incident_id"],
            status=Status(row["status"]),
            findings=json.loads(row["findings_json"] or "[]"),
            evidence=evidence,
            hypotheses=hypotheses,
            tool_executions=tool_executions,
            recommendations=recommendations,
            errors=[],
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            notes=row["notes"],
        )

    def update_investigation(
        self,
        investigation_id: str,
        status: Optional[Status] = None,
        findings: Optional[List[str]] = None,
        completed_at: Optional[datetime] = None,
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        updates = []
        params = []

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)
        if findings is not None:
            updates.append("findings_json = ?")
            params.append(json.dumps(findings))
        if completed_at is not None:
            updates.append("completed_at = ?")
            params.append(completed_at.isoformat())

        if not updates:
            return

        params.append(investigation_id)
        self._conn.execute(
            f"UPDATE investigations SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        self._conn.commit()

    def list_investigations(self, limit: int = 50, offset: int = 0) -> List[Investigation]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        rows = self._conn.execute(
            "SELECT id FROM investigations ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

        results = []
        for r in rows:
            inv = self.get_investigation(r["id"])
            if inv:
                results.append(inv)
        return results

    # --- Evidence ---

    def save_evidence(self, investigation_id: str, evidence: EvidenceItem) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO evidence
               (id, investigation_id, tool_execution_id, evidence_type,
                parameter_name, numeric_value, unit, description, source,
                value_json, timestamp, verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evidence.id,
                investigation_id,
                evidence.tool_execution_id,
                evidence.evidence_type.value,
                evidence.parameter_name,
                evidence.numeric_value,
                evidence.unit,
                evidence.description,
                evidence.source,
                json.dumps(evidence.value) if evidence.value is not None else None,
                evidence.timestamp.isoformat(),
                1 if evidence.verified else 0,
            ),
        )
        self._conn.commit()

    def _load_evidence(self, investigation_id: str) -> List[EvidenceItem]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM evidence WHERE investigation_id = ? ORDER BY timestamp",
            (investigation_id,),
        ).fetchall()

        return [
            EvidenceItem(
                id=row["id"],
                evidence_type=row["evidence_type"],
                tool_execution_id=row["tool_execution_id"] if "tool_execution_id" in row.keys() else None,
                parameter_name=row["parameter_name"] if "parameter_name" in row.keys() else None,
                numeric_value=row["numeric_value"] if "numeric_value" in row.keys() else None,
                unit=row["unit"] if "unit" in row.keys() else None,
                description=row["description"],
                source=row["source"],
                value=json.loads(row["value_json"]) if row["value_json"] else None,
                timestamp=datetime.fromisoformat(row["timestamp"]),
                verified=bool(row["verified"]),
            )
            for row in rows
        ]

    # --- Hypotheses & HypothesisEvidence Junction Table ---

    def save_hypothesis(self, investigation_id: str, hypothesis: Hypothesis) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO hypotheses
               (id, investigation_id, title, explanation, confidence,
                supporting_evidence_json, contradicting_evidence_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hypothesis.id,
                investigation_id,
                hypothesis.title,
                hypothesis.explanation,
                hypothesis.confidence,
                json.dumps(hypothesis.supporting_evidence_ids),
                json.dumps(hypothesis.contradicting_evidence_ids),
                hypothesis.created_at.isoformat(),
            ),
        )

        # Populate normalized junction table for existing evidence rows
        for ev_id in hypothesis.supporting_evidence_ids:
            ev_row = self._conn.execute("SELECT 1 FROM evidence WHERE id = ?", (ev_id,)).fetchone()
            if ev_row:
                self._conn.execute(
                    """INSERT OR REPLACE INTO hypothesis_evidence
                       (hypothesis_id, evidence_id, relationship_type)
                       VALUES (?, ?, ?)""",
                    (hypothesis.id, ev_id, "supports"),
                )

        for ev_id in hypothesis.contradicting_evidence_ids:
            ev_row = self._conn.execute("SELECT 1 FROM evidence WHERE id = ?", (ev_id,)).fetchone()
            if ev_row:
                self._conn.execute(
                    """INSERT OR REPLACE INTO hypothesis_evidence
                       (hypothesis_id, evidence_id, relationship_type)
                       VALUES (?, ?, ?)""",
                    (hypothesis.id, ev_id, "contradicts"),
                )

        self._conn.commit()

    def _load_hypotheses(self, investigation_id: str) -> List[Hypothesis]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM hypotheses WHERE investigation_id = ? ORDER BY created_at",
            (investigation_id,),
        ).fetchall()

        hypotheses = []
        for row in rows:
            hyp_id = row["id"]
            # Query junction table
            links = self._conn.execute(
                "SELECT evidence_id, relationship_type FROM hypothesis_evidence WHERE hypothesis_id = ?",
                (hyp_id,),
            ).fetchall()

            supporting = [l["evidence_id"] for l in links if l["relationship_type"] == "supports"]
            contradicting = [l["evidence_id"] for l in links if l["relationship_type"] == "contradicts"]

            # Fallback to JSON columns if junction table has no rows (legacy rows)
            if not supporting and not contradicting:
                supporting = json.loads(row["supporting_evidence_json"] or "[]")
                contradicting = json.loads(row["contradicting_evidence_json"] or "[]")

            hypotheses.append(
                Hypothesis(
                    id=hyp_id,
                    title=row["title"],
                    explanation=row["explanation"],
                    confidence=row["confidence"],
                    supporting_evidence_ids=supporting,
                    contradicting_evidence_ids=contradicting,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )

        return hypotheses

    # --- Recommendations & Human Approver Reviews ---

    def save_recommendation(
        self, investigation_id: str, recommendation: Recommendation
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT OR REPLACE INTO recommendations
               (id, investigation_id, title, description, requires_approval,
                approval_status, priority, reviewed_by, review_notes, reviewed_at,
                created_at, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                recommendation.id,
                investigation_id,
                recommendation.title,
                recommendation.description,
                1 if recommendation.requires_approval else 0,
                recommendation.approval_status.value,
                recommendation.priority,
                recommendation.reviewed_by,
                recommendation.review_notes,
                recommendation.reviewed_at.isoformat() if recommendation.reviewed_at else None,
                recommendation.created_at.isoformat(),
                recommendation.executed_at.isoformat() if recommendation.executed_at else None,
            ),
        )
        self._conn.commit()

    def review_recommendation(
        self,
        recommendation_id: str,
        approval_status: ApprovalStatus,
        reviewed_by: str,
        review_notes: Optional[str] = None,
    ) -> Optional[Recommendation]:
        """Operator sign-off: approve or reject a recommendation with human audit trail."""
        if self._conn is None:
            raise DatabaseError("Not connected")

        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """UPDATE recommendations
               SET approval_status = ?, reviewed_by = ?, review_notes = ?, reviewed_at = ?
               WHERE id = ?""",
            (approval_status.value, reviewed_by, review_notes, now, recommendation_id),
        )
        self._conn.commit()

        if cursor.rowcount == 0:
            return None

        row = self._conn.execute(
            "SELECT * FROM recommendations WHERE id = ?", (recommendation_id,)
        ).fetchone()

        if row is None:
            return None

        return Recommendation(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            requires_approval=bool(row["requires_approval"]),
            approval_status=ApprovalStatus(row["approval_status"]),
            priority=row["priority"],
            reviewed_by=row["reviewed_by"],
            review_notes=row["review_notes"],
            reviewed_at=(
                datetime.fromisoformat(row["reviewed_at"])
                if row["reviewed_at"]
                else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            executed_at=(
                datetime.fromisoformat(row["executed_at"])
                if row["executed_at"]
                else None
            ),
        )

    def _load_recommendations(self, investigation_id: str) -> List[Recommendation]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM recommendations WHERE investigation_id = ? ORDER BY created_at",
            (investigation_id,),
        ).fetchall()

        return [
            Recommendation(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                requires_approval=bool(row["requires_approval"]),
                approval_status=ApprovalStatus(row["approval_status"]),
                priority=row["priority"],
                reviewed_by=row["reviewed_by"] if "reviewed_by" in row.keys() else None,
                review_notes=row["review_notes"] if "review_notes" in row.keys() else None,
                reviewed_at=(
                    datetime.fromisoformat(row["reviewed_at"])
                    if ("reviewed_at" in row.keys() and row["reviewed_at"])
                    else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
                executed_at=(
                    datetime.fromisoformat(row["executed_at"])
                    if row["executed_at"]
                    else None
                ),
            )
            for row in rows
        ]

    def list_recommendations(
        self,
        approval_status: Optional[ApprovalStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Recommendation]:
        if self._conn is None:
            raise DatabaseError("Not connected")

        if approval_status is not None:
            rows = self._conn.execute(
                "SELECT * FROM recommendations WHERE approval_status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (approval_status.value, limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM recommendations ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

        return [
            Recommendation(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                requires_approval=bool(row["requires_approval"]),
                approval_status=ApprovalStatus(row["approval_status"]),
                priority=row["priority"],
                reviewed_by=row["reviewed_by"] if "reviewed_by" in row.keys() else None,
                review_notes=row["review_notes"] if "review_notes" in row.keys() else None,
                reviewed_at=(
                    datetime.fromisoformat(row["reviewed_at"])
                    if ("reviewed_at" in row.keys() and row["reviewed_at"])
                    else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
                executed_at=(
                    datetime.fromisoformat(row["executed_at"])
                    if row["executed_at"]
                    else None
                ),
            )
            for row in rows
        ]

    # --- Tool Executions ---

    def save_tool_execution(
        self, investigation_id: str, execution: ToolExecution
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        self._conn.execute(
            """INSERT INTO tool_executions
               (id, investigation_id, tool_name, arguments_json, result, status,
                error_message, executed_at, input_size_bytes, output_size_bytes, step_index)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                execution.id,
                investigation_id,
                execution.tool_name,
                json.dumps(execution.arguments),
                execution.result,
                execution.status.value,
                execution.error_message,
                execution.executed_at.isoformat(),
                execution.input_size_bytes,
                execution.output_size_bytes,
                execution.step_index,
            ),
        )
        self._conn.commit()

    def _load_tool_executions(self, investigation_id: str) -> List[ToolExecution]:
        if self._conn is None:
            return []

        rows = self._conn.execute(
            "SELECT * FROM tool_executions WHERE investigation_id = ? ORDER BY executed_at",
            (investigation_id,),
        ).fetchall()

        return [
            ToolExecution(
                id=row["id"],
                tool_name=row["tool_name"],
                arguments=json.loads(row["arguments_json"]),
                result=row["result"],
                status=Status(row["status"]),
                error_message=row["error_message"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
                input_size_bytes=row["input_size_bytes"],
                output_size_bytes=row["output_size_bytes"],
                step_index=row["step_index"] if "step_index" in row.keys() else 0,
            )
            for row in rows
        ]

    # --- Audit Events ---

    def save_audit_event(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        investigation_id: Optional[str] = None,
    ) -> None:
        if self._conn is None:
            raise DatabaseError("Not connected")

        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """INSERT INTO audit_events
               (id, investigation_id, event_type, message, details_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                investigation_id,
                event_type,
                message,
                json.dumps(details),
                now,
            ),
        )
        self._conn.commit()


_db: Optional[Database] = None


def get_database() -> Database:
    """Return the cached database connection, opening it on first access."""
    global _db
    if _db is None:
        config = get_config()
        _db = Database(config.database_path)
        _db.connect()
    return _db


def reset_database() -> None:
    """Close and clear the cached database. Useful for tests."""
    global _db
    if _db is not None:
        _db.close()
        _db = None
