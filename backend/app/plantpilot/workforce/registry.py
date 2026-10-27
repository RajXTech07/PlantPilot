"""Worker registry and qualification management."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import LEVELS, CertificationRecord, WorkerProfile


def parse_datetime(value: str) -> datetime:
    """Parse ISO-8601 string to timezone-aware UTC datetime."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ISO-8601 timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class WorkerRegistry:
    """Handles worker registration, qualifications, certifications, and clearance checks."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_worker(self, payload: Dict[str, Any]) -> WorkerProfile:
        """Register a new worker with full qualifications and clearance."""
        worker_id = (payload.get("worker_id") or f"wrk_{uuid.uuid4().hex[:12]}").strip()
        name = str(payload.get("name", "")).strip()
        role = str(payload.get("role", "")).strip()
        dept = str(payload.get("department", "")).strip()

        if not name:
            raise ValueError("Worker name cannot be empty")
        if not role:
            raise ValueError("Worker role cannot be empty")
        if not dept:
            raise ValueError("Worker department cannot be empty")

        level = str(payload.get("clearance_level", "L1")).upper()
        if level not in LEVELS:
            raise ValueError("clearance_level must be L1, L2, L3, or L4")

        status = str(payload.get("status", "ACTIVE")).upper()
        if status not in {"ACTIVE", "INACTIVE"}:
            raise ValueError("status must be ACTIVE or INACTIVE")

        skills = payload.get("skills") or []
        qualifications = payload.get("qualifications") or []
        raw_certs = payload.get("certifications") or []
        assigned_machines = payload.get("assigned_machines") or []

        # Validate certification objects
        cert_objects = []
        for c in raw_certs:
            if isinstance(c, dict):
                cert_objects.append(CertificationRecord(**c))
            elif isinstance(c, CertificationRecord):
                cert_objects.append(c)

        now = datetime.now(timezone.utc).isoformat()

        # Check duplicate
        existing = self.conn.execute("SELECT worker_id FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        if existing:
            raise ValueError(f"Worker ID {worker_id!r} already exists")

        try:
            self.conn.execute(
                """
                INSERT INTO workers (
                    worker_id, name, role, department, clearance_level,
                    status, skills_json, qualifications_json, certifications_json,
                    assigned_machines_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    worker_id,
                    name,
                    role,
                    dept,
                    level,
                    status,
                    json.dumps(skills),
                    json.dumps(qualifications),
                    json.dumps([c.model_dump() for c in cert_objects]),
                    json.dumps(assigned_machines),
                    now,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Worker creation failed: {exc}") from exc

        return self.get_worker(worker_id)  # type: ignore

    def get_worker(self, worker_id: str) -> Optional[WorkerProfile]:
        """Retrieve worker profile by ID."""
        row = self.conn.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def list_workers(self, limit: int = 100, department: Optional[str] = None) -> List[WorkerProfile]:
        """List worker profiles, optionally filtered by department."""
        query = "SELECT * FROM workers"
        params: List[Any] = []
        if department:
            query += " WHERE department = ?"
            params.append(department)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def update_worker_status(self, worker_id: str, status: str) -> WorkerProfile:
        """Update authorization status (ACTIVE, INACTIVE, SUSPENDED)."""
        st = status.upper()
        if st not in {"ACTIVE", "INACTIVE", "SUSPENDED"}:
            raise ValueError("status must be ACTIVE, INACTIVE, or SUSPENDED")
        if not self.get_worker(worker_id):
            raise LookupError(f"Worker {worker_id!r} not found")
        self.conn.execute("UPDATE workers SET status=? WHERE worker_id=?", (st, worker_id))
        self.conn.commit()
        return self.get_worker(worker_id)  # type: ignore

    def add_certification(self, worker_id: str, cert: CertificationRecord) -> WorkerProfile:
        """Append a new certification to a worker record."""
        worker = self.get_worker(worker_id)
        if not worker:
            raise LookupError(f"Worker {worker_id!r} not found")
        certs = [c.model_dump() for c in worker.certifications]
        certs.append(cert.model_dump())
        self.conn.execute(
            "UPDATE workers SET certifications_json=? WHERE worker_id=?",
            (json.dumps(certs), worker_id),
        )
        self.conn.commit()
        return self.get_worker(worker_id)  # type: ignore

    def check_clearance(self, worker_id: str, required_level: str) -> Dict[str, Any]:
        """Deterministic clearance authorization check.
        
        Hard rules:
        1. Worker must exist.
        2. Worker status must be ACTIVE.
        3. Worker clearance level must be >= required_level.
        """
        worker = self.get_worker(worker_id)
        if not worker:
            raise LookupError(f"Worker {worker_id!r} not found")
        req = required_level.upper()
        if req not in LEVELS:
            raise ValueError("required_level must be L1, L2, L3, or L4")

        is_active = worker.status == "ACTIVE"
        has_level = LEVELS.get(worker.clearance_level, 0) >= LEVELS[req]

        # Check expired certifications or explicitly invalid certifications
        now_dt = datetime.now(timezone.utc)
        expired_certs = []
        for c in worker.certifications:
            if not getattr(c, "valid", True):
                expired_certs.append(c.name)
            elif c.expires_at:
                try:
                    exp_dt = parse_datetime(c.expires_at)
                    if exp_dt < now_dt:
                        expired_certs.append(c.name)
                except Exception:
                    pass

        has_expired = len(expired_certs) > 0
        authorized = is_active and has_level and not has_expired

        reasons = []
        if not is_active:
            reasons.append(f"Worker status is {worker.status} (must be ACTIVE)")
        if not has_level:
            reasons.append(
                f"Worker clearance {worker.clearance_level} is below required level {req}"
            )
        if has_expired:
            reasons.append(
                f"Worker has expired or invalid certifications: {', '.join(expired_certs)}"
            )

        return {
            "worker_id": worker_id,
            "actual_clearance": worker.clearance_level,
            "required_clearance": req,
            "worker_status": worker.status,
            "authorized_by_clearance": authorized,
            "status": "AUTHORIZED" if authorized else "FLAGGED",
            "requires_human_review": not authorized,
            "unmet_requirements": reasons,
            "explanation": "Clearance verified" if authorized else "; ".join(reasons),
        }

    def check_qualifications(
        self, worker_id: str, required_qualifications: List[str]
    ) -> Dict[str, Any]:
        """Verify whether a worker possesses required qualifications and unexpired certifications."""
        worker = self.get_worker(worker_id)
        if not worker:
            raise LookupError(f"Worker {worker_id!r} not found")

        worker_quals = set(q.lower() for q in worker.qualifications)
        missing_quals = [
            rq for rq in required_qualifications if rq.lower() not in worker_quals
        ]

        # Check expired certifications
        now_dt = datetime.now(timezone.utc)
        expired_certs = []
        for c in worker.certifications:
            if c.expires_at:
                try:
                    exp_dt = parse_datetime(c.expires_at)
                    if exp_dt < now_dt:
                        expired_certs.append(c.name)
                except Exception:
                    pass

        is_qualified = (len(missing_quals) == 0) and (worker.status == "ACTIVE")
        return {
            "worker_id": worker_id,
            "qualified": is_qualified,
            "missing_qualifications": missing_quals,
            "expired_certifications": expired_certs,
            "worker_status": worker.status,
        }

    def _row_to_profile(self, row: sqlite3.Row) -> WorkerProfile:
        d = dict(row)
        # Parse JSON fields safely
        skills = self._safe_json(d.get("skills_json"), list)
        qualifications = self._safe_json(d.get("qualifications_json"), list)
        raw_certs = self._safe_json(d.get("certifications_json"), list)
        assigned = self._safe_json(d.get("assigned_machines_json"), list)

        cert_objs = [CertificationRecord(**c) for c in raw_certs if isinstance(c, dict)]

        return WorkerProfile(
            worker_id=d["worker_id"],
            name=d["name"],
            role=d["role"],
            department=d["department"],
            clearance_level=d["clearance_level"],
            status=d.get("status", "ACTIVE"),
            skills=skills,
            qualifications=qualifications,
            certifications=cert_objs,
            assigned_machines=assigned,
            created_at=d["created_at"],
        )

    @staticmethod
    def _safe_json(val: Any, default_type: type) -> Any:
        if not val:
            return default_type()
        try:
            return json.loads(val)
        except Exception:
            return default_type()
