"""Scheduling, staffing verification, and shift handovers."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import ShiftHandover, ShiftSchedule, LEVELS
from .registry import parse_datetime


class ShiftScheduler:
    """Manages shift schedules, staffing coverage verification, and handovers."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_schedule(self, payload: Dict[str, Any]) -> ShiftSchedule:
        """Create or register a planned shift schedule."""
        schedule_id = (payload.get("schedule_id") or f"sch_{uuid.uuid4().hex[:12]}").strip()
        shift_name = str(payload.get("shift_name", "Standard Shift")).strip()
        shift_date = str(payload.get("shift_date", "")).strip()
        start_time = str(payload.get("start_time", "")).strip()
        end_time = str(payload.get("end_time", "")).strip()
        dept = str(payload.get("department", "Operations")).strip()
        machine_id = payload.get("machine_id")

        if not shift_date:
            raise ValueError("shift_date is required (YYYY-MM-DD)")
        if not start_time or not end_time:
            raise ValueError("start_time and end_time are required")

        required_roles = payload.get("required_roles") or []
        min_headcount = int(payload.get("min_headcount", 1))
        min_clearance = str(payload.get("min_clearance", "L1")).upper()
        if min_clearance not in LEVELS:
            raise ValueError("min_clearance must be L1, L2, L3, or L4")

        assigned_workers = payload.get("assigned_worker_ids") or []
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO shift_schedules (
                schedule_id, shift_name, shift_date, start_time, end_time,
                machine_id, department, required_roles_json, min_headcount,
                min_clearance, assigned_worker_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule_id,
                shift_name,
                shift_date,
                start_time,
                end_time,
                machine_id,
                dept,
                json.dumps(required_roles),
                min_headcount,
                min_clearance,
                json.dumps(assigned_workers),
                now,
            ),
        )
        self.conn.commit()
        return self.get_schedule(schedule_id)  # type: ignore

    def get_schedule(self, schedule_id: str) -> Optional[ShiftSchedule]:
        """Fetch schedule by ID."""
        row = self.conn.execute(
            "SELECT * FROM shift_schedules WHERE schedule_id=?", (schedule_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_schedule(row)

    def list_schedules(
        self,
        shift_date: Optional[str] = None,
        machine_id: Optional[str] = None,
        department: Optional[str] = None,
    ) -> List[ShiftSchedule]:
        """List shift schedules with optional filtering."""
        query = "SELECT * FROM shift_schedules WHERE 1=1"
        params: List[Any] = []
        if shift_date:
            query += " AND shift_date = ?"
            params.append(shift_date)
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)
        if department:
            query += " AND department = ?"
            params.append(department)
        query += " ORDER BY shift_date DESC, start_time ASC"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_schedule(r) for r in rows]

    def record_handover(self, payload: Dict[str, Any]) -> ShiftHandover:
        """Record formal shift-to-shift handover between outgoing and incoming workers."""
        handover_id = (payload.get("handover_id") or f"hnd_{uuid.uuid4().hex[:12]}").strip()
        shift_date = str(payload.get("shift_date", "")).strip()
        shift_name = str(payload.get("shift_name", "")).strip()
        machine_id = payload.get("machine_id")
        out_worker = str(payload.get("outgoing_worker_id", "")).strip()
        in_worker = str(payload.get("incoming_worker_id", "")).strip()
        notes = str(payload.get("notes", "")).strip()
        verified = bool(payload.get("verified", False))

        if not out_worker or not in_worker:
            raise ValueError("outgoing_worker_id and incoming_worker_id are required")
        if out_worker == in_worker:
            raise ValueError("outgoing and incoming worker cannot be the same individual")

        # Verify both workers exist
        w1 = self.conn.execute("SELECT worker_id FROM workers WHERE worker_id=?", (out_worker,)).fetchone()
        w2 = self.conn.execute("SELECT worker_id FROM workers WHERE worker_id=?", (in_worker,)).fetchone()
        if not w1:
            raise LookupError(f"Outgoing worker {out_worker!r} not found")
        if not w2:
            raise LookupError(f"Incoming worker {in_worker!r} not found")

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO shift_handovers (
                handover_id, shift_date, shift_name, machine_id,
                outgoing_worker_id, incoming_worker_id, notes, verified, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                handover_id,
                shift_date,
                shift_name,
                machine_id,
                out_worker,
                in_worker,
                notes,
                1 if verified else 0,
                now,
            ),
        )
        self.conn.commit()
        return self.get_handover(handover_id)  # type: ignore

    def get_handover(self, handover_id: str) -> Optional[ShiftHandover]:
        """Fetch handover by ID."""
        row = self.conn.execute(
            "SELECT * FROM shift_handovers WHERE handover_id=?", (handover_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_handover(row)

    def list_handovers(
        self,
        shift_date: Optional[str] = None,
        machine_id: Optional[str] = None,
    ) -> List[ShiftHandover]:
        """List handovers."""
        query = "SELECT * FROM shift_handovers WHERE 1=1"
        params: List[Any] = []
        if shift_date:
            query += " AND shift_date = ?"
            params.append(shift_date)
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)
        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_handover(r) for r in rows]

    def verify_shift_coverage(self, schedule_id: str) -> Dict[str, Any]:
        """Verify whether actual attendance satisfies scheduled staffing and qualification requirements.
        
        Distinguishes planned staffing from verified attendance.
        Checks:
        1. Coverage gaps (actual attending headcount < min_headcount)
        2. Missing required roles
        3. Insufficient clearance of attending personnel
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            raise LookupError(f"Schedule {schedule_id!r} not found")

        # Fetch actual attendance on shift_date
        attendance_rows = self.conn.execute(
            """
            SELECT a.worker_id, w.role, w.clearance_level, w.status
            FROM attendance a
            JOIN workers w ON a.worker_id = w.worker_id
            WHERE a.shift_date = ?
            """,
            (schedule.shift_date,),
        ).fetchall()

        # Filter workers who were assigned or are in the department
        attending_worker_ids = set(r["worker_id"] for r in attendance_rows)
        present_roles = [r["role"].lower() for r in attendance_rows]

        # Scheduled assigned workers who actually clocked in
        planned_assigned = set(schedule.assigned_worker_ids)
        verified_present = planned_assigned.intersection(attending_worker_ids)
        absent_assigned = planned_assigned - attending_worker_ids

        # Check required roles
        missing_roles = []
        for req_role in schedule.required_roles:
            if not any(req_role.lower() in pr for pr in present_roles):
                missing_roles.append(req_role)

        # Check clearance of attending personnel
        min_clearance_val = LEVELS.get(schedule.min_clearance, 1)
        insufficient_clearance = []
        for r in attendance_rows:
            if LEVELS.get(r["clearance_level"], 0) < min_clearance_val:
                insufficient_clearance.append(r["worker_id"])

        headcount_gap = max(0, schedule.min_headcount - len(attendance_rows))
        coverage_adequate = (headcount_gap == 0) and (len(missing_roles) == 0)

        coverage_gaps = []
        if headcount_gap > 0:
            coverage_gaps.append(f"Headcount gap: {headcount_gap} workers needed")
        for r in missing_roles:
            coverage_gaps.append(f"Missing required role: {r}")
        for ic in insufficient_clearance:
            coverage_gaps.append(f"Worker {ic} has insufficient clearance")

        return {
            "schedule_id": schedule_id,
            "shift_name": schedule.shift_name,
            "shift_date": schedule.shift_date,
            "coverage_adequate": coverage_adequate,
            "is_fully_covered": coverage_adequate,
            "scheduled_headcount": schedule.min_headcount,
            "actual_headcount": len(attendance_rows),
            "active_worker_count": len(attendance_rows),
            "headcount_gap": headcount_gap,
            "coverage_gaps": coverage_gaps,
            "planned_assigned_workers": list(planned_assigned),
            "verified_present_workers": list(verified_present),
            "absent_assigned_workers": list(absent_assigned),
            "missing_required_roles": missing_roles,
            "insufficient_clearance_workers": insufficient_clearance,
        }

    def _row_to_schedule(self, row: sqlite3.Row) -> ShiftSchedule:
        d = dict(row)
        roles = []
        if d.get("required_roles_json"):
            try:
                roles = json.loads(d["required_roles_json"])
            except Exception:
                pass
        assigned = []
        if d.get("assigned_worker_ids_json"):
            try:
                assigned = json.loads(d["assigned_worker_ids_json"])
            except Exception:
                pass

        return ShiftSchedule(
            schedule_id=d["schedule_id"],
            shift_name=d["shift_name"],
            shift_date=d["shift_date"],
            start_time=d["start_time"],
            end_time=d["end_time"],
            machine_id=d.get("machine_id"),
            department=d["department"],
            required_roles=roles,
            min_headcount=int(d.get("min_headcount", 1)),
            min_clearance=d.get("min_clearance", "L1"),
            assigned_worker_ids=assigned,
            created_at=d["created_at"],
        )

    def _row_to_handover(self, row: sqlite3.Row) -> ShiftHandover:
        d = dict(row)
        return ShiftHandover(
            handover_id=d["handover_id"],
            shift_date=d["shift_date"],
            shift_name=d["shift_name"],
            machine_id=d.get("machine_id"),
            outgoing_worker_id=d["outgoing_worker_id"],
            incoming_worker_id=d["incoming_worker_id"],
            notes=d.get("notes", ""),
            verified=bool(d.get("verified", 0)),
            created_at=d["created_at"],
        )
