"""Attendance, shift timing, and working-hour tracking."""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import AttendanceRecord
from .registry import parse_datetime


class AttendanceTracker:
    """Manages clock-in/out, shift calculation, break deductions, and attendance anomalies."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def clock_in(
        self,
        worker_id: str,
        timestamp: str,
        shift_date: str,
        break_minutes: int = 0,
    ) -> AttendanceRecord:
        """Record worker clock-in."""
        # Verify worker exists
        row = self.conn.execute("SELECT worker_id FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        if not row:
            raise LookupError(f"Worker {worker_id!r} not found")

        dt = parse_datetime(timestamp)
        if break_minutes < 0 or break_minutes > 1440:
            raise ValueError("break_minutes must be between 0 and 1440")

        # Disallow opening a second record if one is already open
        open_row = self.conn.execute(
            "SELECT record_id FROM attendance WHERE worker_id=? AND clock_out IS NULL",
            (worker_id,),
        ).fetchone()
        if open_row:
            raise ValueError("Worker already has an open attendance record")

        # Check overlapping shift with existing closed records
        overlap = self.conn.execute(
            """
            SELECT record_id, clock_in, clock_out FROM attendance
            WHERE worker_id = ? AND clock_out IS NOT NULL
            AND clock_in <= ? AND clock_out >= ?
            """,
            (worker_id, dt.isoformat(), dt.isoformat()),
        ).fetchone()
        if overlap:
            raise ValueError(f"Clock-in overlaps with existing attendance record {overlap['record_id']}")

        record_id = f"att_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            """
            INSERT INTO attendance (
                record_id, worker_id, shift_date, clock_in, clock_out,
                break_minutes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (record_id, worker_id, shift_date, dt.isoformat(), None, break_minutes, now),
        )
        self.conn.commit()
        return self.get_record(record_id)  # type: ignore

    def clock_out(
        self,
        worker_id: str,
        timestamp: str,
        record_id: Optional[str] = None,
    ) -> AttendanceRecord:
        """Record worker clock-out and compute working hours."""
        dt = parse_datetime(timestamp)

        if record_id:
            row = self.conn.execute(
                "SELECT * FROM attendance WHERE record_id=? AND worker_id=?",
                (record_id, worker_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM attendance WHERE worker_id=? AND clock_out IS NULL ORDER BY clock_in DESC LIMIT 1",
                (worker_id,),
            ).fetchone()

        if not row:
            raise LookupError(f"No open attendance record found for worker {worker_id!r}")

        start_dt = parse_datetime(row["clock_in"])
        if dt <= start_dt:
            raise ValueError("clock-out must be after clock-in")

        self.conn.execute(
            "UPDATE attendance SET clock_out=? WHERE record_id=?",
            (dt.isoformat(), row["record_id"]),
        )
        self.conn.commit()
        return self.get_record(row["record_id"])  # type: ignore

    def get_record(self, record_id: str) -> Optional[AttendanceRecord]:
        """Fetch attendance record with calculated working hours."""
        row = self.conn.execute("SELECT * FROM attendance WHERE record_id=?", (record_id,)).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def list_records_for_worker(self, worker_id: str, limit: int = 100) -> List[AttendanceRecord]:
        """List attendance history for a worker."""
        # Ensure worker exists
        if not self.conn.execute("SELECT worker_id FROM workers WHERE worker_id=?", (worker_id,)).fetchone():
            raise LookupError(f"Worker {worker_id!r} not found")
        rows = self.conn.execute(
            "SELECT * FROM attendance WHERE worker_id=? ORDER BY clock_in DESC LIMIT ?",
            (worker_id, limit),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_records_for_date(self, shift_date: str) -> List[AttendanceRecord]:
        """List all attendance records for a specific date."""
        rows = self.conn.execute(
            "SELECT * FROM attendance WHERE shift_date=? ORDER BY clock_in ASC",
            (shift_date,),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> AttendanceRecord:
        d = dict(row)
        working_hours = None
        if d.get("clock_out"):
            start = parse_datetime(d["clock_in"])
            end = parse_datetime(d["clock_out"])
            total_minutes = (end - start).total_seconds() / 60.0
            net_minutes = max(0.0, total_minutes - float(d.get("break_minutes", 0)))
            working_hours = round(net_minutes / 60.0, 2)

        status = "OPEN" if not d.get("clock_out") else "COMPLETE"

        return AttendanceRecord(
            record_id=d["record_id"],
            worker_id=d["worker_id"],
            shift_date=d["shift_date"],
            clock_in=d["clock_in"],
            clock_out=d.get("clock_out"),
            break_minutes=int(d.get("break_minutes", 0)),
            working_hours=working_hours,
            attendance_status=status,
            created_at=d["created_at"],
        )
