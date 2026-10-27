"""Unified WorkforceService consolidating all modular workforce capabilities."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_DB_PATH
from .models import (
    WorkerProfile,
    AttendanceRecord,
    ShiftSchedule,
    ShiftHandover,
    WorkOrder,
    WorkforceFinding,
    WorkforceIncidentCorrelation,
    LEVELS,
)
from .registry import WorkerRegistry
from .attendance import AttendanceTracker
from .scheduling import ShiftScheduler
from .maintenance import MaintenanceManager
from .analytics import WorkforceAnalyticsEngine
from .correlation import IncidentCorrelator


class WorkforceService:
    """Consolidated facade managing worker registry, attendance, scheduling,
    maintenance work orders, explainable analytics, and incident correlation.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.path = db_path or Path(
            os.environ.get("PLANTPILOT_DATABASE_PATH", str(DEFAULT_DB_PATH))
        ).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()

        self.registry = WorkerRegistry(self.conn)
        self.attendance_tracker = AttendanceTracker(self.conn)
        self.scheduler = ShiftScheduler(self.conn)
        self.maintenance = MaintenanceManager(self.conn)
        self.analytics = WorkforceAnalyticsEngine(self.conn)
        self.correlator = IncidentCorrelator(self.conn)

    def _initialize_schema(self) -> None:
        """Create workforce tables and safely migrate columns."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                role TEXT NOT NULL,
                department TEXT NOT NULL,
                clearance_level TEXT NOT NULL CHECK(clearance_level IN ('L1','L2','L3','L4')),
                status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','INACTIVE','SUSPENDED')),
                skills_json TEXT,
                qualifications_json TEXT,
                certifications_json TEXT,
                assigned_machines_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                record_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                shift_date TEXT NOT NULL,
                clock_in TEXT NOT NULL,
                clock_out TEXT,
                break_minutes INTEGER NOT NULL DEFAULT 0 CHECK(break_minutes >= 0),
                created_at TEXT NOT NULL,
                FOREIGN KEY(worker_id) REFERENCES workers(worker_id)
            );
            CREATE INDEX IF NOT EXISTS idx_attendance_worker_date ON attendance(worker_id, shift_date);

            CREATE TABLE IF NOT EXISTS shift_schedules (
                schedule_id TEXT PRIMARY KEY,
                shift_name TEXT NOT NULL,
                shift_date TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                machine_id TEXT,
                department TEXT NOT NULL,
                required_roles_json TEXT,
                min_headcount INTEGER NOT NULL DEFAULT 1,
                min_clearance TEXT NOT NULL DEFAULT 'L1',
                assigned_worker_ids_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_schedules_date ON shift_schedules(shift_date);

            CREATE TABLE IF NOT EXISTS shift_handovers (
                handover_id TEXT PRIMARY KEY,
                shift_date TEXT NOT NULL,
                shift_name TEXT NOT NULL,
                machine_id TEXT,
                outgoing_worker_id TEXT NOT NULL,
                incoming_worker_id TEXT NOT NULL,
                notes TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(outgoing_worker_id) REFERENCES workers(worker_id),
                FOREIGN KEY(incoming_worker_id) REFERENCES workers(worker_id)
            );

            CREATE TABLE IF NOT EXISTS work_orders (
                order_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                task_description TEXT NOT NULL,
                assigned_worker_ids_json TEXT,
                required_qualifications_json TEXT,
                required_clearance TEXT NOT NULL DEFAULT 'L1',
                scheduled_start TEXT NOT NULL,
                scheduled_end TEXT NOT NULL,
                actual_start TEXT,
                actual_end TEXT,
                status TEXT NOT NULL DEFAULT 'ASSIGNED',
                inspection_required INTEGER NOT NULL DEFAULT 1,
                inspection_evidence_json TEXT,
                verification_outcome TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workforce_findings (
                finding_id TEXT PRIMARY KEY,
                worker_id TEXT,
                machine_id TEXT,
                work_order_id TEXT,
                finding_type TEXT NOT NULL DEFAULT 'ANOMALY',
                category TEXT NOT NULL DEFAULT 'GENERAL',
                severity TEXT NOT NULL,
                rule_name TEXT NOT NULL DEFAULT 'RULE_UNKNOWN',
                explanation TEXT NOT NULL DEFAULT '',
                details_json TEXT NOT NULL,
                recommended_followup TEXT,
                resolution_status TEXT NOT NULL DEFAULT 'OPEN',
                created_at TEXT NOT NULL
            );
            """
        )
        self._migrate_columns()
        self.conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_work_orders_machine ON work_orders(machine_id);
            CREATE INDEX IF NOT EXISTS idx_workforce_findings_cat ON workforce_findings(category);
            """
        )
        self.conn.commit()

    def _migrate_columns(self) -> None:
        """Safely ensure extra columns exist in older SQLite databases."""
        # Migrate workers
        cursor = self.conn.execute("PRAGMA table_info(workers)")
        w_existing = [row["name"] for row in cursor.fetchall()]
        w_cols = [
            ("skills_json", "TEXT"),
            ("qualifications_json", "TEXT"),
            ("certifications_json", "TEXT"),
            ("assigned_machines_json", "TEXT"),
        ]
        for col, c_type in w_cols:
            if col not in w_existing:
                try:
                    self.conn.execute(f"ALTER TABLE workers ADD COLUMN {col} {c_type}")
                except Exception:
                    pass

        # Migrate workforce_findings
        cursor = self.conn.execute("PRAGMA table_info(workforce_findings)")
        f_existing = [row["name"] for row in cursor.fetchall()]
        f_cols = [
            ("category", "TEXT NOT NULL DEFAULT 'GENERAL'"),
            ("machine_id", "TEXT"),
            ("work_order_id", "TEXT"),
            ("rule_name", "TEXT NOT NULL DEFAULT 'RULE_UNKNOWN'"),
            ("explanation", "TEXT NOT NULL DEFAULT ''"),
            ("recommended_followup", "TEXT"),
            ("resolution_status", "TEXT NOT NULL DEFAULT 'OPEN'"),
        ]
        for col, c_type in f_cols:
            if col not in f_existing:
                try:
                    self.conn.execute(f"ALTER TABLE workforce_findings ADD COLUMN {col} {c_type}")
                except Exception:
                    pass

    # --- Worker Registry ---
    def create_worker(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.registry.create_worker(payload).model_dump()

    def get_worker(self, worker_id: str) -> Optional[Dict[str, Any]]:
        w = self.registry.get_worker(worker_id)
        return w.model_dump() if w else None

    def list_workers(self, limit: int = 100) -> List[Dict[str, Any]]:
        return [w.model_dump() for w in self.registry.list_workers(limit=limit)]

    # --- Attendance ---
    def clock_in(
        self, worker_id: str, timestamp: str, shift_date: str, break_minutes: int = 0
    ) -> Dict[str, Any]:
        return self.attendance_tracker.clock_in(
            worker_id, timestamp, shift_date, break_minutes
        ).model_dump()

    def clock_out(self, worker_id: str, timestamp: str) -> Dict[str, Any]:
        return self.attendance_tracker.clock_out(worker_id, timestamp).model_dump()

    def get_attendance_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        r = self.attendance_tracker.get_record(record_id)
        return r.model_dump() if r else None

    def attendance(self, worker_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return [
            r.model_dump()
            for r in self.attendance_tracker.list_records_for_worker(worker_id, limit=limit)
        ]

    # --- Clearance ---
    def clearance_check(self, worker_id: str, required_level: str) -> Dict[str, Any]:
        return self.registry.check_clearance(worker_id, required_level)

    # --- Scheduling ---
    def create_schedule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.scheduler.create_schedule(payload).model_dump()

    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        s = self.scheduler.get_schedule(schedule_id)
        return s.model_dump() if s else None

    def list_schedules(
        self, shift_date: Optional[str] = None, machine_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [
            s.model_dump()
            for s in self.scheduler.list_schedules(shift_date=shift_date, machine_id=machine_id)
        ]

    def verify_shift_coverage(self, schedule_id: str) -> Dict[str, Any]:
        return self.scheduler.verify_shift_coverage(schedule_id)

    def record_handover(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.scheduler.record_handover(payload).model_dump()

    def list_handovers(
        self, shift_date: Optional[str] = None, machine_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return [
            h.model_dump()
            for h in self.scheduler.list_handovers(shift_date=shift_date, machine_id=machine_id)
        ]

    # --- Maintenance Work Orders ---
    def create_work_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.maintenance.create_work_order(payload).model_dump()

    def get_work_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        o = self.maintenance.get_work_order(order_id)
        return o.model_dump() if o else None

    def list_work_orders(
        self, machine_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return [
            o.model_dump()
            for o in self.maintenance.list_work_orders(
                machine_id=machine_id, status=status, limit=limit
            )
        ]

    def record_completion(
        self,
        order_id: str,
        actual_end: str,
        inspection_evidence: Optional[Dict[str, Any]] = None,
        inspector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.maintenance.record_completion(
            order_id=order_id,
            actual_end=actual_end,
            inspection_evidence=inspection_evidence,
            inspector_id=inspector_id,
        ).model_dump()

    # --- Analytics & Findings ---
    def detect(self, max_shift_hours: float = 12.0) -> Dict[str, Any]:
        """Run deterministic anomaly detection across attendance and fatigue."""
        findings = self.analytics.run_all_checks(max_shift_hours=max_shift_hours, persist=True)
        # Format matching existing API response
        legacy_findings = []
        for f in findings:
            item: Dict[str, Any] = {
                "finding_id": f.finding_id,
                "type": f.rule_name.replace("RULE_", ""),
                "severity": f.severity,
                "details": f.explanation,
                "category": f.category,
                "recommended_followup": f.recommended_followup,
            }
            if f.worker_id:
                item["worker_id"] = f.worker_id
            if f.machine_id:
                item["machine_id"] = f.machine_id
            if f.work_order_id:
                item["work_order_id"] = f.work_order_id
            if f.shift_date:
                item["shift_date"] = f.shift_date
            if "working_hours" in f.details:
                item["working_hours"] = f.details["working_hours"]
            if "record_count" in f.details:
                item["record_count"] = f.details["record_count"]
            legacy_findings.append(item)

        return {
            "finding_count": len(legacy_findings),
            "findings": legacy_findings,
            "analysis_mode": "deterministic_precheck",
            "note": "Findings are explainable operational review flags, not conclusions about worker conduct.",
        }

    # --- Incident Correlation ---
    def correlate_incident(
        self,
        incident_id: str,
        machine_id: Optional[str] = None,
        incident_time: Optional[str] = None,
        window_hours: float = 48.0,
    ) -> WorkforceIncidentCorrelation:
        return self.correlator.correlate_incident(
            incident_id=incident_id,
            machine_id=machine_id,
            incident_time=incident_time,
            window_hours=window_hours,
        )

    def close(self) -> None:
        self.conn.close()
