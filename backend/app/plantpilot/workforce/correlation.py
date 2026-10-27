"""Workforce and machine incident correlation module."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .models import (
    WorkforceIncidentCorrelation,
    WorkOrder,
    ShiftHandover,
    LEVELS,
)
from .registry import parse_datetime


class IncidentCorrelator:
    """Correlates workforce, maintenance, and shift timelines with industrial machine incidents."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def correlate_incident(
        self,
        incident_id: str,
        machine_id: Optional[str] = None,
        incident_time: Optional[str] = None,
        window_hours: float = 48.0,
    ) -> WorkforceIncidentCorrelation:
        """Correlate workforce records surrounding an incident timestamp."""
        # 1. Resolve incident time and machine
        inc_row = None
        try:
            inc_row = self.conn.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        except sqlite3.OperationalError:
            pass

        if inc_row:
            if not machine_id and inc_row["machine_id"]:
                machine_id = inc_row["machine_id"]
            if not incident_time:
                incident_time = inc_row["created_at"]

        if not incident_time:
            incident_time = datetime.now(timezone.utc).isoformat()

        inc_dt = parse_datetime(incident_time)
        window_start = inc_dt - timedelta(hours=window_hours)
        window_end = inc_dt + timedelta(hours=6.0)

        # 2. Query work orders on this machine within the window
        recent_orders: List[WorkOrder] = []
        unverified_maint = False
        missing_inspections: List[str] = []
        qualification_issues: List[str] = []

        if machine_id:
            # Query work orders
            orders_rows = self.conn.execute(
                """
                SELECT * FROM work_orders
                WHERE machine_id = ?
                ORDER BY scheduled_start DESC
                """,
                (machine_id,),
            ).fetchall()

            for r in orders_rows:
                sched_start = parse_datetime(r["scheduled_start"])
                # Check if within window
                if window_start <= sched_start <= window_end:
                    wo = self._row_to_order(r)
                    recent_orders.append(wo)

                    # Check inspection status
                    if wo.inspection_required:
                        if wo.verification_outcome != "VERIFIED":
                            unverified_maint = True
                            missing_inspections.append(
                                f"Work order {wo.order_id} ({wo.task_description}) completed with outcome {wo.verification_outcome}; verified inspection missing."
                            )

                    # Check assigned worker qualifications
                    req_clearance_val = LEVELS.get(wo.required_clearance, 1)
                    for wid in wo.assigned_worker_ids:
                        w_row = self.conn.execute(
                            "SELECT name, role, clearance_level, status FROM workers WHERE worker_id=?",
                            (wid,),
                        ).fetchone()
                        if not w_row:
                            qualification_issues.append(f"Worker {wid} assigned to {wo.order_id} not registered.")
                        elif w_row["status"] != "ACTIVE":
                            qualification_issues.append(f"Worker {wid} ({w_row['name']}) assigned while {w_row['status']}.")
                        elif LEVELS.get(w_row["clearance_level"], 0) < req_clearance_val:
                            qualification_issues.append(
                                f"Worker {wid} ({w_row['name']}) clearance {w_row['clearance_level']} below required {wo.required_clearance} for {wo.order_id}."
                            )

        # 3. Query active attendance at incident time
        inc_date = inc_dt.strftime("%Y-%m-%d")
        att_rows = self.conn.execute(
            """
            SELECT a.*, w.name, w.role, w.clearance_level, w.department
            FROM attendance a
            JOIN workers w ON a.worker_id = w.worker_id
            WHERE a.shift_date = ?
            """,
            (inc_date,),
        ).fetchall()

        active_workers = []
        for r in att_rows:
            c_in = parse_datetime(r["clock_in"])
            c_out = parse_datetime(r["clock_out"]) if r["clock_out"] else None
            # Worker is active if clocked in before or at incident, and either no clock out or clock out after incident
            if c_in <= inc_dt and (c_out is None or c_out >= inc_dt):
                active_workers.append({
                    "worker_id": r["worker_id"],
                    "name": r["name"],
                    "role": r["role"],
                    "clearance_level": r["clearance_level"],
                    "department": r["department"],
                    "clock_in": r["clock_in"],
                    "clock_out": r["clock_out"],
                })

        # 4. Query shift handovers around incident date
        handovers: List[ShiftHandover] = []
        h_rows = self.conn.execute(
            """
            SELECT * FROM shift_handovers
            WHERE shift_date = ?
            """,
            (inc_date,),
        ).fetchall()
        for hr in h_rows:
            if not machine_id or not hr["machine_id"] or hr["machine_id"] == machine_id:
                handovers.append(self._row_to_handover(hr))

        # 5. Formulate hypotheses and distinctions
        temporal_corrs: List[str] = []
        hypotheses: List[str] = []
        supported_rels: List[str] = []
        causal_warnings: List[str] = [
            "CAUSAL GUARD: Operational presence or temporal coincidence alone does NOT constitute evidence of technician error or responsibility."
        ]

        if recent_orders:
            for ro in recent_orders:
                temporal_corrs.append(
                    f"Maintenance task {ro.order_id} ({ro.task_description}) was performed on machine {ro.machine_id} within {window_hours}h of incident."
                )

        if unverified_maint:
            hypotheses.append(
                "Maintenance-verification gap: Machine incident occurred following unverified maintenance that lacked verified post-intervention inspection sign-off."
            )
            supported_rels.append(
                "Missing physical inspection verification record directly correlates with operational failure window."
            )

        if not active_workers and machine_id:
            hypotheses.append("Unattended operation: No verified active operator attendance record found at incident timestamp.")

        # 6. Generate structured EvidenceItems for investigation ingestion
        suggested_evidence = []
        if unverified_maint:
            suggested_evidence.append({
                "evidence_type": "observed",
                "description": f"Workforce Audit: Machine {machine_id} had unverified maintenance (Order {recent_orders[0].order_id if recent_orders else 'N/A'}) prior to incident.",
                "source": "workforce_maintenance_audit",
                "verified": True,
                "parameter_name": "maintenance_verification_status",
                "value": {
                    "unverified_orders": [ro.order_id for ro in recent_orders if ro.verification_outcome != "VERIFIED"],
                    "missing_inspections": missing_inspections,
                },
            })

        if active_workers:
            suggested_evidence.append({
                "evidence_type": "observed",
                "description": f"Workforce Attendance: {len(active_workers)} operators clocked in during incident window.",
                "source": "workforce_attendance_tracker",
                "verified": True,
                "parameter_name": "attending_operators_count",
                "numeric_value": float(len(active_workers)),
                "unit": "personnel",
                "value": {"active_workers": [w["worker_id"] for w in active_workers]},
            })

        return WorkforceIncidentCorrelation(
            incident_id=incident_id,
            machine_id=machine_id,
            incident_time=incident_time,
            recent_work_orders=recent_orders,
            unverified_maintenance_detected=unverified_maint,
            missing_inspections=missing_inspections,
            active_workers_at_time=active_workers,
            qualification_issues=qualification_issues,
            shift_handovers=handovers,
            temporal_correlations=temporal_corrs,
            hypotheses=hypotheses,
            supported_relationships=supported_rels,
            causal_warnings=causal_warnings,
            suggested_evidence_items=suggested_evidence,
        )

    def attach_correlation_to_investigation(
        self,
        investigation_id: str,
        correlation: WorkforceIncidentCorrelation,
        db: Any,
    ) -> List[str]:
        """Ingest workforce evidence items into SQLite investigation evidence table."""
        from ..schemas import EvidenceItem, EvidenceType

        created_evidence_ids = []
        for ev_data in correlation.suggested_evidence_items:
            ev_id = f"ev_{uuid.uuid4().hex[:12]}"
            ev_type_str = ev_data.get("evidence_type", "observed").upper()
            ev_type = getattr(EvidenceType, ev_type_str, EvidenceType.OBSERVED)

            item = EvidenceItem(
                id=ev_id,
                evidence_type=ev_type,
                description=ev_data["description"],
                source=ev_data["source"],
                parameter_name=ev_data.get("parameter_name"),
                numeric_value=ev_data.get("numeric_value"),
                unit=ev_data.get("unit"),
                value=ev_data.get("value"),
                verified=bool(ev_data.get("verified", True)),
            )
            db.save_evidence(investigation_id, item)
            created_evidence_ids.append(ev_id)

        return created_evidence_ids

    def _row_to_order(self, row: sqlite3.Row) -> WorkOrder:
        d = dict(row)
        assigned = json.loads(d.get("assigned_worker_ids_json") or "[]")
        quals = json.loads(d.get("required_qualifications_json") or "[]")
        inspection_ev = json.loads(d["inspection_evidence_json"]) if d.get("inspection_evidence_json") else None

        return WorkOrder(
            order_id=d["order_id"],
            machine_id=d["machine_id"],
            task_description=d["task_description"],
            assigned_worker_ids=assigned,
            required_qualifications=quals,
            required_clearance=d.get("required_clearance", "L1"),
            scheduled_start=d["scheduled_start"],
            scheduled_end=d["scheduled_end"],
            actual_start=d.get("actual_start"),
            actual_end=d.get("actual_end"),
            status=d.get("status", "DRAFT"),
            inspection_required=bool(d.get("inspection_required", 1)),
            inspection_evidence=inspection_ev,
            verification_outcome=d.get("verification_outcome", "PENDING"),
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
