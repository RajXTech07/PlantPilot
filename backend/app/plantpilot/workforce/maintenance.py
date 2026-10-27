"""Industrial maintenance work orders, inspection verification, and audit."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import WorkOrder, WorkOrderStatus, VerificationOutcome, LEVELS
from .registry import parse_datetime


class MaintenanceManager:
    """Manages work orders, inspection evidence verification, and overdue maintenance audits."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_work_order(self, payload: Dict[str, Any]) -> WorkOrder:
        """Create a new industrial maintenance work order."""
        order_id = (payload.get("order_id") or f"wo_{uuid.uuid4().hex[:12]}").strip()
        machine_id = str(payload.get("machine_id", "")).strip()
        task_desc = str(payload.get("task_description", "")).strip()
        sched_start = str(payload.get("scheduled_start", "")).strip()
        sched_end = str(payload.get("scheduled_end", "")).strip()

        if not machine_id:
            raise ValueError("machine_id is required")
        if not task_desc:
            raise ValueError("task_description is required")
        if not sched_start or not sched_end:
            raise ValueError("scheduled_start and scheduled_end are required")

        # Validate machine exists if machines table is available
        normalized_machine = machine_id
        try:
            m_row = self.conn.execute(
                "SELECT id FROM machines WHERE id=? OR product_id=?",
                (machine_id, machine_id),
            ).fetchone()
            if m_row:
                normalized_machine = m_row["id"]
        except sqlite3.OperationalError:
            pass

        assigned = payload.get("assigned_worker_ids") or []
        req_quals = payload.get("required_qualifications") or []
        req_clearance = str(payload.get("required_clearance", "L1")).upper()
        if req_clearance not in LEVELS:
            raise ValueError("required_clearance must be L1, L2, L3, or L4")

        inspection_req = bool(payload.get("inspection_required", True))
        status = str(payload.get("status", "ASSIGNED")).upper()
        if status not in WorkOrderStatus.__members__:
            raise ValueError(f"Invalid work order status: {status}")

        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO work_orders (
                order_id, machine_id, task_description, assigned_worker_ids_json,
                required_qualifications_json, required_clearance, scheduled_start,
                scheduled_end, actual_start, actual_end, status,
                inspection_required, inspection_evidence_json, verification_outcome,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id,
                normalized_machine,
                task_desc,
                json.dumps(assigned),
                json.dumps(req_quals),
                req_clearance,
                sched_start,
                sched_end,
                payload.get("actual_start"),
                payload.get("actual_end"),
                status,
                1 if inspection_req else 0,
                json.dumps(payload.get("inspection_evidence")) if payload.get("inspection_evidence") else None,
                payload.get("verification_outcome", "PENDING"),
                now,
            ),
        )
        self.conn.commit()
        return self.get_work_order(order_id)  # type: ignore

    def get_work_order(self, order_id: str) -> Optional[WorkOrder]:
        """Fetch work order by ID."""
        row = self.conn.execute("SELECT * FROM work_orders WHERE order_id=?", (order_id,)).fetchone()
        if not row:
            return None
        return self._row_to_order(row)

    def list_work_orders(
        self,
        machine_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkOrder]:
        """List maintenance work orders."""
        query = "SELECT * FROM work_orders WHERE 1=1"
        params: List[Any] = []
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)
        if status:
            query += " AND status = ?"
            params.append(status.upper())
        query += " ORDER BY scheduled_start DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_order(r) for r in rows]

    def record_completion(
        self,
        order_id: str,
        actual_end: str,
        inspection_evidence: Optional[Dict[str, Any]] = None,
        inspector_id: Optional[str] = None,
    ) -> WorkOrder:
        """Record work order completion.
        
        Enforces evidence verification:
        If inspection is required, completion is ONLY marked VERIFIED if valid inspection evidence
        and an authorized inspector are provided. Otherwise, verification_outcome is UNVERIFIED.
        """
        order = self.get_work_order(order_id)
        if not order:
            raise LookupError(f"Work order {order_id!r} not found")

        dt_end = parse_datetime(actual_end)
        dt_start = parse_datetime(order.actual_start or order.scheduled_start)
        if dt_end < dt_start:
            raise ValueError("actual_end must be after actual_start / scheduled_start")

        # Determine verification outcome
        verification = "VERIFIED"
        if order.inspection_required:
            if not inspection_evidence:
                verification = "UNVERIFIED"
            else:
                # Check inspector if supplied
                if inspector_id:
                    i_row = self.conn.execute(
                        "SELECT clearance_level, status FROM workers WHERE worker_id=?",
                        (inspector_id,),
                    ).fetchone()
                    if not i_row or i_row["status"] != "ACTIVE":
                        verification = "FAILED"
                    elif LEVELS.get(i_row["clearance_level"], 0) < 2:  # Inspector needs at least L2
                        verification = "UNVERIFIED"
                else:
                    # Missing inspector
                    verification = "UNVERIFIED"

        self.conn.execute(
            """
            UPDATE work_orders SET
                status = 'COMPLETED',
                actual_end = ?,
                inspection_evidence_json = ?,
                verification_outcome = ?
            WHERE order_id = ?
            """,
            (
                dt_end.isoformat(),
                json.dumps(inspection_evidence) if inspection_evidence else None,
                verification,
                order_id,
            ),
        )
        self.conn.commit()
        return self.get_work_order(order_id)  # type: ignore

    def audit_maintenance_records(
        self, machine_id: Optional[str] = None, current_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Audit work orders for operational anomalies:
        - Overdue work
        - Unverified completed maintenance (completed without verified inspection)
        - Incomplete work orders
        - Unauthorized worker assignment (assigned worker lacks required clearance)
        """
        now_dt = parse_datetime(current_time) if current_time else datetime.now(timezone.utc)
        query = "SELECT * FROM work_orders WHERE 1=1"
        params: List[Any] = []
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)

        rows = self.conn.execute(query, params).fetchall()
        issues = []

        for r in rows:
            order = self._row_to_order(r)
            # 1. Overdue check
            try:
                sched_end_dt = parse_datetime(order.scheduled_end)
                if order.status not in {"COMPLETED", "CANCELLED"} and now_dt > sched_end_dt:
                    issues.append({
                        "order_id": order.order_id,
                        "machine_id": order.machine_id,
                        "issue_type": "OVERDUE_MAINTENANCE",
                        "severity": "medium",
                        "details": f"Work order {order.order_id} is overdue since {order.scheduled_end}",
                    })
            except Exception:
                pass

            # 2. Unverified completion check
            if order.status == "COMPLETED" and order.inspection_required:
                if order.verification_outcome != "VERIFIED":
                    issues.append({
                        "order_id": order.order_id,
                        "machine_id": order.machine_id,
                        "issue_type": "UNVERIFIED_MAINTENANCE_COMPLETION",
                        "severity": "high",
                        "details": (
                            f"Work order {order.order_id} is marked COMPLETED but inspection "
                            f"evidence is {order.verification_outcome}. Do not assume machine is operational."
                        ),
                    })

            # 3. Worker qualification / clearance check
            req_clearance_val = LEVELS.get(order.required_clearance, 1)
            for wid in order.assigned_worker_ids:
                w_row = self.conn.execute(
                    "SELECT clearance_level, status FROM workers WHERE worker_id=?", (wid,)
                ).fetchone()
                if not w_row:
                    issues.append({
                        "order_id": order.order_id,
                        "machine_id": order.machine_id,
                        "issue_type": "UNKNOWN_WORKER_ASSIGNED",
                        "severity": "high",
                        "details": f"Assigned worker {wid} does not exist in registry.",
                    })
                elif w_row["status"] != "ACTIVE":
                    issues.append({
                        "order_id": order.order_id,
                        "machine_id": order.machine_id,
                        "issue_type": "INACTIVE_WORKER_ASSIGNED",
                        "severity": "high",
                        "details": f"Assigned worker {wid} is {w_row['status']}.",
                    })
                elif LEVELS.get(w_row["clearance_level"], 0) < req_clearance_val:
                    issues.append({
                        "order_id": order.order_id,
                        "machine_id": order.machine_id,
                        "issue_type": "UNAUTHORIZED_WORKER_ASSIGNMENT",
                        "severity": "high",
                        "details": (
                            f"Worker {wid} clearance ({w_row['clearance_level']}) is below "
                            f"required level ({order.required_clearance}) for work order {order.order_id}."
                        ),
                    })

        return issues

    def _row_to_order(self, row: sqlite3.Row) -> WorkOrder:
        d = dict(row)
        assigned = []
        if d.get("assigned_worker_ids_json"):
            try:
                assigned = json.loads(d["assigned_worker_ids_json"])
            except Exception:
                pass

        quals = []
        if d.get("required_qualifications_json"):
            try:
                quals = json.loads(d["required_qualifications_json"])
            except Exception:
                pass

        inspection_ev = None
        if d.get("inspection_evidence_json"):
            try:
                inspection_ev = json.loads(d["inspection_evidence_json"])
            except Exception:
                pass

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
