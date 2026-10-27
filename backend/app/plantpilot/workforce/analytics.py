"""Workforce analytics — deterministic operational risk and anomaly detection."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import WorkforceFinding
from .registry import parse_datetime


class WorkforceAnalyticsEngine:
    """Runs explainable, deterministic rule-based checks for workforce conditions."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def run_all_checks(
        self,
        max_shift_hours: float = 12.0,
        machine_id: Optional[str] = None,
        persist: bool = True,
    ) -> List[WorkforceFinding]:
        """Execute deterministic audit across all workforce rules."""
        findings: List[WorkforceFinding] = []

        findings.extend(self._check_attendance_anomalies(max_shift_hours))
        findings.extend(self._check_qualification_anomalies())
        findings.extend(self._check_maintenance_anomalies(machine_id))
        findings.extend(self._check_staffing_anomalies(machine_id))

        if persist:
            self._persist_findings(findings)

        return findings

    def _check_attendance_anomalies(self, max_shift_hours: float) -> List[WorkforceFinding]:
        findings = []
        rows = self.conn.execute(
            """
            SELECT a.*, w.name worker_name, w.clearance_level, w.status worker_status
            FROM attendance a
            JOIN workers w ON a.worker_id = w.worker_id
            """
        ).fetchall()

        now = datetime.now(timezone.utc).isoformat()

        for r in rows:
            # 1. Missing clock-out
            if r["clock_out"] is None:
                fid = f"wfn_{uuid.uuid4().hex[:12]}"
                findings.append(
                    WorkforceFinding(
                        finding_id=fid,
                        category="ATTENDANCE",
                        severity="medium",
                        worker_id=r["worker_id"],
                        shift_date=r["shift_date"],
                        rule_name="RULE_MISSING_CLOCK_OUT",
                        explanation="Attendance record remains open without a recorded clock-out event.",
                        details={
                            "record_id": r["record_id"],
                            "clock_in": r["clock_in"],
                        },
                        recommended_followup="Verify whether the technician has completed shift or requires administrative clock-out.",
                        created_at=now,
                    )
                )
            else:
                # 2. Excessive hours / fatigue
                start_dt = parse_datetime(r["clock_in"])
                end_dt = parse_datetime(r["clock_out"])
                net_mins = (end_dt - start_dt).total_seconds() / 60.0 - float(r["break_minutes"] or 0)
                hours = round(max(0.0, net_mins) / 60.0, 2)

                if hours > max_shift_hours:
                    fid = f"wfn_{uuid.uuid4().hex[:12]}"
                    findings.append(
                        WorkforceFinding(
                            finding_id=fid,
                            category="FATIGUE",
                            severity="high",
                            worker_id=r["worker_id"],
                            shift_date=r["shift_date"],
                            rule_name="RULE_EXCESSIVE_RECORDED_HOURS",
                            explanation=(
                                f"Recorded continuous shift duration ({hours:.1f} hrs) exceeds "
                                f"the operational fatigue safety limit ({max_shift_hours:.1f} hrs)."
                            ),
                            details={
                                "record_id": r["record_id"],
                                "working_hours": hours,
                                "threshold_hours": max_shift_hours,
                            },
                            recommended_followup="Conduct immediate fatigue risk review and assign secondary operator relief if operational.",
                            created_at=now,
                        )
                    )

        # 3. Duplicate attendance records on the same date
        duplicates = self.conn.execute(
            """
            SELECT worker_id, shift_date, COUNT(*) as count
            FROM attendance
            GROUP BY worker_id, shift_date
            HAVING COUNT(*) > 1
            """
        ).fetchall()

        for d in duplicates:
            fid = f"wfn_{uuid.uuid4().hex[:12]}"
            findings.append(
                WorkforceFinding(
                    finding_id=fid,
                    category="ATTENDANCE",
                    severity="low",
                    worker_id=d["worker_id"],
                    shift_date=d["shift_date"],
                    rule_name="RULE_MULTIPLE_RECORDS_SAME_DATE",
                    explanation=f"Multiple attendance records ({d['count']}) detected on shift date {d['shift_date']}.",
                    details={"record_count": d["count"]},
                    recommended_followup="Review shift logs to confirm whether split or dual shifts were scheduled.",
                    created_at=now,
                )
            )

        return findings

    def _check_qualification_anomalies(self) -> List[WorkforceFinding]:
        findings = []
        now = datetime.now(timezone.utc)
        workers = self.conn.execute("SELECT * FROM workers").fetchall()

        for w in workers:
            raw_certs = w["certifications_json"]
            if not raw_certs:
                continue
            try:
                certs = json.loads(raw_certs)
            except Exception:
                continue

            for c in certs:
                exp_str = c.get("expires_at")
                if exp_str:
                    try:
                        exp_dt = parse_datetime(exp_str)
                        if exp_dt < now:
                            fid = f"wfn_{uuid.uuid4().hex[:12]}"
                            findings.append(
                                WorkforceFinding(
                                    finding_id=fid,
                                    category="QUALIFICATION",
                                    severity="high",
                                    worker_id=w["worker_id"],
                                    rule_name="RULE_QUALIFICATION_EXPIRED_CERTIFICATION",
                                    explanation=(
                                        f"Worker certification {c.get('name')!r} expired on {exp_str}. "
                                        f"Worker must not be assigned to restricted tasks requiring this qualification."
                                    ),
                                    details={"certification": c},
                                    recommended_followup="Schedule recertification assessment and restrict high-voltage/spindle tasks.",
                                    created_at=now.isoformat(),
                                )
                            )
                    except Exception:
                        pass
        return findings

    def _check_maintenance_anomalies(self, machine_id: Optional[str]) -> List[WorkforceFinding]:
        findings = []
        now = datetime.now(timezone.utc)
        query = "SELECT * FROM work_orders WHERE 1=1"
        params: List[Any] = []
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)

        orders = self.conn.execute(query, params).fetchall()
        for o in orders:
            # Overdue maintenance
            sched_end = o["scheduled_end"]
            status = o["status"]
            try:
                end_dt = parse_datetime(sched_end)
                if status not in {"COMPLETED", "CANCELLED"} and now > end_dt:
                    fid = f"wfn_{uuid.uuid4().hex[:12]}"
                    findings.append(
                        WorkforceFinding(
                            finding_id=fid,
                            category="MAINTENANCE",
                            severity="medium",
                            machine_id=o["machine_id"],
                            work_order_id=o["order_id"],
                            rule_name="RULE_MAINTENANCE_OVERDUE",
                            explanation=f"Work order {o['order_id']} for machine {o['machine_id']} is overdue since {sched_end}.",
                            details={"status": status, "scheduled_end": sched_end},
                            recommended_followup="Expedite maintenance execution or reschedule operational window.",
                            created_at=now.isoformat(),
                        )
                    )
            except Exception:
                pass

            # Completed without verified inspection
            if status == "COMPLETED" and o["inspection_required"]:
                outcome = o["verification_outcome"]
                if outcome != "VERIFIED":
                    fid = f"wfn_{uuid.uuid4().hex[:12]}"
                    findings.append(
                        WorkforceFinding(
                            finding_id=fid,
                            category="COMPLIANCE",
                            severity="high",
                            machine_id=o["machine_id"],
                            work_order_id=o["order_id"],
                            rule_name="RULE_MAINTENANCE_UNVERIFIED_COMPLETION",
                            explanation=(
                                f"Maintenance order {o['order_id']} was closed without verified post-work inspection evidence "
                                f"(outcome: {outcome}). Machine cannot be certified operational without verification."
                            ),
                            details={
                                "order_id": o["order_id"],
                                "verification_outcome": outcome,
                            },
                            recommended_followup="Dispatch Level-2 certified inspector for physical inspection sign-off before restarting machine.",
                            created_at=now.isoformat(),
                        )
                    )

        return findings

    def _check_staffing_anomalies(self, machine_id: Optional[str]) -> List[WorkforceFinding]:
        findings = []
        now = datetime.now(timezone.utc).isoformat()
        query = "SELECT * FROM shift_schedules WHERE 1=1"
        params: List[Any] = []
        if machine_id:
            query += " AND machine_id = ?"
            params.append(machine_id)

        schedules = self.conn.execute(query, params).fetchall()
        for s in schedules:
            # Check attendance on that date
            att_count = self.conn.execute(
                "SELECT COUNT(DISTINCT worker_id) as n FROM attendance WHERE shift_date = ?",
                (s["shift_date"],),
            ).fetchone()["n"]

            min_hc = int(s["min_headcount"])
            if att_count < min_hc:
                fid = f"wfn_{uuid.uuid4().hex[:12]}"
                findings.append(
                    WorkforceFinding(
                        finding_id=fid,
                        category="STAFFING",
                        severity="high",
                        machine_id=s["machine_id"],
                        shift_date=s["shift_date"],
                        rule_name="RULE_STAFFING_COVERAGE_GAP",
                        explanation=(
                            f"Shift {s['shift_name']} on {s['shift_date']} had verified attendance "
                            f"of {att_count} personnel against required minimum of {min_hc}."
                        ),
                        details={
                            "schedule_id": s["schedule_id"],
                            "required_headcount": min_hc,
                            "actual_headcount": att_count,
                        },
                        recommended_followup="Call in on-call certified staff to meet minimum operational staffing requirement.",
                        created_at=now,
                    )
                )

        return findings

    def _persist_findings(self, findings: List[WorkforceFinding]) -> None:
        cursor = self.conn.execute("PRAGMA table_info(workforce_findings)")
        cols = {row["name"] for row in cursor.fetchall()}
        has_finding_type = "finding_type" in cols

        for f in findings:
            finding_type_val = f.rule_name.replace("RULE_", "")
            if has_finding_type:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO workforce_findings (
                        finding_id, worker_id, machine_id, work_order_id, category,
                        severity, rule_name, explanation, details_json,
                        recommended_followup, resolution_status, created_at,
                        finding_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f.finding_id,
                        f.worker_id,
                        f.machine_id,
                        f.work_order_id,
                        f.category,
                        f.severity,
                        f.rule_name,
                        f.explanation,
                        json.dumps(f.details),
                        f.recommended_followup,
                        f.resolution_status,
                        f.created_at,
                        finding_type_val,
                    ),
                )
            else:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO workforce_findings (
                        finding_id, worker_id, machine_id, work_order_id, category,
                        severity, rule_name, explanation, details_json,
                        recommended_followup, resolution_status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f.finding_id,
                        f.worker_id,
                        f.machine_id,
                        f.work_order_id,
                        f.category,
                        f.severity,
                        f.rule_name,
                        f.explanation,
                        json.dumps(f.details),
                        f.recommended_followup,
                        f.resolution_status,
                        f.created_at,
                    ),
                )
        self.conn.commit()
