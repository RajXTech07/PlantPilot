"""Workforce records, attendance, clearance checks, and model-assisted review."""
from __future__ import annotations
import json, sqlite3, uuid, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from openai import OpenAI
from .config import get_config, DEFAULT_DB_PATH

LEVELS = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}

class WorkforceStore:
    def __init__(self, db_path: Path | None = None):
        self.path = db_path or Path(os.environ.get("PLANTPILOT_DATABASE_PATH", str(DEFAULT_DB_PATH))).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS workers (
          worker_id TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL,
          department TEXT NOT NULL, clearance_level TEXT NOT NULL CHECK(clearance_level IN ('L1','L2','L3','L4')),
          status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE','INACTIVE')),
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance (
          record_id TEXT PRIMARY KEY, worker_id TEXT NOT NULL, shift_date TEXT NOT NULL,
          clock_in TEXT NOT NULL, clock_out TEXT, break_minutes INTEGER NOT NULL DEFAULT 0 CHECK(break_minutes >= 0),
          created_at TEXT NOT NULL, FOREIGN KEY(worker_id) REFERENCES workers(worker_id)
        );
        CREATE INDEX IF NOT EXISTS idx_attendance_worker_date ON attendance(worker_id, shift_date);
        CREATE TABLE IF NOT EXISTS workforce_findings (
          finding_id TEXT PRIMARY KEY, worker_id TEXT, finding_type TEXT NOT NULL,
          severity TEXT NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        ''')
        self.conn.commit()

    def create_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        wid = payload.get('worker_id') or f"wrk_{uuid.uuid4().hex[:12]}"
        level = str(payload['clearance_level']).upper()
        if level not in LEVELS: raise ValueError('clearance_level must be L1, L2, L3, or L4')
        status = str(payload.get('status', 'ACTIVE')).upper()
        if status not in {'ACTIVE','INACTIVE'}: raise ValueError('status must be ACTIVE or INACTIVE')
        now = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute('INSERT INTO workers VALUES (?,?,?,?,?,?,?)', (wid, payload['name'].strip(), payload['role'].strip(), payload['department'].strip(), level, status, now)); self.conn.commit()
        except sqlite3.IntegrityError as e: raise ValueError('Worker ID already exists or worker data is invalid') from e
        return self.get_worker(wid)  # type: ignore

    def get_worker(self, worker_id: str) -> dict[str, Any] | None:
        row = self.conn.execute('SELECT * FROM workers WHERE worker_id=?', (worker_id,)).fetchone()
        return dict(row) if row else None

    def list_workers(self, limit: int = 100) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute('SELECT * FROM workers ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()]

    def clock_in(self, worker_id: str, timestamp: str, shift_date: str, break_minutes: int = 0) -> dict[str, Any]:
        if not self.get_worker(worker_id): raise LookupError('Worker not found')
        dt = _parse_dt(timestamp)
        if break_minutes < 0 or break_minutes > 1440: raise ValueError('break_minutes must be between 0 and 1440')
        open_row = self.conn.execute('SELECT record_id FROM attendance WHERE worker_id=? AND clock_out IS NULL', (worker_id,)).fetchone()
        if open_row: raise ValueError('Worker already has an open attendance record')
        rid = f"att_{uuid.uuid4().hex[:12]}"
        self.conn.execute('INSERT INTO attendance VALUES (?,?,?,?,?,?,?)', (rid, worker_id, shift_date, dt.isoformat(), None, break_minutes, datetime.now(timezone.utc).isoformat())); self.conn.commit()
        return self.get_attendance_record(rid)  # type: ignore

    def clock_out(self, worker_id: str, timestamp: str) -> dict[str, Any]:
        dt = _parse_dt(timestamp)
        row = self.conn.execute('SELECT * FROM attendance WHERE worker_id=? AND clock_out IS NULL ORDER BY clock_in DESC LIMIT 1', (worker_id,)).fetchone()
        if not row: raise LookupError('No open attendance record for worker')
        start = _parse_dt(row['clock_in'])
        if dt <= start: raise ValueError('clock-out must be after clock-in')
        self.conn.execute('UPDATE attendance SET clock_out=? WHERE record_id=?', (dt.isoformat(), row['record_id'])); self.conn.commit()
        return self.get_attendance_record(row['record_id'])  # type: ignore

    def get_attendance_record(self, record_id: str) -> dict[str, Any] | None:
        row = self.conn.execute('SELECT * FROM attendance WHERE record_id=?', (record_id,)).fetchone()
        return self._attendance_dict(row) if row else None

    def attendance(self, worker_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if not self.get_worker(worker_id): raise LookupError('Worker not found')
        return [self._attendance_dict(r) for r in self.conn.execute('SELECT * FROM attendance WHERE worker_id=? ORDER BY clock_in DESC LIMIT ?', (worker_id, limit)).fetchall()]

    @staticmethod
    def _attendance_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row); d['working_hours'] = None
        if d['clock_out']:
            mins = (_parse_dt(d['clock_out']) - _parse_dt(d['clock_in'])).total_seconds()/60 - d['break_minutes']
            d['working_hours'] = round(max(0, mins)/60, 2)
        d['attendance_status'] = 'OPEN' if not d['clock_out'] else 'COMPLETE'
        return d

    def clearance_check(self, worker_id: str, required_level: str) -> dict[str, Any]:
        worker = self.get_worker(worker_id)
        if not worker: raise LookupError('Worker not found')
        required = required_level.upper()
        if required not in LEVELS: raise ValueError('required_level must be L1, L2, L3, or L4')
        allowed = worker['status'] == 'ACTIVE' and LEVELS[worker['clearance_level']] >= LEVELS[required]
        return {'worker_id': worker_id, 'actual_clearance': worker['clearance_level'], 'required_clearance': required, 'authorized_by_clearance': allowed, 'status': 'AUTHORIZED' if allowed else 'FLAGGED', 'requires_human_review': not allowed}

    def detect(self, max_shift_hours: float = 12.0) -> dict[str, Any]:
        findings = []
        rows = self.conn.execute('SELECT a.*, w.clearance_level, w.status worker_status FROM attendance a JOIN workers w USING(worker_id)').fetchall()
        for r in rows:
            if r['clock_out'] is None:
                findings.append({'type':'MISSING_CLOCK_OUT','severity':'medium','worker_id':r['worker_id'],'record_id':r['record_id'],'details':'Attendance record remains open; verify whether the worker has clocked out.'})
            else:
                hours = self._attendance_dict(r)['working_hours']
                if hours is not None and hours > max_shift_hours:
                    findings.append({'type':'EXCESSIVE_RECORDED_HOURS','severity':'high','worker_id':r['worker_id'],'record_id':r['record_id'],'working_hours':hours,'threshold_hours':max_shift_hours,'details':'Recorded shift exceeds the configured review threshold; verify record and applicable policy.'})
        duplicates = self.conn.execute('SELECT worker_id, shift_date, COUNT(*) n FROM attendance GROUP BY worker_id, shift_date HAVING COUNT(*) > 1').fetchall()
        for r in duplicates: findings.append({'type':'MULTIPLE_RECORDS_SAME_DATE','severity':'low','worker_id':r['worker_id'],'shift_date':r['shift_date'],'record_count':r['n'],'details':'Multiple attendance records exist on the same date; review if multiple shifts are expected.'})
        for f in findings:
            fid=f"wfn_{uuid.uuid4().hex[:12]}"
            self.conn.execute('INSERT INTO workforce_findings VALUES (?,?,?,?,?,?)',(fid,f.get('worker_id'),f['type'],f['severity'],json.dumps(f),datetime.now(timezone.utc).isoformat()))
            f['finding_id']=fid
        self.conn.commit()
        return {'finding_count':len(findings),'findings':findings,'analysis_mode':'deterministic_precheck','note':'Findings are review flags, not conclusions about worker conduct.'}

    def close(self): self.conn.close()

def _parse_dt(value: str) -> datetime:
    try: dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (TypeError, ValueError) as exc: raise ValueError('timestamp must be ISO-8601') from exc
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt


def model_assess_workforce(findings: dict[str, Any]) -> dict[str, Any]:
    """Ask the configured existing model to summarize rule findings; never replaces deterministic checks."""
    
    config = get_config()
    client = OpenAI(base_url=config.nvidia_base_url, api_key=config.nvidia_api_key, timeout=config.api_timeout_seconds)
    response = client.chat.completions.create(model=config.model_identifier, temperature=0.1, max_tokens=1200, messages=[
      {'role':'system','content':'You review industrial workforce data quality and operational coverage flags. Use only supplied records. Do not infer health, intent, productivity, or misconduct. Do not invent facts. Return concise JSON with summary, prioritized_findings (finding_id, rationale, suggested_human_followup), uncertainties. These are flags for human review, not decisions.'},
      {'role':'user','content':json.dumps(findings, ensure_ascii=False)}])
    content = response.choices[0].message.content or '{}'
    try: parsed=json.loads(content)
    except json.JSONDecodeError: parsed={'summary':content[:4000], 'prioritized_findings':[], 'uncertainties':['Model response was not valid JSON; deterministic findings remain authoritative.']}
    return {'analysis_mode':'model_assisted_review','deterministic_findings':findings,'model_review':parsed}


_workforce_store: WorkforceStore | None = None

def get_workforce_store(db_path: Path | None = None) -> WorkforceStore:
    global _workforce_store
    if _workforce_store is None:
        _workforce_store = WorkforceStore(db_path=db_path)
    return _workforce_store

def reset_workforce_store() -> None:
    global _workforce_store
    if _workforce_store is not None:
        try:
            _workforce_store.close()
        except Exception:
            pass
        _workforce_store = None

