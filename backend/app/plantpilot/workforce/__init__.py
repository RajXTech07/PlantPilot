"""Workforce Intelligence subsystem exports and backward-compatible singleton facade."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from ..config import get_config
from .models import (
    LEVELS,
    ClearanceLevel,
    WorkerStatus,
    AttendanceStatus,
    WorkOrderStatus,
    VerificationOutcome,
    FindingCategory,
    FindingSeverity,
    CertificationRecord,
    WorkerProfile,
    AttendanceRecord,
    ShiftSchedule,
    ShiftHandover,
    WorkOrder,
    WorkforceFinding,
    WorkforceIncidentCorrelation,
)
from .registry import WorkerRegistry
from .attendance import AttendanceTracker
from .scheduling import ShiftScheduler
from .maintenance import MaintenanceManager
from .analytics import WorkforceAnalyticsEngine
from .correlation import IncidentCorrelator
from .service import WorkforceService

# Backward compatibility alias
WorkforceStore = WorkforceService

_workforce_store: Optional[WorkforceService] = None


def get_workforce_store(db_path: Optional[Path] = None) -> WorkforceService:
    """Return singleton WorkforceService instance."""
    global _workforce_store
    if _workforce_store is None:
        _workforce_store = WorkforceService(db_path=db_path)
    return _workforce_store


def reset_workforce_store() -> None:
    """Close and clear cached WorkforceService instance (used by test fixtures)."""
    global _workforce_store
    if _workforce_store is not None:
        try:
            _workforce_store.close()
        except Exception:
            pass
        _workforce_store = None


def model_assess_workforce(findings: Dict[str, Any]) -> Dict[str, Any]:
    """Ask the configured model to summarize rule findings; never replaces deterministic checks."""
    config = get_config()
    client = OpenAI(
        base_url=config.nvidia_base_url,
        api_key=config.nvidia_api_key,
        timeout=config.api_timeout_seconds,
    )
    response = client.chat.completions.create(
        model=config.model_identifier,
        temperature=0.1,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": (
                    "You review industrial workforce data quality and operational coverage flags. "
                    "Use only supplied records. Do not infer health, intent, productivity, or misconduct. "
                    "Do not invent facts. Return concise JSON with summary, prioritized_findings "
                    "(finding_id, rationale, suggested_human_followup), uncertainties. "
                    "These are flags for human review, not decisions."
                ),
            },
            {"role": "user", "content": json.dumps(findings, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "summary": content[:4000],
            "prioritized_findings": [],
            "uncertainties": [
                "Model response was not valid JSON; deterministic findings remain authoritative."
            ],
        }
    return {
        "analysis_mode": "model_assisted_review",
        "deterministic_findings": findings,
        "model_review": parsed,
    }


__all__ = [
    "LEVELS",
    "ClearanceLevel",
    "WorkerStatus",
    "AttendanceStatus",
    "WorkOrderStatus",
    "VerificationOutcome",
    "FindingCategory",
    "FindingSeverity",
    "CertificationRecord",
    "WorkerProfile",
    "AttendanceRecord",
    "ShiftSchedule",
    "ShiftHandover",
    "WorkOrder",
    "WorkforceFinding",
    "WorkforceIncidentCorrelation",
    "WorkerRegistry",
    "AttendanceTracker",
    "ShiftScheduler",
    "MaintenanceManager",
    "WorkforceAnalyticsEngine",
    "IncidentCorrelator",
    "WorkforceService",
    "WorkforceStore",
    "get_workforce_store",
    "reset_workforce_store",
    "model_assess_workforce",
]
