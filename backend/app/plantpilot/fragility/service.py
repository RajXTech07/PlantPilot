"""Fragility service coordinator and SQLite persistence layer."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid

from ..config import DEFAULT_DB_PATH
from .models import (
    FragilitySearchRequest,
    FragilitySearchResponse,
    OptimalityStatus,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
)
from .search import search_smallest_disturbance
from .sensitivity import analyze_parameter_sensitivities


class FragilityService:
    """Orchestrates smallest feasible disturbance search and parameter sensitivity ranking."""

    def __init__(self, db_path: Optional[Path] = None):
        self.path = db_path or Path(
            os.environ.get("PLANTPILOT_DATABASE_PATH", str(DEFAULT_DB_PATH))
        ).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create fragility tables if not exists."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fragility_searches (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                status TEXT NOT NULL,
                baseline_margin REAL NOT NULL,
                disturbance_cost REAL NOT NULL,
                evaluations_used INTEGER NOT NULL,
                optimal_disturbance_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fragility_asset ON fragility_searches(asset_id, created_at);
            """
        )
        self.conn.commit()

    def search_fragility(self, request: FragilitySearchRequest) -> FragilitySearchResponse:
        """Find the smallest feasible operating disturbance that causes a threshold breach."""
        response = search_smallest_disturbance(request)
        self._persist_search(response, request)
        return response

    def evaluate_sensitivity(self, request: SensitivityAnalysisRequest) -> SensitivityAnalysisResponse:
        """Rank operating parameters by individual fragility."""
        return analyze_parameter_sensitivities(request)

    run_sensitivity_analysis = evaluate_sensitivity

    def _persist_search(self, response: FragilitySearchResponse, request: FragilitySearchRequest) -> None:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.conn.execute(
                """
                INSERT OR REPLACE INTO fragility_searches (
                    id, asset_id, status, baseline_margin, disturbance_cost,
                    evaluations_used, optimal_disturbance_json, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.search_id,
                    response.asset_id,
                    response.status.value,
                    response.baseline_margin,
                    response.disturbance_cost if response.disturbance_cost != float("inf") else -1.0,
                    response.evaluations_used,
                    json.dumps(response.optimal_disturbance),
                    now_iso,
                    json.dumps(response.model_dump(), default=str),
                ),
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass


_fragility_service: Optional[FragilityService] = None


def get_fragility_service() -> FragilityService:
    global _fragility_service
    if _fragility_service is None:
        _fragility_service = FragilityService()
    return _fragility_service


def reset_fragility_service() -> None:
    global _fragility_service
    if _fragility_service is not None:
        try:
            _fragility_service.conn.close()
        except Exception:
            pass
    _fragility_service = None
