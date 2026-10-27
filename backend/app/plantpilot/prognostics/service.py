"""Prognostics service coordinator and SQLite persistence layer."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from ..config import DEFAULT_DB_PATH
from .models import (
    CrossingPolicy,
    ForecastRequest,
    ForecastResponse,
    ForecastStatus,
    PhysicalFailureRequest,
    PhysicalFailureResponse,
    PredictionTarget,
    ThresholdDefinition,
)
from .threshold import find_crossing_index, safety_margin
from .forecasting import forecast_trajectory, ForecastValidationError
from .failure_model import evaluate_physical_failure
from .uncertainty import calculate_prediction_interval


class PrognosticsService:
    """Orchestrates Time-to-Breach forecasting and physical failure time analysis."""

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
        """Create prognostics table if not exists."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prognostics_forecasts (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                status TEXT NOT NULL,
                prediction_target TEXT NOT NULL,
                threshold_json TEXT NOT NULL,
                predicted_time TEXT,
                predicted_seconds REAL,
                baseline_margin REAL NOT NULL,
                model_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_prognostics_asset ON prognostics_forecasts(asset_id, created_at);
            """
        )
        self.conn.commit()

    def forecast_breach(self, request: ForecastRequest) -> ForecastResponse:
        """Estimate earliest time to threshold breach."""
        # 1. Validation and trajectory fit
        try:
            forecast_t, forecast_y, t_last_epoch, metadata = forecast_trajectory(request)
        except ForecastValidationError as exc:
            return ForecastResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INVALID_INPUT,
                prediction_target=request.prediction_target,
                baseline_margin=0.0,
                assumptions=[],
                limitations=[str(exc)],
            )
        except Exception as exc:
            return ForecastResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.INVALID_INPUT,
                prediction_target=request.prediction_target,
                baseline_margin=0.0,
                limitations=[f"Forecast computation error: {exc}"],
            )

        # Baseline margin at last observed point
        last_obs_val = None
        for k, v in request.observations[-1].values.items():
            if k.lower().replace(" ", "_") == request.threshold.parameter.lower().replace(" ", "_"):
                last_obs_val = float(v)
                break
        base_margin = safety_margin(last_obs_val or forecast_y[0], request.threshold)

        # 2. Check crossing policy on trajectory
        crossing_idx, crossing_t_sec = find_crossing_index(
            times=forecast_t.tolist(),
            values=forecast_y.tolist(),
            threshold=request.threshold,
            policy=request.crossing_policy,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        sample_pts = [
            {"t_sec": round(float(t), 1), "value": round(float(v), 3)}
            for t, v in zip(forecast_t[:: max(1, len(forecast_t) // 10)], forecast_y[:: max(1, len(forecast_y) // 10)])
        ]

        if crossing_t_sec is not None and crossing_t_sec <= request.forecast_horizon_seconds:
            # Breach occurs within horizon
            pred_dt = datetime.fromtimestamp(t_last_epoch + crossing_t_sec, tz=timezone.utc)
            uncertainty = calculate_prediction_interval(
                t_breach_sec=crossing_t_sec,
                t_last_epoch=t_last_epoch,
                metadata=metadata,
                threshold=request.threshold,
            )
            response = ForecastResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.BREACH_PREDICTED,
                prediction_target=request.prediction_target,
                predicted_time=pred_dt,
                predicted_time_seconds_from_last_obs=round(crossing_t_sec, 2),
                uncertainty=uncertainty,
                baseline_margin=round(base_margin, 4),
                assumptions=[
                    f"Model '{request.model_type}' extrapolated over {request.forecast_horizon_seconds}s horizon.",
                    f"Crossing policy: {request.crossing_policy.value}.",
                    "Steady operating conditions without unmodeled step-disturbances.",
                ],
                evidence_references=[f"obs_{len(request.observations)}_points"],
                model_version="1.0.0",
                limitations=[
                    "Forecast reliability decreases as lookahead approaches the horizon boundary.",
                    "Physical equipment degradation may accelerate non-linearly near catastrophic failure.",
                ],
                trajectory_sample=sample_pts,
            )
        else:
            # No breach within horizon
            response = ForecastResponse(
                asset_id=request.asset_id,
                status=ForecastStatus.NO_BREACH_WITHIN_HORIZON,
                prediction_target=request.prediction_target,
                predicted_time=None,
                predicted_time_seconds_from_last_obs=None,
                uncertainty=None,
                baseline_margin=round(base_margin, 4),
                assumptions=[
                    f"Forecast lookahead evaluated to horizon of {request.forecast_horizon_seconds}s.",
                ],
                evidence_references=[f"obs_{len(request.observations)}_points"],
                model_version="1.0.0",
                limitations=[
                    f"No threshold crossing was detected within the validated {request.forecast_horizon_seconds}s horizon.",
                    "IMPORTANT: Absence of predicted crossing within the horizon does NOT constitute proof that the asset is safe.",
                ],
                trajectory_sample=sample_pts,
            )

        # Persist forecast record
        self._persist_forecast(response, request)
        return response

    generate_forecast = forecast_breach

    def evaluate_failure(self, request: PhysicalFailureRequest) -> PhysicalFailureResponse:
        """Estimate physical time-to-failure."""
        response = evaluate_physical_failure(request)
        self._persist_failure_result(response, request)
        return response

    evaluate_physical_failure = evaluate_failure

    def _persist_forecast(self, response: ForecastResponse, request: ForecastRequest) -> None:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            pred_time_iso = response.predicted_time.isoformat() if response.predicted_time else None
            self.conn.execute(
                """
                INSERT OR REPLACE INTO prognostics_forecasts (
                    id, asset_id, status, prediction_target, threshold_json,
                    predicted_time, predicted_seconds, baseline_margin, model_type,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.forecast_id,
                    response.asset_id,
                    response.status.value,
                    response.prediction_target.value,
                    json.dumps(request.threshold.model_dump()),
                    pred_time_iso,
                    response.predicted_time_seconds_from_last_obs,
                    response.baseline_margin,
                    request.model_type,
                    now_iso,
                    json.dumps(response.model_dump(), default=str),
                ),
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass

    def _persist_failure_result(self, response: PhysicalFailureResponse, request: PhysicalFailureRequest) -> None:
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            pred_time_iso = response.estimated_failure_time.isoformat() if response.estimated_failure_time else None
            self.conn.execute(
                """
                INSERT OR REPLACE INTO prognostics_forecasts (
                    id, asset_id, status, prediction_target, threshold_json,
                    predicted_time, predicted_seconds, baseline_margin, model_type,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    response.request_id,
                    response.asset_id,
                    response.status.value,
                    PredictionTarget.PHYSICAL_FAILURE.value,
                    json.dumps({"failure_mode": response.failure_mode}),
                    pred_time_iso,
                    response.estimated_rul_seconds,
                    0.0,
                    response.degradation_model,
                    now_iso,
                    json.dumps(response.model_dump(), default=str),
                ),
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            pass


_prognostics_service: Optional[PrognosticsService] = None


def get_prognostics_service() -> PrognosticsService:
    global _prognostics_service
    if _prognostics_service is None:
        _prognostics_service = PrognosticsService()
    return _prognostics_service


def reset_prognostics_service() -> None:
    global _prognostics_service
    if _prognostics_service is not None:
        try:
            _prognostics_service.conn.close()
        except Exception:
            pass
    _prognostics_service = None
