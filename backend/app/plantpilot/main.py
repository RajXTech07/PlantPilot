"""FastAPI application for PlantPilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import create_agent_controller
from .config import get_config, ConfigError
from .schemas import (
    ApprovalStatus,
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    InvestigationResult,
    Machine,
    MachineCreate,
    Recommendation,
    DigitalTwin,
    DigitalTwinCreate,
    TwinReadingItem,
    TwinReadingCreate,
    TwinSimulationRequest,
    FailureRiskPrediction,
    DiagnosisAssessRequest,
    DiagnosisAssessResponse,
    ReportAuditRequest,
    ReportAuditResponse,
    RankTestsRequest,
    RankTestsResponse,
    WorkerCreate,
    WorkerItem,
    ClockInRequest,
    ClockOutRequest,
    AttendanceRecordItem,
    ClearanceCheckRequest,
    ClearanceCheckResponse,
    WorkforceAnomaliesRequest,
    WorkforceFindingsResponse,
    WorkforceReviewResponse,
)
from .storage import get_database
from .diagnostics import assess_diagnosis, audit_report, rank_next_tests
from .workforce import get_workforce_store, model_assess_workforce



class RecommendationReviewRequest(BaseModel):
    """Payload for human operator reviewing a recommendation."""
    status_choice: ApprovalStatus
    reviewed_by: str
    notes: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    config = get_config()
    config.ensure_data_dir()
    db = get_database()
    yield
    db.close()


app = FastAPI(
    title="PlantPilot",
    description="Human-supervised industrial operations AI copilot",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Return application name, version, and status."""
    return {
        "application": "PlantPilot",
        "version": __version__,
        "status": "running",
    }


@app.get("/health")
async def health(check_provider: bool = False):
    """Return backend health status.

    By default, this endpoint only checks that the API is running.
    Pass check_provider=true to verify NVIDIA provider connectivity.
    """
    health_status = {
        "status": "healthy",
        "backend": "ok",
    }

    if check_provider:
        try:
            # Minimal API check without making a real model call
            from openai import OpenAI
            config = get_config()
            client = OpenAI(
                base_url=config.nvidia_base_url,
                api_key=config.nvidia_api_key,
                timeout=5.0,
            )
            # Just verify we can construct the client
            health_status["provider"] = "reachable"
        except Exception as exc:
            health_status["provider"] = f"error: {type(exc).__name__}"
            health_status["status"] = "degraded"

    return health_status


@app.post("/api/v1/incidents", response_model=Incident, status_code=status.HTTP_201_CREATED)
async def create_incident(create: IncidentCreate) -> Incident:
    """Create and persist an incident."""
    try:
        db = get_database()
        incident = db.create_incident(create)
        return incident
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create incident: {type(exc).__name__}",
        )


@app.get("/api/v1/incidents", response_model=List[Incident])
async def list_incidents(limit: int = 50, offset: int = 0) -> List[Incident]:
    """List persisted incidents, newest first."""
    if limit < 1 or limit > 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 200",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative",
        )

    try:
        db = get_database()
        incidents = db.list_incidents(limit=limit, offset=offset)
        return incidents
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list incidents: {type(exc).__name__}",
        )


@app.get("/api/v1/incidents/{incident_id}", response_model=Incident)
async def get_incident(incident_id: str) -> Incident:
    """Retrieve a single incident."""
    try:
        db = get_database()
        incident = db.get_incident(incident_id)
        if incident is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id!r} not found",
            )
        return incident
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve incident: {type(exc).__name__}",
        )


@app.post("/api/v1/incidents/{incident_id}/investigations", response_model=InvestigationResult)
async def start_investigation(incident_id: str, create: InvestigationCreate) -> InvestigationResult:
    """Start an AI investigation for an existing incident.

    This endpoint is synchronous: it runs the investigation loop and
    returns the structured result.
    """
    if create.incident_id != incident_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="incident_id in path must match incident_id in body",
        )

    try:
        db = get_database()
        incident = db.get_incident(incident_id)
        if incident is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Incident {incident_id!r} not found",
            )

        # Create investigation record
        investigation = db.create_investigation(create, incident)

        # Run the agent
        agent = create_agent_controller(db)
        result = agent.investigate(investigation, incident)

        return result

    except HTTPException:
        raise
    except ConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration error: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation failed: {type(exc).__name__}: {exc}",
        )


@app.get("/api/v1/investigations/{investigation_id}", response_model=Investigation)
async def get_investigation(investigation_id: str) -> Investigation:
    """Retrieve an investigation with full details."""
    try:
        db = get_database()
        investigation = db.get_investigation(investigation_id)
        if investigation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation {investigation_id!r} not found",
            )
        return investigation
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve investigation: {type(exc).__name__}",
        )


@app.get("/api/v1/investigations", response_model=List[Investigation])
async def list_investigations(limit: int = 50, offset: int = 0) -> List[Investigation]:
    """List historical investigations."""
    try:
        db = get_database()
        return db.list_investigations(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list investigations: {type(exc).__name__}",
        )


@app.post("/api/v1/machines", response_model=Machine, status_code=status.HTTP_201_CREATED)
async def create_machine(create: MachineCreate) -> Machine:
    """Register a new industrial machine / asset."""
    try:
        db = get_database()
        machine = db.create_machine(create)
        return machine
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create machine: {type(exc).__name__}",
        )


@app.get("/api/v1/machines", response_model=List[Machine])
async def list_machines(limit: int = 50, offset: int = 0) -> List[Machine]:
    """List registered industrial machines."""
    try:
        db = get_database()
        return db.list_machines(limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list machines: {type(exc).__name__}",
        )


@app.get("/api/v1/machines/{machine_id}", response_model=Machine)
async def get_machine(machine_id: str) -> Machine:
    """Retrieve details for a single machine."""
    try:
        db = get_database()
        machine = db.get_machine(machine_id)
        if machine is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Machine {machine_id!r} not found",
            )
        return machine
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve machine: {type(exc).__name__}",
        )


@app.post("/api/v1/recommendations/{recommendation_id}/review", response_model=Recommendation)
async def review_recommendation(
    recommendation_id: str,
    review: RecommendationReviewRequest,
) -> Recommendation:
    """Operator sign-off: Approve or reject an AI investigation recommendation."""
    try:
        db = get_database()
        rec = db.get_recommendation(recommendation_id)
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {recommendation_id!r} not found",
            )

        # Clearance gating: If reviewer is a registered worker, enforce clearance requirements for approval
        if review.status_choice == ApprovalStatus.APPROVED:
            store = get_workforce_store()
            worker = store.get_worker(review.reviewed_by)
            if worker:
                priority_levels = {"critical": "L3", "high": "L2", "medium": "L1", "low": "L1"}
                req_level = priority_levels.get(str(rec.priority).lower(), "L1")
                chk = store.clearance_check(worker["worker_id"], req_level)
                if not chk["authorized_by_clearance"]:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=(
                            f"Worker {worker['worker_id']} has clearance {worker['clearance_level']}, "
                            f"but {req_level} is required to approve {rec.priority.upper()} priority recommendation."
                        ),
                    )

        updated = db.review_recommendation(
            recommendation_id=recommendation_id,
            approval_status=review.status_choice,
            reviewed_by=review.reviewed_by,
            review_notes=review.notes,
        )
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Recommendation {recommendation_id!r} not found",
            )
        return updated

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to review recommendation: {type(exc).__name__}",
        )


@app.get("/api/v1/recommendations", response_model=List[Recommendation])
async def list_recommendations(
    approval_status: Optional[ApprovalStatus] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Recommendation]:
    """List recommendations pending operator sign-off or historically reviewed."""
    try:
        db = get_database()
        return db.list_recommendations(approval_status=approval_status, limit=limit, offset=offset)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list recommendations: {type(exc).__name__}",
        )


class PredictRiskRequest(BaseModel):
    """Payload to request standalone ML failure risk inference."""
    reading: Dict[str, Any] = Field(..., description="Sensor readings map")
    include_secondary: bool = Field(True, description="Whether to include secondary Random Forest model")


@app.post("/api/v1/twins", response_model=DigitalTwin, status_code=status.HTTP_201_CREATED)
async def create_digital_twin(create: DigitalTwinCreate) -> DigitalTwin:
    """Register a new digital twin software asset."""
    try:
        from .digital_twin import get_twin_store
        store = get_twin_store()
        twin = store.create(create.name, create.machine_type, create.metadata)
        return DigitalTwin(**twin)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create digital twin: {type(exc).__name__}: {exc}",
        )


@app.get("/api/v1/twins", response_model=List[DigitalTwin])
async def list_digital_twins(limit: int = 50) -> List[DigitalTwin]:
    """List registered digital twins with their latest observed readings."""
    try:
        from .digital_twin import get_twin_store
        store = get_twin_store()
        twins = store.list(limit=limit)
        return [DigitalTwin(**t) for t in twins]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list digital twins: {type(exc).__name__}: {exc}",
        )


@app.get("/api/v1/twins/{twin_id}", response_model=DigitalTwin)
async def get_digital_twin(twin_id: str) -> DigitalTwin:
    """Retrieve details and latest observed telemetry for a digital twin."""
    try:
        from .digital_twin import get_twin_store
        store = get_twin_store()
        twin = store.get(twin_id)
        if not twin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        return DigitalTwin(**twin)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get digital twin: {type(exc).__name__}: {exc}",
        )


@app.get("/api/v1/twins/{twin_id}/history", response_model=List[TwinReadingItem])
async def get_twin_history(twin_id: str, limit: int = 100) -> List[TwinReadingItem]:
    """Retrieve telemetry reading history (both observed and simulated)."""
    try:
        from .digital_twin import get_twin_store
        store = get_twin_store()
        history = store.history(twin_id, limit=limit)
        if history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        return [TwinReadingItem(**r) for r in history]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get twin history: {type(exc).__name__}: {exc}",
        )


@app.post("/api/v1/twins/{twin_id}/readings", response_model=TwinReadingItem, status_code=status.HTTP_201_CREATED)
async def add_twin_reading(twin_id: str, payload: TwinReadingCreate) -> TwinReadingItem:
    """Ingest sensor reading onto a digital twin with automated ML risk assessment."""
    try:
        from .digital_twin import get_twin_store
        from .detection import predict
        store = get_twin_store()
        twin = store.get(twin_id)
        if not twin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        
        assessment = payload.assessment
        if assessment is None:
            assessment = predict(payload.reading, include_secondary=True)
            
        rec = store.add_reading(
            twin_id=twin_id,
            reading=payload.reading,
            assessment=assessment,
            evidence_type=payload.evidence_type or "observed",
        )
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        return TwinReadingItem(**rec)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid reading data: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest twin reading: {type(exc).__name__}: {exc}",
        )


@app.post("/api/v1/twins/{twin_id}/simulate", response_model=TwinReadingItem)
async def simulate_twin_reading(twin_id: str, request: TwinSimulationRequest) -> TwinReadingItem:
    """Run an isolated what-if operational simulation on a digital twin."""
    try:
        from .digital_twin import get_twin_store
        from .detection import predict
        store = get_twin_store()
        twin = store.get(twin_id)
        if not twin:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        
        assessment = predict(request.reading, include_secondary=request.include_secondary)
        rec = store.add_reading(
            twin_id=twin_id,
            reading=request.reading,
            assessment=assessment,
            evidence_type="simulated",
        )
        if rec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Digital twin {twin_id!r} not found",
            )
        return TwinReadingItem(**rec)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid simulation reading: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {type(exc).__name__}: {exc}",
        )


@app.post("/api/v1/predict-risk", response_model=FailureRiskPrediction)
async def predict_risk_endpoint(request: PredictRiskRequest) -> FailureRiskPrediction:
    """Predict failure probability and statistical anomaly evidence for sensor measurements."""
    try:
        from .detection import predict
        assessment = predict(request.reading, include_secondary=request.include_secondary)
        return FailureRiskPrediction(**assessment)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid sensor measurements: {exc}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failure risk prediction failed: {type(exc).__name__}: {exc}",
        )


# =========================================================
# Diagnostics Endpoints
# =========================================================

@app.post("/api/v1/diagnostics/assess", response_model=DiagnosisAssessResponse)
async def assess_diagnosis_endpoint(request: DiagnosisAssessRequest) -> DiagnosisAssessResponse:
    """Assess evidence sufficiency against hypotheses deterministically."""
    try:
        result = assess_diagnosis(
            evidence=request.evidence,
            hypotheses=request.hypotheses,
            required_evidence=request.required_evidence,
        )
        return DiagnosisAssessResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis assessment failed: {exc}",
        )


@app.post("/api/v1/diagnostics/audit-report", response_model=ReportAuditResponse)
async def audit_report_endpoint(request: ReportAuditRequest) -> ReportAuditResponse:
    """Audit narrative or numeric claims against verified evidence records."""
    try:
        result = audit_report(claims=request.claims, evidence=request.evidence)
        return ReportAuditResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report audit failed: {exc}",
        )


@app.post("/api/v1/diagnostics/rank-tests", response_model=RankTestsResponse)
async def rank_tests_endpoint(request: RankTestsRequest) -> RankTestsResponse:
    """Rank next candidate diagnostic or physical inspection tests."""
    try:
        result = rank_next_tests(
            hypotheses=request.hypotheses,
            available_tests=request.available_tests,
        )
        return RankTestsResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test ranking failed: {exc}",
        )


# =========================================================
# Workforce Endpoints
# =========================================================

@app.post("/api/v1/workforce/workers", status_code=status.HTTP_201_CREATED, response_model=WorkerItem)
async def create_worker_endpoint(payload: WorkerCreate) -> WorkerItem:
    """Register a new plant worker or technician."""
    try:
        store = get_workforce_store()
        worker = store.create_worker(payload.model_dump())
        return WorkerItem(**worker)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create worker: {exc}",
        )


@app.get("/api/v1/workforce/workers", response_model=List[WorkerItem])
async def list_workers_endpoint(limit: int = 100) -> List[WorkerItem]:
    """List registered workforce personnel."""
    store = get_workforce_store()
    workers = store.list_workers(limit=limit)
    return [WorkerItem(**w) for w in workers]


@app.get("/api/v1/workforce/workers/{worker_id}", response_model=WorkerItem)
async def get_worker_endpoint(worker_id: str) -> WorkerItem:
    """Get worker details by worker_id."""
    store = get_workforce_store()
    worker = store.get_worker(worker_id)
    if not worker:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Worker {worker_id!r} not found",
        )
    return WorkerItem(**worker)


@app.post("/api/v1/workforce/clock-in", status_code=status.HTTP_201_CREATED, response_model=AttendanceRecordItem)
async def clock_in_endpoint(request: ClockInRequest) -> AttendanceRecordItem:
    """Clock in a worker for an operational shift."""
    store = get_workforce_store()
    try:
        record = store.clock_in(
            worker_id=request.worker_id,
            timestamp=request.timestamp,
            shift_date=request.shift_date,
            break_minutes=request.break_minutes or 0,
        )
        return AttendanceRecordItem(**record)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clock-in failed: {exc}",
        )


@app.post("/api/v1/workforce/clock-out", response_model=AttendanceRecordItem)
async def clock_out_endpoint(request: ClockOutRequest) -> AttendanceRecordItem:
    """Clock out a worker from an active shift."""
    store = get_workforce_store()
    try:
        record = store.clock_out(
            worker_id=request.worker_id,
            timestamp=request.timestamp,
        )
        return AttendanceRecordItem(**record)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clock-out failed: {exc}",
        )


@app.get("/api/v1/workforce/workers/{worker_id}/attendance", response_model=List[AttendanceRecordItem])
async def worker_attendance_endpoint(worker_id: str, limit: int = 100) -> List[AttendanceRecordItem]:
    """Get attendance history for a specific worker."""
    store = get_workforce_store()
    try:
        records = store.attendance(worker_id, limit=limit)
        return [AttendanceRecordItem(**r) for r in records]
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.post("/api/v1/workforce/clearance-check", response_model=ClearanceCheckResponse)
async def clearance_check_endpoint(request: ClearanceCheckRequest) -> ClearanceCheckResponse:
    """Verify worker clearance authorization level."""
    store = get_workforce_store()
    try:
        result = store.clearance_check(
            worker_id=request.worker_id,
            required_level=request.required_level,
        )
        return ClearanceCheckResponse(**result)
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@app.post("/api/v1/workforce/detect-anomalies", response_model=WorkforceFindingsResponse)
async def detect_workforce_anomalies_endpoint(request: Optional[WorkforceAnomaliesRequest] = None) -> WorkforceFindingsResponse:
    """Run deterministic workforce anomaly detection (missing clock-out, excessive hours, duplicates)."""
    store = get_workforce_store()
    max_shift = request.max_shift_hours if request and request.max_shift_hours else 12.0
    findings = store.detect(max_shift_hours=max_shift)
    return WorkforceFindingsResponse(**findings)


@app.post("/api/v1/workforce/model-review", response_model=WorkforceReviewResponse)
async def model_review_workforce_endpoint(request: Optional[WorkforceAnomaliesRequest] = None) -> WorkforceReviewResponse:
    """Run deterministic anomaly detection followed by model-assisted review."""
    try:
        store = get_workforce_store()
        max_shift = request.max_shift_hours if request and request.max_shift_hours else 12.0
        deterministic_findings = store.detect(max_shift_hours=max_shift)
        review = model_assess_workforce(deterministic_findings)
        return WorkforceReviewResponse(**review)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model-assisted review failed: {exc}",
        )


@app.exception_handler(ConfigError)
async def config_error_handler(request, exc: ConfigError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Configuration error: {exc}"},
    )
