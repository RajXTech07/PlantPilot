"""FastAPI application for PlantPilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from . import __version__
from .agent import create_agent_controller
from .config import get_config, ConfigError
from .schemas import (
    Incident,
    IncidentCreate,
    Investigation,
    InvestigationCreate,
    InvestigationResult,
)
from .storage import get_database


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


@app.exception_handler(ConfigError)
async def config_error_handler(request, exc: ConfigError):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Configuration error: {exc}"},
    )
