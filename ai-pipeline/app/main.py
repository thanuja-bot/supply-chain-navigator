"""
Governed, Auditable AI Content Pipeline — FastAPI Application
=============================================================
Entry point for the FastAPI application.

Endpoints:
  POST /generate          → runs the full 4-agent pipeline, returns RunArtifact
  GET  /history           → returns stored run artifacts (filterable by user_id)
  GET  /runs/{run_id}     → returns a specific run artifact by ID
  GET  /healthz           → liveness probe

Environment variables:
  OPENAI_API_KEY          → required for all LLM agents
  DATABASE_URL            → SQLite path or PostgreSQL URL (default: ./ai_pipeline.db)
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import RunRecord, get_db, init_db
from app.orchestrator import run_pipeline
from app.schemas import ContentInput, RunArtifact

# Load .env file for local development (no-op in production)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database")
    init_db()
    logger.info("Database ready")
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Governed, Auditable AI Content Pipeline",
    description=(
        "A production-grade multi-agent pipeline for generating, reviewing, "
        "refining, and tagging educational content with full audit trails."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "service": "ai-content-pipeline"}


@app.post(
    "/generate",
    response_model=RunArtifact,
    tags=["Pipeline"],
    summary="Run the full 4-agent content generation pipeline",
    response_description="Complete RunArtifact with full audit trail",
)
def generate(
    payload: ContentInput,
    db: Session = Depends(get_db),
) -> RunArtifact:
    """
    Execute the governed AI content pipeline:

    1. **Generator** — produces a structured educational draft
    2. **Reviewer** — deterministically scores and gates the draft (pass/fail)
    3. **Refiner** — improves failed drafts (max 2 attempts)
    4. **Tagger** — classifies approved content only

    Returns a complete **RunArtifact** capturing every step, score,
    refinement, and final decision for full auditability.
    """
    logger.info(
        "POST /generate — grade=%d topic='%s' user_id=%s",
        payload.grade,
        payload.topic,
        payload.user_id,
    )
    try:
        artifact = run_pipeline(payload, db)
        return artifact
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/history",
    tags=["History"],
    summary="Retrieve stored run artifacts",
)
def get_history(
    user_id: Optional[str] = Query(default=None, description="Filter by user ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(
        default=None, description="Filter by status: 'approved' or 'rejected'"
    ),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Retrieve stored run artifacts with optional filtering and pagination.

    - `user_id`: filter to runs from a specific user
    - `status`: filter by final status (`approved` or `rejected`)
    - `limit` / `offset`: pagination controls
    """
    query = db.query(RunRecord)

    if user_id:
        query = query.filter(RunRecord.user_id == user_id)
    if status:
        if status not in ("approved", "rejected"):
            raise HTTPException(
                status_code=400,
                detail="status must be 'approved' or 'rejected'",
            )
        query = query.filter(RunRecord.status == status)

    total = query.count()
    records = (
        query.order_by(RunRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": [json.loads(r.artifact_json) for r in records],
    }


@app.get(
    "/runs/{run_id}",
    tags=["History"],
    summary="Retrieve a specific run artifact by ID",
)
def get_run(run_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve the full RunArtifact for a specific run by its UUID."""
    record = db.query(RunRecord).filter(RunRecord.run_id == run_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return json.loads(record.artifact_json)
