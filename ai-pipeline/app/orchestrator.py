"""
Pipeline Orchestrator
=====================
Runs a deterministic, bounded multi-agent workflow and compiles a complete
RunArtifact that serves as the full audit trail.

Workflow (deterministic, bounded):
  1. Generator Agent → draft (retry-once on validation failure)
  2. Reviewer Agent  → score + pass/fail (enforced deterministically by schema)
     a. If pass → Tagger Agent → tags → RunArtifact(status=approved)
     b. If fail → Refiner Agent (up to MAX_REFINEMENTS=2 attempts)
        After each refinement:
          → Reviewer Agent re-evaluates
          → If pass → Tagger → approved
          → If still fail after all refinements → rejected
  3. Persist RunArtifact to database.

LLM call budget (worst case):
  - Generator : 2  (1 initial + 1 retry)
  - Reviewer  : 3  (1 initial + 1 per refinement)
  - Refiner   : 2  (MAX_REFINEMENTS)
  - Tagger    : 1  (approved only)
  Total       : ≤ 8 LLM calls per run
"""

import logging
import os
from datetime import datetime, timezone

from openai import OpenAI
from sqlalchemy.orm import Session

from app.agents.generator import run_generator
from app.agents.refiner import run_refiner
from app.agents.reviewer import run_reviewer
from app.agents.tagger import run_tagger
from app.database import RunRecord
from app.schemas import (
    AttemptRecord,
    ContentInput,
    FinalRecord,
    FinalStatus,
    GeneratorOutput,
    RunArtifact,
    Timestamps,
)

logger = logging.getLogger(__name__)

MAX_REFINEMENTS = 2


def _make_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to your .env file and restart the server."
        )
    return OpenAI(api_key=api_key)


def run_pipeline(content_input: ContentInput, db: Session) -> RunArtifact:
    """
    Execute the full governed pipeline and return a persisted RunArtifact.

    Args:
        content_input: Validated grade + topic input.
        db: SQLAlchemy session for persistence.

    Returns:
        RunArtifact: Complete audit trail of the run, regardless of outcome.
    """
    client = _make_client()
    started_at = datetime.now(timezone.utc)

    artifact = RunArtifact(
        input=content_input.model_dump(exclude={"user_id"}),
        timestamps=Timestamps(started_at=started_at),
    )

    logger.info(
        "Pipeline START run_id=%s grade=%d topic='%s'",
        artifact.run_id,
        content_input.grade,
        content_input.topic,
    )

    # -----------------------------------------------------------------------
    # Step 1: Generate initial draft
    # -----------------------------------------------------------------------
    try:
        current_draft: GeneratorOutput = run_generator(client, content_input)
    except RuntimeError as exc:
        # Generator failed after its internal retry budget — reject immediately.
        artifact.final = FinalRecord(
            status=FinalStatus.rejected,
            error=str(exc),
        )
        artifact.timestamps.finished_at = datetime.now(timezone.utc)
        logger.error("Pipeline FAILED at Generator: %s", exc)
        _persist(artifact, content_input, db)
        return artifact

    # -----------------------------------------------------------------------
    # Step 2: Review → Refine loop (bounded by MAX_REFINEMENTS)
    # -----------------------------------------------------------------------
    final_status: FinalStatus | None = None
    approved_draft: GeneratorOutput | None = None

    for refinement_num in range(MAX_REFINEMENTS + 1):
        # --- Review current draft ---
        try:
            review = run_reviewer(client, content_input, current_draft)
        except RuntimeError as exc:
            logger.error(
                "Reviewer failed on refinement round %d: %s", refinement_num, exc
            )
            artifact.final = FinalRecord(
                status=FinalStatus.rejected,
                error=str(exc),
            )
            artifact.timestamps.finished_at = datetime.now(timezone.utc)
            _persist(artifact, content_input, db)
            return artifact

        # Build attempt record (refined will be populated below if needed)
        attempt_record = AttemptRecord(
            attempt=refinement_num + 1,
            draft=current_draft.model_dump(),
            review=review.model_dump(by_alias=True),
            refined=None,
        )

        if review.passed:
            # --- Approved ---
            logger.info(
                "Pipeline: draft APPROVED on attempt %d (avg=%.2f)",
                refinement_num + 1,
                review.scores.average,
            )
            artifact.attempts.append(attempt_record)
            approved_draft = current_draft
            final_status = FinalStatus.approved
            break

        # --- Not approved ---
        logger.info(
            "Pipeline: draft FAILED review on attempt %d "
            "(avg=%.2f, min=%d, feedback_items=%d)",
            refinement_num + 1,
            review.scores.average,
            review.scores.min_score,
            len(review.feedback),
        )

        if refinement_num >= MAX_REFINEMENTS:
            # Refinement budget exhausted — reject.
            logger.warning(
                "Pipeline: max refinements (%d) exhausted. Rejecting.",
                MAX_REFINEMENTS,
            )
            artifact.attempts.append(attempt_record)
            final_status = FinalStatus.rejected
            break

        # --- Refine ---
        try:
            refined_draft = run_refiner(
                client,
                content_input,
                current_draft,
                review,
                attempt_number=refinement_num + 1,
            )
        except RuntimeError as exc:
            logger.error(
                "Refiner failed on attempt %d: %s", refinement_num + 1, exc
            )
            attempt_record.refined = {"error": str(exc)}
            artifact.attempts.append(attempt_record)
            final_status = FinalStatus.rejected
            break

        attempt_record.refined = refined_draft.model_dump()
        artifact.attempts.append(attempt_record)
        current_draft = refined_draft  # Feed improved draft back into review loop

    # -----------------------------------------------------------------------
    # Step 3: Tag (only on approval)
    # -----------------------------------------------------------------------
    tags = None
    if final_status == FinalStatus.approved and approved_draft:
        try:
            tagger_output = run_tagger(client, content_input, approved_draft)
            tags = tagger_output.model_dump()
        except RuntimeError as exc:
            logger.warning(
                "Tagger failed (non-fatal, content still approved): %s", exc
            )
            tags = {"error": str(exc)}

    # -----------------------------------------------------------------------
    # Step 4: Compile final RunArtifact
    # -----------------------------------------------------------------------
    artifact.final = FinalRecord(
        status=final_status or FinalStatus.rejected,
        content=approved_draft.model_dump() if approved_draft else None,
        tags=tags,
    )
    artifact.timestamps.finished_at = datetime.now(timezone.utc)

    logger.info(
        "Pipeline COMPLETE run_id=%s status=%s attempts=%d elapsed=%.1fs",
        artifact.run_id,
        artifact.final.status,
        len(artifact.attempts),
        (artifact.timestamps.finished_at - artifact.timestamps.started_at).total_seconds(),
    )

    # -----------------------------------------------------------------------
    # Step 5: Persist to database
    # -----------------------------------------------------------------------
    _persist(artifact, content_input, db)
    return artifact


def _persist(artifact: RunArtifact, content_input: ContentInput, db: Session) -> None:
    """Persist the RunArtifact to the database.  Rolls back on failure."""
    try:
        record = RunRecord(
            run_id=artifact.run_id,
            user_id=content_input.user_id,
            topic=content_input.topic,
            grade=str(content_input.grade),
            status=artifact.final.status.value if artifact.final else "unknown",
            artifact_json=artifact.model_dump_json(),
            started_at=artifact.timestamps.started_at,
            finished_at=artifact.timestamps.finished_at,
        )
        db.add(record)
        db.commit()
        logger.info("Persisted run_id=%s to database", artifact.run_id)
    except Exception as exc:
        logger.error("Failed to persist run_id=%s: %s", artifact.run_id, exc)
        db.rollback()
        raise
