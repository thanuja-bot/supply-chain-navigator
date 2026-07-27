"""
Tests 2 & 3: Orchestration Flow Tests
======================================

Test 2: Fail → Refine → Pass  (approved path)
  - Generator produces a draft that fails initial review.
  - Refiner improves it on the first attempt.
  - Second review passes.
  - Tagger runs and tags are attached.
  - Final status: "approved"
  - RunArtifact has exactly 2 attempt records.

Test 3: Fail → Refine → Fail → Reject  (rejected path)
  - Generator produces a draft that fails initial review.
  - Refiner attempts twice (MAX_REFINEMENTS=2) — both still fail.
  - Final status: "rejected"
  - RunArtifact has 3 attempt records (initial + 2 refinements).
  - Tagger is never called.

Additional tests cover:
  - Generator failure → immediate rejected RunArtifact (no attempts)
  - Immediate pass on first draft (1 attempt, 3 LLM calls total)
  - Input snapshot excludes user_id
  - run_id is a valid UUID
  - DB persistence fields are correct for both outcomes
"""

from __future__ import annotations

import json
from unittest.mock import patch

from app.orchestrator import MAX_REFINEMENTS, run_pipeline
from app.schemas import Explanation, FinalStatus, GeneratorOutput
from tests.conftest import configure_mock_parse


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_refined_draft(base_draft: GeneratorOutput) -> GeneratorOutput:
    """Return a slightly modified draft to represent a Refiner's output."""
    return GeneratorOutput(
        explanation=Explanation(
            text=base_draft.explanation.text + " [Revised for clarity.]",
            grade=base_draft.explanation.grade,
        ),
        mcqs=base_draft.mcqs,
        teacher_notes=base_draft.teacher_notes,
    )


# ---------------------------------------------------------------------------
# Test 2: Fail → Refine → Pass (approved)
# ---------------------------------------------------------------------------

class TestOrchestrationFailRefinedPass:
    def test_fail_refine_pass_produces_approved_artifact(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        failing_review,
        passing_review,
        valid_tags,
        db_session,
    ):
        """
        Full flow: initial draft fails → refiner produces improved draft →
        second review passes → tagger tags it → status = 'approved'.
        """
        refined = _make_refined_draft(valid_draft)

        configure_mock_parse(
            mock_openai_client,
            valid_draft,     # Generator: initial draft
            failing_review,  # Reviewer:  fails initial draft
            refined,         # Refiner:   improved draft
            passing_review,  # Reviewer:  passes refined draft
            valid_tags,      # Tagger:    classifies approved content
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        # --- Final status ---
        assert artifact.final is not None
        assert artifact.final.status == FinalStatus.approved

        # --- Content and tags populated ---
        assert artifact.final.content is not None
        assert artifact.final.tags is not None
        assert artifact.final.tags.get("subject") == "Mathematics"
        assert artifact.final.error is None

        # --- Exactly 2 attempt records ---
        assert len(artifact.attempts) == 2
        assert artifact.attempts[0].attempt == 1
        assert artifact.attempts[1].attempt == 2

        # --- Attempt 1: failed review, has a refined draft ---
        assert artifact.attempts[0].review["pass"] is False
        assert artifact.attempts[0].refined is not None

        # --- Attempt 2: passing review, no further refinement ---
        assert artifact.attempts[1].review["pass"] is True
        assert artifact.attempts[1].refined is None

        # --- Timestamps both present ---
        assert artifact.timestamps.started_at is not None
        assert artifact.timestamps.finished_at is not None

    def test_approved_artifact_persisted_correctly(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        failing_review,
        passing_review,
        valid_tags,
        db_session,
    ):
        """Approved run must be stored in the DB with the correct indexed fields."""
        refined = _make_refined_draft(valid_draft)

        configure_mock_parse(
            mock_openai_client,
            valid_draft,
            failing_review,
            refined,
            passing_review,
            valid_tags,
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        from app.database import RunRecord
        record = db_session.query(RunRecord).filter_by(run_id=artifact.run_id).first()

        assert record is not None
        assert record.status == "approved"
        assert record.grade == "5"
        assert record.topic == "Fractions as parts of a whole"
        assert record.user_id == "test-user"

        stored = json.loads(record.artifact_json)
        assert stored["run_id"] == artifact.run_id
        assert stored["final"]["status"] == "approved"
        assert stored["final"]["content"] is not None
        assert stored["final"]["tags"] is not None


# ---------------------------------------------------------------------------
# Test 3: Fail → Refine → Fail → Reject
# ---------------------------------------------------------------------------

class TestOrchestrationFailRefineReject:
    def test_fail_refine_fail_produces_rejected_artifact(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        failing_review,
        db_session,
    ):
        """
        Full flow: initial draft fails → refiner attempts twice → both still
        fail → status = 'rejected'.

        Attempt count = MAX_REFINEMENTS + 1 = 3:
          Attempt 1: initial draft reviewed (fails), refined → attempt 2
          Attempt 2: 1st refined draft reviewed (fails), refined → attempt 3
          Attempt 3: 2nd refined draft reviewed (fails), budget exhausted
        """
        refined_1 = _make_refined_draft(valid_draft)
        refined_2 = _make_refined_draft(refined_1)

        configure_mock_parse(
            mock_openai_client,
            valid_draft,    # Generator:  initial draft
            failing_review, # Reviewer:   fails initial draft
            refined_1,      # Refiner:    1st refinement
            failing_review, # Reviewer:   fails 1st refined draft
            refined_2,      # Refiner:    2nd refinement
            failing_review, # Reviewer:   fails 2nd refined draft → exhausted
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        # --- Final status ---
        assert artifact.final is not None
        assert artifact.final.status == FinalStatus.rejected

        # --- No approved content or tags on rejection ---
        assert artifact.final.content is None
        assert artifact.final.tags is None

        # --- Exactly MAX_REFINEMENTS + 1 = 3 attempt records ---
        assert len(artifact.attempts) == MAX_REFINEMENTS + 1

        # --- All reviews show failure ---
        for attempt in artifact.attempts:
            assert attempt.review["pass"] is False

        # --- First two attempts have refined drafts; last one does not ---
        assert artifact.attempts[0].refined is not None
        assert artifact.attempts[1].refined is not None
        assert artifact.attempts[2].refined is None  # Budget exhausted here

    def test_rejected_artifact_persisted_correctly(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        failing_review,
        db_session,
    ):
        """Rejected run must be stored in the DB with status='rejected'."""
        refined_1 = _make_refined_draft(valid_draft)
        refined_2 = _make_refined_draft(refined_1)

        configure_mock_parse(
            mock_openai_client,
            valid_draft, failing_review,
            refined_1,   failing_review,
            refined_2,   failing_review,
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        from app.database import RunRecord
        record = db_session.query(RunRecord).filter_by(run_id=artifact.run_id).first()

        assert record is not None
        assert record.status == "rejected"

        stored = json.loads(record.artifact_json)
        assert stored["final"]["status"] == "rejected"
        assert stored["final"]["content"] is None

    def test_tagger_never_called_on_rejection(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        failing_review,
        db_session,
    ):
        """
        The Tagger must never be called when the pipeline rejects content.
        Total LLM calls = 1 gen + 3 reviews + 2 refinements = 6 (no Tagger).
        """
        refined_1 = _make_refined_draft(valid_draft)
        refined_2 = _make_refined_draft(refined_1)

        configure_mock_parse(
            mock_openai_client,
            valid_draft, failing_review,
            refined_1,   failing_review,
            refined_2,   failing_review,
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        assert mock_openai_client.beta.chat.completions.parse.call_count == 6
        assert artifact.final.tags is None


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestOrchestrationEdgeCases:
    def test_generator_failure_produces_rejected_artifact_with_error(
        self, mock_openai_client, sample_input, db_session
    ):
        """
        If the Generator fails both attempts, the run must be rejected
        immediately with an error message and 0 attempt records.
        """
        configure_mock_parse(
            mock_openai_client,
            RuntimeError("LLM unavailable"),
            RuntimeError("LLM still unavailable"),
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        assert artifact.final.status == FinalStatus.rejected
        assert artifact.final.error is not None   # Error field populated
        assert artifact.final.content is None
        assert len(artifact.attempts) == 0

    def test_immediate_pass_on_first_draft(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        passing_review,
        valid_tags,
        db_session,
    ):
        """
        If the initial draft passes review immediately:
          - Only 1 attempt record is created.
          - No refined draft on that attempt.
          - Total LLM calls = 3 (Generator + Reviewer + Tagger).
        """
        configure_mock_parse(
            mock_openai_client,
            valid_draft,
            passing_review,
            valid_tags,
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        assert artifact.final.status == FinalStatus.approved
        assert len(artifact.attempts) == 1
        assert artifact.attempts[0].refined is None  # No refinement needed
        assert mock_openai_client.beta.chat.completions.parse.call_count == 3

    def test_run_artifact_input_snapshot_excludes_user_id(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        passing_review,
        valid_tags,
        db_session,
    ):
        """
        The RunArtifact.input must capture grade and topic but NOT user_id
        (user_id is stored separately as a DB index column).
        """
        configure_mock_parse(
            mock_openai_client, valid_draft, passing_review, valid_tags
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        assert artifact.input["grade"] == 5
        assert artifact.input["topic"] == "Fractions as parts of a whole"
        assert "user_id" not in artifact.input

    def test_run_id_is_valid_uuid(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        passing_review,
        valid_tags,
        db_session,
    ):
        """Each run must have a unique, valid UUID as its run_id."""
        import uuid as _uuid

        configure_mock_parse(
            mock_openai_client, valid_draft, passing_review, valid_tags
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        parsed = _uuid.UUID(artifact.run_id)   # Raises ValueError if not a valid UUID
        assert str(parsed) == artifact.run_id

    def test_elapsed_time_logged_in_completed_run(
        self,
        mock_openai_client,
        sample_input,
        valid_draft,
        passing_review,
        valid_tags,
        db_session,
    ):
        """finished_at must be >= started_at for every completed run."""
        configure_mock_parse(
            mock_openai_client, valid_draft, passing_review, valid_tags
        )

        with patch("app.orchestrator._make_client", return_value=mock_openai_client):
            artifact = run_pipeline(sample_input, db_session)

        assert artifact.timestamps.finished_at is not None
        assert artifact.timestamps.finished_at >= artifact.timestamps.started_at
