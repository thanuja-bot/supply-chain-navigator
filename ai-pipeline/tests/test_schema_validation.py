"""
Test 1: Schema Validation Failure Handling
==========================================
Verifies that:
  a) Pydantic schemas reject invalid data with clear ValidationErrors.
  b) The Generator Agent retries exactly once when the LLM returns invalid
     data, then raises RuntimeError after both attempts fail.
  c) A single validation failure followed by a valid response succeeds.
  d) The Reviewer's pass/fail decision is deterministic (score-driven).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.generator import run_generator
from app.schemas import (
    Explanation,
    GeneratorOutput,
    MCQ,
    ReviewerOutput,
    ReviewScores,
    TeacherNotes,
)
from tests.conftest import configure_mock_parse


# ---------------------------------------------------------------------------
# MCQ schema constraints
# ---------------------------------------------------------------------------

class TestMCQSchema:
    def test_rejects_wrong_option_count(self):
        """MCQ must have exactly 4 options — fewer should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MCQ(
                question="What is 1/2?",
                options=["A half", "A quarter"],  # Only 2 — invalid
                correct_index=0,
            )
        error_text = str(exc_info.value)
        assert "4" in error_text or "min_length" in error_text

    def test_rejects_too_many_options(self):
        """MCQ must have exactly 4 options — more should raise ValidationError."""
        with pytest.raises(ValidationError):
            MCQ(
                question="What is 1/2?",
                options=["A", "B", "C", "D", "E"],  # 5 — invalid
                correct_index=0,
            )

    def test_rejects_invalid_correct_index(self):
        """correct_index must be in [0, 3]."""
        with pytest.raises(ValidationError):
            MCQ(
                question="What is 1/2?",
                options=["A", "B", "C", "D"],
                correct_index=5,  # Out of bounds
            )

    def test_rejects_negative_correct_index(self):
        """correct_index must not be negative."""
        with pytest.raises(ValidationError):
            MCQ(
                question="What is 1/2?",
                options=["A", "B", "C", "D"],
                correct_index=-1,
            )

    def test_accepts_valid_mcq_all_indexes(self):
        """All valid correct_index values (0-3) should be accepted."""
        for idx in range(4):
            mcq = MCQ(
                question="What fraction is one out of four equal parts?",
                options=["1/2", "1/3", "1/4", "1/5"],
                correct_index=idx,
            )
            assert mcq.correct_index == idx


# ---------------------------------------------------------------------------
# GeneratorOutput schema constraints
# ---------------------------------------------------------------------------

class TestGeneratorOutputSchema:
    def test_rejects_empty_mcq_list(self):
        """MCQ list must have at least 1 item."""
        with pytest.raises(ValidationError):
            GeneratorOutput(
                explanation=Explanation(
                    text="Some explanation text here.", grade=5
                ),
                mcqs=[],  # Empty — invalid
                teacher_notes=TeacherNotes(
                    learning_objective="Students will understand fractions.",
                    common_misconceptions=["Numerator and denominator confusion"],
                ),
            )

    def test_rejects_grade_out_of_range_high(self):
        """Grade must be 1–12; 15 is invalid."""
        with pytest.raises(ValidationError):
            Explanation(text="Some text here for students.", grade=15)

    def test_rejects_grade_out_of_range_low(self):
        """Grade must be 1–12; 0 is invalid."""
        with pytest.raises(ValidationError):
            Explanation(text="Some text here for students.", grade=0)

    def test_rejects_short_explanation_text(self):
        """Explanation text must be at least 10 characters."""
        with pytest.raises(ValidationError):
            Explanation(text="Too short", grade=5)  # 9 chars

    def test_rejects_empty_misconceptions(self):
        """common_misconceptions must have at least 1 item."""
        with pytest.raises(ValidationError):
            TeacherNotes(
                learning_objective="Students will understand fractions.",
                common_misconceptions=[],  # Empty — invalid
            )

    def test_accepts_valid_generator_output(self, valid_draft):
        """A fully valid GeneratorOutput must pass all constraints."""
        assert valid_draft.explanation.grade == 5
        assert len(valid_draft.mcqs) >= 1
        assert valid_draft.teacher_notes.learning_objective
        assert len(valid_draft.teacher_notes.common_misconceptions) >= 1


# ---------------------------------------------------------------------------
# ReviewerOutput: deterministic pass/fail enforcement
# ---------------------------------------------------------------------------

class TestReviewerOutputSchema:
    def test_pass_overridden_false_when_average_below_threshold(self):
        """
        Validator must set passed=False when average < 4.0,
        regardless of what the LLM returned.
        """
        review = ReviewerOutput.model_validate(
            {
                "scores": {
                    "age_appropriateness": 2,
                    "correctness": 3,
                    "clarity": 3,
                    "coverage": 3,
                },
                "pass": True,   # LLM incorrectly claims pass
                "feedback": [],
            }
        )
        assert review.passed is False  # Validator must override

    def test_pass_overridden_false_when_any_score_below_3(self):
        """
        Validator must set passed=False when min score < 3,
        even if average is >= 4.0.
        """
        review = ReviewerOutput.model_validate(
            {
                "scores": {
                    "age_appropriateness": 5,
                    "correctness": 5,
                    "clarity": 2,   # Below minimum
                    "coverage": 5,
                },
                "pass": True,   # LLM incorrectly claims pass
                "feedback": [],
            }
        )
        assert review.passed is False

    def test_pass_overridden_true_when_scores_warrant(self):
        """
        Validator must set passed=True when scores meet threshold,
        even if the LLM returned pass=False.
        """
        review = ReviewerOutput.model_validate(
            {
                "scores": {
                    "age_appropriateness": 4,
                    "correctness": 5,
                    "clarity": 4,
                    "coverage": 4,
                },
                "pass": False,  # LLM incorrectly claims fail
                "feedback": [],
            }
        )
        assert review.passed is True  # Validator must override

    def test_pass_true_exactly_at_threshold(self):
        """Average exactly 4.0 with all scores >= 3 should pass."""
        review = ReviewerOutput.model_validate(
            {
                "scores": {
                    "age_appropriateness": 3,
                    "correctness": 5,
                    "clarity": 4,
                    "coverage": 4,
                },
                "pass": False,
                "feedback": [],
            }
        )
        assert review.scores.average == pytest.approx(4.0)
        assert review.scores.min_score == 3
        assert review.passed is True

    def test_pass_false_just_below_threshold(self):
        """Average just below 4.0 must fail."""
        review = ReviewerOutput.model_validate(
            {
                "scores": {
                    "age_appropriateness": 3,
                    "correctness": 4,
                    "clarity": 4,
                    "coverage": 4,
                },
                "pass": True,
                "feedback": [],
            }
        )
        # avg = (3+4+4+4)/4 = 3.75 < 4.0
        assert review.scores.average == pytest.approx(3.75)
        assert review.passed is False

    def test_scores_average_and_min_properties(self):
        scores = ReviewScores(
            age_appropriateness=3,
            correctness=5,
            clarity=4,
            coverage=4,
        )
        assert scores.average == pytest.approx(4.0)
        assert scores.min_score == 3


# ---------------------------------------------------------------------------
# Generator Agent: retry-on-failure behaviour
# ---------------------------------------------------------------------------

class TestGeneratorAgentRetry:
    def test_retries_once_then_raises_on_double_failure(
        self, mock_openai_client, sample_input
    ):
        """
        When both LLM attempts raise exceptions, run_generator must raise
        RuntimeError mentioning '2 attempts' after exactly 2 calls.
        """
        configure_mock_parse(
            mock_openai_client,
            ValueError("Schema mismatch on attempt 1"),
            ValueError("Schema mismatch on attempt 2"),
        )

        with pytest.raises(RuntimeError, match="2 attempts"):
            run_generator(mock_openai_client, sample_input)

        assert mock_openai_client.beta.chat.completions.parse.call_count == 2

    def test_succeeds_on_second_attempt_after_first_failure(
        self, mock_openai_client, sample_input, valid_draft
    ):
        """
        When the first attempt fails and the second succeeds,
        run_generator must return the valid draft after exactly 2 calls.
        """
        configure_mock_parse(
            mock_openai_client,
            ValueError("Validation error on attempt 1"),
            valid_draft,
        )

        result = run_generator(mock_openai_client, sample_input)

        assert result.explanation.grade == 5
        assert len(result.mcqs) >= 1
        assert mock_openai_client.beta.chat.completions.parse.call_count == 2

    def test_succeeds_on_first_attempt_no_retry(
        self, mock_openai_client, sample_input, valid_draft
    ):
        """When the first attempt succeeds, only 1 LLM call is made."""
        configure_mock_parse(mock_openai_client, valid_draft)

        result = run_generator(mock_openai_client, sample_input)

        assert result == valid_draft
        assert mock_openai_client.beta.chat.completions.parse.call_count == 1

    def test_wraps_all_failures_in_runtime_error(
        self, mock_openai_client, sample_input
    ):
        """Generator must always raise RuntimeError, never leak raw exceptions."""
        configure_mock_parse(
            mock_openai_client,
            ConnectionError("Network error attempt 1"),
            ConnectionError("Network error attempt 2"),
        )

        with pytest.raises(RuntimeError):
            run_generator(mock_openai_client, sample_input)
