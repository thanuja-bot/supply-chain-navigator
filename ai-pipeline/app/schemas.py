"""
Pydantic v2 schemas for all agent inputs, outputs, and the RunArtifact.

These are the strict contracts the entire pipeline enforces. Every agent
uses these models for validation. The OpenAI client's .parse() method
guarantees conformance via structured outputs (JSON schema enforcement).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

class ContentInput(BaseModel):
    """Request payload for the generation pipeline."""

    grade: int = Field(..., ge=1, le=12, description="Target school grade (1–12)")
    topic: str = Field(..., min_length=3, max_length=200, description="Educational topic")
    user_id: Optional[str] = Field(default=None, description="Optional caller identifier")


# ---------------------------------------------------------------------------
# Generator Agent schemas
# ---------------------------------------------------------------------------

class Explanation(BaseModel):
    text: str = Field(..., min_length=10)
    grade: int = Field(..., ge=1, le=12)


class MCQ(BaseModel):
    question: str = Field(..., min_length=5)
    options: List[str] = Field(..., min_length=4, max_length=4)
    correct_index: int = Field(..., ge=0, le=3)

    @field_validator("options")
    @classmethod
    def options_must_have_four(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("MCQ must have exactly 4 options")
        return v


class TeacherNotes(BaseModel):
    learning_objective: str = Field(..., min_length=10)
    common_misconceptions: List[str] = Field(..., min_length=1)


class GeneratorOutput(BaseModel):
    """Strict output schema for the Generator Agent."""

    explanation: Explanation
    mcqs: List[MCQ] = Field(..., min_length=1)
    teacher_notes: TeacherNotes


# ---------------------------------------------------------------------------
# Reviewer Agent schemas
# ---------------------------------------------------------------------------

class ReviewScores(BaseModel):
    age_appropriateness: int = Field(..., ge=1, le=5)
    correctness: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    coverage: int = Field(..., ge=1, le=5)

    @property
    def average(self) -> float:
        return (
            self.age_appropriateness
            + self.correctness
            + self.clarity
            + self.coverage
        ) / 4

    @property
    def min_score(self) -> int:
        return min(
            self.age_appropriateness,
            self.correctness,
            self.clarity,
            self.coverage,
        )


class ReviewFeedbackItem(BaseModel):
    field: str = Field(
        ...,
        description="Dot-path to the field with the issue, e.g. 'explanation.text'",
    )
    issue: str = Field(..., min_length=5)


class ReviewerOutput(BaseModel):
    """
    Strict output schema for the Reviewer Agent.

    Pass criteria (documented and deterministically enforced by model_validator):
      - Average score across all four dimensions >= 4.0
      - No individual score below 3

    The model_validator is the single source of truth for pass/fail —
    it always overwrites whatever the LLM returns to guarantee consistency.
    """

    scores: ReviewScores
    # 'pass' is a Python keyword — expose it via alias
    passed: bool = Field(..., alias="pass")
    feedback: List[ReviewFeedbackItem] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def enforce_pass_threshold(self) -> "ReviewerOutput":
        """
        Deterministically set 'passed' based on scores, ignoring whatever
        the LLM returned.  This prevents hallucinated pass/fail values.

        Pass requires ALL of:
          - average score >= 4.0
          - no individual score below 3
        """
        self.passed = (
            self.scores.average >= 4.0 and self.scores.min_score >= 3
        )
        return self


# ---------------------------------------------------------------------------
# Refiner Agent schemas
# ---------------------------------------------------------------------------

class RefinerOutput(BaseModel):
    """The Refiner produces an improved GeneratorOutput (identical field structure)."""

    explanation: Explanation
    mcqs: List[MCQ] = Field(..., min_length=1)
    teacher_notes: TeacherNotes


# ---------------------------------------------------------------------------
# Tagger Agent schemas
# ---------------------------------------------------------------------------

class BloomsLevel(str, Enum):
    remembering = "Remembering"
    understanding = "Understanding"
    applying = "Applying"
    analyzing = "Analyzing"
    evaluating = "Evaluating"
    creating = "Creating"


class DifficultyLevel(str, Enum):
    easy = "Easy"
    medium = "Medium"
    hard = "Hard"


class TaggerOutput(BaseModel):
    """Strict output schema for the Tagger Agent. Runs only on approved content."""

    subject: str = Field(..., min_length=2)
    topic: str = Field(..., min_length=2)
    grade: int = Field(..., ge=1, le=12)
    difficulty: DifficultyLevel
    content_type: List[str] = Field(..., min_length=1)
    blooms_level: BloomsLevel


# ---------------------------------------------------------------------------
# RunArtifact (orchestration audit trail)
# ---------------------------------------------------------------------------

class FinalStatus(str, Enum):
    approved = "approved"
    rejected = "rejected"


class AttemptRecord(BaseModel):
    attempt: int
    draft: Dict[str, Any]
    review: Dict[str, Any]
    refined: Optional[Dict[str, Any]] = None


class FinalRecord(BaseModel):
    status: FinalStatus
    content: Optional[Dict[str, Any]] = None   # Populated only when approved
    tags: Optional[Dict[str, Any]] = None      # Populated only when approved & tagged
    error: Optional[str] = None                # Populated on pipeline-level failures


class Timestamps(BaseModel):
    started_at: datetime
    finished_at: Optional[datetime] = None


class RunArtifact(BaseModel):
    """Complete audit trail for a single pipeline run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: Dict[str, Any]
    attempts: List[AttemptRecord] = Field(default_factory=list)
    final: Optional[FinalRecord] = None
    timestamps: Timestamps

    # Pydantic v2 serializes datetime to ISO-8601 by default in model_dump_json()
    model_config = ConfigDict()
