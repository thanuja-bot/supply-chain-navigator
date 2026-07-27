"""
Shared pytest fixtures for the AI Content Pipeline test suite.

LLM calls are always mocked — tests must never make real API calls.
The mock client fixture is configured per-test via configure_mock_parse().
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base  # Importing Base also registers RunRecord with metadata
from app.schemas import (
    BloomsLevel,
    ContentInput,
    DifficultyLevel,
    Explanation,
    GeneratorOutput,
    MCQ,
    ReviewerOutput,
    TaggerOutput,
    TeacherNotes,
)


# ---------------------------------------------------------------------------
# In-memory test database
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    """Provide a clean in-memory SQLite session for each test."""
    engine = create_engine(
        TEST_DATABASE_URL, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Reusable schema fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input() -> ContentInput:
    return ContentInput(
        grade=5, topic="Fractions as parts of a whole", user_id="test-user"
    )


@pytest.fixture
def valid_draft() -> GeneratorOutput:
    return GeneratorOutput(
        explanation=Explanation(
            text=(
                "A fraction represents a part of a whole. "
                "When we cut a pizza into 4 equal slices and take 1, "
                "we have 1/4 of the pizza."
            ),
            grade=5,
        ),
        mcqs=[
            MCQ(
                question="What does the numerator in a fraction represent?",
                options=[
                    "The total number of equal parts",
                    "The number of parts we have",
                    "The size of each part",
                    "The whole number",
                ],
                correct_index=1,
            ),
            MCQ(
                question="Which fraction represents half of a whole?",
                options=["1/4", "3/4", "1/2", "2/3"],
                correct_index=2,
            ),
            MCQ(
                question=(
                    "If a pizza is cut into 8 slices and you eat 3, "
                    "what fraction did you eat?"
                ),
                options=["3/5", "5/8", "3/8", "8/3"],
                correct_index=2,
            ),
        ],
        teacher_notes=TeacherNotes(
            learning_objective=(
                "Students will be able to identify and name "
                "fractions as parts of a whole."
            ),
            common_misconceptions=[
                "Students often confuse the numerator and denominator.",
                "Students may think larger denominators mean larger fractions.",
            ],
        ),
    )


@pytest.fixture
def passing_review() -> ReviewerOutput:
    """A ReviewerOutput whose scores deterministically produce passed=True."""
    return ReviewerOutput.model_validate(
        {
            "scores": {
                "age_appropriateness": 5,
                "correctness": 5,
                "clarity": 4,
                "coverage": 4,
            },
            "pass": True,
            "feedback": [],
        }
    )


@pytest.fixture
def failing_review() -> ReviewerOutput:
    """A ReviewerOutput whose scores deterministically produce passed=False."""
    return ReviewerOutput.model_validate(
        {
            "scores": {
                "age_appropriateness": 3,
                "correctness": 4,
                "clarity": 2,
                "coverage": 3,
            },
            "pass": False,
            "feedback": [
                {
                    "field": "explanation.text",
                    "issue": (
                        "Explanation uses vocabulary too advanced "
                        "for Grade 5 students."
                    ),
                },
                {
                    "field": "mcqs[0].question",
                    "issue": (
                        "Question phrasing is ambiguous and could "
                        "confuse students."
                    ),
                },
            ],
        }
    )


@pytest.fixture
def valid_tags() -> TaggerOutput:
    return TaggerOutput(
        subject="Mathematics",
        topic="Fractions",
        grade=5,
        difficulty=DifficultyLevel.medium,
        content_type=["Explanation", "Quiz", "Teacher Notes"],
        blooms_level=BloomsLevel.understanding,
    )


# ---------------------------------------------------------------------------
# Mock OpenAI client helpers
# ---------------------------------------------------------------------------

def _make_mock_completion(parsed_object: Any) -> MagicMock:
    """Wrap a Pydantic object in a mock OpenAI completion response."""
    mock_message = MagicMock()
    mock_message.parsed = parsed_object

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """A mock OpenAI client whose .beta.chat.completions.parse is configurable."""
    client = MagicMock()
    client.beta = MagicMock()
    client.beta.chat = MagicMock()
    client.beta.chat.completions = MagicMock()
    client.beta.chat.completions.parse = MagicMock()
    return client


def configure_mock_parse(mock_client: MagicMock, *side_effects: Any) -> None:
    """
    Configure mock_client.beta.chat.completions.parse to return successive
    objects or raise successive exceptions on each call.

    Each item in side_effects can be:
      - A Pydantic model instance → wrapped in a mock completion object
      - An Exception instance     → raised on that call
      - An Exception class        → instantiated and raised on that call
    """
    responses: list[Any] = []
    for item in side_effects:
        if isinstance(item, Exception) or (
            isinstance(item, type) and issubclass(item, Exception)
        ):
            responses.append(item)
        else:
            responses.append(_make_mock_completion(item))

    def _side_effect(*args, **kwargs):
        if not responses:
            raise StopIteration("No more mock responses configured")
        resp = responses.pop(0)
        if isinstance(resp, type) and issubclass(resp, Exception):
            raise resp("Mock exception")
        if isinstance(resp, Exception):
            raise resp
        return resp

    mock_client.beta.chat.completions.parse.side_effect = _side_effect
