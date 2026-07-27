"""
Reviewer Agent (Gatekeeper)
===========================
Quantitatively evaluates a GeneratorOutput draft and decides pass/fail.

Pass criteria (documented and enforced in ReviewerOutput schema):
  - Average score across age_appropriateness, correctness, clarity, coverage >= 4.0
  - No individual score below 3

Feedback must explicitly name the failing field using dot-path notation
(e.g. "explanation.text", "mcqs[0].question") so the Refiner can target
its improvements precisely.
"""

import json
import logging

from openai import OpenAI

from app.schemas import ContentInput, GeneratorOutput, ReviewerOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a strict educational content quality reviewer with expertise in
curriculum standards and age-appropriate pedagogy.

Your task is to evaluate a draft educational artifact against four dimensions:
  1. age_appropriateness (1-5): Is language/content suitable for the stated grade?
  2. correctness (1-5): Is the content factually accurate?
  3. clarity (1-5): Is the content clearly and unambiguously written?
  4. coverage (1-5): Does it adequately cover the topic given the grade level?

Scoring rubric (applies to each dimension):
  5 = Excellent, no improvements needed
  4 = Good, minor improvements possible
  3 = Acceptable, noticeable issues
  2 = Poor, significant issues
  1 = Unacceptable, major revision required

Pass criteria (you MUST be consistent with these):
  - "pass": true  ONLY IF average score >= 4.0 AND all individual scores >= 3
  - "pass": false otherwise

Feedback rules:
  - Every score of 3 or below MUST have at least one feedback entry.
  - Feedback "field" must be a dot-path to the specific field (e.g. "explanation.text",
    "mcqs[1].question", "teacher_notes.learning_objective").
  - "issue" must describe the specific problem, not a generic comment.
  - If "pass" is true you MAY include minor improvement notes but they are not required.
"""


def run_reviewer(
    client: OpenAI,
    content_input: ContentInput,
    draft: GeneratorOutput,
) -> ReviewerOutput:
    """
    Run the Reviewer Agent to score and gate a draft.

    Args:
        client: Authenticated OpenAI client.
        content_input: Original pipeline input (for context).
        draft: The GeneratorOutput to evaluate.

    Returns:
        ReviewerOutput with scores, pass/fail decision, and field-level feedback.

    Raises:
        RuntimeError: If the API call or schema validation fails.
    """
    draft_json = json.dumps(draft.model_dump(), indent=2)
    user_prompt = (
        f"Grade: {content_input.grade}\n"
        f"Topic: '{content_input.topic}'\n\n"
        f"Draft to review:\n{draft_json}\n\n"
        f"Evaluate the draft and return your structured review."
    )

    logger.info("Reviewer: evaluating draft for grade=%d topic='%s'",
                content_input.grade, content_input.topic)
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=ReviewerOutput,
            temperature=0.3,  # More deterministic for evaluation
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned null parsed content")

        # Validate and enforce pass consistency
        result = ReviewerOutput.model_validate(parsed.model_dump(by_alias=True))
        logger.info(
            "Reviewer: avg=%.2f min=%d pass=%s",
            result.scores.average,
            result.scores.min_score,
            result.passed,
        )
        return result

    except Exception as exc:
        logger.error("Reviewer Agent failed: %s", exc)
        raise RuntimeError(f"Reviewer Agent failed: {exc}") from exc
