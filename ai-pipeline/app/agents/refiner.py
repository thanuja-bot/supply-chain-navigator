"""
Refiner Agent
=============
Improves a failed draft using the Reviewer's structured field-level feedback.

Rules:
  - Maximum 2 refinement attempts (tracked by the orchestrator).
  - Each attempt is logged with the feedback it addressed.
  - The Refiner receives the original draft AND the reviewer feedback so it
    can make targeted improvements rather than a full rewrite.
"""

import json
import logging

from openai import OpenAI

from app.schemas import ContentInput, GeneratorOutput, ReviewerOutput, RefinerOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert educational content editor. You receive a draft that failed
quality review and a structured list of specific issues found by the reviewer.

Your task:
  1. Address EVERY issue listed in the feedback.
  2. Preserve content that scored well — do not rewrite everything.
  3. Target improvements to the exact fields named in the feedback.
  4. Maintain or improve grade-appropriateness of language.
  5. Ensure all MCQs have exactly 4 options and a valid correct_index (0-3).

Produce a revised, complete educational artifact.
"""


def run_refiner(
    client: OpenAI,
    content_input: ContentInput,
    failed_draft: GeneratorOutput,
    review: ReviewerOutput,
    attempt_number: int,
) -> GeneratorOutput:
    """
    Run the Refiner Agent to improve a failed draft.

    Args:
        client: Authenticated OpenAI client.
        content_input: Original pipeline input.
        failed_draft: The draft that failed review.
        review: The Reviewer's structured feedback.
        attempt_number: Which refinement attempt this is (1 or 2), for logging.

    Returns:
        GeneratorOutput: Improved draft (validated against strict schema).

    Raises:
        RuntimeError: If refinement or validation fails.
    """
    draft_json = json.dumps(failed_draft.model_dump(), indent=2)
    feedback_json = json.dumps(
        [fb.model_dump() for fb in review.feedback], indent=2
    )
    scores_json = json.dumps(review.scores.model_dump(), indent=2)

    user_prompt = (
        f"Grade: {content_input.grade}\n"
        f"Topic: '{content_input.topic}'\n\n"
        f"Original draft:\n{draft_json}\n\n"
        f"Reviewer scores:\n{scores_json}\n\n"
        f"Issues to fix (all must be addressed):\n{feedback_json}\n\n"
        f"Please produce an improved version of the draft that addresses all issues."
    )

    logger.info(
        "Refiner: attempt %d for grade=%d topic='%s' (addressing %d issues)",
        attempt_number,
        content_input.grade,
        content_input.topic,
        len(review.feedback),
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=RefinerOutput,
            temperature=0.6,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned null parsed content")

        # RefinerOutput and GeneratorOutput share the same field structure
        refined_data = parsed.model_dump()
        validated = GeneratorOutput.model_validate(refined_data)

        logger.info("Refiner: attempt %d succeeded", attempt_number)
        return validated

    except Exception as exc:
        logger.error("Refiner Agent attempt %d failed: %s", attempt_number, exc)
        raise RuntimeError(
            f"Refiner Agent attempt {attempt_number} failed: {exc}"
        ) from exc
