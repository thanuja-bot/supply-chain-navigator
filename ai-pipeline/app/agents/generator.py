"""
Generator Agent
===============
Produces a structured educational draft from a grade + topic input.

Retry policy: attempt once; if validation fails, retry exactly once more,
then raise a descriptive RuntimeError so the orchestrator can fail gracefully.

Uses OpenAI structured outputs (client.beta.chat.completions.parse) which
enforces the JSON schema at the API level, giving us a validated Pydantic
object back directly.
"""

import logging

from openai import OpenAI

from app.schemas import ContentInput, GeneratorOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert educational content creator. You generate clear, accurate,
grade-appropriate learning materials in English.

Your output must be a complete educational artifact with:
1. An explanation written specifically for the given grade level.
2. Multiple-choice questions that test comprehension of the explanation.
3. Teacher notes with a clear learning objective and common misconceptions.

Rules:
- Language complexity must match the grade level (simple for grade 1-3, moderate for 4-6, advanced for 7-12).
- Every MCQ must have exactly 4 distinct options.
- correct_index must be the zero-based index of the correct option in the list.
- Do NOT include meta-commentary or apologies in your output.
"""


def run_generator(client: OpenAI, content_input: ContentInput) -> GeneratorOutput:
    """
    Run the Generator Agent with one automatic retry on failure.

    Args:
        client: Authenticated OpenAI client.
        content_input: Validated pipeline input (grade + topic).

    Returns:
        GeneratorOutput: Validated schema-conformant educational draft.

    Raises:
        RuntimeError: If both attempts fail.
    """
    user_prompt = (
        f"Create educational content for Grade {content_input.grade} "
        f"on the topic: '{content_input.topic}'.\n\n"
        f"Include:\n"
        f"- A grade-appropriate explanation (at least 3 sentences)\n"
        f"- At least 3 multiple-choice questions\n"
        f"- Teacher notes with learning objective and 2+ misconceptions"
    )

    last_error: Exception | None = None

    for attempt_num in range(1, 3):  # attempts 1 and 2
        logger.info("Generator: attempt %d for grade=%d topic='%s'",
                    attempt_num, content_input.grade, content_input.topic)
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=GeneratorOutput,
                temperature=0.7,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("OpenAI returned null parsed content")

            # Additional schema validation (belt-and-suspenders)
            validated = GeneratorOutput.model_validate(parsed.model_dump())
            logger.info("Generator: attempt %d succeeded", attempt_num)
            return validated

        except Exception as exc:
            last_error = exc
            logger.warning(
                "Generator: attempt %d failed — %s: %s",
                attempt_num, type(exc).__name__, exc
            )

    raise RuntimeError(
        f"Generator Agent failed after 2 attempts. Last error: {last_error}"
    )
