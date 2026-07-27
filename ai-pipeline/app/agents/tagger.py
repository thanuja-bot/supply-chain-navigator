"""
Tagger Agent
============
Classifies APPROVED content only. Assigns subject, difficulty, Bloom's level,
and content type tags for downstream search and filtering.

This agent runs ONLY after the Reviewer has passed the content. It must never
be called on rejected or unreviewed content.
"""

import json
import logging

from openai import OpenAI

from app.schemas import ContentInput, GeneratorOutput, TaggerOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert educational metadata specialist. You classify approved
educational content according to a standard taxonomy.

Classification dimensions:
  - subject: The broad academic subject (e.g. "Mathematics", "Science", "English Language Arts")
  - topic: The specific topic as stated (keep it concise, 1-5 words)
  - grade: The target grade as an integer
  - difficulty: "Easy" | "Medium" | "Hard" (relative to the grade level)
  - content_type: List from ["Explanation", "Quiz", "Teacher Notes", "Worksheet"]
    — include all types present in the content
  - blooms_level: One of Bloom's Taxonomy levels:
    "Remembering" | "Understanding" | "Applying" | "Analyzing" | "Evaluating" | "Creating"
    — choose the PRIMARY cognitive level targeted by the MCQs

Be precise. Use the subject taxonomy (e.g. "Mathematics" not "Math").
"""


def run_tagger(
    client: OpenAI,
    content_input: ContentInput,
    approved_content: GeneratorOutput,
) -> TaggerOutput:
    """
    Run the Tagger Agent on approved content.

    Args:
        client: Authenticated OpenAI client.
        content_input: Original pipeline input for context.
        approved_content: The approved GeneratorOutput draft.

    Returns:
        TaggerOutput: Validated classification metadata.

    Raises:
        RuntimeError: If the API call or schema validation fails.
    """
    content_json = json.dumps(approved_content.model_dump(), indent=2)
    user_prompt = (
        f"Grade: {content_input.grade}\n"
        f"Topic: '{content_input.topic}'\n\n"
        f"Approved content:\n{content_json}\n\n"
        f"Please classify this content with appropriate metadata tags."
    )

    logger.info(
        "Tagger: classifying approved content for grade=%d topic='%s'",
        content_input.grade,
        content_input.topic,
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TaggerOutput,
            temperature=0.2,  # Highly deterministic — classification task
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI returned null parsed content")

        validated = TaggerOutput.model_validate(parsed.model_dump())
        logger.info(
            "Tagger: classified as subject='%s' difficulty='%s' blooms='%s'",
            validated.subject,
            validated.difficulty,
            validated.blooms_level,
        )
        return validated

    except Exception as exc:
        logger.error("Tagger Agent failed: %s", exc)
        raise RuntimeError(f"Tagger Agent failed: {exc}") from exc
