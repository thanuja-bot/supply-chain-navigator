# Governed, Auditable AI Content Pipeline

A production-grade multi-agent system for generating, reviewing, refining, and tagging educational content — with **full audit trails**, **bounded retries**, and **strict schema validation** at every step.

---

## Architecture Overview

```
POST /generate
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                         │
│                                                         │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐           │
│  │Generator│───▶│ Reviewer │───▶│  Tagger │ (approved) │
│  │  Agent  │    │  Agent   │    │  Agent  │           │
│  └─────────┘    └──────────┘    └─────────┘           │
│        ▲              │                                 │
│        │         fail │                                 │
│        │              ▼                                 │
│        │        ┌─────────┐                             │
│        └────────│ Refiner │ (max 2 attempts)            │
│                 │  Agent  │                             │
│                 └─────────┘                             │
└─────────────────────────────────────────────────────────┘
      │
      ▼
  RunArtifact (persisted to DB)
```

---

## Agent Roles

### 1. Generator Agent
**Purpose:** Produce a structured educational draft from a grade + topic.

**Input:**
```json
{ "grade": 5, "topic": "Fractions as parts of a whole" }
```

**Output (strict Pydantic schema):**
```json
{
  "explanation": { "text": "...", "grade": 5 },
  "mcqs": [
    { "question": "...", "options": ["A", "B", "C", "D"], "correct_index": 1 }
  ],
  "teacher_notes": {
    "learning_objective": "...",
    "common_misconceptions": ["...", "..."]
  }
}
```

**Retry policy:** 1 automatic retry on validation failure, then fails gracefully.

---

### 2. Reviewer Agent (Gatekeeper)
**Purpose:** Quantitatively score the draft and decide pass/fail.

**Output:**
```json
{
  "scores": {
    "age_appropriateness": 4,
    "correctness": 5,
    "clarity": 4,
    "coverage": 4
  },
  "pass": true,
  "feedback": [
    { "field": "explanation.text", "issue": "Sentence too complex for Grade 5" }
  ]
}
```

**Pass/Fail Criteria (documented and enforced in schema):**
| Criterion | Threshold |
|-----------|-----------|
| Average score (all 4 dimensions) | **≥ 4.0** |
| Minimum individual score | **≥ 3** |

The `ReviewerOutput` model validator **always overwrites** the LLM's `pass` field with the computed result — in both directions. If scores meet both thresholds the validator forces `pass: true`; if either threshold fails it forces `pass: false`. This bidirectional override prevents the LLM from hallucinating either a pass or a fail.

Feedback must use **dot-path field notation** (e.g. `"mcqs[1].question"`) so the Refiner knows exactly what to fix.

---

### 3. Refiner Agent
**Purpose:** Improve a failing draft using the Reviewer's precise field-level feedback.

**Rules:**
- Maximum **2 refinement attempts** (enforced by orchestrator loop).
- Each attempt is logged with the feedback addressed.
- Targets only the flagged fields — preserves content that scored well.
- If still failing after 2 refinements → final status becomes `rejected`.

---

### 4. Tagger Agent
**Purpose:** Classify approved content only.

**Output:**
```json
{
  "subject": "Mathematics",
  "topic": "Fractions",
  "grade": 5,
  "difficulty": "Medium",
  "content_type": ["Explanation", "Quiz", "Teacher Notes"],
  "blooms_level": "Understanding"
}
```

**Hard rule:** The Tagger is **never called on rejected content**. This is enforced in the orchestrator — the Tagger call is gated behind `final_status == "approved"`.

---

## Orchestration Decisions

### Deterministic Flow
The pipeline always follows the same order: Generate → Review → (Refine → Review)* → Tag (if approved). There are no branches or conditional skips except the Tagger gate.

### Bounded Retries
| Agent | Max LLM calls |
|-------|--------------|
| Generator | 2 (1 + 1 retry) |
| Reviewer | MAX_REFINEMENTS + 1 = 3 |
| Refiner | MAX_REFINEMENTS = 2 |
| Tagger | 1 (approved only) |
| **Total** | **≤ 8** |

### Structured Outputs
All agents use `client.beta.chat.completions.parse()` with a Pydantic model as `response_format`. This enforces the JSON schema at the **OpenAI API level**, so malformed responses are caught before reaching application code.

### Pass Threshold Enforcement
The `ReviewerOutput` model validator runs a deterministic check: `passed = (scores.average >= 4.0 AND scores.min >= 3)`. The computed value is **always written back**, overriding whatever the LLM returned in either direction. This makes the quality gate tamper-proof — the LLM cannot hallucinate a pass on a low-scoring draft, nor a fail on a high-scoring one.

### Graceful Degradation
- If the Generator fails entirely, the run is rejected with an error record (no crash).
- If the Tagger fails on an approved run, the failure is logged but the run still returns `approved` (tags contain an error field). This is intentional — tagging is classification metadata, not content quality.

---

## Trade-offs

| Decision | Reason |
|----------|--------|
| SQLite default (not Postgres) | Zero-config for assessment. Switch via `DATABASE_URL`. |
| Synchronous FastAPI (not async) | OpenAI's Python client is sync; mixing async adds complexity without benefit at this scale. |
| Structured outputs over function calling | Fewer tokens, no function dispatch overhead, guaranteed schema. |
| `MAX_REFINEMENTS = 2` | Assessment requirement. Increasing this increases cost linearly. |
| Reviewer score override in validator | Prevents LLM hallucination from bypassing the quality gate. |
| Tagger failure is non-fatal | Tags are metadata. Failing to classify approved content is a logging concern, not a pipeline failure. |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Run the full pipeline. Returns `RunArtifact`. |
| `GET` | `/history` | List stored runs. Query: `user_id`, `status`, `limit`, `offset`. |
| `GET` | `/runs/{run_id}` | Retrieve a specific run by UUID. |
| `GET` | `/healthz` | Liveness check. |
| `GET` | `/docs` | Interactive Swagger UI. |
| `GET` | `/redoc` | ReDoc documentation. |

---

## RunArtifact Schema (Full Audit Trail)

```json
{
  "run_id": "uuid-string",
  "input": { "grade": 5, "topic": "Fractions as parts of a whole" },
  "attempts": [
    {
      "attempt": 1,
      "draft": { "explanation": {...}, "mcqs": [...], "teacher_notes": {...} },
      "review": { "scores": {...}, "pass": false, "feedback": [...] },
      "refined": { "explanation": {...}, "mcqs": [...], "teacher_notes": {...} }
    },
    {
      "attempt": 2,
      "draft": { ... },
      "review": { "scores": {...}, "pass": true, "feedback": [] },
      "refined": null
    }
  ],
  "final": {
    "status": "approved",
    "content": { "explanation": {...}, "mcqs": [...], "teacher_notes": {...} },
    "tags": {
      "subject": "Mathematics",
      "topic": "Fractions",
      "grade": 5,
      "difficulty": "Medium",
      "content_type": ["Explanation", "Quiz"],
      "blooms_level": "Understanding"
    },
    "error": null
  },
  "timestamps": {
    "started_at": "2024-01-15T10:30:00.000Z",
    "finished_at": "2024-01-15T10:30:45.123Z"
  }
}
```

---

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key (GPT-4o access required for structured outputs)

### Installation

```bash
git clone https://github.com/your-org/ai-content-pipeline
cd ai-content-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Running

```bash
python run.py
# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

Or with uvicorn directly:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": 5,
    "topic": "Fractions as parts of a whole",
    "user_id": "teacher-001"
  }'
```

### Retrieve History

```bash
# All runs
curl "http://localhost:8000/history"

# Filter by user
curl "http://localhost:8000/history?user_id=teacher-001"

# Filter by status
curl "http://localhost:8000/history?status=approved&limit=10"
```

---

## Testing

```bash
# Run all tests (no API key needed — LLM calls are mocked)
pytest tests/ -v

# Run a specific test class
pytest tests/test_schema_validation.py -v
pytest tests/test_orchestration.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=app --cov-report=term-missing
```

### Test Coverage

| Test | What it verifies |
|------|-----------------|
| `TestMCQSchema` | Schema rejects wrong option counts and invalid indexes |
| `TestGeneratorOutputSchema` | Schema rejects empty MCQ lists, out-of-range grades |
| `TestReviewerOutputSchema` | Pass field override when scores fail threshold |
| `TestGeneratorAgentRetry` | Retry-once then fail, succeed on second attempt |
| `TestOrchestrationFailRefinedPass` | Full fail→refine→pass flow produces approved artifact |
| `TestOrchestrationFailRefineReject` | Full fail→refine→fail→reject produces rejected artifact |
| `TestOrchestrationEdgeCases` | Generator failure, immediate pass, UUID uniqueness |

---

## Project Structure

```
ai-pipeline/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app, routes, CORS, lifespan
│   ├── database.py       # SQLAlchemy + SQLite setup
│   ├── schemas.py        # All Pydantic v2 schemas
│   ├── orchestrator.py   # Pipeline coordinator
│   └── agents/
│       ├── generator.py  # Generator Agent (with retry)
│       ├── reviewer.py   # Reviewer Agent (gatekeeper)
│       ├── refiner.py    # Refiner Agent (bounded improvement)
│       └── tagger.py     # Tagger Agent (approved content only)
├── tests/
│   ├── conftest.py       # Fixtures, mock client, DB setup
│   ├── test_schema_validation.py   # Test 1: schema + retry
│   └── test_orchestration.py       # Tests 2 & 3: full pipeline flows
├── requirements.txt
├── .env.example
├── run.py
└── README.md
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | OpenAI API key (GPT-4o access required) |
| `DATABASE_URL` | No | `sqlite:///./ai_pipeline.db` | Database connection string |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8000` | Server port |
| `LOG_LEVEL` | No | `info` | Logging level |

---

## GitHub Notes

- Do **not** commit `venv/`, `.env`, or `ai_pipeline.db` — all are in `.gitignore`.
- The SQLite database file is ephemeral. For production, set `DATABASE_URL` to a PostgreSQL connection string.
