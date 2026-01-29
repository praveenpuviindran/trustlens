# TrustLens

TrustLens is a data science–first system for **claim context and credibility scoring** using structured evidence and measurable signals — not LLM judgment.

Given a claim or headline, the system is designed to:
- retrieve corroborating news evidence at scale
- attach structured entity context
- score credibility using calibrated, inspectable features
- explain results grounded in computed signals

The project prioritizes **SQL-based feature engineering, evaluation, and reproducibility** over black-box models.

---

## Project Goal

Build an industry-grade credibility analysis pipeline that:
- uses real-world data sources (news + knowledge graphs)
- produces calibrated credibility scores
- exposes clear evidence and context for each result
- is reproducible, testable, and deployable

Development is organized into **vertical slices**.  
Each slice is runnable, tested, and committed before moving on.

---

## Slice 0 — Project Setup & Health Check (Complete)

**Purpose**  
Establish a correct, production-ready project foundation before adding data or models.

**What this slice does**
- Creates an installable Python package using a `src/` layout
- Sets up a minimal FastAPI application
- Exposes a `/health` endpoint to verify service wiring
- Adds a unit test to validate API imports and execution

**What I implemented**
- Proper Python packaging via `pyproject.toml`
- FastAPI app entry point
- Health endpoint (`GET /health`)
- Pytest-based test to confirm the API loads and responds

**Why this matters**
Correct packaging and testable entry points prevent import bugs, deployment issues, and brittle development later. This slice ensures the system can be extended safely.
