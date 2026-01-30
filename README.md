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

## Slice 1 — Database Foundation & CLI Initialization (Complete)

**Purpose**  
Introduce a persistent, inspectable data layer and formalize system initialization before ingesting real evidence or running models.

**What this slice does**
- Establishes a relational database foundation using DuckDB + SQLAlchemy
- Defines core domain tables (runs, evidence, features, model versions)
- Adds an idempotent database initialization routine
- Introduces a first-class CLI for system operations
- Extends the `/health` endpoint to verify database connectivity

**What I implemented**
- SQLAlchemy engine and session management
- Declarative schema definitions for core entities
- Repository layer for structured database access
- `trustlens init-db` CLI command for safe schema initialization
- Database connectivity checks wired into the API health endpoint
- Updated unit tests to reflect DB-backed system health

**Why this matters**
This slice converts the project from a stateless API into a real, auditable system with:
- persistent state
- reproducible runs
- inspectable features and evidence
- explicit lifecycle control via a CLI

By introducing the database and repository pattern early, later slices can focus on **data logic, feature engineering, and modeling** without reworking infrastructure.

## Slice 2 — Domain Reliability Prior (Complete)

**Purpose**  
Introduce a quantitative, domain-level reliability prior so credibility scoring begins with a measurable baseline signal before corroboration or timing is considered.

**What this slice does**
- Loads the HuggingFace dataset `sergioburdisso/news_media_reliability`
- Normalizes news domains for consistent matching
- Stores domain reliability data in a first-class SQL table: `source_priors`
- Maps reliability labels to a numeric prior score in **[0, 1]**
- Exposes a CLI command to load or refresh priors: `trustlens load-priors`

**What I implemented**
- End-to-end priors ingestion pipeline (HF → cleaned rows → DB upsert)
- `source_priors` schema with:
  - domain
  - reliability_label (-1 / 0 / 1)
  - optional NewsGuard score
  - numeric prior_score
  - updated_at timestamp
- Unit tests validating:
  - domain normalization
  - label → score mapping
  - idempotent DB writes

**Why this matters**
This prior becomes the system’s first **stable credibility signal**:
- fully inspectable in SQL
- reproducible across runs
- easy to ablate during evaluation
- independent of LLM judgment

## Slice 3 — Evidence Ingestion Pipeline (Complete)

**Purpose**  
Introduce a deterministic, testable evidence ingestion pipeline that retrieves real-world news articles and stores them as structured, queryable evidence linked to each analysis run.

**What this slice does**
- Retrieves news articles from the GDELT DOC API
- Normalizes article metadata into a first-class `evidence_items` table
- Enforces URL-level uniqueness to prevent duplicate evidence
- Links each piece of evidence to a specific analysis run
- Separates publication time, retrieval time, and database creation time
- Ensures idempotent ingestion behavior across repeated runs

**What I implemented**
- GDELT client with dependency injection for testability
- Structured `EvidenceItem` schema with:
  - stable primary keys
  - explicit temporal semantics (`published_at`, `retrieved_at`, `created_at`)
  - enforced URL uniqueness
- Repository-layer upsert logic for evidence ingestion
- End-to-end pipeline wiring (`fetch → normalize → persist`)
- Full pipeline unit test using mocked GDELT HTTP responses

**Why this matters**
This slice converts external news data into **auditable, queryable evidence**:
- every article is traceable to a run
- duplicate ingestion is prevented by design
- temporal reasoning (when something was published vs. retrieved) is explicit
- downstream feature engineering can rely on stable evidence invariants

With this slice complete, the system now supports **real-world corroboration at scale** while remaining fully testable and reproducible.