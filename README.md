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

---

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

---

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

---

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

---

## Slice 4 — Feature Extraction (Complete)

**Purpose**  
Transform raw evidence into interpretable, reproducible features that bridge evidence ingestion and credibility scoring.

This slice formalizes the system’s **feature layer**, ensuring all downstream scoring is grounded in auditable, SQL-derived signals rather than opaque heuristics.

---

### What this slice does

- Computes structured features over `evidence_items` and `source_priors`
- Organizes features into semantic groups:
  - volume
  - source_quality
  - temporal
  - corroboration
- Persists features as first-class rows in the `features` table
- Exposes a CLI command to extract features for a completed run
- Ensures **idempotent, deterministic feature computation**
- Handles real-world edge cases:
  - missing priors
  - missing timestamps
  - repeated domains
  - empty evidence sets

---

### Feature Groups

#### 1. Volume Features
- `total_articles`  
  Total number of evidence items retrieved for the run.
- `unique_domains`  
  Number of distinct source domains represented in the evidence.

---

#### 2. Source Quality Features
- `weighted_prior_mean`  
  Mean reliability prior across evidence items.  
  Domains without priors default to **0.5 (unknown)**.
- `high_reliability_ratio`  
  Fraction of **unique domains** whose reliability prior exceeds the high-reliability threshold (≥ 0.7).  
  This avoids overweighting a single prolific outlet.
- `unknown_source_ratio`  
  Fraction of evidence items whose domains have no known prior.

---

#### 3. Temporal Features
- `recency_score`  
  Mean exponential decay score based on publication time, with neutral defaults for missing timestamps.
- `publication_span_hours`  
  Time span between the earliest and latest published articles.
- `missing_timestamp_ratio`  
  Fraction of evidence items lacking publication timestamps.

---

#### 4. Corroboration Features
- `domain_diversity`  
  Shannon entropy of the domain distribution (higher = broader corroboration).
- `max_domain_concentration`  
  Fraction of evidence from the most common domain (higher = less diverse).

---

### Architecture Components

- **`FeatureExtractor` (service)**  
  Pure feature computation logic with deterministic SQL + Python aggregation.
- **`FeatureRepository` (repo)**  
  Persistence and retrieval of feature rows.
- **Feature Engineering Orchestration**  
  Ensures idempotent feature extraction per run.
- **`trustlens extract-features` (CLI)**  
  User-facing command to compute and persist features for a run.

---

### Why this matters

This slice establishes the system’s **feature contract**:

- Every feature is:
  - traceable to data
  - reproducible across runs
  - inspectable via SQL
- Feature semantics are explicit and test-backed
- Statistical shortcuts (e.g., join duplication) are explicitly avoided
- The system is now ready for:
  - scoring
  - calibration
  - ablation
  - evaluation

---

## Slice 5 — Credibility Scoring + Calibration + Explanation (Complete)

**Purpose**  
Convert extracted features into a **calibrated credibility score** with transparent, inspectable explanations.

### What this slice does
- Computes a per-run credibility score in **[0, 1]** using a deterministic, fully inspectable baseline model
- Applies a calibration transform so scores stay stable and comparable across runs
- Produces a simple decision label: `credible`, `uncertain`, or `not_credible`
- Persists results into a first-class `scores` table (one score per run + model version)
- Produces feature-level explanations by surfacing the top positive and negative contributors
- Adds a CLI command to score a run end-to-end after features exist

### Scoring model (baseline_v1)
- Model id: `baseline_v1`
- The raw score is computed as a transparent weighted combination of core signals.

**Positive evidence signals**
- `weighted_prior_mean`
- `domain_diversity`
- `recency_score`
- `unique_domains` (log1p)

**Negative risk signals**
- `max_domain_concentration`
- `missing_timestamp_ratio`
- `unknown_source_ratio`

**Neutral / stabilizer**
- `total_articles` (log1p)

### Calibration
To avoid overly extreme scores from small evidence sets, the raw score is calibrated deterministically:

- `calibrated = clamp(0.05 + 0.90 * raw_prob, 0, 1)`

### Outputs
- `score`: calibrated score in **[0, 1]**
- `label`: `credible` / `uncertain` / `not_credible`
- `explanations`: top positive + negative feature contributions (fully traceable to stored features)

### CLI Usage (exact commands)
```bash
trustlens init-db
trustlens load-priors

trustlens fetch-evidence --claim "COVID-19 vaccines are effective" \
  --query "COVID vaccine effectiveness" --max-records 25

trustlens extract-features --run-id <run_id>

trustlens score-run --run-id <run_id> --model baseline_v1
```

---

## Slice 6 — Evaluation Harness + Metrics (Complete)

**Purpose**  
Add a deterministic evaluation harness that runs the full pipeline on labeled claims and reports metrics + calibration behavior.

**What this slice does**
- Loads a local labeled dataset of claims (CSV)
- Runs the end-to-end pipeline per claim:
  - create run
  - fetch evidence
  - extract features
  - score + explain
- Persists evaluation results to `eval_results`
- Computes metrics:
  - accuracy, precision, recall, f1
  - confusion matrix (tp, fp, tn, fn)
  - Brier score
  - AUROC (when class variety exists)
  - 5-bin calibration summary

**What I implemented**
- `EvalResult` schema + repository
- Evaluation service with pure metric computation
- CLI: `trustlens evaluate` with Rich output + JSON report
- Sample dataset at `data/eval/sample_claims.csv`
- Deterministic integration tests with mocked evidence

**Label Mapping Rule**
For metrics, predicted labels are mapped as:
- `credible` → 1
- `not_credible` → 0
- `uncertain` → 0

**CLI Usage (exact commands)**
```bash
trustlens evaluate --dataset data/eval/sample_claims.csv --dataset-name sample_v1 --max-records 25
```

**Why this matters**
This slice makes scoring measurable:
- you can quantify performance over labeled claims
- calibration quality becomes visible
- results are reproducible and audit-friendly

---

## Slice 7 — LLM Explanation + Grounded Chat (Complete)

**Purpose**  
Add a grounded explanation layer that never changes scores and never invents facts.

**What this slice does**
- Builds a structured JSON context from stored run data
- Generates a grounded summary explanation
- Adds a chat-style Q&A interface constrained to stored context
- Persists explanation artifacts for auditability

**What I implemented**
- LLM client abstraction (`LLMClient`) with a deterministic stub
- `LLMExplainer` service for context building + explanation/chat
- `explanations` table + repository
- CLI commands: `trustlens explain-run` and `trustlens chat-run`
- Tests for context building, explainer behavior, repo, and CLI

**Why this matters**
This slice makes the system interpretable and safe:
- explanations are grounded in evidence, features, and scores
- no hallucinated claims
- chat answers remain auditable and deterministic

---

## Slice 8 — Offline Trainable Model + Versioning + Threshold Tuning (Complete)

**Purpose**  
Add a deterministic, offline training workflow that learns weights from stored features and registers versioned models.

**What this slice does**
- Trains a logistic regression model from feature vectors
- Tunes two thresholds for 3-way labels (credible / uncertain / not_credible)
- Stores model weights, thresholds, and validation metrics in DB
- Enables scoring with trained model IDs via `score-run`

**What I implemented**
- `trained_models` table + repository
- `FeatureVectorizer` for canonical feature ordering
- `ModelTrainer` with deterministic training and threshold tuning
- CLI: `trustlens train-model` and `trustlens list-models`
- Tests for vectorization, training determinism, thresholds, and scoring

**Why this matters**
This slice makes scoring **learnable and reproducible**:
- models are versioned and auditable
- thresholds are tuned for operational labels
- trained weights can be evaluated and compared offline

---

## Slice 9 — Product API + End-to-End Run Endpoint (Complete)

**Purpose**  
Expose TrustLens as a real API surface that runs the full pipeline end-to-end.

**What this slice does**
- Creates runs and orchestrates evidence → features → score
- Supports grounded explanations and chat over stored data
- Adds read endpoints to fetch stored artifacts by run_id

**What I implemented**
- FastAPI routes for `/runs`, `/runs/{id}` and artifact endpoints
- Pydantic request/response schemas with validation
- Dependency injection for DB session, evidence fetcher, and LLM client
- Deterministic API tests with mocked evidence + stub LLM

**Why this matters**
This slice turns the system into a product surface:
- end-to-end runs are auditable and reproducible
- outputs are structured and grounded
- clients can retrieve artifacts and chat safely

**API Examples**
```bash
curl -X POST http://localhost:8000/runs \\
  -H 'Content-Type: application/json' \\
  -d '{"claim_text":"Example claim","query_text":"Example query","max_records":10,"model_id":"baseline_v1","include_explanation":true}'

curl http://localhost:8000/runs/<run_id>/score

curl -X POST http://localhost:8000/runs/<run_id>/chat \\
  -H 'Content-Type: application/json' \\
  -d '{"question":"Why was this labeled credible?"}'
```

---

## Slice 10 — Real-World Benchmark + Ablations (Complete)

**Purpose**  
Benchmark TrustLens on real labeled claim datasets, compare models, and quantify ablations.

**What this slice does**
- Loads real datasets from Hugging Face (LIAR + FEVER)
- Maps labels to binary credible/not_credible deterministically
- Compares `baseline_v1` vs `lr_v1`
- Runs feature-group ablations and reports metric deltas
- Produces reproducible JSON/CSV reports + calibration plots

**What I implemented**
- Dataset loader with deterministic label mapping + hash
- Benchmark runner with offline mode and ablations
- CLI: `trustlens benchmark`
- Calibration plots saved to `reports/benchmarks/...`
- Offline tests with synthetic features (no network)

**CLI Examples**
```bash
# Benchmark baseline vs lr on LIAR
trustlens benchmark --dataset liar --split test --max-examples 500 --model baseline_v1 --model lr_v1 --max-records 25

# Generate reports (saved under reports/benchmarks/<dataset>/)
trustlens benchmark --dataset fever --split validation --max-examples 500 --model baseline_v1 --model lr_v1 --max-records 25

# Interpret ablations: compare metrics across feature groups in the JSON report
```

### Why this matters
This slice turns TrustLens into an actual scoring system with real-world, reproducible evaluation:
- consistent benchmarking across datasets and models
- ablation analysis to understand feature impact
- auditable reports for regression tracking
- the score is reproducible and testable (no LLM judgment)
- calibration makes results stable and comparable
- explanations make every score auditable and debuggable
- the pipeline is now evidence → features → score, ready for evaluation and model iteration

---

## Slice 11 — Feature Expansion + Calibration + Model Selection (Complete)

**Purpose**  
Improve signal quality with richer features, calibrate learned models, and select stronger model versions.

**What this slice does**
- Adds text similarity, entity overlap, and consistency signals from evidence text
- Extends source-quality aggregation (median/min/max priors)
- Adds Platt scaling calibration for trained models with ECE reporting
- Introduces `lr_v2` trained on expanded feature schema
- Benchmarks `baseline_v1` vs `lr_v1` vs `lr_v2` with delta comparisons

**What I implemented**
- New text feature helpers + extraction for similarity/overlap/contradiction
- Evidence schema support for snippets and grounded text processing
- Model training upgrades: schema versioning + calibration params stored in DB
- Scoring uses calibrated probabilities for trained models
- Tests for text features, calibration, and lr_v2 end-to-end training/scoring

**CLI Examples**
```bash
# Extract features including new text/consistency signals
trustlens extract-features --run-id <run_id>

# Train lr_v2 using expanded feature schema + calibration
trustlens train-model --dataset data/eval/sample_claims.csv --model-id lr_v2 --split-ratio 0.8 --feature-schema-version v2

# Benchmark baseline vs lr_v1 vs lr_v2 (with ablations + calibration plots)
trustlens benchmark --dataset liar --split test --max-examples 500 --model baseline_v1 --model lr_v1 --model lr_v2 --max-records 25
```

**Why this matters**
This slice makes model quality **measurably better and more reliable**:
- richer evidence-based features increase signal diversity
- calibrated probabilities are trustworthy for thresholds and ranking
- lr_v2 is a distinct, auditable step forward in model selection

---
