# TrustLens

TrustLens is a data science-first credibility analysis system for claims and headlines.
It scores credibility from structured evidence and measurable signals, not LLM judgment.

Given a claim, TrustLens is built to:
- retrieve corroborating news evidence at scale
- attach domain reliability context
- compute auditable features
- produce calibrated credibility scores
- generate grounded explanations and constrained Q&A

The project emphasizes SQL-based feature engineering, reproducibility, and inspectability.

## Project Goal

Build an industry-grade credibility pipeline that is:
- evidence-grounded
- calibrated and measurable
- reproducible and testable
- deployable as a real product surface

## Pipeline at a Glance

1. Run created for a claim/query
2. Evidence fetched and normalized
3. Features extracted from evidence + priors
4. Score + label computed with model versioning/calibration
5. Explanation/chat generated from stored context
6. Evaluation and benchmarking tracked with artifacts

## Development Phases (with Slice Mapping)

### Phase 1: Foundation and Data Backbone (Slices 0-2)

Delivered:
- production Python package + FastAPI health surface
- persistent relational layer (DuckDB/SQLAlchemy) with repositories
- CLI initialization flow (`trustlens init-db`)
- domain reliability prior ingestion (`trustlens load-priors`)

Why it matters:
- establishes stable infrastructure, persistence, and first quantitative credibility prior

### Phase 2: Evidence and Feature Intelligence (Slices 3-4)

Delivered:
- deterministic GDELT evidence ingestion with URL-level dedupe
- temporal semantics (`published_at`, `retrieved_at`, `created_at`)
- feature extraction across volume, source quality, temporal, corroboration
- idempotent feature persistence via CLI (`trustlens extract-features`)

Why it matters:
- turns external news into auditable, queryable evidence and reusable feature vectors

### Phase 3: Scoring, Calibration, and Evaluation (Slices 5-6)

Delivered:
- baseline scoring model (`baseline_v1`) with transparent weighted signals
- deterministic calibration transform and 3-way labels
- persisted scores + top positive/negative feature contributions
- evaluation harness (`trustlens evaluate`) with accuracy/precision/recall/F1, confusion matrix, Brier score, AUROC (when valid), and calibration bins

Why it matters:
- converts features into measurable, comparable, and explainable decisions

### Phase 4: Explainability, Trainable Models, and Benchmarking (Slices 7-12)

Delivered:
- grounded LLM explanation/chat layer constrained to stored run context
- offline trainable logistic models with versioning + threshold tuning (`lr_v1`, `lr_v2`)
- calibration enhancements (including Platt scaling for trained models)
- benchmark runner for LIAR/FEVER + ablations + reproducible reports (`trustlens benchmark`)
- local reproducible eval dataset, stratified analysis, and error artifacts

Observed benchmark note:
- On a small LIAR real-evidence run (50 samples), `baseline_v1` remained strongest on calibration quality; LR variants need more signal/data in that setup.

Why it matters:
- adds a full learning and diagnostics loop without sacrificing auditability

### Phase 5: Product API, UI, and Deployment (Slices 13-14)

Delivered:
- end-to-end product API for run orchestration and artifact retrieval
- minimal React/Vite demo UI for claim submission, evidence, score, explanation, chat
- `/models` discovery endpoint + CORS for local frontend integration
- single-service deployment pattern (FastAPI serves built UI)
- Docker packaging + AWS App Runner/RDS guidance
- free deployment path (Cloudflare Pages + Render)

Why it matters:
- moves TrustLens from pipeline code to an accessible, deployable product

## Core CLI Flow

```bash
trustlens init-db
trustlens load-priors

trustlens fetch-evidence --claim "COVID-19 vaccines are effective" \
  --query "COVID vaccine effectiveness" --max-records 25

trustlens extract-features --run-id <run_id>
trustlens score-run --run-id <run_id> --model baseline_v1
```

## Evaluation and Benchmarking

```bash
# Labeled evaluation harness
trustlens evaluate --dataset data/eval/sample_claims.csv --dataset-name sample_v1 --max-records 25

# Real-world benchmark + ablations
trustlens benchmark --dataset liar --split test --max-examples 500 \
  --model baseline_v1 --model lr_v1 --model lr_v2 --max-records 25
```

## API Examples

```bash
curl -X POST http://localhost:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"claim_text":"Example claim","query_text":"Example query","max_records":10,"model_id":"baseline_v1","include_explanation":true}'

curl http://localhost:8000/runs/<run_id>/score

curl -X POST http://localhost:8000/runs/<run_id>/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Why was this labeled credible?"}'
```

## Local Development

Backend:
```bash
uvicorn trustlens.api.main:app --reload
```

Frontend:
```bash
cd web
npm install
npm run dev
```

Environment variables:
- Backend: `ALLOWED_ORIGINS=http://localhost:5173`
- Frontend: `VITE_API_BASE_URL=http://localhost:8000`
- DB: `DATABASE_URL` (Postgres supported; DuckDB fallback)

## Deployment Options

Single-service (API + UI) Docker deploy:
```bash
docker build -t trustlens .
docker run -p 8000:8000 -e DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DB trustlens
```

Render start command:
```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Operational safety env vars:
- `EVIDENCE_TIMEOUT_S` (default 10)
- `MAX_RECORDS_CAP` (default 50)
- `RATE_LIMIT_PER_MIN` (default 30)

Deployment docs:
- `docs/deploy/aws_app_runner.md`
