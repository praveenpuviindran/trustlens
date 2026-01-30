from __future__ import annotations

import typer

from trustlens.clients.hf_reliability_dataset import load_reliability_rows
from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db
from trustlens.repos.source_priors_repo import SourcePriorsRepo
from trustlens.services.priors import build_prior_records

app = typer.Typer(help="TrustLens CLI (init DB, pipelines, evaluation).")


@app.command("hello")
def hello() -> None:
    typer.echo("hello from trustlens")


@app.command("init-db")
def init_db_cmd() -> None:
    engine = build_engine()
    init_db(engine)
    typer.echo("✅ Database initialized and reachable.")


@app.command("load-priors")
def load_priors_cmd() -> None:
    """
    Load domain reliability labels from HF and store as priors in DB.
    """
    engine = build_engine()
    init_db(engine)  # safe + ensures source_priors exists

    rows = load_reliability_rows()
    priors = build_prior_records(rows)

    repo = SourcePriorsRepo(engine)
    n = repo.upsert_many(priors)

    typer.echo(f"✅ Loaded {n} domain priors into source_priors.")

from __future__ import annotations

import typer
from sqlalchemy.orm import Session

from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db
from trustlens.services.pipeline_evidence import fetch_and_store_evidence

app = typer.Typer(help="TrustLens CLI (init DB, pipelines, evaluation).")


@app.command()
def hello() -> None:
    typer.echo("hello")


@app.command("init-db")
def init_db_cmd() -> None:
    engine = build_engine()
    init_db(engine)
    typer.echo("✅ Database initialized and reachable.")


@app.command("fetch-evidence")
def fetch_evidence_cmd(
    claim: str = typer.Option(..., help="Claim/headline text."),
    query: str = typer.Option(..., help="GDELT query string."),
    max_records: int = typer.Option(50, help="Max records to request from GDELT."),
) -> None:
    """
    Slice 3: retrieve evidence from GDELT and persist it linked to a Run.
    """
    engine = build_engine()
    init_db(engine)

    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:  # type: Session
        res = fetch_and_store_evidence(
            session=session,
            claim_text=claim,
            query_text=query,
            max_records=max_records,
        )
        session.commit()

    typer.echo(f"✅ Created run_id={res.run_id}")
    typer.echo(f"✅ Inserted evidence_items={res.evidence_inserted}")
