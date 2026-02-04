from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import Session, sessionmaker

from trustlens.clients.hf_reliability_dataset import load_reliability_rows
from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db
from trustlens.repos.source_priors_repo import SourcePriorsRepo
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.pipeline_evidence import fetch_and_store_evidence
from trustlens.services.priors import build_prior_records
from trustlens.services.scoring import MODEL_VERSION
from trustlens.cli.evaluate import evaluate_cmd
from trustlens.cli.explain import explain_cmd
from trustlens.cli.chat import chat_cmd
from trustlens.cli.train_model import train_model_cmd
from trustlens.cli.list_models import list_models_cmd
from trustlens.cli.benchmark import benchmark_cmd

app = typer.Typer(help="TrustLens CLI (init DB, pipelines, evaluation).")
console = Console()


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
    init_db(engine)

    rows = load_reliability_rows()
    priors = build_prior_records(rows)

    repo = SourcePriorsRepo(engine)
    n = repo.upsert_many(priors)

    typer.echo(f"✅ Loaded {n} domain priors into source_priors.")


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


@app.command("extract-features")
def extract_features_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
) -> None:
    """Extract features for a run."""
    console.print(f"[bold blue]Extracting features for run_id={run_id}[/bold blue]")

    engine = build_engine()
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    try:
        with SessionLocal() as session:  # type: Session
            service = FeatureEngineeringService(session)
            feature_count = service.compute_features(run_id)
            features = service.get_features(run_id)

        console.print(f"[green]✓[/green] Extracted {feature_count} features")

        grouped = {}
        for f in features:
            grouped.setdefault(f.feature_group, []).append(f)

        table = Table(title=f"Features for run_id={run_id}")
        table.add_column("Group", style="cyan")
        table.add_column("Feature", style="magenta")
        table.add_column("Value", style="green", justify="right")

        for group_name in sorted(grouped.keys()):
            for i, f in enumerate(sorted(grouped[group_name], key=lambda x: x.feature_name)):
                table.add_row(group_name if i == 0 else "", f.feature_name, f"{f.feature_value:.4f}")

        console.print(table)
    except ValueError as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command("score-run")
def score_run_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    model: str = typer.Option(MODEL_VERSION, "--model", help="Model version"),
) -> None:
    """Compute and persist credibility score for a run."""
    console.print(f"[bold blue]Scoring run_id={run_id} (model={model})[/bold blue]")

    engine = build_engine()
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    try:
        with SessionLocal() as session:  # type: Session
            service = FeatureEngineeringService(session)
            result = service.compute_score_for_run(run_id, model_version=model)

        console.print(f"[green]✓[/green] score={result.score:.4f} label={result.label}")

        table = Table(title="Top Contributions")
        table.add_column("Type", style="cyan")
        table.add_column("Feature", style="magenta")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Contribution", style="yellow", justify="right")

        for item in result.explanation.get("positive", []):
            table.add_row(
                "positive",
                item["feature_name"],
                f"{item['value']:.4f}",
                f"{item['contribution']:.4f}",
            )
        for item in result.explanation.get("negative", []):
            table.add_row(
                "negative",
                item["feature_name"],
                f"{item['value']:.4f}",
                f"{item['contribution']:.4f}",
            )

        console.print(table)
    except ValueError as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


app.command("evaluate")(evaluate_cmd)
app.command("explain-run")(explain_cmd)
app.command("chat-run")(chat_cmd)
app.command("train-model")(train_model_cmd)
app.command("list-models")(list_models_cmd)
app.command("benchmark")(benchmark_cmd)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
