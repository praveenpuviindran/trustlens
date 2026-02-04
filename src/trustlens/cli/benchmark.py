"""CLI command for offline benchmarking."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.api.deps import get_evidence_fetcher
from trustlens.services.benchmarking import BenchmarkConfig, run_benchmark

console = Console()


def benchmark_cmd(
    dataset: str = typer.Option(..., help="Dataset name: liar or fever"),
    split: str = typer.Option("test", help="Dataset split"),
    max_examples: int = typer.Option(500, help="Max examples"),
    seed: int = typer.Option(42, help="Random seed"),
    model: list[str] = typer.Option(..., "--model", help="Model IDs to benchmark"),
    max_records: int = typer.Option(25, help="Max evidence records per claim"),
    no_fetch_evidence: bool = typer.Option(False, help="Skip evidence fetch; use synthetic features"),
) -> None:
    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    config = BenchmarkConfig(
        dataset_name=dataset,
        dataset_split=split,
        max_examples=max_examples,
        seed=seed,
        model_ids=model,
        label_mapping_name=f"{dataset}_mapping",
        max_records=max_records,
        no_fetch_evidence=no_fetch_evidence,
    )

    with SessionLocal() as session:
        report = run_benchmark(
            session=session,
            config=config,
            evidence_fetcher=get_evidence_fetcher() if not no_fetch_evidence else None,
        )

    table = Table(title="Benchmark Summary")
    table.add_column("Model")
    table.add_column("Accuracy", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("ECE", justify="right")

    for model_id, metrics in report["metrics"].items():
        table.add_row(
            model_id,
            f"{metrics['accuracy']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['brier']:.3f}",
            f"{metrics.get('ece', 0.0):.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    typer.run(benchmark_cmd)
