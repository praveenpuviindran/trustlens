"""CLI command for evaluation harness."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import init_db
from trustlens.services.evaluation import run_evaluation
from trustlens.services.gdelt import fetch_gdelt_articles
from trustlens.services.scoring import MODEL_VERSION

console = Console()


def _default_fetcher(query_text: str, max_records: int) -> list[dict]:
    articles = fetch_gdelt_articles(query=query_text, max_records=max_records)
    return [
        {
            "url": a.url,
            "domain": a.domain,
            "title": a.title,
            "seendate": a.seendate,
            "raw": getattr(a, "raw", None),
        }
        for a in articles
    ]


def evaluate_cmd(
    dataset: Path = typer.Option(..., help="Path to evaluation CSV"),
    dataset_name: str = typer.Option(..., help="Dataset name"),
    max_records: int = typer.Option(25, help="Max evidence records per claim"),
) -> None:
    """
    Run evaluation over a labeled dataset and print metrics.
    """
    engine = build_engine()
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        result = run_evaluation(
            session=session,
            dataset_path=dataset,
            dataset_name=dataset_name,
            max_records=max_records,
            evidence_fetcher=_default_fetcher,
        )

    metrics = result["metrics"]
    bins = result["calibration_bins"]

    table = Table(title=f"Evaluation Metrics ({dataset_name})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    for key in ["accuracy", "precision", "recall", "f1", "brier", "auroc"]:
        table.add_row(key, f"{metrics.get(key)}")

    cm = Table(title="Confusion Matrix")
    cm.add_column("TP", style="magenta", justify="right")
    cm.add_column("FP", style="magenta", justify="right")
    cm.add_column("TN", style="magenta", justify="right")
    cm.add_column("FN", style="magenta", justify="right")
    cm.add_row(
        str(metrics.get("tp")),
        str(metrics.get("fp")),
        str(metrics.get("tn")),
        str(metrics.get("fn")),
    )

    calib = Table(title="Calibration (5 bins)")
    calib.add_column("Bin")
    calib.add_column("Count", justify="right")
    calib.add_column("Avg Pred", justify="right")
    calib.add_column("Avg Obs", justify="right")
    for b in bins:
        calib.add_row(
            f"{b['low']:.2f}-{b['high']:.2f}",
            str(b["count"]),
            f"{b['avg_pred']:.3f}",
            f"{b['avg_obs']:.3f}",
        )

    console.print(table)
    console.print(cm)
    console.print(calib)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"eval_{dataset_name}_{timestamp}.json"

    report = {
        "dataset_name": dataset_name,
        "n": result["n"],
        "metrics": metrics,
        "calibration_bins": bins,
        "timestamp": result["timestamp"],
        "model_version": MODEL_VERSION,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print(f"[green]✓[/green] Wrote report to {report_path}")


if __name__ == "__main__":
    typer.run(evaluate_cmd)
