"""CLI command to list trained models."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.repos.trained_model_repo import TrainedModelRepository

console = Console()


def list_models_cmd() -> None:
    """List trained models in the registry."""
    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        repo = TrainedModelRepository(session)
        models = repo.list_models()

    table = Table(title="Trained Models")
    table.add_column("model_id", style="cyan")
    table.add_column("dataset", style="green")
    table.add_column("created_at", style="green")
    table.add_column("t_lo", style="magenta")
    table.add_column("t_hi", style="magenta")

    for m in models:
        thresholds = json.loads(m.thresholds_json)
        table.add_row(
            m.model_id,
            m.dataset_name,
            m.created_at.isoformat() if m.created_at else "",
            f"{thresholds.get('t_lo', 0.0):.3f}",
            f"{thresholds.get('t_hi', 1.0):.3f}",
        )

    console.print(table)


if __name__ == "__main__":
    typer.run(list_models_cmd)
