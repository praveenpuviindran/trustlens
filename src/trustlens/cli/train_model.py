"""CLI command for offline model training."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.services.model_training import ModelTrainer

console = Console()


def train_model_cmd(
    dataset: Path = typer.Option(..., help="Path to CSV dataset (run_id,label)"),
    model_id: str = typer.Option(..., help="Model ID (e.g., lr_v1)"),
    split_ratio: float = typer.Option(0.8, help="Train/val split ratio"),
    feature_schema_version: str = typer.Option(
        "v1",
        help="Feature schema version (v1 for base features, v2 for expanded features)",
    ),
    calibrate: bool = typer.Option(
        True,
        help="Apply Platt scaling on validation probabilities",
    ),
) -> None:
    """Train and register a logistic regression model from stored features."""
    dataset_name = dataset.stem
    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        trainer = ModelTrainer(session)
        model = trainer.train_and_register(
            dataset_path=dataset,
            dataset_name=dataset_name,
            model_id=model_id,
            split_ratio=split_ratio,
            feature_schema_version=feature_schema_version,
            calibrate=calibrate,
        )

    table = Table(title=f"Trained Model: {model.model_id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("dataset", model.dataset_name)
    table.add_row("schema", model.feature_schema_version)
    table.add_row("dataset_hash", model.dataset_hash[:12] + "...")
    console.print(table)


if __name__ == "__main__":
    typer.run(train_model_cmd)
