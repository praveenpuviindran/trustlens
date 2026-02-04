"""CLI command for LLM explanation."""

from __future__ import annotations

import typer
from rich.console import Console

from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.services.llm_client import get_llm_client
from trustlens.services.llm_explainer import LLMExplainer

console = Console()


def explain_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    model: str = typer.Option("baseline_v1", "--model", help="Model ID"),
) -> None:
    """Generate a grounded explanation for a run."""
    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        explainer = LLMExplainer(session, get_llm_client())
        result = explainer.explain_run(run_id, model_id=model)

    console.print(result.response_text)


if __name__ == "__main__":
    typer.run(explain_cmd)
