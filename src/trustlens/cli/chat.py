"""CLI command for grounded chat."""

from __future__ import annotations

import typer
from rich.console import Console

from sqlalchemy.orm import sessionmaker

from trustlens.db.engine import build_engine
from trustlens.db.init_db import ensure_db
from trustlens.services.llm_client import get_llm_client
from trustlens.services.llm_explainer import LLMExplainer

console = Console()


def chat_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Run ID"),
    model: str = typer.Option("baseline_v1", "--model", help="Model ID"),
    question: str | None = typer.Option(None, "--question", help="User question"),
) -> None:
    """Answer a question grounded in stored run data."""
    if not question:
        question = typer.prompt("Question")

    engine = build_engine()
    ensure_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        explainer = LLMExplainer(session, get_llm_client())
        result = explainer.chat(run_id, question, model_id=model)

    console.print(result.response_text)


if __name__ == "__main__":
    typer.run(chat_cmd)
