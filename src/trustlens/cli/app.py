from __future__ import annotations

import typer

from trustlens.db.engine import build_engine, ping_db
from trustlens.db.init_db import init_db

app = typer.Typer(help="TrustLens CLI (init DB, pipelines, evaluation).")


@app.command("init-db")
def init_db_cmd() -> None:
    """
    Initialize database schema (idempotent).

    Usage:
      trustlens init-db
    """
    engine = build_engine()
    init_db(engine)
    result = ping_db(engine)
    if not result.ok:
        raise typer.Exit(code=1)
    typer.echo("✅ Database initialized and reachable.")


# Keep a simple sanity command around
@app.command()
def hello() -> None:
    typer.echo("TrustLens CLI is wired up.")
