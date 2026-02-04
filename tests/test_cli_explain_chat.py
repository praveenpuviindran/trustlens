"""CLI tests for explain/chat commands."""

import json
from pathlib import Path

from importlib import reload
from typer.testing import CliRunner
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

import trustlens.config.settings as settings_mod
from trustlens.db.init_db import init_db
from trustlens.db.schema import EvidenceItem, Feature, Run, Score


def _seed_run(db_url: str) -> str:
    engine = create_engine(db_url)
    init_db(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    run = Run(claim_text="Test claim", query_text="query", status="completed")
    session.add(run)
    session.flush()

    session.add(
        EvidenceItem(
            run_id=run.run_id,
            url="https://example.com/a",
            domain="example.com",
            title="Title A",
            source="test",
        )
    )
    session.add(
        Feature(
            run_id=run.run_id,
            feature_group="source_quality",
            feature_name="weighted_prior_mean",
            feature_value=0.8,
        )
    )
    session.add(
        Score(
            score_id=1,
            run_id=run.run_id,
            model_version="baseline_v1",
            score=0.7,
            label="credible",
            explanation_json=json.dumps({"positive": [], "negative": []}),
        )
    )
    session.commit()
    run_id = run.run_id
    session.close()
    return run_id


def test_cli_explain_and_chat(tmp_path, monkeypatch):
    db_path = tmp_path / "cli_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setenv("TRUSTLENS_DB_URL", db_url)
    monkeypatch.setenv("TRUSTLENS_LLM_PROVIDER", "stub")
    reload(settings_mod)
    import trustlens.db.engine as engine_mod
    reload(engine_mod)
    monkeypatch.setattr(
        engine_mod,
        "build_engine",
        lambda db_url_override=None: create_engine(db_url_override or db_url),
    )
    from trustlens.cli.app import app
    import trustlens.cli.explain as explain_mod
    import trustlens.cli.chat as chat_mod
    monkeypatch.setattr(
        explain_mod,
        "build_engine",
        lambda db_url_override=None: create_engine(db_url_override or db_url),
    )
    monkeypatch.setattr(
        chat_mod,
        "build_engine",
        lambda db_url_override=None: create_engine(db_url_override or db_url),
    )

    run_id = _seed_run(db_url)
    # sanity check that the CLI engine can see the run
    engine = explain_mod.build_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("select count(*) from runs")).fetchone()
        assert rows[0] == 1
    runner = CliRunner()

    # direct call sanity check
    explain_mod.explain_cmd(run_id=run_id, model="baseline_v1")

    res = runner.invoke(app, ["explain-run", "--run-id", run_id, "--model", "baseline_v1"])
    assert res.exit_code == 0
    assert "STUB_RESPONSE" in res.stdout

    res = runner.invoke(
        app,
        ["chat-run", "--run-id", run_id, "--model", "baseline_v1", "--question", "What is this?"],
    )
    assert res.exit_code == 0
    assert "STUB_RESPONSE" in res.stdout
