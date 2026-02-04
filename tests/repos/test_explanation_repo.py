"""Tests for explanation repository."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base
from trustlens.repos.explanation_repo import ExplanationRepository


def test_upsert_and_list_by_run():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    repo = ExplanationRepository(session)
    repo.upsert_latest(
        run_id="run1",
        model_id="baseline_v1",
        mode="summary",
        user_question=None,
        response_text="ok",
        context_json="{}",
    )

    rows = repo.list_by_run("run1")
    assert len(rows) == 1
    assert rows[0].response_text == "ok"
