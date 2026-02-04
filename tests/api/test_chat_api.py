"""API tests for chat endpoint."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.api.main import app
from trustlens.api.deps import get_db, get_evidence_fetcher, get_llm
from trustlens.db.schema import Base, EvidenceItem, Feature, Run, Score
from trustlens.services.llm_client import StubLLMClient


def test_chat_endpoint(tmp_path):
    db_path = tmp_path / "chat_api.duckdb"
    engine = create_engine(f"duckdb:///{db_path}")
    Base.metadata.create_all(engine)
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
            title="A",
            source="test",
        )
    )
    session.add(
        Feature(
            run_id=run.run_id,
            feature_group="g",
            feature_name="total_articles",
            feature_value=1.0,
        )
    )
    session.add(
        Score(
            score_id=1,
            run_id=run.run_id,
            model_version="baseline_v1",
            score=0.5,
            label="uncertain",
            explanation_json="{}",
        )
    )
    session.commit()

    def _db_override():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[get_llm] = lambda: StubLLMClient("CHAT")
    app.dependency_overrides[get_evidence_fetcher] = lambda: lambda q, n: []

    client = TestClient(app)
    res = client.post(f"/runs/{run.run_id}/chat", json={"question": "What is the claim?"})
    assert res.status_code == 200
    assert res.json()["answer"] == "CHAT"

    app.dependency_overrides.clear()
