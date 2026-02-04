"""Tests for LLM explainer service."""

import json
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, EvidenceItem, Feature, Run, Score
from trustlens.services.llm_client import StubLLMClient
from trustlens.services.llm_explainer import LLMExplainer


def test_build_context_and_explain_chat():
    engine = create_engine("duckdb:///:memory:")
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
            created_at=datetime.utcnow(),
        )
    )
    session.commit()

    explainer = LLMExplainer(session, StubLLMClient("STUB"))
    context = explainer.build_context(run.run_id)
    assert context["run"]["claim_text"] == "Test claim"
    assert context["score"]["label"] == "credible"
    assert len(context["features"]) == 1
    assert len(context["evidence"]) == 1

    summary = explainer.explain_run(run.run_id)
    assert summary.response_text == "STUB"

    chat = explainer.chat(run.run_id, "What is the claim?")
    assert chat.response_text == "STUB"
