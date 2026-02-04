"""Integration tests for evaluation harness."""

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustlens.db.schema import Base, SourcePrior
from trustlens.services.evaluation import run_evaluation


@pytest.fixture
def in_memory_db():
    engine = create_engine("duckdb:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_run_evaluation_with_mocked_evidence(tmp_path: Path, in_memory_db):
    dataset_path = tmp_path / "tiny_eval.csv"
    dataset_path.write_text(
        "claim_id,claim_text,query_text,label\n"
        "1,Claim A,query a,1\n"
        "2,Claim B,query b,0\n",
        encoding="utf-8",
    )

    # insert priors so features are stable
    priors = [
        SourcePrior(domain="example.com", reliability_label=1, prior_score=0.9),
        SourcePrior(domain="example.org", reliability_label=0, prior_score=0.6),
    ]
    in_memory_db.add_all(priors)
    in_memory_db.commit()

    def fetcher(query_text: str, max_records: int):
        return [
            {
                "url": f"https://example.com/{query_text}",
                "domain": "example.com",
                "title": "A",
                "seendate": "20250101000000",
                "raw": {},
            },
            {
                "url": f"https://example.org/{query_text}",
                "domain": "example.org",
                "title": "B",
                "seendate": "20250102000000",
                "raw": {},
            },
        ]

    fixed_now = datetime(2025, 1, 1, 0, 0, 0)

    result = run_evaluation(
        session=in_memory_db,
        dataset_path=dataset_path,
        dataset_name="tiny",
        max_records=2,
        evidence_fetcher=fetcher,
        now_fn=lambda: fixed_now,
    )

    assert result["dataset_name"] == "tiny"
    assert result["n"] == 2
    assert "metrics" in result
    assert "calibration_bins" in result
    assert result["metrics"]["n"] == 2
