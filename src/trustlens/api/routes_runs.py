"""Run orchestration API routes."""

from __future__ import annotations

import json
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from trustlens.api.deps import get_db, get_evidence_fetcher, get_llm
from trustlens.api.schemas import (
    ChatRequest,
    ChatResponse,
    ContributionOut,
    EvidenceOut,
    ExplanationResponse,
    ExplanationSummary,
    FeatureOut,
    RunCreateRequest,
    RunCreateResponse,
    RunMetaResponse,
    ScoreResponse,
)
from trustlens.db.schema import EvidenceItem, Explanation, Feature, Run, Score
from trustlens.repos.evidence_repo import EvidenceRepo
from trustlens.repos.trained_model_repo import TrainedModelRepository
from trustlens.services.feature_engineering import FeatureEngineeringService
from trustlens.services.llm_explainer import LLMExplainer
from trustlens.services.pipeline_evidence import create_run

router = APIRouter(prefix="/runs", tags=["runs"])


def _top_contributions(expl: dict) -> list[ContributionOut]:
    items = []
    for group in ("positive", "negative"):
        for item in expl.get(group, []):
            items.append(
                ContributionOut(
                    feature_name=item.get("feature_name", ""),
                    value=float(item.get("value", 0.0)),
                    contribution=float(item.get("contribution", 0.0)),
                    weight=item.get("weight"),
                )
            )
    return items


@router.post("", response_model=RunCreateResponse)
def create_run_endpoint(
    payload: RunCreateRequest,
    session: Session = Depends(get_db),
    evidence_fetcher: Callable[[str, int], list[dict]] = Depends(get_evidence_fetcher),
    llm_client=Depends(get_llm),
):
    try:
        query_text = payload.query_text or payload.claim_text
        run_id = create_run(session, claim_text=payload.claim_text, query_text=query_text)

        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run:
            run.status = "created"
            session.commit()

        evidence_repo = EvidenceRepo(session.get_bind())
        articles = evidence_fetcher(query_text, payload.max_records)
        for a in articles:
            evidence_repo.upsert_from_gdelt(
                session=session,
                run_id=run_id,
                url=a["url"],
                domain=a.get("domain"),
                title=a.get("title"),
                snippet=a.get("snippet"),
                seendate=a.get("seendate"),
                raw=a.get("raw"),
            )
        session.commit()

        service = FeatureEngineeringService(session)
        service.compute_features(run_id)
        score_result = service.compute_score_for_run(run_id, model_version=payload.model_id)

        explanation_summary = None
        if payload.include_explanation:
            explainer = LLMExplainer(session, llm_client)
            expl = explainer.explain_run(run_id, model_id=payload.model_id)
            bullets = [
                f"{c.feature_name}: {c.contribution:.4f}" for c in _top_contributions(score_result.explanation)
            ]
            explanation_summary = ExplanationSummary(summary=expl.response_text, bullets=bullets)

        run = session.query(Run).filter(Run.run_id == run_id).first()
        if run:
            run.status = "completed"
            session.commit()

        features = service.get_features(run_id)
        evidence_count = (
            session.query(func.count(EvidenceItem.evidence_id))
            .filter(EvidenceItem.run_id == run_id)
            .scalar()
            or 0
        )

        return RunCreateResponse(
            run_id=run_id,
            status="completed",
            score=score_result.score,
            label=score_result.label,
            top_contributions=_top_contributions(score_result.explanation),
            evidence_count=int(evidence_count),
            features=[
                FeatureOut(
                    feature_group=f.feature_group,
                    feature_name=f.feature_name,
                    feature_value=float(f.feature_value),
                )
                for f in features
            ],
            explanation=explanation_summary,
        )
    except Exception as e:
        run = session.query(Run).filter(Run.run_id == run_id).first() if "run_id" in locals() else None
        if run:
            run.status = "failed"
            run.error_text = str(e)
            session.commit()
        raise HTTPException(status_code=500, detail={"run_id": run_id if "run_id" in locals() else None, "error": str(e)})


@router.get("/{run_id}", response_model=RunMetaResponse)
def get_run(run_id: str, session: Session = Depends(get_db)):
    run = session.query(Run).filter(Run.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    score = (
        session.query(Score)
        .filter(Score.run_id == run_id)
        .order_by(Score.created_at.desc())
        .first()
    )
    return RunMetaResponse(
        run_id=run.run_id,
        status=run.status,
        claim_text=run.claim_text,
        query_text=run.query_text,
        created_at=run.created_at,
        params_json=run.params_json,
        error_text=run.error_text,
        score=score.score if score else None,
        label=score.label if score else None,
    )


@router.get("/{run_id}/evidence", response_model=list[EvidenceOut])
def get_evidence(
    run_id: str,
    limit: int = Query(50, ge=1, le=250),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_db),
):
    rows = (
        session.query(EvidenceItem)
        .filter(EvidenceItem.run_id == run_id)
        .order_by(EvidenceItem.retrieved_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    return [
        EvidenceOut(
            domain=e.domain,
            title=e.title,
            snippet=e.snippet,
            url=e.url,
            published_at=e.published_at,
            retrieved_at=e.retrieved_at,
        )
        for e in rows
    ]


@router.get("/{run_id}/features", response_model=list[FeatureOut])
def get_features(run_id: str, session: Session = Depends(get_db)):
    rows = (
        session.query(Feature)
        .filter(Feature.run_id == run_id)
        .order_by(Feature.feature_group, Feature.feature_name)
        .all()
    )
    return [
        FeatureOut(
            feature_group=f.feature_group,
            feature_name=f.feature_name,
            feature_value=float(f.feature_value),
        )
        for f in rows
    ]


@router.get("/{run_id}/score", response_model=ScoreResponse)
def get_score(run_id: str, session: Session = Depends(get_db)):
    score = (
        session.query(Score)
        .filter(Score.run_id == run_id)
        .order_by(Score.created_at.desc())
        .first()
    )
    if not score:
        raise HTTPException(status_code=404, detail="Score not found")

    thresholds = None
    model_repo = TrainedModelRepository(session)
    trained = model_repo.get(score.model_version)
    if trained:
        thresholds = json.loads(trained.thresholds_json)

    contributions = None
    if score.explanation_json:
        try:
            contributions = json.loads(score.explanation_json)
        except json.JSONDecodeError:
            contributions = None

    return ScoreResponse(
        run_id=score.run_id,
        model_version=score.model_version,
        score=score.score,
        label=score.label,
        created_at=score.created_at,
        contributions=contributions,
        thresholds=thresholds,
    )


@router.get("/{run_id}/explanation", response_model=ExplanationResponse)
def get_explanation(run_id: str, session: Session = Depends(get_db)):
    row = (
        session.query(Explanation)
        .filter(Explanation.run_id == run_id)
        .order_by(Explanation.created_at.desc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return ExplanationResponse(
        run_id=row.run_id,
        model_id=row.model_id,
        created_at=row.created_at,
        mode=row.mode,
        user_question=row.user_question,
        response_text=row.response_text,
        context_json=row.context_json,
    )


@router.post("/{run_id}/chat", response_model=ChatResponse)
def chat_run(
    run_id: str,
    payload: ChatRequest,
    session: Session = Depends(get_db),
    llm_client=Depends(get_llm),
):
    explainer = LLMExplainer(session, llm_client)
    result = explainer.chat(run_id, payload.question)
    return ChatResponse(run_id=run_id, answer=result.response_text)
