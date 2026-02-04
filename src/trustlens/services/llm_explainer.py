"""LLM explainer service for grounded summaries and chat."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from trustlens.db.schema import EvidenceItem, Feature, Run, Score
from trustlens.repos.explanation_repo import ExplanationRepository
from trustlens.services.llm_client import LLMClient
from trustlens.services.scoring import BaselineScorer


SYSTEM_PROMPT = (
    "You are a grounded explainer. Use ONLY the provided context JSON. "
    "Do not introduce new evidence or facts. "
    "If information is missing, say it is missing. "
    "Cite which signals drove the score. Include a 'What we don't know' section."
)

SUMMARY_PROMPT = "Explain why the system produced this score and label. Use only provided context."

CHAT_PROMPT_PREFIX = (
    "Answer the user's question using only the provided context. "
    "If the answer is not in the context, say you don't know."
)


@dataclass(frozen=True)
class ExplanationResult:
    run_id: str
    response_text: str
    context_json: str


class LLMExplainer:
    """Builds grounded context and generates explanations via an LLM client."""

    def __init__(self, session: Session, client: LLMClient, now_fn: callable = datetime.utcnow):
        self.session = session
        self.client = client
        self.now_fn = now_fn
        self.repo = ExplanationRepository(session)

    def build_context(self, run_id: str, model_id: str = "baseline_v1", top_n: int = 10) -> Dict[str, Any]:
        run = self.session.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} does not exist")

        score_row = (
            self.session.query(Score)
            .filter(Score.run_id == run_id, Score.model_version == model_id)
            .order_by(Score.created_at.desc())
            .first()
        )

        features = (
            self.session.query(Feature)
            .filter(Feature.run_id == run_id)
            .order_by(Feature.feature_group, Feature.feature_name)
            .all()
        )

        evidence = (
            self.session.query(EvidenceItem)
            .filter(EvidenceItem.run_id == run_id)
            .order_by(func.coalesce(EvidenceItem.published_at, EvidenceItem.retrieved_at).desc())
            .limit(top_n)
            .all()
        )

        contributions: Dict[str, Any]
        if score_row and score_row.explanation_json:
            try:
                contributions = json.loads(score_row.explanation_json)
            except json.JSONDecodeError:
                contributions = {"missing": True}
        else:
            # compute contributions if possible
            if features:
                feature_map = {f.feature_name: float(f.feature_value) for f in features}
                scorer = BaselineScorer()
                _, contribs = scorer._raw_score(feature_map)
                contributions = {
                    "positive": [
                        {"feature_name": k, "contribution": v, "value": feature_map.get(k, 0.0)}
                        for k, v in contribs.items()
                        if v >= 0
                    ],
                    "negative": [
                        {"feature_name": k, "contribution": v, "value": feature_map.get(k, 0.0)}
                        for k, v in contribs.items()
                        if v < 0
                    ],
                }
            else:
                contributions = {"missing": True}

        context = {
            "run": {
                "run_id": run.run_id,
                "claim_text": run.claim_text,
                "query_text": run.query_text,
                "created_at": run.created_at.isoformat() if run.created_at else None,
            },
            "score": (
                {
                    "model_id": model_id,
                    "score": score_row.score,
                    "label": score_row.label,
                    "created_at": score_row.created_at.isoformat() if score_row.created_at else None,
                }
                if score_row
                else {"missing": True}
            ),
            "features": [
                {
                    "group": f.feature_group,
                    "name": f.feature_name,
                    "value": float(f.feature_value),
                }
                for f in features
            ],
            "contributions": contributions,
            "evidence": [
                {
                    "domain": e.domain,
                    "title": e.title,
                    "snippet": e.snippet,
                    "url": e.url,
                    "published_at": e.published_at.isoformat() if e.published_at else None,
                }
                for e in evidence
            ],
        }
        return context

    def explain_run(self, run_id: str, model_id: str = "baseline_v1") -> ExplanationResult:
        context = self.build_context(run_id, model_id=model_id)
        context_json = json.dumps(context, sort_keys=True)
        response = self.client.generate(SYSTEM_PROMPT, f"{SUMMARY_PROMPT}\n\nContext:\n{context_json}")

        self.repo.upsert_latest(
            run_id=run_id,
            model_id=model_id,
            mode="summary",
            user_question=None,
            response_text=response,
            context_json=context_json,
            created_at=self.now_fn(),
        )
        return ExplanationResult(run_id=run_id, response_text=response, context_json=context_json)

    def chat(self, run_id: str, question: str, model_id: str = "baseline_v1") -> ExplanationResult:
        context = self.build_context(run_id, model_id=model_id)
        context_json = json.dumps(context, sort_keys=True)
        user_prompt = f"{CHAT_PROMPT_PREFIX}\n\nQuestion:\n{question}\n\nContext:\n{context_json}"
        response = self.client.generate(SYSTEM_PROMPT, user_prompt)

        self.repo.upsert_latest(
            run_id=run_id,
            model_id=model_id,
            mode="chat",
            user_question=question,
            response_text=response,
            context_json=context_json,
            created_at=self.now_fn(),
        )
        return ExplanationResult(run_id=run_id, response_text=response, context_json=context_json)
