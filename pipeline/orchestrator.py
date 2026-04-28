from __future__ import annotations

from typing import Iterable

from engines.base import Engine
from engines.green_symbolic import GreenSymbolicEngine
from engines.red_synthesis import RedSynthesisEngine
from engines.yellow_retrieval import YellowRetrievalEngine

from . import fusion, query_processing
from .confidence import ConfidenceEstimator
from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .router import EpistemicRouter
from .schemas import EpistemicVector, FusedEvidence, Response


class Pipeline:
    def __init__(self) -> None:
        self.domain_clf = DomainClassifier()
        self.intent_clf = IntentClassifier()
        self.router = EpistemicRouter()
        self.confidence = ConfidenceEstimator()
        self.engines: tuple[Engine, ...] = (
            GreenSymbolicEngine(),
            YellowRetrievalEngine(),
            RedSynthesisEngine(),
        )

    def reload(self) -> None:
        self.domain_clf = DomainClassifier()
        self.intent_clf = IntentClassifier()
        self.router = EpistemicRouter()
        self.confidence = ConfidenceEstimator()

    def status(self) -> dict[str, str]:
        return {
            "domain_classifier": "trained" if self.domain_clf.is_trained else "stub",
            "intent_classifier": "trained" if self.intent_clf.is_trained else "stub",
            "epistemic_router":  "trained" if self.router.is_trained else "stub",
            "confidence":        "trained" if self.confidence.is_trained else "stub",
        }

    def run(self, raw: str) -> Response:
        query = query_processing.process(raw)
        cls_domain = self.domain_clf.predict(query)
        cls = self.intent_clf.predict(query, cls_domain)
        epi = self.router.predict(query, cls)
        evidence, engine_status = fusion.fuse(query, cls, epi, self.engines)
        confidence = self.confidence.predict(epi, evidence)
        rendered = render(query, cls, epi, evidence, confidence)

        return Response(
            query=query,
            classification=cls,
            epistemic=epi,
            evidence=evidence,
            confidence=confidence,
            rendered=rendered,
            debug={"engine_status": engine_status, "status": self.status()},
        )


def render(query, cls, epi: EpistemicVector, evidence: FusedEvidence, confidence: float) -> str:
    if not evidence.records:
        return (
            f"I do not have sufficient knowledge to answer that yet. "
            f"(domain={cls.domain}, intent={cls.intent}, no engines produced evidence.)"
        )

    lines: list[str] = []
    by_engine = {"green": [], "yellow": [], "red": []}
    for r in evidence.records:
        by_engine.setdefault(r.engine, []).append(r)

    if by_engine.get("green"):
        for r in by_engine["green"]:
            lines.append(f"Formally: {r.claim}")
    if by_engine.get("yellow"):
        for r in by_engine["yellow"]:
            lines.append(f"Empirically: {r.claim}")
    if by_engine.get("red"):
        for r in by_engine["red"]:
            lines.append(f"Interpretively: {r.claim}")

    if evidence.contradictions:
        lines.append(
            f"Note: {len(evidence.contradictions)} contradiction(s) detected across sources."
        )
    return "\n".join(lines)
