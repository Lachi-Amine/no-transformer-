from __future__ import annotations

import re
from typing import Iterable

from engines.base import Engine
from engines.green_symbolic import GreenSymbolicEngine
from engines.red_synthesis import RedSynthesisEngine
from engines.yellow_retrieval import YellowRetrievalEngine

from . import config as _config
from . import fusion, query_processing
from .confidence import ConfidenceEstimator
from .domain_classifier import DomainClassifier
from .intent_classifier import IntentClassifier
from .router import EpistemicRouter
from .schemas import EpistemicVector, FusedEvidence, Query, Response

_PRONOUN_RE = re.compile(
    r"\b(it|its|that|this|they|them|their|those|these)\b", re.IGNORECASE
)
_TOPIC_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "how", "does", "do", "of", "and",
    "or", "but", "in", "on", "for", "to", "by", "with", "from", "this", "that",
    "those", "these", "it", "they", "them", "their", "its", "be", "was", "were",
    "can", "could", "should", "would", "will", "may", "might", "compare", "between",
    "explain", "tell", "give", "say", "your", "you", "we", "our", "my", "i",
    "why", "when", "where", "who", "which", "which", "kind", "type", "find",
    "compute", "calculate", "solve", "predict", "summarize", "describe",
}


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
        self._max_history = int(_config.get("orchestrator", "history_size", 3))
        self.history: list[Response] = []

    def reload(self) -> None:
        self.domain_clf = DomainClassifier()
        self.intent_clf = IntentClassifier()
        self.router = EpistemicRouter()
        self.confidence = ConfidenceEstimator()

    def forget(self) -> None:
        self.history.clear()

    def bench(self, raw: str) -> dict[str, float]:
        import time
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        expanded, coref_info = _resolve_coref(raw, self.history)
        processed = query_processing.process(expanded)
        if coref_info is not None:
            query = Query(
                raw=raw,
                normalized=processed.normalized,
                tokens=processed.tokens,
                entities={**processed.entities, "coref_topic": coref_info["topic"]},
            )
        else:
            query = processed
        timings["query_processing"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cls_domain = self.domain_clf.predict(query)
        timings["domain_classifier"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        cls = self.intent_clf.predict(query, cls_domain)
        timings["intent_classifier"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        epi = self.router.predict(query, cls)
        timings["epistemic_router"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        evidence, _ = fusion.fuse(query, cls, epi, self.engines)
        timings["fusion_engines"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        confidence = self.confidence.predict(epi, evidence)
        timings["confidence"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        render(query, cls, epi, evidence, confidence)
        timings["render"] = time.perf_counter() - t0

        timings["total"] = sum(timings.values())
        return timings

    def status(self) -> dict[str, str]:
        return {
            "domain_classifier": "trained" if self.domain_clf.is_trained else "stub",
            "intent_classifier": "trained" if self.intent_clf.is_trained else "stub",
            "epistemic_router":  "trained" if self.router.is_trained else "stub",
            "confidence":        "trained" if self.confidence.is_trained else "stub",
        }

    def run(self, raw: str) -> Response:
        expanded, coref_info = _resolve_coref(raw, self.history)

        processed = query_processing.process(expanded)
        if coref_info is not None:
            query = Query(
                raw=raw,
                normalized=processed.normalized,
                tokens=processed.tokens,
                entities={**processed.entities, "coref_topic": coref_info["topic"]},
            )
        else:
            query = processed

        cls_domain = self.domain_clf.predict(query)
        cls = self.intent_clf.predict(query, cls_domain)
        epi = self.router.predict(query, cls)
        evidence, engine_status = fusion.fuse(query, cls, epi, self.engines)
        confidence = self.confidence.predict(epi, evidence)
        rendered = render(query, cls, epi, evidence, confidence)

        debug = {
            "engine_status": engine_status,
            "status": self.status(),
            "coref": coref_info,
        }

        response = Response(
            query=query,
            classification=cls,
            epistemic=epi,
            evidence=evidence,
            confidence=confidence,
            rendered=rendered,
            debug=debug,
        )

        self.history.append(response)
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]
        return response


def _extract_topic(query: Query) -> str | None:
    candidates = [t for t in query.tokens if t.lower() not in _TOPIC_STOPWORDS and len(t) >= 4]
    if not candidates:
        return None
    return max(candidates, key=len)


def _resolve_coref(raw: str, history: list[Response]) -> tuple[str, dict | None]:
    if not history:
        return raw, None
    if not _PRONOUN_RE.search(raw):
        return raw, None

    topic = _extract_topic(history[-1].query)
    if not topic:
        return raw, None

    expanded = _PRONOUN_RE.sub(topic, raw)
    return expanded, {
        "topic": topic,
        "from_query": history[-1].query.raw,
        "expanded": expanded,
    }


_ENGINE_LABELS = {"green": "Formally", "yellow": "Empirically", "red": "Interpretively"}

_SYNTHESIS_TEMPLATES: dict[str, str] = {
    "compute":         "{green} In broader context, {yellow_lc}",
    "define":          "{green} In practice, {yellow_lc}",
    "explain_process": "{green} Empirically, {yellow_lc}",
    "summarize":       "{yellow} Formally, {green_lc}",
}


def render(query, cls, epi: EpistemicVector, evidence: FusedEvidence, confidence: float) -> str:
    if not evidence.records:
        return (
            f"I do not have sufficient knowledge to answer that yet. "
            f"(domain={cls.domain}, intent={cls.intent}, no engines produced evidence.)"
        )

    lines: list[str] = []
    by_engine: dict[str, list] = {"green": [], "yellow": [], "red": []}
    for r in evidence.records:
        by_engine.setdefault(r.engine, []).append(r)

    synthesized = _try_synthesize(by_engine, cls.intent)
    if synthesized is not None:
        text, support = synthesized
        citation = _format_citation(tuple(support))
        line = text if not citation else f"{text} {citation}"
        lines.append(line)
        for record in by_engine.get("red", []):
            cite = _format_citation(record.support)
            r_line = f"Interpretively: {record.claim}"
            if cite:
                r_line = f"{r_line} {cite}"
            lines.append(r_line)
    else:
        for engine_name in ("green", "yellow", "red"):
            for record in by_engine.get(engine_name, []):
                label = _ENGINE_LABELS.get(engine_name, engine_name.capitalize())
                citation = _format_citation(record.support)
                line = f"{label}: {record.claim}"
                if citation:
                    line = f"{line} {citation}"
                lines.append(line)

    if evidence.contradictions:
        lines.append("")
        lines.append(f"Sources disagree ({len(evidence.contradictions)} conflict(s)):")
        for i, j in evidence.contradictions:
            if i < len(evidence.records) and j < len(evidence.records):
                rec_a = evidence.records[i]
                rec_b = evidence.records[j]
                lines.append(f"  - [{rec_a.engine}] {rec_a.claim}")
                lines.append(f"  - [{rec_b.engine}] {rec_b.claim}")

    return "\n".join(lines)


def _try_synthesize(by_engine: dict[str, list], intent: str):
    template = _SYNTHESIS_TEMPLATES.get(intent)
    if template is None:
        return None
    if not by_engine.get("green") or not by_engine.get("yellow"):
        return None

    green = by_engine["green"][0]
    yellow = by_engine["yellow"][0]

    green_text = _ensure_period(green.claim.strip())
    yellow_text = _ensure_period(yellow.claim.strip())

    text = template.format(
        green=green_text,
        yellow=yellow_text,
        green_lc=_lower_first(green_text),
        yellow_lc=_lower_first(yellow_text),
    )

    support = list(green.support) + [s for s in yellow.support if s not in green.support]
    return text, support


def _ensure_period(s: str) -> str:
    if not s:
        return s
    if s[-1] in ".!?":
        return s
    return s + "."


def _lower_first(s: str) -> str:
    if not s:
        return s
    return s[0].lower() + s[1:]


def _format_citation(support: tuple) -> str:
    ids = [s for s in support if isinstance(s, str) and s and not s.startswith("freeform")]
    ids = [s for s in ids if "=" not in s and "*" not in s and "/" not in s]
    if not ids:
        return ""
    if len(ids) == 1:
        return f"[source: {ids[0]}]"
    return f"[sources: {', '.join(ids)}]"
