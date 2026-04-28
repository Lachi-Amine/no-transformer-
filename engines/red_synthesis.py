from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "interpretive"


class RedSynthesisEngine(Engine):
    name = "red"

    def __init__(self) -> None:
        self.entries = load_knowledge(KNOWLEDGE_DIR)
        self.by_topic: dict[str, list[dict]] = defaultdict(list)
        for e in self.entries:
            topic = e.get("topic")
            if topic:
                self.by_topic[topic].append(e)

    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        if not self.by_topic:
            return None

        topic, overlap = self._best_topic(query)
        if topic is None or overlap == 0:
            return None

        perspectives = self.by_topic[topic]
        if len(perspectives) < 2:
            return None

        parts: list[str] = []
        for p in perspectives:
            tradition = p.get("tradition", "unknown")
            text = (p.get("text") or "").strip()
            first = text.split(".")[0].strip()
            if first:
                parts.append(f"From the {tradition} view: {first}.")

        if not parts:
            return None

        claim = " ".join(parts)
        support = tuple(p.get("id", "") for p in perspectives)
        score = min(1.0, len(perspectives) / 3.0)

        return EvidenceRecord(
            engine="red",
            claim=claim,
            support=support,
            score=score,
        )

    def _best_topic(self, query: Query) -> tuple[str | None, int]:
        q_tokens = {t.lower() for t in query.tokens}
        best, best_score = None, 0
        for topic, items in self.by_topic.items():
            keywords: set[str] = set()
            for e in items:
                keywords |= {k.lower() for k in (e.get("keywords") or [])}
                keywords |= {t.lower() for t in (e.get("tags") or [])}
            keywords |= set(topic.lower().replace("_", " ").split())
            overlap = len(q_tokens & keywords)
            if overlap > best_score:
                best, best_score = topic, overlap
        return best, best_score
