from __future__ import annotations

import re
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "empirical"
_TOKEN_RE = re.compile(r"[a-z]+")
_BM25_THRESHOLD = 1.0
_BM25_NORMALIZER = 10.0


class YellowRetrievalEngine(Engine):
    name = "yellow"

    def __init__(self) -> None:
        self.entries = load_knowledge(KNOWLEDGE_DIR)
        self._bm25 = None
        self._corpus: list[list[str]] = []
        if self.entries:
            self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return
        for e in self.entries:
            blob = " ".join([
                e.get("text", ""),
                " ".join(e.get("keywords", []) or []),
                " ".join(e.get("tags", []) or []),
            ]).lower()
            self._corpus.append(_TOKEN_RE.findall(blob))
        self._bm25 = BM25Okapi(self._corpus)

    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        if self._bm25 is None or not self.entries:
            return None

        q_tokens = [t.lower() for t in query.tokens if t.isalpha()]
        if not q_tokens:
            return None

        scores = self._bm25.get_scores(q_tokens)

        candidate_idx = [
            i for i, e in enumerate(self.entries)
            if cls.domain == "general" or e.get("domain") == cls.domain
        ]
        if cls.domain == "general":
            q_set = set(q_tokens)
            candidate_idx = [
                i for i in candidate_idx
                if q_set & {k.lower() for k in (self.entries[i].get("keywords") or [])}
            ]
        if not candidate_idx:
            return None

        ranked = sorted(candidate_idx, key=lambda i: -scores[i])
        top_idx = ranked[0]
        top_score = float(scores[top_idx])
        if top_score < _BM25_THRESHOLD:
            return None
        top = self.entries[top_idx]
        text = (top.get("text") or "").strip()
        first_sentence = text.split(".")[0].strip() + "." if text else top.get("id", "")

        support_ids = tuple(
            (self.entries[i].get("id") or f"entry-{i}")
            for i in ranked[:3]
            if scores[i] >= _BM25_THRESHOLD * 0.5
        )

        return EvidenceRecord(
            engine="yellow",
            claim=first_sentence,
            support=support_ids,
            score=min(1.0, top_score / _BM25_NORMALIZER),
        )
