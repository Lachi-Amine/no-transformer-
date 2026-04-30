from __future__ import annotations

import re
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "empirical"
_TOKEN_RE = re.compile(r"[a-z]+")
_BM25_THRESHOLD = 1.0
_BM25_NORMALIZER = 10.0
_DOMAIN_FALLBACK_RATIO = 2.0


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

        global_best = max(range(len(self.entries)), key=lambda i: scores[i])
        global_best_score = float(scores[global_best])

        if cls.domain == "general":
            q_set = set(q_tokens)
            candidate_idx = [
                i for i, e in enumerate(self.entries)
                if q_set & {k.lower() for k in (e.get("keywords") or [])}
            ]
        else:
            candidate_idx = [
                i for i, e in enumerate(self.entries)
                if e.get("domain") == cls.domain
            ]
        if not candidate_idx:
            return None

        ranked = sorted(candidate_idx, key=lambda i: -scores[i])
        top_idx = ranked[0]
        top_score = float(scores[top_idx])

        if global_best_score > top_score * _DOMAIN_FALLBACK_RATIO and global_best_score >= _BM25_THRESHOLD:
            top_idx = global_best
            top_score = global_best_score
            ranked = sorted(range(len(self.entries)), key=lambda i: -scores[i])

        if top_score < _BM25_THRESHOLD:
            return None

        top_domain = self.entries[top_idx].get("domain")
        secondary_floor = max(_BM25_THRESHOLD * 1.5, top_score * 0.65)
        selected = []
        for i in ranked[:3]:
            if i == top_idx:
                selected.append(i)
                continue
            if self.entries[i].get("domain") != top_domain:
                continue
            if float(scores[i]) >= secondary_floor:
                selected.append(i)

        sentences: list[str] = []
        support_ids: list[str] = []
        seen_starts: set[str] = set()
        for i in selected:
            e = self.entries[i]
            text = (e.get("text") or "").strip()
            if not text:
                continue
            sent = text.split(".")[0].strip()
            if not sent:
                continue
            head = sent[:24].lower()
            if head in seen_starts:
                continue
            seen_starts.add(head)
            sentences.append(sent + ".")
            support_ids.append(e.get("id") or f"entry-{i}")

        if not sentences:
            return None

        claim = " ".join(sentences)
        return EvidenceRecord(
            engine="yellow",
            claim=claim,
            support=tuple(support_ids),
            score=min(1.0, top_score / _BM25_NORMALIZER),
        )
