from __future__ import annotations

import re
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from pipeline import config as _config
from pipeline.knowledge_graph import KnowledgeGraph

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "empirical"
_TOKEN_RE = re.compile(r"[a-z]+")


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


class YellowRetrievalEngine(Engine):
    name = "yellow"

    def __init__(self) -> None:
        self.entries = load_knowledge(KNOWLEDGE_DIR)
        self._bm25 = None
        self._corpus: list[list[str]] = []
        self.graph: KnowledgeGraph | None = None
        if self.entries:
            self._build_index()
            self.graph = KnowledgeGraph(self.entries)

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
            tokens = _TOKEN_RE.findall(blob)
            self._corpus.append([_stem(t) for t in tokens])
        self._bm25 = BM25Okapi(self._corpus)

    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        if self._bm25 is None or not self.entries:
            return None

        q_tokens = [_stem(t.lower()) for t in query.tokens if t.isalpha()]
        if not q_tokens:
            return None

        cfg = _config.load()["yellow"]
        bm25_threshold = float(cfg["bm25_threshold"])
        bm25_normalizer = float(cfg["bm25_normalizer"])
        domain_fallback_ratio = float(cfg["domain_fallback_ratio"])
        secondary_floor_ratio = float(cfg["secondary_floor_ratio"])
        max_passages = int(cfg["max_passages"])
        max_linked = int(cfg.get("max_linked", 2))
        linked_score_floor = float(cfg.get("linked_score_floor", 0.3))

        scores = self._bm25.get_scores(q_tokens)
        q_set = set(q_tokens)

        global_best = max(range(len(self.entries)), key=lambda i: scores[i])
        global_best_score = float(scores[global_best])

        if cls.domain == "general":
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

        if global_best_score > top_score * domain_fallback_ratio and global_best_score >= bm25_threshold:
            top_idx = global_best
            top_score = global_best_score
            ranked = sorted(range(len(self.entries)), key=lambda i: -scores[i])

        if top_score < bm25_threshold:
            return None

        top_entry = self.entries[top_idx]
        top_keywords = {k.lower() for k in (top_entry.get("keywords") or [])}
        top_id = (top_entry.get("id") or "").lower()
        if "-" in top_id:
            slug = top_id.rsplit("-", 1)[0]
            for tok in re.split(r"[-_\s]+", slug):
                if len(tok) >= 4:
                    top_keywords.add(tok)
        q_stems = q_set | {_stem(t) for t in q_set}
        kw_stems = top_keywords | {_stem(k) for k in top_keywords}
        if not (q_stems & kw_stems):
            return None

        top_domain = self.entries[top_idx].get("domain")
        secondary_floor = max(bm25_threshold * 1.5, top_score * secondary_floor_ratio)
        selected = []
        for i in ranked[:max_passages]:
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

        if self.graph is not None and max_linked > 0:
            primary_entry = self.entries[top_idx]
            already = {self.entries[i].get("id") for i in selected if self.entries[i].get("id")}
            linked = self.graph.linked_entries(primary_entry, exclude_ids=already)
            id_to_idx = {e.get("id"): i for i, e in enumerate(self.entries) if e.get("id")}
            scored: list[tuple[int, float]] = []
            for entry in linked:
                eid = entry.get("id")
                if eid not in id_to_idx:
                    continue
                if entry.get("domain") != top_domain:
                    continue
                idx = id_to_idx[eid]
                scored.append((idx, float(scores[idx])))
            scored.sort(key=lambda p: -p[1])

            link_floor = top_score * linked_score_floor
            for idx, sc in scored[:max_linked]:
                if sc < link_floor:
                    break
                e = self.entries[idx]
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
                sentences.append("Related: " + sent + ".")
                support_ids.append(e.get("id") or f"entry-{idx}")

        claim = " ".join(sentences)
        return EvidenceRecord(
            engine="yellow",
            claim=claim,
            support=tuple(support_ids),
            score=min(1.0, top_score / bm25_normalizer),
        )
