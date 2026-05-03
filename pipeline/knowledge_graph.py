from __future__ import annotations

import re
from typing import Iterable

_MIN_TERM_LEN = 4


class KnowledgeGraph:
    """Builds a term -> entry_id index from a collection of YAML entries and
    answers 'what other entries are mentioned by this one?'

    Entries are matched by keywords, tags, and the prefix of their id slug
    (e.g. 'amylase-001' contributes the term 'amylase').
    """

    def __init__(self, entries: Iterable[dict]) -> None:
        self.entries: list[dict] = [e for e in entries if isinstance(e, dict)]
        self.id_to_entry: dict[str, dict] = {
            e["id"]: e for e in self.entries if e.get("id")
        }
        self.term_to_ids: dict[str, set[str]] = {}
        for entry in self.entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            for term in self._terms_for(entry):
                self.term_to_ids.setdefault(term, set()).add(entry_id)

        # Adjacency list: entry_id -> set of linked entry_ids (precomputed)
        self.edges: dict[str, set[str]] = {}
        for entry in self.entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            self.edges[entry_id] = set(self.linked_ids(entry))

    @staticmethod
    def _terms_for(entry: dict) -> set[str]:
        terms: set[str] = set()
        for kw in entry.get("keywords") or []:
            t = str(kw).strip().lower()
            if len(t) >= _MIN_TERM_LEN:
                terms.add(t)
        for tag in entry.get("tags") or []:
            t = str(tag).strip().lower()
            if len(t) >= _MIN_TERM_LEN:
                terms.add(t)
        entry_id = entry.get("id") or ""
        if "-" in entry_id:
            slug = entry_id.rsplit("-", 1)[0]
            for tok in re.split(r"[-_\s]+", slug.lower()):
                if len(tok) >= _MIN_TERM_LEN:
                    terms.add(tok)
        return terms

    def linked_ids(
        self,
        source_entry: dict,
        exclude_ids: set[str] | None = None,
    ) -> list[str]:
        """Return ids of other entries whose terms appear in source's text."""
        text = (source_entry.get("text") or "").lower()
        if not text:
            return []

        excluded: set[str] = set(exclude_ids or set())
        if source_entry.get("id"):
            excluded.add(source_entry["id"])

        hits: set[str] = set()
        for term, ids in self.term_to_ids.items():
            if re.search(rf"\b{re.escape(term)}\b", text):
                hits |= ids
        return [eid for eid in hits if eid not in excluded]

    def linked_entries(
        self,
        source_entry: dict,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        return [
            self.id_to_entry[eid]
            for eid in self.linked_ids(source_entry, exclude_ids)
            if eid in self.id_to_entry
        ]

    def neighbors(self, entry_id: str) -> set[str]:
        """Precomputed adjacency lookup."""
        return self.edges.get(entry_id, set())

    def find_by_term(self, term: str) -> list[dict]:
        """Return all entries matched by an exact term (keyword/tag/slug)."""
        ids = self.term_to_ids.get(term.lower(), set())
        return [self.id_to_entry[i] for i in ids if i in self.id_to_entry]

    def stats(self) -> dict:
        n_nodes = len(self.id_to_entry)
        # undirected unique edges
        unique_pairs: set[tuple[str, str]] = set()
        for src, dsts in self.edges.items():
            for dst in dsts:
                unique_pairs.add(tuple(sorted([src, dst])))
        n_edges = len(unique_pairs)

        isolated = sorted(eid for eid, neigh in self.edges.items() if not neigh)
        most_connected = sorted(
            self.edges.items(), key=lambda kv: -len(kv[1])
        )[:5]

        by_domain: dict[str, dict[str, int]] = {}
        for entry in self.entries:
            d = entry.get("domain") or "unknown"
            by_domain.setdefault(d, {"nodes": 0, "edges": 0})
            by_domain[d]["nodes"] += 1
        # domain-internal edge counts
        for src, dsts in self.edges.items():
            src_entry = self.id_to_entry.get(src)
            if not src_entry:
                continue
            src_domain = src_entry.get("domain") or "unknown"
            for dst in dsts:
                dst_entry = self.id_to_entry.get(dst)
                if not dst_entry:
                    continue
                if dst_entry.get("domain") == src_domain and src < dst:
                    by_domain[src_domain]["edges"] += 1

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "isolated_count": len(isolated),
            "isolated_ids": isolated,
            "most_connected": [(eid, len(neigh)) for eid, neigh in most_connected],
            "by_domain": by_domain,
        }
