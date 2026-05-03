"""Per-intent / per-domain reasoning rules.

Each rule looks at the standard pipeline's response and, if a domain-specific
pattern is present, surfaces a related entry from the knowledge graph as a
"See also:" annotation. Rules are pure functions; they don't modify the
classifier or router decisions, only enrich the rendered output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge_graph import KnowledgeGraph
from .schemas import EvidenceRecord, Response


@dataclass(frozen=True)
class RuleHit:
    rule_name: str
    record: EvidenceRecord
    explanation: str


class _Rule:
    name: str = ""

    def fire(self, response: Response, kg: KnowledgeGraph) -> RuleHit | None:
        raise NotImplementedError


def _first_sentence(entry: dict) -> str:
    text = (entry.get("text") or "").strip()
    return text.split(".", 1)[0].strip()


def _make_record(entry: dict, lead_in: str = "See also") -> EvidenceRecord:
    return EvidenceRecord(
        engine="rule",
        claim=f"{lead_in}: {_first_sentence(entry)}.",
        support=(entry.get("id", ""),) if entry.get("id") else (),
        score=0.5,
    )


def _haystack(response: Response) -> str:
    """Combine the rendered output, raw query, and normalized query into one
    lowercased string for trigger / inhibitor matching."""
    parts = [
        response.rendered or "",
        response.query.raw or "",
        response.query.normalized or "",
    ]
    return " ".join(parts).lower()


def _already_cited(response: Response, entry_id: str) -> bool:
    for rec in response.evidence.records:
        if entry_id in rec.support:
            return True
    return False


class _MentionsTermSurfacesEntry(_Rule):
    """Generic rule: if `triggers` appear in the response (and `inhibitors`
    do not), surface `target_id` from the graph."""

    def __init__(
        self,
        name: str,
        triggers: tuple[str, ...],
        inhibitors: tuple[str, ...],
        target_id: str,
        lead_in: str = "See also",
        explanation: str = "",
    ):
        self.name = name
        self.triggers = triggers
        self.inhibitors = inhibitors
        self.target_id = target_id
        self.lead_in = lead_in
        self.explanation = explanation or f"matched {triggers} -> {target_id}"

    def fire(self, response: Response, kg: KnowledgeGraph) -> RuleHit | None:
        text = _haystack(response)
        if not any(self._has_word(text, t) for t in self.triggers):
            return None
        if any(self._has_word(text, t) for t in self.inhibitors):
            return None
        if _already_cited(response, self.target_id):
            return None
        target = kg.id_to_entry.get(self.target_id)
        if not target:
            return None
        return RuleHit(self.name, _make_record(target, self.lead_in), self.explanation)

    @staticmethod
    def _has_word(text: str, term: str) -> bool:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None


_RULES: tuple[_Rule, ...] = (
    _MentionsTermSurfacesEntry(
        name="chemistry.acid_pH",
        triggers=("acid", "base", "buffer", "alkaline"),
        inhibitors=("ph", "pka", "henderson", "hasselbalch"),
        target_id="ph-001",
        explanation="response mentions acid/base; surfacing the pH scale",
    ),
    _MentionsTermSurfacesEntry(
        name="chemistry.buffer_henderson",
        triggers=("buffer",),
        inhibitors=("henderson", "hasselbalch", "pka"),
        target_id="henderson-hasselbalch-001",
        explanation="response mentions a buffer; surfacing Henderson-Hasselbalch",
    ),
    _MentionsTermSurfacesEntry(
        name="physics.motion_kinetic_energy",
        triggers=("motion", "moving", "velocity"),
        inhibitors=("kinetic energy", "e = m"),
        target_id="kinetic-energy-001",
        explanation="response mentions motion; surfacing kinetic energy formula",
    ),
    _MentionsTermSurfacesEntry(
        name="physics.force_newton",
        triggers=("force",),
        inhibitors=("newton", "f = m"),
        target_id="newton-second-001",
        explanation="response mentions force; surfacing Newton's second law",
    ),
    _MentionsTermSurfacesEntry(
        name="biology.enzyme_catalyst",
        triggers=("enzyme",),
        inhibitors=("catalyst",),
        target_id="catalyst-001",
        explanation="response mentions an enzyme; surfacing catalyst definition",
    ),
    _MentionsTermSurfacesEntry(
        name="economics.interest_compound",
        triggers=("interest",),
        inhibitors=("compound interest",),
        target_id="compound-interest-001",
        explanation="response mentions interest; surfacing compound interest definition",
    ),
    _MentionsTermSurfacesEntry(
        name="math.average_mean",
        triggers=("average",),
        inhibitors=("arithmetic mean", "mu = "),
        target_id="arithmetic-mean-001",
        explanation="response mentions average; surfacing arithmetic mean formula",
    ),
)


def apply_rules(response: Response, kg: KnowledgeGraph) -> list[RuleHit]:
    hits: list[RuleHit] = []
    for rule in _RULES:
        try:
            hit = rule.fire(response, kg)
        except Exception:
            continue
        if hit is not None:
            hits.append(hit)
    return hits
