from __future__ import annotations

import re
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "formal"
_VALUE_RE = re.compile(r"\b([A-Za-z])\s*=\s*(-?\d+(?:\.\d+)?)\b")


class GreenSymbolicEngine(Engine):
    name = "green"

    def __init__(self) -> None:
        self.entries = load_knowledge(KNOWLEDGE_DIR)

    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        if not self.entries:
            return None

        match, overlap = self._best_match(query)
        if match is None or overlap == 0:
            return None

        equation = match.get("equation", "")
        text = match.get("text", "").strip()

        if cls.intent == "compute" and equation:
            computed = self._try_compute(query.raw, equation, match.get("variables", {}))
            if computed is not None:
                return EvidenceRecord(
                    engine="green",
                    claim=f"{text} Computed: {computed}",
                    support=(equation,),
                    score=1.0,
                )

        support = (equation,) if equation else (match.get("id", ""),)
        return EvidenceRecord(
            engine="green",
            claim=text or equation,
            support=support,
            score=min(1.0, 0.5 + 0.25 * overlap),
        )

    def _best_match(self, query: Query) -> tuple[dict | None, int]:
        q_tokens = {t.lower() for t in query.tokens}
        best, best_score = None, 0
        for e in self.entries:
            kw = {k.lower() for k in e.get("keywords", [])} | {t.lower() for t in e.get("tags", [])}
            score = len(q_tokens & kw)
            if score > best_score:
                best, best_score = e, score
        return best, best_score

    def _try_compute(self, raw: str, equation: str, variables: dict) -> str | None:
        values: dict[str, float] = {}
        for name, num in _VALUE_RE.findall(raw):
            values[name] = float(num)
        if not values:
            return None
        try:
            import sympy as sp
        except ImportError:
            return None
        try:
            if "==" in equation:
                lhs, rhs = equation.split("==", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            elif "=" in equation:
                lhs, rhs = equation.split("=", 1)
                expr = sp.sympify(lhs) - sp.sympify(rhs)
            else:
                expr = sp.sympify(equation)
            free = list(expr.free_symbols)
            unknowns = [s for s in free if s.name not in values]
            if len(unknowns) != 1:
                return None
            substituted = expr.subs({s: values[s.name] for s in free if s.name in values})
            sols = sp.solve(substituted, unknowns[0])
            if not sols:
                return None
            return f"{unknowns[0].name} = {sp.nsimplify(sols[0], rational=False)}"
        except Exception:
            return None
