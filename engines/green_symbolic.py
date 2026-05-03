from __future__ import annotations

import re
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query

from .base import Engine, load_knowledge

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "formal"

_SYNONYMS = {
    "speed":    ["velocity"],
    "velocity": ["speed"],
    "voltage":  ["potential"],
    "current":  ["amperage"],
    "length":   ["distance"],
    "mass":     ["weight"],
}

_GLOBAL_CONSTANTS = {
    "g": 9.81,
    "c": 2.998e8,
    "G": 6.674e-11,
    "k": 8.99e9,
    "h": 6.626e-34,
    "R": 8.314,
}

_MATH_VERBS = (
    "compute ", "solve ", "calculate ", "evaluate ", "simplify ",
    "integrate ", "differentiate ", "find the value of ", "the value of ",
)

_MATH_OP_RE = re.compile(r"[+\-*/^=]|\bsqrt\b|\bsin\b|\bcos\b|\btan\b|\bintegral\b|\bderivative\b|\blog\b|\bexp\b")

_EXPLICIT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)")
_NUMUNIT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*([A-Za-z][A-Za-z0-9/^*]*)")


class GreenSymbolicEngine(Engine):
    name = "green"

    def __init__(self) -> None:
        self.entries = load_knowledge(KNOWLEDGE_DIR)

    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        match, overlap = self._best_match(query) if self.entries else (None, 0)

        if match is None or overlap == 0:
            if cls.intent == "compute" or _looks_mathematical(query.raw):
                free = _freeform_compute(query.raw)
                if free:
                    return EvidenceRecord(
                        engine="green",
                        claim=f"Computed: {free}",
                        support=("freeform-sympy",),
                        score=0.85,
                    )
            return None

        equation = match.get("equation", "")
        text = (match.get("text") or "").strip()
        variables = match.get("variables") or {}
        defaults = match.get("defaults") or {}

        computed = _try_compute(query.raw, equation, variables, defaults) if equation else None

        entry_id = match.get("id", "")

        if computed and cls.intent == "compute":
            return EvidenceRecord(
                engine="green",
                claim=f"Computed: {computed}. Using {equation}.",
                support=(entry_id,) if entry_id else (),
                score=1.0,
            )
        if computed:
            return EvidenceRecord(
                engine="green",
                claim=f"{text} Computed: {computed}.",
                support=(entry_id,) if entry_id else (),
                score=1.0,
            )

        return EvidenceRecord(
            engine="green",
            claim=text or equation,
            support=(entry_id,) if entry_id else (),
            score=min(1.0, 0.5 + 0.25 * overlap),
        )

    def _best_match(self, query: Query) -> tuple[dict | None, int]:
        q_tokens = {t.lower() for t in query.tokens}
        q_stems = {_stem(t) for t in q_tokens}
        best, best_key = None, (0, 0)
        for e in self.entries:
            kw = {k.lower() for k in (e.get("keywords") or [])} | {
                t.lower() for t in (e.get("tags") or [])
            }
            kw_with_stems = kw | {_stem(k) for k in kw}
            matches = (q_tokens | q_stems) & kw_with_stems
            if not matches:
                continue
            key = (len(matches), sum(len(m) for m in matches))
            if key > best_key:
                best, best_key = e, key
        return best, best_key[0]


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _extract_values(query: str, variables: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    used_positions: set[int] = set()

    for m in _EXPLICIT_RE.finditer(query):
        var = m.group(1)
        if var in variables:
            values[var] = float(m.group(2))
            used_positions.add(m.start(2))

    unit_to_syms: dict[str, list[str]] = {}
    word_to_sym: dict[str, str] = {}
    for sym, desc in variables.items():
        if not isinstance(desc, str):
            continue
        m = re.search(r"\(([^)]+)\)", desc)
        if m:
            unit = re.sub(r"\s+", "", m.group(1).lower())
            unit_to_syms.setdefault(unit, []).append(sym)
        head = desc.split("(")[0].strip().lower()
        for word in head.split():
            if len(word) > 3 and word not in word_to_sym:
                word_to_sym[word] = sym
                for syn in _SYNONYMS.get(word, []):
                    if syn not in word_to_sym:
                        word_to_sym[syn] = sym

    for nm in _NUMUNIT_RE.finditer(query):
        if nm.start(1) in used_positions:
            continue
        unit = nm.group(2).lower()
        if unit not in unit_to_syms:
            continue
        for s in unit_to_syms[unit]:
            if s not in values:
                values[s] = float(nm.group(1))
                used_positions.add(nm.start(1))
                break

    q_low = query.lower()
    for word, sym in word_to_sym.items():
        if sym in values:
            continue
        pat = re.compile(
            rf"\b{re.escape(word)}\b(?:\s+of|\s*=|\s+is|\s+equal\s+to)?\s*(-?\d+(?:\.\d+)?)"
        )
        m = pat.search(q_low)
        if m and m.start(1) not in used_positions:
            values[sym] = float(m.group(1))
            used_positions.add(m.start(1))

    return values


def _try_compute(raw: str, equation: str, variables: dict, defaults: dict) -> str | None:
    values = _extract_values(raw, variables)
    if not values:
        return None
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return None

    sympy_ctx = {
        "pi": sp.pi,
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "ln": sp.log,
    }
    for var in variables:
        sympy_ctx[var] = sp.Symbol(var)

    try:
        if "==" in equation:
            lhs, rhs = equation.split("==", 1)
        elif "=" in equation:
            lhs, rhs = equation.split("=", 1)
        else:
            return None
        lhs_e = parse_expr(lhs.strip(), local_dict=sympy_ctx)
        rhs_e = parse_expr(rhs.strip(), local_dict=sympy_ctx)
        expr = lhs_e - rhs_e

        free = list(expr.free_symbols)
        sub = {s: values[s.name] for s in free if s.name in values}
        for s in free:
            if s in sub:
                continue
            if s.name in defaults:
                sub[s] = defaults[s.name]
            elif s.name in _GLOBAL_CONSTANTS:
                sub[s] = _GLOBAL_CONSTANTS[s.name]

        substituted = expr.subs(sub)
        unknowns = [s for s in free if s not in sub]
        if len(unknowns) != 1:
            return None

        sols = sp.solve(substituted, unknowns[0])
        if not sols:
            return None

        sol_num = float(sp.N(sols[0]))
        abs_n = abs(sol_num)
        if abs_n >= 1e6 or (0 < abs_n < 1e-3):
            sol_str = f"{sol_num:.4g}"
        elif abs(sol_num - round(sol_num)) < 1e-6:
            sol_str = f"{int(round(sol_num))}"
        else:
            sol_str = f"{sol_num:.4f}".rstrip("0").rstrip(".")

        unit = ""
        desc = variables.get(unknowns[0].name, "")
        if isinstance(desc, str):
            m = re.search(r"\(([^)]+)\)", desc)
            if m:
                unit = " " + m.group(1).strip()

        return f"{unknowns[0].name} = {sol_str}{unit}"
    except Exception:
        return None


def _looks_mathematical(raw: str) -> bool:
    raw_low = raw.lower()
    if any(verb in raw_low for verb in _MATH_VERBS):
        return True
    if _MATH_OP_RE.search(raw_low):
        return True
    return False


def _freeform_compute(raw: str) -> str | None:
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, convert_xor,
        )
    except ImportError:
        return None

    transformations = standard_transformations + (convert_xor,)
    parse = lambda s: parse_expr(s, transformations=transformations)

    text = raw.strip().rstrip("?.").strip()
    text_low = text.lower()
    for prefix in (
        "find the value of ", "what is the value of ", "the value of ",
        "compute ", "solve ", "calculate ", "evaluate ", "simplify ",
        "integrate ", "differentiate ", "find ", "what is ", "what's ",
    ):
        if text_low.startswith(prefix):
            text = text[len(prefix):]
            text_low = text.lower()
            break
    text = re.sub(r"\b(\d+)\s+factorial\b", r"factorial(\1)", text, flags=re.IGNORECASE)
    text = text.strip()

    m = re.match(
        r"(?:the\s+)?integral\s+of\s+(.+?)\s+from\s+(\S+)\s+to\s+(.+)",
        text, re.IGNORECASE,
    )
    if m:
        try:
            f_expr = parse(m.group(1).strip())
            a_expr = parse(m.group(2).strip())
            b_expr = parse(m.group(3).strip())
            x = next(iter(f_expr.free_symbols), sp.Symbol("x"))
            res = sp.integrate(f_expr, (x, a_expr, b_expr))
            return f"integral = {res}"
        except Exception:
            pass

    m = re.match(
        r"(?:the\s+)?derivative\s+of\s+(.+?)(?:\s+with\s+respect\s+to\s+(\w+))?$",
        text, re.IGNORECASE,
    )
    if m:
        try:
            f_expr = parse(m.group(1).strip())
            x = sp.Symbol(m.group(2)) if m.group(2) else next(iter(f_expr.free_symbols), sp.Symbol("x"))
            res = sp.diff(f_expr, x)
            return f"d/d{x.name} = {res}"
        except Exception:
            pass

    if "=" in text and "==" not in text:
        try:
            lhs, rhs = text.split("=", 1)
            lhs_e = parse(lhs.strip())
            rhs_e = parse(rhs.strip())
            expr = lhs_e - rhs_e
            free = list(expr.free_symbols)
            if len(free) == 1:
                sols = sp.solve(expr, free[0])
                if sols:
                    out = sols[0] if len(sols) == 1 else sols
                    return f"{free[0].name} = {out}"
        except Exception:
            pass

    try:
        expr = parse(text)
        if expr.free_symbols:
            simplified = sp.simplify(expr)
            return f"= {simplified}"
        val = sp.N(expr)
        if hasattr(val, "is_Integer") and val.is_Integer:
            return f"= {int(val)}"
        return f"= {val}"
    except Exception:
        return None
