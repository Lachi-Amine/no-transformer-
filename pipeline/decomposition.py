from __future__ import annotations

import re
from dataclasses import dataclass

_COMPARE_VERB_RE = re.compile(
    r"^\s*(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+?)[?.!]?\s*$",
    re.IGNORECASE,
)
_DIFFERENCE_RE = re.compile(
    r"^\s*(?:what(?:'s| is)?\s+the\s+|the\s+)?difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)[?.!]?\s*$",
    re.IGNORECASE,
)
_VS_RE = re.compile(
    r"^\s*(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]?\s*$",
    re.IGNORECASE,
)
_HOW_DIFFER_RE = re.compile(
    r"^\s*how\s+(?:do|does|are|is)\s+(.+?)\s+(?:and|differ\s+from|different\s+from)\s+(.+?)\s+differ[?.!]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decomposition:
    kind: str           # "compare"
    parts: tuple[str, ...]
    original: str


def detect(raw: str) -> Decomposition | None:
    if not raw or not raw.strip():
        return None
    for pat in (_COMPARE_VERB_RE, _DIFFERENCE_RE, _HOW_DIFFER_RE, _VS_RE):
        m = pat.match(raw)
        if not m:
            continue
        x = _clean(m.group(1))
        y = _clean(m.group(2))
        if not x or not y or x == y:
            continue
        if len(x) < 2 or len(y) < 2:
            continue
        return Decomposition(kind="compare", parts=(x, y), original=raw)
    return None


def _clean(part: str) -> str:
    s = part.strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t.?!,")
