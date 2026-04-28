from __future__ import annotations

import re
import unicodedata

from .schemas import Query

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def process(raw: str) -> Query:
    if not raw or not raw.strip():
        raise ValueError("empty query")

    normalized = unicodedata.normalize("NFKC", raw).strip().lower()
    tokens = tuple(_TOKEN_RE.findall(normalized))
    entities = _extract_entities(tokens)

    return Query(raw=raw, normalized=normalized, tokens=tokens, entities=entities)


def _extract_entities(tokens: tuple[str, ...]) -> dict[str, str]:
    entities: dict[str, str] = {}
    for i, t in enumerate(tokens):
        if t in {"enzyme", "protein", "molecule"} and i + 1 < len(tokens):
            entities[t] = tokens[i + 1]
        if t in {"force", "energy", "pressure"} and i + 1 < len(tokens):
            entities[t] = tokens[i + 1]
    return entities
