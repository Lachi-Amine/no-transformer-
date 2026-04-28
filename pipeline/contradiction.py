from __future__ import annotations

import re

from .schemas import EvidenceRecord

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_NEGATION_PAIRS = [
    ("is", "is not"),
    ("does", "does not"),
    ("can", "cannot"),
    ("will", "will not"),
]


def detect(records: tuple[EvidenceRecord, ...]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if _contradicts(records[i].claim, records[j].claim):
                pairs.append((i, j))
    return tuple(pairs)


def _contradicts(a: str, b: str) -> bool:
    a_low, b_low = a.lower(), b.lower()

    a_nums = set(_NUMBER_RE.findall(a_low))
    b_nums = set(_NUMBER_RE.findall(b_low))
    if a_nums and b_nums and a_nums.isdisjoint(b_nums) and _share_topic(a_low, b_low):
        return True

    for pos, neg in _NEGATION_PAIRS:
        if (f" {pos} " in a_low and f" {neg} " in b_low and _share_topic(a_low, b_low)) or \
           (f" {neg} " in a_low and f" {pos} " in b_low and _share_topic(a_low, b_low)):
            return True
    return False


def _share_topic(a: str, b: str) -> bool:
    a_words = {w for w in re.findall(r"[a-z]{4,}", a)}
    b_words = {w for w in re.findall(r"[a-z]{4,}", b)}
    return len(a_words & b_words) >= 2
