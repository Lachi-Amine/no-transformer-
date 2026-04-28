from __future__ import annotations

from typing import Iterable

from engines.base import Engine

from . import contradiction
from .schemas import (
    Classification,
    EpistemicVector,
    EvidenceRecord,
    FusedEvidence,
    Query,
)

_MIN_WEIGHT = 0.05


def fuse(
    query: Query,
    cls: Classification,
    epistemic: EpistemicVector,
    engines: Iterable[Engine],
) -> tuple[FusedEvidence, dict[str, str]]:
    weights = epistemic.as_weights()
    fired: list[EvidenceRecord] = []
    engine_status: dict[str, str] = {}

    for engine in engines:
        w = weights.get(engine.name, 0.0)
        if w < _MIN_WEIGHT:
            engine_status[engine.name] = "skipped"
            continue
        record = engine.run(query, cls)
        if record is None:
            engine_status[engine.name] = "no-result"
            continue
        weighted = EvidenceRecord(
            engine=record.engine,
            claim=record.claim,
            support=record.support,
            score=record.score * w,
        )
        fired.append(weighted)
        engine_status[engine.name] = "fired"

    contradictions = contradiction.detect(tuple(fired))
    return FusedEvidence(records=tuple(fired), contradictions=contradictions), engine_status
