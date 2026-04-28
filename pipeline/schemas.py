from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

DOMAINS = [
    "math", "physics", "biology", "medicine", "economics",
    "history", "philosophy", "general",
]

INTENTS = [
    "define", "explain_process", "compute", "compare",
    "predict", "interpret", "summarize",
]

ENGINES = ("green", "yellow", "red")


@dataclass(frozen=True)
class Query:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    entities: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EpistemicVector:
    g: float
    y: float
    r: float

    def __post_init__(self) -> None:
        total = self.g + self.y + self.r
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"epistemic vector must sum to 1.0, got {total}")
        for name, val in (("g", self.g), ("y", self.y), ("r", self.r)):
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name}={val} out of [0,1]")

    def as_weights(self) -> dict[str, float]:
        return {"green": self.g, "yellow": self.y, "red": self.r}


@dataclass(frozen=True)
class Classification:
    domain: str
    intent: str
    domain_probs: dict[str, float]
    intent_probs: dict[str, float]


@dataclass(frozen=True)
class EvidenceRecord:
    engine: str
    claim: str
    support: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class FusedEvidence:
    records: tuple[EvidenceRecord, ...]
    contradictions: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Response:
    query: Query
    classification: Classification
    epistemic: EpistemicVector
    evidence: FusedEvidence
    confidence: float
    rendered: str
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
