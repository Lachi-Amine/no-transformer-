from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.schemas import Classification, EvidenceRecord, Query


class Engine(ABC):
    name: str = ""

    @abstractmethod
    def run(self, query: Query, cls: Classification) -> EvidenceRecord | None:
        ...


def load_knowledge(folder: Path) -> list[dict]:
    if not folder.exists():
        return []
    try:
        import yaml
    except ImportError:
        return []

    entries: list[dict] = []
    for path in sorted(folder.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            entries.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            entries.append(data)
    return entries
