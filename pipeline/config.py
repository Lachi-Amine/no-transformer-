from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config.yaml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "yellow": {
        "bm25_threshold": 1.0,
        "bm25_normalizer": 10.0,
        "domain_fallback_ratio": 2.0,
        "secondary_floor_ratio": 0.65,
        "max_passages": 3,
    },
    "fusion": {
        "min_engine_weight": 0.05,
    },
    "orchestrator": {
        "history_size": 3,
    },
    "confidence": {
        "contradiction_penalty": 0.1,
    },
}

_cached: dict[str, Any] | None = None


def load(path: Path = DEFAULT_PATH) -> dict[str, Any]:
    """Return the merged config (file overrides defaults). Cached after first call."""
    global _cached
    if _cached is not None:
        return _cached

    merged: dict[str, Any] = {k: dict(v) for k, v in DEFAULTS.items()}
    if path.exists():
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for section, overrides in data.items():
                if section in merged and isinstance(overrides, dict):
                    merged[section].update(overrides)
        except Exception:
            pass

    _cached = merged
    return merged


def reset_cache() -> None:
    """Drop the cached config (useful in tests)."""
    global _cached
    _cached = None


def get(section: str, key: str, default: Any = None) -> Any:
    cfg = load()
    return cfg.get(section, {}).get(key, default)
