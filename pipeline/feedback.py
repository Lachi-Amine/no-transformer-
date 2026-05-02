from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "training" / "datasets" / "feedback.csv"

FIELDS = ["timestamp", "query", "label", "confidence_at_time", "rendered_excerpt"]
EXCERPT_LEN = 160


def record(
    query: str,
    label: float,
    rendered: str,
    confidence: float,
    path: Path = DEFAULT_PATH,
) -> None:
    if not (0.0 <= label <= 1.0):
        raise ValueError(f"label must be in [0,1], got {label}")
    if not query.strip():
        raise ValueError("empty query")

    excerpt = rendered.strip().split("\n", 1)[0][:EXCERPT_LEN]
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query": query.strip(),
        "label": f"{label:.3f}",
        "confidence_at_time": f"{confidence:.3f}",
        "rendered_excerpt": excerpt,
    }

    new_file = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_MINIMAL)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def load(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["label"] = float(r["label"])
                r["confidence_at_time"] = float(r["confidence_at_time"])
            except (KeyError, ValueError):
                continue
            rows.append(r)
    return rows
