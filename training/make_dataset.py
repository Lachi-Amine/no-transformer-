from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

DOMAINS = ["math", "physics", "chemistry", "biology", "medicine", "economics",
           "history", "philosophy", "general"]
INTENTS = ["define", "explain_process", "compute", "compare",
           "predict", "interpret", "summarize"]

DOMAIN_TARGET = 30
INTENT_TARGET = 30
EPISTEMIC_TARGET = 200


def _validate_classification(path: Path, labels: list[str], target: int) -> int:
    counts = Counter()
    issues = []
    rows = 0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += 1
            if r["label"] not in labels:
                issues.append(f"row {rows}: unknown label {r['label']!r}")
                continue
            counts[r["label"]] += 1
    print(f"\n=== {path.name} === total rows: {rows}")
    low = 0
    for label in labels:
        n = counts.get(label, 0)
        marker = "OK " if n >= target else "LOW"
        if n < target:
            low += 1
        print(f"  [{marker}] {label}: {n}")
    if issues:
        print(f"  issues ({len(issues)}):")
        for i in issues[:5]:
            print(f"    - {i}")
    return len(issues) + low


def _validate_epistemic(path: Path) -> int:
    rows = 0
    issues = []
    domain_counts = Counter()
    intent_counts = Counter()
    sum_min, sum_max = 1.0, 1.0
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += 1
            if r["domain"] not in DOMAINS:
                issues.append(f"row {rows}: bad domain {r['domain']!r}")
            if r["intent"] not in INTENTS:
                issues.append(f"row {rows}: bad intent {r['intent']!r}")
            try:
                g = float(r["g"]); y = float(r["y"]); rd = float(r["r"])
            except (KeyError, ValueError):
                issues.append(f"row {rows}: non-numeric g/y/r")
                continue
            s = g + y + rd
            sum_min, sum_max = min(sum_min, s), max(sum_max, s)
            if abs(s - 1.0) > 1e-3:
                issues.append(f"row {rows}: g+y+r = {s:.4f}")
            for name, val in (("g", g), ("y", y), ("r", rd)):
                if val < 0.0 or val > 1.0:
                    issues.append(f"row {rows}: {name}={val} out of [0,1]")
            domain_counts[r["domain"]] += 1
            intent_counts[r["intent"]] += 1
    print(f"\n=== {path.name} === total rows: {rows}")
    print(f"  sum(g+y+r) range: [{sum_min:.4f}, {sum_max:.4f}]")
    print("  domain coverage:")
    for d in DOMAINS:
        print(f"    {d}: {domain_counts.get(d, 0)}")
    print("  intent coverage:")
    for i in INTENTS:
        print(f"    {i}: {intent_counts.get(i, 0)}")
    if issues:
        print(f"  issues ({len(issues)}):")
        for i in issues[:10]:
            print(f"    - {i}")
    if rows < EPISTEMIC_TARGET:
        print(f"  [LOW] total rows {rows} < target {EPISTEMIC_TARGET}")
        return len(issues) + 1
    return len(issues)


def main() -> int:
    fail = 0
    fail += _validate_classification(DATASETS / "domains.csv", DOMAINS, DOMAIN_TARGET)
    fail += _validate_classification(DATASETS / "intents.csv", INTENTS, INTENT_TARGET)
    fail += _validate_epistemic(DATASETS / "epistemic.csv")
    print()
    if fail:
        print(f"FAIL: {fail} issue(s)")
        return 1
    print("OK: all datasets validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
