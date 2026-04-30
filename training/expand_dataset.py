from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "knowledge"
OUT = ROOT / "training" / "datasets"

DOMAIN_FOR_FORMAL_OR_INTERPRETIVE = {
    "interpretive": "philosophy",
}

EXPLAIN_TEMPLATES_BY_FOLDER = {
    "empirical": ["how does {kw} work", "explain {kw}"],
    "formal":    [],
    "interpretive": ["why is {kw} contested", "is {kw} morally right"],
}

INTENT_TEMPLATES_BY_FOLDER = {
    "empirical":    [("what is {kw}", "define"),
                     ("how does {kw} work", "explain_process"),
                     ("explain {kw}", "explain_process")],
    "formal":       [("what is {kw}", "define"),
                     ("compute {kw}", "compute"),
                     ("find {kw}", "compute")],
    "interpretive": [("what is {kw}", "define"),
                     ("why is {kw} contested", "interpret"),
                     ("is {kw} justified", "interpret")],
}


def _load_yaml_entries(folder: Path) -> list[dict]:
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


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    sent = text.strip().split(".")[0].strip()
    sent = " ".join(sent.split())
    return sent


def _safe_csv(text: str) -> str:
    return text.replace(",", "").strip()


def main() -> int:
    domain_rows: list[tuple[str, str]] = []
    intent_rows: list[tuple[str, str]] = []

    for folder_name in ("formal", "empirical", "interpretive"):
        folder = KNOWLEDGE / folder_name
        for entry in _load_yaml_entries(folder):
            domain = entry.get("domain") or DOMAIN_FOR_FORMAL_OR_INTERPRETIVE.get(folder_name)
            if not domain:
                continue
            keywords = entry.get("keywords") or []
            text = entry.get("text") or ""

            if keywords:
                primary = keywords[0]
                domain_rows.append((f"what is {primary}", domain))
                for tmpl in EXPLAIN_TEMPLATES_BY_FOLDER.get(folder_name, []):
                    domain_rows.append((tmpl.format(kw=primary), domain))
                if len(keywords) >= 2:
                    domain_rows.append((f"what is {keywords[1]}", domain))

            sent = _first_sentence(text)
            if sent and len(sent) < 200:
                domain_rows.append((_safe_csv(sent)[:160], domain))

            if keywords:
                primary = keywords[0]
                for tmpl, intent in INTENT_TEMPLATES_BY_FOLDER.get(folder_name, []):
                    intent_rows.append((tmpl.format(kw=primary), intent))

    OUT.mkdir(parents=True, exist_ok=True)

    seen = set()
    deduped_domain = []
    for t, l in domain_rows:
        key = (t.lower(), l)
        if key in seen:
            continue
        seen.add(key)
        deduped_domain.append((t, l))

    seen = set()
    deduped_intent = []
    for t, l in intent_rows:
        key = (t.lower(), l)
        if key in seen:
            continue
        seen.add(key)
        deduped_intent.append((t, l))

    with (OUT / "domains_auto.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(deduped_domain)

    with (OUT / "intents_auto.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(deduped_intent)

    print(f"domains_auto.csv: {len(deduped_domain)} rows")
    print(f"intents_auto.csv: {len(deduped_intent)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
