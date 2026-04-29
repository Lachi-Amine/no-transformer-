from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import Pipeline
from pipeline.schemas import Response


HELP_TEXT = """\
Commands:
  :help          show this help
  :status        show which models are loaded vs stubbed
  :debug on/off  toggle full pipeline trace
  :reload        re-read trained models from disk
  :why           explain the last response (evidence + contradictions)
  :exit          quit (Ctrl-D also works)
Type any other text to ask a question.
"""


def main() -> int:
    pipeline = Pipeline()
    debug = False
    last: Response | None = None

    print("no-transformer reasoning CLI")
    print("type :help for commands\n")
    _print_status(pipeline)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue

        if line.startswith(":"):
            cmd, _, arg = line[1:].partition(" ")
            cmd = cmd.lower().strip()
            arg = arg.strip().lower()

            if cmd in {"exit", "quit"}:
                return 0
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "status":
                _print_status(pipeline)
                continue
            if cmd == "debug":
                if arg in {"on", "true", "1"}:
                    debug = True
                elif arg in {"off", "false", "0"}:
                    debug = False
                else:
                    debug = not debug
                print(f"debug: {'on' if debug else 'off'}")
                continue
            if cmd == "reload":
                pipeline.reload()
                print("models reloaded.")
                _print_status(pipeline)
                continue
            if cmd == "why":
                if last is None:
                    print("no previous response yet.")
                else:
                    _explain(last)
                continue
            print(f"unknown command: :{cmd}  (try :help)")
            continue

        try:
            last = pipeline.run(line)
        except Exception as exc:
            print(f"error: {exc}")
            if debug:
                traceback.print_exc()
            continue

        _print_response(last, debug)


def _print_status(pipeline: Pipeline) -> None:
    print("model status:")
    for name, state in pipeline.status().items():
        marker = "[ok]" if state == "trained" else "[stub]"
        print(f"  {marker} {name}: {state}")
    print()


def _print_response(resp: Response, debug: bool) -> None:
    cls = resp.classification
    epi = resp.epistemic
    print()
    print(f"domain: {cls.domain}   intent: {cls.intent}")
    print(f"epistemic: g={epi.g:.2f}  y={epi.y:.2f}  r={epi.r:.2f}")

    fired = [n for n, s in resp.debug.get("engine_status", {}).items() if s == "fired"]
    skipped = [n for n, s in resp.debug.get("engine_status", {}).items() if s == "skipped"]
    print(f"engines fired: {', '.join(fired) or 'none'}   skipped: {', '.join(skipped) or 'none'}")
    print(f"confidence: {resp.confidence:.2f}")
    print()
    print(resp.rendered)
    print()

    if debug:
        print("--- debug ---")
        print(f"normalized: {resp.query.normalized}")
        print(f"tokens: {list(resp.query.tokens)}")
        print(f"entities: {resp.query.entities}")
        print(f"records: {len(resp.evidence.records)}")
        for i, rec in enumerate(resp.evidence.records):
            print(f"  [{i}] {rec.engine} score={rec.score:.3f}: {rec.claim}")
        print(f"contradictions: {list(resp.evidence.contradictions)}")
        print()


def _explain(resp: Response) -> None:
    print()
    print(f"query: {resp.query.raw}")
    print(f"reason for routing: domain={resp.classification.domain}, intent={resp.classification.intent}")
    print(f"epistemic blend: g={resp.epistemic.g:.2f}, y={resp.epistemic.y:.2f}, r={resp.epistemic.r:.2f}")
    if not resp.evidence.records:
        print("no evidence records were produced (engines are stubs in M1).")
    else:
        print("evidence:")
        for i, rec in enumerate(resp.evidence.records):
            print(f"  [{i}] ({rec.engine}, score={rec.score:.3f}) {rec.claim}")
            for s in rec.support:
                print(f"      - {s}")
    if resp.evidence.contradictions:
        print(f"contradictions between records: {list(resp.evidence.contradictions)}")
    print()


if __name__ == "__main__":
    sys.exit(main())
