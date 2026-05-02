from __future__ import annotations

import argparse
import sys
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import feedback as feedback_mod
from pipeline.orchestrator import Pipeline
from pipeline.schemas import Response

HISTORY_FILE = Path.home() / ".no_transformer_history"
SLASH_COMMANDS = [
    ":help", ":status", ":debug on", ":debug off", ":reload", ":why",
    ":history", ":forget", ":good", ":bad", ":rate ", ":bench ", ":exit",
]


HELP_TEXT = """\
Commands:
  :help          show this help
  :status        show which models are loaded vs stubbed
  :debug on/off  toggle full pipeline trace
  :reload        re-read trained models from disk
  :why           explain the last response (evidence + contradictions)
  :history       show the last few questions and answers
  :forget        clear the in-session conversation memory
  :good          mark the last response as correct (logs to feedback.csv)
  :bad           mark the last response as wrong   (logs to feedback.csv)
  :rate N        rate the last response 1-5        (logs to feedback.csv)
  :bench QUERY   print per-stage latency for QUERY (does not log to history)
  :exit          quit (Ctrl-D also works)
Type any other text to ask a question.
Pronouns (it, that, this, they) in your question are resolved against the
previous question's main topic.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="no-transformer", add_help=True)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="suppress non-fatal warnings (e.g. sklearn version skew)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show all warnings (overrides --quiet)")
    args = parser.parse_args(argv)

    if args.quiet and not args.verbose:
        warnings.filterwarnings("ignore")

    read_line = _make_reader()

    pipeline = Pipeline()
    debug = False
    last: Response | None = None

    print("no-transformer reasoning CLI")
    print("type :help for commands\n")
    _print_status(pipeline)

    while True:
        try:
            line = read_line().strip()
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
            if cmd == "history":
                _print_history(pipeline)
                continue
            if cmd == "forget":
                pipeline.forget()
                print("conversation memory cleared.")
                continue
            if cmd == "bench":
                if not arg:
                    print("usage: :bench QUERY")
                    continue
                _print_bench(pipeline, arg)
                continue
            if cmd in {"good", "bad", "rate"}:
                if last is None:
                    print("no previous response to rate.")
                    continue
                if cmd == "good":
                    label = 1.0
                elif cmd == "bad":
                    label = 0.0
                else:
                    try:
                        n = int(arg)
                    except ValueError:
                        print("usage: :rate N  where N is 1..5")
                        continue
                    if not 1 <= n <= 5:
                        print("rating must be between 1 and 5")
                        continue
                    label = (n - 1) / 4.0
                try:
                    feedback_mod.record(
                        query=last.query.raw,
                        label=label,
                        rendered=last.rendered,
                        confidence=last.confidence,
                    )
                except Exception as exc:
                    print(f"error logging feedback: {exc}")
                    continue
                print(f"logged: label={label:.2f} (confidence at time was {last.confidence:.2f})")
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


def _make_reader():
    """Return a callable that reads one line of input. Uses prompt_toolkit if
    available (file-backed history + tab completion on slash commands); falls
    back to plain input()."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
    except ImportError:
        return lambda: input("> ")

    completer = WordCompleter(SLASH_COMMANDS, ignore_case=True, sentence=True)
    try:
        session = PromptSession(history=FileHistory(str(HISTORY_FILE)), completer=completer)
    except Exception:
        return lambda: input("> ")

    def read():
        return session.prompt("> ")
    return read


def _print_status(pipeline: Pipeline) -> None:
    print("model status:")
    for name, state in pipeline.status().items():
        marker = "[ok]" if state == "trained" else "[stub]"
        print(f"  {marker} {name}: {state}")
    print()


def _print_history(pipeline: Pipeline) -> None:
    if not pipeline.history:
        print("no history yet.")
        return
    print(f"history ({len(pipeline.history)} of {pipeline._max_history} turns):")
    for i, resp in enumerate(pipeline.history):
        first_line = resp.rendered.split("\n", 1)[0]
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        print(f"  [{i}] Q: {resp.query.raw}")
        print(f"      A: {first_line}")
    print()


def _print_bench(pipeline: Pipeline, query: str) -> None:
    try:
        timings = pipeline.bench(query)
    except Exception as exc:
        print(f"error: {exc}")
        return
    total_ms = timings["total"] * 1000
    print(f"\nbench({query!r}) — total {total_ms:.1f} ms")
    for stage, secs in timings.items():
        if stage == "total":
            continue
        ms = secs * 1000
        pct = 100 * secs / timings["total"] if timings["total"] else 0
        bar = "#" * int(pct / 2)
        print(f"  {stage:20s} {ms:6.1f} ms  {pct:5.1f}%  {bar}")
    print()


def _print_response(resp: Response, debug: bool) -> None:
    cls = resp.classification
    epi = resp.epistemic
    print()
    coref = resp.debug.get("coref")
    if coref is not None:
        print(f"(resolved pronoun -> '{coref['topic']}' from previous query)")
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
    cls = resp.classification
    epi = resp.epistemic

    print()
    print(f"query: {resp.query.raw}")
    print(f"normalized: {resp.query.normalized}")
    print(f"tokens: {list(resp.query.tokens)}")
    if resp.query.entities:
        print(f"entities: {resp.query.entities}")

    coref = resp.debug.get("coref")
    if coref is not None:
        print(f"coref: pronoun resolved to '{coref['topic']}' (from: {coref['from_query']!r})")
        print(f"       expanded to: {coref['expanded']!r}")
    print()

    print(f"domain: {cls.domain}")
    if cls.domain_probs:
        top_domains = sorted(cls.domain_probs.items(), key=lambda kv: -kv[1])[:3]
        for name, prob in top_domains:
            marker = " <-- chosen" if name == cls.domain else ""
            print(f"  {prob:.3f}  {name}{marker}")

    print(f"intent: {cls.intent}")
    if cls.intent_probs:
        top_intents = sorted(cls.intent_probs.items(), key=lambda kv: -kv[1])[:3]
        for name, prob in top_intents:
            marker = " <-- chosen" if name == cls.intent else ""
            print(f"  {prob:.3f}  {name}{marker}")

    print(f"epistemic: g={epi.g:.3f}  y={epi.y:.3f}  r={epi.r:.3f}")
    print()

    engine_status = resp.debug.get("engine_status", {})
    weights = epi.as_weights()
    if engine_status:
        print("engines:")
        for engine_name in ("green", "yellow", "red"):
            state = engine_status.get(engine_name, "not-called")
            weight = weights.get(engine_name, 0.0)
            print(f"  {engine_name:6}  weight={weight:.2f}  status={state}")
        print()

    if not resp.evidence.records:
        print("no evidence records were produced.")
    else:
        print(f"evidence ({len(resp.evidence.records)} record(s)):")
        for i, rec in enumerate(resp.evidence.records):
            print(f"  [{i}] {rec.engine} (score={rec.score:.3f})")
            print(f"      claim: {rec.claim}")
            if rec.support:
                print(f"      support: {list(rec.support)}")
        print()

    if resp.evidence.contradictions:
        print(f"contradictions ({len(resp.evidence.contradictions)}):")
        for i, j in resp.evidence.contradictions:
            if i < len(resp.evidence.records) and j < len(resp.evidence.records):
                a = resp.evidence.records[i]
                b = resp.evidence.records[j]
                print(f"  - [{a.engine}] {a.claim}")
                print(f"  - [{b.engine}] {b.claim}")
        print()

    print(f"confidence: {resp.confidence:.3f}")
    print()


if __name__ == "__main__":
    sys.exit(main())
