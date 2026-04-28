from __future__ import annotations

from pathlib import Path

from .schemas import INTENTS, Classification, Query

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_clf.pkl"

_INTENT_PATTERNS = {
    "define": {"what", "define", "definition", "meaning"},
    "explain_process": {"how", "explain", "process", "mechanism", "work", "works"},
    "compute": {"calculate", "compute", "solve", "value"},
    "compare": {"compare", "versus", "vs", "difference", "between"},
    "predict": {"will", "predict", "forecast", "future", "expect"},
    "interpret": {"why", "meaning", "interpret", "significance"},
    "summarize": {"summary", "summarize", "overview", "brief"},
}


class IntentClassifier:
    def __init__(self) -> None:
        self._pipeline = None
        if MODEL_PATH.exists():
            import joblib
            self._pipeline = joblib.load(MODEL_PATH)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def predict(self, query: Query, prior: Classification) -> Classification:
        if self._pipeline is not None:
            probs_arr = self._pipeline.predict_proba([query.normalized])[0]
            classes = list(self._pipeline.classes_)
            intent_probs = {c: float(p) for c, p in zip(classes, probs_arr)}
            intent = max(intent_probs, key=intent_probs.get)
        else:
            intent, intent_probs = _heuristic_intent(query)

        return Classification(
            domain=prior.domain,
            intent=intent,
            domain_probs=prior.domain_probs,
            intent_probs=intent_probs,
        )


def _heuristic_intent(query: Query) -> tuple[str, dict[str, float]]:
    scores = {i: 0.0 for i in INTENTS}
    tokens = set(query.tokens)
    for intent, kws in _INTENT_PATTERNS.items():
        scores[intent] = float(len(tokens & kws))
    total = sum(scores.values())
    if total == 0:
        scores["explain_process"] = 1.0
        total = 1.0
    probs = {i: s / total for i, s in scores.items()}
    intent = max(probs, key=probs.get)
    return intent, probs
