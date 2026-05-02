from __future__ import annotations

from pathlib import Path

from . import config as _config
from .schemas import EpistemicVector, FusedEvidence

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "confidence.pkl"


class ConfidenceEstimator:
    def __init__(self) -> None:
        self._model = None
        if MODEL_PATH.exists():
            import joblib
            self._model = joblib.load(MODEL_PATH)

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def predict(self, epistemic: EpistemicVector, evidence: FusedEvidence) -> float:
        if self._model is not None:
            from .features import confidence_features
            feats = confidence_features(epistemic, evidence).reshape(1, -1)
            return float(max(0.0, min(1.0, self._model.predict(feats)[0])))
        return _heuristic_confidence(epistemic, evidence)


def _heuristic_confidence(epi: EpistemicVector, ev: FusedEvidence) -> float:
    dominance = max(epi.g, epi.y, epi.r)
    scores = [r.score for r in ev.records] or [0.0]
    avg_score = sum(scores) / len(scores)
    penalty_factor = float(_config.get("confidence", "contradiction_penalty", 0.1))
    penalty = penalty_factor * len(ev.contradictions)
    return max(0.0, min(1.0, 0.5 * dominance + 0.5 * avg_score - penalty))
