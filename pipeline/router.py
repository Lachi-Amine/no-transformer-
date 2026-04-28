from __future__ import annotations

import json
from pathlib import Path

from .schemas import Classification, EpistemicVector, Query

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "epistemic_router.pt"
META_PATH = Path(__file__).resolve().parent.parent / "models" / "epistemic_router.json"

_DOMAIN_PRIORS: dict[str, tuple[float, float, float]] = {
    "math":       (0.85, 0.10, 0.05),
    "physics":    (0.75, 0.20, 0.05),
    "biology":    (0.20, 0.70, 0.10),
    "medicine":   (0.15, 0.75, 0.10),
    "economics":  (0.20, 0.65, 0.15),
    "history":    (0.10, 0.30, 0.60),
    "philosophy": (0.10, 0.20, 0.70),
    "general":    (0.34, 0.33, 0.33),
}


class EpistemicRouter:
    def __init__(self) -> None:
        self._model = None
        self._input_dim: int | None = None
        if MODEL_PATH.exists() and META_PATH.exists():
            self._load_torch_model()

    def _load_torch_model(self) -> None:
        import torch
        meta = json.loads(META_PATH.read_text())
        self._input_dim = int(meta["input_dim"])
        self._model = build_mlp(self._input_dim)
        state = torch.load(MODEL_PATH, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def predict(self, query: Query, cls: Classification) -> EpistemicVector:
        if self._model is not None:
            return self._predict_mlp(query, cls)
        return self._predict_heuristic(cls)

    def _predict_mlp(self, query: Query, cls: Classification) -> EpistemicVector:
        import torch
        from .features import router_features

        feats = router_features(query, cls)
        with torch.no_grad():
            x = torch.from_numpy(feats).float().unsqueeze(0)
            probs = self._model(x).squeeze(0).tolist()
        g, y, r = (float(p) for p in probs)
        s = g + y + r
        return EpistemicVector(g=g / s, y=y / s, r=r / s)

    @staticmethod
    def _predict_heuristic(cls: Classification) -> EpistemicVector:
        g, y, r = _DOMAIN_PRIORS.get(cls.domain, _DOMAIN_PRIORS["general"])
        return EpistemicVector(g=g, y=y, r=r)


def build_mlp(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 3),
        nn.Softmax(dim=-1),
    )
