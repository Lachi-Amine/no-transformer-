from __future__ import annotations

from pathlib import Path

from .schemas import Classification, EpistemicVector, FusedEvidence, Query

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TFIDF_PATH = MODELS_DIR / "tfidf.pkl"

_ROUTER_FEATURE_DIM = 24
_CONFIDENCE_FEATURE_DIM = 8


def query_text_features(text: str):
    import numpy as np
    if TFIDF_PATH.exists():
        import joblib
        vec = joblib.load(TFIDF_PATH)
        return vec.transform([text]).toarray()[0]
    return np.zeros(_ROUTER_FEATURE_DIM, dtype=np.float32)


def router_features(query: Query, cls: Classification):
    import numpy as np
    from .schemas import DOMAINS, INTENTS

    domain_one_hot = np.zeros(len(DOMAINS), dtype=np.float32)
    if cls.domain in DOMAINS:
        domain_one_hot[DOMAINS.index(cls.domain)] = 1.0

    intent_one_hot = np.zeros(len(INTENTS), dtype=np.float32)
    if cls.intent in INTENTS:
        intent_one_hot[INTENTS.index(cls.intent)] = 1.0

    n_tokens = len(query.tokens)
    avg_token_len = float(np.mean([len(t) for t in query.tokens])) if n_tokens else 0.0
    n_entities = float(len(query.entities))
    has_question_mark = 1.0 if "?" in query.raw else 0.0

    stats = np.array(
        [n_tokens / 50.0, avg_token_len / 10.0, n_entities / 5.0, has_question_mark],
        dtype=np.float32,
    )
    return np.concatenate([domain_one_hot, intent_one_hot, stats])


def confidence_features(epi: EpistemicVector, ev: FusedEvidence):
    import numpy as np
    scores = [r.score for r in ev.records] or [0.0]
    n_contradictions = float(len(ev.contradictions))
    return np.array(
        [
            epi.g, epi.y, epi.r,
            float(max(epi.g, epi.y, epi.r)),
            float(np.mean(scores)),
            float(np.max(scores)),
            float(len(ev.records)),
            n_contradictions,
        ],
        dtype=np.float32,
    )
