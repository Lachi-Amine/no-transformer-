from __future__ import annotations

from pathlib import Path

from .schemas import DOMAINS, Classification, Query

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "domain_clf.pkl"

_DOMAIN_KEYWORDS = {
    "math": {"equation", "integral", "derivative", "theorem", "proof", "matrix",
             "polynomial", "quadratic", "geometry", "triangle", "hypotenuse", "series"},
    "physics": {"force", "energy", "mass", "velocity", "quantum", "electron", "field",
                "gravity", "pendulum", "period", "oscillation", "wave", "frequency",
                "voltage", "current", "resistance", "pressure", "momentum", "kepler",
                "ohm", "coulomb", "newton", "snell", "doppler",
                "tectonic", "volcano", "earthquake", "rock", "magma"},
    "chemistry": {"acid", "base", "ph", "molecule", "atom", "ion", "ionic", "covalent",
                  "bond", "redox", "oxidation", "reduction", "catalyst", "isotope",
                  "polymer", "stoichiometry", "equilibrium", "henderson", "beer",
                  "lambert", "avogadro", "periodic", "electron", "le", "chatelier"},
    "biology": {"cell", "enzyme", "dna", "rna", "protein", "organism", "evolution",
                "amylase", "photosynthesis", "mitosis", "respiration", "neuron",
                "antibody", "insulin", "mendel", "neuroplasticity", "synapse",
                "neurotransmitter", "dopamine", "serotonin", "memory"},
    "medicine": {"disease", "symptom", "drug", "patient", "diagnosis", "vaccine",
                 "antibiotic", "diabetes", "hypertension", "influenza", "penicillin",
                 "aspirin", "cancer"},
    "economics": {"market", "inflation", "supply", "demand", "price", "gdp",
                  "recession", "interest", "compound", "advantage", "opportunity",
                  "federal", "stock", "easing"},
    "history": {"war", "empire", "century", "revolution", "treaty", "ancient",
                "dynasty", "colonial", "renaissance"},
    "philosophy": {"ethics", "moral", "consciousness", "epistemology", "ontology",
                   "virtue", "trolley", "deontological", "utilitarian",
                   "determinism", "freewill", "dualist"},
}


class DomainClassifier:
    def __init__(self) -> None:
        self._pipeline = None
        if MODEL_PATH.exists():
            import joblib
            self._pipeline = joblib.load(MODEL_PATH)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def predict(self, query: Query) -> Classification:
        if self._pipeline is not None:
            probs_arr = self._pipeline.predict_proba([query.normalized])[0]
            classes = list(self._pipeline.classes_)
            domain_probs = {c: float(p) for c, p in zip(classes, probs_arr)}
            domain = max(domain_probs, key=domain_probs.get)
        else:
            domain, domain_probs = _heuristic_domain(query)

        return Classification(
            domain=domain,
            intent="",
            domain_probs=domain_probs,
            intent_probs={},
        )


def _heuristic_domain(query: Query) -> tuple[str, dict[str, float]]:
    scores = {d: 0.0 for d in DOMAINS}
    for d, kws in _DOMAIN_KEYWORDS.items():
        for tok in query.tokens:
            if tok in kws:
                scores[d] += 1.0
    total = sum(scores.values())
    if total == 0:
        scores["general"] = 1.0
        total = 1.0
    probs = {d: s / total for d, s in scores.items()}
    domain = max(probs, key=probs.get)
    return domain, probs
