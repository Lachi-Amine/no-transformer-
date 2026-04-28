# Implementation Map

This is the build map for the Epistemic Multi-Hypothesis Neuro-Symbolic Reasoning system described in `architecture.md`. It is the source of truth we follow until training is done and the CLI is wired up.

---

## 0. Goals & Constraints

- **Interface**: a single CLI (REPL). No GUI, no web server.
- **No transformer** anywhere in the learned components. We use classical ML (sklearn) and small MLPs (PyTorch) only.
- **No external API calls** for response generation. The final response is produced by a deterministic template formatter.
- **Training happens on Google Colab, never on the user's laptop.** Local code does inference and rule-based logic only. Whenever a step needs training, we stop, tell the user which Colab cell to run, and wait for the artifact.
- The pipeline must be **runnable end-to-end from day one** using stubs, so we can develop the CLI and engines in parallel with training.
- Trained artifacts (`.pkl`, `.pt`) are dropped into `models/` and hot-loaded by the CLI via a `:reload` command.
- All inputs/outputs flowing between modules are typed dataclasses defined in one shared `schemas.py` so training and inference cannot drift.
- **Stack**: Python 3.11, `numpy`, `scikit-learn`, `torch` (CPU, inference only), `sympy`, `rank-bm25`, `pyyaml`, `prompt_toolkit`. Models stored as plain files in `models/`.

---

## 1. Project Layout

```
no-transformer/
├── cli.py                          # REPL entry point
├── requirements.txt
├── README.md
├── map.md                          # this file
├── architecture.md
│
├── pipeline/
│   ├── __init__.py
│   ├── schemas.py                  # all dataclasses + JSON (de)serializers
│   ├── features.py                 # text → feature vectors (TF-IDF, stats)
│   ├── query_processing.py         # validate, normalize, entity extract
│   ├── domain_classifier.py        # wraps domain_clf.pkl
│   ├── intent_classifier.py        # wraps intent_clf.pkl
│   ├── router.py                   # wraps epistemic_router.pt
│   ├── confidence.py               # wraps confidence.pkl
│   ├── contradiction.py            # rule + small classifier
│   ├── fusion.py                   # weighted engine blend
│   └── orchestrator.py             # full pipeline runner
│
├── engines/
│   ├── __init__.py
│   ├── base.py                     # Engine ABC, EvidenceRecord
│   ├── green_symbolic.py           # sympy
│   ├── yellow_retrieval.py         # BM25 / TF-IDF retrieval
│   └── red_synthesis.py            # template + retrieval combiner
│
├── knowledge/
│   ├── formal/                     # equations, axioms (.json/.yaml)
│   ├── empirical/                  # facts, datasets (.json/.yaml)
│   └── interpretive/               # narratives, perspectives (.json/.yaml)
│
├── models/                         # empty until Colab training is done
│   ├── domain_clf.pkl
│   ├── intent_clf.pkl
│   ├── epistemic_router.pt
│   ├── confidence.pkl
│   └── manifest.json               # version, train date, metrics per model
│
├── training/
│   ├── colab_train.ipynb           # ONE notebook, four training cells
│   ├── schemas.py                  # symlinked to pipeline/schemas.py
│   ├── features.py                 # symlinked to pipeline/features.py
│   ├── make_dataset.py             # builds CSVs from raw seeds
│   └── datasets/
│       ├── domains.csv             # text,label
│       ├── intents.csv             # text,label
│       ├── epistemic.csv           # text,domain,intent,g,y,r
│       └── confidence.csv          # features...,confidence
│
└── tests/
    ├── test_schemas.py
    ├── test_engines.py
    ├── test_pipeline_stubs.py
    └── fixtures/
```

Files marked "symlinked" are kept as symlinks so training and inference share one source of truth. On Windows we copy and add a CI check that the copies are identical.

---

## 2. Shared Schemas (`pipeline/schemas.py`)

Every dataclass below is `@dataclass(frozen=True)` and has `.to_json()` / `.from_json()`.

```python
@dataclass(frozen=True)
class Query:
    raw: str
    normalized: str
    tokens: list[str]
    entities: dict[str, str]        # e.g. {"enzyme": "amylase"}

@dataclass(frozen=True)
class EpistemicVector:
    g: float                         # GREEN  — formal certainty
    y: float                         # YELLOW — empirical uncertainty
    r: float                         # RED    — interpretive flexibility
    # invariant: g + y + r == 1.0 (within 1e-6)

@dataclass(frozen=True)
class Classification:
    domain: str                      # one of DOMAINS
    intent: str                      # one of INTENTS
    domain_probs: dict[str, float]
    intent_probs: dict[str, float]

@dataclass(frozen=True)
class EvidenceRecord:
    engine: str                      # "green" | "yellow" | "red"
    claim: str
    support: list[str]               # citations / equations / passages
    score: float                     # engine-internal confidence in [0,1]

@dataclass(frozen=True)
class FusedEvidence:
    records: list[EvidenceRecord]
    contradictions: list[tuple[int, int]]   # indexes into records

@dataclass(frozen=True)
class Response:
    query: Query
    classification: Classification
    epistemic: EpistemicVector
    evidence: FusedEvidence
    confidence: float
    rendered: str                    # final user-facing text
    debug: dict                      # everything the CLI may want to print
```

Closed vocabularies (start small, expand later):

```
DOMAINS  = ["math", "physics", "biology", "medicine", "economics",
            "history", "philosophy", "general"]
INTENTS  = ["define", "explain_process", "compute", "compare",
            "predict", "interpret", "summarize"]
```

---

## 3. Feature Extraction (`pipeline/features.py`)

One function per learned module, each returning a fixed-shape numpy array. **Both Colab and the CLI call these same functions.**

```python
def query_text_features(text: str) -> np.ndarray:        # TF-IDF, fitted vectorizer loaded from models/
def router_features(query: Query, cls: Classification) -> np.ndarray:
def confidence_features(epistemic: EpistemicVector,
                        evidence: FusedEvidence) -> np.ndarray:
```

The TF-IDF vectorizer itself is trained on Colab and saved as `models/tfidf.pkl`. `query_text_features` loads it on first call.

---

## 4. Module Specs

For each learned module: **inputs → outputs**, model, training data, suggested hyperparameters, and the artifact it writes.

### 4.1 Domain Classifier — `pipeline/domain_classifier.py`

- **In**: `Query`
- **Out**: `domain: str`, `domain_probs: dict[str,float]`
- **Model**: `TfidfVectorizer(ngram_range=(1,2), min_df=2)` + `LogisticRegression(max_iter=1000, class_weight="balanced")`
- **Training data**: `training/datasets/domains.csv` — columns `text,label`. Seed with ~50 examples per domain (we hand-write or scrape Wikipedia summaries).
- **Artifact**: `models/domain_clf.pkl` (a sklearn `Pipeline`)
- **Stub behavior** (before training): always returns `"general"` with uniform probs.

### 4.2 Intent Classifier — `pipeline/intent_classifier.py`

- Same shape as domain classifier.
- **Training data**: `training/datasets/intents.csv` — ~30 examples per intent. We can synthesize patterns ("what is …" → define, "how does … work" → explain_process, etc.) and let the model generalize.
- **Artifact**: `models/intent_clf.pkl`
- **Stub behavior**: returns `"explain_process"` with uniform probs.

### 4.3 Epistemic Router — `pipeline/router.py`

This is the centerpiece.

- **In**: `Query`, `Classification`
- **Out**: `EpistemicVector(g, y, r)`
- **Model**: small PyTorch MLP, **no transformer**:
  ```
  Input:   features.router_features(...)  shape (D,)
  Hidden:  Linear(D → 64) → ReLU → Dropout(0.2)
           Linear(64 → 32) → ReLU
  Head:    Linear(32 → 3) → Softmax
  ```
- **Loss**: cross-entropy on soft labels (we predict a distribution, not a class), implemented as `-sum(target * log(pred))`.
- **Training data**: `training/datasets/epistemic.csv` — columns `text,domain,intent,g,y,r`. Each row's `g+y+r == 1`. Seed with ~200 hand-labeled rows spanning domain × intent combinations.
- **Hyperparameters**: Adam, lr=1e-3, batch=32, epochs=30, early stop on val loss.
- **Artifact**: `models/epistemic_router.pt` (state_dict + a tiny JSON sidecar with input dim and architecture hash).
- **Stub behavior**: heuristic mapping by domain — math/physics → (0.8,0.15,0.05), biology/medicine/economics → (0.2,0.7,0.1), history/philosophy → (0.1,0.2,0.7), general → (0.33,0.34,0.33).

### 4.4 Confidence Estimator — `pipeline/confidence.py`

- **In**: `EpistemicVector`, `FusedEvidence` (post-fusion)
- **Out**: `confidence: float ∈ [0,1]`
- **Model**: sklearn `GradientBoostingRegressor(max_depth=3, n_estimators=200)` — small enough to ship.
- **Training data**: `training/datasets/confidence.csv` — generated semi-synthetically by running the pipeline on labeled QA pairs and computing target confidence as accuracy on a held-out reference. Seed with whatever labeled pairs we have; expand later.
- **Artifact**: `models/confidence.pkl`
- **Stub behavior**: `0.5 * (max(g,y,r)) + 0.5 * mean(record.score for record in evidence.records)`.

### 4.5 Contradiction Detector — `pipeline/contradiction.py`

Hybrid: rule pass first, then an optional small classifier for the leftovers.

- Rule pass: numerical disagreement, direct negation patterns, mutually exclusive entity assertions.
- Classifier pass (optional, train later): MLP over pairwise TF-IDF features → `{contradicts, neutral, supports}`.
- **Artifact**: `models/contradiction.pkl` (optional, can ship without).

---

## 5. Engines (no training)

All engines implement `engines/base.py`:

```python
class Engine(ABC):
    name: str
    def run(self, query: Query, cls: Classification) -> EvidenceRecord: ...
```

### 5.1 GREEN — `engines/green_symbolic.py`

- Backed by `sympy`.
- Reads `knowledge/formal/*.{json,yaml}` for known equations and axioms.
- For `intent="compute"`: substitutes known variables, simplifies, returns the closed-form answer as `claim` and the equation steps as `support`.
- For `intent="define"` on math/physics terms: returns the axiom verbatim.
- `score` = 1.0 if exact match, else 0.0 (symbolic is binary).

### 5.2 YELLOW — `engines/yellow_retrieval.py`

- Backed by `rank_bm25` over `knowledge/empirical/`.
- Returns top-k passages, joined into `support`. `claim` is the highest-scoring passage's leading sentence.
- `score` = normalized BM25 score of the top hit.

### 5.3 RED — `engines/red_synthesis.py`

- Retrieves perspectives from `knowledge/interpretive/`.
- Combines them with a deterministic template:
  ```
  "From {tradition_a}: {passage_a}. From {tradition_b}: {passage_b}. ..."
  ```
- `score` = number of distinct traditions retrieved / target_k, clipped to [0,1].

### 5.4 Fusion — `pipeline/fusion.py`

- Calls each engine with weight from `EpistemicVector`. Engines with weight < 0.05 are skipped.
- Concatenates `EvidenceRecord`s into a `FusedEvidence`.
- Runs the contradiction detector on the records.
- The blended *score* per claim is `Σ weight_i × score_i` over engines that produced overlapping claims (string-match or simple semantic overlap).

---

## 6. CLI (`cli.py`)

A single-file REPL. No external deps beyond `prompt_toolkit` (optional, falls back to `input()`).

**Session loop**

```
> What is amylase and how does it work?

domain: biology   intent: explain_process
epistemic: g=0.18  y=0.74  r=0.08
engines: YELLOW (primary), GREEN (skipped), RED (skipped)
confidence: 0.81

Amylase is an enzyme that catalyzes the hydrolysis of starch into sugars.
It is found in saliva and pancreatic secretions, and...
```

**Commands**

| Command | Effect |
|---|---|
| `:help` | list commands |
| `:debug on` / `:debug off` | toggle full pipeline trace printout |
| `:reload` | re-read `models/` from disk (use after dropping in trained artifacts) |
| `:status` | show which models are loaded vs stubbed |
| `:why` | explain the last response's evidence and contradictions |
| `:exit` / Ctrl-D | quit |

**Boot behavior**

- On startup, scans `models/` and prints a manifest. Anything missing falls back to its stub. The user is never blocked by missing models.

---

## 7. Colab Workflow (`training/colab_train.ipynb`)

The notebook has **four cells**, each independently runnable, in this order:

1. **Setup**: clones the repo, installs `requirements.txt`, mounts Drive (optional for dataset persistence).
2. **Train domain + intent classifiers**: loads `domains.csv` / `intents.csv`, fits sklearn pipelines, saves `domain_clf.pkl` / `intent_clf.pkl` + the shared `tfidf.pkl`.
3. **Train epistemic router**: loads `epistemic.csv`, builds features via `pipeline.features.router_features` (note: this requires the trained classifiers from step 2), trains the MLP, saves `epistemic_router.pt`.
4. **Train confidence estimator**: runs the inference pipeline over `confidence.csv` rows to compute features, fits the regressor, saves `confidence.pkl`.

A final cell zips `models/` into `models.zip` with a `manifest.json`. We download it locally and unzip into `models/`.

**Determinism**: every cell sets `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`.

---

## 8. Knowledge Seeds

We ship enough seed data to make the pipeline non-empty and to train usable classifiers from. Sizes locked at **Option B**.

- `knowledge/formal/` — **20 entries**. Pendulum period, ohm's law, ideal gas law, pythagoras, snell's law, kepler, coulomb, kinematics, conservation laws, etc.
- `knowledge/empirical/` — **35 entries**. Short paragraphs across biology, medicine, economics (amylase, photosynthesis, inflation, antibiotic resistance, vaccines, supply/demand, …).
- `knowledge/interpretive/` — **18 entries**. 6 topics × 3 traditions each (e.g. trolley problem × utilitarian/deontological/virtue-ethics).

CSV row counts are listed in §9 (M3).

Each category lives in one YAML file under its folder (`knowledge/<category>/<category>.yaml`) holding a list of entries. Engines scan `*.yaml` in the folder, so adding more files later is fine.

```yaml
- id: amylase-001
  domain: biology
  tags: [enzyme, digestion]
  keywords: [amylase, starch, hydrolysis, saliva]   # used for retrieval/match
  text: |
    Amylase is an enzyme that catalyzes...
  sources: [wikipedia:amylase]
- id: photosynthesis-001
  ...
```

Formal entries also carry `equation` and `variables`. Interpretive entries carry `topic` and `tradition`.

---

## 9. Order of Work (Milestones)

We do these in order. Each milestone is independently demoable.

### M1 — Skeleton (no learning)
- Create the directory tree.
- Write `schemas.py`, `features.py` (TF-IDF stub returning zeros), all module files with **stubs only**.
- Write `cli.py` end-to-end. Typing a question runs the full pipeline on stubs and prints a placeholder response.
- **Demo**: `python cli.py` works. Ship it.

### M2 — Engines
- Implement GREEN (sympy + ~5 formal entries), YELLOW (BM25 + ~10 empirical entries), RED (template + ~5 interpretive entries).
- Fusion produces a real `FusedEvidence`.
- **Demo**: math questions get symbolic answers, biology questions retrieve passages, ethics questions get multi-perspective responses — even though the router is still a heuristic stub.

### M3 — Datasets
- Hand-write/synthesize the CSVs in `training/datasets/` at **Option B** sizes:
  - `domains.csv` — 480 rows (60 × 8 domains)
  - `intents.csv` — 280 rows (40 × 7 intents)
  - `epistemic.csv` — 250 rows (hand-labeled `g,y,r`)
  - `confidence.csv` — generated on Colab from labeled QA pairs
- Add `make_dataset.py` to validate them (label coverage, distribution checks).

### M4 — Colab training
- Build `colab_train.ipynb`. Run it. Download `models.zip`. Unzip into `models/`.
- `:reload` in the CLI swaps stubs for trained artifacts.
- **Demo**: same questions as M2, now routed by a learned model.

### M5 — Polish
- `:why` debugging command with full trace.
- Contradiction detector classifier (optional).
- More knowledge seeds.
- Tests for each module (`tests/`).

---

## 10. Locked Decisions

1. **Final response wording**: deterministic template formatter. No external API calls.
2. **Knowledge corpora**: assistant authors all seed YAML files at Option B sizes (§8).
3. **Stack**: Python 3.11 + `numpy / scikit-learn / torch (CPU, inference) / sympy / rank-bm25 / pyyaml / prompt_toolkit`.
4. **Model storage**: plain files in `models/`. No Git LFS.
5. **Training location**: Google Colab only. The user's laptop never runs training. When training is needed, the assistant stops and prompts.
