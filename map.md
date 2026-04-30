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

### M1 — Skeleton (no learning) ✅
- Created the directory tree.
- Wrote `schemas.py`, `features.py` (TF-IDF stub returning zeros), all module files with stubs only.
- Wrote `cli.py` end-to-end. Pipeline runs on stubs.

### M2 — Engines ✅
- Implemented GREEN (sympy), YELLOW (BM25), RED (template + retrieval).
- Fusion produces a real `FusedEvidence`. Yellow filters by classified domain.

### M3 — Datasets ✅
- 480 / 280 / 250 rows in `domains.csv` / `intents.csv` / `epistemic.csv`.
- `training/make_dataset.py` validates label coverage and `g+y+r=1`.

### M4 — Colab training ✅
- `colab_train.ipynb` trains TF-IDF, domain, intent, router MLP, confidence regressor.
- Models pinned to scikit-learn 1.8.0 to match local venv.
- `:reload` swaps stubs for trained artifacts.

### M5 — Polish (partially done)
- ✅ `:why` debugging command.
- ✅ More knowledge seeds via auto-derivation from YAML (`expand_dataset.py` adds 251 domain rows + 177 intent rows).
- ✅ GREEN freeform compute (sympy parses arbitrary equations / integrals / derivatives without needing a YAML entry).
- ✅ YELLOW multi-passage synthesis (top-3 within domain, 65% score floor).
- ⏳ Contradiction detector classifier — still rule-based only.
- ⏳ pytest suite — not started.

---

## 10. Locked Decisions

1. **Final response wording**: deterministic template formatter. No external API calls.
2. **Knowledge corpora**: assistant authors all seed YAML files at Option B sizes (§8).
3. **Stack**: Python 3.11 + `numpy / scikit-learn / torch (CPU, inference) / sympy / rank-bm25 / pyyaml / prompt_toolkit`.
4. **Model storage**: plain files in `models/`. No Git LFS.
5. **Training location**: Google Colab only. The user's laptop never runs training. When training is needed, the assistant stops and prompts.

---

## 11. Roadmap beyond M5

Each milestone is independently shippable. Pick any order — they don't depend on each other unless noted.

### M6 — Reliability & trust (1–2 days)

The system answers, but you can't yet tell *why* a particular passage was chosen or whether sources disagree. Fixes that.

- **Source citations in output**: every `Empirically:` / `Formally:` / `Interpretively:` line ends with `[source: amylase-001]`. Already in `EvidenceRecord.support`; just plumb it into the renderer.
- **Contradiction-aware rendering**: when the contradiction detector fires across records, the renderer prepends `Sources disagree: ...` and lists the conflicting claims.
- **`:why` upgrade**: show the full reasoning chain — domain probabilities, top-3 router candidates, scoring trace per engine.
- **pytest suite** under `tests/`:
  - `test_schemas.py` — round-trip dataclasses, validate `EpistemicVector` invariant
  - `test_query_processing.py` — tokenization, normalization, entity extraction
  - `test_engines.py` — fixture queries with expected outputs per engine
  - `test_orchestrator.py` — golden-file end-to-end on a small set of queries
  - GitHub Action that runs the suite on every push

### M7 — Knowledge expansion (variable, no training)

Doubling the corpus roughly doubles retrieval coverage. No model retraining strictly needed (auto-derived rows update on next Colab run).

- `knowledge/empirical/` 35 → 70 entries — fill out chemistry, geology, neuroscience, computer science.
- `knowledge/formal/` 20 → 40 entries — add chemistry stoichiometry, optics formulas, statistics distributions, finance formulas.
- `knowledge/interpretive/` 18 → 36 entries — add 6 more topics × 3 traditions (e.g. immigration, animal rights, distributive justice).
- Add a new domain: `chemistry` (or `engineering`). Update `pipeline/schemas.py::DOMAINS`, regenerate auto rows, retrain.
- Document the YAML schema for contributors in `knowledge/README.md`.

### M8 — Real confidence training (3–5 hrs of human labeling)

The confidence regressor is currently fitted to a heuristic target — it learns the rule, not real correctness. To make it actually predict accuracy:

- Hand-curate `training/datasets/qa_pairs.csv` — 80–150 rows of (query, accepted_answer, ground_truth_correct ∈ {0, 1}).
- Update Colab's confidence cell to: run the pipeline on each row, compute features, target = `ground_truth_correct`. Train regressor on real labels.
- Re-train on Colab. New `confidence.pkl`.
- Now `confidence: 0.81` actually means "81% chance this answer is correct based on its features," not "the dominant epistemic mass is 0.8."

### M9 — Conversational state (2–3 days)

Today every query is independent. Adding a small in-session memory enables follow-ups without a transformer or external state.

- Last N (default 3) `Response` objects kept in a `Pipeline` instance attribute.
- `pipeline.run(raw)` checks for pronoun coreference (`it`, `that`, `this`) and pulls the last subject from history when found — purely rule-based.
- New CLI commands:
  - `:history` — show the last few Q&A
  - `:forget` — clear the in-session memory
- New shape in `Response.debug["coref"]` that shows what got resolved, so `:why` explains follow-ups.
- Test cases: "what is amylase?" → "where is it produced?" → second query should prepend "amylase" entity to its tokens.

### M10 — User feedback loop (1 day)

Currently the system never learns from real use. A trivial logging layer turns CLI sessions into future training data.

- New CLI commands `:good` / `:bad` after a response — append `(query, response.rendered, label)` to `training/datasets/feedback.csv`.
- Validator (`make_dataset.py`) gains a `feedback.csv` check.
- Periodic re-train on Colab merges hand + auto + feedback rows.
- Optional: `:rate 1-5` for graded feedback.

### M11 — Engineering polish (1 day)

Quality-of-life wins, no new behavior.

- `pyproject.toml` instead of `requirements.txt`. Pin everything (numpy, torch, sklearn, sympy).
- Replace `print` with `logging`; CLI gets `--verbose` and `--quiet`.
- `config.yaml` for thresholds (BM25 floor, contradiction sensitivity, top-k for yellow). Today they're constants in code.
- CLI: command history (prompt_toolkit), color (ANSI), tab-complete on commands.
- `:bench` — print per-stage latency for a single query (good for catching regressions).

### Recommended order

1. **M6** — fast, makes everything else more debuggable. Do this first.
2. **M11** — small, removes friction (config file, logging, history).
3. **M7** — biggest coverage win for end users.
4. **M10** — only meaningful after **M7** (more knowledge → real feedback worth collecting).
5. **M8** — needs labeled data; collect via M10 first if you don't want to write 80 QA pairs by hand.
6. **M9** — most ambitious, do last when foundation is solid.

### Things deliberately not in the roadmap

- **Transformer-based components.** Locked out by §0.
- **External LLM for response wording.** Locked out by §10.
- **Web fetch / live search.** Possible but would add network dependency and provider risk; revisit after M9.
- **GUI / web frontend.** Requirement §0 says CLI only.
