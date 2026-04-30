# Knowledge base

The three folders here are the only place facts live. The engines load every `*.yaml` file in their folder and treat the contents as a flat list of entries.

```
knowledge/
├── formal/          → GREEN engine  (sympy compute, axioms)
├── empirical/       → YELLOW engine (BM25 retrieval)
└── interpretive/    → RED engine    (multi-tradition synthesis)
```

You can split a folder into multiple YAML files (e.g. `chemistry.yaml`, `physics.yaml`) — anything matching `*.yaml` is loaded.

## Common fields (every entry)

| Field | Required | Notes |
|---|---|---|
| `id` | yes | unique slug, used as the citation token shown to users |
| `domain` | yes | one of the values in `pipeline/schemas.py::DOMAINS` |
| `tags` | yes | list of short topic tags (used by GREEN matcher and as fallback retrieval signal) |
| `keywords` | yes | list of distinctive nouns/terms; **first keyword is the canonical term** for auto-derived training rows |
| `text` | yes | 2–4 sentence prose explanation. The first sentence is what gets returned by YELLOW |
| `sources` | recommended | list of citation strings (e.g. `wikipedia:amylase`, `textbook:mechanics`) |

## Per-engine extras

### Formal entries (GREEN)

```yaml
- id: kinetic-energy-001
  domain: physics
  tags: [mechanics, energy]
  keywords: [kinetic, energy, mass, velocity, motion]
  equation: "E = m*v**2/2"          # sympy-parseable; ^ is treated as exponent in freeform but write ** here
  variables:                          # symbol → "human description (unit)"
    E: kinetic energy (J)
    m: mass (kg)
    v: speed (m/s)
  defaults:                           # optional: per-entry constants for symbols not given by user
    g: 9.81
  text: |
    Kinetic energy of a moving object: E = (1/2) m v^2.
  sources: [textbook:mechanics]
```

- The unit in parentheses (`(kg)`, `(m/s)`) is parsed by GREEN's value-extraction. Use canonical SI units.
- `defaults:` is consulted before the global constants table (`g`, `c`, `G`, `k`, `h`, `R`).

### Empirical entries (YELLOW)

The common fields above are sufficient. YELLOW filters by `domain` (the classified domain of the query), so make sure the entry's domain is in the closed set in `schemas.py::DOMAINS`.

### Interpretive entries (RED)

Add `topic` and `tradition`:

```yaml
- id: trolley-utilitarian-001
  domain: philosophy
  topic: trolley_problem            # groups entries; RED returns all traditions of the matched topic
  tradition: utilitarian            # display label and de-dup key
  tags: [ethics, dilemma, consequentialism]
  keywords: [trolley, problem, dilemma, kill, save, divert, lever, track]
  text: |
    The utilitarian answer pulls the lever: ...
  sources: [singer:practical-ethics]
```

- Each topic should have **at least 2 traditions** for RED to fire.
- Recommended layout: 6 topics × 3 traditions per file.

## Adding a new domain

If you add a domain that doesn't exist in `DOMAINS`, four files need updates:

1. `pipeline/schemas.py` — add to the `DOMAINS` list.
2. `pipeline/domain_classifier.py` — add the domain's keyword set to `_DOMAIN_KEYWORDS` for the heuristic fallback.
3. `pipeline/router.py` — add a `_DOMAIN_PRIORS[<name>] = (g, y, r)` row.
4. `training/datasets/domains.csv` — add ≥30 hand-written training examples for the new domain.

After that, run `python training/expand_dataset.py` to regenerate the auto CSVs, then re-train on Colab.

## Workflow when adding entries

```bash
# 1. Edit the yaml file(s)
# 2. Regenerate auto-derived training rows
python training/expand_dataset.py

# 3. Validate everything
python training/make_dataset.py    # checks label coverage and g+y+r=1

# 4. Re-train on Colab (open the notebook, Run all)
#    Then unzip models.zip into models/ and :reload in the CLI.
```

The CLI does not need to be restarted between knowledge-base edits — engines re-read YAML on `:reload`.

## Style guide

- **Keep `text` to 2–4 sentences.** Longer entries fragment poorly when YELLOW pulls the first sentence.
- **Avoid commas in `id` and `tradition`.** They're displayed as plain strings.
- **Keywords should be distinctive.** Generic words ("system", "process", "the") add noise to BM25 and the GREEN matcher.
- **Use IUPAC / SI / standard notation in equations.** Don't mix `^` and `**` — equations are parsed by sympy.
- **Cite sources.** Even `wikipedia:topic` is more useful than nothing — it's what the user sees in `:why`.
