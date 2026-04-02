# Phase 3 Implementation Plan
## Resilient Housing Bayes
**Focus:** cross-city structural transfer as a falsifiable stress test of posterior decision stability  
**Status entering Phase 3:** Phase 2 completed, with Rotterdam baseline, hazard perturbation stress tests, posterior-derived decision metrics, and robustness interpretation already in place.

---

## Current status update

Completed:
- Phase 0 pipeline validation
- Phase 1 baseline probabilistic pipeline with real hazard proxy
- Posterior-derived decision metrics for Rotterdam
- Phase 2 hazard perturbation runs under Gaussian noise on standardised hazard
- Re-inference and decision metric recomputation across perturbation scenarios
- Aggregate summary tables and comparison figures
- Scenario-specific outputs saved (`summary.json`, `idata.nc`, `asset_metrics.parquet`)
- Phase 2 interpretation memo
- Phase 1 paper updated with robustness section

Main result:
Decision instability remains confined to a narrow prioritisation boundary under moderate hazard perturbation. Borderline share remains low and stable, indicating that the observed unstable region is structural rather than an artefact of a fixed deterministic hazard input.

Interpretive note:
Phase 2 closes the most immediate alternative explanation for the Rotterdam result. Moderate perturbation of the hazard proxy does not produce diffuse system-wide ranking instability. Instead, uncertainty remains concentrated near the decision threshold.

Status:
Phase 3 open — cross-city structural transfer begins.

Operational implication:
The next question is no longer whether the narrow boundary survives within Rotterdam under controlled stress. It is whether the same decision-stability structure persists when the city itself changes.

---

## 1. Goal

Phase 3 is not about making the model richer. It is about testing whether the core empirical structure survives domain shift without changing the model definition every time reality becomes inconvenient.

### Core Phase 3 question
If the generative model is kept fixed and applied to other cities, does the narrow decision boundary persist, or does instability spread across the ranked population?

### Primary test
Apply the same simple model specification, feature definitions, posterior workflow, and decision metrics to additional cities and compare the resulting stability structure.

This turns cross-city transfer into a falsifiable structural experiment:
- if instability remains concentrated near a narrow boundary, the Phase 1-2 result gains structural credibility
- if instability spreads broadly or becomes highly sensitive to preprocessing choices, the result may be city-specific or partly induced by representation choices

---

## 2. Phase 3 scope

### In scope
- fixed-specification transfer to additional cities
- cross-city comparison of posterior decision metrics
- transfer-aware preprocessing and scaling protocol
- support diagnostics to distinguish transfer from extrapolation
- minimal comparative memo

### Out of scope
- latent hazard model
- hierarchical partial pooling across cities
- major expansion of the exposure representation
- hydraulic realism upgrades
- publication polishing
- city-specific model redesign

---

## 3. Working hypothesis

### Baseline hypothesis
If the narrow-boundary result reflects a structural property of the inference-and-ranking pipeline rather than a Rotterdam-specific artefact, then cross-city transfer should still produce a relatively small subset of borderline assets, even if the exact size of that subset changes.

### Competing hypothesis
Under transfer, instability may spread materially across the ranked population, implying that the observed concentration of uncertainty near the decision boundary depends strongly on city-specific feature geometry, scaling choices, or representation artefacts.

---

## 4. Experiment design

## 4.1 Baseline objects already available
From Phases 1 and 2:
- exposure proxy `E_hat_v0`
- deterministic hazard proxy `H`
- synthetic binary outcome formulation
- posterior draws from the baseline logistic model
- citywide decision metrics:
  - posterior mean risk
  - top-k membership probability
  - borderline share
- Phase 2 robustness interpretation showing that instability remains localised under hazard perturbation

## 4.2 New transfer intervention
Construct equivalent asset-level datasets for additional cities using the same minimal feature recipe.

Candidate cities:
- Hamburg
- Donostia / San Sebastián

For each target city:
1. build the same minimal exposure variables
2. build the same hazard proxy type
3. apply the same scaling rule
4. fit the same logistic model
5. recompute posterior decision metrics
6. compare to Rotterdam

### Why this matters
This is the cleanest next stress test because it:
- keeps the model structure intact
- changes the urban context rather than the hazard noise only
- directly tests whether the observed decision boundary is structurally transferable

---

## 5. Fixed vs changing elements

### Fixed
These must remain unchanged in the main Phase 3 experiment:
- model form
- prior family and scale
- feature definitions
- hazard construction logic
- posterior workflow
- decision metric code
- borderline definition (`0.2 < p < 0.8`)
- comparison figure templates
- main `k` values used for decision analysis

### Changing
These are allowed to differ by city:
- building stock size
- building geometry and density
- hydrographic context
- raw predictor distributions
- hazard proxy distribution
- resulting posterior ranking behaviour

### Why this distinction matters
If feature definitions, priors, or scaling rules drift city by city, transfer stops being a structural test and becomes disguised re-fitting. Humans do enjoy changing the exam after seeing the answers.

---

## 6. Scaling and standardisation strategy

This is the most important implementation choice in Phase 3.

## 6.1 Candidate options

### Option A — Per-city standardisation
Standardise each city using its own mean and standard deviation.

Use:
- exploratory sensitivity analysis only

Problem:
- hides domain shift by construction
- makes target cities look artificially well-behaved

### Option B — Rotterdam-anchored standardisation
Fit scaling parameters on Rotterdam and apply those same parameters unchanged to target cities.

Use:
- primary Phase 3 transfer design

Why:
- preserves domain shift
- avoids laundering cross-city differences into local z-scores
- keeps transfer honest

### Option C — Pooled standardisation across all cities
Fit scaling using all available cities.

Use:
- optional secondary analysis only

Problem:
- leaks target-city information into the transfer representation

## 6.2 Recommendation
Use **Rotterdam-anchored standardisation** as the main Phase 3 design.

That means:
- compute mean and standard deviation on Rotterdam features
- save them as transfer scaling artefacts
- apply them unchanged to Hamburg and Donostia

This is the cleanest way to avoid accidental redefinition of the problem.

---

## 7. Outputs to compute for each city

For each transferred city, fit the unchanged model and compute:

### Posterior quantities
- posterior mean impact probability per asset
- posterior standard deviation of impact probability
- posterior predictive summaries

### Decision quantities
For each `k` in `{1000, 2500, 5000}` and optionally matched relative thresholds:
- top-k membership probability
- borderline share (`0.2 < p < 0.8`)
- stable inclusion share (`p >= 0.8`)
- stable exclusion share (`p <= 0.2`)
- rank standard deviation
- normalised rank spread

### Optional but recommended
- ECDF of top-k membership probability
- 90th percentile of rank spread
- boundary concentration ratio
- support-overlap diagnostics relative to Rotterdam

---

## 8. Comparison logic

Each target-city run should be compared to Rotterdam along two dimensions:

### A. Distributional structure
Do posterior probability distributions remain strongly polarised, or do they flatten and diffuse?

### B. Decision structure
Does uncertainty remain concentrated near a relatively narrow prioritisation threshold, or does it spread broadly through the ranked population?

Interpretation should focus on whether instability:
- stays concentrated near a narrow decision boundary
- inflates modestly but remains localised
- expands materially into the broader ranked population

---

## 9. Implementation steps

## Step 1 — Freeze the Phase 3 transfer protocol
Create a short protocol document containing:
- fixed model specification
- fixed priors
- fixed feature definitions
- fixed scaling rule
- fixed metric definitions
- fixed `k` list
- city comparison rules

This should be written before coding new transfer logic so the experiment does not quietly mutate as each city misbehaves.

Suggested location:
```text
docs/phase3/phase3_transfer_protocol.md
```

---

## Step 2 — Build city configuration registry
Implement a simple registry for each city containing:
- city code
- CRS
- boundary source
- buildings source
- hydrography source
- hazard source
- output locations

Suggested module:
```python
CITY_CONFIGS = {
    "RTM": {...},
    "HAM": {...},
    "DON": {...},
}
```

Requirements:
- one standard interface for all cities
- no city-specific logic hidden inside notebooks
- explicit metadata for reproducibility

---

## Step 3 — Create generic city loader
Implement a loader that:
1. reads city-specific raw inputs
2. normalises schema
3. validates required columns
4. outputs a standard asset-level table

Suggested function:
```python
def load_city_assets(city_code: str) -> pd.DataFrame:
    ...
```

Requirements:
- same output schema for all cities
- stable building identifier
- clear validation errors when data are incomplete

---

## Step 4 — Implement transfer scaling module
Create one utility that:
- fits scaling parameters on Rotterdam
- saves those parameters
- applies them unchanged to target cities

Suggested functions:
```python
def fit_reference_scaler(df_ref: pd.DataFrame, cols: list[str]) -> dict:
    ...

def apply_reference_scaler(df: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    ...
```

Requirements:
- deterministic and testable
- no implicit re-fitting on target cities
- scaler metadata saved to disk

Suggested output:
```text
outputs/phase3/config/rtm_scaler.json
```

---

## Step 5 — Reuse the baseline model runner
Use the same Phase 1-2 logistic model runner with:
- same priors
- same inference settings unless explicitly versioned
- same posterior extraction logic

Do not create a “temporary” city-specific variant unless you want future confusion preserved in code.

Suggested scenario naming:
```text
outputs/phase3/RTM/
outputs/phase3/HAM/
outputs/phase3/DON/
```

Or, if versioned more explicitly:
```text
outputs/phase3/exp_001_transfer_rtm_reference/
outputs/phase3/exp_002_transfer_ham_fixedspec/
outputs/phase3/exp_003_transfer_don_fixedspec/
```

---

## Step 6 — Recompute decision metrics
For each fitted city model:
- compute posterior probabilities for all buildings
- derive top-k membership probabilities
- identify borderline assets
- compute rank spread
- summarise stable inclusion and exclusion shares

Save:
- asset metrics as `.parquet`
- model summaries as `.json`
- posterior traces as `.nc`
- figures as `.png`

---

## Step 7 — Add support diagnostics
Implement simple diagnostics to compare target-city predictor distributions against Rotterdam:
- range comparison
- percentile comparison
- share of assets outside Rotterdam support
- optional histogram overlays

This matters because transfer outside support should be interpreted as structural stress or extrapolation, not as ordinary model comparison.

---

## Step 8 — Produce cross-city comparison figures
Minimum figure set:

1. **ECDF of posterior mean probabilities**
   - one line per city

2. **Borderline share vs k**
   - one line per city

3. **Histogram or ECDF of top-k membership probability**
   - compare Rotterdam vs target city

4. **Rank spread summary**
   - distribution or boxplot by city

5. **Support comparison figure**
   - predictor distributions under Rotterdam-anchored scaling

6. **Optional:** boundary concentration figure
   - how much uncertainty mass is carried by assets near the threshold

---

## Step 9 — Write interpretation memo
After the first transfer run, write a short memo answering:

- Does the narrow-boundary result persist in the target city?
- Is instability still localised near the decision threshold?
- Does transfer mainly inflate uncertainty near the boundary, or diffuse it broadly?
- Are observed differences structural, or mostly due to support mismatch?
- Is the current model simple but still defensible for Phase 4?

This memo matters because otherwise the repository will become another sediment layer of unexplained outputs and heroic assumptions.

---

## 10. Proposed repository structure

```text
resilient-housing-bayes/
├── data/
│   ├── raw/
│   │   ├── RTM/
│   │   ├── HAM/
│   │   └── DON/
│   ├── processed/
│   │   ├── RTM/
│   │   ├── HAM/
│   │   └── DON/
│   └── external/
├── outputs/
│   ├── phase1/
│   ├── phase2/
│   └── phase3/
│       ├── config/
│       │   ├── phase3_base.yaml
│       │   ├── transfer_rtm_to_ham.yaml
│       │   ├── transfer_rtm_to_don.yaml
│       │   └── rtm_scaler.json
│       ├── RTM/
│       ├── HAM/
│       ├── DON/
│       └── cross_city_summary/
├── src/
│   └── rhb/
│       ├── data/
│       │   ├── loaders.py
│       │   ├── schemas.py
│       │   └── validation.py
│       ├── preprocessing/
│       │   ├── exposure_features.py
│       │   ├── hazard_features.py
│       │   ├── standardize.py
│       │   └── transfer_scaling.py
│       ├── models/
│       │   ├── logistic_baseline.py
│       │   └── priors.py
│       ├── inference/
│       │   ├── fit.py
│       │   └── diagnostics.py
│       ├── decision/
│       │   ├── topk.py
│       │   ├── rankings.py
│       │   └── cross_city_metrics.py
│       ├── reports/
│       │   ├── summary_tables.py
│       │   └── figures_phase3.py
│       └── pipelines/
│           ├── run_city_pipeline.py
│           ├── run_phase3_transfer.py
│           └── compare_cities.py
├── notebooks/
│   └── phase3/
└── docs/
    └── phase3/
```

---

## 11. Minimal code architecture

### `loaders.py`
- city-aware raw data loading
- schema harmonisation
- asset table construction

### `validation.py`
- required column checks
- uniqueness checks for building IDs
- missingness summaries

### `transfer_scaling.py`
- fit Rotterdam reference scaler
- apply same scaler to target cities
- save and load scaler metadata

### `logistic_baseline.py`
- unchanged Phase 1-2 baseline model
- fixed priors
- standardised inputs

### `cross_city_metrics.py`
- top-k membership probability
- borderline share
- stable inclusion/exclusion shares
- rank spread
- optional boundary concentration metrics

### `run_phase3_transfer.py`
- orchestrates one city transfer run
- saves outputs
- logs metadata

### notebook
Used only for:
- aggregated comparison
- visual inspection
- interpretation

The notebook should still not be the engine. It should merely observe the machinery while humans pretend this counts as civilisation.

---

## 12. Default experiment settings

Suggested defaults:

```python
REFERENCE_CITY = "RTM"
TARGET_CITIES = ["HAM", "DON"]

TOP_K_VALUES = [1000, 2500, 5000]
TOP_K_SHARES = [0.005, 0.01, 0.025]

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8

INFERENCE = {
    "chains": 2,
    "draws": 500,
    "tune": 500,
    "target_accept": 0.9,
}
```

If runtime is too high:
- run Hamburg first
- keep Donostia for second pass
- do not change the model per city
- reduce breadth before reducing methodological clarity

---

## 13. Validation checklist

For every city:
- required schema matches Rotterdam baseline
- feature definitions match the reference implementation
- Rotterdam-anchored scaling applied correctly
- priors unchanged from baseline
- convergence diagnostics checked
- posterior predictive check executed
- decision metrics computed identically
- support diagnostics saved
- outputs saved with city metadata

---

## 14. Decision rules for interpreting results

### Structural robustness
- borderline share remains low
- top-k membership probabilities remain strongly polarised
- instability remains concentrated near the decision boundary
- rank spread inflation is limited or localised

### Moderate structural stress
- borderline region widens somewhat
- target-city uncertainty increases but remains concentrated near the threshold
- broad interpretation still holds

### Structural breakdown
- borderline share increases sharply
- top-k probabilities become much less polarised
- instability spreads across a broad share of assets
- results depend heavily on preprocessing or support mismatch

If structural breakdown appears, that is not failure. It is the experiment working, which is a rare and irritating success.

---

## 15. Minimal viable Phase 3

Start with the smallest experiment that can falsify the idea quickly.

### MVP design
Run **one strict transfer from Rotterdam to Hamburg**:
- same minimal feature set
- same hazard proxy logic
- same synthetic outcome recipe
- same logistic model
- Rotterdam-anchored standardisation
- same decision metrics

### MVP output bundle
Produce:
- `summary.json`
- `idata.nc`
- `asset_metrics.parquet`
- one comparison table
- one comparison figure set
- one short interpretation memo

### Fast falsification criteria
If Hamburg shows:
- low borderline share
- strong probability polarisation
- instability concentrated near the threshold

then the narrow-boundary result survives first transfer.

If Hamburg shows:
- diffuse top-k membership probabilities
- substantial borderline inflation
- instability spread across the citywide ranked population

then the Phase 1-2 structural claim is weakened immediately.

### Why this MVP is enough
It answers the main question before Phase 3 becomes another sprawling monument to future extensibility.

---

## 16. Deliverables

At the end of the minimal Phase 3 segment, produce:

### A. Transfer protocol README
Includes:
- what is fixed
- what is allowed to vary
- how scaling works
- how to run the experiment
- interpretation target

### B. Reproducible transfer runner
One command per city

### C. Comparison notebook
Aggregated figures and interpretation

### D. Short decision memo
One page:
- what transferred
- what widened
- what broke
- recommendation for Phase 4

---

## 17. Closure target

Phase 3 will be considered minimally successful if:
- one transferred city has been processed end-to-end
- the fixed-spec transfer protocol has been respected
- cross-city decision metrics have been compared reproducibly
- the result can be interpreted clearly as either structural robustness or structural breakdown

This phase does not need to prove universal generality. It only needs to tell you whether the narrow-boundary phenomenon survives first contact with another city.

---

## 18. Outcome criterion

Two broad outcomes are possible.

### Outcome A — Structural robustness survives transfer
The narrow-boundary instability result remains visible under fixed-spec cross-city transfer. Borderline assets remain a relatively small subset, and posterior decision instability continues to concentrate near the prioritisation threshold.

Interpretation:
The decision boundary appears to reflect a structural property of the model-plus-data interaction rather than a purely Rotterdam-specific artefact.

### Outcome B — Structural breakdown under transfer
Instability spreads materially across the ranked population, top-k membership becomes much more diffuse, or conclusions become highly sensitive to preprocessing choices.

Interpretation:
The current specification may be too context-dependent to support strong structural claims across cities, and Phase 4 should treat this as a modelling limit rather than simply adding complexity.

## Next small step

Create `docs/phase3/phase3_transfer_protocol.md` first, then implement the Rotterdam-anchored scaler and run Hamburg before touching Donostia.
