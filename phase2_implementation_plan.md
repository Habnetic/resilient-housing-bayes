# Phase 2 Implementation Plan
## Resilient Housing Bayes
**Focus:** posterior decision stability under stress  
**Status entering Phase 2:** Phase 1 baseline completed, with Rotterdam pipeline, baseline Bayesian model, and posterior-derived decision metrics already implemented.

---

## Current status update

Completed:
- Baseline run (σ = 0.00)
- Hazard perturbation runs at σ = 0.10, 0.20, 0.30
- Posterior re-inference and decision metric recomputation
- Borderline share comparison
- Interpretation memo and phase synthesis

Main result:
Decision instability remains confined to a narrow boundary (~1.5–1.7% of assets)
under moderate hazard perturbation.

Pending:
- add σ = 0.05
- save scenario outputs systematically
- generate comparison figures
- freeze reference baseline bundle

## 1. Goal

Phase 2 is not about making the model more impressive. It is about testing whether the main empirical claim from Phase 1 survives controlled perturbation.

### Core Phase 2 question
How stable are posterior-derived prioritisation decisions when the hazard representation is explicitly stressed?

### Primary test
Introduce controlled noise into the deterministic hazard proxy and measure how decision stability changes.

This turns the current baseline into a falsifiable experiment:
- if ranking instability remains concentrated in a narrow boundary, the Phase 1 result gains credibility
- if instability spreads widely, the result may depend too heavily on the deterministic hazard proxy

---

## 2. Phase 2 scope

### In scope
- hazard perturbation experiment
- posterior decision stability re-computation
- comparison against Phase 1 baseline
- reproducible experiment structure
- short interpretation memo

### Out of scope
- latent hazard model
- new city transfer
- hydraulic realism
- major restructuring of the full generative model
- publication polishing

---

## 3. Working hypothesis

### Baseline hypothesis
Small to moderate perturbations in the hazard proxy should mainly affect assets near the prioritisation boundary, while most assets should retain stable prioritisation behaviour.

### Competing hypothesis
Even modest perturbations in hazard may produce large reordering across the ranked set, implying that the current prioritisation stability result is fragile.

---

## 4. Experiment design

## 4.1 Baseline objects already available
From Phase 1:
- exposure proxy `E_hat_v0`
- deterministic hazard proxy `H`
- synthetic binary outcome formulation
- posterior draws from the baseline logistic model
- citywide decision metrics:
  - posterior mean risk
  - top-k membership probability
  - borderline share

## 4.2 New intervention
Construct perturbed hazard variants:

```python
H_perturbed = H + epsilon
epsilon ~ Normal(0, sigma)
```

Test several values of `sigma`, for example:
- `0.05`
- `0.10`
- `0.20`
- `0.30`

The perturbation should be applied after standardisation unless there is a clear reason to test raw-scale noise separately.

### Why this matters
This is the cleanest first stress test because it:
- keeps the existing model structure intact
- directly targets the most simplified component of Phase 1
- reveals whether decision stability depends on unrealistically fixed hazard input

---

## 5. Outputs to compute for each noise level

For each hazard perturbation scenario, re-run the inference and compute:

### Posterior quantities
- posterior mean impact probability per asset
- posterior standard deviation of impact probability
- posterior predictive summaries

### Decision quantities
For each `k` in `{1000, 2500, 5000}`:
- top-k membership probability
- borderline share (`0.2 < p < 0.8`)
- overlap with baseline top-k
- share of assets with strong membership shifts

### Optional but recommended
- rank standard deviation
- rank entropy
- Spearman correlation with baseline posterior mean ranking

---

## 6. Comparison logic

Each perturbed run should be compared to the Phase 1 baseline along two dimensions:

### A. Distributional change
Does the citywide posterior risk distribution change substantially?

### B. Decision change
Do the same assets remain high priority, or does the top-k membership structure spread and destabilise?

Interpretation should focus on whether instability:
- stays concentrated near a narrow decision boundary
- expands materially into the broader ranked population

---

## 7. Implementation steps

## Step 1 — Freeze the Phase 1 baseline reference
Create a clean baseline reference bundle containing:
- model specification summary
- posterior summary tables
- decision metrics for each `k`
- figure outputs used for comparison

Store this in a fixed location so Phase 2 comparisons do not drift.

Suggested output folder:
```text
outputs/phase2/reference_baseline/
```

---

## Step 2 — Create hazard perturbation generator
Implement a small utility function that:
- accepts the baseline hazard vector
- applies seeded Gaussian noise
- returns a perturbed hazard vector
- records `sigma` and `seed` in metadata

Suggested function:
```python
def perturb_hazard(h: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    ...
```

Requirements:
- deterministic under fixed seed
- simple and testable
- no hidden transformations

---

## Step 3 — Build experiment runner
Create one experiment runner that:
1. loads baseline data
2. generates perturbed hazard
3. fits the same logistic model
4. computes posterior decision metrics
5. saves outputs in a scenario-specific folder

Suggested scenario naming:
```text
exp_002_hazard_noise_sigma_005/
exp_003_hazard_noise_sigma_010/
exp_004_hazard_noise_sigma_020/
exp_005_hazard_noise_sigma_030/
```

---

## Step 4 — Recompute decision metrics
For each fitted model:
- compute posterior probabilities for all buildings
- derive top-k membership probabilities
- identify borderline assets
- compare against baseline

Save:
- tables as `.parquet` or `.csv`
- model summaries as `.json`
- figures as `.png`

---

## Step 5 — Produce comparison figures
Minimum figure set:

1. **ECDF of posterior mean probabilities**
   - baseline vs perturbed scenarios

2. **Borderline share vs k**
   - one line per scenario

3. **Histogram of top-k membership probabilities**
   - baseline vs selected perturbed scenario

4. **Overlap with baseline top-k**
   - by noise level and by k

5. **Optional:** rank variability summary
   - boxplot or density of rank standard deviation

---

## Step 6 — Write interpretation memo
After the experiment, write a short memo answering:

- Which noise levels preserve the main Phase 1 result?
- At what point does decision instability materially widen?
- Is the “narrow decision boundary” result robust or fragile?
- What should be changed before cross-city transfer?

This memo matters because otherwise the notebook will become another archaeological layer in the ruin of human documentation.

---

## 8. Proposed repository structure

```text
resilient-housing-bayes/
├── experiments/
│   └── hazard_noise/
│       ├── README.md
│       ├── run.py
│       ├── config.py
│       └── metrics.py
├── outputs/
│   └── phase2/
│       ├── reference_baseline/
│       ├── exp_002_hazard_noise_sigma_005/
│       ├── exp_003_hazard_noise_sigma_010/
│       ├── exp_004_hazard_noise_sigma_020/
│       └── exp_005_hazard_noise_sigma_030/
└── notebooks/
    └── 15_phase2_hazard_noise_analysis.ipynb
```

---

## 9. Minimal code architecture

### `config.py`
- sigma levels
- seeds
- k values
- inference settings

### `run.py`
- orchestrates one scenario
- saves outputs
- logs metadata

### `metrics.py`
- top-k membership probability
- borderline share
- overlap metrics
- optional rank entropy

### notebook
Used only for:
- aggregated comparison
- visual inspection
- interpretation

The notebook should not be the engine. It should be the window. Humanity keeps confusing those two.

---

## 10. Default experiment settings

Suggested defaults:

```python
SIGMAS = [0.05, 0.10, 0.20, 0.30]
SEED = 42
TOP_K_VALUES = [1000, 2500, 5000]

INFERENCE = {
    "chains": 2,
    "draws": 500,
    "tune": 500,
    "target_accept": 0.9,
}
```

If runtime is too high:
- keep chains fixed
- reduce number of scenarios first
- do not keep changing the inference setup per scenario unless explicitly recorded

---

## 11. Validation checklist

For every scenario:
- data dimensions match baseline
- hazard perturbation reproducible under fixed seed
- priors unchanged from baseline
- convergence diagnostics checked
- posterior predictive check executed
- decision metrics computed identically
- outputs saved with metadata

---

## 12. Decision rules for interpreting results

### Strong robustness
- overlap with baseline remains high
- borderline share grows only modestly
- instability remains concentrated

### Moderate robustness
- some top-k churn appears
- borderline region widens but remains small relative to full population
- broad interpretation still holds

### Fragility
- top-k overlap drops sharply
- borderline region spreads materially
- prioritisation becomes highly sensitive even under mild noise

If fragility appears early, that is still a result. An inconvenient result, naturally, which makes it scientifically useful.

---

## 13. Deliverables

At the end of this Phase 2 segment, produce:

### A. Experiment README
Includes:
- what was tested
- how to run it
- inputs
- outputs
- interpretation target

### B. Reproducible experiment code
One command to run each scenario

### C. Comparison notebook
Aggregated figures and interpretation

### D. Short decision memo
One page:
- what changed
- what held
- what broke
- recommendation for next step

---

## 14. Recommended immediate execution order

1. Freeze baseline outputs
2. Implement hazard perturbation utility
3. Run one pilot scenario at `sigma = 0.10`
4. Check runtime, diagnostics, and metric outputs
5. Expand to full sigma grid
6. Generate comparison figures
7. Write interpretation memo

This avoids the classic academic hobby of building an elaborate framework before verifying that the first test even works.

---

## 15. What success looks like

This Phase 2 step is successful if, by the end, you can state one of the following with evidence:

### Outcome A
The narrow-boundary instability result remains robust under plausible hazard perturbation.

### Outcome B
The result weakens under perturbation, indicating that hazard representation dominates ranking stability.

Either outcome is useful. The only useless outcome is a notebook full of half-finished cells and vague optimism.

---

## Assumptions made
- Phase 1 baseline outputs are available and can be reused as reference
- hazard is currently treated as a deterministic standardised proxy
- the first Phase 2 step should minimise structural changes
- computational budget is limited, so experiments should stay close to the existing baseline

## How to validate
- confirm that one perturbed scenario reproduces the baseline workflow end-to-end
- verify that metrics are computed identically across scenarios
- inspect convergence and posterior predictive checks before interpreting ranking changes
- ensure outputs are saved reproducibly with scenario metadata

## Next small step
Implement `perturb_hazard()` and run a single pilot experiment at `sigma = 0.10`.
