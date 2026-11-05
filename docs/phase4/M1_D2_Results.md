# Habnetic | Phase 4 M1-D2: Intercept prior sensitivity results

**Status:** PENDING, protocol and code only. This file is a results **template**. Do not cite it as experimental evidence until the prior predictive audit and posterior runs are reviewed and the values below are filled from generated tables.

**Source protocol:** `docs/phase4/M1_D2_Experimental_Protocol.md`

**Source code:** `src/rhb/experiments/m1d2_prior_predictive.py`, `src/rhb/pipelines/run_m1d2.py`, `src/rhb/pipelines/summarize_m1d2.py`

## A. Prior predictive audit (pending)

- Training predictor scales and sample identity: **TODO** (see `prior_predictive/predictor_summary.csv`).
- Prior event-rate quantiles for P0/P1/P2/P3: **TODO** (see `prior_predictive/prior_predictive_summary.csv`).
- Synthetic training-rate reference range: **TODO** (see `prior_predictive/observed_synthetic_train_rates.csv`).
- Decision to proceed with the pre-registered alternatives, including rationale: **TODO**.
- Audit manifest input and code fingerprints: **TODO**.

## B. Posterior and held-out prediction (pending)

- Alpha and gamma recovery / interval coverage: **TODO**.
- Held-out RMSE, Brier, integrated log predictive density and training PPC: **TODO**.
- Convergence warnings (R-hat, ESS and divergences): **TODO**.

## C. Decision consequences (pending)

- Top-10 overlap with P0, draw-wise inclusion shifts, oracle overlap: **TODO**.
- Paired expected-regret differences by gamma and outcome seed: **TODO**.
- Distinguish parameter recovery from decision quality: **TODO**.

## D. Limitations and interpretation

- All targets, including generating `alpha=-3`, are **synthetic**.
- P1/P2 are *oracle-centred*. They cannot be interpreted as validated real-world priors.
- The population, train/test split and covariates remain fixed; five seeds are not geographic replications.
- `alpha` describes reference covariates `E=H=T=0`, not the mean event rate. Extremely small H variation and broad E prior remain possible failure modes.
- Expected regret assumes the synthetic event probability and uniform, fully avoidable event value, not an empirical intervention effect.

## How to reproduce

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m rhb.experiments.m1d2_prior_predictive
# Inspect the audit before starting any posterior fits.
python -m rhb.pipelines.run_m1d2 --smoke
python -m rhb.pipelines.summarize_m1d2 --smoke
python -m rhb.pipelines.run_m1d2
python -m rhb.pipelines.summarize_m1d2
```

**What to test next:** Review prior predictive tails, diagnostic sensitivity and potential need for slope priors; repeat with larger runs and spatial validation before making empirical claims.
