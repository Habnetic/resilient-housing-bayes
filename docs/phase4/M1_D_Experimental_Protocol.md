# Habnetic | Phase 4, M1-D1: Gamma prior-width sensitivity

**Status:** implementation v0.1, 5 November 2026. Synthetic methodological sensitivity analysis, **not** empirical validation of urban flood risk. This is the first part of M1-D. M1-D2 will examine event-rate-informed intercept priors separately.

## Goal

Does the choice of prior width for the terrain coefficient materially alter posterior parameter recovery, predictive accuracy, top-k membership probabilities, or expected decision regret? We test the robustness of M1-C's apparent decision advantage to **one** modeling decision while holding the data and all other priors fixed.

## Fixed generative mechanism

We reuse M1-C's **already generated** `m1c_scenarios.parquet`, preserving the same 5,000 Rotterdam-derived assets, the same 4,000/1,000 train/test split, five Bernoulli outcome seeds (101, 202, 303, 404, 505), paired random numbers, and three true gamma values (0, -0.4, -0.8).

For each asset: `T = asinh(T50 / 0.5 m)`, `p_true = sigmoid(-3 + E + H + gamma_true * T)`, and `Y ~ Bernoulli(p_true)`. Both the outcome and the oracle probabilities are **synthetic**.

## Models and priors

B0: `logit(p_i) = alpha + beta_E * E_i + beta_H * H_i`.

B1: `logit(p_i) = alpha + beta_E * E_i + beta_H * H_i + gamma_T * T_i`.

Both models retain the Phase 3 priors `alpha, beta_E, beta_H ~ Normal(0, 2.5)`. B1 tests exactly three prior standard deviations:

| Setting | Prior |
| --- | --- |
| Narrow | `gamma_T ~ Normal(0, 0.25)` |
| Baseline (frozen M1-C) | `gamma_T ~ Normal(0, 0.5)` |
| Wide | `gamma_T ~ Normal(0, 1.0)` |

**Only the prior standard deviation of gamma changes.** B0 has no gamma parameter, so its M1-C fit is reused unchanged. The B1 baseline sigma=0.5 fit is also reused unchanged. No fresh outcome generation, re-selection of buildings, or resplitting is performed.

The sensitivity values are exploratory choices; they are not calibrated empirical priors. The data scale and transformation remain fixed.

## Likelihood and posterior quantities

The likelihood is Bernoulli logistic; inference uses PyMC/NUTS with the frozen M1-C configuration (2 chains, 500 tuning and 500 retained draws per chain, target_accept=0.9). These are pilot settings, **not** confirmatory convergence settings.

Quantities of interest include posterior mean and 95% credible interval for gamma, coverage of the **known** synthetic gamma, R-hat, bulk/tail ESS and divergences; prior/posterior predictive event-rate summaries; held-out RMSE vs known `p_true`, Brier score and integrated Bernoulli log predictive density; draw-wise posterior top-k membership (k=10), selected buildings and expected regret against the synthetic oracle.

A sensitivity result is *decision-relevant* if changing the prior moves selected assets or inclusion probabilities, or materially changes oracle-relative expected regret. Parameter changes without decision changes are recorded separately. Compare paired scenario outcomes; do not pretend that 15 scenarios on a single fixed geography are independent geographic replications.

## Provenance and caching

The runner authenticates M1-C baseline fit fingerprints against current M1-C source code/configuration and scenario-data hashes. It never overwrites M1-C artifacts. New fits are cached under `outputs/phase4/RTM/m1d/{smoke,full}/fits/`. Cache identity includes M1-C data and code hashes, the M1-D code and config hashes, and the chosen gamma prior standard deviation. Outputs record Python and dependency versions and Git HEAD. The `--force` option recomputes **only** new M1-D fits.

**Risk:** changes in current M1-C source/config after the original fits cause a deliberate failure. Reconstruct the original M1-C environment first rather than silently comparing incompatible models.

## Prior predictive limitation

The inherited intercept and slope priors remain broad and imply implausibly high prior predictive aggregate event rates (~47–61% medians in M1-C versus ~6% simulated events). Changing *only* gamma's prior width does not resolve this. **M1-D2**, a separate experiment, will compare intercept/event-rate prior assumptions, with the prior chosen transparently and without conditioning on held-out labels.

## Reproduction

From the repository root, using Python 3.11 and the existing `.venv`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1d.py -q
python -m rhb.pipelines.run_m1d --smoke --dry-run
python -m rhb.pipelines.run_m1d --smoke
python -m rhb.pipelines.summarize_m1d --smoke
python -m rhb.pipelines.run_m1d --dry-run
python -m rhb.pipelines.run_m1d
python -m rhb.pipelines.summarize_m1d
```

**Inputs:** `configs/m1d_gamma_prior_rtm.json`, the unchanged M1-C configuration and `outputs/phase4/RTM/m1c/m1c_scenarios.parquet`, plus the M1-C smoke/full fit artifacts. These must already exist.

**Outputs:** `comparison_results.csv`, per-fit NetCDF/decision files for new B1 priors, `manifest.json`, and `summary/summary_by_gamma_prior.csv` under `outputs/phase4/RTM/m1d/`. Outputs are git-ignored; persist inputs and research artifacts in versioned large-file storage if external reproduction is required.

**Checks and failure modes:** Look for nonconvergence (R-hat >1.01, ESS <400, divergences), extreme prior predictive event rates, weakly identified hazard coefficient because of the small H scale, prior-driven shrinkage of gamma, and selection sensitivity hidden by population-average inclusion differences. Distinguish a zero expected regret in one fixed scenario from evidence of real-world optimality. Repeat with four chains and more outcome/population replicates before confirmatory claims.

## What to test next

1. Examine full D1 prior-width sensitivity and paired decision changes.
2. Conduct D2 event-rate-informed intercept prior sensitivity, with prior predictive calibration and an explicit no-test-label-leakage rule.
3. Vary T50 transform, geographic samples and spatially blocked validation after prior sensitivity is understood.
