# Habnetic Phase 4 | M1-C: Synthetic sensitivity and replication protocol

**Status:** implementation v0.1, 5 November 2026. Exploratory robustness experiment; not an empirical flood-risk validation.

## Goal and hypotheses

Determine whether the M1-B result depends on one favourable outcome realization, and identify the conditions in which adding a measured terrain covariate changes constrained selections.

- **H0 (negative control):** for true terrain effect `gamma_true = 0`, model B1 should not systematically outperform B0 on predictive accuracy or expected decision regret. Sampling variation and overfitting are possible.
- **H1 (known signal):** for `gamma_true = -0.4` or `-0.8`, B1 should more often recover a negative effect and reduce expected decision regret relative to B0. This is a *simulation hypothesis*, not a guaranteed outcome.
- **Decision question:** are selections internally stable under posterior uncertainty yet sensitive to model specification? Compare posterior top-k inclusion with overlap across B0/B1.

## Population and sources

- RTM Phase 3 predictors from `outputs/phase3/RTM/phase3_features_scaled.parquet`.
- Terrain from sibling `data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet`.
- Population: **220,456** assets with valid T50. One fixed **5,000**-asset random sample, selected with `sample_seed=20261105`.
- Fixed **4,000/1,000** training/test partition with `split_seed=20261106`. This is a random split, **not spatial validation**.
- Terrain variable: `T = asinh(T50 / 0.5 m)`. Units: T50 in metres, T dimensionless. No refitting of the transformation.
- `E_hat_v0` and `H_pluvial_v1_logrel` retain Phase 3 scales. Small dispersion in H (~0.0076 on the earlier test sample) implies weak coefficient identifiability. Do not interpret its sign as a physical hazard effect.

## Generative story and likelihood

Conditional on the fixed predictors, each asset independently receives a synthetic binary event:

\[
Y_i \sim \operatorname{Bernoulli}(p_i),\qquad
\operatorname{logit}(p_i) = -3 + 1 E_i + 1 H_i + \gamma_{\rm true} T_i.
\]

The true additional terrain effect takes values **0, -0.4, -0.8**. Five outcome-generation seeds are **101, 202, 303, 404, 505**. For each seed, the *same uniform draws* are used for each gamma, so scenarios are paired by random noise. Different seeds generate distinct Bernoulli realizations. The same fixed buildings and split are reused throughout. The generated `p_true` is known **only for evaluation** and must never enter inference.

These events do **not** represent observed building losses or validated flood probabilities.

## Bayesian models

**B0:** `logit(p_i) = alpha + beta_E E_i + beta_H H_i`.

**B1:** `logit(p_i) = alpha + beta_E E_i + beta_H H_i + gamma_T T_i`.

Independent priors: `alpha, beta_E, beta_H ~ Normal(0, 2.5)` in both models, and `gamma_T ~ Normal(0, 0.5)` in B1. Their breadth is retained for direct comparability to the pilot; prior predictive distributions can imply implausibly large event rates. This is a documented limitation, **not a validated prior**. No priors depend on the simulated `gamma_true`.

Pilot MCMC: 2 chains × 500 retained draws, after 500 tuning steps, `target_accept=0.9`; fixed derived sampler seeds. Four chains and longer runs will be required for confirmatory analysis.

## Posterior targets, checks, and model comparison

For every fit, save parameter summaries (including the B1 posterior mean and 95% HDI of gamma), R-hat, bulk/tail ESS and divergence counts. Report coverage of the **known** gamma by the B1 95% HDI across replicate datasets; five repetitions give only a coarse exploratory estimate, **not reliable nominal-coverage evidence**.

- **Prior predictive:** summarize the 5th, 50th and 95th percentiles of simulated training-set event rates, compared with observed synthetic rate. This identifies overly diffuse priors.
- **Posterior predictive:** summarize analogous event-rate percentiles from replicated `Y_obs` on training observations; retain the synthetic observed rate for comparison. This one statistic alone is an incomplete PPC.
- **Held-out prediction:** compute RMSE of posterior-mean probabilities against known `p_true`; Brier score against held-out Bernoulli outcomes; held-out integrated Bernoulli log predictive density (`heldout_elpd`). The same test assets and labels are used for B0/B1. **Do not fit on test labels.** Held-out scoring is the primary comparison; LOO/WAIC is not needed for this first fixed-split study.
- **Decision:** select the top **1% of the 1,000 test assets** (`k=10`) by posterior mean event probability. For each posterior draw, recompute top-k and count inclusion frequency. Report borderline assets with `0.2 < p_topk < 0.8`, model overlap, and differences in inclusion probabilities.
- **Synthetic decision value:** assume equal and perfectly avoidable consequence per selected event. Let `S` be the selected set, and `S*` the oracle top-k using known `p_true`. Report `regret = sum_{i in S*} p_true_i - sum_{i in S} p_true_i`. Smaller is better. This is **not** observed cost, intervention efficacy, or real-world damage prevented.


## Design and limitations

There are 3 effects × 5 random seeds × 2 fitted models = **30 fits**. `--smoke` uses only the first outcome seed across all effects = **6 fits**. The number of buildings, train/test split, predictors, transformation, budgets, model priors, MCMC configuration and decision rule are held constant across scenarios.

The fixed random train/test split does not test geographical transfer, spatial generalization, infrastructure-specific selection, real hazard validity, missing-terrain mechanisms, or sensitivity to the population sample. Future sensitivity experiments must vary prior width (e.g. gamma sigma 0.25/1.0), terrain transform (raw/scaled or `asinh` alternative scale), simulation seeds and spatial blocking independently. Do not interpret one negative-gamma scenario as an empirical estimate of terrain's causal or protective effect.

Prior predictive limitation (2026-11-05). The Phase 3 priors produce extremely broad distributions of aggregate event rates. In the M1-C smoke experiment, prior predictive medians ranged approximately from 47% to 61%, while simulated event rates were approximately 6%. The original priors are retained for controlled comparison. Sensitivity to more informative priors remains necessary before drawing robust methodological conclusions.

## Reproduce from repository root

Activate **the repository's Python 3.11 `.venv`**, and ensure the package is discoverable (`$env:PYTHONPATH = (Resolve-Path .\src).Path` on Windows PowerShell).

```powershell
python -m pytest tests/test_m1c.py tests/test_m1c_metrics.py -q
python -m rhb.experiments.m1c_simulation
python -m rhb.pipelines.run_m1c --smoke --dry-run
python -m rhb.pipelines.run_m1c --smoke
# After reviewing six fits and diagnostics:
python -m rhb.pipelines.run_m1c
```

`--force` recomputes fit checkpoints. Without it, completed fits are reused **only if** data, configuration, and relevant code hashes match. Fits are checkpointed one by one to permit safe resumption after a failure.

The scenario generator writes `outputs/phase4/RTM/m1c/m1c_scenarios.parquet`, `sample_ids.csv`, and `experiment_config.json`. The runner writes to **separate** `smoke/` and `full/` subdirectories, with `fit_results.csv`, `scenario_comparisons.csv`, `manifest.json`, and per-fit NetCDF posterior, CSV summary, Parquet test decisions and JSON fit record. `outputs/**` is Git-ignored. Store large artifacts and source datasets in versioned research storage, with recorded hashes, for independent reproduction.

**Required environment:** Python 3.11.9 reference; PyMC 5.28.1 reference; see `docs/phase4/M1_pilot_environment_py311.txt`. The manifest records actual interpreter, package versions, Git HEAD, input/data/config and code hashes. A `pip freeze` snapshot describes one environment, not a proof of compatible installs on other platforms.

## Interpretation / failure modes

Flag results with divergences, high R-hat, poor ESS, or implausible prior predictive event rates. `r_hat > 1.01`, bulk/tail ESS below 400, or divergences are diagnostic warnings requiring investigation, not automatic evidence against the hypothesis. Inspect traces if any problems occur. A stable internal ranking can still disagree strongly between model specifications. A negligible change in mean inclusion probability over all buildings can conceal a large change for one selected asset. Do not pool correlated, paired scenarios as independent trials.

**What to test next:** prior and terrain-scale sensitivity, spatially blocked validation, and replication across population samples before running citywide or transferring to HAM/DON.
