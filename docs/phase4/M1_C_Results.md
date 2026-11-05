# Habnetic | Phase 4, M1-C: Synthetic replication results

**Status:** exploratory full run, 5 November 2026. **Code reference:** `e920f8c` (`phase4/m1-topography`, M1-C experiment implementation). **Scope:** reproducible synthetic decision experiment on Rotterdam-derived covariates. **Not** empirical validation of urban flood risk.

## Research question

Does adding the measured relative-elevation variable T50 improve posterior-based asset selection when its additional effect is deliberately present in the synthetic event generator? Does the result persist across five outcome realizations, and what happens when the effect is absent?

This document summarizes the full-run tables reported on 5 November 2026. The authoritative, recalculable source is `outputs/phase4/RTM/m1c/full/` (ignored by Git), not the values manually transcribed here. Use `python -m rhb.pipelines.summarize_m1c` to regenerate paired-case and per-effect summaries and record source-file hashes. Commit this document only after checking the generated values against it.

## Experimental specification

- **Fixed population:** 5,000 buildings sampled from 220,456 Rotterdam assets with measured terrain; fixed random 4,000/1,000 train/test split. The same buildings and split appear in all scenarios.
- **Generative story:** `Y_i ~ Bernoulli(p_true_i)` with `logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i`, where `T_i = asinh(T50_i / 0.5 m)`. `gamma_true` is **0.0, -0.4, or -0.8**. Events are simulated, not observed flood losses.
- **Randomness:** seeds **101, 202, 303, 404, 505** generate five outcome realizations per effect. The same uniform variates are used across effect sizes for a given seed (*paired random numbers*).
- **Inference:** B0 uses E and H; B1 uses E, H and T. Independent priors: intercept/exposure/hazard coefficients `Normal(0, 2.5)` and B1 terrain coefficient `Normal(0, 0.5)`. Two NUTS chains, 500 tuning + 500 retained draws per chain, `target_accept=0.9`.
- **Decision:** top **k=10** of 1,000 held-out buildings, selected by posterior mean event probability. Within-model uncertainty uses draw-wise top-k inclusion frequencies. An **oracle** uses the known simulated `p_true` on the same test buildings. Expected regret is `sum(p_true for oracle-selected) - sum(p_true for model-selected)`. This assumes equally valued, completely avoidable events and is not real intervention benefit.

## Results: parameter recovery

The table uses the B1 posterior-mean estimates reported in `fit_results.csv`, rounded there to three decimal places.

| True terrain effect | Mean estimated effect | Mean signed bias | 95% HDI contains true effect |
|---:|---:|---:|---:|
| 0.0 | +0.0512 | +0.0512 | 5/5 |
| -0.4 | -0.3052 | +0.0948 | 5/5 |
| -0.8 | -0.7234 | +0.0766 | 5/5 |

In these realizations, B1 follows the direction and magnitude of the imposed effect, with a tendency toward estimates closer to zero for negative gamma. Five realizations per effect cannot establish nominal 95% interval coverage or isolate prior shrinkage from sampling variation.

## Results: selection and expected regret

| True effect | Mean B0 regret | Mean B1 regret | Reduction in mean regret | B1 better / tie / B0 better |
|---:|---:|---:|---:|---:|
| 0.0 | 0.000630 | 0.000630 | 0.0% | 0 / 5 / 0 |
| -0.4 | 0.056997 | 0.008162 | 85.7% | 5 / 0 / 0 |
| -0.8 | 0.235471 | 0.003015 | 98.7% | 5 / 0 / 0 |

The percentage is the **reduction in the group-mean regret**, not the mean of replicate-level percentage reductions. B1 reduces expected regret in all five generated outcome realizations for each nonzero gamma. For gamma=0, B0 and B1 select the same top-10 in all five runs and have identical expected regret in the reported data.

| True effect | Mean B0/B1 top-10 overlap | Mean B0/oracle overlap | Mean B1/oracle overlap |
|---:|---:|---:|---:|
| 0.0 | 10.0/10 | 9.8/10 | 9.8/10 |
| -0.4 | 9.0/10 | 9.0/10 | 9.6/10 |
| -0.8 | 7.8/10 | 8.0/10 | 9.8/10 |

These findings distinguish three quantities: **within-model posterior selection uncertainty** (inclusion probability), **between-model specification sensitivity** (B0/B1 disagreement), and **decision quality** under a synthetic oracle (expected regret). Two models can each be internally confident while selecting different buildings.

A zero expected regret means the same expected-value top-k as the **synthetic oracle**, not zero damage, zero actual risk, or proven intervention effectiveness.

## Computational checks and warnings

All 30 fits reported **zero divergences**. The diagnostic filter in the experimental protocol flags two fits for further attention:

| True effect | Seed | Model | Warning | Reported value |
|---:|---:|---|---|---:|
| -0.8 | 101 | B1 | Tail ESS below 400 | 394 |
| -0.8 | 404 | B0 | R-hat above 1.01 | 1.02 |

These are **warnings**, not proof of failure. Inspect their traces and, if necessary, rerun those cases with four chains and longer sampling before treating coefficient intervals as confirmatory. R-hat and ESS are calculated from summary values and reported at finite precision.

The broad inherited priors fail a plausibility check: in the six-fit smoke experiment, **prior predictive median aggregate event rates ranged roughly 47–61%**, compared with simulated training event rates near 6%. Posterior predictive **aggregate** rates tracked the simulated event frequency, but that alone does not establish calibration or asset-level predictive adequacy. More informative priors require a separate prespecified sensitivity experiment; do not quietly change them within the present baseline.

The full run also calculates held-out RMSE versus `p_true`, Brier score versus simulated binary outcomes, and integrated Bernoulli held-out log predictive density. **Those aggregate predictive results are not included here because their values were not supplied in the reviewed console output.** The summary script computes paired B1-minus-B0 changes and saves them in `summary_by_gamma.csv`; review them before claiming improvement in predictive accuracy.

## Scope, dependencies and failure modes

This study fixes one population sample, one train/test split, one selection budget, one terrain transformation, and the E/H predictor scales. The five outcome seeds quantify conditional simulation variability, **not** variation between population samples, spatial blocks, hazards, or cities. The same buildings and paired uniform random numbers create dependence between runs; do not count the 15 scenarios as independent geographic validations.

The hazard predictor H has very little dispersion on earlier pilots (sample standard deviation approximately 0.0076), leaving its regression coefficient weakly identified; do not infer a physical negative hazard effect from its posterior sign. At the chosen synthetic mechanism, B1 has an intentionally relevant predictor. Superior performance under this design does not demonstrate that T50 improves actual flood-risk prioritization.

The experimental source inputs (`phase3_features_scaled.parquet` and `terrain_features_m1_5m.parquet`), per-fit posteriors, and generated CSVs remain local unless separately deposited. Git history plus an environment snapshot alone is not independent reproducibility. Preserve the original `full/manifest.json`, input hashes and large results in versioned research storage.

## Reproduce the summary

From the root of `resilient-housing-bayes` in the project's Python 3.11 environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1c.py tests/test_m1c_metrics.py -q
python -m rhb.pipelines.summarize_m1c
```

**Inputs expected:** `outputs/phase4/RTM/m1c/full/{fit_results.csv,scenario_comparisons.csv,manifest.json}`. Generate these through the protocol in `docs/phase4/M1_C_Experimental_Protocol.md` if absent. The summary command never reruns inference.

**Outputs created** under `outputs/phase4/RTM/m1c/full/summary/`: `paired_cases.csv`, `summary_by_gamma.csv`, `diagnostic_flags.csv`, `summary_report.md`, `summary_manifest.json` (including hashes of the source artifacts). This directory is excluded from Git by `outputs/**`.

## What to test next

1. Inspect flagged traces and rerun flagged fits using four chains and more draws; assess the sensitivity of the intervals and selection probabilities.
2. Evaluate paired held-out RMSE, Brier and log predictive density; distinguish predictive accuracy from decision quality.
3. Prespecify a **prior sensitivity** study: more realistic intercept/event-rate priors, alternate gamma-prior widths, and prior predictive plausibility.
4. Replicate across **new population samples, train/test splits, terrain scales, spatial blocks and selection budgets** before claiming general robustness or transfer to Hamburg/Donostia.
