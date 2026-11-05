# Habnetic | Phase 4 M1-D2: Intercept prior sensitivity

**Protocol status:** v0.1, 5 November 2026. **Experiment status:** not yet run. This is a *synthetic, controlled sensitivity study*, not an empirical calibration of urban flood risk.

## Goal and testable question

Our Phase 3 / M1-C baseline uses `alpha ~ Normal(0, 2.5)` in a logistic Bernoulli model. Its prior predictive distribution for the aggregate event rate places much of its mass far above the ~6% synthetic event rates. Will alternative, transparently selected intercept priors yield less diffuse prior predictive event rates? Do they alter posterior recovery, held-out predictions, posterior top-k membership or decision regret?

**Important:** A prior centred on the *known* synthetic generating intercept is an **oracle-centred sensitivity condition**, not an independently justified prior for real flood outcomes. The purpose is to quantify the effect of prior assumptions under a known generator, not to discover the correct flood-loss prior.

## Dataset and predictor-scale audit (from the M1-C full scenario table)

- 75,000 rows represent **5,000 distinct buildings repeated across 3 gamma effects and 5 outcome seeds**. Each scenario uses the same 4,000 training / 1,000 held-out buildings. The exact 4,000 training covariates, not all 75,000 repeated rows, are used for prior predictive simulation.
- `E_hat_v0`: sample mean approximately -0.0021; standard deviation 0.833, with an extreme positive tail (maximum approximately 11.99).
- `H_pluvial_v1_logrel`: sample mean approximately 0.00095; standard deviation approximately 0.00761. This column is *not unit-standardized* and a `Normal(0,2.5)` coefficient contributes little variation at its current numerical scale.
- `T = asinh(T50 / 0.5 m)`: mean approximately 0.0231; standard deviation approximately 0.409; T50 itself is in metres.
- These full-table statistics describe repeated predictors; the audit writes train-only statistics to verify actual fit-time scales.

The logistic intercept is the log odds at `E=H=T=0`, **not** necessarily the logit of the population-average event rate. Therefore `alpha=-3` must not be misrepresented as a direct 4,000-building event-rate prior. Changes to the intercept alone cannot guarantee reasonable tail behaviour if the exposure slope remains broad.

## Generative story and likelihood

For an asset `i`, the known synthetic data generator is:

```text
T_i        = asinh(T50_i / 0.5 m)
p_true_i   = sigmoid(-3 + E_i + H_i + gamma_true * T_i)
Y_i        ~ Bernoulli(p_true_i)
gamma_true in {0.0, -0.4, -0.8}
```

Five outcome seeds (101, 202, 303, 404, 505) preserve paired random variates across gamma scenarios. The fitted B1 model uses the **same** Bernoulli-logistic likelihood and all three covariates. Observed test labels are never passed to inference. `p_true` is used only to evaluate simulated oracle decisions and prediction error.

## Pre-registered prior comparison

Only the B1 **intercept** prior changes. `beta_E, beta_H ~ Normal(0, 2.5)` and `gamma_T ~ Normal(0, 0.5)` remain unchanged.

| ID | `alpha` prior | Experimental role |
|---|---|---|
| **P0** | `Normal(0, 2.5)` | Frozen Phase 3 / M1-C reference; reuse M1-C B1 fits. |
| **P1** | `Normal(-3, 1.0)` | Moderately concentrated, *oracle-centred* at synthetic generator. |
| **P2** | `Normal(-3, 0.5)` | More concentrated, *oracle-centred* at synthetic generator. |
| **P3** | `Normal(-1.5, 1.0)` | Deliberately shifted centre; stress test of misspecified prior information. |

P3 guards against the misleading impression that a prior centred at the known generating value is universally beneficial. These are **controlled alternatives**, not elicited empirical priors. Do not select a favourite using the held-out outcomes; do not silently change the priors after viewing the audit.

## Stage A: prior predictive audit before any new fit

Run `python -m rhb.experiments.m1d2_prior_predictive` first. The audit draws 2,000 independent sets of `(alpha, beta_E, beta_H, gamma_T)` per prior from the declared joint prior; computes asset-level probabilities on the *same 4,000 training predictors*; and samples `Y_rep ~ Bernoulli(p)` to obtain prior predictive event-rate distributions.

For every prior, record 5th, 50th and 95th percentiles of **both** mean event probability and Bernoulli-replicated event rate. Also report all training-set observed synthetic event rates for context. These observed outcomes are **reported, not used to adjust prior values**. The prior audit and per-prior artifacts have SHA-256 input/output fingerprints and a fixed seed. If the audit's config/source hashes change, the posterior runner refuses to proceed until the audit is regenerated explicitly.

The audit is a **qualitative screening and sensitivity analysis**, not a claim of statistical calibration. Assess extreme rates and widths, not solely whether the median lies near 6%. The extremely broad original slope prior may dominate upper tails even after shifting the intercept.

## Stage B: posterior/decision comparison (only after reviewing Stage A)

M1-D2 reuses the frozen M1-C `B0` and `P0`/B1 results for **all 15 scenarios**. New fits are B1 with P1, P2 and P3. For smoke testing, use only seed 101 across all three gamma effects.

| Stage | Cases | Priors per case | New fits | Reused B1 fits | Comparisons |
|---|---:|---:|---:|---:|---:|
| Smoke | 3 | 4 | 9 | 3 | 12 |
| Full | 15 | 4 | 45 | 15 | 60 |

Fit using the original M1-C NUTS settings (2 chains, 500 tune + 500 retained draws per chain, `target_accept=0.9`) for exploratory compatibility. Results require follow-up with at least 4 chains and longer runs before confirmatory claims.

**Posterior quantities:** posterior means and 95% HDI for `alpha` and `gamma_T`; Monte Carlo coverage of known synthetic `alpha_true=-3` and `gamma_true`; maximum R-hat, minimum bulk/tail ESS, number of divergences. `alpha` recovery from P0 is read from the original M1-C `posterior_summary.csv` without refitting.

**Checks:** Bernoulli posterior predictive training event-rate quantiles (one incomplete but transparent PPC); held-out RMSE of posterior mean probability versus known `p_true`, held-out Brier versus `Y`, and integrated Bernoulli log predictive density. Decision budget top **k=10** of the held-out 1,000 selected by posterior mean predicted event probability; inclusion probabilities calculated per posterior draw. Compare selection against the synthetic `p_true` oracle and compute expected regret with equal, avoidable event consequences. Compare each P1/P2/P3 with **P0 within the same gamma and seed**; also compare each with B0.

An intercept prior may improve coefficient recovery without improving top-k decisions; this is a central test, not an error. Do not interpret small `p_topk` averages as proof that no individual boundary decision changes.

## Diagnostics and failure modes

- Flag `R-hat > 1.01`, bulk/tail ESS `< 400`, or any divergences. Inspect traces for any flagged case; do not automatically discard a scenario.
- A zero regret is relative to the **synthetic fixed-test oracle**, not real flood damage prevented.
- The fixed sample, train/test split and five outcome seeds do not establish spatial generalization, population robustness, external calibration or causal topography effects.
- Fixed slope priors, extreme E values and tiny H variance may explain why intercept-only adjustments do not resolve unrealistic predictive tails.
- Prior predictive conclusions based on the known generator's intercept are optimistic; P3 partially explores misspecification but is not sufficient to validate real-world prior elicitation.
- All input/output paths under `outputs/` are ignored by Git; the dataset and result artifacts must be preserved separately for external reproduction.

## Reproduction: one command per stage

Repository root, Python 3.11 / existing `.venv`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1d2.py -q
python -m rhb.experiments.m1d2_prior_predictive
```

**Stop and inspect before fitting:**

```powershell
Import-Csv outputs/phase4/RTM/m1d2/prior_predictive/prior_predictive_summary.csv | Format-Table -AutoSize
Import-Csv outputs/phase4/RTM/m1d2/prior_predictive/predictor_summary.csv | Format-Table -AutoSize
```

After deciding to proceed with the pre-registered variants:

```powershell
python -m rhb.pipelines.run_m1d2 --smoke --dry-run
python -m rhb.pipelines.run_m1d2 --smoke
python -m rhb.pipelines.summarize_m1d2 --smoke
python -m rhb.pipelines.run_m1d2 --dry-run
python -m rhb.pipelines.run_m1d2
python -m rhb.pipelines.summarize_m1d2
```

**Inputs:** frozen M1-C config `configs/m1c_rtm.json`, generated M1-C scenario data, per-fit `fit_record.json`, `test_decisions.parquet`, `posterior_summary.csv` and `idata.nc`, plus `configs/m1d2_baseline_prior_rtm.json`. Source M1-C fit fingerprints are verified. Paths are repository-relative.

**Outputs:** `outputs/phase4/RTM/m1d2/prior_predictive/{prior_predictive_draws.parquet,prior_predictive_summary.csv,predictor_summary.csv,observed_synthetic_train_rates.csv,audit_manifest.json}`; `outputs/phase4/RTM/m1d2/{smoke,full}/{comparison_results.csv,manifest.json,fits/,summary/}`. Generated `summary/summary_by_gamma_prior.csv`, `summary/diagnostic_flags.csv`, `summary/paired_cases.csv` and `summary_report.md` are reproducible.

## What to test next

Inspect the Stage A prior predictive tails before fitting. If still unrealistic, consider a **separate, pre-registered slope-prior sensitivity** or predictor rescaling, not silent post-hoc changes to D2. Then assess smoke-run convergence and P1/P2/P3 decisions, followed by five-seed replication, larger samples and spatially blocked evaluation.
