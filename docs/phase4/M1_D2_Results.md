# Habnetic | Phase 4 M1-D2: Intercept prior sensitivity results

**Status:** exploratory full run completed, 5 November 2026. **Scope:** controlled synthetic prior-sensitivity experiment using Rotterdam-derived covariates; **not** empirical flood-risk calibration or validation.

**Protocol:** `docs/phase4/M1_D2_Experimental_Protocol.md`<br>
**Implementation:** `src/rhb/experiments/m1d2_prior_predictive.py`, `src/rhb/pipelines/run_m1d2.py`, `src/rhb/pipelines/summarize_m1d2.py`<br>
**Authoritative generated evidence:** `outputs/phase4/RTM/m1d2/prior_predictive/` and `outputs/phase4/RTM/m1d2/full/` (not tracked by Git). Numerical values below were transcribed from the completed audit and full-run summaries on 5 November 2026; the generated CSVs and manifests remain the reproducible source of truth.

## Research question and experimental design

Can more plausible intercept priors correct the implausibly diffuse *prior predictive* event-rate distribution of the Phase 3 / M1-C model? Do these priors materially change posterior coefficient recovery, held-out predictive scores, or the top-10 asset selection?

- **Generative story:** For each building, `Y_i ~ Bernoulli(p_true_i)`, with `logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i`; `T_i = asinh(T50_i / 0.5 m)`, and `gamma_true in {0, -0.4, -0.8}`. All events are synthetic.
- **Population:** Fixed 5,000-building sample with measured terrain; common 4,000 / 1,000 training/test split; outcome seeds `101, 202, 303, 404, 505` for each effect size. Five outcome realizations on the **same assets** are not five independent geographical validations.
- **Likelihood:** Bernoulli-logistic B1 with exposure (`E`), hazard (`H`), and topography (`T`). The B0 reference omits topography. Test outcomes are never used for fitting.
- **Unchanged priors:** `beta_E, beta_H ~ Normal(0, 2.5)` and `gamma_T ~ Normal(0, 0.5)` independently. Only the intercept prior changes:

| Prior ID | Intercept prior | Role |
|---|---|---|
| P0 | `Normal(0, 2.5)` | Frozen Phase 3 / M1-C B1 reference; reused fit artifacts |
| P1 | `Normal(-3, 1.0)` | Oracle-centred, moderately informative |
| P2 | `Normal(-3, 0.5)` | Oracle-centred, more concentrated |
| P3 | `Normal(-1.5, 1.0)` | Deliberately shifted, misspecification stress test |

**Crucial design limitation:** P1/P2 use the known generating intercept, so they are **oracle-centred sensitivity conditions**, not independently elicited real-world priors. The intercept describes log odds at `E=H=T=0`, not the mean event rate across buildings.

**Computation:** The prior audit used 2,000 joint-prior draws per configuration on the 4,000 training covariate vectors. The posterior study completed **45 new B1 fits**, reused **15 P0/B1 fits** and B0 references, and produced **60 prior-by-scenario comparisons**. Fits used two NUTS chains, 500 tuning and 500 retained draws per chain, `target_accept=0.9`; longer four-chain fits would be needed for confirmatory analysis. Eight M1-D2 unit tests passed before the runs.

## A. Prior predictive audit

The audit distinguished the expected aggregate event rate (mean of the building probabilities) from the Bernoulli-replicated aggregate event rate. The table below gives the **5th, 50th, and 95th percentiles of the expected aggregate rate**, expressed as percentages:

| Prior | 5th | Median | 95th |
|---|---:|---:|---:|
| P0 | 3.92% | **51.08%** | 97.12% |
| P1 | 1.48% | **10.43%** | 30.78% |
| P2 | 3.25% | **9.65%** | 24.53% |
| P3 | 6.69% | **25.37%** | 51.48% |

The Bernoulli-replicated rate quantiles were nearly identical at this sample size. The synthetic event rate was approximately **6.2%**, varying across scenario realizations; this is a **reference for interpretation**, not a value used to tune the priors during the audit.

**Interpretation:** P1 and P2 drastically reduce the implausibly high prior predictive median relative to P0, but retain wide upper tails. P2 is the most concentrated candidate in this set. P3 demonstrates that shifting the intercept to an incorrect centre can preserve substantial prior predictive mismatch. None of these audits proves a prior is suitable for observed flood losses.

**Training predictor scales (4,000 distinct assets):** `E_hat_v0` mean 0.00213, population SD 0.81610, max 7.9900; `H_pluvial_v1_logrel` mean 0.000964, population SD 0.007618, max 0.029841; transformed `T` mean 0.02485, population SD 0.41293, max 2.6004. The broad exposure slope and its positive covariate tail contribute to wide prior predictions. At the present scale, the hazard covariate has very little numerical variation, raising a separate identifiability/scaling issue.

**Audit provenance:** `prior_predictive/prior_predictive_summary.csv`, `predictor_summary.csv`, `observed_synthetic_train_rates.csv`, and `audit_manifest.json` contain the recalculable results and fingerprints. Exact hashes should be read from those files; none is reproduced from memory here.

## B. Posterior recovery and held-out prediction

### B1. Intercept recovery

Mean posterior intercept by true synthetic topographic effect, over five outcome seeds. The known generating intercept equals **-3**.

| True `gamma` | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 0.0 | -3.0132 | -3.0144 | -3.0154 | -3.0076 |
| -0.4 | -3.0104 | -3.0132 | -3.0112 | -3.0020 |
| -0.8 | -2.9930 | -2.9950 | -2.9948 | -2.9842 |

The posterior intercept varied little between these priors, including P3. All reported 95% intercept intervals contained the generating value in **5/5 replicates per configuration/effect**. This suggests the likelihood is informative about `alpha` with 4,000 training outcomes. Five replicates per cell cannot establish accurate interval coverage, and smaller training sets have not been tested.

### B2. Topographic coefficient recovery

| True `gamma` | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 0.0 | +0.0512 | +0.0490 | +0.0512 | +0.0500 |
| -0.4 | -0.3052 | -0.3008 | -0.3052 | -0.3012 |
| -0.8 | -0.7234 | -0.7204 | -0.7242 | -0.7196 |

Changing only the intercept prior had little effect on the estimated topographic coefficient. All reported 95% `gamma` intervals contained the true value in 5/5 replicates per cell, again an exploratory count rather than validated coverage.

### B3. Posterior predictive and held-out checks

Training posterior predictive **median aggregate event rates** were approximately **6.1–6.3%** across priors and effects, close to the corresponding synthetic training event rates. For example, for `gamma_true=-0.8`, the median PPC rates were P0 **6.220%**, P1 **6.230%**, P2 **6.215%**, and P3 **6.270%**, versus an observed mean training event rate of **6.230%**.

Held-out predictive-score differences against P0 were small. Mean integrated Bernoulli log predictive density changes (`Pj - P0`) ranged approximately **-0.0164 to +0.0168** across the reported effect/prior combinations; Brier score changes were of order **10^-6**, and RMSE-against-known-probabilities changes were small. These are differences in the experiment's reported score units; no substantive superiority claim follows from their sign at this level of replication. Aggregate event-rate PPC is only one diagnostic and cannot establish individual calibration or spatial validity.

## C. Decision consequences

The decision is to select the **top 10** among the 1,000 held-out buildings by posterior mean event probability. The oracle ranks assets by known synthetic `p_true`. Expected regret is the oracle sum of `p_true` minus the model-selected sum, assuming equal and completely avoidable events. It is not observed flood damage or estimated intervention benefit.

### C1. Mean expected selection regret (five outcome realizations)

| True `gamma` | P0 | P1 | P2 | P3 |
|---|---:|---:|---:|---:|
| 0.0 | 0.000630 | 0.000630 | 0.000630 | 0.000630 |
| -0.4 | **0.008162** | 0.012243 | 0.012243 | **0.008162** |
| -0.8 | 0.003015 | 0.003015 | 0.003015 | 0.003015 |

At `gamma_true=-0.4`, P1 and P2 each had **one worse case and four ties** relative to P0; the mean paired regret increase was **0.004081**. The changed selection occurred in seed **101**, with an individual-case regret increase of approximately **0.020406**. P3 tied P0 in all five replicates. At `gamma_true=0` and `-0.8`, all four priors tied P0 in regret across all five replications.

### C2. Top-k overlap and posterior inclusion probabilities

| True `gamma` | P1 / P0 mean overlap | P2 / P0 mean overlap | P3 / P0 mean overlap |
|---|---:|---:|---:|
| 0.0 | 10.0/10 | 10.0/10 | 10.0/10 |
| -0.4 | 9.8/10 | 9.8/10 | 10.0/10 |
| -0.8 | 10.0/10 | 10.0/10 | 10.0/10 |

Top-10 lists were nearly identical, even though the reported maximum shifts in draw-wise inclusion probability relative to P0 reached **0.054** (5.4 percentage points) for P2 when `gamma_true=-0.8`. Mean overlap describes *between-prior* agreement, not certainty within any individual posterior model.

**Interpretation:** A large improvement in prior predictive plausibility did **not** translate into a consistent held-out decision improvement. The isolated P1/P2 decision degradation at `gamma=-0.4`, seed 101 is not decisive evidence that an informative intercept prior is harmful: the rankings are near a top-k boundary, and two-chain Monte Carlo variation is a possible contributor. A targeted longer-chain check would be needed to distinguish a systematic prior effect from sampling noise.

## D. Computational diagnostics and limitations

The completed full summary identified **two flagged fit/scenario combinations** using the protocol's criteria (`R-hat > 1.01`, minimum bulk/tail ESS `< 400`, or any divergences):

| True `gamma` | Seed | Prior | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Divergences | Reason |
|---|---:|---|---:|---:|---:|---:|---|
| -0.8 | 101 | P0 | 1.00 | 868 | **394** | 0 | Tail ESS below 400; reused M1-C fit |
| -0.8 | 505 | P2 | 1.01 | 531 | **373** | 0 | Tail ESS below 400; new M1-D2 fit |

For these listed scenarios, no divergences were reported, and neither has `R-hat > 1.01`. The flag is a **diagnostic prompt**, not proof that the fitted model or scientific conclusion is invalid. The detailed traces for these two cases have **not** been reviewed as part of this report. Results remain *exploratory* with two chains and 500 retained draws per chain.

Further limitations:

- Only five paired **outcome seeds** per effect were varied; the 5,000 buildings, predictors, train/test assignment, selection budget and oracle probabilities were held fixed. Results do not establish population, spatial, temporal or cross-city generalization.
- This is a synthetic experiment with the generating mechanism deliberately available to the researcher. It is not evidence of an empirical topographic effect on observed flooding.
- P1/P2 are oracle-centred. P3 is one misspecified centre, not an exhaustive study of prior misspecification or robust prior elicitation.
- The exposure prior remains diffuse, and the hazard predictor's extremely small variance remains unaddressed. Intercept-only adjustments cannot establish joint-prior adequacy.
- Selection regret assumes identical consequences and complete preventability of a synthetic Bernoulli event. It has no direct CHF, damage, resilience or intervention-effect interpretation.
- Posterior predictive aggregate event-rate agreement is necessary but insufficient. No spatial blocking, missing-terrain sensitivity, covariate measurement error study or real loss data were assessed.
- MCMC variability near the selection boundary may influence differences as small as one building; targeted re-fitting is appropriate before attributing the difference solely to an intercept prior.

## Reproduction and source-of-truth files

From the root of `resilient-housing-bayes`, with the project `.venv` and generated M1-C inputs available:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1d2.py -q
python -m rhb.experiments.m1d2_prior_predictive
# Review the prior audit before fitting; the commands below reproduce the recorded exploratory runs.
python -m rhb.pipelines.run_m1d2 --smoke
python -m rhb.pipelines.summarize_m1d2 --smoke
python -m rhb.pipelines.run_m1d2
python -m rhb.pipelines.summarize_m1d2
```

**Inputs:** `configs/m1d2_baseline_prior_rtm.json`, `configs/m1c_rtm.json`, M1-C generated scenario data and reference fit artifacts (including posterior samples/decisions). **Outputs:** `outputs/phase4/RTM/m1d2/prior_predictive/`, `outputs/phase4/RTM/m1d2/full/comparison_results.csv`, `full/manifest.json`, `full/fits/`, and `full/summary/{summary_by_gamma_prior.csv,diagnostic_flags.csv,paired_cases.csv,summary_report.md}`. The audit and run manifests contain the provenance/fingerprints; the `outputs/` tree is intentionally Git-ignored and must be preserved separately for independent reproduction.

## Scientific conclusion

In this fixed synthetic Rotterdam experiment, replacing the broad intercept prior with oracle-centred informative priors substantially improved **prior predictive plausibility**, while posterior intercept estimates, aggregate posterior predictive rates and held-out top-10 decisions were largely unchanged. One boundary selection changed under P1/P2 in one of the five outcomes at `gamma=-0.4`, worsening synthetic expected regret by 0.020406 for that seed. This is an observed exploratory sensitivity, **not** confirmation of a reliable prior-induced harm or benefit. Prior plausibility, posterior robustness and decision robustness must be evaluated separately.

## Addendum: M1-E four-chain computational confirmation (5 November 2026)

**This addendum supersedes the *interpretation* of the isolated boundary-swap observation above, not the historical M1-D2 output table.** See `docs/phase4/M1_E_Experimental_Protocol.md`, `docs/phase4/M1_E_Results.md`, and the preserved generated data in `outputs/phase4/RTM/m1e/`.

The M1-E reanalysis refitted the three distinct diagnostic-warning cases using four NUTS chains, 1,000 tuning and 1,000 retained draws per chain. All three now satisfy the pre-specified R-hat and ESS screening thresholds, with **zero divergences**. The prior warning in the reused M1-C/P0 fit at `gamma=-0.8`, seed 101, and the new P2 warning at `gamma=-0.8`, seed 505, were resolved in their targeted refits. The additional M1-C B0 warning (seed 404) was also resolved. Three previously flagged trace plots were visually reviewed without evident mixing pathologies; review of the six boundary trace plots remains to be recorded.

Most importantly, M1-E refitted **P0, P1, and P2 twice independently** at `gamma=-0.4`, seed 101. **All six refits selected the same top 10 assets**, both across priors and within independent repeats. All six reported synthetic oracle-relative regret of **0.020406**. The new P0 top-10 differs from the original P0 top-10 by **one asset**, while P1/P2 retain their original selection. The original P1/P2-versus-P0 difference in M1-D2 therefore **did not reproduce under longer sampling**.

**Revised conclusion:** The one-seed M1-D2 apparent disadvantage of P1/P2 is **not reliable evidence of a substantive intercept-prior effect on selection**. Finite MCMC noise near a discrete decision boundary is a plausible explanation, although the precise cause has not been uniquely identified. The M1-D2 prior predictive findings and full exploratory tables remain valid as records of their original runs. Broader claims about all M1-D2 scenarios or real flood damage still require further evidence.

**What to test next:** Complete six boundary trace inspections and archive M1-E manifests; then test population/split sensitivity in M1-F. Joint-prior or hazard-scaling follow-up should be driven by a separate, documented hypothesis, not by this unconfirmed one-asset swap.
