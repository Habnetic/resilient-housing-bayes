# Habnetic | M1-G2 Visual MCMC Trace Review

**Review date:** 2026-11-15  
**Scope:** 12 PNG trace diagnostics: 3 cities (DON, RTM, HAM) × 2 synthetic terrain scenarios (γ = 0 and γ = −0.8) × 2 fitted models (B0 and B1).  
**Inputs reviewed:** `M1G2_traces_review.zip` (12 PNGs), accompanying automated audit `AUDIT_REPORT.md` and `audit_results.json`.  
**Status:** **VISUAL_TRACE_REVIEW_PASS_WITH_LIMITATIONS**. This is a visual diagnostic review, not final scientific acceptance or empirical flood-risk validation.

## 1. Inspection criteria and observations

The plots show four post-warmup chains per fit, with 1,000 retained draws per chain. For all 12 fits and all displayed parameters:

- **Stationarity:** No visually persistent upward/downward drift or sustained regime shift was detected.
- **Between-chain overlap:** The four chains occupy comparable ranges throughout the retained draws. Their marginal histograms overlap substantially.
- **Mixing:** Frequent movement through the bulk of the posterior is visible; no persistent chain trapping or isolated alternative mode is apparent.
- **Marginal shape:** Displayed marginal posterior distributions are broadly unimodal. Width differs considerably by parameter; this is not, by itself, poor MCMC mixing.
- **Terrain parameter (B1 only):** `gamma_T` is centered near zero in γ = 0 controls and near −0.8 in γ = −0.8 positive controls, with overlapping marginal distributions across chains.

These observations agree with the automated checks: no divergences, R-hat at or below approximately 1.01, adequate bulk/tail ESS, and BFMI well above typical low-BFMI warning thresholds.

## 2. File-by-file visual review

| City | Scenario | Model | Visual conclusion |
|---|---|---|---|
| DON | γ = 0 | B0 | PASS: alpha, beta_E, beta_H show overlap and no persistent trends. |
| DON | γ = 0 | B1 | PASS: same; gamma_T fluctuates around zero without visible chain separation. |
| DON | γ = −0.8 | B0 | PASS: traces mix; beta_E shifts under the intentionally omitted predictor (interpretational issue, not a sampler failure). |
| DON | γ = −0.8 | B1 | PASS: gamma_T clusters around −0.81 with overlapping chains. |
| RTM | γ = 0 | B0 | PASS: traces are stationary and between-chain histograms overlap. |
| RTM | γ = 0 | B1 | PASS: gamma_T close to zero; no visible drift or mode trapping. |
| RTM | γ = −0.8 | B0 | PASS: no visual sampler warning despite omitted-variable model misspecification. |
| RTM | γ = −0.8 | B1 | PASS: gamma_T centered around −0.80 and all chains overlap. |
| HAM | γ = 0 | B0 | PASS: four chains overlap across all displayed parameters. |
| HAM | γ = 0 | B1 | PASS: gamma_T near −0.017 with visibly overlapping chains. |
| HAM | γ = −0.8 | B0 | PASS **for MCMC only**: broad beta_H posterior centered far above its B1 estimate merits methodological interpretation. |
| HAM | γ = −0.8 | B1 | PASS: gamma_T near −0.819; no persistent inter-chain separation. |

## 3. Issues that must not be confused with sampler failure

### Weakly resolved hazard coefficient

`beta_H` has wide marginal posteriors in several fits, including models fitted to very large populations. As one example, the automated audit reports HAM, γ = −0.8:

- B0: posterior `beta_H` mean **5.905**, 95% HDI **[3.886, 7.756]**.
- B1: posterior `beta_H` mean **0.506**, 95% HDI **[−1.573, 2.452]**.

This substantial change is consistent with the model-specification contrast, but the traces alone cannot determine whether weak predictor variation, collinearity, covariate scaling, or other posterior dependencies are responsible. No causal or physical interpretation of the hazard coefficient follows from the plots.

**Targeted check recommended:** inspect the empirical distributions/standard deviations of `H`, `E`, and transformed `T` by city, plus pairwise predictor correlations and posterior parameter correlations. Include joint pair plots if there are concerning dependencies. Do **not** rerun NUTS merely because marginal intervals are broad.

### Limits of visual traces

- These figures show univariate marginal histograms and time-series traces, not joint posterior geometry, pairwise correlations, energy diagnostics or spatial residual dependence.
- The audit's posterior predictive checks are in-sample and synthetic.
- MCMC convergence is not proof of model correctness, physical validity, or transferability.

## 4. Decision

**Visual trace gate: PASS for all 12 fits. No NUTS refit is indicated by these trace figures and the available automatic convergence diagnostics.**

**Full M1-G2 scientific acceptance: NOT YET ASSIGNED.** Remaining independent review gates:

1. GIS/topography QA: vertical datums, CRS, DEM coverage; spatially inspect DON T50 extremes and 868 RTM terrain exclusions.
2. Compare findings to M1-G1 held-out experiments and document any differences.
3. Conduct a focused covariate-identifiability diagnostic for `beta_H` and document appropriate interpretation.
4. Preserve large NetCDF/Parquet results and manifests with verified SHA256 outside Git, then commit all currently untracked M1-G2 source code, configs, tests and documentation before tagging.

## 5. Paper-ready methods sentence

> Visual review of trace plots for all twelve full-population NUTS fits showed overlapping chains, broadly unimodal marginal posteriors, and no persistent post-warmup drift or chain trapping. These observations were consistent with the automated R-hat, effective-sample-size, divergence, and BFMI diagnostics. This evidence supports numerical convergence of the synthetic experiments but does not establish empirical flood-risk validity.
