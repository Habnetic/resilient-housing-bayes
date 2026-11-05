# Habnetic | Phase 4, M1-D1: Gamma Prior Sensitivity Results

**Status:** Exploratory full run completed, 5 November 2026.  
**Implementation reference:** `047d763` (`phase4/m1-topography`).  
**Preceding experiment:** M1-C, implemented at `e920f8c` and documented at `e8d0561`.  
**Scope:** Sensitivity of Bayesian synthetic selection decisions to the prior width of the topographic coefficient. **Not** empirical validation of flood risk or of a causal terrain effect.

## 1. Research question

How much do inference and constrained top-*k* decisions depend on the prior standard deviation assigned to the topographic coefficient, holding the observed synthetic outcomes, likelihood, covariates, split and decision rule fixed?

M1-C showed that adding the known synthetic topographic signal can lower expected selection regret. M1-D1 tests whether that conclusion is sensitive to an analyst's choice of prior for the added coefficient.

## 2. Experimental specification

### Generative story and data

- Fixed Rotterdam-derived population: **5,000 buildings**, with **4,000 training** and **1,000 held-out** observations.
- Covariates: exposure `E`, hazard `H`, and terrain transformation `T = asinh(T50 / 0.5 m)`.
- Synthetic outcomes: `Y_i ~ Bernoulli(p_true_i)` with `logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i`.
- Effect scenarios: `gamma_true ∈ {0, -0.4, -0.8}`.
- Outcome seeds: `101, 202, 303, 404, 505`; outcomes are paired across effect sizes using common uniform variates.
- The building sample, features, train/test split and known oracle probabilities are fixed across outcome replications.

### Likelihood and priors

- **B0 (reference):** Bayesian logistic regression with `E` and `H`.
- **B1 (terrain):** Bayesian logistic regression with `E`, `H` and `T`.
- Shared priors: `alpha, beta_E, beta_H ~ Normal(0, 2.5)` independently.
- The **only changed model assumption** is `gamma_T ~ Normal(0, sigma_gamma)`, with `sigma_gamma ∈ {0.25, 0.50, 1.00}`. Here `Normal(mu, sigma)` denotes a distribution parameterized by standard deviation.
- The B1 `sigma_gamma=0.50` fits and B0 reference results are reused from M1-C. The narrow and wide B1 variants require **30 new fits**.
- Exploratory inference: **2 NUTS chains**, 500 tuning steps and 500 retained draws per chain, `target_accept=0.9`.

### Posterior quantities and decision rule

- Posterior mean and 95% interval for `gamma_T`; empirical interval inclusion of the known `gamma_true`.
- Held-out prediction via the posterior; held-out log predictive score relative to B0.
- For each B1 prior, select the **top 10 of 1,000 held-out assets** using posterior mean event probability.
- Compare selections with the same synthetic oracle, defined by known `p_true`, and with B1 at `sigma_gamma=0.50`.
- **Expected selection regret** is `sum(p_true over oracle-selected) - sum(p_true over model-selected)`. It measures loss in this idealized selection objective, **not** actual prevented flood damage, where vulnerability, consequences and intervention effectiveness would matter.
- Within-model uncertainty is summarized through draw-wise top-*k* inclusion probabilities; the maximum absolute change in inclusion probability relative to baseline measures prior sensitivity of these posterior quantities.

**Experiment size:** 3 true effects × 5 outcome seeds × 3 prior widths = **45 B1 prior comparisons**, using **30 new** and **15 reused** B1 fits. B0 comparisons use the existing M1-C reference outputs.

## 3. Parameter recovery

Means across five outcome realizations for each effect/prior combination:

| True `gamma` | Prior SD | Mean posterior `gamma` | Mean bias (estimate − truth) | 95% interval includes truth |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.25 | +0.0424 | +0.0424 | 5/5 |
| 0.0 | 0.50 | +0.0512 | +0.0512 | 5/5 |
| 0.0 | 1.00 | +0.0538 | +0.0538 | 5/5 |
| −0.4 | 0.25 | −0.2338 | +0.1662 | 5/5 |
| −0.4 | 0.50 | −0.3052 | +0.0948 | 5/5 |
| −0.4 | 1.00 | −0.3262 | +0.0738 | 5/5 |
| −0.8 | 0.25 | −0.5650 | +0.2350 | 4/5 |
| −0.8 | 0.50 | −0.7234 | +0.0766 | 5/5 |
| −0.8 | 1.00 | −0.7722 | +0.0278 | 5/5 |

**Interpretation.** The narrower prior shrinks the estimated terrain coefficient more strongly toward zero. At `gamma_true=-0.8`, the widest prior yields the smallest mean coefficient bias, but this does **not** imply optimal decision performance. The single interval miss under the narrow prior is a diagnostic observation, not a reliable estimate of frequentist coverage: five replications per condition are insufficient.

## 4. Expected selection regret

Mean expected selection regret across five seeds; **lower is better**:

| True `gamma` | B0, no terrain | B1, SD 0.25 | B1, SD 0.50 | B1, SD 1.00 |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.000630 | 0.000630 | 0.000630 | 0.000630 |
| −0.4 | 0.056997 | 0.016324 | 0.008162 | **0.004465** |
| −0.8 | 0.235471 | 0.007960 | **0.003015** | 0.006030 |

Across the five outcome seeds, every B1 prior improves on B0 in all five cases at both nonzero effects; all B1 variants tie B0 on regret when `gamma_true=0`.

At `gamma_true=-0.4`, the widest prior has the lowest *mean* regret; at `gamma_true=-0.8`, the baseline prior has the lowest mean regret despite the wide prior having smaller mean parameter bias. This illustrates that coefficient estimation and decisions under a discontinuous top-*k* selection rule should be evaluated separately. **It does not establish a universally optimal prior:** the sample and oracle are fixed, the number of outcome replications is small, and mean regret can change when a single borderline building is exchanged.

## 5. Prior-induced selection changes

Mean overlap with the baseline B1 selection (`sigma_gamma=0.50`), out of 10 held-out assets:

| True `gamma` | SD 0.25 | SD 0.50 | SD 1.00 |
| ---: | ---: | ---: | ---: |
| 0.0 | 10.0 | 10.0 | 10.0 |
| −0.4 | 9.6 | 10.0 | 9.6 |
| −0.8 | 9.2 | 10.0 | 9.8 |

Maximum **absolute change in posterior top-*k* inclusion probability** relative to the baseline, across the five seeds and evaluated buildings:

| True `gamma` | SD 0.25 | SD 1.00 |
| ---: | ---: | ---: |
| 0.0 | 0.091 | 0.023 |
| −0.4 | 0.204 | 0.075 |
| −0.8 | **0.382** | 0.135 |

At `gamma_true=-0.8`, the narrow prior can alter one building's estimated top-*k* inclusion probability by **38.2 percentage points**, even though the mean selected-set overlap is still 9.2/10. A stable selected set therefore does not imply that posterior uncertainty or boundary membership probabilities are insensitive to priors.

Three related quantities must remain distinct:

1. **Within-model posterior decision stability:** draw-wise inclusion probabilities conditional on a chosen model and prior.
2. **Model/specification sensitivity:** changed decisions when the predictors or likelihood are changed (M1-C's B0 versus B1).
3. **Prior sensitivity:** changed decisions and inclusion probabilities under alternative priors with otherwise fixed model structure (M1-D1).

## 6. Held-out predictive scores

Mean held-out log predictive score difference, **B1 minus B0** (positive favors B1 under the implemented scoring convention):

| True `gamma` | SD 0.25 | SD 0.50 | SD 1.00 |
| ---: | ---: | ---: | ---: |
| 0.0 | +0.164 | +0.221 | +0.237 |
| −0.4 | +0.339 | +0.352 | +0.360 |
| −0.8 | +1.444 | +1.425 | +1.380 |

These are descriptive results for the single fixed held-out split, not evidence of out-of-city generalization or statistically resolved prior superiority. Do not use a difference of a few hundredths as a model-selection verdict without a suitable uncertainty analysis. The scoring metric and reference data must remain unchanged if future experiments are compared.

## 7. Checks and diagnostics

**Prior predictive checks.** The mean of the per-fit prior predictive median synthetic event rates is approximately **48–52%**, whereas the simulated realized event rates are approximately **6%**. This mismatch is substantial. Altering only the terrain-coefficient prior does not repair broad priors for the intercept and other coefficients. The observed synthetic prevalence should not simply be inserted into a purportedly independent prior without acknowledging data reuse.

**Posterior predictive checks.** M1-C's pipeline checks replicated training-set event rates; the current summaries do not constitute a full calibration assessment. M1-D2 should explicitly check prior predictive event-rate distributions, and future validation should add calibration and spatial checks alongside held-out scores.

**MCMC checks.** The M1-D1 summary records **one diagnostic flag** among the B1 variants: the **reused** `sigma_gamma=0.50`, `gamma_true=-0.8`, seed `101` fit has tail ESS **394**, below the protocol threshold of 400. The **30 new narrow/wide B1 fits have no diagnostic flags in the reported summary**. The original M1-C audit also flagged a B0 reference fit at `gamma_true=-0.8`, seed `404` with maximum `R-hat=1.02`; it is not included in the M1-D1 B1-prior flag count. No reported divergences were present in the M1-C audit or the M1-D1 smoke log; full-run divergence counts should be verified from fit-level outputs before a stronger claim. Two chains with 500 retained draws remain an exploratory rather than confirmatory inference configuration.

**Sensitivity still untested.** Prior center (rather than only width), baseline-rate prior, exposure/hazard coefficient priors, transformation of `T50`, population sampling, train/test split, spatial blocks, decision budget, unequal asset consequences, missing-terrain mechanism, and city transfer.

## 8. Conclusions

**Supported by this controlled synthetic experiment:**

- Stronger prior regularization of `gamma_T` moves negative effect estimates toward zero.
- The selected top-*k* assets and their posterior inclusion probabilities can change when only the prior width changes.
- Lower average coefficient bias does not necessarily coincide with lower average selection regret.
- All three tested B1 priors substantially outperform the no-terrain B0 reference in mean expected regret when the synthetic generator includes a nonzero terrain effect; this behavior is **conditional on the generator and fixed sample**.

**Not supported:** identification of a physically correct prior, a real protective effect of relative elevation, validated flood-damage probabilities, transferability, robust credible-interval coverage, or general decision superiority beyond this synthetic design.

## 9. Reproducibility

From the `resilient-housing-bayes` repository root with the recorded Python environment and M1-C source outputs available:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1d.py -q
python -m rhb.pipelines.run_m1d --dry-run
python -m rhb.pipelines.run_m1d
python -m rhb.pipelines.summarize_m1d
```

**Inputs:** `configs/m1d_gamma_prior_rtm.json`, the fixed M1-C synthetic scenarios/features and M1-C reference fit outputs, plus the corresponding data files.  
**Primary result source:** `outputs/phase4/RTM/m1d/full/comparison_results.csv`.  
**Aggregated result:** `outputs/phase4/RTM/m1d/full/summary/summary_by_gamma_prior.csv`.  
**Versioning:** this document and the executable pipeline belong in Git; generated `outputs/**` are ignored and must be retained or archived separately with the manifest/configuration and data provenance. Do not assume the Git commit alone preserves results.

## 10. What to test next

**M1-D2: baseline-rate and shared-coefficient prior sensitivity.** Before fitting, inspect the scale and centering of `E` and `H`, determine what the intercept represents at `E=H=T=0`, and compare candidate prior specifications through **prior predictive simulations**. Use scientifically defensible prevalence ranges rather than silently calibrating priors to the test outcomes. Then compare inference, calibration, held-out scoring, top-*k* selection and regret on the frozen synthetic scenarios. Follow with longer four-chain runs for any conclusion intended to be confirmatory, and eventually vary sample, geography and budget.
