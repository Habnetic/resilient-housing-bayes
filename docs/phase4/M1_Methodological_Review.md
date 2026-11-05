# Habnetic | Phase 4 M1: Methodological Review

**Version:** 0.1.0, 5 November 2026<br>
**Status:** Evidence synthesis and decision memo, *draft for repository verification*. M1-A through M1-D2 are exploratory synthetic experiments; this review does not certify empirical flood-risk validity or close all M1 work.<br>
**Study area:** Rotterdam (RTM).<br>
**Scientific baseline:** Phase 3/M0, commit `5e19254`, tag `phase3-code-freeze-20261022`.<br>
**Code references independently recorded in experiment history:** M1-C implementation `e920f8c`, M1-C results/summary `e8d0561`, M1-D1 implementation `047d763`, M1-D1 results `133d7a7`. Verify the M1-D2 implementation/results commit IDs locally before changing this document's status.
**Evidence class:** Measured terrain covariates, **synthetic Bernoulli outcomes**, posterior inference and oracle-based decisions. The two evidence types must never be conflated.

## Executive assessment

**Decision:** Accept M1-A/B/C/D1/D2 as a coherent *exploratory, controlled synthetic-methods programme*. Do **not** classify M1 as an empirically validated flood-risk model, declare any prior universally optimal, or transfer its decision-accuracy claims to another city. Preserve the frozen experiments and move to a short, pre-specified validation programme rather than building an increasingly elaborate model immediately.

The strongest result is conditional and defensible: **when an additional terrain-associated effect is deliberately injected into the synthetic outcome generator, a model containing that variable generally yields better oracle-relative top-10 selections than a model omitting it, on the fixed test population.** The associated benefits persist across five realizations of the synthetic outcomes per effect. Changing priors can alter posterior coefficients and individual decisions without producing a monotone improvement in decision regret. More plausible intercept priors repair much of the original prior-predictive implausibility, while most fitted decisions remain unchanged.

The largest scientific gap is not another prior variant. It is **external relevance**: one fixed building sample, synthetic response, idealized equal-value asset selection, and no observed flood losses or spatial holdout.

## 1. Research question and what M1 actually tests

M1 asks whether richer geospatial covariates and explicitly modeled epistemic uncertainty change *constrained asset selection*, how stable such decisions are, and under which controlled conditions the additional information lowers synthetic expected loss. It does **not** estimate an observed causal effect of relative elevation, flood probability or intervention benefit.

### Generative story

For building `i` on the fixed Rotterdam sample:

```text
T_i = asinh(T50_i / 0.5 m)
logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i
Y_i ~ Bernoulli(p_true_i)
gamma_true in {0, -0.4, -0.8}
```

`E = E_hat_v0` is the Phase 3 exposure feature; `H = H_pluvial_v1_logrel` is the hazard feature. `T50` is an **observed terrain contrast**, not an observed flood outcome. Common random uniforms pair outcome realizations across effect scenarios.

### Likelihood, priors and posterior targets

- **B0:** Bernoulli-logistic regression with intercept, `E` and `H`.
- **B1:** The same likelihood with additional `T` and coefficient `gamma_T`.
- **M1-C reference priors:** independent `alpha, beta_E, beta_H ~ Normal(0, 2.5)`; B1 `gamma_T ~ Normal(0, 0.5)`. Here the second Normal argument is the standard deviation.
- **M1-D1 intervention:** change *only* `gamma_T` prior SD to `0.25`, `0.50` or `1.00`.
- **M1-D2 intervention:** change *only* intercept prior among P0 `N(0, 2.5)`, P1 `N(-3, 1.0)`, P2 `N(-3, 0.5)`, P3 `N(-1.5, 1.0)`, leaving B1's other priors unchanged. P1/P2 use the **known synthetic intercept** and are not externally elicited priors.
- **Posterior targets:** coefficients and uncertainty; posterior predictive probabilities; draw-wise top-*k* inclusion probabilities; selected-set membership; oracle-relative expected selection regret. Diagnostics include R-hat, bulk/tail ESS, divergences, prior predictive and aggregate posterior predictive checks, plus held-out scores.

### Population, decision and loss function

M1-C/D use **5,000 buildings with measured terrain**, a fixed **4,000 training / 1,000 test** random split, and five outcome seeds `101, 202, 303, 404, 505` for each `gamma_true`. The decision is **top 10 of 1,000 test buildings**, by posterior-mean `p_i`. The synthetic oracle uses known `p_true_i` on the *same* 1,000 buildings. With uniform benefits, regret is:

```text
regret = sum(p_true_i over oracle top-10) - sum(p_true_i over selected top-10)
```

This is **not** monetary damage, flood depth, avoided losses, physical resilience, cost effectiveness, or a causal treatment effect. A regret of zero means oracle-equivalent selection under the defined synthetic objective.

## 2. Measured terrain: available evidence and restrictions

The Rotterdam AHN5 5 m terrain extraction joined to **221,324** OSM-derived building records, with **220,456 measured** `T50` values (99.608%) and **868 missing** (0.392%). `T50` is the median local-minus-neighbourhood ground elevation, with local annulus 0–20 m outside the footprint and a neighbourhood ring from 20–100 m. Positive values mean a higher *relative local ground median*, not necessarily hydrological protection.

The documented source-to-mosaic and 200-building extraction checks establish pipeline reproducibility. The 200-building 5 m vs 0.5 m pilot found strong rank agreement (Spearman **0.9804** for T50), but a maximum absolute discrepancy of **0.6698 m**. Extreme, mixed, infrastructure-associated terrain and missing measurements warrant review; raster exactness is **not** hydraulic validity. The 36-asset spatial QA set was generated but not fully adjudicated. The OSM asset universe is also not a certified *residential* building inventory.

**Implication:** M1's `T50` experiment is an informative *methodological sensitivity study*. It cannot establish that T50 measures, predicts or reduces actual flood damage. Citywide prevalence of measurable terrain is not evidence of relevance for the missing 868, nor of unbiased inclusion in the sample.

## 3. Experiment-by-experiment evidence

| Study | Controlled change | Supported observation | Important qualification |
|---|---|---|---|
| **M1-A** negative control | Add terrain although the original synthetic generator has no additional terrain effect | Pilot `gamma` mean about `-0.032`, credible interval includes 0, B0/B1 top-50 overlap **49/50** | Single pilot; one swap or borderline-share movement does not imply a real effect |
| **M1-B** positive control | Inject known `gamma_true=-0.8` in 5,000-asset synthetic sample | Pilot `gamma` posterior ≈ `-0.673` (interval `[-0.994,-0.345]`); top-10 oracle agreement **8/10** (B0) versus **9/10** (B1); regret **0.141091** versus **0.020654** | Designed so B1 contains a genuinely useful generating covariate |
| **M1-C** replication | `gamma_true={0,-0.4,-0.8}`, five paired outcome seeds, B0/B1 | When `gamma=0`, B0/B1 ties in all five; when negative, B1 has lower regret in **5/5** for each effect | One fixed population and test split; five Bernoulli realizations, not geographical replications |
| **M1-D1** terrain prior | B1 `sigma_gamma={0.25,0.5,1.0}` | Narrower prior shrinks `gamma` toward zero; some selected buildings and inclusion probabilities change | No prior dominates regret across both nonzero effects; only five outcomes per condition |
| **M1-D2** intercept prior | P0/P1/P2/P3 intercept priors, with slope priors fixed | P1/P2 sharply lower prior predictive median rate; posteriors and top-10 choices mostly unchanged | Oracle-centred alternatives, broad remaining slope priors, possible MCMC boundary noise |

M1-A and M1-B were pilots, not the same repeated 15-scenario experiment as M1-C. Do not combine their numbers as if they had identical samples, budgets or validation procedures.

### 3.1 M1-C: replicated synthetic selection benefit

Means across **five outcome seeds per effect**:

| `gamma_true` | B0 mean regret | B1 mean regret | Reduction in mean regret | B0/B1 mean top-10 overlap | B1/oracle mean overlap |
|---:|---:|---:|---:|---:|---:|
| `0.0` | 0.000630 | 0.000630 | 0.0% | 10.0/10 | 9.8/10 |
| `-0.4` | 0.056997 | 0.008162 | **85.7%** | 9.0/10 | 9.6/10 |
| `-0.8` | 0.235471 | 0.003015 | **98.7%** | 7.8/10 | 9.8/10 |

The percentage is calculated from **group-mean regret** rather than averaged replicate-level percentage improvements. These results support a *conditional synthetic information-value statement*, not an empirical claim about T50 in Rotterdam. A model can have internally stable posterior selections and still disagree substantially with a better-specified model or with the synthetic oracle.

B1 parameter recovery by effect: mean posterior gamma `+0.0512`, `-0.3052`, `-0.7234`; the five reported 95% intervals per effect each contained the true value. That is a **count**, not a validated 95% coverage rate.

### 3.2 M1-D1: prior sensitivity of the terrain coefficient

| `gamma_true` | Narrow SD 0.25: mean gamma / regret | Baseline SD 0.50 | Wide SD 1.00 |
|---:|---:|---:|---:|
| `0.0` | `+0.0424 / 0.000630` | `+0.0512 / 0.000630` | `+0.0538 / 0.000630` |
| `-0.4` | `-0.2338 / 0.016324` | `-0.3052 / 0.008162` | `-0.3262 / 0.004465` |
| `-0.8` | `-0.5650 / 0.007960` | `-0.7234 / 0.003015` | `-0.7722 / 0.006030` |

For `gamma_true=-0.8`, the widest prior has the smallest mean gamma bias, but **not** the lowest mean regret. Against the SD-0.50 baseline, average selected-set overlap remains 9.2/10 under the narrow prior at `gamma=-0.8`; the greatest observed absolute shift in draw-wise inclusion probability is **0.382**. A non-changing top-*k* can conceal changed posterior boundary uncertainty.

The narrow prior's 95% gamma interval contains the true `-0.8` in **4/5** simulations, whereas the other tested D1 cells report **5/5**. This is too little replication for formal coverage assessment.

### 3.3 M1-D2: intercept calibration versus decision robustness

The prior audit used **2,000 prior draws** for each candidate on the actual **4,000 training covariate vectors**. The following are quantiles of the distribution of the **expected aggregate event rate**, not parameter intervals:

| Intercept prior | 5th | Median | 95th |
|---|---:|---:|---:|
| P0 `N(0, 2.5)` | 3.92% | **51.08%** | 97.12% |
| P1 `N(-3, 1.0)` | 1.48% | **10.43%** | 30.78% |
| P2 `N(-3, 0.5)` | 3.25% | **9.65%** | 24.53% |
| P3 `N(-1.5, 1.0)` | 6.69% | **25.37%** | 51.48% |

Synthetic observed rates are around **6.2%**. P1/P2 substantially reduce the P0 mismatch, although their tails remain broad because the exposure and hazard slope priors were unchanged. The intercept describes `E=H=T=0`, **not** the sample-average event rate. An intercept prior centered on the known generator's `-3` does not ensure a prior predictive median near 6%.

The full M1-D2 study completed **45 new B1 fits**, reused **15 P0 fits**, and compared **60 prior/scenario combinations**. Across effect sizes, mean posterior intercepts remained near `-3`, including under the shifted P3; training posterior predictive *aggregate median* rates stayed around 6.1–6.3%. This does not imply individual or geographic calibration.

| `gamma_true` | P0 mean regret | P1 mean regret | P2 mean regret | P3 mean regret |
|---:|---:|---:|---:|---:|
| `0.0` | 0.000630 | 0.000630 | 0.000630 | 0.000630 |
| `-0.4` | 0.008162 | 0.012243 | 0.012243 | 0.008162 |
| `-0.8` | 0.003015 | 0.003015 | 0.003015 | 0.003015 |

The only increased P1/P2 regret occurred at `gamma=-0.4`, seed **101**: one changed top-10 asset and **+0.020406** case-level synthetic regret; four other seeds tied P0. Differences in held-out log predictive density, Brier and RMSE were small in the reported units. This boundary event could reflect prior sensitivity, MCMC variation or both, so the result is **not** evidence that informative intercept priors are harmful.

## 4. Cross-cutting scientific findings

### 4.1 Distinguish four kinds of evidence

1. **Internal decision uncertainty:** posterior `P(i in Top-k | data, model, prior)`, with chosen capacity and candidate universe.
2. **Model specification sensitivity:** changes when relevant predictors or likelihood assumptions change (B0 versus B1).
3. **Prior sensitivity:** changes under alternate prior distributions within otherwise fixed B1 (D1/D2).
4. **Decision quality:** expected selection regret against an *available synthetic oracle*, separate from fit quality, parameter bias or predictive scores.

These quantities answer different questions. Posterior certainty conditional on a model is **not** evidence that the model is correctly specified. Lower parameter bias does **not** imply lower top-*k* regret. Better prior predictive plausibility does **not** guarantee a different or better posterior decision.

### 4.2 Prior predictive and posterior predictive checks are not interchangeable

- **Prior predictive:** under P0, almost half/half median aggregate events and a 3.9–97.1% middle 90% range are poorly suited to the present synthetic generator; P1/P2 improve plausibility but are *oracle-centred*.
- **Posterior predictive:** aggregate observed and replicated rates near 6% show the model can reproduce *one statistic*. They do **not** validate individual risk ordering, tail prediction, calibration, spatial structure or flood physics.
- **Held-out prediction:** M1-C/D1/D2 calculate predictive metrics; any claim of substantive superiority needs uncertainty accounting and a held-out evaluation design suited to the intended deployment (including spatial transfer). LOO/WAIC is not needed for the first fixed train/test split; consider proper scoring and predictive calibration rather than adding model-selection machinery reflexively.

### 4.3 Diagnostics constrain inference strength

| Experiment | Case | Reported issue | Action |
|---|---|---|---|
| M1-C | `gamma=-0.8`, seed `101`, B1 | Tail ESS **394** (<400) | Inspect trace; targeted longer-chain repeat |
| M1-C | `gamma=-0.8`, seed `404`, B0 | Max R-hat **1.02** (>1.01) | Inspect trace; targeted longer-chain repeat |
| M1-D1 | `gamma=-0.8`, seed `101`, SD `0.50` | Same **reused** B1 tail ESS 394 | Do not count as a distinct new failure |
| M1-D2 | `gamma=-0.8`, seed `101`, P0 | Same **reused** B1 tail ESS 394 | Do not count as independent evidence |
| M1-D2 | `gamma=-0.8`, seed `505`, P2 | Tail ESS **373** (<400), new fit | Inspect trace; targeted longer-chain repeat |

These are warnings rather than demonstrated model failure. Flag counts across experiments **must not be added together** as though reused fits were independent. Reported M1-C fits had no divergences; for new runs, verify fit-level divergence data before a blanket statement. Most experiments use **two chains with 500 retained draws each**: adequate for exploratory prototyping but below the planned confirmatory standard of at least four chains and longer sampling. Differences of one selected building warrant particular caution when posterior probabilities are close at the decision boundary.

### 4.4 Scaling and interpretation remain unsettled

The M1-D2 training audit gives `E` SD **0.81610** with a positive extreme of **7.99**, `H` SD **0.007618**, and transformed `T` SD **0.41293**. The negligible numerical spread in `H` means its coefficient has weak likelihood information at the current scale; its numerical coefficient should **not** be interpreted as a reliable physical hazard effect. Rescaling `H` for comparability would require explicitly transforming the coefficient and its prior, and checking what is truly identifiable. Do not simply standardize while silently leaving its prior unchanged and call the results comparable.

## 5. Known limitations and potential failure modes

| Limitation / failure mode | Why it matters | Required response |
|---|---|---|
| **Synthetic outcomes** | Only the injected generative signal is known; no observed flood losses | State `outcome_evidence=synthetic` on every artifact and visualization |
| **Oracle-centred priors** | P1/P2 benefit from knowledge unavailable in real deployment | Separate prior elicitation from synthetic parameter-recovery checks |
| **Fixed asset population and split** | Outcome seeds do not test other samples, spatial generalization or changes in population | Independent building samples and spatially blocked evaluation |
| **Top-10 budget with equal consequences** | May exaggerate the importance of a few boundary swaps; not a realistic allocation model | Vary `k`, eligibility, vulnerability, benefit, cost and feasibility one at a time |
| **Measured terrain ≠ hydraulic variable** | Relative terrain rings ignore drainage, barriers, flow and defences | Mechanism/measurement sensitivity and external validation |
| **Missing/extreme terrain** | 868 citywide missing; mixed landforms and infrastructure | Explicit missingness and asset-definition sensitivity |
| **E/H predictor design** | Broad `E` prior and near-constant `H` complicate prior checks and identification | Scale-aware priors and controlled hazard-design audit |
| **Limited MCMC** | Top-k swaps may reflect sampler variation as well as meaningful prior change | Refit flagged/boundary cases using 4 chains, longer draws and matched design |
| **Unarchived generated artifacts** | `outputs/**` is Git-ignored, so code alone does not reproduce numeric findings | Preserve manifests, input SHA-256, configs, environment and generated bundles in versioned research storage |

No inference here directly quantifies **hazard → vulnerability → impact → recovery**, measures within-building component hierarchies, estimates actual probability of flooding, or tests transfer to Hamburg/Donostia.

## 6. Methodological decisions and ordered next experiments

### Decision M1-REV-01: Keep the synthetic evidence, narrow the claim

**Accepted claim:** On the fixed RTM synthetic setup, inclusion of an intentionally causal-generative terrain term increases oracle-relative top-10 selection quality for negative injected terrain effects; prior changes can affect coefficients, posterior selection uncertainty and occasional top-10 membership.

**Rejected claims:** `T50` has a verified physical flood effect; the posterior predicts real flood damage; a prior or model is optimal across deployment contexts; 5 repeated outcomes imply geographical robustness; stable posterior inclusion probabilities certify model validity.

**Release classification:** M1-A/B/C/D are **exploratory synthetic evidence**, not confirmed or empirical risk validation. Do not replace or rewrite frozen experiment outputs to make the conclusions stronger.

### Priority 1: Targeted computational confirmation, not another full grid

- Inspect saved traces and fit-level diagnostics for the **three distinct flagged fits** above (M1-C B1/seed101; M1-C B0/seed404; M1-D2 P2/seed505 at `gamma=-0.8`). Refit only if needed using **4 chains**, at least **1,000 post-tuning draws per chain**, and enough tuning for stable diagnostics.
- Recheck the **M1-D2 `gamma=-0.4`, seed101 P0/P1/P2** top-10 boundary with matched higher-quality sampler settings. Determine whether the *same asset swap* is reproducible and report inclusion probability and regret changes with their Monte Carlo sensitivity.
- Acceptance gate: `R-hat <= 1.01`, reasonable bulk/tail ESS, no unexplained divergences, trace review, stable key decisions under repeated sampling. A flagged result is not automatically rejected, but must be explicitly qualified.

### Priority 2: Independent population and decision sensitivity

Before spatial transfer, conduct a **small, pre-registered factorial check** that varies *building sample*, *split* and *selection budget* independently. Begin with two or three additional building samples (or spatially separated test blocks), one or two budget settings, and selected `gamma_true` levels; do not immediately multiply all old prior grids. Keep scenario-level pairing and report uncertainty in mean regret and top-*k* overlap across *sampling units*, not just repeated Bernoulli labels.

Acceptance gate: direction of conditional synthetic B1 benefit is not solely driven by one fixed population, and degradation modes are transparently reported. With spatial blocks, examine support shift and asset composition rather than interpreting every failed transfer as a model bug.

### Priority 3: Covariate and terrain-design sensitivity

- Investigate whether `H` is nearly constant because of feature engineering/normalization; assess effective predictor contribution, posterior-predictive sensitivity and scale-aware priors. Report identifiable quantities (`beta_H * H`) rather than celebrating unstable raw coefficients.
- Perform a limited, paired `T50` versus `T25` or alternative `asinh` scale test **after** defining what physical contrast is relevant; avoid choosing the transform post hoc by oracle regret.
- Audit the subset of buildings with extreme, weak-coverage or missing terrain and distinguish physical objects from intervention-eligible homes. Do not silently drop unmeasured observations in an eventual citywide claim.

### Priority 4: Empirical relevance and eventual transfer

Identify an observed outcome and target estimand, e.g., independently documented event occurrence, claims or damage with time and spatial provenance. Model vulnerability, consequences, action feasibility and unequal costs separately before asserting practical intervention value. Then attempt blocked validation within Rotterdam and only afterward a carefully defined cross-city protocol for HAM/DON, with covariate/support comparability and local calibration checks. If suitable outcomes remain unavailable, label the work as **synthetic decision-method validation** instead of field prediction.

## 7. Reproducibility and evidence ledger

**GitHub is the source of truth for code, tests, protocols and this review; the derived experimental `outputs/**` remain ignored** and must be archived in an identified research dataset location with verifiable hashes. The independent data repository holds the source terrain features and underlying geospatial assets. A scientific bundle should preserve the `analysis_kind`, `outcome_evidence`, code commit, input hashes, seeds, environment, split IDs, parameterization, diagnostics and decision metrics, following `Experiment_Artifact_Contract.md` where adopted. No public artifact should expose private local filesystem paths.

| Source / artifact | Purpose | Reproduce from `resilient-housing-bayes` root |
|---|---|---|
| `docs/phase4/M1_C_Experimental_Protocol.md`, `M1_C_Results.md` | Fixed design and M1-C evidence | `python -m rhb.pipelines.summarize_m1c` |
| `outputs/phase4/RTM/m1c/full/summary/` | Paired cases, diagnostics, summary and source hashes | Depends on preserved full-run CSVs/manifest |
| `docs/phase4/M1_D_Experimental_Protocol.md`, `M1_D1_Results.md` | D1 prior-width design and results | `python -m rhb.pipelines.summarize_m1d` |
| `outputs/phase4/RTM/m1d/full/summary/` | D1 recalculable aggregates | Depends on preserved M1-C and D1 fit artifacts |
| `docs/phase4/M1_D2_Experimental_Protocol.md`, `M1_D2_Results.md` | D2 intercept-prior design and results | `python -m rhb.pipelines.summarize_m1d2` |
| `outputs/phase4/RTM/m1d2/prior_predictive/` | Joint-prior rate audit, predictor scales, hashes | `python -m rhb.experiments.m1d2_prior_predictive` |
| `outputs/phase4/RTM/m1d2/full/summary/` | D2 recalculable aggregates and flags | Depends on preserved M1-C and D2 fit artifacts |
| `../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet` | Citywide measured terrain from data repo | See terrain extraction/data-quality report |
| `outputs/phase3/RTM/phase3_features_scaled.parquet` | Phase 3 predictor snapshot | Version/hash must be retained externally |

**Minimal local audit commands** (with `.venv` and correct `PYTHONPATH`):

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1c.py tests/test_m1c_metrics.py tests/test_m1c_summary.py tests/test_m1d.py tests/test_m1d2.py -q
python -m rhb.pipelines.summarize_m1c
python -m rhb.pipelines.summarize_m1d
python -m rhb.experiments.m1d2_prior_predictive
python -m rhb.pipelines.summarize_m1d2
```

**Inputs:** Existing generated M1-C/D1/D2 full-run outputs, manifests and the phase3/terrain Parquet data; Python project environment (3.11.x in this study). **Outputs:** Regenerated summaries under ignored `outputs/phase4/RTM/`. **Reproduction limit:** these commands **do not** independently recreate the entire posterior history or raw raster acquisition; they recompute aggregates/audit from available inputs. For full reproduction, use each experiment's protocol and run command and retain hashes/data sources. Do not re-run citywide raster scripts over existing files without checking their overwrite semantics.

## 8. Review checklist and decision log

- [x] Separate observed geospatial features from synthetic target outcomes.
- [x] Distinguish likelihood, priors, generative process, posterior quantities and decision loss.
- [x] Summarize M1-C, D1 and D2 from reported full-run outputs.
- [x] Record documented MCMC warnings without double counting reused fits.
- [x] Keep fixed population and idealized loss limitations explicit.
- [ ] Verify current Git HEAD / whether final M1-D2 results document was committed.
- [ ] Confirm input and result-manifest hashes and archive generated bundles outside Git.
- [ ] Review flagged traces and top-k boundary Monte Carlo robustness.
- [ ] Produce the minimal independent-sample/spatial-block experiment protocol.
- [ ] Select and justify future real-world outcome evidence and deployment estimand.

**Review decision:** Proceed to targeted diagnostic checks and a *new-population/holdout* experiment; **pause expansion of prior grids** until those tests provide a reason to do so. Keep the Phase 3 freeze, M1-C and M1-D outputs immutable.

**What to test next:** First, the three unique flagged fits and M1-D2 seed-101 boundary. Then an independently sampled or spatially blocked population. Only afterward reassess the scientific need for additional terrain/hazard complexity or cross-city transfer.
