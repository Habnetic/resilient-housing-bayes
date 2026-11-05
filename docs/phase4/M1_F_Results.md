# Habnetic | Phase 4 M1-F: Population robustness results

**Version:** 0.3.0, 5 November 2026<br>
**Status:** **ACCEPTED as an exploratory within-Rotterdam synthetic population-robustness experiment.** Full-run numeric diagnostic audit and independent population-ID integrity verification completed on 5 November 2026. One low-tail-ESS fit is retained with an explicit warning; source-manifest hash verification, asset-level exception review, and independent archival remain open. No geographical or empirical flood-risk validation is claimed.<br>
**Protocol:** `docs/phase4/M1_F_Experimental_Protocol.md`<br>
**Implementation:** `src/rhb/experiments/m1f_population_sampling.py`, `src/rhb/pipelines/run_m1f.py`, `src/rhb/pipelines/summarize_m1f.py`<br>
**Generated evidence:** `outputs/phase4/RTM/m1f/` (Git-ignored). The numbers below were transcribed from the full-run summary tables reported on 5 November 2026; use the generated CSVs and manifests as the recalculable source.

## 1. Research question and design

Does the conditional synthetic decision benefit of adding a measured topographic feature persist when the underlying set of Rotterdam buildings changes, rather than merely when synthetic Bernoulli outcomes are regenerated on one fixed sample?

- **Universe:** Rotterdam assets with measured 5 m terrain features. The sampling protocol requires 220,456 measured assets, exclusion of all 5,000 M1-C buildings, and three new **mutually disjoint** samples (`pop01`, `pop02`, `pop03`) of 5,000 buildings each. The generator enforces these conditions, and the separate post-run integrity script independently verified the selected IDs and splits (Section 6). The complete source-manifest hash chain has not been independently inspected.
- **Split:** 4,000 train and 1,000 held-out test buildings per population. The same assets and split are retained across the two true-effect levels and outcome seeds within a population.
- **Generative story:** `Y_i ~ Bernoulli(p_true_i)` with `logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i`, `T_i = asinh(T50_i / 0.5 m)`, and `gamma_true in {0, -0.8}`. Synthetic outcomes are produced using seeds 101 and 202 per population, with paired random uniforms across effect levels.
- **Likelihood:** Bayesian Bernoulli-logistic models: B0 uses exposure E and hazard H; B1 adds topography T.
- **Priors:** `alpha, beta_E, beta_H ~ Normal(0, 2.5)` in both models; B1 additionally uses `gamma_T ~ Normal(0, 0.5)`; independent coefficients. These are the original M1-C priors.
- **Posterior decision:** Select the 10 largest posterior-mean event probabilities among 1,000 test buildings (1%). Compare with an **oracle** using known generating probabilities. Expected selection regret is the sum of oracle-selected `p_true` minus the sum of model-selected `p_true`; lower is better. This idealized metric assumes equal assets, costs and perfectly effective intervention, not observed prevented flood losses.
- **Inference:** Two NUTS chains, 500 tuning plus 500 posterior draws per chain, `target_accept=0.9`. This is an exploratory population screen; four-chain M1-E confirmation applies only to its separately registered cases.

The full summary accounts for **12 paired population/effect/seed cases**, **24 B0/B1 fits**, three populations, two effect sizes and two outcome seeds per effect/population. The outcome seeds are repeated realizations on the same buildings, **not six independent geographical samples**.

## 2. Primary result: synthetic decision regret

Figures below are means over six paired cases per effect, from `full_summary/summary_by_gamma.csv`.

| True terrain effect | B0 mean regret | B1 mean regret | Mean paired improvement (B0 − B1) | B1 wins / ties / B0 wins |
|---|---:|---:|---:|---:|
| `gamma_true = 0.0` | 0.000000 | 0.001958 | −0.001958 | 0 / 5 / 1 |
| `gamma_true = -0.8` | 0.147047 | 0.022663 | +0.124384 | 5 / 1 / 0 |

For `gamma_true = -0.8`, the reduction in **mean synthetic oracle regret** is approximately **84.6% relative to mean B0 regret**; this percentage is a ratio of means, not an average of case-level percentage reductions. B1 does **not** outperform B0 in every case: one is a tie.

For the negative control, `gamma_true = 0`, both models tie in five cases, and B1 incurs additional regret in **one** case. The mean B1 disadvantage (0.001958) should be retained in the result instead of silently describing all null-effect selections as identical.

### 2.1 Population-specific results

These are means of **two outcome realizations** per population and effect, from `full_summary/summary_by_population_gamma.csv`.

| Population | True effect | B0 mean regret | B1 mean regret | Mean B1 improvement | B1 wins / ties / B0 wins |
|---|---:|---:|---:|---:|---:|
| pop01 | −0.8 | 0.158116 | 0.027597 | +0.130519 | 2 / 0 / 0 |
| pop02 | −0.8 | 0.235084 | 0.016423 | +0.218662 | 2 / 0 / 0 |
| pop03 | −0.8 | 0.047941 | 0.023971 | +0.023971 | 1 / 1 / 0 |
| pop01 | 0.0 | 0.000000 | 0.000000 | 0.000000 | 0 / 2 / 0 |
| pop02 | 0.0 | 0.000000 | 0.000000 | 0.000000 | 0 / 2 / 0 |
| pop03 | 0.0 | 0.000000 | 0.005875 | −0.005875 | 0 / 1 / 1 |

The **direction of mean positive-control improvement** is consistent across all three new random Rotterdam samples, but the magnitude varies substantially. `pop03` is the weakest positive-control population and also contains the negative-control degradation. The source tables should be inspected per seed before attributing either to population composition or MCMC approximation.

## 3. Held-out prediction and coefficient recovery

Reported means over six paired cases per effect from the full summary. Score differences are **B1 minus B0**. For held-out log predictive density (ELPD), a positive value favours B1; for Brier score and RMSE, a negative value favours B1.

| True effect | Mean B1 γ estimate | 95% γ interval includes truth | Mean Δ held-out ELPD | Mean Δ Brier | Mean Δ RMSE to known `p_true` |
|---|---:|---:|---:|---:|---:|
| −0.8 | −0.776833 | 6/6 | +1.815913 | −0.000165703 | −0.013688617 |
| 0.0 | +0.024833 | 6/6 | −0.227289 | +0.0000375524 | +0.001424316 |

When terrain has a true synthetic effect, the extended model improves the mean held-out predictive metrics as well as decision regret. Under the null control, B1 slightly worsens average held-out scores. These scores come from the same fixed test split per population. The **6/6 interval-inclusion counts are descriptive**, not evidence of calibrated nominal 95% coverage with so few outcomes.

## 4. Posterior inclusion and decision sensitivity

The original `pop01` / seed 101 smoke case showed:

| True effect | B0/B1 top-10 overlap | B0 oracle overlap | B1 oracle overlap | B0 regret | B1 regret |
|---|---:|---:|---:|---:|---:|
| 0.0 | 10/10 | 10/10 | 10/10 | 0 | 0 |
| −0.8 | 7/10 | 7/10 | 10/10 | 0.158116 | 0 |

For the positive-control smoke case, the largest absolute shift in posterior top-10 inclusion probability was **1.000** for at least one asset, despite a small mean shift over all test assets. These are specific smoke-case observations, **not population-wide averages**. Complete per-population inclusion and oracle-overlap metrics are available in the generated full summary, but are not reproduced here because they were not included in the supplied full-run console tables.

Selection changes near the top-10 boundary should not automatically be attributed to a substantive model difference: targeted M1-E refits showed that finite MCMC approximation can change borderline membership even when standard convergence screens pass.

## 5. Prior predictive, PPC and sampling diagnostics

**Prior predictive limitation:** M1-F retains the original diffuse Phase 3 / M1-C priors. In the `pop01` smoke cases, B1 prior predictive median aggregate event rates were approximately 0.614 (`gamma=0`) and 0.486 (`gamma=-0.8`), versus observed synthetic training rates 0.063 and 0.065. These are smoke-case figures only. M1-D2 showed that intercept-prior alternatives can improve prior predictive plausibility without establishing a universally superior decision prior. Do not present M1-F priors as independently calibrated.

**Posterior predictive:** In the same smoke cases, posterior predictive median aggregate event rates matched the training-event rates closely (0.063 and 0.065). Aggregate event-rate agreement is only a partial PPC. Full fit records contain further information that has not been independently checked here.

**Full-run MCMC diagnostics: AUDITED, 1 of 24 fits flagged.** The `full_summary/diagnostic_flags.csv` check identified exactly one exploratory fit with a registered threshold warning:

| Population | True gamma | Outcome seed | Model | R-hat max | Bulk ESS min | Tail ESS min | Divergences | Flag |
|---|---:|---:|---|---:|---:|---:|---:|---|
| `pop02` | −0.8 | 202 | B0 | 1.01 | 415 | **331** | 0 | Tail ESS below 400 |

The registered screen flags R-hat `>1.01`, bulk/tail ESS `<400`, or any divergence. This case has **only the tail ESS warning**; no divergences were reported. Its original selected top-10 has **7/10 oracle overlap** and **expected synthetic regret 0.235084**. The low tail ESS does **not** establish that this regret is caused by sampling approximation; B0 omits a terrain effect present in the generator.

**Decision:** Retain this single fit and its warning for the exploratory summary; **do not claim that all 24 fits pass diagnostics**. A targeted four-chain rerun would be appropriate if this specific case later supports a confirmatory claim. A new 24-fit grid is not necessary to preserve the present descriptive result.

## 6. Data integrity and reproducibility audit

An **independent post-run ID/split integrity check PASSED** on 5 November 2026. It read the stored `population_ids.csv` and the original M1-C scenario Parquet and verified:

- **15,000** unique `bldg_id` values across **three** populations (`pop01`, `pop02`, `pop03`), each with **5,000** buildings.
- Exactly **4,000 train** and **1,000 test** buildings per population; no duplication across populations (hence no train/test ID overlap between populations).
- **Zero shared building IDs** with the original M1-C population.

The check verifies stored population IDs and allocation. It does **not** independently validate the upstream raster-to-terrain processing or inspect every manifest checksum. The generated manifest/source hashes and external evidence preservation must remain available for a full independent reproduction.

**Outstanding bounded follow-up:**

1. Confirm source, configuration, code and output checksums in `sampling_manifest.json` and `full_summary/summary_manifest.json` against the actual archived inputs and outputs; the generated files were not supplied for independent hash inspection here.
2. Inspect the individual `pop03`, `gamma=0` exception (one case where B1 has additional regret) at asset-ID and posterior-mean-rank level. Consider finite-MCMC sensitivity; do not yet assert that the extra regret is solely due to the irrelevant predictor.
3. Copy/retain the Git-ignored result directories, posterior samples and manifests to versioned research storage. Source code commits do not archive the outputs.

### How to reproduce

Run from the repository root with its local **Python 3.11** environment and unchanged versioned inputs:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& .\.venv\Scripts\python.exe -m pytest tests/test_m1f.py -q
& .\.venv\Scripts\python.exe -m rhb.experiments.m1f_population_sampling  # only if not already generated
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f
```

**Important:** The population sampler refuses to overwrite an existing frozen sampling manifest unless explicitly forced. Do **not** rerun the sampler on the completed experiment without planning a new version. The original source snapshots and environment must remain available to reproduce every fit.

## 7. Interpretation and decision

**Accepted as a completed exploratory population-robustness experiment, with an explicitly retained single-fit diagnostic warning and remaining archive/asset-level follow-up.** The expected benefit of including informative terrain persisted **in mean synthetic regret across all three disjoint new Rotterdam samples**, with 5/6 positive-control wins and one tie. A zero-effect terrain variable gave 5/6 ties and one case with small additional B1 regret. Predictive improvements followed the same conditional direction on average. The experiment tests sensitivity to *population membership within Rotterdam*, not observed flood outcomes, spatially blocked generalization, independent geographical transfer, or actual avoided damage.

**What to test next:** Preserve the archived evidence and verify its fingerprints; optionally inspect the pop03 null-effect selection boundary at asset level. No additional model grid is needed for the scoped exploratory M1-F conclusion. Hamburg and Donostia feature comparability belongs to the next transfer checkpoint.
