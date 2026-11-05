# Habnetic | Phase 4 M1: Rotterdam topographic integration and controlled synthetic validation

**Version:** 0.2.0, 5 November 2026<br>
**Status:** **SCIENTIFIC M1 CLOSURE ACCEPTED for the bounded exploratory synthetic methodology milestone.** Numerical M1-F diagnostic triage, independent population-ID checks and all nine M1-E trace reviews are complete. **Formal evidence-archive/fingerprint sign-off remains open**, together with optional asset-level examination of an isolated null-control exception. Do not interpret scientific acceptance as a completed archival freeze or empirical flood validation.<br>
**Study area:** Rotterdam (RTM). **Evidence class:** observed geospatial terrain/exposure covariates with synthetic Bernoulli outcome and synthetic oracle-based decision evaluation. **Not** empirical flood-loss validation.

## Executive decision

The Rotterdam M1 research programme has completed a coherent sequence of controlled experiments: a measured topographic feature was integrated into the building-level covariates; negative and positive synthetic controls were run; outcome replications and coefficient/intercept prior sensitivity were explored; numerical convergence and an apparent prior-induced boundary swap were tested with longer sampling; and the decision comparison was reproduced on three new disjoint Rotterdam building samples.

**Scientific decision (5 November 2026):** **Accept M1 as complete within its predefined Rotterdam topographic-integration and controlled synthetic-validation scope.** All substantive exploratory experiments, targeted numerical follow-up, recorded visual inspection and independent population-ID integrity checks are complete. One marginal tail-ESS warning remains disclosed. **Operational reproducibility freeze is not yet certified** because off-repository archiving and manifest/hash verification have not been independently signed off. No additional large inference grid is justified for this bounded claim. M1 does **not** validate observed flood losses, hazard causality, intervention effectiveness or cross-city transfer.

## 1. Definition of M1 and what was actually delivered

**M1 objective:** Introduce measured high-resolution local topographic context into the existing posterior-based asset-selection framework, and determine in controlled experiments when this additional information can change inference and decisions under a fixed selection budget.

**Observed input data:** RTM buildings with Phase 3 exposure E and hazard proxy H, plus a 5 m terrain-derived relative elevation contrast T50. The accepted measured-terrain subset was **220,456** buildings from a clipped inventory of **221,324**; **868** did not have valid measured terrain in this snapshot. The model uses `T = asinh(T50 / 0.5 m)`. Dataset/provenance details belong to the existing M1 terrain quality report and source manifest.

**Generative story:** `Y_i ~ Bernoulli(p_true_i)` and `logit(p_true_i) = -3 + E_i + H_i + gamma_true*T_i`. All M1-A/F outcomes analyzed here are synthetically generated. Thus, terrain's presence or absence as a *predictive effect* is explicitly under experimenter control.

**Likelihood and priors:** B0: logit-Bernoulli with E and H; B1: B0 plus T. Historical baseline priors `Normal(0, 2.5)` for intercept/E/H coefficients and `Normal(0, 0.5)` for the B1 T coefficient, with separate sensitivity experiments that modify topographic and intercept priors. The inherited priors produce overly diffuse prior predictive event-rate distributions.

**Decision estimand:** Top 1% (k=10 of 1,000 held-out buildings in each pilot) ranked by posterior mean event probability. Synthetic **expected selection regret** is measured against an oracle informed by known generating probabilities. It assumes equally weighted outcomes and an idealized fully effective intervention. It is **not** measured economic loss, prevented damage or validated operational benefit.

## 2. Evidence ledger

| Stage | Registered purpose | Observed result and appropriate interpretation |
|---|---|---|
| M1-A | Negative control: terrain without additional generative effect | Small change to the baseline; no evidence of a systematic synthetic benefit from an absent added effect in that pilot. |
| M1-B | Positive control: known terrain effect | Extended model recovered a negative coefficient and reduced synthetic selection regret in the pilot; conditional on the designed generator. |
| M1-C | Replicate synthetic outcomes on one fixed 5,000-building sample | At `gamma=-0.4` and `-0.8`, mean B1-vs-B0 regret reductions were about 85.7% and 98.7%, respectively; at `gamma=0`, the models tied on selection regret. Five outcome seeds per effect are *not* population replicates. |
| M1-D1 | Change the prior width on the terrain coefficient | Prior width changes coefficient shrinkage and some top-10 decisions; closest parameter recovery is not always lowest oracle regret. Five seeds per condition are exploratory. |
| M1-D2 | Change the intercept prior and audit prior predictive distributions | Priors centred on the known synthetic intercept made prior predictive rates more plausible; posterior estimates and most selections changed little. The originally observed isolated P0/P1/P2 swap was later *not reproduced* under longer MCMC; retain the historical numbers but downgrade the original interpretation. |
| M1-E | Targeted computational confirmation of flagged fits and prior-sensitive boundary | Nine four-chain refits: **zero diagnostic flags and divergences**; six independent boundary fits gave identical top-10 selections. All nine confirmation trace plots were individually visually inspected and passed screening; the original boundary swap did not reproduce. |
| **M1-F** | Vary population membership within RTM | Three new disjoint 5,000-building samples, two outcomes per effect/population. At `gamma=-0.8`, B1 improved mean oracle regret in **all three** populations (5 wins, 1 tie over six cases); at `gamma=0`, 5 ties and 1 B1 disadvantage. **24 fits audited: one B0 tail-ESS warning; independently verified 15,000 unique IDs, disjoint populations and correct splits. Full source-manifest/archival checks remain open.** |

### M1-F quantitative synthesis

| True effect | Mean B0 regret | Mean B1 regret | Mean improvement B0 − B1 | B1 wins / ties / B0 wins |
|---|---:|---:|---:|---:|
| `gamma=0` | 0.000000 | 0.001958 | −0.001958 | 0 / 5 / 1 |
| `gamma=-0.8` | 0.147047 | 0.022663 | +0.124384 | 5 / 1 / 0 |

In the positive control the ratio of mean improvements to mean baseline regret is **84.6%**. Across the six positive-control cases, mean B1 γ was **−0.776833** versus the generating −0.8, with 6/6 reported interval inclusions. At `gamma=0`, mean B1 γ was **+0.024833** with 6/6 interval inclusions. These are small experimental summaries, not precision claims about population robustness or coverage.

The held-out score comparisons align directionally: for `gamma=-0.8`, mean `ΔELPD=+1.815913`, `ΔBrier=-0.000165703`, `ΔRMSE=-0.013688617`; for `gamma=0`, mean `ΔELPD=-0.227289`, `ΔBrier=+0.0000375524`, `ΔRMSE=+0.001424316` (all deltas are B1 minus B0). The fact that these decision and predictive comparisons are consistent under the chosen synthetic generator does not establish real-world predictive performance.

## 3. Findings that survive the experimental cycle

**Conditional synthetic value of terrain.** A measured terrain-derived predictor can contribute information that B0 lacks. When the synthetic generator includes that effect, B1 generally yields better selections; the direction of mean benefit survived three new disjoint RTM samples.

**No free benefit from unnecessary complexity.** When `gamma_true=0`, an extra predictor does not necessarily improve anything. One M1-F null-effect case produced B1-specific regret, consistent with finite-sample estimation/decision-boundary variability.

**Decision, posterior and specification sensitivity are distinct.** Posterior top-k inclusion variation within a model, differences between models using different features, differences induced by priors, and numerical variability from finite MCMC sampling must be measured and reported separately. M1-E showed an apparently prior-induced boundary difference that did not survive more intensive inference.

**Prior predictive plausibility is not sufficient.** Calibrating the intercept prior in an oracle-centred synthetic check reduced implausibly high prior aggregate event rates, yet the resulting posterior and decision improvements were not systematic. Future priors for real flood outcomes need external justification rather than access to the generating parameters.

## 4. Boundaries: what M1 did not establish

- **No empirical outcomes:** Synthetic Bernoulli labels are not observed floods, losses, insurance claims, repair costs or recovery times.
- **No physical causal effect:** The negative synthetic terrain coefficient is imposed by the generator, not inferred as a causal relationship from field evidence.
- **No spatial or cross-city validation:** All experiments use Rotterdam data. M1-F changes random building membership, but does not implement spatially blocked holdouts or transfer to Hamburg/Donostia.
- **No operational intervention model:** Regret assumes homogeneous asset value/cost and ideal interventions. The concept of avoided real damage must be designed and calibrated separately.
- **No complete model adequacy claim:** The hazard proxy H has extremely small variance (`SD` around 0.0076 on the pilot sample), leaving `beta_H` weakly identified. Missing/poor terrain coverage and unusual exposure tails require downstream scrutiny; event-rate PPC alone is insufficient.
- **Limited replication:** Three new RTM samples and two outcomes per sample/effect are an exploratory robustness screen. Sample-level variability and correlated seed realizations preclude precise uncertainty intervals for population-level effect sizes.

## 5. Reproducibility, ownership and evidence locations

**Source of truth for code, tests, protocols and decisions:** GitHub, branch `phase4/m1-topography` until reviewed and merged. **Data/source assets and large generated experiment artifacts:** the dedicated `data` repository or controlled Drive/archive storage, with SHA-256 hashes. `outputs/**` is Git-ignored; commits alone do **not** preserve full posterior samples, decisions or source raster assets.

| Evidence | Location or command |
|---|---|
| Original Phase 3 frozen code | commit `5e19254`, tag `phase3-code-freeze-20261022` |
| M1-C implementation and full summary | commits `e920f8c`, `e8d0561`; `python -m rhb.pipelines.summarize_m1c` |
| M1-D1 implementation/results | commits `047d763`, `133d7a7`; `python -m rhb.pipelines.summarize_m1d` |
| M1-D2 implementation/results | commits `4ed5199`, `e9f9f63`; `python -m rhb.pipelines.summarize_m1d2` |
| M1-E implementation and revised interpretation | commit `259fac1`; `python -m rhb.pipelines.summarize_m1e` |
| M1-F implementation/results | commits `a917e09` (implementation) and `2b2ab89` (results/provisional closure); `python -m rhb.pipelines.summarize_m1f` |
| RTM terrain data | `../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet` |
| M1-F sampling manifest | `outputs/phase4/RTM/m1f/sampling_manifest.json` |
| M1-F full fit and summary manifests | `outputs/phase4/RTM/m1f/full_manifest.json`, `full_summary/summary_manifest.json` |
| M1-F per-case evidence | `full_fit_results.csv`, `full_comparisons.csv`, `full_summary/matched_cases.csv` |

**One-command-style local reproduction** (from repo root, project-local Python **3.11**, unchanged source data and fingerprints):

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& .\.venv\Scripts\python.exe -m pytest tests/test_m1f.py -q
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f
```

Full reconstruction of the data snapshot from original raster sources needs the separately versioned terrain-processing pipeline; the commands above assume those artifacts exist. Re-running the M1-F population generator over a frozen `sampling_manifest.json` is **not** part of normal reproduction and requires explicit controlled regeneration.

## 6. Closure gate and final classification

**Decision classification: CLOSED FOR THE SCOPED EXPLORATORY SCIENTIFIC QUESTION; EVIDENCE-ARCHIVE SIGN-OFF OPEN.** This is a documented scope decision, not a claim that all possible validation tasks are complete. M1 used measured Rotterdam topography but *synthetic* outcomes, and it does not cover empirical floods or transfer.

### Completed research and QA gates (5 November 2026)

- [x] M1-A through M1-F experimental computations completed and documented.
- [x] M1-F includes 12 paired cases, 24 fitted models, three distinct disjoint Rotterdam samples and the recorded positive/negative controls.
- [x] B1 positive-control improvement appears in mean synthetic regret in all three new populations; the one null-effect B1 disadvantage is disclosed.
- [x] Six positive-control coefficient estimates recover mean `gamma=-0.776833`, with the full-run predictive score comparisons recorded (see M1-F report).
- [x] M1-F full-run diagnostic file inspected: **one flagged fit of 24**, `pop02`, `gamma=-0.8`, seed 202, B0, `ESS_tail=331` against a threshold of 400; `R-hat=1.01`, `ESS_bulk=415`, `divergences=0`. Retained with warning, not reclassified as passing.
- [x] Independent M1-F population ID audit passed: **15,000 distinct IDs**, three disjoint groups of 5,000, each 4,000 train/1,000 test, and **zero overlap with original M1-C IDs**.
- [x] M1-E: all nine four-chain confirmation fits satisfy the numerical screens; **all nine trace plots** (three originally flagged and six boundary confirmations) individually reviewed without obvious mixing problems. Six boundary refits return the same top-10.
- [x] M1-F implementation commit `a917e09` and results/provisional closure commit `2b2ab89` recorded in the evidence ledger.

### Residual operational and narrowly scoped follow-up (not certified complete)

- [ ] Verify the actual source, config, code and output hashes from `sampling_manifest.json`, `full_manifest.json` and `full_summary/summary_manifest.json` against preserved local/archived files. The **ID integrity test passed**; that is not equivalent to complete provenance verification.
- [ ] Confirm a controlled, retrievable **off-repository archive** for large source data, posterior artifacts, decision outputs, raster inputs and manifests. `outputs/**` is Git-ignored; Git commits alone are insufficient.
- [ ] Inspect the isolated `pop03`, `gamma=0` negative-control selection at the asset-ID level if a causal/numerical explanation is to be published. Currently it is accurately reported as **one observed exploratory exception**, not attributed conclusively to the predictor or MCMC.
- [ ] Record the final closure decision and any archive URI/fingerprints in the KnowledgeLab/Notion decision log. A final signed reproducibility tag should follow archive QA rather than precede it.

**Acceptance statement:** The scientifically bounded M1 milestone may now be used as the starting point for HAM/DON terrain-feature portability and subsequent Phase 4 research, **without reopening M1-F**. A final immutable release/tag that certifies a self-contained reproducibility evidence bundle must wait for the operational items above. Later empirical or cross-city work is explicitly outside this acceptance decision.

## 7. Handoff after M1

**Next transfer checkpoint (not an M1 deliverable):** Compare terrain-feature definitions, coverage, support and distributions across **Hamburg (HAM)** and **Donostia (DON)**. Reproduce T50 with explicit local dataset, horizontal and vertical reference, spatial resolution and missingness checks. Only after establishing feature comparability should RTM-to-HAM/DON probabilistic transfer be tested, using an explicit target, matched selection budgets and appropriate calibration evidence.

**Parallel Phase 4 work:** M2 should focus on improving hazard representation and uncertainty; M3 should connect exposure/vulnerability to consequences and a meaningful decision objective. Neither should be retroactively claimed as validated in M1.

**What to test next:** Verify source/output manifest digests, archive the generated evidence and record a durable archive reference; then issue the reproducibility-frozen Git tag. In parallel, design the Hamburg/Donostia terrain-feature portability checkpoint without reopening the Rotterdam synthetic experiment grid.
