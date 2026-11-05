# Habnetic | Phase 4 M1: Rotterdam topographic integration and controlled synthetic validation

**Version:** 0.1.0, 5 November 2026<br>
**Status:** **PROVISIONAL CLOSURE REVIEW**. The experimental cycle M1-A through M1-F is executed and documented, but the final M1-F diagnostics/provenance audit and limited M1-E visual QA are not yet signed off. Do not relabel this document *fully closed* until the gate in Section 6 is satisfied.<br>
**Study area:** Rotterdam (RTM). **Evidence class:** observed geospatial terrain/exposure covariates with synthetic Bernoulli outcome and synthetic oracle-based decision evaluation. **Not** empirical flood-loss validation.

## Executive decision

The Rotterdam M1 research programme has completed a coherent sequence of controlled experiments: a measured topographic feature was integrated into the building-level covariates; negative and positive synthetic controls were run; outcome replications and coefficient/intercept prior sensitivity were explored; numerical convergence and an apparent prior-induced boundary swap were tested with longer sampling; and the decision comparison was reproduced on three new disjoint Rotterdam building samples.

**Provisional decision:** The *research and computation* of M1 are sufficient for **conditional closure as an exploratory synthetic methodology milestone**. Final sign-off requires the auditable checks listed below. No additional large-scale fit grid is justified by the results currently presented. Closure must **not** be described as validation of real flood losses, physical hazard causality, cost-effective interventions or cross-city transfer.

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
| M1-E | Targeted computational confirmation of flagged fits and prior-sensitive boundary | Nine four-chain refits: **zero diagnostic flags and divergences**; six independent boundary fits gave identical top-10 selections. Three originally flagged trace plots were visually reviewed; six boundary traces remain pending individual sign-off. |
| **M1-F** | Vary population membership within RTM | Three new disjoint 5,000-building samples, two outcomes per effect/population. At `gamma=-0.8`, B1 improved mean oracle regret in **all three** populations (5 wins, 1 tie over six cases); at `gamma=0`, 5 ties and 1 B1 disadvantage. **M1-F numeric diagnostics and ID/provenance checks remain to be signed off.** |

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
| M1-F implementation/results | **TODO: record exact implementation and report commit IDs after commit.** `python -m rhb.pipelines.summarize_m1f` |
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

The experimental work is **substantially complete**, but formal acceptance is conditional. The responsible reviewer must verify the following before changing the status above to **CLOSED (exploratory synthetic M1)**:

- [x] The M1-F full summary reports six paired comparisons per effect and 24 model fits in total.
- [x] The positive-control mean regret improvement is positive in all three new random RTM populations, and the `gamma=0` degradation is disclosed.
- [x] Held-out predictive comparisons and B1 coefficient recovery are recorded from user-supplied full-run summaries.
- [ ] Read `full_summary/summary_manifest.json` and `full_summary/diagnostic_flags.csv`; record the exact flag count and follow up on any flagged fits.
- [ ] Verify `population_ids.csv`, `sampling_manifest.json`, source hashes and exclusion of the original M1-C buildings in the local experiment artifacts.
- [ ] Inspect the `pop03`, `gamma=0` negative-control exception at case level, including its selected asset IDs and potential finite-MCMC sensitivity.
- [ ] Individually inspect and record the **six M1-E boundary trace plots** and, if the boundary narrative requires it, the exact original-versus-confirmed P0 swapped IDs.
- [ ] Archive/manifests: confirm retained posterior artifacts and matching digests outside Git-ignored transient output folders.
- [ ] Update the two documents' status lines, record M1-F commit IDs, and sign the closure decision in the project decision log.

**Provisional decision:** Conditional methodological closure is appropriate once these bounded audit items are completed. If a substantive convergence, provenance or leakage failure is found, investigate that failure rather than treating an attractive aggregate table as proof of success. Do not automatically expand the research design in response to an isolated finite-sample exception.

## 7. Handoff after M1

**Next transfer checkpoint (not an M1 deliverable):** Compare terrain-feature definitions, coverage, support and distributions across **Hamburg (HAM)** and **Donostia (DON)**. Reproduce T50 with explicit local dataset, horizontal and vertical reference, spatial resolution and missingness checks. Only after establishing feature comparability should RTM-to-HAM/DON probabilistic transfer be tested, using an explicit target, matched selection budgets and appropriate calibration evidence.

**Parallel Phase 4 work:** M2 should focus on improving hazard representation and uncertainty; M3 should connect exposure/vulnerability to consequences and a meaningful decision objective. Neither should be retroactively claimed as validated in M1.

**What to test next:** Complete the short closure checklist, freeze the M1 documentation and evidence bundle, and proceed to HAM/DON feature-portability design without reopening the Rotterdam synthetic experiment grid.
