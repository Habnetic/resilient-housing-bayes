# Habnetic | Phase 4 — M1-G2 Full-City Terrain Experiment

**Results record and scientific interpretation**  
**Experiment:** `M1-G2` · **Implementation:** `m1g_citywide_v0.1` · **Runner:** `RUN_ALL_v0.2`  
**Execution completed:** 14 November 2026 (local run log)  
**Record prepared:** 15 November 2026  
**Evidence type:** Synthetic Bernoulli outcomes; citywide measured terrain inputs  
**Status:** **Computations completed; scientific review and archival acceptance pending.**  
**Intended use:** Reproducibility record, input for a future scientific paper, and specification of scientific layers for the Habnetic Operations Center.

> **Interpretation boundary:** These are controlled *synthetic* experiments, **not** observed flood-damage probabilities, independently validated flood maps, or evidence that terrain interventions improve real-world decisions. All numerical results below were transcribed from the completed workstation run and the six summary tables supplied in the project conversation. This document was checked against the packaged engine and protocol, but the actual local NetCDF/Parquet files have **not** been independently opened for this report.

## 1. Research question

How does adding a terrain-derived covariate change posterior top-\(k\) building prioritisation when the full eligible populations of Rotterdam, Hamburg and Donostia are fitted under identical within-city synthetic outcomes and selection budgets?

Two preregistered scenarios distinguish a **negative control** (terrain has no true effect, \(\gamma_\mathrm{true}=0\)) from a **positive control** (terrain has an imposed non-zero effect, \(\gamma_\mathrm{true}=-0.8\)). B0 excludes terrain; B1 includes it. The primary estimands are posterior top-\(k\) inclusion probabilities, selected-set changes and synthetic oracle regret.

This is **not** a test of intercity parameter transfer or partial pooling: all models are fitted independently within each city. It is also not a comparison of identical observations against the original frozen Phase 3 fits. B0 is newly refitted for a matched M1-G2 comparison.

## 2. Eligible populations, datasets and experimental design

| City | Code | Phase 3 buildings | Terrain-eligible buildings | Excluded from matched M1-G2 | Budget \(k\) |
|---|---|---:|---:|---:|---:|
| Donostia / San Sebastián | DON | 7,755 | 7,755 | 0 | 78 |
| Rotterdam | RTM | 221,324 | 220,456 | 868 | 2,205 |
| Hamburg | HAM | 341,530 | 341,530 | 0 | 3,415 |
| **Total unique eligible buildings** | | **570,609** | **569,741** | **868** | **5,698** |

**Design:** 3 cities × 2 models (B0, B1) × 2 synthetic scenarios (\(\gamma=0,-0.8\)) = **12 full-population Bayesian fits**, accompanied by building-level scoring and **6 paired B0/B1 comparisons**. The 5,000-building experiments in M1-G1 are methodological checks, **not** the dataset used to fit M1-G2.

Each M1-G2 comparison uses the same eligible building IDs, synthetic labels, deterministic top-\(k\) tie-breaking procedure and budget for B0 and B1. The 868 Rotterdam records lacking valid terrain are **not imputed** or included in the matched comparisons. The 1% budget uses `round(0.01 * N)` (Python's rounding rule).

### Covariates and terrain transformation

- `E_hat_v0`: pre-existing Phase 3 exposure-related feature.
- `H_pluvial_v1_logrel`: pre-existing Phase 3 pluvial-hazard proxy.
- `T50`: terrain-derived quantity in metres, available for every eligible record.
- `T = asinh(T50 / 0.5 m)`: dimensionless predictor used for inference.

**Important:** `T50` should not be described as absolute elevation above sea level without verifying the actual upstream definition, sampling method and vertical reference system. The recorded dataset audit ranges were: DON \([-23.459,43.269]\) m; RTM \([-11.229,16.083]\) m; HAM \([-12.975,19.692]\) m. Physical/topographic QA remains a separate task.

## 3. Generative model, likelihood and priors

For building \(i\) in city \(c\), the known synthetic data-generating probability is

\[
 p_{i,c}^{\mathrm{true}} = \operatorname{logit}^{-1}\!\left(
 -3 + E_{i,c} + H_{i,c} + \gamma_{\mathrm{true}} T_{i,c}
 \right),
\qquad
Y_{i,c}^{\mathrm{synthetic}} \sim \operatorname{Bernoulli}(p_{i,c}^{\mathrm{true}}).
\]

The two scenarios are \(\gamma_{\mathrm{true}}=0\) and \(\gamma_{\mathrm{true}}=-0.8\). Within each city, the two scenarios use the same underlying uniform random variates to generate paired synthetic outcomes. Seed and generation parameters are frozen in `configs/m1g_citywide_v0.1.json`; the outcome seed is `20261116` (a seed identifier, not the execution date).

**B0 (terrain omitted):**

\[
Y_i \sim \operatorname{Bernoulli}(p_i), \qquad
\operatorname{logit}(p_i) = \alpha + \beta_E E_i + \beta_H H_i.
\]

**B1 (terrain included):**

\[
Y_i \sim \operatorname{Bernoulli}(p_i), \qquad
\operatorname{logit}(p_i) = \alpha + \beta_E E_i + \beta_H H_i + \gamma_T T_i.
\]

**Priors:** independent \(\alpha,\beta_E,\beta_H\sim\mathcal N(0,2.5)\); for B1, \(\gamma_T\sim\mathcal N(0,0.5)\). Parameter \(\gamma_T\) measures the log-odds effect per **one unit of transformed terrain \(T\)**, not per metre, and is not a direct percentage of flood damage.

**Inference configuration:** PyMC NUTS; 4 chains, 1,000 tuning iterations and 1,000 retained draws per chain, `target_accept=0.90`, 2 concurrent worker processes. To avoid storing massive observation-level arrays, the engine does not save the full log-likelihood tensor.

## 4. Posterior decision quantities and their definitions

For each fitted model, the engine selects up to **250 approximately evenly spaced posterior draws per chain**, totaling **1,000 coefficient draws** in the reference configuration. Every retained draw scores and ranks **all** buildings eligible in that city, rather than a 5,000-building subsample.

- **Posterior mean event score** (`p_mean`): average \(p_i\) across the selected coefficient draws, under the synthetic model.
- **Posterior top-\(k\) inclusion** (`p_topk`): fraction of selected coefficient draws in which building \(i\) belongs to the highest-scoring \(k\) buildings.
- **Deterministic reported selected set** (`selected_by_p_mean`): top \(k\) buildings ranked by their *posterior mean event score*. This is **not** the same estimator as selecting the highest `p_topk` values.
- **Uncertain/borderline membership:** \(0.2 < p_{\mathrm{top}k,i} < 0.8\) (strict inequalities); a descriptive decision-boundary category, not a calibrated confidence level.
- **Entries/exits**: buildings entering/leaving the **posterior-mean selected set** when replacing B0 by B1. Entries equal exits because both selected sets contain exactly \(k\) assets.
- **Synthetic oracle set**: the top \(k\) buildings ranked by *known generating* \(p_i^{\mathrm{true}}\). This oracle is available only because the experiment is synthetic.

**Synthetic oracle regret** is implemented as

\[
 R(M) = \sum_{i\in S_{\mathrm{oracle}}}p_i^{\mathrm{true}}
 - \sum_{i\in S_M}p_i^{\mathrm{true}},
\]

with non-negative numerical clipping and \(|S_M|=|S_{\mathrm{oracle}}|=k\). It measures the shortfall in **expected synthetic event count captured by the selected set**, *not* money, actual observed damage or a fraction of buildings damaged. Its absolute scale varies with city population, selection budget and the synthetic generating probabilities; **do not compare absolute regret values between cities as though they were normalised effect sizes**.

A built-in internal consistency check verifies \(\sum_i p_{\mathrm{top}k,i}=k\) (within numerical tolerance). The comparison output also checks identical building IDs and identical synthetic reference probabilities for B0/B1.

## 5. Results: selected-set changes and regret

Values below come from the six `comparison_manifest.json` records as read in PowerShell after the completed run. **The reported top-\(k\) overlap refers to deterministic sets selected using posterior mean event scores**, not the overlap of all draw-wise selected sets.

| City | True \(\gamma\) | \(N\) | \(k\) | Shared B0/B1 | Enter B1 (= exit B0) | Replaced share of \(k\) | Borderline B0 | Borderline B1 | Regret B0 | Regret B1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DON | 0.0 | 7,755 | 78 | 78 | 0 | 0.00% | 10 | 24 | 0.000000 | 0.000000 |
| RTM | 0.0 | 220,456 | 2,205 | 2,198 | 7 | 0.32% | 26 | 64 | 0.002933 | 0.015649 |
| HAM | 0.0 | 341,530 | 3,415 | 3,389 | 26 | 0.76% | 46 | 79 | 0.005606 | 0.067903 |
| DON | −0.8 | 7,755 | 78 | 29 | 49 | **62.82%** | 14 | 10 | 31.367451 | 0.003040 |
| RTM | −0.8 | 220,456 | 2,205 | 1,712 | 493 | **22.36%** | 35 | 68 | 59.120462 | 0.009864 |
| HAM | −0.8 | 341,530 | 3,415 | 1,904 | 1,511 | **44.25%** | 47 | 100 | 282.578002 | 0.130386 |

The replacement percentage is \(100\times \mathrm{entries}_{B1}/k\), *not* the percentage of the city's total building population. For example, under the imposed terrain effect, 49 of the 78 priority slots change in DON, not 49 of all 7,755 buildings.

### 5.1 Negative control: \(\gamma_{\mathrm{true}}=0\)

- DON: B0 and B1 select the same 78 assets; both have zero synthetic regret at the recorded precision.
- RTM: 7 of 2,205 priority slots change (0.32%); oracle regret increases from 0.002933 (B0) to 0.015649 (B1).
- HAM: 26 of 3,415 priority slots change (0.76%); oracle regret increases from 0.005606 (B0) to 0.067903 (B1).
- Borderline membership **increases** after adding terrain in each city (DON 10→24, RTM 26→64, HAM 46→79).

**Interpretation:** With a zero imposed terrain effect, the additional B1 coefficient does not materially alter the posterior-mean selected sets, though it can redistribute inclusion uncertainty and modestly worsen synthetic oracle regret. These small differences may reflect finite data/sampling and estimation uncertainty; they are not evidence that physical terrain information is detrimental in actual flood risk analysis.

### 5.2 Positive control: \(\gamma_{\mathrm{true}}=-0.8\)

- DON: only 29/78 selected assets remain common to B0/B1; **49 (62.82%)** priority slots change. Synthetic regret falls from 31.367451 to 0.003040.
- RTM: 1,712/2,205 selected assets remain common; **493 (22.36%)** priority slots change. Synthetic regret falls from 59.120462 to 0.009864.
- HAM: 1,904/3,415 selected assets remain common; **1,511 (44.25%)** priority slots change. Synthetic regret falls from 282.578002 to 0.130386.
- Borderline results are **not monotone or universal**: DON 14→10, RTM 35→68, HAM 47→100. Adding an informative variable can improve oracle selection while changing uncertainty near the decision boundary in either direction.

**Interpretation:** Under a synthetic generator deliberately containing the terrain covariate, B1 has the correctly specified functional form while B0 omits that known effect. B1's much lower synthetic oracle regret is therefore expected, but the **size and geographic distribution of the priority-set changes** are genuine outputs of this controlled full-city experiment. These results are methodological evidence about specification sensitivity, **not empirical evidence of benefits from real-world terrain-based flood prioritisation**.

## 6. Results: recovery of the injected terrain coefficient

B1's posterior summaries were read from `B1/posterior_summary.csv` for each city/scenario. Numbers below preserve the displayed rounding in the user's PowerShell extraction. The **95% HDI** is the ArviZ highest-density interval computed with `hdi_prob=0.95`; it is not a frequentist confidence interval. The recovery criterion here is descriptive compatibility with the known synthetic generating coefficient, not formal validation against independent observed data.

| City | Injected \(\gamma_{\mathrm{true}}\) | Posterior mean \(\gamma_T\) | 95% HDI lower | 95% HDI upper | Contains generating value? |
|---|---:|---:|---:|---:|---|
| DON | 0.0 | −0.001 | −0.028 | 0.028 | Yes |
| RTM | 0.0 | 0.012 | −0.030 | 0.055 | Yes |
| HAM | 0.0 | −0.017 | −0.036 | 0.003 | Yes |
| DON | −0.8 | −0.810 | −0.851 | −0.768 | Yes |
| RTM | −0.8 | −0.797 | −0.839 | −0.757 | Yes |
| HAM | −0.8 | −0.819 | −0.837 | −0.799 | Yes |

**Finding:** All six displayed 95% HDIs include the injected generating coefficient. In the null scenario all include zero; in the imposed-effect scenario all include −0.8, and none includes zero. The coefficient is recovered at the reported precision in all three cities. This is **internal parameter recovery under a correctly specified synthetic data generator**, not independent empirical predictive validation.

**PowerShell note:** `Import-Csv` warned about an unnamed first CSV header because the file was written by `DataFrame.to_csv()` with the parameter name as the index. Those warnings alone do not invalidate the six extracted `gamma_T` rows. For archival use, retain the original CSVs and independently verify the extraction during the final QA.

## 7. Execution, diagnostics and reproducibility

The run logged completion of DON, RTM and HAM in both scenarios, with full B0/B1 fitting, citywide scoring and comparison. The final runner message was:

```text
ALL REQUESTED CITY/SCENARIO ANALYSES COMPLETE.
M1-G2 finished for DON, RTM and HAM; gamma -0.8 and 0; B0/B1 fit/score/compare.
```

**Automated tests:** `25 passed` in the user's terminal. The runner verifies for each full fit: number of observations, exact output SHA-256, \(\hat R\leq1.01\), bulk ESS ≥ 400, tail ESS ≥ 400, and zero divergences; failed gates stop the run. **All 12 fits passed the runner's automated gates**, as indicated by its successful final completion. This report does not reproduce all 12 diagnostics individually; the original `fit_diagnostics.json` files remain the source of truth.

For HAM under \(\gamma=0\), the terminal explicitly reported:

| Fit | \(N\) | Max \(\hat R\) | Min bulk ESS | Min tail ESS | Divergences |
|---|---:|---:|---:|---:|---:|
| B0 | 341,530 | 1.0000 | 1,137 | 1,044 | 0 |
| B1 | 341,530 | 1.0000 | 1,401 | 1,661 | 0 |

The saved fit SHA-256 hashes for these two HAM \(\gamma=0\) NetCDF objects, as shown in the execution log, are:

```text
HAM/gamma_0/B0/idata.nc
c8d6b191108664ababb6a5f9cdfcdcba53219161fc27370ecb0c6b2087147a9a
HAM/gamma_0/B1/idata.nc
c7e59cc55f2246f0e482e410a36a18f80b9bafcb1fe57e6d0ad46459421f5cf6
```

**Code/config fingerprints logged by the original execution:**

```text
configs/m1g_citywide_v0.1.json
068b3a5d12b5e2bd410ce6a092cfba520901e2892bedbfa2c71e6b9eeff405c4
src/rhb/pipelines/m1g_citywide.py
e3481e2cf32c0aeb0e72b5cff017f5e3c52062f1b5aac55b87df9d4cf045fc15
src/rhb/experiments/m1f_population_sampling.py
e2ff7ad8f45a1878e1f7ab1d2c405bd045f7822f0305c77b82dd203503169d6f
src/rhb/experiments/m1g_cross_city_sampling.py
bd9ea284c225b3f5e61490c401933d2905d15b55b8fedcd9fe1c3fa8fa28e3e5
```

### Output inventory

For each city `CITY ∈ {DON, RTM, HAM}`:

```text
outputs/phase4/CITY/m1g_citywide_v0.1/
  preparation_manifest.json
  gamma_0_scenario.parquet
  gamma_neg0p8_scenario.parquet
  gamma_0/
    B0/{idata.nc, posterior_summary.csv, fit_diagnostics.json,
        citywide_decisions.parquet, scoring_manifest.json}
    B1/{...}
    citywide_comparison.parquet
    comparison_manifest.json
  gamma_neg0p8/
    B0/{...}
    B1/{...}
    citywide_comparison.parquet
    comparison_manifest.json
```

Runner log: `outputs/phase4/_m1g2_runner/m1g2_runner.log`.

Each comparison parquet includes building ID, terrain feature in metres, B0/B1 posterior means and top-\(k\) inclusion estimates, their delta, posterior-mean selection flags, a four-way decision-change class, and rank uncertainty. It **does not contain building geometry or map tiles**.

### One-command execution / reproduction

From the scientific repo root on Windows PowerShell, in the existing **Python 3.11** virtual environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1g.py tests/test_m1g_citywide.py -q
.\RUN_M1G2_ALL.ps1
```

The runner is resume-safe and validates existing complete files rather than blindly overwriting them. To reproduce *from scratch*, use a separate clean working directory with the same frozen source datasets and configuration (and archive the original output hashes first). Do **not** delete existing results simply to re-run an already completed experiment.

**Environment pinning required before final archive:** Python exact version, PyMC, ArviZ, numpy, pandas, SciPy, PyArrow and platform/environment lockfile. Save a `pip freeze` alongside the scientific artifacts, review it, and document the original dataset/repository versions. Exact versions cannot be established from the terminal snippets used in this report.

## 8. Interpretation and limitations

1. **Synthetic target:** Labels were generated from a known logistic function, not observed inundation, losses or claims. Results test a method's behaviour under controlled conditions.
2. **Built-in advantage for B1 in the positive control:** The synthetic generator explicitly uses \(T\) and B1 is specified with that term. Near-oracle synthetic performance is therefore informative about implementation and finite-sample effects, not surprising independent scientific discovery.
3. **No hierarchical transfer:** Models are separately fitted in three cities. The experiment cannot support claims of learned cross-city knowledge transfer or benefits from partial pooling.
4. **No causal terrain effect identified:** The imposed coefficient is a scenario assumption. The experiment does not identify the causal physical effect of ground level, slope, rainfall or hydraulic defences.
5. **Rank stability ≠ prediction accuracy:** Top-\(k\) probabilities measure stability under the fitted posterior. A stable shortlist can still be wrong under an incorrect physical model.
6. **Posterior draw approximation:** `p_topk` is estimated from 1,000 scored draws, not integrated analytically; assess stability under alternate draw subsets. Do not interpret changes of 0.001 as exact to arbitrary precision.
7. **Urban source comparability:** Terrain definitions, measured coverage, vertical reference systems, and outliers require geographic/source QA before spatial interpretation or release of maps. RTM has 868 excluded records.
8. **Predictive checks incomplete:** The engine records prior/posterior replicated *aggregate event rates* (up to 100 replicate draws), but those checks have not been inspected here and do not validate spatial structure, feature-stratified fit or observed flood risk.
9. **Sensitivity incomplete:** An alternate terrain prior width, alternate likelihood/generator and posterior-scoring draw sensitivity have not been reported; these should be completed before strong publication claims.
10. **Regret comparability:** Regret is an unnormalised synthetic expected-event-count loss relative to an oracle. Cross-city magnitudes cannot be compared directly without accounting for different sizes, budgets and generating distributions.
11. **Frozen Phase 3:** The Phase 3 result set differs from matched M1-G2 B0 in source population and synthetic outcomes; comparisons to the original Phase 3 must be separately labelled and cannot attribute changes solely to terrain.
12. **External empirical tests:** Historical flood-damage maps are not present-day labels; changed terrain, buildings and defences need explicit consideration in future M2/M3 observational work.

## 9. Proposed paper-ready results paragraph (draft; revise after QA)

> We evaluated a matched full-city synthetic Bayesian prioritisation experiment on 569,741 terrain-eligible buildings in Rotterdam, Hamburg and Donostia–San Sebastián. For each city, independently fitted logistic models without (B0) and with (B1) a transformed terrain predictor were compared under two prespecified synthetic outcome-generating scenarios: no terrain effect (\(\gamma_{\mathrm{true}}=0\)) and a negative terrain effect (\(\gamma_{\mathrm{true}}=-0.8\)). When the generating terrain effect was absent, the posterior-mean top-1% selected sets were nearly unchanged: 0, 7 and 26 substitutions in Donostia, Rotterdam and Hamburg, respectively. When the generating effect was non-zero, 49 of 78, 493 of 2,205 and 1,511 of 3,415 selected positions changed, corresponding to 62.82%, 22.36% and 44.25% of priority capacity. In the non-zero-effect scenario, B1 substantially reduced regret relative to a synthetic oracle in all cities, consistent with inclusion of the predictor used by the data-generating mechanism. B1's posterior 95% highest-density intervals for the terrain coefficient contained the injected values in every city and scenario. These findings demonstrate sensitivity of posterior-informed selection to a controlled terrain-specification change; they do not establish observed flood-risk accuracy or real-world intervention benefit.

**Suggested paper figures (after map readiness and QA):** (i) coefficient recovery plot with 95% HDI and injected \(\gamma\); (ii) selected-set overlap / entry–exit plot; (iii) paired regret plots with careful city scale handling; (iv) B0 vs B1 citywide geographic difference maps with synthetic-evidence labels; (v) posterior inclusion boundary counts. Paper tables should report both exact counts and denominators.

## 10. Acceptance checklist and next research actions

**Verified from the supplied completed run and engine:**

- [x] Complete eligible population specified and audited for all three cities.
- [x] 12 full-population fits completed; 6 comparisons generated.
- [x] Automated convergence/ESS/divergence gates passed in the runner.
- [x] Full-city posterior decision scoring and exact-\(k\) integrity checks applied.
- [x] B1 terrain coefficient recovery summaries extracted (six scenarios/cities).
- [x] Negative and positive synthetic controls show different expected behaviour.

**Required before declaring scientific acceptance/tag:**

- [ ] Open and review **all 12** posterior diagnostics (not just examples), trace behaviour and numerical warnings.
- [ ] Review stored prior/posterior predictive rate checks, and add spatial/feature-stratified PPC as appropriate.
- [ ] Run and document at least one alternative terrain prior width and draw-subsampling sensitivity test; compare with M1-G1 held-out evidence.
- [ ] Verify original `posterior_summary.csv` fields and the six manifest records directly from the local output files.
- [ ] Audit transformed terrain semantics, vertical references, exclusions and extreme values, especially DON's negative `T50` range.
- [ ] Pin and archive exact Python/environment versions, source dataset versions, run manifests and hashes.
- [ ] Decide the scope of the Phase 4/M1 scientific acceptance tag and archive large NetCDF/Parquet outputs externally with immutable hashes (do not assume a Git tag archives ignored outputs).

**After acceptance:** Publish geospatial extracts by joining comparison outputs on canonical `bldg_id` to original building geometries; add 2D/3D building hover and inspector in the Operations Center. The original Phase 3 view must remain a **separately labelled historical reference**, while matched B0/B1 is the controlled causal-*specification* contrast. Prepare independent F0 geographic showcases for isolated facilities without fabricating module-level damage probabilities.

## 11. Git recording guidance

This Markdown document can be committed and pushed **now as a results record**, without prematurely declaring full scientific acceptance. Suggested commit message:

```text
docs(phase4): record M1-G2 full-city synthetic results and gamma recovery
```

Do not create a release tag claiming scientific acceptance until Section 10's open gates are resolved. Code/config changes should be reviewed and committed separately if still untracked, and the large NetCDF/Parquet artifacts should not be accidentally added to Git.

---

**Source files reviewed for this report:** `configs/m1g_citywide_v0.1.json`, `src/rhb/pipelines/m1g_citywide.py`, `docs/phase4/M1_G2_Full_City_Protocol.md`, and `scripts/run_m1g2_all.py` from the two previously supplied M1-G2 ZIP packages. **Numerical evidence:** the PowerShell six-row comparison-manifest extraction, six B1 `gamma_T` posterior summary rows, and HAM/runner diagnostics shared after completing the workstation run. **Not reviewed directly:** the workstation's underlying NetCDF, Parquet, and local provenance manifest bytes.
