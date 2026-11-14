# Habnetic | Phase 4 M1-G: bounded HAM/DON terrain portability experiment

**Design version:** 0.1.0 · **Registered:** 14 November 2026 · **Status:** protocol and implementation delivered, **scientific runs not yet executed or reviewed**.

## Goal

Test whether the *synthetic* decision effects observed with Rotterdam terrain in the frozen M1-C/M1-F design recur when its fixed algorithm is applied to Hamburg and Donostia's different, measured terrain-feature distributions.

**Not tested:** geographic transfer of posterior coefficients, real flood-damage prediction, empirical benefit of terrain, or domain transfer. The two cities are independently refit. Treat each as an experimental covariate-distribution portability screen, not trained-model transfer (reserved for M3).

## Frozen scientific specification

| Item | M1-G |
| --- | --- |
| Buildings available | HAM: 341,530; DON: 7,755; exact Phase 3/terrain ID join required |
| City sample | One deterministic *random, non-spatial* sample of 5,000 per city |
| Train/test | 4,000/1,000, same IDs and split for all gamma/seed/model combinations |
| Generator | `p_true = sigmoid(-3 + E_hat_v0 + H_pluvial_v1_logrel + gamma_true * asinh(T50/0.5 m))` |
| Likelihood | Bernoulli synthetic outcome `Y` |
| Generator scenarios | `gamma_true ∈ {0, −0.8}` (null, known terrain signal), each with seeds 101 and 202 |
| B0 | `logit(p) = alpha + beta_E E + beta_H H` |
| B1 | B0 plus `gamma_T * T` |
| Priors | `alpha,beta_E,beta_H ~ N(0,2.5)`; `gamma_T ~ N(0,0.5)`; independent |
| Sampler | Two NUTS chains, 500 tune and 500 draws per chain, `target_accept=0.90` |
| Budget | Top 10 of the same 1,000 held-out buildings, selection based on posterior mean p |
| Cases/fits | Four matched scenarios, eight B0/B1 fits per city; 8 cases/16 fits total |
| Frozen precedent | RTM M1 tag `phase4-m1-complete-v1.0`; unchanged scientific functions from `run_m1c` |

Paired labels use common uniform draws for the two synthetic generating effects at fixed city and simulation seed. Within each scenario, B0/B1 are fitted using identical generated outcomes, train/test split and selection capacity.

## Data input contract

1. `outputs/phase3/{HAM,DON}/phase3_features_scaled.parquet`: requires `bldg_id`, `E_hat_v0`, `H_pluvial_v1_logrel` as in RTM M1-F.
2. `../data/runs/ham_m1_citywide/terrain_batches/`: all 342 Parquet batches and final extraction manifest. Every batch SHA-256 must match.
3. `../data/runs/don_m1_citywide/terrain_features_m1_5m.parquet`: single consolidated 7,755-row terrain table.
4. Both terrain inputs require `bldg_id`, `T50`, `terrain_measured` and complete, unique building-ID correspondence with the Phase 3 features. **No silent joins, rescaling, imputation or outlier removal.**

Input audit is explicitly a gate. If the HAM/DON Phase 3 `phase3_features_scaled.parquet` filenames or columns differ from the Rotterdam M1-F precedent, stop and inspect real files. Do not guess a substitute.

## Statistical checks and cautions

- **Prior predictive:** original M1-C `run_fit` samples prior-predictive synthetic event rates; compare with known generated base rates. The original broad baseline priors were discussed in M1-D2; they are not retrained to make M1-G look better.
- **Posterior predictive:** original M1-C `run_fit` samples posterior-predictive event rates; this is a *coarse rate check*, not a comprehensive spatial PPC.
- **Convergence:** flag max R-hat >1.01, bulk/tail ESS <400, any divergences; inspect traces of flagged fits before scientific acceptance. Two chains are exploratory.
- **Predictive comparison:** held-out integrated Bernoulli log predictive score, Brier, RMSE against synthetic true p. No LOO needed for this fixed held-out pilot.
- **Decision comparison:** paired oracle regret, posterior-derived top-10 inclusion probability, borderline counts, top-k overlap, turnover. Sum of all held-out inclusion probabilities must equal 10.
- **Sensitivity:** gamma 0 is the negative control; gamma −0.8 known-signal control. Previous M1-D/D2 explored prior variations in RTM; **HAM/DON prior sensitivity is NOT performed here**. If city results or diagnostics suggest sensitivity, preregister and rerun a separate new version rather than tune this one post hoc.
- **Units:** T50 in m, `T=asinh(T50 / 0.5m)` dimensionless; existing Phase 3 E/H feature scales preserved. No claim that 0.5m is universally meaningful.
- **Sampling limitation:** one nonspatial sample per city. The two outcomes per city reuse the exact same assets/split, so four cases per city do not constitute four independent geographic confirmations.

## Outputs

Under `outputs/phase4/<CITY>/m1g/`:

- `m1g_scenarios.parquet`: synthetic p_true and `Y_m1g` paired across scenarios; not observed labels.
- `population_ids.csv`: exact sampled IDs and training/test split.
- `sampling_manifest.json`: frozen config and SHA256 source/code/scenario inventory.
- `fits/gamma_*/B{0,1}/`: M1-C-compatible `idata.nc`, `fit_record.json`, `posterior_summary.csv`, `test_decisions.parquet`.
- `smoke_{fit_results,comparisons,manifest}` and `full_{fit_results,comparisons,manifest}`.
- `full_summary/`: matched cases, gamma summaries, flagged fits, draft report and **draft** artifact-contract manifest.
- `outputs/phase4/m1g_cross_city_summary/`: side-by-side *descriptive* summary, not pooled posterior inference.

Results and raw data are ignored by Git; archive finished outputs separately and preserve hashes with the source code commit.

## Run in Windows PowerShell

From `C:\Users\C.Price\Habnetic\resilient-housing-bayes`, using that repository's Python 3.11 `.venv`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& .\.venv\Scripts\python.exe -m pytest tests/test_m1g.py -q
& .\.venv\Scripts\python.exe -m rhb.experiments.m1g_cross_city_sampling --city all --check
& .\.venv\Scripts\python.exe -m rhb.experiments.m1g_cross_city_sampling --city all
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1g --city all --smoke --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1g --city all --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1g --city HAM --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1g --city DON --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1g --city all --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1g --city all
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1g --city HAM
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1g --city DON
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1g --city all
```

**No inference commands until the input check and unit tests pass.** The full run reuses smoke checkpoints. The runner does not silently overwrite generated scenarios or rerun cached fits whose fingerprints differ. Every fit has a user-reviewable diagnostic record.

To archive after scientific review, from the same directory:

```powershell
& .\.venv\Scripts\python.exe -m rhb.pipelines.archive_m1g --city all --check
& .\.venv\Scripts\python.exe -m rhb.pipelines.archive_m1g --city all --archive-dir ..\archives
```

### Versioning and tagging gate

**Do not tag based on passing a Python script alone.** First commit the M1-G implementation on branch `phase4/m1g-cross-city` (retaining frozen RTM tag), run the full experiments using that committed code version, and review warnings and results. Then add a reviewed results note and create an annotated tag such as `phase4-m1g-cross-city-v0.1`. Git tags protect code and review documents; the SHA256 archives in Drive preserve generated posterior outputs, which are intentionally Git-ignored.

## Minimum decision record before tag

Record separately for HAM/DON: missingness (must be 0), train/test counts, feature T distribution, generated event rates at gamma 0 and −0.8, all diagnostics, model comparison Brier/ELPD, regret and top-k overlap, null-scenario performance, sensitivity limitations, archive filenames/hashes, full code SHA, and acceptance status (checked synthetic vs flagged exploratory). Do not rewrite the frozen Rotterdam reports.

**What to test next:** run the read-only city input audit, then the 4-fit-per-city smoke, check diagnostics, and only then proceed to the full 16 fits.
