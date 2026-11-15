# Habnetic | M1-G2 Full-City Run Launcher v0.2

This **additive runner** works with the previously installed `Habnetic_M1G2_FullCity_v0.1.zip`. It does not replace scientific code or change synthetic priors, predictions, outcomes or thresholds.

## Install and execute (Windows PowerShell)

From `C:\Users\C.Price\Habnetic\resilient-housing-bayes` with the Python **3.11** virtual environment active:

```powershell
Expand-Archive -Path "$HOME\Downloads\Habnetic_M1G2_RUN_ALL_v0.2.zip" -DestinationPath .
.\RUN_M1G2_ALL.ps1
```

If PowerShell execution policy blocks the `.ps1`, invoke the equivalent commands manually:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1g.py tests/test_m1g_citywide.py -q
python .\scripts\run_m1g2_all.py
```

## What runs

1. Checks each city and prepares both frozen synthetic scenarios where not yet prepared.
2. Runs a small Donostia smoke test for B0 and B1, gamma = -0.8 (unless `--skip-smoke`).
3. Runs **12 full city fits**: DON, RTM, HAM × B0/B1 × gamma -0.8 and 0.
4. Checks full-population N, posterior convergence diagnostics, fit SHA256, score count/hash, and comparison count/hash; **halts rather than bypassing a failed diagnostic**.
5. Scores every terrain-eligible building and computes citywide B0/B1 comparisons for both scenarios.

**Resume:** Restart the same command after an interruption. Completed files are left untouched. Partially created fit directories or inconsistent results cause a halt for inspection, not destructive overwrite. A terminated process before saving a fit can require manual cleanup after verifying incomplete files. Do not delete any scientific result by guesswork.

Logs: `outputs/phase4/_m1g2_runner/m1g2_runner.log`.

Results for each city/scenario:

`outputs/phase4/<CITY>/m1g_citywide_v0.1/<gamma_0|gamma_neg0p8>/citywide_comparison.parquet`

Plus `B0/` and `B1/` posterior netCDF, diagnostics and per-building scores.

## Scientific and Operations Center boundaries

- **Synthetic outcomes**, not observed flood losses or validated hazard probabilities.
- The B0 model is a **matched full-city refit without terrain**, not the immutable original Phase 3 M0 results. M0 must be read separately to display a Phase 3 versus M1 comparison.
- The output comparison parquet includes `bldg_id`, `T50_m`, `p_mean_B0/B1`, `p_topk_B0/B1`, `delta_p_topk_B1_minus_B0`, selection flags and `decision_change`. It **does not contain building geometry, coordinates, or interactive map tiles**. The later map-publishing join requires source building geometry, CRS/license review and validation.
- Full city fitting can consume significant CPU/RAM/time; this was not benchmarked here and no actual city fits were run in this packaging environment.
- If convergence thresholds fail, stop and diagnose; do not claim the results are scientific deliverables.
- As specified in the protocol, RTM excludes 868 records lacking terrain; DO NOT silently impute these.

## Validation

Run `python -m pytest tests/test_m1g.py tests/test_m1g_citywide.py -q` and `python -m rhb.pipelines.m1g_citywide --city all --check`. After execution, confirm 6 comparison files (3 cities × 2 scenarios), 12 complete fit diagnostics and all expected counts/hashes. Map QA remains a separate stage.
