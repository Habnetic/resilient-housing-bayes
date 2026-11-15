# Habnetic M1-G2 | additive full-city implementation v0.1.0

**Scope**: RTM 220,456 eligible, HAM 341,530 eligible, DON 7,755 eligible. Synthetic B0/B1 full-universe fits and building-by-building posterior top-k decisions for both predeclared terrain-effect scenarios. Neither frozen Phase 3/RTM M1 nor M1-G1 files are modified.

**Do not use** `Expand-Archive -Force` against a modified version of these same filenames. This package adds *new* files only.

## Install (PowerShell, from `resilient-housing-bayes` branch `phase4/m1g-cross-city`)

```
Expand-Archive -Path "$HOME\Downloads\Habnetic_M1G2_FullCity_v0.1.zip" -DestinationPath .
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1g_citywide.py -q
python -m rhb.pipelines.m1g_citywide --city all --check
```

**Stop after the source audit.** The next computational stages require first inspecting the actual input report and sampling resources on this PC.

After the audit, commit these additive files as a new implementation commit, and only then prepare scenarios. Do **not** create a scientific release tag until all city fits, scoring and map-publication QA have passed.

## Files added

- `configs/m1g_citywide_v0.1.json`
- `src/rhb/pipelines/m1g_citywide.py`
- `tests/test_m1g_citywide.py`
- `docs/phase4/M1_G2_Full_City_Protocol.md`
- `docs/phase4/Operations_Center_v0.1_Maps_Amendment.md`
- `M1G2_SETUP_README.md`

## Provenance and runtime notes

The script reads existing RTM M1-F and HAM/DON M1-G1 full-city input adapters. Those two earlier implementations and their citywide derived terrain datasets must be present. Python 3.11, SciPy, pandas, NumPy, PyMC and ArviZ required. Existing 15 M1-G1 tests should continue passing. New source-free tests verify decision cardinality, pairing and probability-sum invariants; **no full-city PyMC run was executed by the package author**.

Do not present M1-G2 as observed validation. The complete 569,741 eligible buildings require two B0/B1 fits per gamma scenario per city, 12 fits total, plus posterior-draw scoring. Full 341,530-building Hamburg fits can be computationally intensive.

See `docs/phase4/M1_G2_Full_City_Protocol.md` for model assumptions, QA, reproduction and further stages.
