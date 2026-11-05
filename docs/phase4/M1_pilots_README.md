# Habnetic Phase 4 | M1-A / M1-B reproducible pilots

**Status:** Pilot only, not full Phase 4 M1 validation. Synthetic outcomes; not empirical flood-damage validation.

## Code and versioning

- Git repository: `resilient-housing-bayes` (scripts, tests, documentation, model source of truth).
- Frozen Phase 3 base: `phase3-code-freeze-20261022` (annotated tag, commit `5e19254`). Do not modify Phase 3.
- Existing source: `src/rhb/models/logistic_m1.py`, `src/rhb/pipelines/run_m1a_pilot.py`, `src/rhb/pipelines/run_m1b_pilot.py`.
- Added: `src/rhb/pipelines/audit_m1_pilots.py` (persisted decision calculations), `src/rhb/pipelines/run_m1_pilots.py` (single-command workflow), `tests/test_m1_pilot_audit.py`.
- Run commands from the repository root; requires `rhb` importable (`python -m pip install -e .` if packaging supports it, otherwise set `PYTHONPATH`).

## Input data

- `outputs/phase3/RTM/phase3_features_scaled.parquet` (221,324 RTM assets; existing scaled E/H and frozen synthetic `Y_damage`).
- `../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet` (`T50`, measured flag, `bldg_id`).
- M1-A reuses the Phase 3 synthetic labels on **the same 5,000 assets** in A0/A1, selected from the 220,456 measured-T50 assets.
- M1-B reuses those assets, generates **new synthetic labels** under `alpha=-3`, `beta_E=1`, `beta_H=1`, `gamma_T=-0.8`, seed `20261105`, then trains on 4,000 assets and tests on 1,000.

**Critical limitation:** Input Parquet files may be Git-ignored and therefore not available to another researcher cloning the repo. Reproducibility requires archiving or publishing these inputs (subject to licensing) with checksums and a documented source/preparation pipeline. The audit writes SHA-256 hashes for its direct inputs but hashes alone cannot replace missing datasets.

## Model specification

- A0/B0: `logit(p) = alpha + beta_E E + beta_H H`.
- A1/B1: `logit(p) = alpha + beta_E E + beta_H H + gamma_T T`.
- `T = asinh(T50_metres / 0.5 metres)`; unchanged raw T50 is preserved.
- Priors: `alpha, beta_E, beta_H ~ Normal(0, 2.5)`; `gamma_T ~ Normal(0, 0.5)`.
- Pilot inference: 2 chains x 500 tuning + 500 posterior draws, `target_accept=0.9`, seed `20261105`.
- M1-A generator is Phase 3 synthetic E+H (without T50). M1-B generator adds `-0.8 T`.

## Reproduce

Activate the Python environment, then in PowerShell at repo root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1_pilot_audit.py -q
```

**Audit existing inference results without refitting** (recommended first):

```powershell
python -m rhb.pipelines.audit_m1_pilots
```

**Regenerate pilot datasets, refit all four models and audit in one command** (overwrites existing pilot outputs):

```powershell
python -m rhb.pipelines.run_m1_pilots
```

Record the active environment separately, for example `python -m pip freeze > requirements-m1-pilot.lock.txt`; inspect and commit only if portable. Preserve a hash/version of all upstream input artifacts.

## Outputs

`outputs/phase4/RTM/m1a_pilot/`:
- `pilot_sample.csv`, `A0_idata.nc`, `A1_idata.nc`, `A0_summary.csv`, `A1_summary.csv`
- **Added audit outputs:** `A0_decisions.csv`, `A1_decisions.csv`, `decision_audit.json`

`outputs/phase4/RTM/m1b_pilot/`:
- `m1b_dataset.parquet`, `B0_idata.nc`, `B1_idata.nc`, `B0_summary.csv`, `B1_summary.csv`
- **Added audit outputs:** `B0_decisions.csv`, `B1_decisions.csv`, `decision_audit.json`

The JSON audits include versions, current Git state, checksums of audit inputs, and decision metrics. **Commit code/tests/README to Git**; handle larger derived NetCDF/Parquet files with a documented data/artifact release rather than silently committing them.

## Decision definitions

- `selected_by_p_mean`: assets with the highest **posterior mean** event probability (k=50 for M1-A; k=10 for M1-B test).
- `p_topk`: fraction of posterior draws in which the asset belongs to the per-draw top-k.
- `borderline`: strictly `0.2 < p_topk < 0.8`.
- `oracle`: highest **known synthetic generator** event probabilities, only for M1-B; `expected_regret` is expected missed events relative to that oracle under equal per-event consequences.

## Pilot outcomes from 5 Nov 2026 (one random split/seed)

- M1-A: `gamma_T` posterior mean `-0.031`, 95% HDI `[-0.310, 0.236]`; A0/A1 top-50 overlap **49/50**; borderline **0 / 4**; mean absolute inclusion-probability change **0.000406**.
- M1-B: true `gamma_T=-0.8`; B1 posterior mean `-0.672`, 95% HDI `[-0.971, -0.351]`; top-10 overlap B0/B1 **9/10**, oracle overlaps **8/10 vs 9/10**; expected regret **0.141091 vs 0.020654**. Maximum asset-level change in inclusion probability **0.902**, despite **0 borderline** in either model.
- M1-B held-out probability RMSE improves (`0.025493 -> 0.011641`), Brier score does **not** improve in this single test draw (`0.058699 -> 0.058757`).
- Test-set H standard deviation only **0.007584**: the `beta_H` coefficient is weakly identified on this scale; revisit predictor scaling and prior predictive implications in the definitive study. Never interpret synthetic coefficients as empirical flooding effects.

## Checks required before scientific claims

1. Four-chain inference, traces, `R-hat`, bulk/tail ESS, divergences, BFMI.
2. Prior predictive plausibility and posterior predictive checks (not just aggregated event rate).
3. Evaluate out-of-sample probabilities and decisions across multiple seeds, synthetic effect sizes, and independent test populations; spatial splits where appropriate.
4. Sensitivity to `gamma_T` prior, E/H prior scales and T50 transformation (asinh, clipped/robust alternatives); inspect extreme topographic assets and measuredness.
5. Evaluate predictive model comparison (e.g. LOO on common training observations with pointwise log likelihood, and held-out log score/Brier); ensure fair A0/A1, B0/B1 splits.
6. Record input data versions, environment, software commit and output checksums in a repeatable research release.

**What to test next:** Repeat M1-B across multiple random seeds and realistic effect magnitudes before scaling to all 220,456 assets.
