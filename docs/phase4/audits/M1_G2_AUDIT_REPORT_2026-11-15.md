# Habnetic | M1-G2 scientific audit (automated evidence)

**Run:** 2026-11-15T15:39:41.366543+00:00  
**Status:** AUTOMATED_CHECKS_PASS_WITH_SCOPED_LIMITATIONS  
**Scope:** Frozen 3-city synthetic experiment; no empirical flood-damage validation.

## Provenance

- Scientific Git commit: `d1396ebf8e5d784260fd4ecc1f69316441fc2aa2`
- Python: `3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`
- M1-G2 engine/config/source checksums checked against existing manifests.

## Citywide results

| City | Scenario γ | N | k | Replacements | Borderline B0/B1 | Regret B0/B1 | Δ expected logloss B1−B0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DON | 0.0 | 7,755 | 78 | 0 (0.00%) | 10/24 | 0.000000/0.000000 | +0.00000094 |
| DON | -0.8 | 7,755 | 78 | 49 (62.82%) | 14/10 | 31.367451/0.003040 | -0.15334123 |
| RTM | 0.0 | 220,456 | 2,205 | 7 (0.32%) | 26/64 | 0.002933/0.015649 | +0.00000040 |
| RTM | -0.8 | 220,456 | 2,205 | 493 (22.36%) | 35/68 | 59.120462/0.009864 | -0.00287962 |
| HAM | 0.0 | 341,530 | 3,415 | 26 (0.76%) | 46/79 | 0.005606/0.067903 | +0.00000479 |
| HAM | -0.8 | 341,530 | 3,415 | 1,511 (44.25%) | 47/100 | 282.578002/0.130386 | -0.01069641 |

## B1 coefficient recovery and fitting

| City | γ true | γ mean | γ 95% HDI | Max R-hat | Min ESS bulk/tail | BFMI min |
|---|---:|---:|---|---:|---:|---:|
| DON | 0.0 | -0.001 | [-0.028, 0.028] | 1.000 | 1702/1754 | 0.919 |
| DON | -0.8 | -0.810 | [-0.851, -0.768] | 1.000 | 1441/1350 | 0.988 |
| RTM | 0.0 | 0.012 | [-0.030, 0.055] | 1.000 | 3179/2708 | 1.061 |
| RTM | -0.8 | -0.797 | [-0.839, -0.757] | 1.000 | 3518/2877 | 1.094 |
| HAM | 0.0 | -0.017 | [-0.036, 0.003] | 1.000 | 1401/1661 | 0.988 |
| HAM | -0.8 | -0.819 | [-0.837, -0.799] | 1.010 | 1506/1387 | 0.998 |

## Full audit checks

Automated: original raw-input fingerprints, deterministic synthetic generation, building-ID alignment, parquet SHA256 checks, complete fit cardinalities, netCDF/CSV posteriors, MCMC diagnostics, per-draw top-k sums, rankings and regret.
Trace plots are in `traces/`.

### Feature-stratified synthetic posterior predictive checks

| City | γ | Model | Missed 95% intervals (E/H/T bins) | Maximum split-draw Δp_topk |
|---|---:|---|---|---:|
| DON | 0.0 | B0 | 0/0/1 | 0.0420 |
| DON | 0.0 | B1 | 0/0/0 | 0.0580 |
| DON | -0.8 | B0 | 0/1/5 | 0.0360 |
| DON | -0.8 | B1 | 0/0/0 | 0.0460 |
| RTM | 0.0 | B0 | 0/0/0 | 0.0520 |
| RTM | 0.0 | B1 | 0/0/0 | 0.0860 |
| RTM | -0.8 | B0 | 2/0/4 | 0.0420 |
| RTM | -0.8 | B1 | 0/0/0 | 0.0740 |
| HAM | 0.0 | B0 | 0/0/2 | 0.0540 |
| HAM | 0.0 | B1 | 0/0/1 | 0.0640 |
| HAM | -0.8 | B0 | 1/0/5 | 0.0800 |
| HAM | -0.8 | B1 | 0/0/1 | 0.0700 |

### Approximate terrain-prior sensitivity (B1, importance reweighting)

| City | γ | Prior σ | Importance ESS | Weighted γ | Top-k replacements |
|---|---:|---:|---:|---:|---:|
| DON | 0.0 | 0.25 | 1000.0 | -0.0010 | 0 |
| DON | 0.0 | 1.0 | 1000.0 | -0.0010 | 0 |
| DON | -0.8 | 0.25 | 959.5 | -0.8063 | 0 |
| DON | -0.8 | 1.0 | 997.3 | -0.8117 | 0 |
| RTM | 0.0 | 0.25 | 1000.0 | 0.0106 | 0 |
| RTM | 0.0 | 1.0 | 1000.0 | 0.0107 | 0 |
| RTM | -0.8 | 0.25 | 960.6 | -0.7930 | 3 |
| RTM | -0.8 | 1.0 | 997.4 | -0.7983 | 1 |
| HAM | 0.0 | 0.25 | 1000.0 | -0.0168 | 0 |
| HAM | 0.0 | 1.0 | 1000.0 | -0.0168 | 0 |
| HAM | -0.8 | 0.25 | 991.4 | -0.8187 | 2 |
| HAM | -0.8 | 1.0 | 999.5 | -0.8198 | 0 |

## Remaining *manual / external* scientific acceptance gates

1. Confirm vertical datums, CRS, original DEM footprint, RTM 868 excluded locations and DON extreme T50 in the source GIS.
2. Review trace PNGs for mixing and trends; automated R-hat/ESS alone is not complete review.
3. Interpret stratified PPC failures relative to intentionally misspecified B0 under γ = -0.8.
4. If importance ESS is inadequate, refit B1 under an alternative gamma prior; importance weighting is an approximation, not a true refit.
5. Compare with M1-G1 held-out evaluations and independently validate Phase 3 baselines.
6. Archive NetCDF, Parquet and manifests off-Git with checksums; tag only after the review.

## Interpretation boundary

These synthetic labels are generated with a logistic terrain effect and B1 contains that exact covariate. Recovery of γ and reduced oracle regret under γ = -0.8 are positive controls, not validation on observed flood impacts. γ = 0 is a negative control. The expected replicate log loss is calculated analytically from the known synthetic generating probabilities on the same geographic feature rows.

## Warnings

- No automated warnings.

## What to test next

Resolve the manual gates and compare feature-stratified PPC to M1-G1 held-out results before assigning a scientific-acceptance Git tag.
