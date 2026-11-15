# Habnetic M1-G2 | Scientific audit package v0.1

## What it is

An **independent, read-only** retrospective scientific audit of the *already finished* M1-G2 complete-city inference run. This package never fits the original models again and never modifies their frozen CSV, NetCDF or Parquet files. It uses the original pipeline's reader and a separate verification script.

**What it is not:** independent validation on observed flood damage. The evidence remains synthetic and B1 includes the covariate used to generate the positive-control outcomes. The script cannot independently verify Swiss/German/Spanish vertical datum, building geometries or historical flood observations without external sources.

## Install

Extract this ZIP into the existing `resilient-housing-bayes` repository root (do **not** delete anything already installed):

```powershell
Expand-Archive `
  -Path "$HOME\Downloads\Habnetic_M1G2_Scientific_Audit_v0.1.zip" `
  -DestinationPath "." -Force
```

Ensure the Python 3.11 virtual environment is active. The script uses the repository's existing PyMC, ArviZ, numpy, pandas, scipy, pyarrow, xarray and matplotlib packages. It does not require a dependency upgrade.

## One-command scientific audit

```powershell
.\RUN_M1G2_SCIENTIFIC_AUDIT.ps1
```

The scientific audit does **not rerun the 12 Bayesian fits**. It recomputes summaries, checks all 12 full-posteriors and results, generates feature-stratified synthetic posterior predictive checks, checks selected-draw Monte Carlo sensitivity, evaluates an alternate-prior approximation, and writes trace images.

A complete audit is more computationally involved than a quick integrity check because posterior ranking sensitivity requires re-ranking the whole population. It is nevertheless lighter than rerunning NUTS.

Alternative quick check:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python .\scripts\audit_m1g2_full.py --quick
```

`--quick` skips feature-stratified predictive checks, scoring-draw sensitivity, and prior sensitivity. **Quick is not the complete audit.**

## Outputs (new, separate from frozen experiment)

```text
outputs/phase4/_m1g2_scientific_audit_v0.1/
  AUDIT_REPORT.md
  audit_results.json
  traces/
    DON_gamma_0_B0.png
    ... (one trace chart for each of the 12 fitted models)
```

If an integrity requirement fails, the script exits non-zero and records `audit_failure.txt` in that separate directory. None of the frozen outputs is edited.

## QA performed

1. Raw source fingerprints and frozen code fingerprints; city eligibility and building ID completeness.
2. Source and scenario integrity, deterministic regeneration of synthetic labels, SHA256 of artifacts.
3. Exact B0/B1 matching, top-1% sizes, inclusion sum `= k`, selections, grouping, regret.
4. Read posterior NetCDF; recompute parameter R-hat, ESS, HDIs, divergences, BFMI, and traces for all 12 fits.
5. Recovery of generating coefficients for B1 (coverage is reported, not treated as a deterministic guarantee).
6. Original aggregate prior/posterior predictive event-rate checks are inspected and retained.
7. Synthetic posterior predictive event rates stratified separately into quintiles by E/H/T.
8. Expected out-of-sample *replicate* Brier/log loss **under the synthetic known generating probabilities** at the same building features; no label leakage from observed damage because there are no observed damages.
9. Draw sensitivity of 1,000-score-draw top-k probabilities (two interleaved halves) with exact recomputation against frozen `p_topk`.
10. Alternate terrain Gaussian prior `sigma_T=0.25` and `1.0` assessed by self-normalised posterior importance reweighting with effective-sample-size diagnostics. This is **not a prior refit** and may fail if weighted support is inadequate.

## Manual scientific acceptance gates

- Independently inspect terrain vertical datums, projections, and extreme T50, especially Donostia; examine spatial distribution of RTM's 868 omitted buildings.
- Inspect trace plots and meaningful stratified predictive discrepancies. The intentional B0 misspecification under γ = -0.8 may cause nontrivial mismatches.
- Confirm that the alternate-prior weights have adequate effective sample sizes; if not, rerun alternate-prior B1 full fits in a **new, separate experiment**, never overwriting the original one.
- Cross-check M1-G1 held-out evidence, archived raw data versions, selection of fitted logistic likelihood, and actual Phase 3 output alignment before claiming empirical improvement.
- Archive all original NetCDF/Parquet artifacts and SHA256 manifests outside Git.

**Do not tag scientific acceptance solely because the automated audit exits 0.** The report marks the automated gates as passed with explicit remaining external/manual limitations.

## Where to commit

Commit the script, test, this README and the audit protocol under `docs/phase4/`. Do **not** commit huge local posterior files or the audit's output folder automatically. After manual scientific review, a short curated `AUDIT_REPORT.md` can be copied into the documentation repository for publication.

## Assumptions made

The original unmodified M1-G2 engine and configuration, frozen inputs, 12 model fits, and 6 comparisons are present in the exact paths from the original `RUN_ALL_v0.2` setup.

## How to validate

The included unit tests run before the audit, the audit exits with a failing code on detected integrity problems, and the report enumerates everything that still requires human review.

## Next small step

Run the one-command audit and share `AUDIT_REPORT.md` (plus any failing traceback if it stops). The results will tell us whether acceptance is justified or further work is needed.
