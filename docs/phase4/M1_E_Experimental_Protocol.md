# Habnetic | Phase 4 M1-E: Computational confirmation protocol

**Version:** 0.1.0, 5 November 2026
**Status:** Pre-registered targeted reanalysis; no confirmation result yet.
**Evidence:** Rotterdam-derived covariates with synthetic Bernoulli outcomes only.
**Prerequisite:** Full M1-C and M1-D2 fits, both unchanged and authenticated.

## Goal

Check whether selected M1-C/M1-D2 diagnostic warnings disappear under longer, four-chain NUTS inference and whether the single M1-D2 prior-dependent top-10 swap is reproducible rather than MCMC boundary noise. This is a **computational** check, not new population or geographical validation.

## Fixed generative story, likelihood and priors

- Unchanged M1-C simulator: `Y ~ Bernoulli(sigmoid(-3 + E + H + gamma_true * T))`, `T=asinh(T50/0.5m)`, synthetic truth available solely to evaluate selections.
- Unchanged 5,000 Rotterdam buildings, train/test IDs (4,000/1,000), 10-of-1,000 held-out selection, fixed outcome seeds. No test labels during fitting.
- B0: Phase 3 E/H logistic model; intercept/E/H priors independently `N(0,2.5)`.
- B1 P0: same plus terrain T with `gamma_T ~ N(0,0.5)`.
- B1 P1/P2: same B1 likelihood and slope priors; intercept respectively `N(-3,1)` and `N(-3,0.5)`. These are oracle-centred synthetic alternatives, **not** priors defensible for field deployment.
- Compare posterior coefficient summaries, predictive scores, posterior predictive aggregate rates, draw-wise top-k inclusion and oracle-relative regret **under the same original synthetic outcomes**.

## Registered confirmations

Three *distinct* diagnostic warnings (not repeated counts from reused fits):

| Source | `gamma_true` | Outcome seed | Model/prior | Original warning |
|---|---:|---:|---|---|
| M1-C | -0.8 | 101 | B1 / P0 | tail ESS 394 |
| M1-C | -0.8 | 404 | B0 | R-hat 1.02 |
| M1-D2 | -0.8 | 505 | B1 / P2 | tail ESS 373 |

Boundary case: `gamma_true=-0.4`, outcome seed `101`, B1 priors **P0/P1/P2**. Original P1 and P2 each differed from P0 by one of the ten selected assets. Refit **each prior twice independently** and compare P0/P1/P2 within each repeat; compare independent repeats under each prior and compare against the original decisions. This design does **not** assume that a one-asset swap proves meaningful prior sensitivity.

**Total:** three flagged refits + six boundary refits = **nine** new fits, not another factorial experiment. Smoke mode computes four of these; full mode reuses smoke checkpoints.

## Computation and diagnostic checks

- Four chains, 1,000 tuning iterations and 1,000 retained draws *per chain*, `target_accept=0.92`, up to two concurrent workers, stable per-case/repeat random seeds.
- Warning thresholds: maximum R-hat **>1.01**, minimum bulk/tail ESS **<400**, any divergences. More detailed trace inspection is mandatory even if these thresholds pass; these thresholds are pragmatic screening rules, not scientific proof of convergence. BFMI/energy behaviour may be examined if doubts persist.
- Recompute original diagnostics directly from stored NetCDF draws, rather than trusting the exported rounded summary alone.
- Review saved parameter trace plots `trace.png` for chain mixing, pathologies and persistent trends.
- Recompute held-out Brier, integrated Bernoulli log predictive density and RMSE against known synthetic probabilities; prior predictive and posterior predictive aggregate event-rate checks.
- Check top-10 overlaps, swapped asset IDs, maximum changes to inclusion probabilities and synthetic oracle-relative regret. Nonzero differences at the boundary may also reflect Monte Carlo variation.

## Acceptance and stop conditions

1. Existing references authenticate, including input hashes and M1-D2 prior-audit identity. Do not modify or overwrite M1-C, M1-D1 or M1-D2.
2. All registered fits finish with four chains and adequate R-hat/ESS, no unexplained divergences, inspected traces; any remaining warning is recorded and investigated.
3. For the boundary, report the original decisions, within-prior repeat agreement, and P0 versus P1/P2 comparisons **for each independent repeat**. Only call a swap reproducible when its membership difference persists with acceptable diagnostics and is not plausibly dominated by MCMC error.
4. If selection changes between independent repeats under the *same* prior, label boundary prior-sensitivity attribution **unresolved**. Do not cherry-pick a sampler seed.
5. Preserve generated CSVs, trace PNGs, NetCDFs, manifests and SHA-256 data/code references in controlled research storage. `outputs/**` is Git-ignored.

M1-E cannot verify flood damage, causality, physical validity of T50, or generalization to Hamburg/Donostia. Those are separate scientific questions.

## Reproduce from repository root

Activate the project's Python 3.11 `.venv`; set `PYTHONPATH` to `src` for module resolution.

```powershell
python -m pytest tests/test_m1e.py -q
python -m rhb.pipelines.run_m1e --audit-only --trace-plots
python -m rhb.pipelines.run_m1e --dry-run
python -m rhb.pipelines.run_m1e --smoke
python -m rhb.pipelines.summarize_m1e --smoke
python -m rhb.pipelines.run_m1e
python -m rhb.pipelines.summarize_m1e
```

Inputs: `outputs/phase4/RTM/m1c/m1c_scenarios.parquet` and original full fit artifacts, plus `outputs/phase4/RTM/m1d2/prior_predictive/` identity manifest. Outputs: `outputs/phase4/RTM/m1e/` with `reference_audit.csv`, `fits/<case_id>/repeat_<n>/{idata.nc,trace.png,posterior_summary.csv,fit_record.json,test_decisions.parquet}`, `smoke_results.csv`, `full_results.csv`, `full_summary/` and manifests. No Git-tracked result binaries are created.

**What to test next:** Confirm original diagnostics first, run the four-fit smoke, and inspect whether P0/P1/P2 differences persist when each model is independently resampled. If confirmed, complete the nine-fit suite; if not, document the boundary as computationally unstable rather than extending M1 indefinitely.
