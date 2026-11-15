# Habnetic | M1-G2: Full-city matched terrain-effect experiment

**Protocol version:** 0.1.0 · **Proposed:** 14 November 2026 · **Status:** implementation prepared, synthetic source-free tests passed, **not executed on real datasets or PyMC-reviewed**.

## Goal and interpretation

Fit two **independent, within-city** Bayesian logistic models on exactly the same terrain-eligible complete building universe, synthetic outcomes, and fixed budget. Compare posterior building-level selection with and without a measured relative-terrain feature. Three cities: Rotterdam, Hamburg, Donostia.

**This is not flood forecasting validated by observations.** The results are conditional on a synthetic Bernoulli outcome generator. Citywide model ranking is not a prediction of observed flood damage. A change in priorities does not prove a planning benefit.

## Frozen baseline, scope, and eligibility

- Phase 3/M0 code and RTM M1-C to M1-F remain frozen and unchanged.
- RTM Phase 3: **221,324** buildings; 220,456 terrain measured; **868 are unscored in matched B0/B1** (report separately, not imputed).
- HAM: 341,530 Phase 3 and terrain-eligible buildings.
- DON: 7,755 Phase 3 and terrain-eligible buildings.
- **Total measured candidate universe: 569,741 buildings**. Full-city means **all eligible** assets, not necessarily all original Phase 3 assets.
- B0 is **refit** on the same full eligible population and generated outcome realization as B1; Phase 3 posterior is a contextual reference, **not** a paired comparator.

## Generative story

For every eligible building i, preserve Phase 3 standardized E and H predictors and the measured T50 in metres, transformed to dimensionless `T = asinh(T50/0.5 m)`.

`p_true_i = sigmoid(-3.0 + 1.0*E_i + 1.0*H_i + gamma_true*T_i)`

`Y_i ~ Bernoulli(p_true_i)`; two **predeclared synthetic scenarios**: gamma=0 and gamma=-0.8. The same uniform random numbers are used across the gamma scenarios within each city, providing paired synthetic draws. No empirical flood impacts are used.

## Inference models

B0: `logit(p_i) = alpha + beta_E*E_i + beta_H*H_i`.

B1: `logit(p_i) = alpha + beta_E*E_i + beta_H*H_i + gamma_T*T_i`.

Likelihood `Y_i ~ Bernoulli(p_i)` in both models. Independent baseline priors `alpha, beta_E, beta_H ~ Normal(0, 2.5)`; B1 additionally `gamma_T ~ Normal(0, 0.5)`. 4 NUTS chains, 1,000 tune, 1,000 draws/chain, target accept .90, cores=2. Parameter names and priors match frozen Rotterdam M1-C. The new implementation omits the *observation-level* deterministic and stored log-likelihood tensors to keep RAM bounded; this is an implementation/performance choice, not a likelihood change.

B0/B1 are trained on **all** eligible buildings, rather than sample fitting followed by extrapolation. Training and validation are not conflated: the separate M1-G1 4,000/1,000 held-out experiments test predictive behaviour; M1-G2 describes full-city conditional decisions, without out-of-sample predictive validation by itself.

## Decision quantities

- Every retained posterior scoring draw calculates probabilities for **every** eligible asset, ranks the entire city, and selects exactly k (1%, `round(0.01*N)`; ties by ascending canonical building ID).
- Compute per-building posterior-mean probability, full-city top-k inclusion probability, rank SD, mean-risk selected status and synthetic-oracle selected status.
- Score **at most 250 evenly spaced draws per MCMC chain** (1,000 total with the reference sampling configuration) to bound CPU and memory. Probabilities are **finite-draw Monte Carlo estimates**, not exact integrals; record draw count and assess Monte Carlo sensitivity before publication.
- The mean-risk top-k and posterior draw-wise top-k are distinct estimands. Sum of inclusion probabilities must equal k to within 1e-6.
- Compare B0 vs B1 using exact matching IDs, budgets and synthetic truth: overlap, entries/exits, borderline count (`0.2 < inclusion < 0.8`), deltas in inclusion probabilities, rank variability, and **synthetic** oracle regret.

## Quality gates

1. **Input audit:** complete one-to-one joins; fixed population counts; finite T50, E/H; frozen original source SHA-256 (HAM batches included); RTM missingness explicitly accounted for.
2. **Preparation:** deterministic paired synthetic outcomes, SHA256 scenario files and run manifest, no overwrites.
3. **PyMC smoke:** small fixed subset (not accepted as citywide); check model builds, sampling, traces and numerical warnings.
4. **Full fits:** all buildings, both B0/B1 and both gamma scenarios. Review divergences=0, R-hat <=1.01, bulk/tail ESS >=400, plus trace behaviour.
5. **Prior/posterior predictive:** full-city aggregate simulated event-rate distributions recorded alongside synthetic observed rate; additional feature-stratified PPC desirable before interpreting substantive effects. Aggregate-only PPC is **not** a complete model check.
6. **Sensitivity:** null gamma=0 negative control; gamma=-0.8 positive control. Additional alternate gamma prior width and posterior-draw sensitivity must be recorded before stronger empirical claims. Held-out predictive comparisons come from M1-G1; LOO/WAIC not used for enormous in-sample citywide draws.
7. **Output integrity:** every model scores exactly all eligible buildings; `sum_i P(i in top-k) = k`; selected B0/B1 exactly k; entries=exits; matched ID order; SHA256 output manifest.
8. **Scientific acceptance:** no tag/publication until reviewing inference, PPC, source coverage, sample-vs-full differences and sensitivity. Git tag alone does not archive ignored NetCDF/Parquet outputs.

## Run instructions (from scientific repo root)

```
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m pytest tests/test_m1g_citywide.py -q
python -m rhb.pipelines.m1g_citywide --city all --check
python -m rhb.pipelines.m1g_citywide --city DON --prepare
python -m rhb.pipelines.m1g_citywide --city DON --smoke --gamma -0.8
```

After checking smoke records, generate full Donostia results **one scenario at a time**:

```
python -m rhb.pipelines.m1g_citywide --city DON --fit --gamma -0.8
python -m rhb.pipelines.m1g_citywide --city DON --score --gamma -0.8
python -m rhb.pipelines.m1g_citywide --city DON --compare --gamma -0.8
```

Then repeat `--fit/--score/--compare` for `--gamma 0`, and repeat both scenarios for RTM and HAM after running `--prepare` separately. For interrupted two-model fits, use `--model B1` (or B0) to resume the incomplete model, instead of overwriting a successful model output. No raw source or scientific outputs are written to Git; outputs live in `outputs/phase4/{CITY}/m1g_citywide_v0.1/`.

**Reproduction:** pin Python 3.11.9, scientific dependency versions, full code and data Git SHAs; preserve configuration and frozen scenario SHA256; command sequence above. On acceptance, separately archive NetCDF/Parquet and publish only curated map-ready extracts via a reviewed adapter.

## What to test next

Start with **DON**, the smallest full eligible population, and complete a verified full citywide B0/B1 comparison. Inspect gamma recovery, negative-control results, convergence and posterior predictive diagnostics; then extend to RTM/HAM without methodological changes.
