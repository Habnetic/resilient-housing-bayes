# Habnetic | Phase 4 M1-F: Population robustness results

**Status:** TEMPLATE. Do not mark complete or insert unobserved results.<br>
**Protocol:** `docs/phase4/M1_F_Experimental_Protocol.md`.<br>
**Evidence:** three Rotterdam samples excluding the original M1-C assets, with synthetic outcomes only.

## Registered question and design

Compare original Phase 3 B0 (E+H) and M1 B1 (E+H+T) under `gamma_true in {0, -0.8}` on three new disjoint 5,000-building populations. Use two synthetic outcome seeds per population/effect and fixed 4,000/1,000 train/test splits. Compare top-10 of held-out buildings on synthetic oracle regret, posterior top-k stability, held-out prediction and diagnostics.

## Source integrity and population audit

TODO after `m1f_population_sampling`: document asset-source checksums, excluded 5,000 M1-C buildings, 15,000 distinct selected M1-F assets, population-specific covariate distributions and sampled outcome rates. State any terrain-selection exclusions and inventory mismatches.

## Main result: synthetic decision regret

TODO after `summarize_m1f`: insert results from `full_summary/summary_by_gamma.csv`; report separately from `full_summary/summary_by_population_gamma.csv`. Include all population/seed cases, negative-control behaviours, B1 wins/ties/losses, oracle overlap and relative regret effect sizes. Do not treat two outcome seeds as independent geographic replications.

## Held-out predictive accuracy

TODO: report Brier, held-out integrated log predictive density and RMSE against known synthetic p, with per-case contrasts as well as average contrasts. Distinguish predictive score improvement from decision improvement.

## Parameter recovery and diagnostics

TODO: inspect B1 gamma posterior and 95% intervals, R-hat/ESS/divergences and actual trace plots if flagged. Note broad baseline priors and small variation of hazard predictor H. Examine prior/PPC aggregate event-rate summaries.

## Scientific interpretation and limits

TODO: precisely state which findings extend across population samples and which fail. Three disjoint random samples within the same city are **not** a spatially blocked holdout, cross-city transfer or empirical flood-risk validation. The synthetic generator explicitly includes a terrain effect for the positive control. Equal and complete intervention effectiveness is assumed in oracle regret.

## Reproducibility

Run `python -m rhb.experiments.m1f_population_sampling`, then `python -m rhb.pipelines.run_m1f`, then `python -m rhb.pipelines.summarize_m1f` from the root of `resilient-housing-bayes`, using its Python 3.11 `.venv` and `PYTHONPATH=src`. The exact source and code SHA-256 fingerprints are in `outputs/phase4/RTM/m1f/*manifest.json`.

**What to test next:** final M1 methodological closure review. Cross-city (HAM/DON) feature portability and spatial transfer are separate experiments.
