# M1-G2 | Scientific audit archive record

**Date:** 2026-11-15  
**Status:** completed synthetic full-city experiments and scientific checks; GIS/data closure accepted with explicit limitations. **Not** empirical flood-risk validation and **not** formal M1-G1/M1-G2 synthesis.  
**Archive tag proposed:** `phase4-m1g2-scientific-audit-2026-11-15`.

## Frozen scope

- Three cities: DON 7,755, RTM 220,456, HAM 341,530 eligible buildings; total 569,741. Rotterdam's 868 original buildings without T50 remain accounted for and unscored.
- Two injected synthetic terrain coefficients: 0 and -0.8. Two new within-city fits (B0 and B1) for each scenario. 12 four-chain PyMC NUTS fits; 6 paired top-1% comparisons.
- Building-level decision estimands from frozen posterior draws: `p_topk` is inclusion frequency; `selected_by_p_mean` is a distinct posterior-mean ranking. Synthetic oracle regret is expected synthetic event count, not real monetary loss.
- Original SHA-256 frozen config: `068b3a5d12b5e2bd410ce6a092cfba520901e2892bedbfa2c71e6b9eeff405c4`.

## Review evidence

1. `docs/phase4/M1_G2_Full_City_Results_2026-11-14.md`.
2. `docs/phase4/M1_G2_Visual_Trace_Review_2026-11-15.md`.
3. `docs/phase4/audits/M1_G2_AUDIT_REPORT_2026-11-15.md` and `M1_G2_audit_results_2026-11-15.json`: preserved copies of automated scientific audit evidence.
4. `docs/phase4/audits/M1_G2_GIS_AUDIT_REPORT_2026-11-15.md` and `M1_G2_audit_gis_results_2026-11-15.json`: preserved earlier automated GIS findings. **Its GIS_REVIEW_REQUIRED status is superseded by the subsequent human review, not rewritten.**
5. `data/docs/phase4/M1_G2_DATA_GIS_AUDIT_CLOSURE_2026-11-15.md`: final later manual GIS acceptance and limitations.
6. `data/docs/phase4/M1_G2_FROZEN_ARTIFACT_INDEX_2026-11-15.json`: local immutable hashes of the 6 comparisons, 12 posteriors and related output files, built at archive time.

## Decision

Accept the M1-G2 geographic/data inputs and statistical analysis for the **bounded synthetic-methodological claim**. No reason emerged to recompute terrain or rerun fits. Exact per-tile vertical-datum equivalence across HAM source providers was not independently proved; no material inconsistency was identified. The M0/Phase 3 versus M1-G2 context must stay clearly labeled, since B0 is a synthetic refit, not the historical Phase 3 result.

The Operations Center can now be **implemented** using frozen outputs, once building geometry joins, static tiles, licenses and an application UI are created. It is not yet a ready-to-deploy application.

This tag records the source-code/documentation revision for the audit. It does **not** incorporate ignored `idata.nc`, Parquet or raster binaries into Git. See the data-repository SHA256 artifact index and independently back up those source files. Do not create a broad empirical-validity release tag by implication.
