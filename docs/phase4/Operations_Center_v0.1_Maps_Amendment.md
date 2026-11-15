# Habnetic Operations Center v0.1 | Geographic and terrain comparison amendment

**Proposal date:** 14 November 2026. **Status:** approved research requirement, implementation not yet started.

This amendment supersedes the 29 October 2026 specification's deferral of citywide maps and M0–M1 decision comparison. Keep the original source document unchanged for audit; merge this approved amendment into the canonical `Habnetic/docs` specification during implementation.

## v0.1 required geographic features

- City selector: RTM, HAM, DON.
- M0 (Phase 3) citywide decision map, plus M1-G2 matched B0 and terrain-informed B1 map.
- B0/B1 difference layer: stayed selected, entered, left, not selected; continuous delta in posterior top-k inclusion probability.
- Full terrain-eligible building population behind the map; explicit RTM no-terrain exclusion count (868) and optional distinct unscored overlay.
- Every interactive building can be inspected for ID, model, scenario, posterior-mean probability, inclusion probability, selected status, rank variability, T50, inference evidence and provenance.
- Capacity k must correspond to the fixed 1% target candidate set for each M1-G2 experiment. Expose alternatives only when newly calculated, not by reusing inclusion probabilities computed for a different k.
- Display synthetic evidence prominently: **model-conditional decisions, not observed flood hazard, flood probability, or certified risk**.
- Responsive read-only interface with accessible keyboard navigation and meaningful color alternatives/legends.

## Geographic artifact pipeline

1. Scientific source: citywide score `citywide_comparison.parquet`, immutable manifest and SHA256.
2. Spatial join in `Habnetic/data` using authoritative original building footprints by `bldg_id`. For each city, prove unique IDs, CRS consistency, complete matched geometry, count == eligible population, no missing geometry.
3. Publish versioned browser-efficient geographic tiles (or compact spatial layers at appropriate zoom); do **not** ship 569,741 polygon features/NetCDF posterior draws into a single browser JSON.
4. Provide curated JSON summaries and immutable URLs through existing static GitHub-backed publication layer. Keep raw copyrighted/licensed data excluded if their licenses disallow public redistribution; document rights and attribution for each city.
5. Validate counts, B0/B1 membership k, `P(top-k)` sum, SHA256 provenance, map-asset join and rendering at city and building zoom levels before public index update.

The initial viewer is **blocked** for its M1 geographic comparison until real full-city scientific results pass review. The existing Phase 3 data can populate a separate map view while M1-G2 runs proceed. The scientific pipeline does not depend on the browser.

## What to test next

Verify source building footprint accessibility/license per city and validate the GIS join on the first accepted Donostia M1-G2 decision Parquet before exporting nationwide or cross-city tiles.
