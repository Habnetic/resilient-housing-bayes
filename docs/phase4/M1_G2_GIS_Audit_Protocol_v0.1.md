# Habnetic | M1-G2 GIS audit protocol v0.1

**State:** External geographic verification is pending. Statistical audit has passed within its synthetic scope. No observed flood-damage validation is claimed.

## Scope and questions

- **DON:** Are extreme *derived relative-terrain* values `T50 < -20 m` (4 assets), or `T50 > 20 m` (22 assets), spatially and physically plausible? What exact geometry and terrain neighborhood produced them? Verify the original computation and source datum.
- **RTM:** Which 868 Phase 3 building IDs are missing from the terrain-eligible universe? Are they systematically located at boundaries, water/coast, or other unusual situations? *Do not infer the cause from counts alone.*
- **HAM:** Are the full building IDs and geometry valid, and do their original terrain sources and CRS/vertical reference match the declared scientific input?
- **All:** Is the derived `T50` feature actually reproducible from the source terrain and building geometry? Does spatial support or input chronology limit interpretation?

## What the accompanying code checks

1. Reloads original frozen M1-G2 eligible universes from the existing repository engine.
2. Summarizes physical T50 range, quantiles, and threshold counts without recasting T50 as an absolute elevation.
3. Exports exceptional DON building IDs and their predictors, with all threshold exceptions plus upper/lower-50 diagnostic records.
4. Computes the exact RTM Phase 3 identifier difference and exports Phase 3 attributes for those 868 exclusions.
5. Summarizes E/H/T scaling and correlations to support the separate beta_H identifiability review.
6. Optionally matches authoritative building footprints via an explicit `bldg_id` or a supplied ID-column override; records match completeness, geometry validity and horizontal CRS, and exports small GeoJSON review layers.
7. Optionally inspects original DEM raster header (bounds, resolution, units, CRS, tags). **Raster header inspection does not prove vertical datum or correctness of T50.**

## Acceptance gates

| Gate | Evidence | Decision |
|---|---|---|
| T50 source formula | Terrain-generation code + independent recomputation or targeted GIS spot checks | Pending |
| DON extreme sites | Spatial overlay with terrain and imagery; explain extremes or flag errors | Pending |
| RTM exclusion geography | Map the 868 excluded buildings and quantify clustering/selection bias | Pending |
| Source provenance | Original DTM/DEM edition, source date, horizontal CRS, vertical datum, units, nodata | Pending |
| Geometry join | One-to-one building IDs, coverage, valid shapes, licensing | Pending |
| Statistical evidence | M1-G2 12 fits, PPC, MCMC, priors | Previously passed with scoped limitations |

## Scientific caution

The model's `T50 = ...` **cannot be inferred from the variable name alone**. M1-G2 preserves an upstream derived metric and applies `T = asinh(T50 / 0.5 m)`; the underlying GIS algorithm must be traced to its producing code. The `T50` value is a relative terrain difference, not an elevation relative to sea level. A negative value is not by itself implausible.

No amount of passing MCMC diagnostics can validate the vertical reference, terrain inputs, or the 868 spatial exclusions. A successful script run is *evidence collection*, not GIS scientific acceptance.

## Outputs (outside Git)

`outputs/phase4/_m1g2_gis_audit_v0.1/GIS_AUDIT_REPORT.md`, `audit_gis_results.json`, `terrain_profiles.csv`, `DON_T50_extremes_for_GIS_review.csv`, `RTM_excluded_from_terrain_population.csv`, `predictor_scales.csv`, `predictor_correlations.csv`, plus optional QGIS-ready GeoJSON layers.

## What to test next

Locate the original footprint and terrain raster paths in the data repository and rerun with explicit geometry/DEM arguments, then independently review the resulting site overlays in QGIS before signing the geographic acceptance.
