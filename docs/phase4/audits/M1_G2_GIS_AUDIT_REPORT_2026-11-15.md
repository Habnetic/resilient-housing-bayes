# Habnetic | M1-G2 geographic source audit

**Run (UTC):** 2026-11-15T18:06:12.652724+00:00
**Status:** AUTOMATED_TABLES_PASS_GIS_REVIEW_REQUIRED

The numerical audit is separate and previously passed. This audit does not validate physical flood hazard or observations.

## Derived relative terrain (T50, metres)

| City | Eligible | Min | Median | Max | T50<-20 m | T50>20 m |
|---|---:|---:|---:|---:|---:|---:|
| DON | 7,755 | -23.459 | 0.169 | 43.269 | 4 | 22 |
| RTM | 220,456 | -11.229 | 0.017 | 16.083 | 0 | 0 |
| HAM | 341,530 | -12.975 | 0.043 | 19.692 | 0 | 0 |

## Evidence per city

### DON

- Eligible buildings: 7,755
- Building geometry: **GEOMETRY_CHECKED_REQUIRES_HUMAN_GIS_REVIEW**
- Source DEM: **RASTER_METADATA_READ_NOT_VERTICAL_DATUM_VERIFIED**
- Review records exported: 100
- |T50| > 20m: 26

### RTM

- Eligible buildings: 220,456
- Building geometry: **GEOMETRY_CHECKED_REQUIRES_HUMAN_GIS_REVIEW**
- Source DEM: **PENDING_ORIGINAL_DEM_OR_SOURCE_METADATA**
- Excluded Phase 3 buildings: 868

### HAM

- Eligible buildings: 341,530
- Building geometry: **GEOMETRY_CHECKED_REQUIRES_HUMAN_GIS_REVIEW**
- Source DEM: **RASTER_METADATA_READ_NOT_VERTICAL_DATUM_VERIFIED**

## What remains external / manual

1. Load the review GeoJSON (if generated) or join review CSV on `bldg_id` to original footprints in QGIS; inspect DON extrema and RTM exclusion clustering.
2. Verify the original raster/terrain source name, acquisition date, resolution, coverage, horizontal CRS, vertical datum and vertical units *from provider documentation*.
3. Verify **T50 formula**, neighborhood scale and nodata handling against the source code that originally generated the terrain parquet; do not equate T50 with absolute elevation.
4. Inspect RTM missingness mechanism and whether the 868 exclusions concentrate spatially in meaningful areas; no blanket extrapolation to excluded assets.
5. Record the source evidence, screenshots and a signed GIS review decision in the paper trail.

## Not established

No empirical flood-damage validation; no proof the three vertical datums are compatible; no proof that derived T50 values are physically correct.

## Next small step

Review the CSV outputs and, when authoritative footprints are identified, rerun with `--don-geometries`, `--rtm-geometries`, `--ham-geometries`.
