# RTM Phase 1 — Pluvial Hazard Design (ERA5-based)

Status: DESIGN LOCKED (Phase 1)
Scope: Deterministic pluvial hazard intensity only. No UI, no new outcomes, no new hazards.

## Goal

Replace the synthetic pluvial hazard proxy (H_pluvial_v0) with a REAL, deterministic,
ERA5-based pluvial hazard intensity signal for Rotterdam (RTM), producing H_pluvial_v1.

This Phase 1 hazard is an *intensity index*, not a probability of flooding and not damage.
Probabilistic hazard (return periods, GEV, etc.) is explicitly deferred to Phase 2.

## Data source (locked)

Dataset: ERA5-Land hourly (Copernicus/ECMWF).
Rationale: Higher spatial resolution than ERA5; buildings are on land.  
Refs:
- ERA5-Land hourly dataset overview. :contentReference[oaicite:2]{index=2}

## Variable (locked)

Primary variable: `tp` = total precipitation.
Unit handling:
- Convert meters to millimeters: `tp_mm = tp * 1000`.

Notes on accumulation semantics:
- ERA5-family precipitation variables are accumulated quantities; for hourly usage we treat the
  provided hourly steps as hourly totals. ARCO documentation explicitly discusses de-accumulation
  of accumulated variables (e.g., total precipitation) to hourly resolution. :contentReference[oaicite:3]{index=3}

## Hazard definition (locked)

We define a single deterministic pluvial hazard intensity metric:

**H_pluvial_v1 = mean annual maximum 1-hour precipitation over 1991–2020** (mm)

Formal definition per grid cell g:

1) Hourly precipitation series:
   P_t(g) = hourly total precipitation (mm) at time t

2) Annual maxima:
   AMAX_y(g) = max over all hours t in year y of P_t(g)

3) Climatological aggregation (deterministic intensity):
   H_pluvial_v1(g) = mean over years y in [1991..2020] of AMAX_y(g)

Interpretation:
- “Typical” annual worst-hour rainfall intensity at the ERA5-Land grid scale.
- Not a return level. Not a probability. Pure deterministic intensity index.

## Spatial aggregation to buildings (locked)

Target: building-level hazard aligned with existing RTM building index (`bldg_id`).

Building mapping:
- Compute building centroid in EPSG:28992, reproject to lat/lon as needed for ERA5-Land.
- Assign hazard via bilinear interpolation from the four nearest grid points:
  H_pluvial_v1_bldg = interp_bilinear(H_pluvial_v1_grid, centroid)

Store both:
- Grid product: H_pluvial_v1_grid (NetCDF and/or GeoTIFF)
- Building product: table keyed by `bldg_id`

## Outputs (expected)

1) Grid-level hazard (deterministic):
   processed/RTM/hazards/pluvial/H_pluvial_v1_grid.nc
   (optional companion: .tif for QGIS sanity)

2) Building-level hazard:
   processed/RTM/hazards/pluvial/H_pluvial_v1_buildings.parquet

Schema (buildings parquet):
- bldg_id (stable id from Phase 0)
- H_pluvial_v1_mm (float)
- hazard_src = "ERA5-Land"
- hazard_metric = "mean_annual_max_1h_1991_2020"
- hazard_version = "v1"
- optional: lat, lon (centroid) for debugging only

## Quality checks (must pass)

Spatial sanity:
- Join building hazard to buildings_rtm.gpkg and render in QGIS:
  expect coherent coastal-to-inland gradients and no obvious tiling artifacts.

Range sanity:
- H_pluvial_v1_mm must be >= 0 everywhere.
- Inspect distribution (min/median/p95/max). Look for spikes or zeros from missing data.

Completeness:
- 1 row per bldg_id, no drops.
- No NaNs after interpolation (if NaNs exist, fill using nearest-neighbor and flag count).

Reproducibility:
- Record CDS request metadata: years, variable list, spatial bbox, time zone handling,
  and dataset version/month of retrieval.

## Failure modes (known)

- Grid too coarse for urban hydrology: this is accepted in Phase 1 (we’re building the scaffold).
- Coastal cells / mixed land-sea behavior: mitigated by using ERA5-Land and restricting to buildings.
- Accumulation semantics misunderstandings: mitigated by treating hourly steps consistently and
  validating against known Dutch rainfall magnitudes.

## Non-goals (explicit)

- No surface runoff modeling.
- No drainage capacity / imperviousness / sewer system.
- No event-based flood depth.
- No probabilistic return periods (Phase 2).

## What to test next

- Compare H_pluvial_v1 spatial pattern with KNMI climatology at a coarse level (qualitative).
- Sensitivity: swap period to 1981–2010 and confirm patterns are stable.
- Optional internal check: compute also p99 hourly precip and confirm it correlates strongly with AMAX metric.
