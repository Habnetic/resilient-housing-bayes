# Habnetic M1-G2 | GIS/source audit package v0.1

**Goal:** Collect concrete geographic evidence needed to close M1-G2 without refitting Bayesian models or altering the frozen outputs.

## Install into scientific repository

From `C:\Users\C.Price\Habnetic\resilient-housing-bayes`:

```powershell
Expand-Archive -Path "$HOME\Downloads\Habnetic_M1G2_GIS_Audit_v0.1.zip" -DestinationPath . -Force
```

## Run without installing any new dependency

```powershell
.\RUN_M1G2_GIS_AUDIT.ps1
```

The PowerShell runner explicitly invokes `resilient-housing-bayes\.venv\Scripts\python.exe`, even if a neighboring `data\.venv` was accidentally activated. Requires only pandas, numpy, pyarrow and the previously working project code. A regression test runs first. **This first pass does not require GIS geometries.**

Then inspect:

```powershell
Get-Content .\outputs\phase4\_m1g2_gis_audit_v0.1\GIS_AUDIT_REPORT.md
```

Upload `GIS_AUDIT_REPORT.md`, `audit_gis_results.json`, `DON_T50_extremes_for_GIS_review.csv`, `RTM_excluded_from_terrain_population.csv`, `predictor_scales.csv`, and `predictor_correlations.csv` for interpretation.

## Second pass with authoritative source footprints and optional source DEM

**Replace the `TODO` paths with the real, existing files in `Habnetic/data` or your source archive.** Paths are illustrative, not discovered data:

```powershell
.\RUN_M1G2_GIS_AUDIT.ps1 `
  --don-geometries "C:\PATH\TO\DON_buildings.gpkg" `
  --rtm-geometries "C:\PATH\TO\RTM_buildings.gpkg" `
  --ham-geometries "C:\PATH\TO\HAM_buildings.gpkg" `
  --don-dem "C:\PATH\TO\DON_source_dem.tif"
```

For files whose ID field is not `bldg_id`, supply `--don-id-column`, `--rtm-id-column`, or `--ham-id-column`. For multi-layer GeoPackages, supply `--don-layer`, etc. Supported footprints: .gpkg, .shp, .geojson, GeoParquet. The second pass requires `geopandas`; DEM metadata requires `rasterio`, both only if their respective optional arguments are used.

Exports small QGIS-ready GeoJSON files for exceptional buildings, provided they can be joined to the source geometries. The script **does not** compute a new T50 from the DEM; that requires examining the actual upstream terrain algorithm and verifying geographic source coverage.

## Important limitations

- A plain CSV of T50 cannot establish geographic validity. Without authoritative footprints, the spatial gate stays **pending**.
- DEM CRS and raster headers may not contain vertical datum. Document provider specification separately.
- DON extremes can be legitimate relative-terrain depressions/rises; do not flag them as erroneous only because values are large or negative.
- The 868 RTM exclusions do not invalidate the *matched eligible B0/B1 contrast*, but may restrict citywide coverage and interpretation.
- Source map licensing and chronology (what edition/date terrain and buildings represent) must also be recorded.
- The model remains synthetic. This GIS review is not a validation against observed inundation or damages.

## Git guidance

Commit `scripts/audit_m1g2_gis.py`, `tests/test_m1g2_gis_audit.py`, the `.ps1` runner, and both documentation files, but do not automatically commit large raw geographic data, GIS extracts, or NetCDF/Parquet outputs.

## Reproduction

Command: `./RUN_M1G2_GIS_AUDIT.ps1`. Input: frozen repository M1-G2 config/outputs, source Phase 3 and terrain parquet; optional source footprints/DEM. Output: read-only QA report and diagnostic CSV/GeoJSON under `outputs/phase4/_m1g2_gis_audit_v0.1`. It records run time and SHA256 of the M1-G2 config. A separate Git commit and data hashes must be archived for full reproducibility.

## Next small step

Run the default first pass, then share the report and source geometry paths for a physical review.
