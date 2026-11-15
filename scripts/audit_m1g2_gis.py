"""Read-only evidence audit of M1-G2 terrain and population geography.

Input: frozen M1-G2 config and city tables, plus optional authoritative building
footprints and original DEM. The script never changes frozen scientific outputs.

Examples:
  python scripts/audit_m1g2_gis.py
  python scripts/audit_m1g2_gis.py --don-geometries ../data/geo/don_buildings.gpkg \
      --rtm-geometries ../data/geo/rtm_buildings.gpkg --don-dem ../data/dtm/don.tif

IMPORTANT: T50 is a derived *relative* terrain measurement, not an absolute
DEM elevation. Raster metadata inspection does not prove T50 values are valid.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
LOG = logging.getLogger("m1g2.gis")
CITIES = ("DON", "RTM", "HAM")
BAD_IDS = {"", "nan", "none", "<na>", "nat"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normal_ids(data: pd.Series) -> pd.Series:
    """Canonicalize IDs without converting nulls into valid identifiers."""
    result = data.astype("string").str.strip()
    bad = result.isna() | result.str.lower().isin(BAD_IDS).fillna(True)
    if bad.any():
        raise ValueError(f"Missing or invalid IDs: {int(bad.sum())}")
    if result.duplicated().any():
        raise ValueError(f"Duplicate identifiers: {int(result.duplicated().sum())}")
    return result


def missing_ids(reference: pd.Series, eligible: pd.Series) -> pd.Series:
    ref = normal_ids(reference)
    valid = normal_ids(eligible)
    leftover = ref[~ref.isin(set(valid))].reset_index(drop=True)
    return leftover


def terrain_profile(frame: pd.DataFrame, city: str) -> dict[str, Any]:
    t = pd.to_numeric(frame["T50"], errors="coerce").to_numpy(float)
    if not np.isfinite(t).all():
        raise ValueError(f"{city}: nonfinite T50")
    return {
        "city": city, "n": int(len(t)), "min_m": float(np.min(t)),
        "p0_1_m": float(np.quantile(t, 0.001)),
        "p1_m": float(np.quantile(t, 0.01)),
        "median_m": float(np.median(t)),
        "p99_m": float(np.quantile(t, 0.99)),
        "p99_9_m": float(np.quantile(t, 0.999)),
        "max_m": float(np.max(t)),
        "n_below_minus20_m": int((t < -20).sum()),
        "n_above_plus20_m": int((t > 20).sum()),
        "definition": "Derived relative terrain; not absolute height above sea level",
    }


def don_extreme_rows(frame: pd.DataFrame, threshold: float = 20.0, n_each: int = 50) -> pd.DataFrame:
    """Union of threshold exceptions and fixed upper/lower review samples."""
    data = frame[["bldg_id", "T50", "T", "E_hat_v0", "H_pluvial_v1_logrel"]].copy()
    t = data["T50"].to_numpy(float)
    s = np.argsort(t, kind="stable")
    positions = set(np.flatnonzero(np.abs(t) > threshold).tolist())
    positions.update(s[:n_each].tolist())
    positions.update(s[-n_each:].tolist())
    out = data.iloc[sorted(positions)].copy()
    out["review_reason"] = np.where(np.abs(out["T50"].to_numpy(float)) > threshold,
                                    "abs(T50)>20m", "upper_or_lower_50")
    out["abs_T50_m"] = np.abs(out["T50"].to_numpy(float))
    return out.sort_values(["abs_T50_m", "bldg_id"], ascending=[False, True]).reset_index(drop=True)


def predictor_diagnostics(frame: pd.DataFrame, city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["E_hat_v0", "H_pluvial_v1_logrel", "T"]
    data = frame[cols].apply(pd.to_numeric, errors="coerce")
    desc = []
    for col in cols:
        values = data[col].to_numpy(float)
        desc.append({"city": city, "predictor": col, "n": len(values),
                     "unique": int(data[col].nunique(dropna=True)),
                     "sd": float(np.std(values)), "p01": float(np.quantile(values, 0.01)),
                     "median": float(np.median(values)), "p99": float(np.quantile(values, .99))})
    cor = data.corr().reset_index().rename(columns={"index": "predictor"})
    cor.insert(0, "city", city)
    return pd.DataFrame(desc), cor


def inspect_geometry(path: Path, frame: pd.DataFrame, city: str, out: Path,
                     *, id_column: str = "bldg_id", layer: str | None = None,
                     missing: pd.Series | None = None, extremes: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compare authoritative footprints by IDs; emit small review GeoJSON layers."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        return {"status": "BLOCKED_GEOPANDAS_MISSING", "reason": str(exc), "path": str(path)}
    if not path.is_file():
        return {"status": "BLOCKED_GEOMETRY_NOT_FOUND", "path": str(path)}
    try:
        if path.suffix.lower() in {".parquet", ".geoparquet"}:
            geo = gpd.read_parquet(path)
        else:
            kw = {"layer": layer} if layer else {}
            geo = gpd.read_file(path, **kw)
    except Exception as exc:
        return {"status": "BLOCKED_READ_ERROR", "path": str(path), "reason": repr(exc)}
    if id_column not in geo.columns:
        return {"status": "BLOCKED_ID_COLUMN_MISSING", "path": str(path),
                "columns": [str(x) for x in geo.columns[:25]]}
    try:
        ids = normal_ids(geo[id_column])
    except ValueError as exc:
        return {"status": "BLOCKED_ID_INVALID", "path": str(path), "reason": str(exc)}
    geo = geo[[id_column, geo.geometry.name]].copy()
    geo["bldg_id"] = ids.to_numpy()
    wanted = set(normal_ids(frame["bldg_id"]))
    geom_ids = set(ids)
    nonempty = (~geo.geometry.isna()) & (~geo.geometry.is_empty)
    geom_good = geo.geometry.is_valid.fillna(False) & nonempty
    found = geo["bldg_id"].isin(wanted)
    eligible_not_found = wanted - geom_ids
    invalid_in_eligible = int((found & ~geom_good).sum())
    crs = geo.crs
    detail: dict[str, Any] = {
        "status": "GEOMETRY_CHECKED_REQUIRES_HUMAN_GIS_REVIEW", "path": str(path),
        "n_geometry_rows": int(len(geo)), "n_matched": int(found.sum()),
        "n_missing_eligible_geometries": int(len(eligible_not_found)),
        "n_invalid_or_empty_eligible_geometries": invalid_in_eligible,
        "crs": str(crs) if crs is not None else None,
        "crs_is_projected": bool(crs.is_projected) if crs is not None else None,
        "bounds_source_units": [float(x) for x in geo.total_bounds],
        "n_geometries_for_rtm_excluded": int(geo["bldg_id"].isin(set(missing)).sum()) if missing is not None else None,
    }
    if crs is None:
        detail["status"] = "BLOCKED_MISSING_HORIZONTAL_CRS"
        return detail
    if len(eligible_not_found) or invalid_in_eligible:
        detail["status"] = "GEOMETRY_QA_FAILURE_REVIEW_REQUIRED"
    # Only small diagnostically relevant extracts, never export full city geometries.
    try:
        if missing is not None:
            ids_missing = set(missing)
            missing_geom = geo[geo["bldg_id"].isin(ids_missing) & geom_good].copy()
            if not missing_geom.empty:
                missing_geom.to_crs("EPSG:4326").to_file(
                    out / f"{city}_excluded_buildings_review.geojson", driver="GeoJSON")
                detail["excluded_geojson"] = f"{city}_excluded_buildings_review.geojson"
        if extremes is not None:
            extreme_geo = geo[geo["bldg_id"].isin(set(extremes["bldg_id"])) & geom_good].copy()
            if not extreme_geo.empty:
                extra_cols = extremes[["bldg_id", "T50", "review_reason"]]
                extreme_geo = extreme_geo.merge(extra_cols, on="bldg_id", validate="one_to_one")
                extreme_geo.to_crs("EPSG:4326").to_file(
                    out / f"{city}_T50_extremes_review.geojson", driver="GeoJSON")
                detail["extremes_geojson"] = f"{city}_T50_extremes_review.geojson"
    except Exception as exc:
        detail["export_warning"] = repr(exc)
    return detail


def dem_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "BLOCKED_DEM_NOT_FOUND", "path": str(path)}
    try:
        import rasterio
    except ImportError:
        return {"status": "BLOCKED_RASTERIO_MISSING", "path": str(path)}
    try:
        with rasterio.open(path) as src:
            crs = src.crs
            return {"status": "RASTER_METADATA_READ_NOT_VERTICAL_DATUM_VERIFIED",
                    "path": str(path), "crs": str(crs) if crs is not None else None,
                    "shape": [int(src.height), int(src.width)],
                    "bounds": [float(v) for v in src.bounds],
                    "resolution": [float(v) for v in src.res],
                    "nodata": src.nodata, "units": list(src.units),
                    "tags": src.tags(),
                    "warning": "Horizontal CRS/height units do not establish vertical datum or T50 computation correctness"}
    except Exception as exc:
        return {"status": "BLOCKED_DEM_READ_ERROR", "path": str(path), "reason": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/phase4/_m1g2_gis_audit_v0.1")
    for city in CITIES:
        prefix = city.lower()
        parser.add_argument(f"--{prefix}-geometries", type=Path)
        parser.add_argument(f"--{prefix}-id-column", default="bldg_id")
        parser.add_argument(f"--{prefix}-layer")
        parser.add_argument(f"--{prefix}-dem", type=Path)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    frozen = [ROOT / "outputs/phase4" / city / "m1g_citywide_v0.1" for city in CITIES]
    if any(out == p.resolve() or p.resolve() in out.parents for p in frozen):
        parser.error("Output must not overwrite any frozen city experiment directory")
    out.mkdir(parents=True, exist_ok=True)
    from rhb.pipelines import m1g_citywide as eng
    cfg = eng.read_config()
    result: dict[str, Any] = {
        "audit_version": "0.1.0", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "GIS/source audit of frozen synthetic experiment; no physical flood-risk validation",
        "m1g2_config_sha256": digest(eng.CONFIG_PATH),
        "cities": {},
        "interpretation_warning": "T50 is relative terrain, not DEM elevation; source DEM provenance and vertical datum require manual verification",
    }
    profiles: list[dict] = []
    summaries: list[pd.DataFrame] = []
    correlations: list[pd.DataFrame] = []
    for city in CITIES:
        LOG.info("Reading original eligible universe: %s", city)
        frame, _ = eng.load_city(city, cfg)
        frame["bldg_id"] = normal_ids(frame["bldg_id"])
        base = {"n_eligible": int(len(frame)), "n_phase3": int(cfg["cities"][city]["expected_phase3"])}
        profile = terrain_profile(frame, city)
        profiles.append(profile)
        diag, corr = predictor_diagnostics(frame, city)
        summaries.append(diag)
        correlations.append(corr)
        detail: dict[str, Any] = {**base, "terrain_profile": profile}
        extremes = None
        excluded = None
        if city == "DON":
            extremes = don_extreme_rows(frame)
            extremes.to_csv(out / "DON_T50_extremes_for_GIS_review.csv", index=False)
            detail["extreme_rows_exported"] = int(len(extremes))
            detail["extreme_above_20_abs_m"] = int((extremes["T50"].abs() > 20).sum())
            LOG.info("DON: %d locations with |T50|>20m; %d unique records in review CSV",
                     detail["extreme_above_20_abs_m"], len(extremes))
        if city == "RTM":
            features = (ROOT / cfg["cities"][city]["features_path"]).resolve()
            original = pd.read_parquet(features)
            original["bldg_id"] = normal_ids(original["bldg_id"])
            excluded = missing_ids(original["bldg_id"], frame["bldg_id"])
            full_excluded = original[original["bldg_id"].isin(set(excluded))].copy()
            full_excluded.to_csv(out / "RTM_excluded_from_terrain_population.csv", index=False)
            detail["n_excluded"] = int(len(excluded))
            detail["excluded_export_columns"] = [str(s) for s in full_excluded.columns]
            expected_excluded = base["n_phase3"] - base["n_eligible"]
            if len(excluded) != expected_excluded:
                raise ValueError(f"RTM exclusion set mismatch: found {len(excluded)}, expected {expected_excluded}")
            # Metadata-only peek; avoid interpreting raw terrain missingness without its join logic.
            raw_path = (ROOT / cfg["cities"]["RTM"]["terrain_path"]).resolve()
            try:
                import pyarrow.parquet as pq
                schema = pq.ParquetFile(raw_path).schema_arrow
                detail["original_terrain_table"] = {"path": str(raw_path),
                    "columns": schema.names, "sha256": digest(raw_path)}
            except Exception as exc:
                detail["original_terrain_table_warning"] = repr(exc)
            LOG.info("RTM: %d Phase 3 buildings excluded; IDs exported", len(excluded))
        geompath = getattr(args, f"{city.lower()}_geometries")
        if geompath is not None:
            detail["geometry"] = inspect_geometry(
                geompath.resolve(), frame, city, out,
                id_column=getattr(args, f"{city.lower()}_id_column"),
                layer=getattr(args, f"{city.lower()}_layer"),
                missing=excluded, extremes=extremes)
        else:
            detail["geometry"] = {"status": "PENDING_AUTHORITATIVE_BUILDING_GEOMETRIES"}
        dempath = getattr(args, f"{city.lower()}_dem")
        detail["original_dem"] = dem_metadata(dempath.resolve()) if dempath else {
            "status": "PENDING_ORIGINAL_DEM_OR_SOURCE_METADATA"}
        result["cities"][city] = detail
    pd.DataFrame(profiles).to_csv(out / "terrain_profiles.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(out / "predictor_scales.csv", index=False)
    pd.concat(correlations, ignore_index=True).to_csv(out / "predictor_correlations.csv", index=False)
    result["status"] = "AUTOMATED_TABLES_PASS_GIS_REVIEW_REQUIRED"
    (out / "audit_gis_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    lines = [
        "# Habnetic | M1-G2 geographic source audit", "",
        f"**Run (UTC):** {result['timestamp_utc']}",
        "**Status:** AUTOMATED_TABLES_PASS_GIS_REVIEW_REQUIRED", "",
        "The numerical audit is separate and previously passed. This audit does not validate physical flood hazard or observations.", "",
        "## Derived relative terrain (T50, metres)", "",
        "| City | Eligible | Min | Median | Max | T50<-20 m | T50>20 m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        lines.append(f"| {p['city']} | {p['n']:,} | {p['min_m']:.3f} | {p['median_m']:.3f} | "
                     f"{p['max_m']:.3f} | {p['n_below_minus20_m']} | {p['n_above_plus20_m']} |")
    lines += ["", "## Evidence per city", ""]
    for city, info in result["cities"].items():
        lines += [f"### {city}", "", f"- Eligible buildings: {info['n_eligible']:,}",
                  f"- Building geometry: **{info['geometry']['status']}**",
                  f"- Source DEM: **{info['original_dem']['status']}**"]
        if city == "DON":
            lines += [f"- Review records exported: {info['extreme_rows_exported']}",
                      f"- |T50| > 20m: {info['extreme_above_20_abs_m']}"]
        if city == "RTM":
            lines += [f"- Excluded Phase 3 buildings: {info['n_excluded']}"]
        lines += [""]
    lines += ["## What remains external / manual", "",
              "1. Load the review GeoJSON (if generated) or join review CSV on `bldg_id` to original footprints in QGIS; inspect DON extrema and RTM exclusion clustering.",
              "2. Verify the original raster/terrain source name, acquisition date, resolution, coverage, horizontal CRS, vertical datum and vertical units *from provider documentation*.",
              "3. Verify **T50 formula**, neighborhood scale and nodata handling against the source code that originally generated the terrain parquet; do not equate T50 with absolute elevation.",
              "4. Inspect RTM missingness mechanism and whether the 868 exclusions concentrate spatially in meaningful areas; no blanket extrapolation to excluded assets.",
              "5. Record the source evidence, screenshots and a signed GIS review decision in the paper trail.", "",
              "## Not established", "",
              "No empirical flood-damage validation; no proof the three vertical datums are compatible; no proof that derived T50 values are physically correct.", "",
              "## Next small step", "",
              "Review the CSV outputs and, when authoritative footprints are identified, rerun with `--don-geometries`, `--rtm-geometries`, `--ham-geometries`."]
    (out / "GIS_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Geographic audit tables complete. Report: %s", out / "GIS_AUDIT_REPORT.md")
    LOG.warning("GIS scientific acceptance still requires original geometry/DEM provenance and human review")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
