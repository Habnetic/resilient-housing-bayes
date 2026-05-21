from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

try:
    import osmnx as ox
except ImportError:
    ox = None


CITIES = ["RTM", "HAM", "DON"]

BASE_PATH = Path("outputs") / "phase3"
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "maps"

GEOMETRY_PATHS = {
    "RTM": ("derived", "buildings_rtm.gpkg"),
    "HAM": ("derived", "buildings_ham.gpkg"),
    "DON": ("derived", "buildings_don.gpkg"),
}

TOPK_COL = "topk_prob_k1000"
BORDERLINE_COL = "borderline_k1000"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8

# Habnetic palette
NAVY = "#0B132B"
TEAL = "#09303A"
CORAL = "#FF4B3A"
WHITE = "#FFFFFF"

OTHER_BUILDINGS = "#A3A3A3"
WATER_FILL = "#D0EDF2"
WATER_EDGE = "#9FD7DF"

MIN_VISIBLE_TOPK_PROB = 0.05


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    project_root = get_project_root()
    habnetic_root = project_root.parent
    return habnetic_root / "data" / "processed"


def get_geometry_path(city: str) -> Path:
    city = city.upper()
    subdir, filename = GEOMETRY_PATHS[city]
    return get_data_root() / city / subdir / filename


def get_features_path(city: str) -> Path:
    return BASE_PATH / city.upper() / "phase3_features_scaled.parquet"


def get_asset_metrics_path(city: str) -> Path:
    return BASE_PATH / city.upper() / "asset_metrics.parquet"


def get_osm_water_cache_path(city: str) -> Path:
    return OUTPUT_DIR / "_cache" / f"{city.upper()}_osm_water.gpkg"


def load_city_geodata(city: str) -> gpd.GeoDataFrame:
    city = city.upper()

    geom_path = get_geometry_path(city)
    features_path = get_features_path(city)
    asset_metrics_path = get_asset_metrics_path(city)

    if not geom_path.exists():
        raise FileNotFoundError(f"Missing geometry file: {geom_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not asset_metrics_path.exists():
        raise FileNotFoundError(f"Missing asset metrics file: {asset_metrics_path}")

    print(f"[maps] loading geometry: {geom_path}")
    gdf = gpd.read_file(geom_path)

    print(f"[maps] loading features: {features_path}")
    features = pd.read_parquet(features_path)

    print(f"[maps] loading asset metrics: {asset_metrics_path}")
    metrics = pd.read_parquet(asset_metrics_path)

    for name, df in {"geometry": gdf, "features": features, "metrics": metrics}.items():
        if "bldg_id" not in df.columns:
            raise KeyError(f"{name} missing bldg_id")

    if TOPK_COL not in metrics.columns:
        topk_candidates = [c for c in metrics.columns if c.startswith("topk_prob_k")]
        if not topk_candidates:
            raise KeyError(f"No top-k probability columns found in {asset_metrics_path}")
        topk_col = sorted(topk_candidates)[0]
        print(f"[maps] warning: {TOPK_COL} missing for {city}; using {topk_col}")
    else:
        topk_col = TOPK_COL

    feature_cols = [
        "bldg_id",
        "E_hat_v0",
        "H_pluvial_v1_mm",
        "H_pluvial_v1_logrel",
        "Y_damage",
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]

    metrics_small = metrics[["bldg_id", topk_col]].copy()
    if topk_col != TOPK_COL:
        metrics_small = metrics_small.rename(columns={topk_col: TOPK_COL})

    df = features[feature_cols].merge(metrics_small, on="bldg_id", how="inner")

    df[BORDERLINE_COL] = (
        (df[TOPK_COL] > BORDERLINE_LOW)
        & (df[TOPK_COL] < BORDERLINE_HIGH)
    ).astype(int)

    gdf = gdf.merge(df, on="bldg_id", how="inner", validate="one_to_one")

    if gdf.empty:
        raise ValueError(f"Join produced empty GeoDataFrame for {city}")

    print(f"[maps] joined rows for {city}: {len(gdf):,}")
    return gdf


def fetch_osm_water_for_city(city: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    if ox is None:
        print("[maps] osmnx not installed; skipping OSM water")
        return None

    city = city.upper()
    cache_path = get_osm_water_cache_path(city)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[maps] loading cached OSM water: {cache_path}")
        water = gpd.read_file(cache_path)
        if water.crs != gdf.crs:
            water = water.to_crs(gdf.crs)
        return water

    print(f"[maps] fetching OSM water for {city}")

    # OSM works in WGS84.
    gdf_wgs = gdf.to_crs(4326)
    xmin, ymin, xmax, ymax = gdf_wgs.total_bounds

    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    west = xmin - pad_x
    south = ymin - pad_y
    east = xmax + pad_x
    north = ymax + pad_y

    tags = {
        "natural": ["water", "bay", "strait", "coastline"],
        "waterway": ["river", "canal", "stream", "dock"],
        "landuse": ["reservoir"],
        "water": True,
    }

    try:
        # Newer osmnx
        water = ox.features_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            tags=tags,
        )
    except TypeError:
        # Older osmnx
        water = ox.geometries_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            tags=tags,
        )

    if water.empty:
        print(f"[maps] warning: OSM water empty for {city}")
        return None

    water = water.reset_index()

    # Keep only geometry. We do not want roads, labels, cafés, or humanity’s clutter.
    water = gpd.GeoDataFrame(water[["geometry"]], geometry="geometry", crs=4326)
    water = water[~water.geometry.is_empty & water.geometry.notna()].copy()
    water = water.to_crs(gdf.crs)

    # Clip to a padded building extent for cleaner output.
    clip_poly = gdf.buffer(0).unary_union.envelope
    water = gpd.clip(water, clip_poly)

    if water.empty:
        print(f"[maps] warning: clipped OSM water empty for {city}")
        return None

    water.to_file(cache_path, driver="GPKG")
    print(f"[maps] cached OSM water: {cache_path}")

    return water


def set_tight_bounds(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    pad_fraction: float = 0.03,
) -> None:
    xmin, ymin, xmax, ymax = gdf.total_bounds

    pad_x = (xmax - xmin) * pad_fraction
    pad_y = (ymax - ymin) * pad_fraction

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(
        png_path,
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor=WHITE,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.015,
        facecolor=WHITE,
    )

    print(f"[maps] saved: {png_path}")
    print(f"[maps] saved: {pdf_path}")

    return [png_path, pdf_path]


def plot_water(water: gpd.GeoDataFrame | None, ax: plt.Axes) -> None:
    if water is None or water.empty:
        return

    polygons = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    lines = water[water.geometry.geom_type.isin(["LineString", "MultiLineString"])]

    if not polygons.empty:
        polygons.plot(
            ax=ax,
            color=WATER_FILL,
            edgecolor="none",
            linewidth=0,
            alpha=0.42,
            zorder=0,
        )

    if not lines.empty:
        lines.plot(
            ax=ax,
            color=WATER_EDGE,
            linewidth=0.22,
            alpha=0.55,
            zorder=0,
        )


def plot_topk_onepager_map(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
    output_dir: Path,
) -> list[Path]:
    city = city.upper()

    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    plot_water(water, ax)

    gdf.plot(
        ax=ax,
        color=OTHER_BUILDINGS,
        linewidth=0,
        alpha=0.93,
        zorder=1,
    )

    signal = gdf[gdf[TOPK_COL] > MIN_VISIBLE_TOPK_PROB].copy()

    if signal.empty:
        print(f"[maps] warning: no visible top-k signal for {city}")
    else:
        signal.plot(
            column=TOPK_COL,
            ax=ax,
            cmap="inferno",
            norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=0.92),
            linewidth=0.03,
            edgecolor="none",
            antialiased=False,
            alpha=0.98,
            legend=False,
            zorder=2,
        )

    borderline = gdf[gdf[BORDERLINE_COL] == 1]
    if not borderline.empty:
        borderline.boundary.plot(
            ax=ax,
            color=CORAL,
            linewidth=0.30,
            alpha=0.85,
            zorder=3,
        )

    set_tight_bounds(ax, gdf)

    ax.set_title(
        f"CITYWIDE OVERVIEW — {city}",
        fontsize=10,
        fontweight="semibold",
        color=TEAL,
        loc="left",
        pad=3,
    )
    ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=0.92),
    )
    sm._A = []

    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.022,
        pad=0.01,
    )

    cbar.set_label(
        "Posterior top-k probability",
        fontsize=11,
    )

    cbar.set_ticks([0.1, 0.5, 1.0])
    cbar.set_ticklabels(["0.1", "0.5", "1.0"])
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(left=0.005, right=0.94, top=0.94, bottom=0.005)

    return save_figure(fig, output_dir, f"{city}_topk_onepager_map")


def plot_uncertain_boundary_map(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
    output_dir: Path,
) -> list[Path]:
    city = city.upper()

    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    plot_water(water, ax)

    base = gdf[gdf[BORDERLINE_COL] == 0]
    borderline = gdf[gdf[BORDERLINE_COL] == 1]

    base.plot(
        ax=ax,
        color=OTHER_BUILDINGS,
        linewidth=0,
        alpha=0.93,
        zorder=1,
    )

    if not borderline.empty:
        borderline.plot(
            ax=ax,
            color=CORAL,
            linewidth=0,
            alpha=0.95,
            zorder=2,
        )

    share = len(borderline) / len(gdf) if len(gdf) else 0.0

    set_tight_bounds(ax, gdf)

    ax.set_title(
        f"{city} — Uncertain decision boundary "
        f"({BORDERLINE_LOW} < P(top-k) < {BORDERLINE_HIGH})",
        fontsize=10,
        color=TEAL,
        loc="left",
        pad=3,
    )

    ax.text(
        0.01,
        0.01,
        f"Boundary share: {share:.3%}",
        transform=ax.transAxes,
        fontsize=8,
        color=NAVY,
        bbox={"facecolor": WHITE, "alpha": 0.85, "edgecolor": "none"},
    )

    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005)

    return save_figure(fig, output_dir, f"{city}_uncertain_boundary_map")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[maps] data_root={get_data_root()}")
    print(f"[maps] output_dir={OUTPUT_DIR}")

    for city in CITIES:
        print(f"\n[maps] city={city}")

        gdf = load_city_geodata(city)
        water = fetch_osm_water_for_city(city, gdf)

        plot_topk_onepager_map(
            gdf=gdf,
            water=water,
            city=city,
            output_dir=OUTPUT_DIR,
        )

        plot_uncertain_boundary_map(
            gdf=gdf,
            water=water,
            city=city,
            output_dir=OUTPUT_DIR,
        )

    print("[maps] done")


if __name__ == "__main__":
    main()