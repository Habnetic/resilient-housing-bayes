from __future__ import annotations

"""Generate transparent hero PNG maps for Habnetic website v1.

This script reuses the Phase 3 map-loading logic from the one-pager/paper map
scripts, but produces website-specific assets only:

- transparent PNGs
- no titles
- no legends
- no colourbars
- no axes
- consistent styling for Rotterdam, Hamburg and Donostia

Expected run location
---------------------
Run from the repository root that contains::

    outputs/phase3/<CITY>/phase3_features_scaled.parquet
    outputs/phase3/<CITY>/asset_metrics.parquet

The script expects the Habnetic data repository to be a sibling of this project
repository, matching the existing Phase 3 scripts::

    <habnetic-root>/resilient-housing-bayes/...
    <habnetic-root>/data/processed/<CITY>/derived/buildings_<city>.gpkg

Default output
--------------
By default, images are written to::

    ../habnetic.github.io/assets/images/hero/

If that folder does not exist, it is created. You can override it with:

    python phase3_maps_website.py --output-dir path/to/assets/images/hero

Example
-------

    python phase3_maps_website.py
    python phase3_maps_website.py --cities RTM HAM DON
    python phase3_maps_website.py --output-dir ../habnetic.github.io/assets/images/hero

Outputs
-------

    rotterdam.png
    hamburg.png
    donostia.png

"""

from dataclasses import dataclass
from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

try:
    import osmnx as ox
except ImportError:  # Optional dependency. The script still runs without water context.
    ox = None


CITIES = ["RTM", "HAM", "DON"]

BASE_PATH = Path("outputs") / "phase3"
DEFAULT_WEBSITE_OUTPUT_DIR = Path("..") / "habnetic.github.io" / "assets" / "images" / "hero"
CACHE_DIR = BASE_PATH / "cross_city_summary" / "maps_website" / "_cache"

GEOMETRY_PATHS = {
    "RTM": ("derived", "buildings_rtm.gpkg"),
    "HAM": ("derived", "buildings_ham.gpkg"),
    "DON": ("derived", "buildings_don.gpkg"),
}

CITY_NAMES = {
    "RTM": "rotterdam",
    "HAM": "hamburg",
    "DON": "donostia",
}

TOPK_COL = "topk_prob_k1000"
BORDERLINE_COL = "borderline_k1000"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8
MIN_VISIBLE_TOPK_PROB = 0.08

# Website palette. Tuned for dark navy background, not for white paper.
BASE_BUILDINGS = "#7D8791"
BASE_BUILDINGS_ZOOM = "#8B949E"
WATER_FILL = "#2C5676"
WATER_EDGE = "#4B7895"
BOUNDARY = "#FF4B3A"

# Hero bounds, in each city's projected CRS. Copied from phase3_maps_paper.py
# ZOOM_BOUNDS because those are already the visually inspected focus areas.
HERO_BOUNDS = {
    "RTM": (
        96084.3556087867,
        440050.47772143496,
        97259.00481460664,
        440786.6052452254,
    ),
    "HAM": (
        563966.5591559211,
        5931932.797120256,
        568317.873611889,
        5934659.672786485,
    ),
    "DON": (
        580052.0267986986,
        4794535.501698487,
        581438.974184864,
        4795404.671949442,
    ),
}


@dataclass(frozen=True)
class ZoomConfig:
    min_borderline_assets: int = 3
    fallback_top_quantile: float = 0.98
    buffer_fraction: float = 0.35
    min_side_m: float = 2200.0
    max_side_m: float = 5_000.0


ZOOM = ZoomConfig()


def get_project_root() -> Path:
    """Infer project root using the same convention as the Phase 3 scripts."""
    # If this file is stored in scripts/.../... this matches phase3_maps_paper.py.
    # If it is placed directly in the repo root, fall back gracefully.
    here = Path(__file__).resolve()
    if (here.parent / "outputs").exists():
        return here.parent
    try:
        candidate = here.parents[3]
    except IndexError:
        candidate = here.parent
    return candidate


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
    return CACHE_DIR / f"{city.upper()}_osm_water.gpkg"


def load_city_geodata(city: str) -> gpd.GeoDataFrame:
    city = city.upper()

    geom_path = get_geometry_path(city)
    features_path = get_features_path(city)
    asset_metrics_path = get_asset_metrics_path(city)

    for path in [geom_path, features_path, asset_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print(f"[website maps] loading geometry: {geom_path}")
    gdf = gpd.read_file(geom_path)

    print(f"[website maps] loading features: {features_path}")
    features = pd.read_parquet(features_path)

    print(f"[website maps] loading asset metrics: {asset_metrics_path}")
    metrics = pd.read_parquet(asset_metrics_path)

    for name, df in {"geometry": gdf, "features": features, "metrics": metrics}.items():
        if "bldg_id" not in df.columns:
            raise KeyError(f"{name} missing bldg_id")

    topk_col = TOPK_COL
    if TOPK_COL not in metrics.columns:
        candidates = sorted(c for c in metrics.columns if c.startswith("topk_prob_k"))
        if not candidates:
            raise KeyError(f"No top-k probability columns found in {asset_metrics_path}")
        topk_col = candidates[0]
        print(f"[website maps] warning: {TOPK_COL} missing for {city}; using {topk_col}")

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
        (df[TOPK_COL] > BORDERLINE_LOW) & (df[TOPK_COL] < BORDERLINE_HIGH)
    ).astype(int)

    gdf = gdf.merge(df, on="bldg_id", how="inner", validate="one_to_one")
    if gdf.empty:
        raise ValueError(f"Join produced empty GeoDataFrame for {city}")

    print(f"[website maps] joined rows for {city}: {len(gdf):,}")
    return gdf


def fetch_osm_water_for_city(city: str, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame | None:
    if ox is None:
        print("[website maps] osmnx not installed; skipping OSM water")
        return None

    city = city.upper()
    cache_path = get_osm_water_cache_path(city)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[website maps] loading cached OSM water: {cache_path}")
        water = gpd.read_file(cache_path)
        if water.crs != gdf.crs:
            water = water.to_crs(gdf.crs)
        return water

    print(f"[website maps] fetching OSM water for {city}")
    gdf_wgs = gdf.to_crs(4326)
    xmin, ymin, xmax, ymax = gdf_wgs.total_bounds
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04

    tags = {
        "natural": ["water", "bay", "strait", "coastline"],
        "waterway": ["river", "canal", "stream", "dock"],
        "landuse": ["reservoir"],
        "water": True,
    }

    try:
        water = ox.features_from_bbox(
            north=ymax + pad_y,
            south=ymin - pad_y,
            east=xmax + pad_x,
            west=xmin - pad_x,
            tags=tags,
        )
    except TypeError:
        water = ox.geometries_from_bbox(
            north=ymax + pad_y,
            south=ymin - pad_y,
            east=xmax + pad_x,
            west=xmin - pad_x,
            tags=tags,
        )

    if water.empty:
        print(f"[website maps] warning: OSM water empty for {city}")
        return None

    water = water.reset_index()
    water = gpd.GeoDataFrame(water[["geometry"]], geometry="geometry", crs=4326)
    water = water[~water.geometry.is_empty & water.geometry.notna()].copy()
    water = water.to_crs(gdf.crs)

    clip_poly = gdf.buffer(0).unary_union.envelope
    water = gpd.clip(water, clip_poly)

    if water.empty:
        print(f"[website maps] warning: clipped OSM water empty for {city}")
        return None

    water.to_file(cache_path, driver="GPKG")
    print(f"[website maps] cached OSM water: {cache_path}")
    return water


def split_water(
    water: gpd.GeoDataFrame | None,
) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    if water is None or water.empty:
        return None, None
    polygons = water[water.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    lines = water[water.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    return polygons if not polygons.empty else None, lines if not lines.empty else None


def plot_water(water: gpd.GeoDataFrame | None, ax: plt.Axes) -> None:
    polygons, lines = split_water(water)
    if polygons is not None:
        polygons.plot(
            ax=ax,
            color=WATER_FILL,
            edgecolor="none",
            linewidth=0,
            alpha=0.70,
            zorder=0,
        )
    if lines is not None:
        lines.plot(
            ax=ax,
            color=WATER_EDGE,
            linewidth=0.45,
            alpha=0.70,
            zorder=0,
        )


def bounds_with_padding(
    bounds: tuple[float, float, float, float],
    pad_fraction: float = 0.08,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    dx = xmax - xmin
    dy = ymax - ymin
    return (
        xmin - dx * pad_fraction,
        ymin - dy * pad_fraction,
        xmax + dx * pad_fraction,
        ymax + dy * pad_fraction,
    )


def set_bounds(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def make_square_bounds(
    bounds: tuple[float, float, float, float],
    config: ZoomConfig = ZOOM,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = bounds
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side = max(xmax - xmin, ymax - ymin)
    side = max(side * (1 + config.buffer_fraction), config.min_side_m)
    side = min(side, config.max_side_m)
    return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)


def get_auto_hero_bounds(gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    borderline = gdf[gdf[BORDERLINE_COL] == 1].copy()
    if len(borderline) >= ZOOM.min_borderline_assets:
        print(f"[website maps] auto bounds based on borderline assets: {len(borderline):,}")
        return make_square_bounds(tuple(borderline.total_bounds))

    threshold = float(gdf[TOPK_COL].quantile(ZOOM.fallback_top_quantile))
    focus = gdf[gdf[TOPK_COL] >= threshold].copy()
    print(
        "[website maps] auto bounds fallback based on high top-k probability "
        f"q={ZOOM.fallback_top_quantile}: {len(focus):,} assets"
    )
    if focus.empty:
        focus = gdf.nlargest(min(50, len(gdf)), TOPK_COL)
    return make_square_bounds(tuple(focus.total_bounds))


def get_hero_bounds(city: str, gdf: gpd.GeoDataFrame, pad_fraction: float = 0.30) -> tuple[float, float, float, float]:
    city = city.upper()
    manual = HERO_BOUNDS.get(city)
    if manual is None:
        print(f"[website maps] no manual hero bounds for {city}; using automatic fallback")
        manual = get_auto_hero_bounds(gdf)
    else:
        print(f"[website maps] using manual hero bounds for {city}: {manual}")
    return bounds_with_padding(manual, pad_fraction=pad_fraction)


def clip_to_bounds(
    gdf: gpd.GeoDataFrame | None,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return None
    xmin, ymin, xmax, ymax = bounds
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


def add_probability_layer(gdf: gpd.GeoDataFrame, ax: plt.Axes) -> None:
    signal = gdf[gdf[TOPK_COL] > MIN_VISIBLE_TOPK_PROB].copy()
    if signal.empty:
        return
    signal.plot(
        column=TOPK_COL,
        ax=ax,
        cmap="inferno",
        norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=1.0),
        linewidth=0.0,
        edgecolor="none",
        antialiased=False,
        alpha=0.98,
        legend=False,
        zorder=2,
    )


def add_borderline_layer(gdf: gpd.GeoDataFrame, ax: plt.Axes) -> None:
    borderline = gdf[gdf[BORDERLINE_COL] == 1]
    if borderline.empty:
        return
    borderline.boundary.plot(
        ax=ax,
        color=BOUNDARY,
        linewidth=0.55,
        alpha=0.95,
        zorder=3,
    )


def plot_hero_map(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
    output_dir: Path,
    dpi: int = 240,
) -> Path:
    """Plot one transparent hero PNG for the website."""
    city = city.upper()
    bounds = get_hero_bounds(city, gdf)
    hero_gdf = clip_to_bounds(gdf, bounds)
    hero_water = clip_to_bounds(water, bounds)

    if hero_gdf is None or hero_gdf.empty:
        raise ValueError(f"Hero bounds produced no building geometries for {city}: {bounds}")

    # 12 x 8 at 240 dpi -> roughly 2880 x 1920 before tight cropping.
    # The website will scale down. Let the browser do the boring part.
    fig, ax = plt.subplots(figsize=(12.0, 8.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    plot_water(hero_water, ax)

    hero_gdf.plot(
        ax=ax,
        color=BASE_BUILDINGS_ZOOM,
        linewidth=0,
        alpha=0.58,
        zorder=1,
    )
    add_probability_layer(hero_gdf, ax)
    add_borderline_layer(hero_gdf, ax)

    set_bounds(ax, bounds)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{CITY_NAMES[city]}.png"
    path = output_dir / filename
    fig.savefig(
        path,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.20,
    )
    plt.close(fig)

    print(f"[website maps] saved: {path}")
    return path


def write_readme(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "README.md"
    path.write_text(
        "# Habnetic website hero maps\n\n"
        "Generated by `phase3_maps_website.py`.\n\n"
        "These are transparent PNG assets for the Habnetic website v1 hero section.\n"
        "They intentionally contain no title, legend, axes, colourbar, labels, or paper-specific annotations.\n\n"
        "Outputs:\n\n"
        "- `rotterdam.png`\n"
        "- `hamburg.png`\n"
        "- `donostia.png`\n\n"
        "The maps are visual website assets and spatial diagnostics, not hydraulic validation.\n",
        encoding="utf-8",
    )
    print(f"[website maps] saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transparent PNG hero maps for Habnetic website v1."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=CITIES,
        choices=CITIES,
        help="Cities to render. Default: RTM HAM DON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WEBSITE_OUTPUT_DIR,
        help=(
            "Output directory for PNGs. Default: "
            "../habnetic.github.io/assets/images/hero"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG export DPI. Default: 240.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not write README.md into the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    print(f"[website maps] project_root={get_project_root()}")
    print(f"[website maps] data_root={get_data_root()}")
    print(f"[website maps] output_dir={output_dir}")

    for city in args.cities:
        print(f"\n[website maps] city={city}")
        gdf = load_city_geodata(city)
        water = fetch_osm_water_for_city(city, gdf)
        plot_hero_map(gdf=gdf, water=water, city=city, output_dir=output_dir, dpi=args.dpi)

    if not args.no_readme:
        write_readme(output_dir)

    print("[website maps] done")


if __name__ == "__main__":
    main()
