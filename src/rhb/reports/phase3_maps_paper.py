from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "maps_paper"

GEOMETRY_PATHS = {
    "RTM": ("derived", "buildings_rtm.gpkg"),
    "HAM": ("derived", "buildings_ham.gpkg"),
    "DON": ("derived", "buildings_don.gpkg"),
}

TOPK_COL = "topk_prob_k1000"
BORDERLINE_COL = "borderline_k1000"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8
MIN_VISIBLE_TOPK_PROB = 0.08

# Paper palette: restrained enough for print, not dead enough for LaTeX.
BLACK = "#111111"
BASE_BUILDINGS = "#B8B8B8"
WATER_FILL = "#D7EEF2"
WATER_EDGE = "#9FD7DF"
BOUNDARY = "#D6453D"
WHITE = "#FFFFFF"

# Optional manual zoom bounds, in each city's projected CRS.
# Fill these after inspecting candidate outputs in QGIS / generated citywide maps:
#     "RTM": (xmin, ymin, xmax, ymax)
# Leave as None to use the automatic fallback.
ZOOM_BOUNDS = {
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

CITYWIDE_BOUNDS = {
    "RTM": None,

    "HAM": (
        556752.1893756775,
        5929209.802277068,
        577244.5602281991,
        5942051.932623321,
    ),

    "DON": (
        579370.3635517282,
        4793876.048668751,
        586668.9002346192,
        4798449.885444063,
    ),
}

@dataclass(frozen=True)
class ZoomConfig:
    min_borderline_assets: int = 3
    fallback_top_quantile: float = 0.98
    buffer_fraction: float = 0.30
    min_side_m: float = 2200.0
    max_side_m: float = 5_000.0


ZOOM = ZoomConfig()


def get_citywide_bounds(
    city: str,
    gdf: gpd.GeoDataFrame,
) -> tuple[float, float, float, float]:

    city = city.upper()

    manual = CITYWIDE_BOUNDS.get(city)

    if manual is not None:
        print(f"[maps] using manual citywide bounds for {city}: {manual}")
        return manual

    return bounds_with_padding(gdf)

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

    for path in [geom_path, features_path, asset_metrics_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    print(f"[maps] loading geometry: {geom_path}")
    gdf = gpd.read_file(geom_path)

    print(f"[maps] loading features: {features_path}")
    features = pd.read_parquet(features_path)

    print(f"[maps] loading asset metrics: {asset_metrics_path}")
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
        print(f"[maps] warning: {TOPK_COL} missing for {city}; using {topk_col}")

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
        print(f"[maps] warning: OSM water empty for {city}")
        return None

    water = water.reset_index()
    water = gpd.GeoDataFrame(water[["geometry"]], geometry="geometry", crs=4326)
    water = water[~water.geometry.is_empty & water.geometry.notna()].copy()
    water = water.to_crs(gdf.crs)

    clip_poly = gdf.buffer(0).unary_union.envelope
    water = gpd.clip(water, clip_poly)

    if water.empty:
        print(f"[maps] warning: clipped OSM water empty for {city}")
        return None

    water.to_file(cache_path, driver="GPKG")
    print(f"[maps] cached OSM water: {cache_path}")
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
            alpha=0.55,
            zorder=0,
        )
    if lines is not None:
        lines.plot(
            ax=ax,
            color=WATER_EDGE,
            linewidth=0.24,
            alpha=0.55,
            zorder=0,
        )


def bounds_with_padding(
    gdf: gpd.GeoDataFrame,
    pad_fraction: float = 0.035,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = gdf.total_bounds
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


def get_auto_zoom_bounds(
    gdf: gpd.GeoDataFrame,
    config: ZoomConfig = ZOOM,
) -> tuple[float, float, float, float]:
    """Choose a fallback zoom if manual bounds are missing.

    Preference order:
    1. borderline assets, because that is the actual decision-stability object;
    2. high top-k assets, because some cities have almost no borderline assets;
    3. highest-probability assets as a final sanity fallback.
    """
    borderline = gdf[gdf[BORDERLINE_COL] == 1].copy()

    if len(borderline) >= config.min_borderline_assets:
        print(f"[maps] auto zoom based on borderline assets: {len(borderline):,}")
        return make_square_bounds(borderline.total_bounds, config)

    threshold = float(gdf[TOPK_COL].quantile(config.fallback_top_quantile))
    focus = gdf[gdf[TOPK_COL] >= threshold].copy()
    print(
        "[maps] auto zoom fallback based on high top-k probability "
        f"q={config.fallback_top_quantile}: {len(focus):,} assets"
    )

    if focus.empty:
        focus = gdf.nlargest(min(50, len(gdf)), TOPK_COL)

    return make_square_bounds(focus.total_bounds, config)


def get_zoom_bounds(city: str, gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    city = city.upper()
    manual = ZOOM_BOUNDS.get(city)
    if manual is not None:
        print(f"[maps] using manual zoom bounds for {city}: {manual}")
        return manual
    print(f"[maps] no manual zoom bounds for {city}; using automatic fallback")
    return get_auto_zoom_bounds(gdf)


def clip_to_bounds(
    gdf: gpd.GeoDataFrame | None,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return None
    xmin, ymin, xmax, ymax = bounds
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


def add_probability_layer(gdf: gpd.GeoDataFrame, ax: plt.Axes, zoom: bool = False) -> None:
    signal = gdf[gdf[TOPK_COL] > MIN_VISIBLE_TOPK_PROB].copy()
    if signal.empty:
        return
    signal.plot(
        column=TOPK_COL,
        ax=ax,
        cmap="inferno",
        norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=1.0),
        linewidth=0.02 if not zoom else 0.08,
        edgecolor="none",
        antialiased=False,
        alpha=0.98,
        legend=False,
        zorder=2,
    )


def add_borderline_layer(gdf: gpd.GeoDataFrame, ax: plt.Axes, zoom: bool = False) -> None:
    borderline = gdf[gdf[BORDERLINE_COL] == 1]
    if borderline.empty:
        return
    borderline.boundary.plot(
        ax=ax,
        color=BOUNDARY,
        linewidth=0.25 if not zoom else 0.75,
        alpha=0.9 if not zoom else 0.98,
        zorder=3,
    )


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["png", "pdf"]:
        path = OUTPUT_DIR / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.015, "facecolor": WHITE}
        if ext == "png":
            kwargs["dpi"] = 450
        fig.savefig(path, **kwargs)
        print(f"[maps] saved: {path}")
        paths.append(path)
    return paths


def plot_citywide_topk_map(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
) -> list[Path]:
    city = city.upper()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    plot_water(water, ax)
    gdf.plot(ax=ax, color=BASE_BUILDINGS, linewidth=0, alpha=0.65, zorder=1)
    add_probability_layer(gdf, ax, zoom=False)
    add_borderline_layer(gdf, ax, zoom=False)

    set_bounds(ax, get_citywide_bounds(city, gdf))
    ax.set_title(f"{city} citywide posterior top-k probability", fontsize=9, loc="left", pad=3)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=1.0),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, fraction=0.026, pad=0.01)
    cbar.set_label("P(top-k)", fontsize=8)
    cbar.set_ticks([0.1, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.005, right=0.93, top=0.94, bottom=0.005)
    return save_figure(fig, f"{city}_paper_citywide_topk_map")


def plot_boundary_zoom_map(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
) -> list[Path]:
    city = city.upper()
    zoom_bounds = get_zoom_bounds(city, gdf)
    zoom_gdf = clip_to_bounds(gdf, zoom_bounds)
    zoom_water = clip_to_bounds(water, zoom_bounds)

    if zoom_gdf is None or zoom_gdf.empty:
        raise ValueError(f"Zoom bounds produced no building geometries for {city}: {zoom_bounds}")

    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=WHITE)
    ax.set_facecolor(WHITE)

    plot_water(zoom_water, ax)
    zoom_gdf.plot(ax=ax, color=BASE_BUILDINGS, linewidth=0, alpha=0.72, zorder=1)
    add_probability_layer(zoom_gdf, ax, zoom=True)
    add_borderline_layer(zoom_gdf, ax, zoom=True)

    set_bounds(ax, zoom_bounds)

    boundary_count = int((zoom_gdf[BORDERLINE_COL] == 1).sum())
    top_count = int((zoom_gdf[TOPK_COL] > MIN_VISIBLE_TOPK_PROB).sum())

    ax.set_title(
        f"{city} local transition structure",
        fontsize=8.5,
        loc="left",
        pad=3,
    )
    ax.axis("off")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005)
    return save_figure(fig, f"{city}_paper_boundary_zoom_map")


def plot_citywide_plus_zoom(
    gdf: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame | None,
    city: str,
) -> list[Path]:
    city = city.upper()
    zoom_bounds = get_zoom_bounds(city, gdf)
    zoom_gdf = clip_to_bounds(gdf, zoom_bounds)
    zoom_water = clip_to_bounds(water, zoom_bounds)

    if zoom_gdf is None or zoom_gdf.empty:
        raise ValueError(f"Zoom bounds produced no building geometries for {city}: {zoom_bounds}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.4),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.02},
        facecolor=WHITE,
    )

    for ax in axes:
        ax.set_facecolor(WHITE)
        ax.axis("off")

    ax0, ax1 = axes

    plot_water(water, ax0)
    gdf.plot(ax=ax0, color=BASE_BUILDINGS, linewidth=0, alpha=0.65, zorder=1)
    add_probability_layer(gdf, ax0, zoom=False)
    add_borderline_layer(gdf, ax0, zoom=False)
    set_bounds(ax0, bounds_with_padding(gdf))
    ax0.set_title(f"{city}: citywide", fontsize=8.5, loc="left", pad=2)

    xmin, ymin, xmax, ymax = zoom_bounds
    ax0.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        color=BLACK,
        linewidth=0.6,
        alpha=0.85,
        zorder=4,
    )

    plot_water(zoom_water, ax1)
    zoom_gdf.plot(ax=ax1, color=BASE_BUILDINGS, linewidth=0, alpha=0.72, zorder=1)
    add_probability_layer(zoom_gdf, ax1, zoom=True)
    add_borderline_layer(zoom_gdf, ax1, zoom=True)
    set_bounds(ax1, zoom_bounds)
    ax1.set_title("local transition structure", fontsize=8.5, loc="left", pad=2)

    sm = plt.cm.ScalarMappable(
        cmap="inferno",
        norm=Normalize(vmin=MIN_VISIBLE_TOPK_PROB, vmax=1.0),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.022, pad=0.012)
    cbar.set_label("P(top-k)", fontsize=8)
    cbar.set_ticks([0.1, 0.5, 1.0])
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"Posterior top-k probability and uncertain boundary — {city}", fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.004, right=0.93, top=0.88, bottom=0.004)
    return save_figure(fig, f"{city}_paper_citywide_plus_boundary_zoom")


def write_readme() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    readme = f"""# Phase 3 paper maps

Generated by `phase3_maps_paper.py`.

Main outputs per city:

- `*_paper_citywide_topk_map.pdf`: citywide posterior top-k probability map.
- `*_paper_boundary_zoom_map.pdf`: zoom around the selected boundary / high-priority diagnostic area.
- `*_paper_citywide_plus_boundary_zoom.pdf`: combined citywide + zoom figure for paper/demo use.

Manual zoom bounds can be provided in `ZOOM_BOUNDS` as `(xmin, ymin, xmax, ymax)` in each city's projected CRS.
If a city has `None` zoom bounds, the script uses an automatic fallback based first on borderline assets and then on high posterior top-k probability assets.

These maps are spatial diagnostics, not hydraulic validation.
"""
    path = OUTPUT_DIR / "README.md"
    path.write_text(readme, encoding="utf-8")
    print(f"[maps] saved: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[maps] data_root={get_data_root()}")
    print(f"[maps] output_dir={OUTPUT_DIR}")

    for city in CITIES:
        print(f"\n[maps] city={city}")
        gdf = load_city_geodata(city)
        water = fetch_osm_water_for_city(city, gdf)

        plot_citywide_topk_map(gdf, water, city)
        plot_boundary_zoom_map(gdf, water, city)
        plot_citywide_plus_zoom(gdf, water, city)

    write_readme()
    print("[maps] done")


if __name__ == "__main__":
    main()
