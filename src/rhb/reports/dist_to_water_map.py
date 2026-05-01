from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]

BASE_PATH = Path("outputs") / "phase3"
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "maps_dist_to_water"

GEOMETRY_PATHS = {
    "RTM": ("derived", "buildings_rtm.gpkg"),
    "HAM": ("derived", "buildings_ham.gpkg"),
    "DON": ("derived", "buildings_don.gpkg"),
}

DEFAULT_PRIORS_FILE = "building_water_proximity.parquet"
DON_V2_PRIORS_FILE = "building_water_proximity_v2_coast.parquet"


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


def get_priors_path(city: str) -> Path:
    city = city.upper()

    if city == "DON":
        return get_data_root() / city / "priors" / "building_water_proximity_v2_coast.parquet"

    return get_data_root() / city / "priors" / "building_water_proximity_v3.parquet"


def get_output_stem(city: str) -> str:
    city = city.upper()
    if city == "DON":
        return "DON_dist_to_water_v2_coast"
    return f"{city}_dist_to_water_v3"


def load_data(city: str) -> gpd.GeoDataFrame:
    city = city.upper()

    geom_path = get_geometry_path(city)
    priors_path = get_priors_path(city)

    print(f"[DEBUG] USING PRIORS FILE: {priors_path}")

    if not geom_path.exists():
        raise FileNotFoundError(f"Missing geometry: {geom_path}")
    if not priors_path.exists():
        raise FileNotFoundError(f"Missing priors: {priors_path}")

    print(f"[load] {city} geometry: {geom_path}")
    print(f"[load] {city} priors: {priors_path}")

    gdf = gpd.read_file(geom_path)
    df = pd.read_parquet(priors_path)

    if "bldg_id" not in gdf.columns:
        raise KeyError(f"{city} geometry missing bldg_id")
    if "bldg_id" not in df.columns:
        raise KeyError(f"{city} priors missing bldg_id")
    if "dist_to_water_m" not in df.columns:
        raise KeyError(f"{city} priors missing dist_to_water_m")

    df = df[["bldg_id", "dist_to_water_m"]].copy()
    gdf = gdf.merge(df, on="bldg_id", how="inner", validate="one_to_one")

    if gdf.empty:
        raise ValueError(f"{city} join produced empty GeoDataFrame")

    print(f"[load] {city} joined rows: {len(gdf):,}")
    return gdf


def get_bounds(values: pd.Series, q_low: float = 0.02, q_high: float = 0.98) -> tuple[float, float]:
    values = values.dropna()

    if values.empty:
        raise ValueError("No valid values for plotting")

    vmin = values.quantile(q_low)
    vmax = values.quantile(q_high)

    if vmin == vmax:
        vmin = values.min()
        vmax = values.max()

    return float(vmin), float(vmax)


def plot_map(gdf: gpd.GeoDataFrame, city: str) -> None:
    city = city.upper()
    col = "dist_to_water_m"

    values = gdf[col].dropna()
    vmin, vmax = get_bounds(values)

    fig, ax = plt.subplots(figsize=(9, 8))

    gdf.plot(
        column=col,
        ax=ax,
        cmap="viridis",  # dark = near water, bright = far
        linewidth=0,
        markersize=2.2,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={
            "shrink": 0.45,
            "label": "Distance to water [m]",
        },
    )

    title = (
        "DON — Distance to water (v2 local coast)"
        if city == "DON"
        else f"{city} — Distance to water (v3 baseline)"
    )

    ax.set_title(title, fontsize=11)
    ax.axis("off")
    ax.set_aspect("equal")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = get_output_stem(city)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {png_path}")
    print(f"[saved] {pdf_path}")


def main() -> None:
    print(f"[maps] data root: {get_data_root()}")
    print(f"[maps] output dir: {OUTPUT_DIR}")

    for city in CITIES:
        print(f"\n[city] {city}")
        gdf = load_data(city)
        plot_map(gdf, city)


if __name__ == "__main__":
    main()