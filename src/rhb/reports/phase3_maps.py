from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]

BASE_PATH = Path("outputs") / "phase3"
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "maps"

GEOMETRY_PATHS = {
    "RTM": ("derived", "buildings_rtm.gpkg"),
    "HAM": ("derived", "buildings_ham.gpkg"),
    "DON": ("derived", "buildings_don.gpkg"),
}

TOPK_COL = "topk_prob_k1000"

CONTINUOUS_MAPS = {
    "E_hat_v0": "Exposure proxy",
    "H_pluvial_v1_mm": "Pluvial hazard proxy [mm]",
    TOPK_COL: "P(top-k), k=1000",
}

BORDERLINE_COL = "borderline_k1000"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8


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
    print(f"[maps] loading features: {features_path}")
    print(f"[maps] loading asset metrics: {asset_metrics_path}")

    gdf = gpd.read_file(geom_path)
    features = pd.read_parquet(features_path)
    metrics = pd.read_parquet(asset_metrics_path)

    for name, df in {
        "geometry": gdf,
        "features": features,
        "metrics": metrics,
    }.items():
        if "bldg_id" not in df.columns:
            raise KeyError(f"{name} missing bldg_id")

    if TOPK_COL not in metrics.columns:
        topk_candidates = [c for c in metrics.columns if c.startswith("topk_prob_k")]
        if not topk_candidates:
            raise KeyError(
                f"No top-k probability columns found in {asset_metrics_path}"
            )
        fallback = sorted(topk_candidates)[0]
        print(f"[maps] warning: {TOPK_COL} missing for {city}; using {fallback}")
        topk_col = fallback
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

    metric_cols = ["bldg_id", topk_col]
    metrics_small = metrics[metric_cols].copy()

    if topk_col != TOPK_COL:
        metrics_small = metrics_small.rename(columns={topk_col: TOPK_COL})

    features_small = features[feature_cols].copy()

    df = features_small.merge(metrics_small, on="bldg_id", how="inner")

    df[BORDERLINE_COL] = (
        (df[TOPK_COL] > BORDERLINE_LOW) & (df[TOPK_COL] < BORDERLINE_HIGH)
    ).astype(int)

    gdf = gdf.merge(df, on="bldg_id", how="inner", validate="one_to_one")

    if gdf.empty:
        raise ValueError(f"Join produced empty GeoDataFrame for {city}")

    print(f"[maps] joined rows for {city}: {len(gdf):,}")
    return gdf


def maybe_sample_gdf(
    gdf: gpd.GeoDataFrame,
    max_features: int = 100_000,
    random_seed: int = 42,
) -> gpd.GeoDataFrame:
    if len(gdf) <= max_features:
        return gdf

    return gdf.sample(max_features, random_state=random_seed)


def get_quantile_bounds(
    gdf: gpd.GeoDataFrame,
    column: str,
    q_low: float = 0.02,
    q_high: float = 0.98,
) -> tuple[float, float]:
    values = gdf[column].dropna()

    if values.empty:
        raise ValueError(f"No valid values for column {column}")

    vmin = float(values.quantile(q_low))
    vmax = float(values.quantile(q_high))

    if vmin == vmax:
        vmin = float(values.min())
        vmax = float(values.max())

    if vmin == vmax:
        raise ValueError(f"Column {column} has no visible variation")

    return vmin, vmax


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=260, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)

    print(f"[maps] saved: {png_path}")
    print(f"[maps] saved: {pdf_path}")

    return [png_path, pdf_path]


def plot_continuous_map(
    gdf: gpd.GeoDataFrame,
    city: str,
    column: str,
    label: str,
    output_dir: Path,
    max_features: int = 100_000,
) -> list[Path]:
    city = city.upper()

    if column not in gdf.columns:
        raise KeyError(f"Column {column} not found for {city}")

    plot_gdf = maybe_sample_gdf(gdf, max_features=max_features)
    vmin, vmax = get_quantile_bounds(plot_gdf, column)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    # Plot without automatic legend so the map can breathe like it pays rent.
    plot_gdf.plot(
        column=column,
        ax=ax,
        linewidth=0,
        cmap="plasma",
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
        legend=False,
    )

    ax.set_title(f"{city} — {label}", fontsize=11, pad=4)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="plasma",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm._A = []

    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.026,
        pad=0.01,
    )
    cbar.set_label(label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.01, right=0.92, top=0.94, bottom=0.01)

    stem = f"{city}_{column}_map"
    paths = save_figure(fig, output_dir, stem)
    plt.close(fig)

    return paths


def plot_borderline_map(
    gdf: gpd.GeoDataFrame,
    city: str,
    output_dir: Path,
    max_features: int = 100_000,
) -> list[Path]:
    city = city.upper()

    if BORDERLINE_COL not in gdf.columns:
        raise KeyError(f"Column {BORDERLINE_COL} not found for {city}")

    plot_gdf = maybe_sample_gdf(gdf, max_features=max_features)

    base = plot_gdf[plot_gdf[BORDERLINE_COL] == 0]
    borderline = plot_gdf[plot_gdf[BORDERLINE_COL] == 1]

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    base.plot(
        ax=ax,
        color="#d9d9d9",
        linewidth=0,
        alpha=0.35,
    )

    if not borderline.empty:
        borderline.plot(
            ax=ax,
            color="#d7191c",
            linewidth=0,
            alpha=0.95,
        )

    share = len(borderline) / len(plot_gdf) if len(plot_gdf) else 0.0

    ax.set_title(
        f"{city} — Borderline assets, k=1000 "
        f"({BORDERLINE_LOW} < P(top-k) < {BORDERLINE_HIGH})",
        fontsize=10,
        pad=4,
    )
    ax.text(
        0.01,
        0.01,
        f"Borderline share in plotted sample: {share:.3%}",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)

    stem = f"{city}_{BORDERLINE_COL}_map"
    paths = save_figure(fig, output_dir, stem)
    plt.close(fig)

    return paths


def export_city_sample(
    gdf: gpd.GeoDataFrame,
    city: str,
    output_dir: Path,
    n: int = 5,
) -> Path:
    city = city.upper()

    sample_cols = [
        "bldg_id",
        "E_hat_v0",
        "H_pluvial_v1_mm",
        "H_pluvial_v1_logrel",
        TOPK_COL,
        BORDERLINE_COL,
        "Y_damage",
    ]

    available_cols = [c for c in sample_cols if c in gdf.columns]
    sample = gdf[available_cols].head(n)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{city}_phase3_feature_sample.csv"

    sample.to_csv(out_path, index=False)

    print(f"[maps] saved sample: {out_path}")
    return out_path


def write_map_readme(output_dir: Path) -> None:
    readme = f"""# Phase 3 GIS Sanity Maps

These maps are visual sanity checks for the Phase 3 transfer inputs and decision outputs.

They are not calibrated flood-risk maps. Their purpose is to show that:

- spatial feature structure exists,
- exposure and hazard proxies are joined correctly to building geometries,
- decision outputs have interpretable spatial structure,
- borderline assets can be located spatially.

Main maps generated per city:

1. `E_hat_v0`  
   Deterministic exposure proxy.

2. `H_pluvial_v1_mm`  
   Raw pluvial hazard proxy in millimetres.

3. `{TOPK_COL}`  
   Posterior probability that an asset belongs to the top-k priority set.

4. `{BORDERLINE_COL}`  
   Binary indicator for assets with `{BORDERLINE_LOW} < P(top-k) < {BORDERLINE_HIGH}`.

Continuous maps use a 2–98% quantile stretch for visual contrast.  
This improves legibility but means colour scales should not be interpreted as directly comparable across cities.

Both PNG and PDF outputs are generated.
"""
    path = output_dir / "README.md"
    path.write_text(readme, encoding="utf-8")
    print(f"[maps] saved readme: {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[maps] data_root={get_data_root()}")
    print(f"[maps] output_dir={OUTPUT_DIR}")

    for city in CITIES:
        print(f"[maps] city={city}")

        gdf = load_city_geodata(city)

        for column, label in CONTINUOUS_MAPS.items():
            if column in gdf.columns:
                plot_continuous_map(
                    gdf=gdf,
                    city=city,
                    column=column,
                    label=label,
                    output_dir=OUTPUT_DIR,
                )
            else:
                print(f"[maps] skipping missing column for {city}: {column}")

        plot_borderline_map(
            gdf=gdf,
            city=city,
            output_dir=OUTPUT_DIR,
        )

        export_city_sample(
            gdf=gdf,
            city=city,
            output_dir=OUTPUT_DIR,
        )

    write_map_readme(OUTPUT_DIR)

    print("[maps] done")


if __name__ == "__main__":
    main()