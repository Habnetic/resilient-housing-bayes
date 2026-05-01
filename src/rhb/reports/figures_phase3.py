from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]
DEFAULT_BASE_PATH = Path("outputs") / "phase3"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_PATH / "cross_city_summary" / "figures"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8
MIDPOINT = 0.5


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y values for an empirical CDF."""
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def find_topk_columns(df: pd.DataFrame) -> list[str]:
    """Find top-k probability columns in asset metrics."""
    cols = [
        c
        for c in df.columns
        if c.startswith("topk_prob_k")
        or c.startswith("topk_prob")
        or c.startswith("p_topk")
        or c.startswith("top_k_prob")
    ]

    if not cols:
        raise KeyError(
            "No top-k probability columns found. Expected columns starting with "
            "'topk_prob_k', 'topk_prob', 'p_topk', or 'top_k_prob'. "
            f"Available columns: {df.columns.tolist()}"
        )

    return sorted(cols)


def load_asset_metrics(
    base_path: Path,
    cities: list[str],
    max_rows_per_city: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load asset-level Phase 3 metrics for all cities."""
    frames: list[pd.DataFrame] = []

    for city in cities:
        path = base_path / city / "asset_metrics.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Missing asset metrics file: {path}")

        df = pd.read_parquet(path)

        if max_rows_per_city is not None and len(df) > max_rows_per_city:
            df = df.sample(max_rows_per_city, random_state=random_seed)

        df = df.copy()
        df["city"] = city
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def choose_primary_topk_column(df: pd.DataFrame) -> str:
    """
    Choose one top-k probability column for the primary comparison figure.

    Preference order:
    - k=5000 if available
    - k=2500 if available
    - k=1000 if available
    - first detected top-k column
    """
    cols = find_topk_columns(df)

    preferred_cols = [
        "topk_prob_k5000",
        "topk_prob_k2500",
        "topk_prob_k1000",
        "topk_prob_5000",
        "topk_prob_2500",
        "topk_prob_1000",
        "p_topk_5000",
        "p_topk_2500",
        "p_topk_1000",
        "top_k_prob_5000",
        "top_k_prob_2500",
        "top_k_prob_1000",
    ]

    for preferred in preferred_cols:
        if preferred in cols:
            return preferred

    return cols[0]


def _plot_ecdf_core(
    df: pd.DataFrame,
    cities: list[str],
    topk_col: str,
    title: str,
    out_path: Path,
    xlim: tuple[float, float],
    annotate: bool = False,
) -> Path:
    """Internal ECDF plotting helper."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for city in cities:
        values = df.loc[df["city"] == city, topk_col].dropna().to_numpy()
        x, y = ecdf(values)
        ax.plot(x, y, label=city)

    ax.axvspan(
        BORDERLINE_LOW,
        BORDERLINE_HIGH,
        alpha=0.2,
        label="borderline zone",
    )

    ax.axvline(
        MIDPOINT,
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="decision midpoint",
    )

    if annotate:
        ax.text(
            0.03,
            0.92,
            "RTM/HAM: near-binary decisions",
            transform=ax.transAxes,
            fontsize=9,
        )
        ax.text(
            0.03,
            0.86,
            "DON: distribution shift, not uncertainty diffusion",
            transform=ax.transAxes,
            fontsize=9,
        )

    ax.set_xlabel("Top-k membership probability")
    ax.set_ylabel("ECDF")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.legend()
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[figures-phase3] saved: {out_path}")
    return out_path


def plot_topk_ecdf(
    base_path: Path = DEFAULT_BASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cities: list[str] | None = None,
    topk_col: str | None = None,
    max_rows_per_city: int | None = 50_000,
) -> list[Path]:
    """
    Plot ECDF of top-k membership probability by city.

    Saves:
    - full ECDF
    - low-probability zoom
    - high-probability zoom
    """
    cities = cities or CITIES
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_asset_metrics(
        base_path=base_path,
        cities=cities,
        max_rows_per_city=max_rows_per_city,
    )

    if topk_col is None:
        topk_col = choose_primary_topk_column(df)

    if topk_col not in df.columns:
        raise KeyError(f"Column not found: {topk_col}")

    outputs = []

    outputs.append(
        _plot_ecdf_core(
            df=df,
            cities=cities,
            topk_col=topk_col,
            title=f"Phase 3 — Top-k Membership Probability ECDF ({topk_col})",
            out_path=output_dir / f"phase3_topk_ecdf_full_{topk_col}.png",
            xlim=(0.0, 1.0),
            annotate=False,
        )
    )

    outputs.append(
        _plot_ecdf_core(
            df=df,
            cities=cities,
            topk_col=topk_col,
            title=f"Phase 3 — Top-k ECDF, Low-Probability Zoom ({topk_col})",
            out_path=output_dir / f"phase3_topk_ecdf_low_zoom_{topk_col}.png",
            xlim=(0.0, 0.3),
            annotate=True,
        )
    )

    outputs.append(
        _plot_ecdf_core(
            df=df,
            cities=cities,
            topk_col=topk_col,
            title=f"Phase 3 — Top-k ECDF, High-Probability Zoom ({topk_col})",
            out_path=output_dir / f"phase3_topk_ecdf_high_zoom_{topk_col}.png",
            xlim=(0.7, 1.0),
            annotate=False,
        )
    )

    return outputs


def plot_borderline_share_vs_k(
    base_path: Path = DEFAULT_BASE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cities: list[str] | None = None,
) -> Path:
    """Plot borderline share vs k using decision_metrics.json files."""
    cities = cities or CITIES
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for city in cities:
        path = base_path / city / "decision_metrics.json"

        if not path.exists():
            raise FileNotFoundError(f"Missing decision metrics file: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            rows.append(
                {
                    "city": city,
                    "k": int(row["k"]),
                    "borderline_share": float(row["borderline_share"]),
                }
            )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))

    for city in cities:
        sub = df[df["city"] == city].sort_values("k")
        ax.plot(sub["k"], sub["borderline_share"], marker="o", label=city)

    ax.set_xlabel("k")
    ax.set_ylabel("Borderline share")
    ax.set_title("Phase 3 — Borderline Share vs k")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "phase3_borderline_share_vs_k.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[figures-phase3] saved: {out_path}")
    return out_path


def main() -> None:
    """Generate minimum Phase 3 comparison figures."""
    plot_topk_ecdf()
    plot_topk_ecdf(topk_col="topk_prob_k1000")
    plot_borderline_share_vs_k()


if __name__ == "__main__":
    main()