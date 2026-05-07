from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]
BASE_PATH = Path("outputs") / "phase3"
OUT_DIR = BASE_PATH / "cross_city_summary" / "onepager_figures"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_topk_cols(columns: list[str]) -> dict[int, str]:
    out = {}
    for col in columns:
        m = re.fullmatch(r"topk_prob_k(\d+)", col)
        if m:
            out[int(m.group(1))] = col
    return out


def select_k_for_share(metrics: pd.DataFrame, share: float) -> tuple[int, str]:
    n = len(metrics)
    target_k = max(1, int(round(n * share)))
    candidates = parse_topk_cols(metrics.columns.tolist())

    if not candidates:
        raise KeyError("No topk_prob_k* columns found in asset_metrics.parquet")

    k = min(candidates.keys(), key=lambda kk: abs(kk - target_k))
    return k, candidates[k]


def load_city(city: str, share: float) -> pd.DataFrame:
    city = city.upper()

    features_path = BASE_PATH / city / "phase3_features_scaled.parquet"
    metrics_path = BASE_PATH / city / "asset_metrics.parquet"
    summary_path = BASE_PATH / city / "summary.json"

    features = pd.read_parquet(features_path)
    metrics = pd.read_parquet(metrics_path)
    summary = pd.read_json(summary_path, orient="index")

    k, topk_col = select_k_for_share(metrics, share)

    df = features.merge(
        metrics[["bldg_id", topk_col]],
        on="bldg_id",
        how="inner",
        validate="one_to_one",
    )

    alpha = float(summary.loc["alpha", "mean"])
    beta_e = float(summary.loc["beta_E", "mean"])
    beta_h = float(summary.loc["beta_H", "mean"])

    df["expected_risk"] = sigmoid(
        alpha
        + beta_e * df["E_hat_v0"].to_numpy()
        + beta_h * df["H_pluvial_v1_logrel"].to_numpy()
    )

    df = df.rename(columns={topk_col: "topk_prob"})
    df["city"] = city
    df["selected_k"] = k
    df["selected_share"] = k / len(df)

    return df


def prepare_ranked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("expected_risk", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_norm"] = (out["rank"] - 1) / max(1, len(out) - 1)
    out["expected_risk_relative"] = out["expected_risk"] / out["expected_risk"].max()
    return out


def rolling_band(
    x: pd.Series,
    y: pd.Series,
    bins: int = 120,
) -> pd.DataFrame:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    tmp["bin"] = pd.cut(tmp["x"], bins=bins, labels=False, include_lowest=True)

    g = tmp.groupby("bin", observed=True)

    band = pd.DataFrame(
        {
            "x": g["x"].median(),
            "q25": g["y"].quantile(0.25),
            "median": g["y"].median(),
            "q75": g["y"].quantile(0.75),
        }
    ).dropna()

    return band


def save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    svg = OUT_DIR / f"{stem}.svg"

    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    plt.close(fig)

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")
    print(f"[saved] {svg}")


def plot_deterministic_ranking(df: pd.DataFrame, city: str) -> None:
    ranked = prepare_ranked(df)
    k = int(ranked["selected_k"].iloc[0])

    fig, ax = plt.subplots(figsize=(4.2, 3.3))

    ax.plot(ranked["rank"], ranked["expected_risk_relative"], linewidth=2)
    ax.axvline(k, linestyle="--", linewidth=1.5)
    ax.fill_between(
        ranked["rank"],
        0,
        1,
        where=ranked["rank"] <= k,
        alpha=0.12,
    )

    ax.set_title("Assets ranked by expected risk")
    ax.set_xlabel("Asset rank")
    ax.set_ylabel("Expected risk (relative)")
    ax.set_ylim(0, 1.03)

    ax.text(
        0.05,
        0.15,
        f"Top {ranked['selected_share'].iloc[0]:.1%}",
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
    )

    fig.tight_layout()
    save(fig, f"{city}_01_deterministic_ranking")


def plot_probability_ranking(df: pd.DataFrame, city: str) -> None:
    ranked = prepare_ranked(df)
    band = rolling_band(ranked["rank_norm"], ranked["topk_prob"])

    fig, ax = plt.subplots(figsize=(4.2, 3.3))

    ax.fill_between(
        band["x"],
        band["q25"],
        band["q75"],
        alpha=0.18,
        label="Local 25–75% band",
    )
    ax.plot(
        band["x"],
        band["median"],
        linewidth=2,
        label="Median local probability",
    )

    ax.set_title("Probability of being in top set")
    ax.set_xlabel("Asset rank (normalised)")
    ax.set_ylabel("P(in top set)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save(fig, f"{city}_02_probability_ranking")


def plot_stability_classification(df: pd.DataFrame, city: str) -> None:
    ranked = prepare_ranked(df)
    band = rolling_band(ranked["rank_norm"], ranked["topk_prob"])

    fig, ax = plt.subplots(figsize=(4.2, 3.3))

    ax.axhspan(0.8, 1.0, alpha=0.10)
    ax.axhspan(0.2, 0.8, alpha=0.12)
    ax.axhspan(0.0, 0.2, alpha=0.10)

    ax.axhline(BORDERLINE_LOW, linestyle="--", linewidth=1)
    ax.axhline(BORDERLINE_HIGH, linestyle="--", linewidth=1)

    ax.fill_between(
        band["x"],
        band["q25"],
        band["q75"],
        alpha=0.18,
    )
    ax.plot(band["x"], band["median"], linewidth=2)

    ax.text(0.06, 0.86, "Stable in", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(0.42, 0.48, "Unstable\nboundary", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(0.72, 0.10, "Stable out", transform=ax.transAxes, fontsize=9, weight="bold")

    ax.set_title("Classification by top-set probability")
    ax.set_xlabel("Asset rank (normalised)")
    ax.set_ylabel("P(in top set)")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    save(fig, f"{city}_03_stability_classification")


def plot_cross_city(cities_df: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), sharey=True)

    rows = []

    for ax, city in zip(axes, CITIES):
        ranked = prepare_ranked(cities_df[city])
        band = rolling_band(ranked["rank_norm"], ranked["topk_prob"], bins=100)

        ax.fill_between(band["x"], band["q25"], band["q75"], alpha=0.18)
        ax.plot(band["x"], band["median"], linewidth=2)

        ax.axhline(BORDERLINE_LOW, linestyle="--", linewidth=1)
        ax.axhline(BORDERLINE_HIGH, linestyle="--", linewidth=1)

        k = int(ranked["selected_k"].iloc[0])
        share = float(ranked["selected_share"].iloc[0])

        ax.set_title(f"{city}\nk={k} ({share:.2%})", fontsize=9)
        ax.set_xlabel("Rank (normalised)")
        ax.set_ylim(-0.02, 1.02)

        borderline = ranked[
            (ranked["topk_prob"] > BORDERLINE_LOW)
            & (ranked["topk_prob"] < BORDERLINE_HIGH)
        ]

        rows.append(
            {
                "city": city,
                "n_assets": len(ranked),
                "k": k,
                "k_share": share,
                "borderline_share": len(borderline) / len(ranked),
                "stable_in_share": (ranked["topk_prob"] >= BORDERLINE_HIGH).mean(),
                "stable_out_share": (ranked["topk_prob"] <= BORDERLINE_LOW).mean(),
            }
        )

    axes[0].set_ylabel("P(in top set)")

    fig.suptitle("Same decision-stability structure across cities", y=1.08)
    fig.tight_layout()
    save(fig, "04_cross_city_probability_structure")

    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "onepager_summary.csv", index=False)
    summary.to_markdown(OUT_DIR / "onepager_summary.md", index=False)
    print(summary)


def plot_tiny_boundary_inset(df: pd.DataFrame, city: str) -> None:
    ranked = prepare_ranked(df)
    band = rolling_band(ranked["rank_norm"], ranked["topk_prob"])

    fig, ax = plt.subplots(figsize=(2.6, 1.3))

    ax.axhspan(0.2, 0.8, alpha=0.12)
    ax.plot(band["x"], band["median"], linewidth=2)
    ax.axhline(0.2, linestyle="--", linewidth=1)
    ax.axhline(0.8, linestyle="--", linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    fig.tight_layout(pad=0.05)
    save(fig, f"{city}_05_tiny_boundary_inset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-city",
        default="RTM",
        choices=CITIES,
        help="City used for the three main explanatory panels.",
    )
    parser.add_argument(
        "--share",
        type=float,
        default=0.01,
        help="Target top-k share. Script selects nearest available topk_prob_k* column.",
    )
    args = parser.parse_args()

    cities_df = {city: load_city(city, args.share) for city in CITIES}

    ref = args.reference_city.upper()
    ref_df = cities_df[ref]

    plot_deterministic_ranking(ref_df, ref)
    plot_probability_ranking(ref_df, ref)
    plot_stability_classification(ref_df, ref)
    plot_cross_city(cities_df)
    plot_tiny_boundary_inset(ref_df, ref)

    print(f"\n[done] figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()