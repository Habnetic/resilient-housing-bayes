from __future__ import annotations

"""Generate publication/printer-oriented Phase 3 figures.

This script produces the compact figure set used by the Phase 3 paper.
It is intentionally conservative: grayscale / monochrome-safe, PDF/SVG/PNG
exports, direct labels where useful, and no exploratory ECDF clutter.

Expected inputs, relative to repository root:
    outputs/phase3/<CITY>/phase3_features_scaled.parquet
    outputs/phase3/<CITY>/asset_metrics.parquet
    outputs/phase3/<CITY>/summary.json
    outputs/phase3/<CITY>/decision_metrics.json
    outputs/phase3/<CITY>/prior_predictive_summary*.json   optional

Run from repository root:
    $env:PYTHONPATH="src"
    python -m rhb.reports.figures_phase3

Or directly:
    python src/rhb/reports/figures_phase3.py

Outputs:
    outputs/phase3/cross_city_summary/figures/

Core figures generated:
    fig_prior_predictive_event_rate.*
    phase3_decision_stability_structure_RTM.*
    phase3_certainty_composition_bar.*
    04_cross_city_probability_structure.*
    phase3_cross_city_stability_structure.*
    phase3_expected_risk_ranking_RTM.*
    phase3_borderline_share_vs_k.*

Notes:
    - Top-k columns are selected by closest available threshold to --share.
    - Default --share is 0.01, i.e. comparable 1% prioritisation thresholds.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]
CITY_LABELS = {
    "RTM": "Rotterdam",
    "HAM": "Hamburg",
    "DON": "Donostia--San Sebastián",
}

BASE_PATH = Path("outputs") / "phase3"
OUT_DIR = BASE_PATH / "cross_city_summary" / "figures"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8

# Grayscale-safe palette. Color had a meeting and mostly produced legend debt.
BLACK = "#111111"
DARK = "#333333"
MID = "#737373"
LIGHT = "#D9D9D9"
VERY_LIGHT = "#F2F2F2"
GRID = "#E6E6E6"
UNCERTAIN = "#BDBDBD"
STABLE_LOW = "#EFEFEF"
STABLE_HIGH = "#4A4A4A"
WHITE = "#FFFFFF"

CITY_STYLES = {
    "RTM": {"color": BLACK, "linestyle": "-", "marker": "o"},
    "HAM": {"color": "#555555", "linestyle": "--", "marker": "s"},
    "DON": {"color": "#999999", "linestyle": "-.", "marker": "^"},
}


def apply_paper_style() -> None:
    """Apply sober matplotlib settings for print/PDF figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_topk_cols(columns: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for col in columns:
        match = re.fullmatch(r"topk_prob_k(\d+)", col)
        if match:
            out[int(match.group(1))] = col
    return out


def select_k_for_share(metrics: pd.DataFrame, share: float) -> tuple[int, str]:
    """Select available top-k column nearest to a target asset share."""
    n_assets = len(metrics)
    target_k = max(1, int(round(n_assets * share)))
    candidates = parse_topk_cols(metrics.columns)
    if not candidates:
        raise KeyError("No topk_prob_k* columns found in asset_metrics.parquet")
    selected_k = min(candidates.keys(), key=lambda k: abs(k - target_k))
    return selected_k, candidates[selected_k]


def load_city(city: str, base_path: Path, share: float) -> pd.DataFrame:
    """Load Phase 3 features, asset metrics, and posterior summary for one city."""
    city = city.upper()
    city_dir = base_path / city

    features_path = city_dir / "phase3_features_scaled.parquet"
    metrics_path = city_dir / "asset_metrics.parquet"
    summary_path = city_dir / "summary.json"

    for path in [features_path, metrics_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    features = pd.read_parquet(features_path)
    metrics = pd.read_parquet(metrics_path)
    summary = pd.read_json(summary_path, orient="index")

    selected_k, topk_col = select_k_for_share(metrics, share)

    required_features = {"bldg_id", "E_hat_v0", "H_pluvial_v1_logrel"}
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise KeyError(f"Missing feature columns for {city}: {sorted(missing_features)}")

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
    df["selected_k"] = selected_k
    df["selected_share"] = selected_k / len(df)
    df["stable_low"] = df["topk_prob"] <= BORDERLINE_LOW
    df["unstable_boundary"] = (df["topk_prob"] > BORDERLINE_LOW) & (
        df["topk_prob"] < BORDERLINE_HIGH
    )
    df["stable_high"] = df["topk_prob"] >= BORDERLINE_HIGH
    return df


def load_all_cities(base_path: Path, share: float) -> dict[str, pd.DataFrame]:
    return {city: load_city(city, base_path, share) for city in CITIES}


def prepare_ranked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("expected_risk", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_norm"] = (out["rank"] - 1) / max(1, len(out) - 1)
    out["log_rank"] = np.log10(out["rank"])
    max_risk = out["expected_risk"].max()
    out["expected_risk_relative"] = (
        out["expected_risk"] / max_risk if max_risk > 0 else 0.0
    )
    return out


def rolling_band(x: pd.Series, y: pd.Series, bins: int = 120) -> pd.DataFrame:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()
    tmp["bin"] = pd.cut(tmp["x"], bins=bins, labels=False, include_lowest=True)
    grouped = tmp.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "x": grouped["x"].median(),
            "q25": grouped["y"].quantile(0.25),
            "median": grouped["y"].median(),
            "q75": grouped["y"].quantile(0.75),
        }
    ).dropna()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"[figures-phase3] saved: {path}")
    plt.close(fig)


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def _find_prior_summary_path(city_dir: Path) -> Path | None:
    """Prefer low-event prior predictive summary when present, else any summary."""
    preferred = city_dir / "prior_predictive_summary_low_event_weakly_informative.json"
    if preferred.exists():
        return preferred
    generic = city_dir / "prior_predictive_summary.json"
    if generic.exists():
        return generic
    matches = sorted(city_dir.glob("prior_predictive_summary*.json"))
    return matches[0] if matches else None


def plot_prior_predictive_event_rate(base_path: Path, output_dir: Path) -> None:
    """Prior predictive event-rate check across cities, if summaries exist."""
    rows: list[dict[str, float | str]] = []
    for city in CITIES:
        path = _find_prior_summary_path(base_path / city)
        if path is None:
            print(f"[figures-phase3] skipping prior predictive; missing summary for {city}")
            return
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        event = data.get("prior_event_rate_percentiles", {})
        rows.append(
            {
                "city": city,
                "observed": float(data.get("observed_synthetic_event_rate", np.nan)),
                "p05": float(event.get("p05", np.nan)),
                "p50": float(event.get("p50", np.nan)),
                "p95": float(event.get("p95", np.nan)),
            }
        )

    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    yerr = np.vstack([df["p50"] - df["p05"], df["p95"] - df["p50"]])

    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.errorbar(
        x,
        df["p50"],
        yerr=yerr,
        fmt="o",
        color=BLACK,
        ecolor=MID,
        elinewidth=1.2,
        capsize=3,
        label="Prior predictive event rate, p05--p95",
    )
    ax.scatter(
        x,
        df["observed"],
        marker="x",
        s=40,
        color=BLACK,
        linewidths=1.5,
        label="Observed synthetic event rate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["city"])
    ax.set_ylabel("Event rate")
    ax.set_title("Prior predictive event-rate check")
    ax.legend(frameon=False, loc="upper left")
    clean_axis(ax, grid=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_prior_predictive_event_rate")


def plot_decision_stability_structure(df: pd.DataFrame, output_dir: Path, city: str = "RTM") -> None:
    ranked = prepare_ranked(df)
    band = rolling_band(ranked["log_rank"], ranked["topk_prob"], bins=140)
    k = int(ranked["selected_k"].iloc[0])
    share = float(ranked["selected_share"].iloc[0])
    uncertain_share = float(ranked["unstable_boundary"].mean())

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.axhspan(0.8, 1.0, color=VERY_LIGHT, alpha=1.0)
    ax.axhspan(0.2, 0.8, color=LIGHT, alpha=0.75)
    ax.axhspan(0.0, 0.2, color=VERY_LIGHT, alpha=1.0)
    ax.fill_between(band["x"], band["q25"], band["q75"], color=LIGHT, alpha=0.9, linewidth=0)
    ax.plot(band["x"], band["median"], color=BLACK, linewidth=2.0)
    ax.axhline(BORDERLINE_LOW, color=MID, linestyle="--", linewidth=0.9)
    ax.axhline(BORDERLINE_HIGH, color=MID, linestyle="--", linewidth=0.9)

    ax.text(0.04, 0.88, "stable high-priority", transform=ax.transAxes, fontsize=8.5, weight="bold", color=DARK)
    ax.text(0.43, 0.48, "unstable\nboundary", transform=ax.transAxes, fontsize=8.5, weight="bold", color=DARK, ha="center", va="center")
    ax.text(0.70, 0.12, "stable low-priority", transform=ax.transAxes, fontsize=8.5, weight="bold", color=DARK)
    ax.text(
        0.97,
        0.52,
        f"{pct(uncertain_share, 2)} uncertain",
        transform=ax.transAxes,
        fontsize=9,
        weight="bold",
        color=BLACK,
        ha="right",
        va="center",
        bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.25"},
    )

    ax.set_title(f"Decision-stability structure ({CITY_LABELS.get(city, city)})")
    ax.set_xlabel("Asset rank (log10 scale)")
    ax.set_ylabel("Top-k membership probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, ranked["log_rank"].max())
    clean_axis(ax, grid=False)
    fig.text(
        0.125,
        -0.03,
        f"Selected threshold: k={k:,} ({share:.2%} of assets). Borderline: {BORDERLINE_LOW} < P(i in Top-k) < {BORDERLINE_HIGH}.",
        fontsize=8,
        color=DARK,
    )
    fig.tight_layout()
    save_figure(fig, output_dir, f"phase3_decision_stability_structure_{city}")


def plot_certainty_composition_bar(cities_df: dict[str, pd.DataFrame], output_dir: Path) -> None:
    rows = []
    for city, df in cities_df.items():
        rows.append(
            {
                "city": city,
                "stable_low": float(df["stable_low"].mean()),
                "unstable": float(df["unstable_boundary"].mean()),
                "stable_high": float(df["stable_high"].mean()),
                "k": int(df["selected_k"].iloc[0]),
                "share": float(df["selected_share"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows).set_index("city").loc[CITIES]

    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    y = np.arange(len(summary))
    left = np.zeros(len(summary))
    parts = [
        ("stable_low", "Stable low-priority", STABLE_LOW),
        ("unstable", "Borderline", UNCERTAIN),
        ("stable_high", "Stable high-priority", STABLE_HIGH),
    ]
    for col, label, color in parts:
        vals = summary[col].to_numpy()
        ax.barh(y, vals, left=left, color=color, edgecolor=WHITE, height=0.58, label=label)
        left += vals

    for i, city in enumerate(summary.index):
        uncertain = float(summary.loc[city, "unstable"])
        ax.text(1.01, i, f"{pct(uncertain, 2)} borderline", va="center", ha="left", fontsize=8.5, color=BLACK, weight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([CITY_LABELS.get(city, city) for city in summary.index])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Share of assets")
    ax.set_title("Most prioritisation decisions remain stable at comparable 1% thresholds")
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{100*x:.0f}%")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    clean_axis(ax, grid=True)
    fig.tight_layout()
    save_figure(fig, output_dir, "phase3_certainty_composition_bar")

    summary_out = summary.reset_index().rename(columns={"index": "city"})
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(output_dir / "phase3_certainty_composition_summary.csv", index=False)
    summary_out.to_markdown(output_dir / "phase3_certainty_composition_summary.md", index=False)


def plot_cross_city_probability_structure(cities_df: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Monochrome replacement for the original onepager 04_cross_city_probability_structure."""
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    rows = []
    for ax, city in zip(axes, CITIES):
        ranked = prepare_ranked(cities_df[city])
        band = rolling_band(ranked["rank_norm"], ranked["topk_prob"], bins=100)
        style = CITY_STYLES[city]

        ax.axhspan(0.2, 0.8, color=LIGHT, alpha=0.45)
        ax.fill_between(band["x"], band["q25"], band["q75"], color=LIGHT, alpha=0.75, linewidth=0)
        ax.plot(band["x"], band["median"], color=style["color"], linestyle=style["linestyle"], linewidth=1.8)
        ax.axhline(BORDERLINE_LOW, color=MID, linestyle="--", linewidth=0.8)
        ax.axhline(BORDERLINE_HIGH, color=MID, linestyle="--", linewidth=0.8)

        uncertain_share = float(ranked["unstable_boundary"].mean())
        k = int(ranked["selected_k"].iloc[0])
        share = float(ranked["selected_share"].iloc[0])
        ax.text(
            0.97,
            0.52,
            f"{pct(uncertain_share, 2)}\nborderline",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.2"},
        )
        ax.set_title(f"{city}\nk={k:,} ({share:.2%})", fontsize=9)
        ax.set_xlabel("Rank (normalised)")
        ax.set_ylim(-0.02, 1.02)
        clean_axis(ax, grid=False)

        rows.append(
            {
                "city": city,
                "n_assets": len(ranked),
                "k": k,
                "k_share": share,
                "borderline_share": uncertain_share,
                "stable_in_share": float((ranked["topk_prob"] >= BORDERLINE_HIGH).mean()),
                "stable_out_share": float((ranked["topk_prob"] <= BORDERLINE_LOW).mean()),
            }
        )

    axes[0].set_ylabel("P(asset in Top-k)")
    fig.suptitle("Same decision-stability structure across cities", fontsize=11, weight="bold", y=1.05)
    fig.tight_layout()
    save_figure(fig, output_dir, "04_cross_city_probability_structure")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cross_city_probability_structure_summary.csv", index=False)
    summary.to_markdown(output_dir / "cross_city_probability_structure_summary.md", index=False)


def plot_cross_city_stability_structure(cities_df: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)
    for ax, city in zip(axes, CITIES):
        ranked = prepare_ranked(cities_df[city])
        band = rolling_band(ranked["log_rank"], ranked["topk_prob"], bins=120)
        style = CITY_STYLES[city]
        uncertain_share = float(ranked["unstable_boundary"].mean())
        k = int(ranked["selected_k"].iloc[0])
        share = float(ranked["selected_share"].iloc[0])

        ax.axhspan(0.2, 0.8, color=LIGHT, alpha=0.65)
        ax.fill_between(band["x"], band["q25"], band["q75"], color=LIGHT, alpha=0.75)
        ax.plot(band["x"], band["median"], color=style["color"], linestyle=style["linestyle"], linewidth=1.9)
        ax.axhline(BORDERLINE_LOW, color=MID, linestyle="--", linewidth=0.8)
        ax.axhline(BORDERLINE_HIGH, color=MID, linestyle="--", linewidth=0.8)
        ax.text(
            0.97,
            0.52,
            f"{pct(uncertain_share, 2)}\nborderline",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            weight="bold",
            bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.2"},
        )
        ax.set_title(f"{city}\nk={k:,} ({share:.2%})", fontsize=9)
        ax.set_xlabel("Asset rank\n(log10 scale)")
        ax.set_xlim(0, ranked["log_rank"].max())
        ax.set_ylim(-0.02, 1.02)
        clean_axis(ax, grid=False)

    axes[0].set_ylabel("P(asset in Top-k)")
    fig.suptitle("Same decision-stability structure across cities", fontsize=11, weight="bold", y=1.05)
    fig.tight_layout()
    save_figure(fig, output_dir, "phase3_cross_city_stability_structure")


def plot_expected_risk_ranking(df: pd.DataFrame, output_dir: Path, city: str = "RTM") -> None:
    ranked = prepare_ranked(df)
    k = int(ranked["selected_k"].iloc[0])
    share = float(ranked["selected_share"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.fill_between(ranked["rank"], 0, ranked["expected_risk_relative"], color=VERY_LIGHT, linewidth=0)
    ax.plot(ranked["rank"], ranked["expected_risk_relative"], color=BLACK, linewidth=1.8)
    ax.axvline(k, color=MID, linestyle="--", linewidth=1.0)
    x_annot = min(max(k * 4.0, 9000), len(ranked) * 0.22)
    ax.annotate(
        f"Top {share:.1%}\nk={k:,}",
        xy=(k, 0.94),
        xytext=(x_annot, 0.70),
        textcoords="data",
        fontsize=8.5,
        weight="bold",
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": MID, "lw": 0.8},
        bbox={"facecolor": WHITE, "edgecolor": LIGHT, "boxstyle": "round,pad=0.25"},
    )
    ax.set_title(f"Expected risk is concentrated in a small upper tail ({CITY_LABELS.get(city, city)})")
    ax.set_xlabel("Asset rank")
    ax.set_ylabel("Expected risk (relative)")
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, len(ranked))
    clean_axis(ax, grid=True)
    fig.tight_layout()
    save_figure(fig, output_dir, f"phase3_expected_risk_ranking_{city}")


def plot_borderline_share_vs_k(base_path: Path, output_dir: Path, cities: list[str] | None = None) -> None:
    cities = cities or CITIES
    rows: list[dict[str, float | int | str]] = []
    for city in cities:
        path = base_path / city / "decision_metrics.json"
        if not path.exists():
            print(f"[figures-phase3] skipping borderline-vs-k; missing: {path}")
            return
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            n_assets = row.get("n_assets")
            rows.append(
                {
                    "city": city,
                    "k": int(row["k"]),
                    "k_share": int(row["k"]) / float(n_assets) if n_assets else np.nan,
                    "borderline_share_pct": 100 * float(row["borderline_share"]),
                }
            )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for city in cities:
        sub = metrics[metrics["city"] == city].sort_values("k")
        style = CITY_STYLES[city]
        ax.plot(
            sub["k"],
            sub["borderline_share_pct"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.5,
            markersize=4.0,
        )
        last = sub.iloc[-1]
        ax.annotate(
            city,
            xy=(float(last["k"]), float(last["borderline_share_pct"])),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            weight="bold",
            color=style["color"],
        )

    ax.set_title("Borderline share remains small across prioritisation thresholds", pad=14)
    ax.text(
        0.5,
        1.01,
        "Including under fixed-specification cross-city transfer",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color=DARK,
    )
    ax.set_xlabel("Top-k threshold")
    ax.set_ylabel("Borderline share (% of assets)")
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_ylim(0, max(1.2, metrics["borderline_share_pct"].max() * 1.35))
    ax.set_xlim(0, metrics["k"].max() * 1.12)
    ax.set_yticks([0, 0.01, 0.05, 0.1, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.01", "0.05", "0.1", "0.5", "1.0"])
    clean_axis(ax, grid=True)
    fig.subplots_adjust(top=0.82)
    save_figure(fig, output_dir, "phase3_borderline_share_vs_k")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--reference-city", default="RTM", choices=CITIES)
    parser.add_argument(
        "--share",
        type=float,
        default=0.01,
        help="Target top-k share. Nearest available topk_prob_k* column is selected per city.",
    )
    args = parser.parse_args()

    apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cities_df = load_all_cities(args.base_path, args.share)
    reference_city = args.reference_city.upper()
    reference_df = cities_df[reference_city]

    plot_prior_predictive_event_rate(args.base_path, args.output_dir)
    plot_decision_stability_structure(reference_df, args.output_dir, city=reference_city)
    plot_certainty_composition_bar(cities_df, args.output_dir)
    plot_cross_city_probability_structure(cities_df, args.output_dir)
    plot_cross_city_stability_structure(cities_df, args.output_dir)
    plot_expected_risk_ranking(reference_df, args.output_dir, city=reference_city)
    plot_borderline_share_vs_k(args.base_path, args.output_dir)

    print(f"\n[figures-phase3] done. Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
