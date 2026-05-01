from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]
K = 1000

BASE_PATH = Path("outputs") / "phase3"
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "figures"


def load_metrics(city: str) -> pd.DataFrame:
    path = BASE_PATH / city / "asset_metrics.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    col = f"topk_prob_k{K}"

    if col not in df.columns:
        raise KeyError(f"{city} missing {col}. Available: {df.columns.tolist()}")

    return df[[col]].copy()


def plot_topk_histogram() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    col = f"topk_prob_k{K}"

    for city in CITIES:
        df = load_metrics(city)
        ax.hist(
            df[col],
            bins=50,
            histtype="step",
            linewidth=1.6,
            density=True,
            label=city,
        )

    ax.set_title(f"Posterior top-k membership probability, k={K}")
    ax.set_xlabel(f"P(asset in top-{K})")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    ax.text(
        0.02,
        0.95,
        "Mass near 0 or 1 indicates stable prioritisation.\n"
        "Mass between 0.2 and 0.8 indicates borderline decisions.",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    fig.tight_layout()

    png = OUTPUT_DIR / f"topk_probability_comparison_k{K}.png"
    pdf = OUTPUT_DIR / f"topk_probability_comparison_k{K}.pdf"

    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def main() -> None:
    plot_topk_histogram()


if __name__ == "__main__":
    main()