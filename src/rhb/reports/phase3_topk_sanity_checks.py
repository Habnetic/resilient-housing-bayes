from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CITIES = ["RTM", "HAM", "DON"]
BASE = Path("outputs") / "phase3"
OUT = BASE / "cross_city_summary" / "sanity_checks"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8


def find_topk_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("topk_prob_k")]
    if not cols:
        raise KeyError(f"No topk_prob_k* columns found. Columns: {df.columns.tolist()}")
    return sorted(cols)


def summarize_topk_column(df: pd.DataFrame, city: str, col: str) -> dict:
    p = df[col]

    return {
        "city": city,
        "topk_col": col,
        "n_assets": int(len(df)),
        "min": float(p.min()),
        "p01": float(p.quantile(0.01)),
        "p05": float(p.quantile(0.05)),
        "p10": float(p.quantile(0.10)),
        "p50": float(p.quantile(0.50)),
        "p90": float(p.quantile(0.90)),
        "p95": float(p.quantile(0.95)),
        "p99": float(p.quantile(0.99)),
        "max": float(p.max()),
        "share_near_zero": float((p <= 0.05).mean()),
        "share_borderline": float(((p > BORDERLINE_LOW) & (p < BORDERLINE_HIGH)).mean()),
        "share_near_one": float((p >= 0.95).mean()),
    }


def plot_histogram(df_all: pd.DataFrame, col: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))

    for city in CITIES:
        sub = df_all[df_all["city"] == city]
        ax.hist(
            sub[col],
            bins=50,
            histtype="step",
            density=True,
            label=city,
        )

    ax.axvspan(BORDERLINE_LOW, BORDERLINE_HIGH, alpha=0.15)
    ax.set_xlabel("Top-k membership probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Phase 3 — Top-k Probability Histogram ({col})")
    ax.legend()

    fig.tight_layout()

    out_path = OUT / f"hist_{col}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    rows = []

    for city in CITIES:
        path = BASE / city / "asset_metrics.parquet"
        print(f"[sanity] loading {path}")

        df = pd.read_parquet(path)
        topk_cols = find_topk_cols(df)

        df = df.copy()
        df["city"] = city
        frames.append(df)

        for col in topk_cols:
            rows.append(summarize_topk_column(df, city, col))

    df_all = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(rows)

    summary_path = OUT / "topk_probability_sanity_summary.csv"
    summary.to_csv(summary_path, index=False)

    json_path = OUT / "topk_probability_sanity_summary.json"
    json_path.write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print(f"[sanity] saved summary: {summary_path}")
    print(f"[sanity] saved json: {json_path}")

    # Plot selected k values if available
    preferred_cols = [
        "topk_prob_k1000",
        "topk_prob_k2500",
        "topk_prob_k5000",
    ]

    for col in preferred_cols:
        if col in df_all.columns:
            out_path = plot_histogram(df_all, col)
            print(f"[sanity] saved histogram: {out_path}")

    print("[sanity] done")


if __name__ == "__main__":
    main()