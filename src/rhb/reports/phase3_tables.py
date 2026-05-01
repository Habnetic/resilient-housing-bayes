from __future__ import annotations

from pathlib import Path
import pandas as pd

CITIES = ["RTM", "HAM", "DON"]
BASE_PATH = Path("outputs") / "phase3"
OUTPUT_DIR = BASE_PATH / "cross_city_summary" / "tables"

BORDERLINE_LOW = 0.2
BORDERLINE_HIGH = 0.8


def find_topk_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("topk_prob_k")])


def compute_summary(df: pd.DataFrame, city: str) -> list[dict]:
    rows = []

    topk_cols = find_topk_columns(df)

    for col in topk_cols:
        k = int(col.split("k")[-1])

        values = df[col].dropna()

        rows.append(
            {
                "city": city,
                "k": k,
                "n_assets": len(values),
                "share_near_zero": (values < 0.01).mean(),
                "share_borderline": ((values >= BORDERLINE_LOW) & (values <= BORDERLINE_HIGH)).mean(),
                "share_near_one": (values > 0.99).mean(),
                "p50": values.quantile(0.50),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
            }
        )

    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for city in CITIES:
        path = BASE_PATH / city / "asset_metrics.parquet"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)

        rows = compute_summary(df, city)
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows).sort_values(["city", "k"])

    # Save CSV
    csv_path = OUTPUT_DIR / "phase3_decision_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    # Save markdown (human-readable)
    md_path = OUTPUT_DIR / "phase3_decision_summary.md"
    md_path.write_text(summary_df.to_markdown(index=False))

    print(f"[phase3-tables] saved: {csv_path}")
    print(f"[phase3-tables] saved: {md_path}")


if __name__ == "__main__":
    main()