from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def compute_support_diagnostics(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    cols: list[str],
    reference_city: str = "RTM",
    target_city: str = "HAM",
) -> pd.DataFrame:
    rows = []

    for col in cols:
        if col not in df_ref.columns:
            raise KeyError(f"Reference DataFrame missing column: {col}")
        if col not in df_target.columns:
            raise KeyError(f"Target DataFrame missing column: {col}")

        ref = df_ref[col]
        tgt = df_target[col]

        if ref.isna().any():
            raise ValueError(f"Reference column '{col}' contains NaNs")
        if tgt.isna().any():
            raise ValueError(f"Target column '{col}' contains NaNs")

        ref_min = float(ref.min())
        ref_max = float(ref.max())

        share_below_ref_min = float((tgt < ref_min).mean())
        share_above_ref_max = float((tgt > ref_max).mean())
        share_outside_ref_support = share_below_ref_min + share_above_ref_max

        rows.append(
            {
                "feature": col,
                "reference_city": reference_city,
                "target_city": target_city,
                "ref_min": ref_min,
                "ref_p05": float(ref.quantile(0.05)),
                "ref_p50": float(ref.quantile(0.50)),
                "ref_p95": float(ref.quantile(0.95)),
                "ref_max": float(ref_max),
                "target_min": float(tgt.min()),
                "target_p05": float(tgt.quantile(0.05)),
                "target_p50": float(tgt.quantile(0.50)),
                "target_p95": float(tgt.quantile(0.95)),
                "target_max": float(tgt.max()),
                "share_below_ref_min": share_below_ref_min,
                "share_above_ref_max": share_above_ref_max,
                "share_outside_ref_support": share_outside_ref_support,
            }
        )

    return pd.DataFrame(rows)


def save_support_diagnostics(
    diagnostics_df: pd.DataFrame,
    output_json_path: str | Path,
    output_csv_path: str | Path | None = None,
) -> None:
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    records = diagnostics_df.to_dict(orient="records")
    output_json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df.to_csv(output_csv_path, index=False)