from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rhb.decision.phase3_decision_metrics import (
    compute_posterior_probabilities,
    compute_topk_membership,
    load_posterior_params,
    save_json,
    summarize_decision_metrics,
)


DEFAULT_K_VALUES = [1000, 2500, 5000]
DEFAULT_K_SHARES = [0.005, 0.01, 0.025]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City code: RTM, HAM, DON")
    parser.add_argument(
        "--max-draws",
        type=int,
        default=None,
        help="Optional cap on posterior draws for faster testing",
    )
    args = parser.parse_args()

    city = args.city.upper()

    project_root = Path(__file__).resolve().parents[3]
    city_dir = project_root / "outputs" / "phase3" / city

    features_path = city_dir / "phase3_features_scaled.parquet"
    idata_path = city_dir / "idata.nc"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")
    if not idata_path.exists():
        raise FileNotFoundError(f"Missing idata file: {idata_path}")

    print(f"[phase3-decision] city={city}")
    print(f"[phase3-decision] features={features_path}")
    print(f"[phase3-decision] idata={idata_path}")

    df = pd.read_parquet(features_path)

    required = ["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    e = df["E_hat_v0"].to_numpy()
    h = df["H_pluvial_v1_logrel"].to_numpy()

    alpha, beta_e, beta_h = load_posterior_params(idata_path)

    if args.max_draws is not None:
        alpha = alpha[: args.max_draws]
        beta_e = beta_e[: args.max_draws]
        beta_h = beta_h[: args.max_draws]

    print(f"[phase3-decision] posterior draws={len(alpha)}")
    print(f"[phase3-decision] assets={len(df)}")

    p_draws = compute_posterior_probabilities(
        alpha=alpha,
        beta_e=beta_e,
        beta_h=beta_h,
        e=e,
        h=h,
    )

    df_metrics = df[["bldg_id"]].copy()
    df_metrics["p_mean"] = p_draws.mean(axis=0)
    df_metrics["p_std"] = p_draws.std(axis=0)

    k_values = list(DEFAULT_K_VALUES)
    for share in DEFAULT_K_SHARES:
        k_values.append(max(1, int(round(len(df) * share))))

    k_values = sorted(set(k_values))

    summaries = []

    for k in k_values:
        print(f"[phase3-decision] computing top-k metrics for k={k}")

        topk_prob, rank_std = compute_topk_membership(p_draws, k=k)

        df_metrics[f"topk_prob_k{k}"] = topk_prob
        df_metrics[f"rank_std_k{k}"] = rank_std

        summaries.append(
            summarize_decision_metrics(
                topk_prob=topk_prob,
                rank_std=rank_std,
                k=k,
            )
        )

    asset_metrics_path = city_dir / "asset_metrics.parquet"
    decision_summary_path = city_dir / "decision_metrics.json"

    df_metrics.to_parquet(asset_metrics_path, index=False)
    save_json(summaries, decision_summary_path)

    print(f"[phase3-decision] saved asset metrics: {asset_metrics_path}")
    print(f"[phase3-decision] saved decision summary: {decision_summary_path}")


if __name__ == "__main__":
    main()