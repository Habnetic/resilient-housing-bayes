"""Summarize M1-D1 prior-width sensitivity without rerunning Bayesian fits.

    python -m rhb.pipelines.summarize_m1d --smoke
    python -m rhb.pipelines.summarize_m1d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rhb.pipelines.run_m1d import OUTPUT_ROOT, sha256


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate independent outcome seeds within each gamma/prior setting."""
    required = {
        "gamma_true", "simulation_seed", "gamma_prior_sigma", "gamma_posterior_mean",
        "gamma_covered", "expected_regret_B0", "expected_regret_B1",
        "overlap_B1_baseline", "max_inclusion_change_B1_baseline",
        "heldout_elpd_B0", "heldout_elpd_B1", "rhat_max", "ess_bulk_min",
        "ess_tail_min", "divergences", "prior_predictive_rate_p50_B1",
    }
    if required.difference(df.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(df.columns))}")
    if df.duplicated(["gamma_true", "simulation_seed", "gamma_prior_sigma"]).any():
        raise ValueError("Duplicate gamma/seed/prior combinations")
    grouped = []
    for (gamma, sigma), sub in df.groupby(["gamma_true", "gamma_prior_sigma"], sort=True):
        if sub["simulation_seed"].nunique() != len(sub):
            raise ValueError("Duplicate simulation seed")
        a = sub["expected_regret_B0"].astype(float)
        b = sub["expected_regret_B1"].astype(float)
        delta = a - b
        estimate = sub["gamma_posterior_mean"].astype(float)
        eps = 1e-10
        grouped.append({
            "gamma_true": float(gamma),
            "gamma_prior_sigma": float(sigma),
            "replications": len(sub),
            "mean_gamma_posterior": float(estimate.mean()),
            "mean_gamma_bias": float((estimate - float(gamma)).mean()),
            "coverage_count": int(sub["gamma_covered"].astype(bool).sum()),
            "mean_expected_regret_B0": float(a.mean()),
            "mean_expected_regret_B1": float(b.mean()),
            "mean_paired_regret_improvement": float(delta.mean()),
            "B1_better": int((delta > eps).sum()),
            "ties": int((np.abs(delta) <= eps).sum()),
            "B0_better": int((delta < -eps).sum()),
            "mean_overlap_B1_baseline": float(sub["overlap_B1_baseline"].mean()),
            "max_change_inclusion_vs_baseline": float(sub["max_inclusion_change_B1_baseline"].max()),
            "mean_heldout_elpd_delta_B1_minus_B0": float(
                (sub["heldout_elpd_B1"].astype(float) - sub["heldout_elpd_B0"].astype(float)).mean()
            ),
            "mean_prior_predictive_rate_p50_B1": float(sub["prior_predictive_rate_p50_B1"].mean()),
            "diagnostic_flag_count": int((
                (sub["rhat_max"].astype(float) > 1.01)
                | (sub["ess_bulk_min"].astype(float) < 400)
                | (sub["ess_tail_min"].astype(float) < 400)
                | (sub["divergences"].astype(int) > 0)
            ).sum()),
        })
    return pd.DataFrame(grouped).sort_values(
        ["gamma_true", "gamma_prior_sigma"], ascending=[False, True]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stage = "smoke" if args.smoke else "full"
    folder = OUTPUT_ROOT / stage
    path = folder / "comparison_results.csv"
    df = pd.read_csv(path)
    agg = summarize(df)
    report_dir = folder / "summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(report_dir / "summary_by_gamma_prior.csv", index=False)
    (report_dir / "source_hash.json").write_text(
        json.dumps({"comparison_results_sha256": sha256(path)}, indent=2),
        encoding="utf-8",
    )
    print(agg.to_string(index=False))
    print(f"M1-D1 {stage.upper()} SUMMARY: COMPLETE")


if __name__ == "__main__":
    main()
