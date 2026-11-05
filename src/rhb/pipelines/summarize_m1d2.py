"""Reproducible M1-D2 per-prior and paired decision summaries.

    python -m rhb.pipelines.summarize_m1d2 --smoke
    python -m rhb.pipelines.summarize_m1d2

No Bayesian inference. Never silently reconstruct missing case results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "outputs/phase4/RTM/m1d2"
THRESHOLD = 1e-10


def sha256_file(path: Path) -> str:
    """File identity for the report manifest."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_to_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Match each experiment setting to P0 within the *same* gamma and seed."""
    required = {
        "gamma_true", "simulation_seed", "prior_id", "alpha_prior_mean", "alpha_prior_sigma",
        "alpha_posterior_mean", "gamma_posterior_mean",
        "gamma_covered", "alpha_covered", "expected_regret_B0", "expected_regret_B1",
        "overlap_B1_P0", "max_inclusion_change_B1_P0", "heldout_elpd_B1",
        "heldout_elpd_B0", "brier_B1", "rmse_vs_true_p_B1", "rhat_max", "ess_bulk_min",
        "ess_tail_min", "divergences", "train_damage_rate", "posterior_predictive_rate_p50_B1",
    }
    if missing := required - set(df.columns):
        raise ValueError(f"Missing M1-D2 columns: {sorted(missing)}")
    keys = ["gamma_true", "simulation_seed", "prior_id"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate gamma/seed/prior combinations")
    settings = set(df["prior_id"].unique())
    if "P0" not in settings:
        raise ValueError("Missing reference prior P0")
    for (gamma, seed), group in df.groupby(["gamma_true", "simulation_seed"]):
        if set(group["prior_id"]) != settings:
            raise ValueError(f"Missing prior variant in gamma={gamma}, seed={seed}")
        if group["expected_regret_B0"].max() - group["expected_regret_B0"].min() > 1e-8:
            raise ValueError("B0 reference regret changed across priors")
        if group["train_damage_rate"].max() - group["train_damage_rate"].min() > 1e-8:
            raise ValueError("Training outcomes changed across priors")
    p0 = df.loc[df["prior_id"].eq("P0"), [
        "gamma_true", "simulation_seed", "expected_regret_B1", "heldout_elpd_B1",
        "brier_B1", "rmse_vs_true_p_B1",
    ]].rename(columns={
        "expected_regret_B1": "P0_regret",
        "heldout_elpd_B1": "P0_heldout_elpd",
        "brier_B1": "P0_brier",
        "rmse_vs_true_p_B1": "P0_rmse_true_p",
    })
    paired = df.merge(p0, on=["gamma_true", "simulation_seed"], how="left", validate="many_to_one")
    if paired["P0_regret"].isna().any():
        raise ValueError("Some scenarios do not match the P0 reference")
    paired["regret_change_vs_P0"] = paired["expected_regret_B1"] - paired["P0_regret"]
    paired["heldout_elpd_change_vs_P0"] = paired["heldout_elpd_B1"] - paired["P0_heldout_elpd"]
    paired["brier_change_vs_P0"] = paired["brier_B1"] - paired["P0_brier"]
    paired["rmse_true_p_change_vs_P0"] = paired["rmse_vs_true_p_B1"] - paired["P0_rmse_true_p"]
    paired["diagnostic_warning"] = (
        paired["rhat_max"].astype(float).gt(1.01)
        | paired["ess_bulk_min"].astype(float).lt(400)
        | paired["ess_tail_min"].astype(float).lt(400)
        | paired["divergences"].astype(int).gt(0)
    )
    return paired.sort_values(
        ["gamma_true", "simulation_seed", "prior_id"], ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across five outcome realizations without claiming new geographies."""
    rows = []
    for (gamma, prior_id), sub in paired.groupby(["gamma_true", "prior_id"], sort=True):
        r = sub["regret_change_vs_P0"].astype(float)
        rows.append({
            "gamma_true": float(gamma),
            "prior_id": prior_id,
            "replications": len(sub),
            "alpha_prior_mean": float(sub["alpha_prior_mean"].iloc[0]),
            "alpha_prior_sigma": float(sub["alpha_prior_sigma"].iloc[0]),
            "mean_alpha_posterior": float(sub["alpha_posterior_mean"].mean()),
            "mean_alpha_bias_vs_generator_minus3": float((sub["alpha_posterior_mean"] + 3.0).mean()),
            "alpha_coverage_count": int(sub["alpha_covered"].astype(bool).sum()),
            "mean_gamma_posterior": float(sub["gamma_posterior_mean"].mean()),
            "gamma_coverage_count": int(sub["gamma_covered"].astype(bool).sum()),
            "mean_expected_regret_B0": float(sub["expected_regret_B0"].mean()),
            "mean_expected_regret_B1": float(sub["expected_regret_B1"].mean()),
            "mean_regret_change_vs_P0": float(r.mean()),
            "prior_better_than_P0": int((r < -THRESHOLD).sum()),
            "prior_tied_with_P0": int((r.abs() <= THRESHOLD).sum()),
            "prior_worse_than_P0": int((r > THRESHOLD).sum()),
            "mean_topk_overlap_with_P0": float(sub["overlap_B1_P0"].mean()),
            "max_inclusion_probability_shift": float(sub["max_inclusion_change_B1_P0"].max()),
            "mean_heldout_elpd_change_vs_P0": float(sub["heldout_elpd_change_vs_P0"].mean()),
            "mean_brier_change_vs_P0": float(sub["brier_change_vs_P0"].mean()),
            "mean_rmse_true_p_change_vs_P0": float(sub["rmse_true_p_change_vs_P0"].mean()),
            "mean_ppc_event_rate_p50": float(sub["posterior_predictive_rate_p50_B1"].mean()),
            "mean_train_event_rate": float(sub["train_damage_rate"].mean()),
            "diagnostic_flag_count": int(sub["diagnostic_warning"].sum()),
        })
    return pd.DataFrame(rows).sort_values(
        ["gamma_true", "prior_id"], ascending=[False, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stage = "smoke" if args.smoke else "full"
    folder = DATA_ROOT / stage
    path = folder / "comparison_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run `python -m rhb.pipelines.run_m1d2{' --smoke' if args.smoke else ''}` first")
    df = pd.read_csv(path)
    paired = compare_to_reference(df)
    result = summarize(paired)
    output = folder / "summary"
    output.mkdir(parents=True, exist_ok=True)
    paired_path = output / "paired_cases.csv"
    summary_path = output / "summary_by_gamma_prior.csv"
    flags_path = output / "diagnostic_flags.csv"
    report_path = output / "summary_report.md"
    paired.to_csv(paired_path, index=False)
    result.to_csv(summary_path, index=False)
    paired.loc[paired["diagnostic_warning"], [
        "gamma_true", "simulation_seed", "prior_id", "rhat_max", "ess_bulk_min",
        "ess_tail_min", "divergences",
    ]].to_csv(flags_path, index=False)
    report_path.write_text(
        "# M1-D2 automated results\n\n"
        "Synthetic sensitivity only. True generating intercept = -3, known to the experimenter. "
        "P1/P2 are oracle-centred and cannot be treated as independently elicited priors. "
        "This report does not validate real flood-risk predictions.\n\n"
        + "```text\n" + result.round(6).to_string(index=False) + "\n```\n\n"
        + f"Scenarios: {len(df.groupby(['gamma_true', 'simulation_seed']))}; "
        + f"prior comparisons: {len(df)}; diagnostic flags: {int(paired['diagnostic_warning'].sum())}.\n",
        encoding="utf-8",
    )
    (output / "summary_manifest.json").write_text(json.dumps({
        "source_comparisons_sha256": sha256_file(path),
        "paired_cases_sha256": sha256_file(paired_path),
        "summary_by_gamma_prior_sha256": sha256_file(summary_path),
        "diagnostic_flags_sha256": sha256_file(flags_path),
    }, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print(f"M1-D2 {stage.upper()} SUMMARY COMPLETE: {len(paired)} comparisons, "
          f"{int(paired['diagnostic_warning'].sum())} flags")


if __name__ == "__main__":
    main()
