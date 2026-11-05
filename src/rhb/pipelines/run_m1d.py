"""M1-D1 prior-width sensitivity for the M1-C Rotterdam synthetic experiment.

Run from the repository root (Python 3.11 environment):
    python -m rhb.pipelines.run_m1d --smoke --dry-run
    python -m rhb.pipelines.run_m1d --smoke
    python -m rhb.pipelines.run_m1d --dry-run
    python -m rhb.pipelines.run_m1d

Reuses the frozen M1-C B0 and sigma=0.5 B1 fits. Only new B1 priors
(sigma=0.25 and 1.0) are fitted. No changes to M1-C or Phase 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rhb.decision.m1c_metrics import compare_decisions
from rhb.pipelines.run_m1c import (
    DATA_DIR as M1C_DIR,
    ROOT,
    git_head,
    label,
    read_inputs,
    run_fit,
    sha256,
)

LOG = logging.getLogger("rhb.m1d")
CONFIG_PATH = ROOT / "configs/m1d_gamma_prior_rtm.json"
OUTPUT_ROOT = ROOT / "outputs/phase4/RTM/m1d"


def validate_config(config: dict[str, Any]) -> list[float]:
    """Check that M1-C's original prior is a registered, unchanged reference."""
    if config.get("experiment") != "M1-D1":
        raise ValueError("Unexpected experiment identifier")
    values = [float(value) for value in config["gamma_prior_sigmas"]]
    baseline = float(config["baseline_gamma_prior_sigma"])
    if not values or len(values) != len(set(values)):
        raise ValueError("Expected distinct gamma-prior widths")
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Gamma prior widths must be finite and positive")
    if baseline not in values:
        raise ValueError("Baseline prior width is missing")
    if config.get("city") != "RTM":
        raise ValueError("Only the fixed Rotterdam M1-C dataset is supported")
    return values


def sigma_label(sigma: float) -> str:
    """Stable directory name for a gamma prior standard deviation."""
    return f"sigma_{sigma:g}".replace(".", "p")


def reference_fit(case_dir: Path, name: str, fingerprints: dict[str, str]) -> tuple[dict, pd.DataFrame]:
    """Read and authenticate one existing M1-C fit (never change it)."""
    fit_dir = case_dir / name
    record_path = fit_dir / "fit_record.json"
    decisions_path = fit_dir / "test_decisions.parquet"
    posterior_path = fit_dir / "idata.nc"
    for required in (record_path, decisions_path, posterior_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing M1-C baseline artifact: {required}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("fingerprints") != fingerprints:
        raise ValueError(f"M1-C baseline fingerprints do not match: {fit_dir}")
    return record, pd.read_parquet(decisions_path)


def make_fingerprints(
    source_fingerprints: dict[str, str], sigma: float,
) -> dict[str, str]:
    """Include M1-D configuration, code and chosen prior in cache identity."""
    result = dict(source_fingerprints)
    result["m1d_gamma_prior_sigma"] = f"{sigma:.17g}"
    result["m1d_config_sha256"] = sha256(CONFIG_PATH)
    result["m1d_runner_sha256"] = sha256(Path(__file__))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="First outcome seed only")
    parser.add_argument("--dry-run", action="store_true", help="Validate reference fits; list planned work")
    parser.add_argument("--force", action="store_true", help="Recompute only M1-D nonbaseline fits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    d_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    sigmas = validate_config(d_config)
    baseline_sigma = float(d_config["baseline_gamma_prior_sigma"])
    c_config, scenarios, source_hashes = read_inputs()
    if float(c_config["gamma_prior_sigma"]) != baseline_sigma:
        raise ValueError("M1-C's frozen baseline gamma prior does not match M1-D")

    stage = "smoke" if args.smoke else "full"
    gamma_values = [float(value) for value in c_config["gamma_true_values"]]
    seeds = [int(value) for value in c_config["simulation_seeds"]]
    if args.smoke:
        seeds = seeds[:1]
    n_test = int(c_config["test_size"])
    k = max(1, int(round(n_test * float(c_config["selection_share"]))))
    n_new = len(gamma_values) * len(seeds) * sum(value != baseline_sigma for value in sigmas)
    LOG.info("M1-D1 %s: %d cases, %d prior widths, %d new B1 fits, top-k=%d",
             stage, len(gamma_values) * len(seeds), len(sigmas), n_new, k)

    source_stage = M1C_DIR / stage / "fits"
    out = OUTPUT_ROOT / stage
    # Run even for --dry-run: confirm M1-C reference provenance and artifacts.
    for gamma in gamma_values:
        for seed in seeds:
            reference = source_stage / label(gamma, seed)
            reference_fit(reference, "B0", source_hashes)
            reference_fit(reference, "B1", source_hashes)
            if args.dry_run:
                LOG.info("DRY RUN gamma=%s seed=%s gamma-prior sigmas=%s (reuse sigma=%g)",
                         gamma, seed, sigmas, baseline_sigma)
    if args.dry_run:
        return

    records: list[dict[str, Any]] = []
    for gamma in gamma_values:
        for seed in seeds:
            case_label = label(gamma, seed)
            reference = source_stage / case_label
            record_b0, decisions_b0 = reference_fit(reference, "B0", source_hashes)
            record_baseline, decisions_baseline = reference_fit(reference, "B1", source_hashes)
            group = scenarios.loc[
                np.isclose(scenarios["gamma_true"], gamma, rtol=0, atol=1e-12)
                & scenarios["simulation_seed"].eq(seed)
            ].reset_index(drop=True)
            if len(group) != int(c_config["sample_size"]):
                raise ValueError(f"Missing synthetic case: {case_label}")
            for sigma in sigmas:
                if sigma == baseline_sigma:
                    record_b1, decisions_b1 = record_baseline, decisions_baseline
                    source = "M1-C reference B1"
                else:
                    patched_config = dict(c_config)
                    patched_config["gamma_prior_sigma"] = sigma
                    fit_dir = out / "fits" / case_label / sigma_label(sigma) / "B1"
                    record_b1 = run_fit(
                        group, patched_config, make_fingerprints(source_hashes, sigma),
                        gamma, seed, "B1", fit_dir, k, args.force,
                    )
                    decisions_b1 = pd.read_parquet(fit_dir / "test_decisions.parquet")
                    source = "M1-D1 new fit"
                cmp_b0 = compare_decisions(decisions_b0, decisions_b1, k=k)
                cmp_baseline = compare_decisions(decisions_baseline, decisions_b1, k=k)
                row: dict[str, Any] = {
                    "gamma_true": gamma,
                    "simulation_seed": seed,
                    "gamma_prior_sigma": sigma,
                    "fit_source": source,
                    "gamma_posterior_mean": record_b1["gamma_posterior_mean"],
                    "gamma_hdi_low": record_b1["gamma_hdi_low"],
                    "gamma_hdi_high": record_b1["gamma_hdi_high"],
                    "gamma_covered": record_b1["gamma_covered"],
                    "rhat_max": record_b1["rhat_max"],
                    "ess_bulk_min": record_b1["ess_bulk_min"],
                    "ess_tail_min": record_b1["ess_tail_min"],
                    "divergences": record_b1["divergences"],
                    "prior_predictive_rate_p50_B1": record_b1["prior_predictive_rate_p50"],
                    "prior_predictive_rate_p95_B1": record_b1["prior_predictive_rate_p95"],
                    "posterior_predictive_rate_p50_B1": record_b1["posterior_predictive_rate_p50"],
                    "train_damage_rate": record_b1["train_damage_rate"],
                    "rmse_vs_true_p_B0": record_b0["rmse_vs_true_p"],
                    "rmse_vs_true_p_B1": record_b1["rmse_vs_true_p"],
                    "brier_B0": record_b0["brier"],
                    "brier_B1": record_b1["brier"],
                    "heldout_elpd_B0": record_b0["heldout_elpd"],
                    "heldout_elpd_B1": record_b1["heldout_elpd"],
                    **cmp_b0,
                    "overlap_B1_baseline": cmp_baseline["overlap"],
                    "max_inclusion_change_B1_baseline": cmp_baseline["max_inclusion_change"],
                    "mean_abs_inclusion_change_B1_baseline": cmp_baseline["mean_abs_inclusion_change"],
                    "regret_shift_B1_vs_baseline": (
                        float(cmp_b0["expected_regret_B1"])
                        - float(record_baseline["expected_regret"])
                    ),
                }
                if abs(row["expected_regret_B0"] - float(record_b0["expected_regret"])) > 1e-9:
                    raise AssertionError("B0 expected regret does not match authenticated baseline")
                records.append(row)
                LOG.info("gamma=%g seed=%s sigma=%g overlap B1/base=%d/%d regret B1=%.5g",
                         gamma, seed, sigma, row["overlap_B1_baseline"], k,
                         row["expected_regret_B1"])

    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records).sort_values(
        ["gamma_true", "simulation_seed", "gamma_prior_sigma"],
        ascending=[False, True, True],
    )
    table.to_csv(out / "comparison_results.csv", index=False)
    version_names = ("pymc", "arviz", "numpy", "pandas", "scipy")
    manifest = {
        "experiment": "M1-D1",
        "stage": stage,
        "reused_m1c_baseline": True,
        "new_B1_fits": n_new,
        "comparison_rows": len(table),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "dependencies": {name: metadata.version(name) for name in version_names},
        "source_m1c_hashes": source_hashes,
        "m1d_config_sha256": sha256(CONFIG_PATH),
        "m1d_runner_sha256": sha256(Path(__file__)),
        "m1d_config": d_config,
        "m1c_config": c_config,
        "note": "Synthetic sensitivity study only; not empirical flood-risk validation.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("M1-D1 %s COMPLETE: %d comparisons, %d new Bayesian fits, %s",
             stage.upper(), len(table), n_new, out)


if __name__ == "__main__":
    main()
