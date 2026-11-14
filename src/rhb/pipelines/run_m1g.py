"""M1-G: refit frozen M1-C B0/B1 for HAM/DON synthetic portability screen.

    python -m rhb.pipelines.run_m1g --city HAM --smoke --dry-run
    python -m rhb.pipelines.run_m1g --city HAM --smoke
    python -m rhb.pipelines.run_m1g --city HAM

M1-G changes only the city-specific covariate/terrain distribution. It does
not copy Phase 3 posterior probabilities or claim real-outcome transfer.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rhb.experiments.m1f_population_sampling import sha256, validate_scenarios
from rhb.experiments.m1g_cross_city_sampling import (
    ROOT, check_python, generate_city_scenarios, load_config,
    load_universe, output_dir, select_population, source_fingerprints,
)

LOGGER = logging.getLogger("rhb.m1g.run")


def read_registered(city: str) -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    """Fail closed if frozen scenarios, population, code or input data changed."""
    cfg = load_config()
    out = output_dir(city)
    path = out / "m1g_scenarios.parquet"
    sample_path = out / "population_ids.csv"
    snapshot = out / "sampling_manifest.json"
    for file in (path, sample_path, snapshot):
        if not file.is_file():
            raise FileNotFoundError(f"{city}: missing M1-G preparation artifact: {file}")
    manifest = json.loads(snapshot.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "M1-G" or manifest.get("city") != city or manifest.get("config") != cfg:
        raise ValueError(f"{city}: stale or mismatched M1-G sampling manifest")
    if manifest.get("output_hashes") != {"scenarios": sha256(path), "population_ids": sha256(sample_path)}:
        raise ValueError(f"{city}: sampled output hashes changed")
    # Re-read the complete city inputs and check their current SHA-256 values.
    universe, inputs = load_universe(city, cfg)
    hashes = source_fingerprints(city, cfg, inputs)
    if hashes != manifest.get("input_fingerprints"):
        raise ValueError(f"{city}: code/config/source changed after M1-G preparation")
    expected_pop = select_population(universe, city, cfg)
    expected_cases = generate_city_scenarios(expected_pop, city, cfg)
    scenarios = pd.read_parquet(path)
    pd.testing.assert_frame_equal(
        scenarios.reset_index(drop=True), expected_cases.reset_index(drop=True),
        check_dtype=True, check_exact=True,
    )
    saved_ids = pd.read_csv(sample_path, dtype={"bldg_id": str})
    expected_ids = expected_pop[["population_id", "bldg_id", "split"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(saved_ids, expected_ids, check_dtype=False)
    return cfg, scenarios, {
        **hashes,
        "scenarios_sha256": sha256(path),
        "population_ids_sha256": sha256(sample_path),
        "sampling_manifest_sha256": sha256(snapshot),
    }


def cases_for(cfg: dict[str, Any], smoke: bool) -> list[tuple[float, int]]:
    seeds = cfg["simulation_seeds"][:1] if smoke else cfg["simulation_seeds"]
    return [(float(gamma), int(seed)) for gamma in cfg["gamma_true_values"] for seed in seeds]


def git_head() -> str | None:
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    return run.stdout.strip() if run.returncode == 0 else None


def run_city(city: str, smoke: bool, dry_run: bool) -> None:
    check_python()
    cfg, scenarios, hashes = read_registered(city)
    cases = cases_for(cfg, smoke)
    k = int(round(cfg["test_size"] * cfg["selection_share"]))
    if k != 10:
        raise ValueError("M1-G selection must be top 10 of 1000")
    stage = "smoke" if smoke else "full"
    out = output_dir(city)
    prior_manifest_path = out / f"{stage}_manifest.json"
    if prior_manifest_path.exists():
        prior = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        fit_path = out / f"{stage}_fit_results.csv"
        comp_path = out / f"{stage}_comparisons.csv"
        if prior.get("input_code_fingerprints") != hashes:
            raise ValueError(f"{city}: existing {stage} run is from different inputs/code")
        if not fit_path.is_file() or not comp_path.is_file() or prior.get("output_hashes") != {
            "fit_results": sha256(fit_path), "comparisons": sha256(comp_path),
        }:
            raise ValueError(f"{city}: existing {stage} output was changed; refusing overwrite")
        LOGGER.info("%s %s verified existing complete run; no overwrite", city, stage)
        return
    LOGGER.info("M1-G %s %s: %d matched cases / %d model fits / top k=%d",
                city, stage, len(cases), 2 * len(cases), k)
    if dry_run:
        for gamma, seed in cases:
            LOGGER.info("DRY RUN city=%s gamma=%g seed=%s models=B0,B1", city, gamma, seed)
        return
    # Crucial: use the identical, already-tested Rotterdam M1-C model/fitting/decision code.
    from rhb.decision.m1c_metrics import compare_decisions
    from rhb.pipelines.run_m1c import label, run_fit

    m1c_config = {"gamma_prior_sigma": cfg["gamma_prior_sigma"], "mcmc": cfg["mcmc"]}
    fits: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    out = output_dir(city)
    for gamma, seed in cases:
        frame = scenarios.loc[
            np.isclose(scenarios["gamma_true"], gamma, atol=1e-12, rtol=0)
            & scenarios["simulation_seed"].eq(seed)
        ].reset_index(drop=True)
        if len(frame) != cfg["sample_size"] or not frame["bldg_id"].is_unique:
            raise ValueError(f"{city}: broken case gamma={gamma} seed={seed}")
        for_fit = frame.rename(columns={"Y_m1g": "Y_m1c"})
        case_dir = out / "fits" / label(gamma, seed)
        decisions = {}
        for model in ("B0", "B1"):
            fit_dir = case_dir / model
            result = run_fit(
                for_fit, m1c_config, hashes, gamma, seed,
                model, fit_dir, k=k, force=False,
            )
            fits.append({"city": city, "population_id": "city01", **{
                key: val for key, val in result.items() if key != "fingerprints"
            }})
            decisions[model] = pd.read_parquet(fit_dir / "test_decisions.parquet")
        comparison = compare_decisions(decisions["B0"], decisions["B1"], k=k)
        comparisons.append({"city": city, "population_id": "city01",
                            "gamma_true": gamma, "simulation_seed": seed, **comparison})
        LOGGER.info("%s gamma=%g seed=%d overlap=%d/%d regret B0=%.5f B1=%.5f",
                    city, gamma, seed, comparison["overlap"], k,
                    comparison["expected_regret_B0"], comparison["expected_regret_B1"])
    fit_path = out / f"{stage}_fit_results.csv"
    comparison_path = out / f"{stage}_comparisons.csv"
    pd.DataFrame(fits).to_csv(fit_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    versions = {pkg: metadata.version(pkg) for pkg in ("pymc", "arviz", "numpy", "pandas", "scipy")}
    manifest = {
        "experiment": "M1-G", "city": city, "stage": stage,
        "outcome_evidence": "synthetic", "run_kind": "retrained_per_city",
        "case_count": len(cases), "model_fits": len(fits), "k": k,
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "git_head": git_head(), "dependencies": versions,
        "configuration": cfg, "input_code_fingerprints": hashes,
        "output_hashes": {"fit_results": sha256(fit_path), "comparisons": sha256(comparison_path)},
        "note": "B0/B1 fitted independently within each city on matched synthetic outcomes. No empirical flood prediction, geographic transfer of posterior coefficients or observed loss evidence.",
    }
    (out / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    LOGGER.info("M1-G %s %s INFERENCE COMPLETE: %d cases, %d fits -> %s",
                city, stage.upper(), len(cases), len(fits), out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("HAM", "DON", "all"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check frozen inputs and list fits, no inference")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for city in (("HAM", "DON") if args.city == "all" else (args.city,)):
        run_city(city, smoke=args.smoke, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
