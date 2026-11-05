r"""M1-F: refit frozen M1-C B0/B1 on disjoint new Rotterdam building samples.

Run in order:
  .\.venv\Scripts\python.exe -m rhb.experiments.m1f_population_sampling
  .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --smoke --dry-run
  .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --smoke
  .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f

Reuses M1-C's tested fitting and decision functions, *not* its 5,000 assets.
Checkpoints are shared by smoke and full modes, and generated outputs are ignored by Git.
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

from rhb.experiments.m1f_population_sampling import (
    CONFIG_PATH, OUTPUT_DIR, ROOT, check_python, input_fingerprints,
    load_reference_ids, sha256, validate_config, validate_scenarios,
)

LOG = logging.getLogger("rhb.m1f.run")


def read_registered_inputs() -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    """Authenticate generated scenario and original source checksums."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    validate_config(config)
    manifest_path = OUTPUT_DIR / "sampling_manifest.json"
    scenario_path = OUTPUT_DIR / "m1f_scenarios.parquet"
    populations_path = OUTPUT_DIR / "population_ids.csv"
    for path in (manifest_path, scenario_path, populations_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required M1-F sampling artifact missing: {path}. "
                "Run python -m rhb.experiments.m1f_population_sampling first."
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "M1-F" or manifest.get("stage") != "sampling":
        raise ValueError("Unexpected M1-F sampling manifest")
    if manifest.get("config") != config:
        raise ValueError("M1-F configuration changed after population generation")
    fingerprints = input_fingerprints(config)
    if manifest.get("input_fingerprints") != fingerprints:
        raise ValueError("M1-F sampling inputs or generator changed. Do not reuse stale scenarios")
    if manifest.get("output_sha256") != {
        "scenarios": sha256(scenario_path), "population_ids": sha256(populations_path)
    }:
        raise ValueError("Generated M1-F artifacts changed since sampling")
    scenarios = pd.read_parquet(scenario_path)
    excluded = load_reference_ids((ROOT / config["exclude_reference_path"]).resolve(),
                                  int(config["expected_reference_sample_size"]))
    validate_scenarios(scenarios, config, excluded)
    ids = pd.read_csv(populations_path)
    expected = (scenarios.loc[(scenarios["gamma_true"] == float(config["gamma_true_values"][0]))
                              & (scenarios["simulation_seed"] == int(config["simulation_seeds"][0])),
                              ["population_id", "bldg_id", "split"]].reset_index(drop=True))
    if not ids.equals(expected):
        raise ValueError("Population ID and split inventory is inconsistent with scenarios")
    extra_paths = [
        ROOT / "src/rhb/pipelines/run_m1f.py",
        ROOT / "src/rhb/pipelines/run_m1c.py",
        ROOT / "src/rhb/decision/m1c_metrics.py",
        ROOT / "src/rhb/decision/phase3_decision_metrics.py",
        ROOT / "src/rhb/models/logistic_m1.py",
        ROOT / "src/rhb/models/logistic_phase3.py",
    ]
    fingerprints = {
        **fingerprints,
        "sampling_manifest_sha256": sha256(manifest_path),
        "m1f_scenarios_sha256": sha256(scenario_path),
        "population_ids_sha256": sha256(populations_path),
        **{f"code_sha256:{p.relative_to(ROOT).as_posix()}": sha256(p) for p in extra_paths},
    }
    return config, scenarios, fingerprints


def cases_for(config: dict[str, Any], smoke: bool) -> list[tuple[str, float, int]]:
    pops = config["population_ids"][:1] if smoke else config["population_ids"]
    seeds = config["simulation_seeds"][:1] if smoke else config["simulation_seeds"]
    return [(str(pop), float(gamma), int(seed)) for pop in pops
            for gamma in config["gamma_true_values"] for seed in seeds]


def git_head() -> str | None:
    run = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=False)
    return run.stdout.strip() if run.returncode == 0 else None


def main() -> None:
    check_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One population, both effects, one outcome seed; four fits")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report fit plan, no inference")
    parser.add_argument("--force", action="store_true", help="Recompute existing fits after explicit review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg, scenarios, hashes = read_registered_inputs()
    cases = cases_for(cfg, args.smoke)
    k = int(round(int(cfg["test_size"]) * float(cfg["selection_share"])))
    if k != 10:
        raise ValueError("M1-F registered decision budget must be k=10")
    stage = "smoke" if args.smoke else "full"
    LOG.info("M1-F %s: %d cases, %d B0/B1 fits, top-k=%d",
             stage.upper(), len(cases), 2 * len(cases), k)
    if args.dry_run:
        for pop, gamma, seed in cases:
            LOG.info("DRY RUN pop=%s gamma=%g seed=%d models=B0,B1", pop, gamma, seed)
        return

    # Deliberately import PyMC-dependent M1-C pipeline only for actual inference.
    from rhb.decision.m1c_metrics import compare_decisions
    from rhb.pipelines.run_m1c import label, run_fit

    model_config = {"gamma_prior_sigma": cfg["gamma_prior_sigma"], "mcmc": cfg["mcmc"]}
    fit_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for pop, gamma, seed in cases:
        group = scenarios.loc[
            scenarios["population_id"].eq(pop)
            & np.isclose(scenarios["gamma_true"], gamma, atol=1e-12, rtol=0)
            & scenarios["simulation_seed"].eq(seed)
        ].reset_index(drop=True)
        if len(group) != int(cfg["sample_size"]):
            raise ValueError(f"Incomplete registered case {pop}, {gamma}, {seed}")
        # Adapter: M1-C's validated inference implementation expects Y_m1c.
        # No data transfer, population sampling or generator code is reused from M1-C.
        for_fit = group.rename(columns={"Y_m1f": "Y_m1c"})
        case_dir = OUTPUT_DIR / "fits" / pop / label(gamma, seed)
        decisions = {}
        for model in ("B0", "B1"):
            fit_dir = case_dir / model
            result = run_fit(
                for_fit, model_config, hashes, gamma, seed, model,
                fit_dir, k=k, force=args.force,
            )
            fit_rows.append({"population_id": pop, **{
                name: value for name, value in result.items() if name != "fingerprints"
            }})
            decisions[model] = pd.read_parquet(fit_dir / "test_decisions.parquet")
        comparison = compare_decisions(decisions["B0"], decisions["B1"], k=k)
        comparison_rows.append({"population_id": pop, "gamma_true": gamma,
                                "simulation_seed": seed, **comparison})
        LOG.info("CASE %s gamma=%g seed=%d overlap=%d/%d regret B0=%.5g B1=%.5g",
                 pop, gamma, seed, comparison["overlap"], k,
                 comparison["expected_regret_B0"], comparison["expected_regret_B1"])

    fit_path = OUTPUT_DIR / f"{stage}_fit_results.csv"
    comparison_path = OUTPUT_DIR / f"{stage}_comparisons.csv"
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    versions = {name: metadata.version(name) for name in ("pymc", "arviz", "numpy", "pandas", "scipy")}
    manifest = {
        "experiment": "M1-F", "stage": stage, "outcome_evidence": "synthetic",
        "populations": list(cfg["population_ids"][:1] if args.smoke else cfg["population_ids"]),
        "case_count": len(cases), "model_fits": len(fit_rows),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "dependencies": versions, "git_head": git_head(),
        "input_code_fingerprints": hashes, "config": cfg,
        "output_hashes": {"fit_results": sha256(fit_path), "comparisons": sha256(comparison_path)},
        "note": "Fixed synthetic generator; three disjoint RTM samples excluding the M1-C population. Not spatial generalization or real flood validation.",
    }
    (OUTPUT_DIR / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    LOG.info("M1-F %s COMPLETE: %d cases, %d Bayesian fits -> %s",
             stage.upper(), len(cases), len(fit_rows), OUTPUT_DIR)


if __name__ == "__main__":
    main()
