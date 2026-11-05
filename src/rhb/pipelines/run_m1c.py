"""M1-C: paired synthetic experiments on fixed Rotterdam assets.

Examples (from the repository root):
    python -m rhb.pipelines.run_m1c --smoke --dry-run
    python -m rhb.pipelines.run_m1c --smoke
    python -m rhb.pipelines.run_m1c
    python -m rhb.pipelines.run_m1c --smoke --force

Completed fits are cached with configuration, data, and code fingerprints.
Do not edit Phase 3 or M1-A/B artifacts. Only writes under outputs/phase4/RTM/m1c/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit

from rhb.decision.m1c_metrics import compare_decisions, decision_metrics, predictive_scores
from rhb.models.logistic_m1 import build_m1_model
from rhb.models.logistic_phase3 import build_phase3_model

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/m1c_rtm.json"
DATA_DIR = ROOT / "outputs/phase4/RTM/m1c"
LOG = logging.getLogger("rhb.m1c")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_inputs() -> tuple[dict[str, Any], pd.DataFrame, dict[str, str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    path = DATA_DIR / "m1c_scenarios.parquet"
    snapshot = DATA_DIR / "experiment_config.json"
    if not path.exists() or not snapshot.exists():
        raise FileNotFoundError(
            "Run `python -m rhb.experiments.m1c_simulation` first."
        )
    if json.loads(snapshot.read_text(encoding="utf-8-sig")) != config:
        raise ValueError("Configuration changed after scenario generation. Regenerate scenarios.")
    df = pd.read_parquet(path)
    required = {
        "bldg_id", "E_hat_v0", "H_pluvial_v1_logrel", "T", "p_true",
        "Y_m1c", "gamma_true", "simulation_seed", "split",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if len(df) != config["sample_size"] * len(config["gamma_true_values"]) * len(config["simulation_seeds"]):
        raise ValueError("Unexpected number of scenario rows")

    for (gamma, seed), group in df.groupby(["gamma_true", "simulation_seed"]):
        if len(group) != config["sample_size"] or not group["bldg_id"].is_unique:
            raise ValueError(f"Invalid or duplicated population for gamma={gamma}, seed={seed}")
        n_train = int(group["split"].eq("train").sum())
        if n_train != config["train_size"]:
            raise ValueError("Wrong training population size")
        e = group["E_hat_v0"].to_numpy(dtype=float)
        h = group["H_pluvial_v1_logrel"].to_numpy(dtype=float)
        t = group["T"].to_numpy(dtype=float)
        expected_p = expit(-3.0 + e + h + float(gamma) * t)
        if not np.allclose(expected_p, group["p_true"], rtol=1e-12, atol=1e-12):
            raise ValueError("Synthetic probability does not match registered generator")
        if not np.all(np.isin(group["Y_m1c"].to_numpy(), [0, 1])):
            raise ValueError("Nonbinary simulated outcomes")

    hashes = {
        "config_sha256": sha256(CONFIG_PATH),
        "scenarios_sha256": sha256(path),
    }
    source_paths = [
        ROOT / "src/rhb/pipelines/run_m1c.py",
        ROOT / "src/rhb/decision/m1c_metrics.py",
        ROOT / "src/rhb/decision/phase3_decision_metrics.py",
        ROOT / "src/rhb/experiments/m1c_simulation.py",
        ROOT / "src/rhb/models/logistic_m1.py",
        ROOT / "src/rhb/models/logistic_phase3.py",
    ]
    for source in source_paths:
        hashes[f"sha256:{source.relative_to(ROOT).as_posix()}"] = sha256(source)
    return config, df, hashes


def seed_for(gamma: float, seed: int, name: str) -> int:
    """Sampler seed is stable regardless of scenario traversal order."""
    payload = f"M1-C-v0.1|{gamma:g}|{seed}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def label(gamma: float, seed: int) -> str:
    sign = "neg" if gamma < 0 else "pos"
    value = f"{abs(gamma):g}".replace(".", "p")
    return f"gamma_{sign}{value}_seed_{seed}"


def posterior_probabilities(idata: az.InferenceData, df: pd.DataFrame, extended: bool) -> np.ndarray:
    """Predict held-out probabilities from parameter draws, without test-label leakage."""
    e = df["E_hat_v0"].to_numpy(dtype=float)
    h = df["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    t = df["T"].to_numpy(dtype=float)
    posterior = idata.posterior

    def draws(name: str) -> np.ndarray:
        return posterior[name].values.reshape(-1, 1)

    logits = draws("alpha") + draws("beta_E") * e[None, :] + draws("beta_H") * h[None, :]
    if extended:
        logits += draws("gamma_T") * t[None, :]
    return expit(logits)


def run_fit(
    group: pd.DataFrame,
    config: dict[str, Any],
    hashes: dict[str, str],
    gamma: float,
    simulation_seed: int,
    name: str,
    fit_dir: Path,
    k: int,
    force: bool,
) -> dict[str, Any]:
    """Fit one model, evaluate held-out decisions, checkpoint its results."""
    fit_dir.mkdir(parents=True, exist_ok=True)
    record_path = fit_dir / "fit_record.json"
    idata_path = fit_dir / "idata.nc"
    decision_path = fit_dir / "test_decisions.parquet"
    if record_path.exists() and not force:
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("fingerprints") != hashes:
            raise ValueError(f"Stale cached fit: {fit_dir}. Use --force only after reviewing inputs.")
        if not idata_path.exists() or not decision_path.exists():
            raise ValueError(f"Incomplete cached fit: {fit_dir}. Use --force to rerun.")
        LOG.info("Reusing %s", fit_dir)
        return prior

    train = group.loc[group["split"].eq("train")].reset_index(drop=True)
    test = group.loc[group["split"].eq("test")].reset_index(drop=True)
    e = train["E_hat_v0"].to_numpy(dtype=float)
    h = train["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    t = train["T"].to_numpy(dtype=float)
    y = train["Y_m1c"].to_numpy(dtype=int)
    extended = name == "B1"

    if extended:
        model = build_m1_model(e, h, t, y, gamma_sigma=float(config["gamma_prior_sigma"]))
    else:
        model = build_phase3_model(e, h, y)

    sampler_seed = seed_for(gamma, simulation_seed, name)
    mcmc = config["mcmc"]
    LOG.info("Fitting gamma=%s simulation=%s model=%s", gamma, simulation_seed, name)
    with model:
        prior = pm.sample_prior_predictive(samples=100, random_seed=sampler_seed)
        idata = pm.sample(
            draws=int(mcmc["draws"]), tune=int(mcmc["tune"]),
            chains=int(mcmc["chains"]), cores=min(2, int(mcmc["chains"])),
            target_accept=float(mcmc["target_accept"]),
            random_seed=sampler_seed, progressbar=False,
        )
        ppc = pm.sample_posterior_predictive(
            idata, var_names=["Y_obs"], random_seed=sampler_seed,
            progressbar=False,
        )

    var_names = ["alpha", "beta_E", "beta_H"] + (["gamma_T"] if extended else [])
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.95)
    divergences = int(idata.sample_stats["diverging"].sum().item())
    prior_rates = prior.prior_predictive["Y_obs"].values.reshape(-1, len(train)).mean(axis=1)
    posterior_rates = ppc.posterior_predictive["Y_obs"].values.reshape(-1, len(train)).mean(axis=1)
    probabilities = posterior_probabilities(idata, test, extended)
    scores = predictive_scores(
        probabilities,
        test["Y_m1c"].to_numpy(dtype=int),
        test["p_true"].to_numpy(dtype=float),
    )
    decisions, dmetrics = decision_metrics(
        probabilities,
        test["bldg_id"].to_numpy(),
        test["p_true"].to_numpy(dtype=float),
        k=k,
    )

    record: dict[str, Any] = {
        "gamma_true": float(gamma),
        "simulation_seed": int(simulation_seed),
        "model": name,
        "sampler_seed": int(sampler_seed),
        "train_size": len(train),
        "test_size": len(test),
        "train_damage_rate": float(y.mean()),
        "test_damage_rate": float(test["Y_m1c"].mean()),
        "divergences": divergences,
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "prior_predictive_rate_p05": float(np.percentile(prior_rates, 5)),
        "prior_predictive_rate_p50": float(np.percentile(prior_rates, 50)),
        "prior_predictive_rate_p95": float(np.percentile(prior_rates, 95)),
        "posterior_predictive_rate_p05": float(np.percentile(posterior_rates, 5)),
        "posterior_predictive_rate_p50": float(np.percentile(posterior_rates, 50)),
        "posterior_predictive_rate_p95": float(np.percentile(posterior_rates, 95)),
        "gamma_posterior_mean": float(summary.loc["gamma_T", "mean"]) if extended else None,
        "gamma_hdi_low": float(summary.loc["gamma_T", "hdi_2.5%"]) if extended else None,
        "gamma_hdi_high": float(summary.loc["gamma_T", "hdi_97.5%"]) if extended else None,
        "gamma_covered": (
            bool(summary.loc["gamma_T", "hdi_2.5%"] <= gamma <= summary.loc["gamma_T", "hdi_97.5%"])
            if extended else None
        ),
        "fingerprints": hashes,
        **scores,
        **dmetrics,
    }
    # Persist model-specific results. Posterior predictive replicas are summarized,
    # not written to NetCDF (this avoids 30 huge observed-variable arrays).
    az.to_netcdf(idata, idata_path)
    summary.to_csv(fit_dir / "posterior_summary.csv")
    decisions.to_parquet(decision_path, index=False)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    LOG.info("Completed %s: regret=%.5f gamma_mean=%s", fit_dir, record["expected_regret"], record["gamma_posterior_mean"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One simulation seed, all three gamma values (six fits)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list scenarios without fitting")
    parser.add_argument("--force", action="store_true", help="Recompute existing fit checkpoints")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config, scenarios, hashes = read_inputs()
    stage = "smoke" if args.smoke else "full"
    out = DATA_DIR / stage
    gamma_values = [float(x) for x in config["gamma_true_values"]]
    simulation_seeds = [int(config["simulation_seeds"][0])] if args.smoke else [int(x) for x in config["simulation_seeds"]]
    n_test = int(config["test_size"])
    k = max(1, int(round(n_test * float(config["selection_share"]))))
    if n_test != int(config["sample_size"]) - int(config["train_size"]):
        raise ValueError("Config test_size must equal sample_size minus train_size")
    if not 0 < k <= n_test:
        raise ValueError("Invalid selection budget")

    LOG.info("M1-C stage=%s, scenarios=%s, fits=%s, top-k=%s", stage, len(gamma_values)*len(simulation_seeds), 2*len(gamma_values)*len(simulation_seeds), k)
    if args.dry_run:
        for gamma in gamma_values:
            for seed in simulation_seeds:
                LOG.info("DRY RUN gamma=%s seed=%s models=B0,B1", gamma, seed)
        return

    fit_records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for gamma in gamma_values:
        for seed in simulation_seeds:
            group = scenarios.loc[
                np.isclose(scenarios["gamma_true"], gamma, atol=1e-12, rtol=0)
                & scenarios["simulation_seed"].eq(seed)
            ].reset_index(drop=True)
            if len(group) != int(config["sample_size"]):
                raise ValueError(f"Missing scenario gamma={gamma}, seed={seed}")
            case_dir = out / "fits" / label(gamma, seed)
            per_model: dict[str, pd.DataFrame] = {}
            for name in ("B0", "B1"):
                fit_dir = case_dir / name
                record = run_fit(group, config, hashes, gamma, seed, name, fit_dir, k, args.force)
                fit_records.append({key: val for key, val in record.items() if key != "fingerprints"})
                per_model[name] = pd.read_parquet(fit_dir / "test_decisions.parquet")

            comparison = compare_decisions(per_model["B0"], per_model["B1"], k=k)
            comparison.update({"gamma_true": gamma, "simulation_seed": seed})
            comparisons.append(comparison)
            LOG.info("CASE gamma=%s seed=%s overlap=%d/%d regret B0=%.5f B1=%.5f",
                     gamma, seed, comparison["overlap"], k,
                     comparison["expected_regret_B0"], comparison["expected_regret_B1"])

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fit_records).to_csv(out / "fit_results.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "scenario_comparisons.csv", index=False)
    versions: dict[str, str] = {}
    for pkg in ("pymc", "arviz", "numpy", "pandas", "scipy"):
        versions[pkg] = metadata.version(pkg)
    manifest = {
        "experiment": "M1-C",
        "stage": stage,
        "scenario_count": len(comparisons),
        "model_fits": len(fit_records),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "dependencies": versions,
        "git_head": git_head(),
        "input_and_code_fingerprints": hashes,
        "config": config,
        "note": "Synthetic controlled experiment. No empirical flood-risk validation.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("M1-C %s COMPLETE. Wrote %s", stage.upper(), out)


if __name__ == "__main__":  # Required for multiprocess sampling on Windows
    main()
