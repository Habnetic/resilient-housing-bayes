r"""M1-F: prepare disjoint Rotterdam populations with paired synthetic outcomes.

From repository root:
    .\.venv\Scripts\python.exe -m rhb.experiments.m1f_population_sampling

Source data are read-only. All new artifacts go under outputs/phase4/RTM/m1f/.
No empirical flood-loss observations are used in this experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/m1f_population_robustness_rtm.json"
OUTPUT_DIR = ROOT / "outputs/phase4/RTM/m1f"
LOG = logging.getLogger("rhb.m1f.sample")


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(*parts: object) -> int:
    """Independent, reproducible positive 32-bit RNG seed per named purpose."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def validate_config(cfg: dict[str, Any]) -> None:
    if cfg.get("experiment") != "M1-F" or cfg.get("city") != "RTM":
        raise ValueError("Expected M1-F Rotterdam config")
    pops = cfg.get("population_ids", [])
    if len(pops) != 3 or len(set(pops)) != 3 or any(not isinstance(x, str) or not x for x in pops):
        raise ValueError("Exactly three distinct named populations required")
    if (int(cfg["sample_size"]), int(cfg["train_size"]), int(cfg["test_size"])) != (5000, 4000, 1000):
        raise ValueError("Frozen study design requires 5,000 / 4,000 / 1,000 assets")
    if [float(v) for v in cfg["gamma_true_values"]] != [0.0, -0.8]:
        raise ValueError("Frozen effect scenarios are gamma=0 and gamma=-0.8")
    seeds = cfg.get("simulation_seeds", [])
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(int(s) < 0 for s in seeds):
        raise ValueError("Exactly two distinct nonnegative outcome seeds required")
    if float(cfg["terrain_transform_scale_m"]) != 0.5:
        raise ValueError("M1-C terrain transformation must remain asinh(T50 / 0.5 m)")
    if (float(cfg["generator_alpha"]), float(cfg["generator_beta_E"]), float(cfg["generator_beta_H"])) != (-3.0, 1.0, 1.0):
        raise ValueError("M1-C synthetic generator coefficients must remain frozen")
    if float(cfg["gamma_prior_sigma"]) != 0.5 or float(cfg["selection_share"]) != 0.01:
        raise ValueError("M1-C terrain prior and top-1% selection must remain frozen")
    if int(cfg["expected_reference_sample_size"]) != 5000 or int(cfg["expected_measured_assets"]) != 220456:
        raise ValueError("Unexpected source inventory counts")
    mcmc = cfg["mcmc"]
    if int(mcmc["chains"]) != 2 or int(mcmc["draws"]) != 500 or int(mcmc["tune"]) != 500 or float(mcmc["target_accept"]) != 0.9:
        raise ValueError("M1-F exploratory inference must match M1-C (2x500 draws/tune)")
    for key in ("population_seed", "split_seed"):
        if int(cfg[key]) < 0:
            raise ValueError(f"Invalid {key}")


def check_python() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"M1-F requires project Python 3.11, found {sys.version.split()[0]}. "
            "Use .\\.venv\\Scripts\\python.exe (resilient-housing-bayes, not data)."
        )


def source_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {key: (ROOT / cfg[key]).resolve() for key in
            ("features_path", "terrain_path", "exclude_reference_path")}


def load_universe(features_path: Path, terrain_path: Path, expected_count: int) -> pd.DataFrame:
    """One-to-one join on bldg_id; include only genuinely measured T50."""
    for path in (features_path, terrain_path):
        if not path.is_file():
            raise FileNotFoundError(f"M1-F source not found: {path}")
    features = pd.read_parquet(features_path, columns=["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel"])
    terrain = pd.read_parquet(terrain_path, columns=["bldg_id", "T50", "terrain_measured"])
    for name, table in (("Phase 3 features", features), ("terrain", terrain)):
        if table["bldg_id"].isna().any() or not table["bldg_id"].is_unique:
            raise ValueError(f"Missing or duplicate bldg_id in {name}")
    if terrain["terrain_measured"].isna().any():
        raise ValueError("Unknown terrain_measured flags must be reviewed")
    if not terrain["terrain_measured"].isin([True, False]).all():
        raise ValueError("terrain_measured must be a boolean flag")
    measured = terrain.loc[terrain["terrain_measured"].eq(True), ["bldg_id", "T50"]]
    if len(measured) != expected_count:
        raise ValueError(f"Measured terrain universe changed: {len(measured)} vs {expected_count}")
    # A measured asset must have Phase 3 predictors; no silent population loss.
    joined = measured.merge(features, on="bldg_id", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Measured terrain assets without matching Phase 3 predictors")
    joined = joined.drop(columns="_merge")
    matrix = joined[["T50", "E_hat_v0", "H_pluvial_v1_logrel"]].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Invalid or missing predictors in measured population")
    joined["T"] = np.arcsinh(joined["T50"].to_numpy(dtype=float) / 0.5)
    return joined.sort_values("bldg_id", kind="stable").reset_index(drop=True)


def load_reference_ids(path: Path, expected_count: int) -> set[Any]:
    """Exclude the original M1-C buildings to test a genuinely new asset set."""
    if not path.is_file():
        raise FileNotFoundError(f"Original M1-C scenarios missing: {path}")
    refs = pd.read_parquet(path, columns=["bldg_id"])["bldg_id"]
    unique = set(refs.to_list())
    if len(unique) != expected_count:
        raise ValueError(f"M1-C reference must have exactly {expected_count} unique assets")
    return unique


def select_populations(universe: pd.DataFrame, excluded: set[Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Choose 3 disjoint random samples without replacement, excluding M1-C."""
    available = universe.loc[~universe["bldg_id"].isin(excluded)].reset_index(drop=True)
    n = len(cfg["population_ids"]) * int(cfg["sample_size"])
    if len(available) < n:
        raise ValueError("Insufficient eligible assets for three disjoint populations")
    rng = np.random.default_rng(int(cfg["population_seed"]))
    indices = rng.choice(len(available), size=n, replace=False)
    groups = []
    for index, pop in enumerate(cfg["population_ids"]):
        first = index * int(cfg["sample_size"])
        subset = available.iloc[indices[first:first + int(cfg["sample_size"])]].copy()
        subset = subset.sort_values("bldg_id", kind="stable").reset_index(drop=True)
        split_rng = np.random.default_rng(derived_seed("M1-F-split", cfg["split_seed"], pop))
        train_indices = split_rng.choice(len(subset), size=int(cfg["train_size"]), replace=False)
        split = np.full(len(subset), "test", dtype=object)
        split[train_indices] = "train"
        subset["population_id"] = pop
        subset["split"] = split
        groups.append(subset)
    result = pd.concat(groups, ignore_index=True)
    if not result["bldg_id"].is_unique or result["bldg_id"].isin(excluded).any():
        raise AssertionError("Population overlap or M1-C reference leakage")
    for _, group in result.groupby("population_id"):
        if len(group) != int(cfg["sample_size"]) or group["split"].eq("train").sum() != int(cfg["train_size"]):
            raise AssertionError("Invalid population split")
    return result


def generate_scenarios(populations: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Reuse uniforms for paired gamma effects at fixed population/outcome seed."""
    scenarios = []
    for pop in cfg["population_ids"]:
        subset = populations.loc[populations["population_id"].eq(pop)].reset_index(drop=True)
        e = subset["E_hat_v0"].to_numpy(dtype=float)
        h = subset["H_pluvial_v1_logrel"].to_numpy(dtype=float)
        t = subset["T"].to_numpy(dtype=float)
        for seed in cfg["simulation_seeds"]:
            random_seed = derived_seed(cfg["outcome_seed_prefix"], pop, seed)
            uniforms = np.random.default_rng(random_seed).random(len(subset))
            for gamma in cfg["gamma_true_values"]:
                p = expit(float(cfg["generator_alpha"]) + float(cfg["generator_beta_E"]) * e
                           + float(cfg["generator_beta_H"]) * h + float(gamma) * t)
                block = subset.copy()
                block["gamma_true"] = float(gamma)
                block["simulation_seed"] = int(seed)
                block["p_true"] = p
                block["Y_m1f"] = (uniforms < p).astype("int8")
                scenarios.append(block)
    return pd.concat(scenarios, ignore_index=True)


def validate_scenarios(df: pd.DataFrame, cfg: dict[str, Any], excluded: set[Any]) -> None:
    """No leakage, consistent population/split and correct generative truth."""
    n_cases = len(cfg["population_ids"]) * len(cfg["gamma_true_values"]) * len(cfg["simulation_seeds"])
    if len(df) != n_cases * int(cfg["sample_size"]):
        raise ValueError("Wrong number of scenario rows")
    population_sets: dict[str, set[Any]] = {}
    for (pop, gamma, seed), group in df.groupby(["population_id", "gamma_true", "simulation_seed"], sort=False):
        if len(group) != int(cfg["sample_size"]) or not group["bldg_id"].is_unique:
            raise ValueError("Wrong case size or duplicated buildings")
        if group["split"].eq("train").sum() != int(cfg["train_size"]) or group["split"].eq("test").sum() != int(cfg["test_size"]):
            raise ValueError("Invalid train/test split")
        p = expit(-3.0 + group["E_hat_v0"].to_numpy(dtype=float)
                  + group["H_pluvial_v1_logrel"].to_numpy(dtype=float)
                  + float(gamma) * group["T"].to_numpy(dtype=float))
        if not np.allclose(p, group["p_true"].to_numpy(dtype=float), rtol=0, atol=1e-14):
            raise ValueError("Synthetic generator truth changed")
        uniforms = np.random.default_rng(derived_seed(cfg["outcome_seed_prefix"], pop, seed)).random(len(group))
        if not np.array_equal((uniforms < p).astype("int8"), group["Y_m1f"].to_numpy(dtype="int8")):
            raise ValueError("Paired outcome stream does not reproduce")
        ids = set(group["bldg_id"].to_list())
        if pop in population_sets and population_sets[pop] != ids:
            raise ValueError("Population changed between effect/seed cases")
        population_sets[pop] = ids
        # All effect/seed cases must retain identical ID order and split.
        reference = df.loc[(df["population_id"] == pop)
                           & (df["gamma_true"] == float(cfg["gamma_true_values"][0]))
                           & (df["simulation_seed"] == int(cfg["simulation_seeds"][0]))]
        if not group["bldg_id"].reset_index(drop=True).equals(reference["bldg_id"].reset_index(drop=True)):
            raise ValueError("Population asset order differs across cases")
        if not group["split"].reset_index(drop=True).equals(reference["split"].reset_index(drop=True)):
            raise ValueError("Population split differs across cases")
    if set().union(*population_sets.values()) & excluded:
        raise ValueError("Original M1-C assets were reused")
    for a in cfg["population_ids"]:
        for b in cfg["population_ids"]:
            if a != b and population_sets[a] & population_sets[b]:
                raise ValueError("Different M1-F populations overlap")


def input_fingerprints(cfg: dict[str, Any]) -> dict[str, str]:
    paths = source_paths(cfg)
    return {
        "config_sha256": sha256(CONFIG_PATH),
        "generator_sha256": sha256(Path(__file__)),
        **{f"source_sha256:{name}": sha256(path) for name, path in paths.items()},
    }


def main() -> None:
    check_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Explicitly regenerate snapshots after review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    snapshot = OUTPUT_DIR / "sampling_manifest.json"
    if snapshot.exists() and not args.force:
        raise FileExistsError(f"Sampling already frozen: {snapshot}. Do not overwrite silently.")
    paths = source_paths(cfg)
    LOG.info("Reading source inventory and excluding original M1-C population...")
    universe = load_universe(paths["features_path"], paths["terrain_path"], int(cfg["expected_measured_assets"]))
    excluded = load_reference_ids(paths["exclude_reference_path"], int(cfg["expected_reference_sample_size"]))
    population = select_populations(universe, excluded, cfg)
    scenarios = generate_scenarios(population, cfg)
    validate_scenarios(scenarios, cfg, excluded)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = input_fingerprints(cfg)
    scenario_path = OUTPUT_DIR / "m1f_scenarios.parquet"
    sample_path = OUTPUT_DIR / "population_ids.csv"
    population[["population_id", "bldg_id", "split"]].to_csv(sample_path, index=False)
    scenarios.to_parquet(scenario_path, index=False)
    manifest = {
        "experiment": "M1-F", "stage": "sampling", "evidence": "synthetic",
        "population_sizes": {p: int(cfg["sample_size"]) for p in cfg["population_ids"]},
        "excluded_reference_assets": len(excluded),
        "input_fingerprints": sources,
        "output_sha256": {"scenarios": sha256(scenario_path), "population_ids": sha256(sample_path)},
        "config": cfg,
        "reproduction": ".\\.venv\\Scripts\\python.exe -m rhb.experiments.m1f_population_sampling",
    }
    snapshot.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    LOG.info("M1-F SAMPLING COMPLETE: %s disjoint populations, %d cases, %d rows -> %s",
             len(cfg["population_ids"]), len(cfg["population_ids"]) * len(cfg["gamma_true_values"]) * len(cfg["simulation_seeds"]),
             len(scenarios), OUTPUT_DIR)
    for (pop, gamma), group in scenarios.groupby(["population_id", "gamma_true"], sort=False):
        LOG.info("%s gamma=%g mean true p=%.4f simulated rate=%.4f", pop, gamma,
                 group["p_true"].mean(), group["Y_m1f"].mean())


if __name__ == "__main__":
    main()
