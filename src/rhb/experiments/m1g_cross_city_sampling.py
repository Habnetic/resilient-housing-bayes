"""M1-G: prepare paired synthetic HAM/DON scenarios from frozen M1-C design.

One command, from resilient-housing-bayes with PYTHONPATH=src:
    python -m rhb.experiments.m1g_cross_city_sampling --city HAM --check
    python -m rhb.experiments.m1g_cross_city_sampling --city HAM

Inputs are read-only. Output is isolated to outputs/phase4/<CITY>/m1g/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rhb.experiments.m1f_population_sampling import (
    derived_seed, generate_scenarios, sha256, validate_scenarios,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/m1g_cross_city.json"
LOGGER = logging.getLogger("rhb.m1g.sampling")


def check_python() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("M1-G requires the scientific Python 3.11 environment")


def load_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject changes to the registered experiment, not merely malformed JSON."""
    if cfg.get("experiment") != "M1-G" or set(cfg["cities"]) != {"HAM", "DON"}:
        raise ValueError("M1-G must target exactly HAM and DON")
    if cfg["population_ids"] != ["city01"]:
        raise ValueError("Exactly one named population per city")
    if (cfg["sample_size"], cfg["train_size"], cfg["test_size"]) != (5000, 4000, 1000):
        raise ValueError("Frozen M1-C population design is 5000/4000/1000")
    if cfg["gamma_true_values"] != [0.0, -0.8] or cfg["simulation_seeds"] != [101, 202]:
        raise ValueError("M1-G fixed gamma and outcome seeds changed")
    if (cfg["generator_alpha"], cfg["generator_beta_E"], cfg["generator_beta_H"]) != (-3.0, 1.0, 1.0):
        raise ValueError("Frozen synthetic generating model changed")
    if (cfg["terrain_transform_scale_m"], cfg["gamma_prior_sigma"], cfg["selection_share"]) != (0.5, 0.5, 0.01):
        raise ValueError("Frozen transform, gamma prior or decision budget changed")
    if cfg["mcmc"] != {"chains": 2, "draws": 500, "tune": 500, "target_accept": 0.9}:
        raise ValueError("Frozen exploratory inference settings changed")
    for city, n in (("HAM", 341530), ("DON", 7755)):
        if cfg["cities"][city]["expected_buildings"] != n:
            raise ValueError(f"Unexpected {city} population contract")
    if cfg["cities"]["HAM"]["expected_terrain_files"] != 342:
        raise ValueError("Expected Hamburg batch count changed")
    if cfg["frozen_reference"]["rtm_m1_tag"] != "phase4-m1-complete-v1.0":
        raise ValueError("Rotterdam reference tag changed")


def normalize_ids(series: pd.Series, label: str) -> pd.Series:
    """Preserve IDs as exact strings, rejecting silent null/float corruption."""
    if series.isna().any():
        raise ValueError(f"{label}: null building IDs")
    if pd.api.types.is_float_dtype(series):
        if not np.equal(series.to_numpy(), np.floor(series.to_numpy())).all():
            raise ValueError(f"{label}: noninteger floating-point building IDs")
        return series.astype("int64").astype(str)
    return series.astype(str)


def verified_ham_batches(folder: Path, manifest_path: Path, expected: int, expected_rows: int = 341530) -> tuple[pd.DataFrame, dict[str, str]]:
    """Verify every HAM Parquet batch against its frozen extraction manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("outputs", [])
    files = sorted(folder.glob("terrain_*.parquet"))
    if len(records) != expected or len(files) != expected:
        raise ValueError(f"HAM terrain count mismatch: manifest={len(records)}, files={len(files)}")
    seen = {str(p.resolve()) for p in files}
    frames: list[pd.DataFrame] = []
    counts = 0
    for record in records:
        path = ((ROOT.parent / "data") / record["path"]).resolve()
        if str(path) not in seen or path.parent != folder.resolve():
            raise ValueError(f"HAM batch outside expected inventory: {path}")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"HAM batch checksum mismatch: {path}")
        table = pd.read_parquet(path, columns=["bldg_id", "T50", "terrain_measured"])
        if len(table) != int(record["count"]):
            raise ValueError(f"HAM batch row mismatch: {path}")
        counts += len(table)
        frames.append(table)
    df = pd.concat(frames, ignore_index=True)
    if counts != expected_rows or len(df) != expected_rows:
        raise ValueError("HAM terrain population changed")
    return df, {"ham_terrain_manifest_sha256": sha256(manifest_path)}


def load_universe(city: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    """One-to-one audit of entire city before sampling; no silent drops."""
    if city not in ("HAM", "DON"):
        raise ValueError(city)
    sc = cfg["cities"][city]
    n = int(sc["expected_buildings"])
    features_path = (ROOT / sc["features_path"]).resolve()
    if not features_path.is_file():
        raise FileNotFoundError(f"M1-G Phase 3 features missing: {features_path}")
    features = pd.read_parquet(features_path, columns=["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel"])
    fingerprints = {"phase3_features_sha256": sha256(features_path)}
    if city == "HAM":
        directory = (ROOT / sc["terrain_batch_dir"]).resolve()
        manifest_path = (ROOT / sc["terrain_manifest"]).resolve()
        terrain, fingerprints_ham = verified_ham_batches(directory, manifest_path, sc["expected_terrain_files"])
        fingerprints.update(fingerprints_ham)
    else:
        path = (ROOT / sc["terrain_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"M1-G DON terrain consolidated file missing: {path}")
        terrain = pd.read_parquet(path, columns=["bldg_id", "T50", "terrain_measured"])
        fingerprints["terrain_sha256"] = sha256(path)
    if len(features) != n or len(terrain) != n:
        raise ValueError(f"{city}: expected {n} exact Phase3/terrain rows; got {len(features)}/{len(terrain)}")
    for table, label in ((features, "Phase3"), (terrain, "terrain")):
        table["bldg_id"] = normalize_ids(table["bldg_id"], label)
        if not table["bldg_id"].is_unique:
            raise ValueError(f"{city} duplicate IDs in {label}")
    if not terrain["terrain_measured"].eq(True).all():
        raise ValueError(f"{city}: unmeasured terrain, cannot silently impute")
    joined = features.merge(terrain[["bldg_id", "T50"]], on="bldg_id", how="outer", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != n:
        raise ValueError(f"{city}: exact Phase3/terrain building-ID match failed")
    joined = joined.drop(columns="_merge")
    values = joined[["T50", "E_hat_v0", "H_pluvial_v1_logrel"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{city}: missing/nonfinite predictors")
    joined["T"] = np.arcsinh(joined["T50"].to_numpy(dtype=float) / cfg["terrain_transform_scale_m"])
    return joined.sort_values("bldg_id", kind="stable").reset_index(drop=True), fingerprints


def select_population(universe: pd.DataFrame, city: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Use a fixed, city-specific sample and held-out split, shared by all scenarios."""
    n = cfg["sample_size"]
    rng = np.random.default_rng(derived_seed("M1-G-population", cfg["population_seed"], city))
    index = rng.choice(len(universe), size=n, replace=False)
    selected = universe.iloc[index].sort_values("bldg_id", kind="stable").reset_index(drop=True).copy()
    rng_split = np.random.default_rng(derived_seed("M1-G-split", cfg["split_seed"], city))
    train_indices = rng_split.choice(n, size=cfg["train_size"], replace=False)
    split = np.full(n, "test", dtype=object)
    split[train_indices] = "train"
    selected["split"] = split
    selected["population_id"] = cfg["population_ids"][0]
    return selected


def generate_city_scenarios(population: pd.DataFrame, city: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Delegate the scientific generator to frozen M1-F, with city-specific RNG prefix."""
    local = {**cfg, "outcome_seed_prefix": cfg["outcome_seed_prefix"] + "-" + city}
    result = generate_scenarios(population, local)
    validate_scenarios(result, local, excluded=set())
    return result.rename(columns={"Y_m1f": "Y_m1g"})


def source_fingerprints(city: str, cfg: dict[str, Any], inputs: dict[str, str]) -> dict[str, str]:
    code = [
        ROOT / "src/rhb/experiments/m1g_cross_city_sampling.py",
        ROOT / "src/rhb/experiments/m1f_population_sampling.py",
        ROOT / "src/rhb/pipelines/run_m1g.py",
        ROOT / "src/rhb/pipelines/run_m1c.py",
        ROOT / "src/rhb/decision/m1c_metrics.py",
        ROOT / "src/rhb/decision/phase3_decision_metrics.py",
        ROOT / "src/rhb/models/logistic_m1.py",
        ROOT / "src/rhb/models/logistic_phase3.py",
    ]
    return {
        "config_sha256": sha256(CONFIG),
        **inputs,
        **{f"code_sha256:{p.relative_to(ROOT).as_posix()}": sha256(p) for p in code},
        "city": city,
    }


def output_dir(city: str) -> Path:
    return ROOT / "outputs/phase4" / city / "m1g"


def prepare(city: str, check_only: bool) -> None:
    check_python()
    cfg = load_config()
    universe, inputs = load_universe(city, cfg)
    LOGGER.info("%s input audit PASS: %d IDs, T50 measured and finite", city, len(universe))
    pop = select_population(universe, city, cfg)
    scenarios = generate_city_scenarios(pop, city, cfg)
    assert len(scenarios) == 4 * cfg["sample_size"]
    hashes = source_fingerprints(city, cfg, inputs)
    if check_only:
        LOGGER.info("%s DRY INPUT CHECK: sample %d, 4 cases, 8 B0/B1 fits", city, len(pop))
        return
    out = output_dir(city)
    snapshot = out / "sampling_manifest.json"
    if snapshot.exists() or (out / "m1g_scenarios.parquet").exists():
        raise FileExistsError(f"{city}: M1-G scenarios already frozen: {out}; never overwrite")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "m1g_scenarios.parquet"
    ids_path = out / "population_ids.csv"
    scenarios.to_parquet(path, index=False)
    pop[["population_id", "bldg_id", "split"]].to_csv(ids_path, index=False)
    manifest = {
        "experiment": "M1-G", "city": city, "stage": "sampling", "evidence": "synthetic",
        "input_fingerprints": hashes,
        "population": cfg["sample_size"], "train": cfg["train_size"], "test": cfg["test_size"],
        "cases": 4, "seeds": cfg["simulation_seeds"], "effects": cfg["gamma_true_values"],
        "config": cfg,
        "output_hashes": {"scenarios": sha256(path), "population_ids": sha256(ids_path)},
        "reproduction": f"python -m rhb.experiments.m1g_cross_city_sampling --city {city}",
    }
    snapshot.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("%s scenarios frozen: %d rows, 4 paired cases -> %s", city, len(scenarios), out)
    for gamma, group in scenarios.groupby("gamma_true", sort=False):
        LOGGER.info("%s gamma=%g mean synthetic p=%.5f event rate=%.5f", city, gamma,
                    group["p_true"].mean(), group["Y_m1g"].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("HAM", "DON", "all"), required=True)
    parser.add_argument("--check", action="store_true", help="Validate inputs without writing anything")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for city in (("HAM", "DON") if args.city == "all" else (args.city,)):
        prepare(city, check_only=args.check)


if __name__ == "__main__":
    main()
