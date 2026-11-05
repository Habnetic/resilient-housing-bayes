from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def assign_split(
    df: pd.DataFrame,
    train_size: int,
    seed: int,
) -> pd.DataFrame:
    """Assign a reproducible train/test split."""

    if not 0 < train_size < len(df):
        raise ValueError("Invalid train_size")

    out = df.copy()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(out))

    split = np.full(len(out), "test", dtype=object)
    split[indices[:train_size]] = "train"

    out["split"] = split

    return out


def prepare_population(
    config: dict,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Load and prepare the fixed Rotterdam population."""

    phase3_path = (
        root
        / "outputs/phase3/RTM/phase3_features_scaled.parquet"
    )

    terrain_path = (
        root.parent
        / "data/runs/rtm_m1_citywide/"
        "terrain_features_m1_5m.parquet"
    )

    phase3 = pd.read_parquet(
        phase3_path,
        columns=[
            "bldg_id",
            "E_hat_v0",
            "H_pluvial_v1_logrel",
        ],
    )

    terrain = pd.read_parquet(
        terrain_path,
        columns=[
            "bldg_id",
            "T50",
            "terrain_measured",
        ],
    )

    if not phase3["bldg_id"].is_unique:
        raise ValueError("Duplicate Phase 3 building IDs")

    if not terrain["bldg_id"].is_unique:
        raise ValueError("Duplicate terrain building IDs")

    df = phase3.merge(
        terrain,
        on="bldg_id",
        how="left",
        validate="one_to_one",
    )

    if len(df) != len(phase3):
        raise ValueError("Population size changed during join")

    df = df.loc[
        df["terrain_measured"].eq(True)
        & df["T50"].notna()
    ].copy()

    expected_population = 220456

    if len(df) != expected_population:
        raise ValueError(
            f"Expected {expected_population} valid buildings, "
            f"found {len(df)}"
        )

    df = (
        df.sample(
            n=config["sample_size"],
            random_state=config["sample_seed"],
        )
        .sort_values("bldg_id")
        .reset_index(drop=True)
    )

    scale = float(config["terrain_scale_m"])

    if scale <= 0:
        raise ValueError("terrain_scale_m must be positive")

    if config["terrain_transform"] != "asinh":
        raise ValueError("Unsupported terrain transformation")

    df["T"] = np.arcsinh(
        df["T50"].to_numpy(dtype=float) / scale
    )

    predictors = [
        "E_hat_v0",
        "H_pluvial_v1_logrel",
        "T",
    ]

    if not np.isfinite(df[predictors].to_numpy()).all():
        raise ValueError("Non-finite predictor values")

    return assign_split(
        df,
        train_size=config["train_size"],
        seed=config["split_seed"],
    )


def simulate_case(
    df: pd.DataFrame,
    gamma_true: float,
    seed: int,
) -> pd.DataFrame:
    """Generate Bernoulli outcomes with a known T effect."""

    E = df["E_hat_v0"].to_numpy(dtype=float)
    H = df["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)

    logit_true = -3.0 + E + H + gamma_true * T

    p_true = expit(logit_true)

    rng = np.random.default_rng(seed)
    uniforms = rng.random(len(df))

    y = (uniforms < p_true).astype(np.int8)

    out = df.copy()

    out["gamma_true"] = gamma_true
    out["simulation_seed"] = seed
    out["p_true"] = p_true
    out["Y_m1c"] = y

    return out


def main() -> None:
    """Generate the complete M1-C synthetic dataset."""

    config_path = PROJECT_ROOT / "configs/m1c_rtm.json"

    config = json.loads(
        config_path.read_text(encoding="utf-8-sig")
    )

    population = prepare_population(config)

    scenarios = []

    for gamma in config["gamma_true_values"]:
        for seed in config["simulation_seeds"]:

            scenario = simulate_case(
                population,
                gamma_true=float(gamma),
                seed=int(seed),
            )

            scenarios.append(scenario)

    results = pd.concat(scenarios, ignore_index=True)

    out_dir = (
        PROJECT_ROOT / "outputs/phase4/RTM/m1c"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_parquet(
        out_dir / "m1c_scenarios.parquet",
        index=False,
    )

    population[
        ["bldg_id", "split", "T50", "T"]
    ].to_csv(
        out_dir / "sample_ids.csv",
        index=False,
    )

    (out_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\n=== M1-C DATA GENERATION ===")
    print("Buildings per scenario:", len(population))
    print("Training:", (population["split"] == "train").sum())
    print("Test:", (population["split"] == "test").sum())
    print("Scenarios:", len(scenarios))
    print("Total simulated rows:", len(results))

    print("\nSynthetic damage rates:")
    print(
        results.groupby(
            ["gamma_true", "simulation_seed"]
        )["Y_m1c"].mean()
    )

    print("\nM1-C DATA GENERATION: PASS")


if __name__ == "__main__":
    main()