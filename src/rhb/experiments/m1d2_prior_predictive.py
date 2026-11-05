"""M1-D2 prior predictive audit; no posterior fitting and no test-label use.

From the repository root with PYTHONPATH=src:
    python -m rhb.experiments.m1d2_prior_predictive

Samples from the joint coefficient priors, followed by the Bernoulli
likelihood on the fixed M1-C *training* covariates. Does not tune priors.
"""
from __future__ import annotations

import json
import logging
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/m1d2_baseline_prior_rtm.json"
OUTPUT_ROOT = ROOT / "outputs/phase4/RTM/m1d2"
AUDIT_DIR = OUTPUT_ROOT / "prior_predictive"
LOG = logging.getLogger("rhb.m1d2.prior")


def validate_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate pre-registered alternatives and frozen M1-C reference."""
    if config.get("experiment") != "M1-D2" or config.get("city") != "RTM":
        raise ValueError("Expected the RTM M1-D2 experiment")
    priors = config.get("intercept_priors")
    if not isinstance(priors, list) or len(priors) < 3:
        raise ValueError("Expected at least three intercept priors")
    ids = [prior["id"] for prior in priors]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"P[0-9]+", ident) for ident in ids):
        raise ValueError("Prior ids must be distinct P<number> values")
    for item in priors:
        if not np.isfinite(float(item["mean"])) or not np.isfinite(float(item["sigma"])):
            raise ValueError("Prior parameters must be finite")
        if float(item["sigma"]) <= 0:
            raise ValueError("Prior standard deviation must be positive")
    reference = config.get("reference_prior_id")
    if reference != "P0" or ids[0] != "P0":
        raise ValueError("P0 must be the first and frozen reference prior")
    if (float(priors[0]["mean"]), float(priors[0]["sigma"])) != (0.0, 2.5):
        raise ValueError("P0 must exactly match the Phase 3 intercept prior")
    if float(config["fixed_beta_sigma"]) != 2.5 or float(config["fixed_gamma_sigma"]) != 0.5:
        raise ValueError("M1-C slope priors must remain unchanged")
    if int(config["prior_draws"]) < 100 or int(config["prior_batch_size"]) < 1:
        raise ValueError("Invalid prior predictive simulation size")
    if int(config["prior_seed"]) < 0:
        raise ValueError("Prior seed must be nonnegative")
    return priors


def reference_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """Use identical training predictors, regardless of synthetic effect/seed."""
    required = {
        "bldg_id", "E_hat_v0", "H_pluvial_v1_logrel", "T", "split",
        "gamma_true", "simulation_seed", "Y_m1c",
    }
    if required - set(df.columns):
        raise ValueError(f"Missing M1-C scenario columns: {sorted(required - set(df.columns))}")
    groups = list(df.groupby(["gamma_true", "simulation_seed"], sort=True))
    if not groups:
        raise ValueError("No synthetic scenarios")
    ref = groups[0][1].reset_index(drop=True)
    same_cols = ["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel", "T", "split"]
    for key, group in groups[1:]:
        group = group.reset_index(drop=True)
        if len(group) != len(ref):
            raise ValueError(f"Population size changed: {key}")
        if not ref["bldg_id"].equals(group["bldg_id"]) or not ref["split"].equals(group["split"]):
            raise ValueError(f"Building ids, ordering or split changed: {key}")
        for col in same_cols[1:-1]:
            if not np.array_equal(ref[col].to_numpy(), group[col].to_numpy(), equal_nan=True):
                raise ValueError(f"Covariate {col} changed across scenarios: {key}")
    train = ref.loc[ref["split"].eq("train")].copy().reset_index(drop=True)
    if not len(train) or train["bldg_id"].duplicated().any():
        raise ValueError("Missing or duplicated training assets")
    if not np.isfinite(train[["E_hat_v0", "H_pluvial_v1_logrel", "T"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite training predictors")
    return train


def draw_prior_predictive(
    train: pd.DataFrame,
    prior: dict[str, Any],
    draws: int,
    seed: int,
    beta_sigma: float = 2.5,
    gamma_sigma: float = 0.5,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Return prior predictive event rates, using only training predictors.

    Samples beta/gamma and alpha independently under exactly the declared
    normal priors; then samples Bernoulli outcomes. The event-rate distribution
    is over whole-population average rates, not merely sigmoid(alpha).
    """
    if draws < 1 or batch_size < 1:
        raise ValueError("draws and batch_size must be positive")
    x = train[["E_hat_v0", "H_pluvial_v1_logrel", "T"]].to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("Need nonempty finite training covariates")
    if any(not np.isfinite(s) or s <= 0 for s in (beta_sigma, gamma_sigma)):
        raise ValueError("Slope sigmas must be finite and positive")
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for start in range(0, draws, batch_size):
        n = min(batch_size, draws - start)
        alpha = rng.normal(float(prior["mean"]), float(prior["sigma"]), size=n)
        beta_e = rng.normal(0.0, beta_sigma, size=n)
        beta_h = rng.normal(0.0, beta_sigma, size=n)
        gamma_t = rng.normal(0.0, gamma_sigma, size=n)
        logits = (
            alpha[:, None]
            + beta_e[:, None] * x[None, :, 0]
            + beta_h[:, None] * x[None, :, 1]
            + gamma_t[:, None] * x[None, :, 2]
        )
        probabilities = expit(logits)
        rates = probabilities.mean(axis=1)
        bernoulli = rng.binomial(1, probabilities)
        parts.append(pd.DataFrame({
            "prior_id": prior["id"],
            "draw": np.arange(start, start + n),
            "alpha": alpha,
            "beta_E": beta_e,
            "beta_H": beta_h,
            "gamma_T": gamma_t,
            "mean_event_probability": rates,
            "sampled_event_rate": bernoulli.mean(axis=1),
        }))
    return pd.concat(parts, ignore_index=True)


def prior_summary(draws: pd.DataFrame) -> pd.DataFrame:
    """Describe both expectation and replicated Bernoulli event-rate tails."""
    rows = []
    for prior_id, sub in draws.groupby("prior_id", sort=False):
        p = sub["mean_event_probability"].to_numpy(dtype=float)
        y = sub["sampled_event_rate"].to_numpy(dtype=float)
        rows.append({
            "prior_id": prior_id,
            "draws": len(sub),
            "expected_rate_p05": float(np.quantile(p, 0.05)),
            "expected_rate_p50": float(np.quantile(p, 0.50)),
            "expected_rate_p95": float(np.quantile(p, 0.95)),
            "replicated_rate_p05": float(np.quantile(y, 0.05)),
            "replicated_rate_p50": float(np.quantile(y, 0.50)),
            "replicated_rate_p95": float(np.quantile(y, 0.95)),
        })
    return pd.DataFrame(rows)


def predictor_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Audit actual predictor scales; H is not variance-standardized."""
    items = []
    for name in ("E_hat_v0", "H_pluvial_v1_logrel", "T"):
        v = train[name].to_numpy(dtype=float)
        items.append({
            "predictor": name,
            "n": len(v),
            "mean": float(np.mean(v)),
            "std_population": float(np.std(v, ddof=0)),
            "min": float(v.min()),
            "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.median(v)),
            "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max()),
        })
    return pd.DataFrame(items)


def synthetic_training_rates(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Descriptive reference only; not used to choose prior parameters."""
    train = scenarios.loc[scenarios["split"].eq("train")]
    return (
        train.groupby(["gamma_true", "simulation_seed"], sort=True)
        .agg(train_events=("Y_m1c", "sum"), train_n=("Y_m1c", "size"))
        .reset_index()
        .assign(train_rate=lambda x: x["train_events"] / x["train_n"])
    )


def main() -> None:
    import argparse
    from rhb.pipelines.run_m1c import read_inputs, sha256, git_head

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing audit after explicit review")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    priors = validate_config(cfg)
    c_cfg, scenarios, hashes = read_inputs()
    if float(c_cfg["gamma_prior_sigma"]) != float(cfg["fixed_gamma_sigma"]):
        raise ValueError("M1-C gamma sigma differs from the D2 reference")
    train = reference_training_set(scenarios)
    if len(train) != int(c_cfg["train_size"]):
        raise ValueError("Unexpected train size")
    manifest_path = AUDIT_DIR / "audit_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Prior audit already exists: {manifest_path}. Use --force after review.")
    draws = pd.concat([
        draw_prior_predictive(
            train, prior, int(cfg["prior_draws"]), int(cfg["prior_seed"]) + 1009 * i,
            beta_sigma=float(cfg["fixed_beta_sigma"]),
            gamma_sigma=float(cfg["fixed_gamma_sigma"]),
            batch_size=int(cfg["prior_batch_size"]),
        ) for i, prior in enumerate(priors)
    ], ignore_index=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "draws": AUDIT_DIR / "prior_predictive_draws.parquet",
        "summary": AUDIT_DIR / "prior_predictive_summary.csv",
        "predictors": AUDIT_DIR / "predictor_summary.csv",
        "observed": AUDIT_DIR / "observed_synthetic_train_rates.csv",
    }
    draws.to_parquet(paths["draws"], index=False)
    prior_summary(draws).to_csv(paths["summary"], index=False)
    predictor_summary(train).to_csv(paths["predictors"], index=False)
    synthetic_training_rates(scenarios).to_csv(paths["observed"], index=False)
    manifest = {
        "experiment": "M1-D2",
        "stage": "prior_predictive",
        "source_m1c_fingerprints": hashes,
        "config_sha256": sha256(CONFIG_PATH),
        "prior_audit_code_sha256": sha256(Path(__file__)),
        "git_head": git_head(),
        "python": platform.python_version(),
        "train_n": len(train),
        "no_test_outcomes_used": True,
        "seed": int(cfg["prior_seed"]),
        "draws_per_prior": int(cfg["prior_draws"]),
        "outputs_sha256": {k: sha256(v) for k, v in paths.items()},
        "note": "Synthetic prior audit; oracle-centred -3 is known from the generator, not empirically elicited.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("M1-D2 PRIOR AUDIT COMPLETE: %d training assets, %d prior candidates, %s",
             len(train), len(priors), AUDIT_DIR)
    print(prior_summary(draws).to_string(index=False))
    print("\nTraining predictor scales:\n" + predictor_summary(train).to_string(index=False))


if __name__ == "__main__":
    main()
