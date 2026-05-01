from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from rhb.models.logistic_phase3 import build_phase3_model


PRIOR_PROFILES = {
    "baseline": {
        "alpha_mu": 0.0,
        "alpha_sigma": 2.5,
        "beta_e_mu": 0.0,
        "beta_e_sigma": 2.5,
        "beta_h_mu": 0.0,
        "beta_h_sigma": 2.5,
        "label": "baseline_normal_0_2p5",
    },
    "low_event": {
        "alpha_mu": -3.0,
        "alpha_sigma": 1.0,
        "beta_e_mu": 0.0,
        "beta_e_sigma": 1.0,
        "beta_h_mu": 0.0,
        "beta_h_sigma": 1.0,
        "label": "low_event_weakly_informative",
    },
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="City code: RTM, HAM, DON")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--prior-profile",
        default="baseline",
        choices=list(PRIOR_PROFILES.keys()),
        help="Prior profile to evaluate",
    )
    args = parser.parse_args()

    city = args.city.upper()
    prior_cfg = PRIOR_PROFILES[args.prior_profile]

    project_root = Path(__file__).resolve().parents[3]
    city_dir = project_root / "outputs" / "phase3" / city
    features_path = city_dir / "phase3_features_scaled.parquet"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    print(f"[prior-predictive] city={city}")
    print(f"[prior-predictive] prior_profile={args.prior_profile}")
    print(f"[prior-predictive] features={features_path}")

    df = pd.read_parquet(features_path)

    required = ["E_hat_v0", "H_pluvial_v1_logrel", "Y_damage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    e = df["E_hat_v0"].to_numpy()
    h = df["H_pluvial_v1_logrel"].to_numpy()
    y = df["Y_damage"].to_numpy()

    model = build_phase3_model(
        E=e,
        H=h,
        y=y,
        alpha_mu=prior_cfg["alpha_mu"],
        alpha_sigma=prior_cfg["alpha_sigma"],
        beta_e_mu=prior_cfg["beta_e_mu"],
        beta_e_sigma=prior_cfg["beta_e_sigma"],
        beta_h_mu=prior_cfg["beta_h_mu"],
        beta_h_sigma=prior_cfg["beta_h_sigma"],
    )

    with model:
        prior = pm.sample_prior_predictive(
            samples=args.draws,
            random_seed=args.seed,
        )

    suffix = prior_cfg["label"]
    prior_path = city_dir / f"prior_predictive_{suffix}.nc"
    az.to_netcdf(prior, prior_path)

    rng = np.random.default_rng(args.seed)

    alpha = rng.normal(
        prior_cfg["alpha_mu"],
        prior_cfg["alpha_sigma"],
        size=args.draws,
    )
    beta_e = rng.normal(
        prior_cfg["beta_e_mu"],
        prior_cfg["beta_e_sigma"],
        size=args.draws,
    )
    beta_h = rng.normal(
        prior_cfg["beta_h_mu"],
        prior_cfg["beta_h_sigma"],
        size=args.draws,
    )

    max_assets = min(len(df), 50_000)
    sample_idx = rng.choice(len(df), size=max_assets, replace=False)

    e_s = e[sample_idx]
    h_s = h[sample_idx]

    logit = (
        alpha[:, None]
        + beta_e[:, None] * e_s[None, :]
        + beta_h[:, None] * h_s[None, :]
    )
    p = sigmoid(logit)
    event_rates = p.mean(axis=1)

    summary = {
        "city": city,
        "prior_profile": args.prior_profile,
        "prior_label": prior_cfg["label"],
        "n_assets_total": int(len(df)),
        "n_assets_used_for_probability_summary": int(max_assets),
        "n_prior_draws": int(args.draws),
        "prior_probability_percentiles": {
            "p01": float(np.quantile(p, 0.01)),
            "p05": float(np.quantile(p, 0.05)),
            "p10": float(np.quantile(p, 0.10)),
            "p50": float(np.quantile(p, 0.50)),
            "p90": float(np.quantile(p, 0.90)),
            "p95": float(np.quantile(p, 0.95)),
            "p99": float(np.quantile(p, 0.99)),
        },
        "prior_event_rate_percentiles": {
            "p01": float(np.quantile(event_rates, 0.01)),
            "p05": float(np.quantile(event_rates, 0.05)),
            "p10": float(np.quantile(event_rates, 0.10)),
            "p50": float(np.quantile(event_rates, 0.50)),
            "p90": float(np.quantile(event_rates, 0.90)),
            "p95": float(np.quantile(event_rates, 0.95)),
            "p99": float(np.quantile(event_rates, 0.99)),
        },
        "observed_synthetic_event_rate": float(y.mean()),
        "model": (
            "logit(p_i) = alpha + beta_E * E_hat_v0_i "
            "+ beta_H * H_pluvial_v1_logrel_i"
        ),
        "priors": {
            "alpha": f"Normal({prior_cfg['alpha_mu']}, {prior_cfg['alpha_sigma']})",
            "beta_E": f"Normal({prior_cfg['beta_e_mu']}, {prior_cfg['beta_e_sigma']})",
            "beta_H": f"Normal({prior_cfg['beta_h_mu']}, {prior_cfg['beta_h_sigma']})",
        },
    }

    summary_path = city_dir / f"prior_predictive_summary_{suffix}.json"
    save_json(summary, summary_path)

    print(f"[prior-predictive] saved prior predictive: {prior_path}")
    print(f"[prior-predictive] saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()