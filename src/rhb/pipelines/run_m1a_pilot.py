from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from rhb.models.logistic_phase3 import build_phase3_model
from rhb.models.logistic_m1 import build_m1_model


def main() -> None:
    ROOT = Path(__file__).resolve().parents[3]
    OUT = ROOT / "outputs/phase4/RTM/m1a_pilot"
    OUT.mkdir(parents=True, exist_ok=True)

    SEED = 20261105
    N = 5000

    # --------------------------------------------------
    # 1. Load original Phase 3 data and M1 terrain
    # --------------------------------------------------

    phase3 = pd.read_parquet(
        ROOT / "outputs/phase3/RTM/phase3_features_scaled.parquet"
    )

    terrain = pd.read_parquet(
        ROOT.parent
        / "data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet"
    )

    assert phase3["bldg_id"].is_unique
    assert terrain["bldg_id"].is_unique

    df = phase3.merge(
        terrain[["bldg_id", "T50", "terrain_measured"]],
        on="bldg_id",
        how="inner",
        validate="one_to_one",
    )

    assert len(df) == 221324

    df = df.loc[
        df["terrain_measured"].eq(True) & df["T50"].notna()
    ].copy()

    assert len(df) == 220456

    # Reproducible paired sample
    df = (
        df.sample(n=N, random_state=SEED)
        .sort_values("bldg_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 2. Prepare predictors
    # --------------------------------------------------

    E = df["E_hat_v0"].to_numpy(dtype=float)
    H = df["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    T = np.arcsinh(df["T50"].to_numpy(dtype=float) / 0.5)
    y = df["Y_damage"].to_numpy(dtype=int)

    assert np.isfinite(E).all()
    assert np.isfinite(H).all()
    assert np.isfinite(T).all()
    assert np.isin(y, [0, 1]).all()

    df[["bldg_id", "Y_damage", "T50"]].to_csv(
        OUT / "pilot_sample.csv", index=False
    )

    print("\n=== M1-A PILOT DATA ===")
    print("Buildings:", len(df))
    print("Synthetic damage rate:", round(y.mean(), 5))
    print("Transformed T50 range:", T.min(), T.max())

    # --------------------------------------------------
    # 3. Compare baseline and topographic model
    # --------------------------------------------------

    models = {
        "A0": build_phase3_model(E, H, y),
        "A1": build_m1_model(E, H, T, y),
    }

    for name, model in models.items():

        print(f"\n=== FITTING {name} ===")

        with model:

            # Prior predictive check on actual predictors
            prior = pm.sample_prior_predictive(
                samples=100,
                random_seed=SEED,
            )

            prior_rates = (
                prior.prior_predictive["Y_obs"]
                .values.reshape(-1, len(y))
                .mean(axis=1)
            )

            print(
                "Prior predictive damage-rate percentiles:",
                np.round(np.percentile(prior_rates, [5, 50, 95]), 4),
            )

            idata = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=2,
                target_accept=0.9,
                random_seed=SEED,
            )

            idata.extend(
                pm.sample_posterior_predictive(
                    idata,
                    var_names=["Y_obs"],
                    random_seed=SEED,
                )
            )

        variables = ["alpha", "beta_E", "beta_H"]

        if name == "A1":
            variables.append("gamma_T")

        summary = az.summary(
            idata,
            var_names=variables,
            hdi_prob=0.95,
        )

        print("\nPOSTERIOR SUMMARY")
        print(summary)

        divergences = int(
            idata.sample_stats["diverging"].sum().item()
        )

        print("\nDivergences:", divergences)

        posterior_rates = (
            idata.posterior_predictive["Y_obs"]
            .values.reshape(-1, len(y))
            .mean(axis=1)
        )

        print(
            "Posterior predictive damage-rate percentiles:",
            np.round(np.percentile(posterior_rates, [5, 50, 95]), 4),
        )

        summary.to_csv(OUT / f"{name}_summary.csv")

        az.to_netcdf(
            idata,
            OUT / f"{name}_idata.nc",
        )

    print("\nM1-A PILOT: COMPLETE")


if __name__ == "__main__":
    main()
