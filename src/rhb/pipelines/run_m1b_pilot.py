from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from scipy.special import expit

from rhb.models.logistic_phase3 import build_phase3_model
from rhb.models.logistic_m1 import build_m1_model


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/phase4/RTM/m1b_pilot"

SEED = 20261105
GAMMA_TRUE = -0.8


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Reuse M1-A buildings

    sample = pd.read_csv(
        ROOT / "outputs/phase4/RTM/m1a_pilot/pilot_sample.csv"
    )

    features = pd.read_parquet(
        ROOT / "outputs/phase3/RTM/phase3_features_scaled.parquet",
        columns=[
            "bldg_id",
            "E_hat_v0",
            "H_pluvial_v1_logrel",
        ],
    )

    df = sample[["bldg_id", "T50"]].merge(
        features,
        on="bldg_id",
        validate="one_to_one",
    )

    assert len(df) == 5000

    E = df["E_hat_v0"].to_numpy(dtype=float)
    H = df["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    T = np.arcsinh(df["T50"].to_numpy(dtype=float) / 0.5)

    assert np.isfinite(E).all()
    assert np.isfinite(H).all()
    assert np.isfinite(T).all()

    # 2. Generate outcomes with known topographic effect

    logit_true = -3.0 + E + H + GAMMA_TRUE * T
    p_true = expit(logit_true)

    rng = np.random.default_rng(SEED)
    y = rng.binomial(1, p_true)

    # 3. Train/test split

    indices = rng.permutation(len(df))
    train = indices[:4000]
    test = indices[4000:]

    df["p_true_m1b"] = p_true
    df["Y_m1b"] = y
    df["split"] = "test"
    df.loc[train, "split"] = "train"

    df.to_parquet(OUT / "m1b_dataset.parquet", index=False)

    print("\n=== M1-B SYNTHETIC EXPERIMENT ===")
    print("Buildings:", len(df))
    print("Training:", len(train))
    print("Test:", len(test))
    print("True gamma:", GAMMA_TRUE)
    print("Synthetic damage rate:", round(y.mean(), 5))

    models = {
        "B0": build_phase3_model(
            E[train], H[train], y[train]
        ),
        "B1": build_m1_model(
            E[train], H[train], T[train], y[train]
        ),
    }

    # 4. Bayesian inference

    for name, model in models.items():

        print(f"\n=== FITTING {name} ===")

        with model:
            idata = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=2,
                target_accept=0.9,
                random_seed=SEED,
            )

        variables = ["alpha", "beta_E", "beta_H"]

        if name == "B1":
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

        print("Divergences:", divergences)

        # 5. Predict held-out buildings

        posterior = idata.posterior

        def draws(variable):
            return posterior[variable].values.reshape(-1, 1)

        logits = (
            draws("alpha")
            + draws("beta_E") * E[test][None, :]
            + draws("beta_H") * H[test][None, :]
        )

        if name == "B1":
            logits = (
                logits
                + draws("gamma_T") * T[test][None, :]
            )

        p_draws = expit(logits)
        p_mean = p_draws.mean(axis=0)

        # Predictive error against known synthetic truth
        rmse = np.sqrt(
            np.mean((p_mean - p_true[test]) ** 2)
        )

        # Brier score against held-out binary outcomes
        brier = np.mean(
            (p_mean - y[test]) ** 2
        )

        # Equal-capacity top-k decision
        k = 10

        selected = np.argpartition(p_mean, -k)[-k:]
        oracle = np.argpartition(p_true[test], -k)[-k:]

        expected_selected = p_true[test][selected].sum()
        expected_oracle = p_true[test][oracle].sum()

        regret = expected_oracle - expected_selected

        print("\nHELD-OUT EVALUATION")
        print("RMSE against true probabilities:", round(rmse, 6))
        print("Brier score:", round(brier, 6))
        print("Expected selected damage:", round(expected_selected, 6))
        print("Oracle expected damage:", round(expected_oracle, 6))
        print("Selection regret:", round(regret, 6))

        summary.to_csv(OUT / f"{name}_summary.csv")
        az.to_netcdf(idata, OUT / f"{name}_idata.nc")

    print("\nM1-B PILOT: COMPLETE")


if __name__ == "__main__":
    main()
