from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd
import pymc as pm
import arviz as az

from rhb.models.logistic_phase3 import build_phase3_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    args = parser.parse_args()

    city = args.city.upper()

    project_root = Path(__file__).resolve().parents[3]


    data_path = project_root / "outputs" / "phase3" / city / "phase3_features_scaled.parquet"
    out_dir = project_root / "outputs" / "phase3" / city

    print(f"[phase3-model] project_root={project_root}")
    print(f"[phase3-model] data_path={data_path}")

    df = pd.read_parquet(data_path)

    E = df["E_hat_v0"].values
    H = df["H_pluvial_v1_logrel"].values
    y = df["Y_damage"].values

    model = build_phase3_model(E, H, y)

    with model:
        idata = pm.sample(
            draws=500,
            tune=500,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=42,
        )

    # Save idata
    idata_path = out_dir / "idata.nc"
    az.to_netcdf(idata, idata_path)

    # Save summary
    summary = az.summary(idata)
    summary_path = out_dir / "summary.json"
    summary.to_json(summary_path, orient="index")

    print(f"[phase3-model] saved idata to {idata_path}")
    print(f"[phase3-model] saved summary to {summary_path}")


if __name__ == "__main__":
    main()