from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from rhb.transfer_scaling import (
    fit_reference_scaler,
    apply_reference_scaler,
    derive_e_hat_v0,
    save_reference_scaler,
    load_reference_scaler,
)

from rhb.support_diagnostics import (
    compute_support_diagnostics,
    save_support_diagnostics,
)


EXPOSURE_RAW_COLS = [
    "dist_to_water_m",
    "water_len_density_250m",
    "water_len_density_500m",
    "water_len_density_1000m",
]

HAZARD_RAW_COLS = [
    "H_pluvial_v1_mm",
]

SCALER_COLS = EXPOSURE_RAW_COLS + HAZARD_RAW_COLS

SUPPORT_CHECK_COLS = [
    "dist_to_water_m",
    "water_len_density_250m",
    "water_len_density_500m",
    "water_len_density_1000m",
    "H_pluvial_v1_mm",
]


def load_city_features(city_code: str, data_root: Path) -> pd.DataFrame:
    """
    Load pre-built Phase 3 asset table for one city.
    """
    city_code = city_code.upper()

    candidate_paths = {
        "RTM": data_root / "RTM" / "phase3_assets.parquet",
        "HAM": data_root / "HAM" / "phase3_assets.parquet",
        "DON": data_root / "DON" / "phase3_assets.parquet",
    }

    if city_code not in candidate_paths:
        raise ValueError(f"Unsupported city_code: {city_code}")

    path = candidate_paths[city_code]
    if not path.exists():
        raise FileNotFoundError(f"City features file not found: {path}")

    return pd.read_parquet(path)


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def save_run_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, help="Target city code, e.g. HAM")
    parser.add_argument(
        "--reference-city",
        default="RTM",
        help="Reference city code used to fit the scaler",
    )
    parser.add_argument(
        "--refit-scaler",
        action="store_true",
        help="Refit the reference scaler even if a saved scaler exists",
    )
    args = parser.parse_args()

    target_city = args.city.upper()
    reference_city = args.reference_city.upper()

    project_root = Path(__file__).resolve().parents[2]
    habnetic_root = project_root.parent
    data_root = habnetic_root / "data" / "processed"

    outputs_dir = project_root / "outputs" / "phase3"
    config_dir = outputs_dir / "config"
    scaler_path = config_dir / "rtm_feature_scaler.json"

    print(f"[phase3] project_root={project_root}")
    print(f"[phase3] habnetic_root={habnetic_root}")
    print(f"[phase3] data_root={data_root}")
    print(f"[phase3] reference_city={reference_city}")
    print(f"[phase3] target_city={target_city}")

    # Load reference city
    df_ref = load_city_features(reference_city, data_root)
    validate_required_columns(df_ref, SCALER_COLS)

    # Fit or load scaler
    if scaler_path.exists() and not args.refit_scaler:
        print(f"[phase3] loading existing scaler: {scaler_path}")
        scaler = load_reference_scaler(scaler_path)
    else:
        print("[phase3] fitting new reference scaler")
        scaler = fit_reference_scaler(
            df_ref=df_ref,
            cols=SCALER_COLS,
            reference_city=reference_city,
        )
        save_reference_scaler(scaler, scaler_path)
        print(f"[phase3] saved scaler to: {scaler_path}")

    # Load target city
    df_target = load_city_features(target_city, data_root)
    validate_required_columns(df_target, SUPPORT_CHECK_COLS)

    # Apply scaler and derive exposure proxy
    print(f"[phase3] applying reference scaler to {target_city}")
    df_target = apply_reference_scaler(df_target, scaler)

    print(f"[phase3] deriving E_hat_v0 for {target_city}")
    df_target = derive_e_hat_v0(df_target)

    ref_hazard_median = float(df_ref["H_pluvial_v1_mm"].median())

    df_target["H_pluvial_v1_rel"] = df_target["H_pluvial_v1_mm"] / ref_hazard_median
    df_target["H_pluvial_v1_logrel"] = np.log(df_target["H_pluvial_v1_rel"])

    # support diagnostics
    validate_required_columns(df_ref, SUPPORT_CHECK_COLS)
    validate_required_columns(df_target, SUPPORT_CHECK_COLS)

    diagnostics_df = compute_support_diagnostics(
        df_ref=df_ref,
        df_target=df_target,
        cols=SUPPORT_CHECK_COLS,
        reference_city=reference_city,
        target_city=target_city,
    )

    # Save transformed features for inspection
    city_out_dir = outputs_dir / target_city
    city_out_dir.mkdir(parents=True, exist_ok=True)

    transformed_path = city_out_dir / "phase3_features_scaled.parquet"
    diagnostics_json_path = city_out_dir / "support_diagnostics.json"
    diagnostics_csv_path = city_out_dir / "support_diagnostics.csv"
    metadata_path = city_out_dir / "run_metadata.json"

    df_target.to_parquet(transformed_path, index=False)

    save_support_diagnostics(
        diagnostics_df,
        output_json_path=diagnostics_json_path,
        output_csv_path=diagnostics_csv_path,
    )

    metadata = {
        "phase": "phase3",
        "reference_city": reference_city,
        "target_city": target_city,
        "data_root": str(data_root),
        "scaler_path": str(scaler_path),
        "exposure_raw_cols": EXPOSURE_RAW_COLS,
        "hazard_raw_cols": HAZARD_RAW_COLS,
        "scaler_cols": SCALER_COLS,
        "support_check_cols": SUPPORT_CHECK_COLS,
        "n_assets_reference": int(len(df_ref)),
        "hazard_model_col": "H_pluvial_v1_logrel",
        "hazard_reference_median_mm": ref_hazard_median,
        "n_assets_target": int(len(df_target)),
        "output_file": str(transformed_path),
        "support_diagnostics_json": str(diagnostics_json_path),
        "support_diagnostics_csv": str(diagnostics_csv_path),
    }
    save_run_metadata(metadata_path, metadata)

    print(f"[phase3] saved transformed target features to: {transformed_path}")
    print(f"[phase3] saved support diagnostics to: {diagnostics_json_path}")
    print(f"[phase3] saved run metadata to: {metadata_path}")
    print("[phase3] preprocessing stage complete")


if __name__ == "__main__":
    main()