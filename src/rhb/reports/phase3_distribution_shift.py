from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wasserstein_distance


BASE_PATH = Path("outputs") / "phase3"
OUT_DIR = BASE_PATH / "cross_city_summary"

CITIES = ["RTM", "HAM", "DON"]

FEATURES = {
    "E_hat_v0": "Exposure",
    "H_pluvial_v1_logrel": "Hazard",
}


def load_feature(city: str, feature: str) -> pd.Series:
    """Load one Phase 3 input feature for a city."""
    path = BASE_PATH / city / "phase3_features_scaled.parquet"

    df = pd.read_parquet(path, columns=[feature])

    return df[feature].dropna()


def summarise_feature(city: str, feature: str) -> dict[str, float | str | int]:
    """Return simple distribution summaries."""
    x = load_feature(city, feature)

    return {
        "city": city,
        "feature": feature,
        "n": len(x),
        "mean": x.mean(),
        "sd": x.std(),
        "q25": x.quantile(0.25),
        "median": x.median(),
        "q75": x.quantile(0.75),
    }


def compute_shift(feature: str, city: str) -> dict[str, float | str]:
    """Measure distribution shift relative to Rotterdam."""
    reference = load_feature("RTM", feature)
    target = load_feature(city, feature)

    distance = wasserstein_distance(reference, target)

    reference_sd = reference.std()

    normalised_distance = (
        distance / reference_sd
        if reference_sd > 0
        else float("nan")
    )

    return {
        "city": city,
        "feature": feature,
        "wasserstein": distance,
        "wasserstein_rtm_sd": normalised_distance,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for city in CITIES:
        for feature in FEATURES:
            summaries.append(
                summarise_feature(city, feature)
            )

    summary_df = pd.DataFrame(summaries)

    shifts = []

    for city in ["HAM", "DON"]:
        for feature in FEATURES:
            shifts.append(
                compute_shift(feature, city)
            )

    shift_df = pd.DataFrame(shifts)

    summary_path = OUT_DIR / "phase3_input_distribution_summary.csv"
    shift_path = OUT_DIR / "phase3_input_distribution_shift.csv"

    summary_df.to_csv(summary_path, index=False)
    shift_df.to_csv(shift_path, index=False)

    print("\nDistribution summaries")
    print(summary_df.to_string(index=False))

    print("\nShift relative to Rotterdam")
    print(shift_df.to_string(index=False))

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {shift_path}")


if __name__ == "__main__":
    main()