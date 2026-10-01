from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path.cwd()

CITY_CONFIG = {
    "RTM": {
        "n": 221_324,
        "representative_k": 2213,
    },
    "HAM": {
        "n": 341_530,
        "representative_k": 3415,
    },
    "DON": {
        "n": 7_755,
        "representative_k": 78,
    },
}

THRESHOLDS = [
    (0.10, 0.90),
    (0.20, 0.80),
    (0.25, 0.75),
]

OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "phase3"
    / "cross_city_summary"
    / "threshold_sensitivity"
)

FIGURE_DIR = REPO_ROOT / "figures"

SUMMARY_PATH = OUT_DIR / "threshold_sensitivity_summary.csv"
ALL_K_PATH = OUT_DIR / "threshold_sensitivity_all_k.csv"
MASS_CHECK_PATH = OUT_DIR / "topk_membership_mass_check.csv"

FIGURE_PDF = FIGURE_DIR / "phase3_threshold_sensitivity.pdf"
FIGURE_PNG = FIGURE_DIR / "phase3_threshold_sensitivity.png"


def threshold_label(low: float, high: float) -> str:
    return f"{low:.2f}/{high:.2f}"


def load_city(city: str) -> pd.DataFrame:
    path = (
        REPO_ROOT
        / "outputs"
        / "phase3"
        / city
        / "asset_metrics.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_parquet(path)


def available_topk_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    result = []

    pattern = re.compile(r"^topk_prob_k(\d+)$")

    for col in df.columns:
        match = pattern.match(col)

        if match:
            result.append(
                (
                    int(match.group(1)),
                    col,
                )
            )

    return sorted(result)


def compute_threshold_metrics(
    probabilities: np.ndarray,
    n_assets: int,
    k: int,
    low: float,
    high: float,
) -> dict[str, float | int]:

    borderline = (
        (probabilities > low)
        & (probabilities < high)
    )

    count = int(borderline.sum())

    return {
        "low_threshold": low,
        "high_threshold": high,
        "threshold_label": threshold_label(low, high),
        "borderline_count": count,
        "borderline_share_N": count / n_assets,
        "borderline_share_k": count / k,
    }


def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []
    all_k_rows = []
    mass_rows = []

    for city, config in CITY_CONFIG.items():

        print(f"\n{'=' * 70}")
        print(city)
        print("=" * 70)

        df = load_city(city)

        n_assets = len(df)

        if n_assets != config["n"]:
            raise ValueError(
                f"{city}: expected N={config['n']:,}, "
                f"found N={n_assets:,}"
            )

        topk_columns = available_topk_columns(df)

        if not topk_columns:
            raise ValueError(
                f"{city}: no topk_prob_k* columns found"
            )

        # ----------------------------------------------------------
        # Verify sum_i pi_{i,k} = k for every available k
        # ----------------------------------------------------------

        for k, col in topk_columns:

            probabilities = (
                df[col]
                .to_numpy(dtype=float)
            )

            membership_mass = float(
                probabilities.sum()
            )

            error = membership_mass - k

            mass_rows.append(
                {
                    "city": city,
                    "k": k,
                    "column": col,
                    "sum_pi": membership_mass,
                    "expected_k": k,
                    "absolute_error": error,
                    "relative_error":
                        error / k,
                }
            )

            print(
                f"k={k:>6,}: "
                f"sum(pi)={membership_mass:,.6f}, "
                f"error={error:+.6e}"
            )

            # ------------------------------------------------------
            # Threshold sensitivity for every available k
            # ------------------------------------------------------

            for low, high in THRESHOLDS:

                metrics = compute_threshold_metrics(
                    probabilities=probabilities,
                    n_assets=n_assets,
                    k=k,
                    low=low,
                    high=high,
                )

                all_k_rows.append(
                    {
                        "city": city,
                        "n_assets": n_assets,
                        "k": k,
                        "prioritised_share": k / n_assets,
                        **metrics,
                    }
                )

        # ----------------------------------------------------------
        # Main paper comparison at ~1% k
        # ----------------------------------------------------------

        representative_k = config["representative_k"]

        representative_col = (
            f"topk_prob_k{representative_k}"
        )

        if representative_col not in df.columns:
            raise ValueError(
                f"{city}: missing {representative_col}"
            )

        probabilities = (
            df[representative_col]
            .to_numpy(dtype=float)
        )

        print(
            f"\nRepresentative k={representative_k:,}"
        )

        for low, high in THRESHOLDS:

            metrics = compute_threshold_metrics(
                probabilities=probabilities,
                n_assets=n_assets,
                k=representative_k,
                low=low,
                high=high,
            )

            summary_rows.append(
                {
                    "city": city,
                    "n_assets": n_assets,
                    "k": representative_k,
                    "prioritised_share":
                        representative_k / n_assets,
                    **metrics,
                }
            )

            print(
                f"  {threshold_label(low, high)}: "
                f"{metrics['borderline_count']} assets; "
                f"{metrics['borderline_share_N']:.4%} of N; "
                f"{metrics['borderline_share_k']:.2%} of k"
            )

    # --------------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------------

    summary = pd.DataFrame(summary_rows)

    all_k = pd.DataFrame(all_k_rows)

    mass_check = pd.DataFrame(mass_rows)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    all_k.to_csv(
        ALL_K_PATH,
        index=False,
    )

    mass_check.to_csv(
        MASS_CHECK_PATH,
        index=False,
    )

    # --------------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------------

    # Wider interval 0.1/0.9 must contain at least as many
    # borderline assets as 0.2/0.8, which must contain at least
    # as many as 0.25/0.75.
    for city in CITY_CONFIG:

        city_summary = (
            summary[summary["city"] == city]
            .set_index("threshold_label")
        )

        counts = [
            city_summary.loc[
                "0.10/0.90",
                "borderline_count",
            ],
            city_summary.loc[
                "0.20/0.80",
                "borderline_count",
            ],
            city_summary.loc[
                "0.25/0.75",
                "borderline_count",
            ],
        ]

        if not (
            counts[0]
            >= counts[1]
            >= counts[2]
        ):
            raise AssertionError(
                f"{city}: threshold nesting failed: {counts}"
            )

    # Membership-mass identity should hold to numerical precision.
    max_abs_error = float(
        mass_check["absolute_error"]
        .abs()
        .max()
    )

    print(
        "\nMaximum |sum(pi)-k| across all city/k combinations:",
        f"{max_abs_error:.8f}",
    )

    # --------------------------------------------------------------
    # Figure: threshold sensitivity at representative ~1% k
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    labels = [
        "0.10 / 0.90",
        "0.20 / 0.80",
        "0.25 / 0.75",
    ]

    x = np.arange(
        len(labels)
    )

    for city in CITY_CONFIG:

        city_data = (
            summary[
                summary["city"] == city
            ]
            .copy()
        )

        order = {
            "0.10/0.90": 0,
            "0.20/0.80": 1,
            "0.25/0.75": 2,
        }

        city_data["order"] = (
            city_data[
                "threshold_label"
            ]
            .map(order)
        )

        city_data = city_data.sort_values(
            "order"
        )

        y = (
            city_data[
                "borderline_share_k"
            ]
            .to_numpy()
            * 100
        )

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.5,
            label=city,
        )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        "Decision-stability thresholds"
    )

    ax.set_ylabel(
        "Borderline assets (% of prioritisation capacity k)"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_PNG,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(SUMMARY_PATH)
    print(ALL_K_PATH)
    print(MASS_CHECK_PATH)
    print(FIGURE_PDF)
    print(FIGURE_PNG)


if __name__ == "__main__":
    main()