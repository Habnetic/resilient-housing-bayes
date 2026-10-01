from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ROOT = Path.cwd()

FEATURE_PATH = (
    REPO_ROOT
    / "outputs"
    / "phase3"
    / "RTM"
    / "phase3_features_scaled.parquet"
)

OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "phase2"
    / "hazard_perturbation_replicates_final"
)

FIGURE_DIR = REPO_ROOT / "figures"

RAW_PATH = OUT_DIR / "phase2_hazard_replicates.csv"
SUMMARY_PATH = OUT_DIR / "phase2_hazard_summary.csv"

FIGURE_PDF = FIGURE_DIR / "fig06_hazard_perturbation_repeated.pdf"
FIGURE_PNG = FIGURE_DIR / "fig06_hazard_perturbation_repeated.png"


# Experimental sample
N_ASSETS = 50_000
TOP_K = 500

SAMPLE_SEED = 42
MCMC_SEED = 20261001

PERTURBATION_SEEDS = list(range(1001, 1021))

SIGMAS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]

# Bayesian inference
DRAWS = 1000
TUNE = 1000
CHAINS = 4
TARGET_ACCEPT = 0.90

# Decision-stability thresholds
LOW_THRESHOLD = 0.20
HIGH_THRESHOLD = 0.80


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def load_experiment_data() -> tuple[pd.DataFrame, float]:
    """
    Load the current Rotterdam Phase 3 features used by the paper.

    Returns
    -------
    sample:
        Fixed 50,000-asset Rotterdam sample.
    hazard_sd_rtm:
        Full-Rotterdam SD of H_pluvial_v1_logrel. Perturbation sigma is
        expressed as a fraction of this reference SD.
    """

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 3 feature file: {FEATURE_PATH}"
        )

    df_full = pd.read_parquet(FEATURE_PATH)

    required = {
        "bldg_id",
        "E_hat_v0",
        "H_pluvial_v1_logrel",
        "Y_damage",
    }

    missing = required.difference(df_full.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}\n"
            f"Available columns: {list(df_full.columns)}"
        )

    if len(df_full) < N_ASSETS:
        raise ValueError(
            f"Requested N={N_ASSETS:,}, "
            f"but only {len(df_full):,} rows are available."
        )

    hazard_sd_rtm = float(
        df_full["H_pluvial_v1_logrel"].std(ddof=1)
    )

    sample = (
        df_full[
            [
                "bldg_id",
                "E_hat_v0",
                "H_pluvial_v1_logrel",
                "Y_damage",
            ]
        ]
        .sample(
            n=N_ASSETS,
            random_state=SAMPLE_SEED,
        )
        .reset_index(drop=True)
    )

    print(f"Full Rotterdam N = {len(df_full):,}")
    print(
        "Full Rotterdam event rate = "
        f"{df_full['Y_damage'].mean():.4%}"
    )
    print(
        "Full Rotterdam H SD = "
        f"{hazard_sd_rtm:.8f}"
    )

    print()
    print(f"Experimental N = {len(sample):,}")
    print(f"k = {TOP_K:,}")
    print(
        "Prioritised share = "
        f"{TOP_K / len(sample):.1%}"
    )
    print(
        "Sample event rate = "
        f"{sample['Y_damage'].mean():.4%}"
    )

    return sample, hazard_sd_rtm


# ---------------------------------------------------------------------
# Bayesian model
# ---------------------------------------------------------------------

def fit_model(
    exposure: np.ndarray,
    hazard: np.ndarray,
    outcome: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | int],
]:
    """Fit logistic model and return posterior draws plus diagnostics."""

    with pm.Model() as model:

        e_data = pm.Data(
            "E",
            exposure,
            mutable=False,
        )

        h_data = pm.Data(
            "H",
            hazard,
            mutable=False,
        )

        alpha = pm.Normal(
            "alpha",
            mu=0.0,
            sigma=2.5,
        )

        beta_e = pm.Normal(
            "beta_E",
            mu=0.0,
            sigma=2.5,
        )

        beta_h = pm.Normal(
            "beta_H",
            mu=0.0,
            sigma=2.5,
        )

        logit_p = (
            alpha
            + beta_e * e_data
            + beta_h * h_data
        )

        pm.Bernoulli(
            "Y_obs",
            logit_p=logit_p,
            observed=outcome,
        )

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=MCMC_SEED,
            return_inferencedata=True,
        )

    parameter_names = [
        "alpha",
        "beta_E",
        "beta_H",
    ]

    rhat = az.rhat(
        idata,
        var_names=parameter_names,
    ).to_array()

    ess_bulk = az.ess(
        idata,
        var_names=parameter_names,
        method="bulk",
    ).to_array()

    divergences = int(
        idata.sample_stats["diverging"]
        .sum()
        .item()
    )

    diagnostics = {
        "rhat_max": float(rhat.max()),
        "ess_bulk_min": float(ess_bulk.min()),
        "divergences": divergences,
    }

    posterior = idata.posterior.stack(
        sample=("chain", "draw")
    )

    alpha_draws = (
        posterior["alpha"]
        .transpose("sample")
        .values
    )

    beta_e_draws = (
        posterior["beta_E"]
        .transpose("sample")
        .values
    )

    beta_h_draws = (
        posterior["beta_H"]
        .transpose("sample")
        .values
    )

    return (
        alpha_draws,
        beta_e_draws,
        beta_h_draws,
        diagnostics,
    )


# ---------------------------------------------------------------------
# Posterior decision metric
# ---------------------------------------------------------------------

def compute_topk_probability(
    alpha: np.ndarray,
    beta_e: np.ndarray,
    beta_h: np.ndarray,
    exposure: np.ndarray,
    hazard: np.ndarray,
    k: int,
    batch_size: int = 100,
) -> np.ndarray:
    """
    Compute posterior top-k membership probability.

    Logistic probability is monotonic in the linear predictor, so ranks
    can be computed directly from eta without materialising probabilities.
    """

    n_draws = len(alpha)
    n_assets = len(exposure)

    counts = np.zeros(
        n_assets,
        dtype=np.int32,
    )

    for start in range(
        0,
        n_draws,
        batch_size,
    ):

        stop = min(
            start + batch_size,
            n_draws,
        )

        eta = (
            alpha[start:stop, None]
            + beta_e[start:stop, None]
            * exposure[None, :]
            + beta_h[start:stop, None]
            * hazard[None, :]
        )

        top_idx = np.argpartition(
            eta,
            -k,
            axis=1,
        )[:, -k:]

        np.add.at(
            counts,
            top_idx.ravel(),
            1,
        )

    return counts / n_draws


def calculate_metrics(
    topk_prob: np.ndarray,
) -> tuple[int, float, float]:

    borderline = (
        (topk_prob > LOW_THRESHOLD)
        & (topk_prob < HIGH_THRESHOLD)
    )

    count = int(
        borderline.sum()
    )

    share_n = (
        count / N_ASSETS
    )

    share_k = (
        count / TOP_K
    )

    return (
        count,
        share_n,
        share_k,
    )


# ---------------------------------------------------------------------
# Single perturbation run
# ---------------------------------------------------------------------

def run_single(
    df: pd.DataFrame,
    hazard_sd_rtm: float,
    sigma: float,
    perturbation_seed: int,
) -> dict[str, float | int]:

    rng = np.random.default_rng(
        perturbation_seed
    )

    z = rng.normal(
        loc=0.0,
        scale=1.0,
        size=len(df),
    )

    hazard_base = (
        df["H_pluvial_v1_logrel"]
        .to_numpy(dtype=float)
    )

    # sigma is expressed relative to the full Rotterdam hazard SD.
    noise = (
        sigma
        * hazard_sd_rtm
        * z
    )

    hazard_perturbed = (
        hazard_base
        + noise
    )

    exposure = (
        df["E_hat_v0"]
        .to_numpy(dtype=float)
    )

    outcome = (
        df["Y_damage"]
        .to_numpy(dtype=int)
    )

    (
        alpha,
        beta_e,
        beta_h,
        diagnostics,
    ) = fit_model(
        exposure=exposure,
        hazard=hazard_perturbed,
        outcome=outcome,
    )

    topk_prob = compute_topk_probability(
        alpha=alpha,
        beta_e=beta_e,
        beta_h=beta_h,
        exposure=exposure,
        hazard=hazard_perturbed,
        k=TOP_K,
    )

    (
        borderline_count,
        borderline_share_n,
        borderline_share_k,
    ) = calculate_metrics(
        topk_prob
    )

    if sigma == 0:
        hazard_corr = 1.0
    else:
        hazard_corr = float(
            np.corrcoef(
                hazard_base,
                hazard_perturbed,
            )[0, 1]
        )

    return {
        "sigma": sigma,
        "perturbation_seed": perturbation_seed,
        "mcmc_seed": MCMC_SEED,
        "n_assets": N_ASSETS,
        "top_k": TOP_K,
        "prioritised_share": TOP_K / N_ASSETS,
        "hazard_sd_rtm": hazard_sd_rtm,
        "noise_sd": sigma * hazard_sd_rtm,
        "hazard_correlation": hazard_corr,
        "borderline_count": borderline_count,
        "borderline_share_N": borderline_share_n,
        "borderline_share_k": borderline_share_k,
        **diagnostics,
    }


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def load_existing_results() -> pd.DataFrame:

    if not RAW_PATH.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        RAW_PATH
    )

    print(
        f"Loaded {len(existing)} "
        "existing checkpoint rows."
    )

    return existing


def is_completed(
    results: pd.DataFrame,
    sigma: float,
    seed: int,
) -> bool:

    if results.empty:
        return False

    mask = (
        np.isclose(
            results["sigma"],
            sigma,
        )
        & (
            results["perturbation_seed"]
            == seed
        )
    )

    return bool(
        mask.any()
    )


def save_checkpoint(
    rows: list[dict[str, float | int]],
) -> pd.DataFrame:

    results = pd.DataFrame(
        rows
    )

    results = (
        results
        .sort_values(
            [
                "sigma",
                "perturbation_seed",
            ]
        )
        .reset_index(drop=True)
    )

    results.to_csv(
        RAW_PATH,
        index=False,
    )

    return results


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarise_results(
    results: pd.DataFrame,
) -> pd.DataFrame:

    summary_rows = []

    for sigma, group in results.groupby(
        "sigma",
        sort=True,
    ):

        values_n = (
            group["borderline_share_N"]
        )

        values_k = (
            group["borderline_share_k"]
        )

        if sigma == 0:

            p10_n = np.nan
            p90_n = np.nan
            p10_k = np.nan
            p90_k = np.nan

        else:

            p10_n = float(
                values_n.quantile(0.10)
            )

            p90_n = float(
                values_n.quantile(0.90)
            )

            p10_k = float(
                values_k.quantile(0.10)
            )

            p90_k = float(
                values_k.quantile(0.90)
            )

        summary_rows.append(
            {
                "sigma": sigma,
                "n_replicates": len(group),
                "median_borderline_count":
                    float(
                        group[
                            "borderline_count"
                        ].median()
                    ),
                "median_borderline_share_N":
                    float(
                        values_n.median()
                    ),
                "p10_borderline_share_N":
                    p10_n,
                "p90_borderline_share_N":
                    p90_n,
                "median_borderline_share_k":
                    float(
                        values_k.median()
                    ),
                "p10_borderline_share_k":
                    p10_k,
                "p90_borderline_share_k":
                    p90_k,
                "max_rhat":
                    float(
                        group[
                            "rhat_max"
                        ].max()
                    ),
                "min_ess_bulk":
                    float(
                        group[
                            "ess_bulk_min"
                        ].min()
                    ),
                "total_divergences":
                    int(
                        group[
                            "divergences"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    return summary


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def make_figure(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:

    baseline = float(
        summary.loc[
            summary["sigma"] == 0,
            "median_borderline_share_N",
        ].iloc[0]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    # Individual paired perturbation paths.
    for seed in PERTURBATION_SEEDS:

        seed_data = results[
            (
                results[
                    "perturbation_seed"
                ]
                == seed
            )
            & (
                results["sigma"]
                > 0
            )
        ].sort_values("sigma")

        if len(seed_data) != 4:
            continue

        x = np.concatenate(
            [
                [0.0],
                seed_data[
                    "sigma"
                ].to_numpy(),
            ]
        )

        y = np.concatenate(
            [
                [baseline],
                seed_data[
                    "borderline_share_N"
                ].to_numpy(),
            ]
        ) * 100

        ax.plot(
            x,
            y,
            linewidth=0.7,
            alpha=0.25,
            color="0.65",
        )

    x = summary[
        "sigma"
    ].to_numpy()

    median = (
        summary[
            "median_borderline_share_N"
        ].to_numpy()
        * 100
    )

    ax.plot(
        x,
        median,
        marker="o",
        linewidth=2.0,
        color="black",
        label="Median",
    )

    nonzero = (
        summary["sigma"] > 0
    )

    x_band = summary.loc[
        nonzero,
        "sigma",
    ].to_numpy()

    lower = (
        summary.loc[
            nonzero,
            "p10_borderline_share_N",
        ].to_numpy()
        * 100
    )

    upper = (
        summary.loc[
            nonzero,
            "p90_borderline_share_N",
        ].to_numpy()
        * 100
    )

    ax.fill_between(
        x_band,
        lower,
        upper,
        alpha=0.20,
        color="0.6",
        label="Empirical 10th–90th percentile",
    )

    ax.set_xlabel(
        "Hazard perturbation σ "
        "(fraction of Rotterdam hazard SD)"
    )

    ax.set_ylabel(
        "Borderline share of assets (%)"
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, hazard_sd_rtm = (
        load_experiment_data()
    )

    existing = (
        load_existing_results()
    )

    rows = (
        existing
        .to_dict("records")
        if not existing.empty
        else []
    )

    # -------------------------------------------------------------
    # sigma = 0 baseline
    # -------------------------------------------------------------

    baseline_seed = (
        PERTURBATION_SEEDS[0]
    )

    if not is_completed(
        existing,
        sigma=0.0,
        seed=baseline_seed,
    ):

        print(
            "\nRunning baseline "
            "sigma = 0.00"
        )

        result = run_single(
            df=df,
            hazard_sd_rtm=hazard_sd_rtm,
            sigma=0.0,
            perturbation_seed=baseline_seed,
        )

        rows.append(result)

        existing = save_checkpoint(
            rows
        )

    else:

        print(
            "\nBaseline already complete."
        )

    # -------------------------------------------------------------
    # Repeated non-zero perturbations
    # -------------------------------------------------------------

    for seed in PERTURBATION_SEEDS:

        # One z-vector per seed. run_single regenerates exactly
        # the same z for every sigma because the seed is identical.
        print(
            f"\nPerturbation seed {seed}"
        )

        for sigma in SIGMAS[1:]:

            if is_completed(
                existing,
                sigma=sigma,
                seed=seed,
            ):

                print(
                    f"  sigma={sigma:.2f} "
                    "already complete"
                )

                continue

            print(
                f"  sigma = {sigma:.2f}"
            )

            result = run_single(
                df=df,
                hazard_sd_rtm=hazard_sd_rtm,
                sigma=sigma,
                perturbation_seed=seed,
            )

            rows.append(result)

            existing = save_checkpoint(
                rows
            )

            print(
                "    borderline = "
                f"{result['borderline_count']} "
                f"({result['borderline_share_N']:.4%} of N; "
                f"{result['borderline_share_k']:.2%} of k)"
            )

            print(
                "    diagnostics: "
                f"R-hat max={result['rhat_max']:.4f}, "
                f"ESS bulk min={result['ess_bulk_min']:.0f}, "
                f"divergences={result['divergences']}"
            )

    # -------------------------------------------------------------
    # Final summary + figure
    # -------------------------------------------------------------

    results = pd.read_csv(
        RAW_PATH
    )

    summary = summarise_results(
        results
    )

    make_figure(
        results=results,
        summary=summary,
    )

    print("\n========================================")
    print("FINAL SUMMARY")
    print("========================================")

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Raw results: {RAW_PATH}"
    )
    print(
        f"Summary:     {SUMMARY_PATH}"
    )
    print(
        f"Figure PDF:  {FIGURE_PDF}"
    )
    print(
        f"Figure PNG:  {FIGURE_PNG}"
    )


if __name__ == "__main__":
    main()