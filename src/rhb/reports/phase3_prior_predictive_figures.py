from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CITIES = ["RTM", "HAM", "DON"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = project_root()
    out_dir = root / "outputs" / "phase3" / "cross_city_summary" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for city in CITIES:
        city_dir = root / "outputs" / "phase3" / city

        # Use the baseline priors from the main Phase 3 model
        path = city_dir / "prior_predictive_summary.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing baseline prior predictive summary: {path}"
            )

        data = json.loads(path.read_text(encoding="utf-8"))

        # Verify consistency with the published Bayesian model
        expected_priors = {
            "Normal(0, 2.5)",
            "Normal(0.0, 2.5)",
        }

        for parameter in ("alpha", "beta_E", "beta_H"):
            actual = data.get("priors", {}).get(parameter)

            if actual not in expected_priors:
                raise ValueError(
                    f"Unexpected prior for {parameter} in {path}: {actual}"
                )

        er = data["prior_event_rate_percentiles"]

        rows.append(
            {
                "city": city,
                "p01": er["p01"],
                "p05": er["p05"],
                "p10": er["p10"],
                "p50": er["p50"],
                "p90": er["p90"],
                "p95": er["p95"],
                "p99": er["p99"],
                "observed": data["observed_synthetic_event_rate"],
            }
        )

    if not rows:
        raise RuntimeError("No prior predictive summaries found.")

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(rows))
    med = np.array([r["p50"] for r in rows])
    lo = np.array([r["p05"] for r in rows])
    hi = np.array([r["p95"] for r in rows])
    obs = np.array([r["observed"] for r in rows])

    ax.errorbar(
        x,
        med,
        yerr=[med - lo, hi - med],
        fmt="o",
        capsize=4,
        label="Prior-implied mean event probability (P5-P95)",
    )
    ax.scatter(x, obs, marker="x", label="Observed synthetic event rate")

    ax.set_xticks(x)
    ax.set_xticklabels([r["city"] for r in rows])
    ax.set_ylabel("Mean event probability / event rate")
    ax.set_title("Prior-implied event rate (baseline priors)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png = out_dir / "fig_prior_predictive_event_rate.png"
    pdf = out_dir / "fig_prior_predictive_event_rate.pdf"

    fig.tight_layout()
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


if __name__ == "__main__":
    main()