from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CITIES = ["RTM", "HAM", "DON"]
PROFILE = "low_event_weakly_informative"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    root = project_root()
    out_dir = root / "outputs" / "phase3" / "cross_city_summary" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for city in CITIES:
        city_dir = root / "outputs" / "phase3" / city
        candidates = list(city_dir.glob(f"prior_predictive_summary_*{PROFILE}*.json"))
        if not candidates:
            print(f"[skip] no prior predictive summary for {city}")
            continue

        path = candidates[0]
        data = json.loads(path.read_text(encoding="utf-8"))
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
        label="Prior predictive event rate, p05–p95",
    )
    ax.scatter(x, obs, marker="x", label="Observed synthetic event rate")

    ax.set_xticks(x)
    ax.set_xticklabels([r["city"] for r in rows])
    ax.set_ylabel("Event rate")
    ax.set_title("Prior predictive event-rate check")
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