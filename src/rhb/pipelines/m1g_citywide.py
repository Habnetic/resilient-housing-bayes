"""M1-G2: synthetic full-city Bayesian B0/B1 comparison, without changing frozen M1.

Run from the scientific repository root, with PYTHONPATH=src:
    python -m rhb.pipelines.m1g_citywide --city all --check
    python -m rhb.pipelines.m1g_citywide --city DON --prepare
    python -m rhb.pipelines.m1g_citywide --city DON --smoke --gamma -0.8
    python -m rhb.pipelines.m1g_citywide --city DON --fit --gamma -0.8
    python -m rhb.pipelines.m1g_citywide --city DON --score --gamma -0.8

All results are synthetic and conditional on model assumptions. Citywide B0 and B1
use the same exact eligible building IDs, outcomes, budget, and tie-breaks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from rhb.experiments.m1f_population_sampling import derived_seed, load_universe as load_rtm
from rhb.experiments.m1g_cross_city_sampling import load_config as load_g1, load_universe as load_ham_don

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/m1g_citywide_v0.1.json"
LOG = logging.getLogger("rhb.m1g.citywide")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config() -> dict[str, Any]:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    if cfg["experiment"] != "M1-G2" or set(cfg["cities"]) != {"RTM", "HAM", "DON"}:
        raise ValueError("Invalid M1-G2 city configuration")
    if cfg["gamma_scenarios"] != [0.0, -0.8] or cfg["terrain_scale_m"] != 0.5:
        raise ValueError("Registered synthetic scenarios or terrain transformation changed")
    if cfg["baseline_prior_sigma"] != 2.5 or cfg["terrain_prior_sigma"] != 0.5:
        raise ValueError("Frozen M1 prior specification changed")
    return cfg


def city_dir(city: str) -> Path:
    return ROOT / "outputs" / "phase4" / city / "m1g_citywide_v0.1"


def code_fingerprint() -> dict[str, str]:
    files = [
        CONFIG_PATH,
        Path(__file__),
        ROOT / "src/rhb/experiments/m1f_population_sampling.py",
        ROOT / "src/rhb/experiments/m1g_cross_city_sampling.py",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def load_city(city: str, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read full measured population, auditing Phase 3 alignment and input integrity."""
    info = cfg["cities"][city]
    if city == "RTM":
        feat_path = (ROOT / info["features_path"]).resolve()
        terrain_path = (ROOT / info["terrain_path"]).resolve()
        features = pd.read_parquet(feat_path, columns=["bldg_id"])
        if len(features) != info["expected_phase3"] or not features["bldg_id"].is_unique:
            raise ValueError("RTM Phase 3 population is not the frozen universe")
        universe = load_rtm(feat_path, terrain_path, info["expected_eligible"])
        fingerprints = {"rtm_phase3_features": digest(feat_path), "rtm_terrain": digest(terrain_path)}
    else:
        universe, fingerprints = load_ham_don(city, load_g1())
    if len(universe) != info["expected_eligible"]:
        raise ValueError(f"{city}: expected {info['expected_eligible']} eligible buildings, got {len(universe)}")
    universe = universe.copy()
    universe["bldg_id"] = universe["bldg_id"].astype(str)
    if universe["bldg_id"].isna().any() or not universe["bldg_id"].is_unique:
        raise ValueError(f"{city}: duplicate or missing building IDs")
    required = ["E_hat_v0", "H_pluvial_v1_logrel", "T50", "T"]
    if not np.isfinite(universe[required].to_numpy(dtype=float)).all():
        raise ValueError(f"{city}: nonfinite terrain/exposure/hazard predictors")
    t = np.arcsinh(universe["T50"].to_numpy(dtype=float) / cfg["terrain_scale_m"])
    if not np.allclose(t, universe["T"].to_numpy(dtype=float), atol=1e-12, rtol=0):
        raise ValueError(f"{city}: terrain transform inconsistent with RTM M1")
    universe = universe[["bldg_id", *required]].sort_values("bldg_id", kind="stable").reset_index(drop=True)
    return universe, fingerprints


def gamma_label(gamma: float) -> str:
    if gamma == 0.0:
        return "gamma_0"
    if gamma == -0.8:
        return "gamma_neg0p8"
    raise ValueError("Only preregistered gamma scenarios are allowed")


def scenario_for(universe: pd.DataFrame, city: str, gamma: float, cfg: dict[str, Any]) -> pd.DataFrame:
    e, h, t = (universe[key].to_numpy(dtype=np.float64) for key in
               ("E_hat_v0", "H_pluvial_v1_logrel", "T"))
    p = expit(cfg["alpha_generating"] + cfg["beta_E_generating"] * e
              + cfg["beta_H_generating"] * h + gamma * t)
    # Common random uniform stream across both gamma scenarios within each city.
    seed = derived_seed("M1-G2-outcomes", cfg["outcome_seed"], city)
    uniforms = np.random.default_rng(seed).random(len(p))
    result = universe.copy()
    result["p_true"] = p
    result["Y_synthetic"] = (uniforms < p).astype(np.int8)
    return result


def provenance(cfg: dict[str, Any], fingerprints: dict[str, str]) -> dict[str, Any]:
    return {"experiment": "M1-G2", "configuration": cfg,
            "source_hashes": fingerprints, "code_hashes": code_fingerprint()}


def audit(city: str, cfg: dict[str, Any]) -> None:
    u, hashes = load_city(city, cfg)
    if len(u) != cfg["cities"][city]["expected_eligible"]:
        raise ValueError("City population count changed")
    label = "terrain unavailable" if city == "RTM" else "terrain complete"
    LOG.info("%s PASS: Phase3=%d; terrain-eligible=%d; excluded=%d (%s)", city,
             cfg["cities"][city]["expected_phase3"], len(u),
             cfg["cities"][city]["expected_phase3"] - len(u), label)
    LOG.info("%s T50 range [%.3f, %.3f]m; source fingerprints=%s",
             city, float(u["T50"].min()), float(u["T50"].max()), sorted(hashes))


def prepare(city: str, cfg: dict[str, Any]) -> None:
    u, input_hashes = load_city(city, cfg)
    dest = city_dir(city)
    manifest_path = dest / "preparation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Already prepared: {manifest_path}; never overwrite frozen outputs")
    dest.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    for gamma in cfg["gamma_scenarios"]:
        file = dest / f"{gamma_label(float(gamma))}_scenario.parquet"
        if file.exists():
            raise FileExistsError(file)
        frame = scenario_for(u, city, float(gamma), cfg)
        frame.to_parquet(file, index=False)
        cases[gamma_label(float(gamma))] = {
            "gamma_true": float(gamma), "file": file.name, "sha256": digest(file),
            "n": len(frame), "events": int(frame["Y_synthetic"].sum()),
            "mean_p_true": float(frame["p_true"].mean()),
        }
    manifest = {**provenance(cfg, input_hashes), "city": city,
                "eligible_count": len(u), "phase3_count": cfg["cities"][city]["expected_phase3"],
                "outcome_evidence": "synthetic", "scenarios": cases,
                "status": "prepared_not_fitted"}
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOG.info("%s prepared: %d eligible buildings, 2 scenarios -> %s", city, len(u), dest)


def read_frozen(city: str, gamma: float, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dest = city_dir(city)
    file = dest / f"{gamma_label(gamma)}_scenario.parquet"
    path = dest / "preparation_manifest.json"
    if not path.is_file() or not file.is_file():
        raise FileNotFoundError(f"Prepare {city} scenarios first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["city"] != city or manifest["configuration"] != cfg:
        raise ValueError("Prepared configuration changed")
    if manifest["code_hashes"] != code_fingerprint():
        raise ValueError("Implementation changed after preparation; do not silently reuse")
    if manifest["scenarios"][gamma_label(gamma)]["sha256"] != digest(file):
        raise ValueError("Prepared scenario SHA-256 mismatch")
    _, source_hashes = load_city(city, cfg)
    if manifest["source_hashes"] != source_hashes:
        raise ValueError("Source data changed after preparation")
    frame = pd.read_parquet(file)
    if len(frame) != cfg["cities"][city]["expected_eligible"] or not frame["bldg_id"].is_unique:
        raise ValueError("Prepared candidate population is not complete")
    if not np.isfinite(frame["p_true"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite synthetic ground truth")
    regenerated = scenario_for(frame[["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel", "T50", "T"]], city, gamma, cfg)
    if (not np.array_equal(frame["Y_synthetic"].to_numpy(dtype=int), regenerated["Y_synthetic"].to_numpy(dtype=int))
            or not np.allclose(frame["p_true"].to_numpy(dtype=float), regenerated["p_true"].to_numpy(dtype=float), atol=1e-14, rtol=0)):
        raise ValueError("Synthetic probabilities or outcomes cannot be reproduced")
    return frame, manifest


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k; ties broken by canonical sorted building ID order."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not 1 <= k <= len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid top-k input")
    return np.lexsort((np.arange(len(values)), -values))[:k]


def posterior_indices(posterior: Any, max_per_chain: int) -> dict[str, np.ndarray]:
    """Thin evenly *within each chain* for bounded citywide decision computation."""
    names = ["alpha", "beta_E", "beta_H"] + (["gamma_T"] if "gamma_T" in posterior else [])
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        tensor = np.asarray(posterior[name].values, dtype=np.float64)
        if tensor.ndim != 2:
            raise ValueError(f"Posterior {name} must have chain, draw dimensions")
        idx = np.linspace(0, tensor.shape[1] - 1, min(tensor.shape[1], max_per_chain), dtype=np.int64)
        arrays[name] = tensor[:, idx].reshape(-1)
    return arrays


def score_draws(frame: pd.DataFrame, draws: dict[str, np.ndarray], k: int) -> pd.DataFrame:
    """All-building posterior top-k probabilities using O(N) additional RAM.

    Every selected posterior draw ranks the *entire* eligible city. No chunk-local
    top-k calculation, no geometric imputation, no 5,000-building decision subset.
    """
    ids = frame["bldg_id"].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate building IDs")
    e, h, t = (frame[col].to_numpy(dtype=np.float64) for col in
               ("E_hat_v0", "H_pluvial_v1_logrel", "T"))
    n = len(ids)
    if not 0 < k <= n:
        raise ValueError("Invalid citywide budget")
    size = len(draws["alpha"])
    if size < 2 or any(len(a) != size or not np.isfinite(a).all() for a in draws.values()):
        raise ValueError("Invalid posterior coefficient draws")
    means = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int32)
    rank_sum = np.zeros(n, dtype=np.float64)
    rank_square_sum = np.zeros(n, dtype=np.float64)
    ranks = np.empty(n, dtype=np.int32)
    for i in range(size):
        logits = draws["alpha"][i] + draws["beta_E"][i] * e + draws["beta_H"][i] * h
        if "gamma_T" in draws:
            logits += draws["gamma_T"][i] * t
        p = expit(logits)
        order = topk_indices(p, n)  # complete ranks, not local chunk selections
        ranks[order] = np.arange(1, n + 1)
        counts[order[:k]] += 1
        means += p
        rank_sum += ranks
        rank_square_sum += ranks.astype(np.float64) ** 2
    means /= size
    p_topk = counts.astype(np.float64) / size
    if not np.isclose(p_topk.sum(), k, rtol=0, atol=1e-6):
        raise AssertionError("Sum of citywide inclusion probabilities must equal k")
    rank_sd = np.sqrt(np.maximum(rank_square_sum / size - (rank_sum / size) ** 2, 0))
    selected = np.zeros(n, dtype=bool)
    selected[topk_indices(means, k)] = True
    oracle = np.zeros(n, dtype=bool)
    oracle[topk_indices(frame["p_true"].to_numpy(dtype=float), k)] = True
    return pd.DataFrame({
        "bldg_id": ids,
        "p_true_synthetic": frame["p_true"].to_numpy(dtype=float),
        "T50_m": frame["T50"].to_numpy(dtype=float),
        "p_mean": means,
        "p_topk": p_topk,
        "rank_sd": rank_sd,
        "selected_by_p_mean": selected,
        "selected_by_oracle": oracle,
    })


def comparison_frame(b0: pd.DataFrame, b1: pd.DataFrame, k: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not b0["bldg_id"].equals(b1["bldg_id"]):
        raise ValueError("B0/B1 eligible building IDs or order differ")
    a = b0["selected_by_p_mean"].to_numpy(dtype=bool)
    b = b1["selected_by_p_mean"].to_numpy(dtype=bool)
    if int(a.sum()) != k or int(b.sum()) != k:
        raise ValueError("B0/B1 top-k cardinalities differ")
    overlap = int(np.logical_and(a, b).sum())
    if overlap + int(np.logical_and(~a, b).sum()) != k:
        raise AssertionError("B1 entry count is inconsistent")
    if not np.array_equal(b0["p_true_synthetic"], b1["p_true_synthetic"]):
        raise ValueError("Different synthetic ground truth")
    delta = b1["p_topk"].to_numpy(float) - b0["p_topk"].to_numpy(float)
    truth = b0["p_true_synthetic"].to_numpy(float)
    oracle = b0["selected_by_oracle"].to_numpy(bool)
    oracle_val = float(truth[oracle].sum())
    regret0 = max(0., oracle_val - float(truth[a].sum()))
    regret1 = max(0., oracle_val - float(truth[b].sum()))
    group = np.select([a & b, (~a) & b, a & (~b)],
                      ["stayed_selected", "entered_B1", "left_B1"], default="not_selected")
    result = pd.DataFrame({
        "bldg_id": b0["bldg_id"], "T50_m": b0["T50_m"],
        "p_mean_B0": b0["p_mean"], "p_mean_B1": b1["p_mean"],
        "p_topk_B0": b0["p_topk"], "p_topk_B1": b1["p_topk"],
        "delta_p_topk_B1_minus_B0": delta,
        "selected_B0": a, "selected_B1": b,
        "decision_change": group,
        "rank_sd_B0": b0["rank_sd"], "rank_sd_B1": b1["rank_sd"],
    })
    metrics = {
        "k": k, "n": len(b0), "overlap": overlap, "entries_B1": int(((~a) & b).sum()),
        "exits_B1": int((a & (~b)).sum()),
        "borderline_B0": int(((b0["p_topk"] > 0.2) & (b0["p_topk"] < 0.8)).sum()),
        "borderline_B1": int(((b1["p_topk"] > 0.2) & (b1["p_topk"] < 0.8)).sum()),
        "mean_abs_delta_inclusion": float(np.abs(delta).mean()),
        "regret_B0": regret0, "regret_B1": regret1,
        "regret_improvement_B0_minus_B1": regret0 - regret1,
    }
    return result, metrics


def fit_one(frame: pd.DataFrame, name: str, cfg: dict[str, Any], dest: Path, smoke: bool, seed: int) -> None:
    """Full-population NUTS fit. Do not retain huge observation-scale predictive arrays."""
    if dest.exists():
        raise FileExistsError(f"Fit output already exists, refusing overwrite: {dest}")
    import arviz as az
    import pymc as pm

    if smoke:
        # Deliberate subset: smoke outputs live in a separate path and cannot
        # masquerade as full-city inference.
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(frame), min(1200, len(frame)), replace=False))
        frame = frame.iloc[indices].reset_index(drop=True)
    e, h, t = (frame[x].to_numpy(dtype=np.float64) for x in
               ("E_hat_v0", "H_pluvial_v1_logrel", "T"))
    y = frame["Y_synthetic"].to_numpy(dtype=np.int8)
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=cfg["baseline_prior_sigma"])
        beta_e = pm.Normal("beta_E", mu=0, sigma=cfg["baseline_prior_sigma"])
        beta_h = pm.Normal("beta_H", mu=0, sigma=cfg["baseline_prior_sigma"])
        logit = alpha + beta_e * e + beta_h * h
        if name == "B1":
            gamma = pm.Normal("gamma_T", mu=0, sigma=cfg["terrain_prior_sigma"])
            logit = logit + gamma * t
        pm.Bernoulli("Y_obs", logit_p=logit, observed=y)
        mcmc = cfg["mcmc"] if not smoke else {"chains": 2, "draws": 250, "tune": 250, "target_accept": 0.9, "cores": 2}
        idata = pm.sample(
            draws=mcmc["draws"], tune=mcmc["tune"], chains=mcmc["chains"],
            cores=mcmc["cores"], target_accept=mcmc["target_accept"],
            random_seed=seed, progressbar=True, idata_kwargs={"log_likelihood": False},
        )
    names = ["alpha", "beta_E", "beta_H"] + (["gamma_T"] if name == "B1" else [])
    summary = az.summary(idata, var_names=names, hdi_prob=0.95)
    diverged = int(idata.sample_stats["diverging"].sum().item())
    diagnostics = {"n_fit": len(frame), "model": name, "smoke": smoke,
                   "rhat_max": float(summary["r_hat"].max()),
                   "ess_bulk_min": float(summary["ess_bulk"].min()),
                   "ess_tail_min": float(summary["ess_tail"].min()),
                   "divergences": diverged}
    dest.mkdir(parents=True, exist_ok=False)
    az.to_netcdf(idata, dest / "idata.nc")
    summary.to_csv(dest / "posterior_summary.csv")
    diagnostics["idata_sha256"] = digest(dest / "idata.nc")
    diagnostics["code_hashes"] = code_fingerprint()
    (dest / "fit_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOG.info("Fit %s complete: %s", name, diagnostics)


def predictive_rate_check(frame: pd.DataFrame, draws: dict[str, np.ndarray],
                          cfg: dict[str, Any], seed: int, n_check: int = 100) -> dict[str, Any]:
    """Aggregate prior/posterior predictive synthetic event-rate replication.

    Draw complete replicated Y vectors at up to n_check posterior parameter draws.
    This does NOT constitute a spatial PPC or observed-outcome validation.
    """
    rng = np.random.default_rng(seed)
    e = frame["E_hat_v0"].to_numpy(dtype=np.float64)
    h = frame["H_pluvial_v1_logrel"].to_numpy(dtype=np.float64)
    t = frame["T"].to_numpy(dtype=np.float64)
    n = len(frame)
    observed = float(frame["Y_synthetic"].mean())
    posterior_n = len(draws["alpha"])
    ix = np.linspace(0, posterior_n - 1, min(n_check, posterior_n), dtype=int)
    prior_rates = []
    posterior_rates = []
    for j in ix:
        al, be, bh = rng.normal(0, cfg["baseline_prior_sigma"], size=3)
        lp_prior = al + be * e + bh * h
        if "gamma_T" in draws:
            lp_prior += rng.normal(0, cfg["terrain_prior_sigma"]) * t
        prior_rates.append(float(np.mean(rng.random(n) < expit(lp_prior))))
        lp = draws["alpha"][j] + draws["beta_E"][j] * e + draws["beta_H"][j] * h
        if "gamma_T" in draws:
            lp += draws["gamma_T"][j] * t
        posterior_rates.append(float(np.mean(rng.random(n) < expit(lp))))
    def stats(values: list[float]) -> dict[str, float]:
        return {"p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95))}
    return {"observed_synthetic_rate": observed, "prior_predictive_rate": stats(prior_rates),
            "posterior_predictive_rate": stats(posterior_rates),
            "replications": len(ix), "check_kind": "aggregate_event_rate_only",
            "status": "recorded_not_scientifically_reviewed"}


def apply_scoring(frame: pd.DataFrame, fit_dir: Path, k: int,
                  draws_per_chain: int, cfg: dict[str, Any], seed: int) -> None:
    """Score all eligible buildings from fitted NUTS coefficient draws."""
    import arviz as az
    if not (fit_dir / "idata.nc").exists():
        raise FileNotFoundError("Fit first")
    result_path = fit_dir / "citywide_decisions.parquet"
    if result_path.exists():
        raise FileExistsError(result_path)
    fit_meta = json.loads((fit_dir / "fit_diagnostics.json").read_text(encoding="utf-8"))
    if fit_meta["n_fit"] != len(frame) or fit_meta["smoke"]:
        raise ValueError("Refusing citywide score from partial/smoke fit")
    if fit_meta["code_hashes"] != code_fingerprint() or fit_meta["idata_sha256"] != digest(fit_dir / "idata.nc"):
        raise ValueError("Fitted posterior SHA256 or code fingerprint changed")
    idata = az.from_netcdf(fit_dir / "idata.nc")
    draws = posterior_indices(idata.posterior, draws_per_chain)
    result = score_draws(frame, draws, k)
    if len(result) != len(frame):
        raise AssertionError("Incomplete citywide scoring")
    result.to_parquet(result_path, index=False)
    checks = predictive_rate_check(frame, draws, cfg, seed)
    metadata = {"n": len(frame), "k": k, "scoring_draws": len(draws["alpha"]),
                "draw_limit_per_chain": draws_per_chain, "result_sha256": digest(result_path),
                "interpretation": "synthetic outcomes; fitted on all eligible buildings (unless this is a smoke fit)",
                "inclusion_probability_se": "Monte Carlo approximation from subsampled posterior draws",
                "predictive_checks": checks,
                "candidate_universe": "all terrain-eligible buildings, no 5000-building subset"}
    (fit_dir / "scoring_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOG.info("Scored %d citywide buildings, k=%d, %d coefficient draws", len(result), k, len(draws["alpha"]))


def compare(city: str, gamma: float, cfg: dict[str, Any], frame: pd.DataFrame) -> None:
    dest = city_dir(city) / gamma_label(gamma)
    path = dest / "citywide_comparison.parquet"
    if path.exists():
        raise FileExistsError(path)
    b0_path = dest / "B0" / "citywide_decisions.parquet"
    b1_path = dest / "B1" / "citywide_decisions.parquet"
    for src in (b0_path, b1_path):
        if not src.exists():
            raise FileNotFoundError(f"Full-city scoring missing: {src}")
    for model in ("B0", "B1"):
        fit_dir = dest / model
        fit_record = json.loads((fit_dir / "fit_diagnostics.json").read_text(encoding="utf-8"))
        if fit_record["smoke"] or fit_record["n_fit"] != len(frame):
            raise ValueError("Citywide comparison requires two full-universe model fits")
        score_record = json.loads((fit_dir / "scoring_manifest.json").read_text(encoding="utf-8"))
        if score_record["n"] != len(frame) or score_record["result_sha256"] != digest(fit_dir / "citywide_decisions.parquet"):
            raise ValueError(f"{model} citywide result integrity failed")
    b0, b1 = pd.read_parquet(b0_path), pd.read_parquet(b1_path)
    k = max(1, round(cfg["selection_share"] * len(frame)))
    result, metrics = comparison_frame(b0, b1, k)
    result.to_parquet(path, index=False)
    record = {"experiment": "M1-G2", "city": city, "gamma_true": gamma,
              "n_eligible": len(frame), "n_phase3": cfg["cities"][city]["expected_phase3"],
              "outcome_evidence": "synthetic", "comparison_sha256": digest(path),
              "metrics": metrics, "status": "computed_not_scientifically_reviewed",
              "limitations": ["Synthetic outcomes", "No observed flood impacts",
                              "Citywise refit, not geographic coefficient transfer",
                              "B0 and B1 use equal candidate sets and same synthetic labels"]}
    (dest / "comparison_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOG.info("%s gamma=%s citywide comparison: %s", city, gamma, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=["RTM", "HAM", "DON", "all"], required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    for flag in ["check", "prepare", "smoke", "fit", "score", "compare"]:
        action.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--model", choices=["B0", "B1", "both"], default="both",
                        help="Fit/score only one model to resume a partially finished city")
    parser.add_argument("--gamma", type=float, default=-0.8,
                        help="registered gamma scenario, 0 or -0.8 (default -0.8)")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("M1-G2 requires scientific Python 3.11")
    cfg = read_config()
    if args.gamma not in cfg["gamma_scenarios"]:
        raise ValueError("Only preregistered gamma=0 and gamma=-0.8 are supported")
    if args.city == "all" and args.smoke:
        raise ValueError("Run smoke on one city to inspect numerical behaviour")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cities = list(cfg["cities"]) if args.city == "all" else [args.city]
    for city in cities:
        if args.check:
            audit(city, cfg)
        elif args.prepare:
            prepare(city, cfg)
        else:
            frame, manifest = read_frozen(city, args.gamma, cfg)
            stage = f"smoke_{gamma_label(args.gamma)}" if args.smoke else gamma_label(args.gamma)
            root = city_dir(city) / stage
            if args.fit or args.smoke:
                for model in (("B0", "B1") if args.model == "both" else (args.model,)):
                    fit_one(frame, model, cfg, root / model, args.smoke,
                            derived_seed("M1-G2-NUTS", city, args.gamma, model))
            elif args.score:
                k = max(1, round(cfg["selection_share"] * len(frame)))
                for model in (("B0", "B1") if args.model == "both" else (args.model,)):
                    apply_scoring(frame, root / model, k, cfg["scoring"]["draws_per_chain"], cfg,
                                  derived_seed("M1-G2-PPC", city, args.gamma, model))
            elif args.compare:
                compare(city, args.gamma, cfg, frame)


if __name__ == "__main__":
    main()
