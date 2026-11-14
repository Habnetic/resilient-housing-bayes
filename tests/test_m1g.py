"""Pure M1-G protocol tests. Run alongside existing scientific repository tests."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from rhb.experiments.m1g_cross_city_sampling import (
    generate_city_scenarios, normalize_ids, select_population,
    validate_config, verified_ham_batches,
)
from rhb.pipelines.run_m1g import cases_for
from rhb.pipelines.summarize_m1f import check_result_tables, diagnostics


def config():
    return {
        "experiment": "M1-G", "cities": {"HAM": {"expected_buildings": 341530,
                         "expected_terrain_files": 342}, "DON": {"expected_buildings": 7755}},
        "population_ids": ["city01"], "sample_size": 5000, "train_size": 4000,
        "test_size": 1000, "gamma_true_values": [0.0, -0.8], "simulation_seeds": [101, 202],
        "population_seed": 201, "split_seed": 202, "outcome_seed_prefix": "test-m1g",
        "generator_alpha": -3.0, "generator_beta_E": 1.0, "generator_beta_H": 1.0,
        "terrain_transform_scale_m": 0.5, "gamma_prior_sigma": 0.5,
        "selection_share": 0.01, "mcmc": {"chains": 2, "draws": 500, "tune": 500, "target_accept": 0.9},
        "frozen_reference": {"rtm_m1_tag": "phase4-m1-complete-v1.0"},
    }


def universe(n=5200):
    ix = np.arange(n)
    return pd.DataFrame({
        "bldg_id": (ix + 1).astype(str),
        "E_hat_v0": np.sin(ix / 30),
        "H_pluvial_v1_logrel": np.cos(ix / 19) * 0.002,
        "T50": np.sin(ix / 11) * 0.3,
        "T": np.arcsinh(np.sin(ix / 11) * 0.3 / 0.5),
    })


def test_frozen_configuration_accepts_registered():
    validate_config(config())


@pytest.mark.parametrize("change", [
    ("gamma_true_values", [0.0, -0.4]),
    ("sample_size", 4000),
    ("gamma_prior_sigma", 2.5),
    ("selection_share", 0.1),
])
def test_configuration_changes_refused(change):
    cfg = config()
    cfg[change[0]] = change[1]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_city_sample_reproducible_and_disjoint_split():
    cfg = config()
    u = universe()
    one = select_population(u, "HAM", cfg)
    two = select_population(u, "HAM", cfg)
    pd.testing.assert_frame_equal(one, two)
    assert len(one) == 5000 and one.bldg_id.is_unique
    assert one.split.value_counts().to_dict() == {"train": 4000, "test": 1000}
    assert len(cases_for(cfg, smoke=True)) == 2
    assert len(cases_for(cfg, smoke=False)) == 4


def test_scenarios_reuse_frozen_generator_and_paired_truth():
    cfg = config()
    pop = select_population(universe(), "HAM", cfg)
    out = generate_city_scenarios(pop, "HAM", cfg)
    assert len(out) == 20000 and "Y_m1g" in out.columns
    for gamma in (0.0, -0.8):
        rows = out.loc[(out.gamma_true == gamma) & (out.simulation_seed == 101)]
        np.testing.assert_allclose(
            rows.p_true.to_numpy(),
            expit(-3.0 + rows.E_hat_v0 + rows.H_pluvial_v1_logrel + gamma * rows["T"]),
            atol=1e-14, rtol=0,
        )
        assert rows.bldg_id.is_unique
    first = out.loc[(out.gamma_true == 0.0) & (out.simulation_seed == 101)]
    second = out.loc[(out.gamma_true == -0.8) & (out.simulation_seed == 101)]
    assert first.bldg_id.reset_index(drop=True).equals(second.bldg_id.reset_index(drop=True))
    assert first.split.reset_index(drop=True).equals(second.split.reset_index(drop=True))
    assert not out.Y_m1g.isna().any()


def test_city_seed_independence():
    cfg = config()
    pop = select_population(universe(), "HAM", cfg)
    ham = generate_city_scenarios(pop, "HAM", cfg)
    don = generate_city_scenarios(pop, "DON", cfg)
    assert not ham.Y_m1g.equals(don.Y_m1g)


def test_normalize_ids_no_silent_truncation():
    with pytest.raises(ValueError, match="noninteger"):
        normalize_ids(pd.Series([1.25, 2.0]), "test")
    with pytest.raises(ValueError, match="null"):
        normalize_ids(pd.Series([1, None]), "test")
    assert normalize_ids(pd.Series([1, 2]), "test").tolist() == ["1", "2"]
    assert normalize_ids(pd.Series(["001", "002"]), "test").tolist() == ["001", "002"]


def test_ham_missing_manifest_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        verified_ham_batches(tmp_path, tmp_path / "nonexistent.json", 1, expected_rows=1)


def test_synthetic_comparison_requires_both_models():
    fits = pd.DataFrame({"population_id": ["city01", "city01"],
                         "gamma_true": [0., 0.], "simulation_seed": [101, 101],
                         "model": ["B0", "B1"], "expected_regret": [0.2, 0.1]})
    comps = pd.DataFrame({"population_id": ["city01"], "gamma_true": [0.],
                          "simulation_seed": [101], "k": [10], "overlap": [9],
                          "expected_regret_B0": [0.2], "expected_regret_B1": [0.1],
                          "regret_improvement_B0_minus_B1": [0.1]})
    with pytest.raises(ValueError):
        check_result_tables(fits, comps, populations=1, seeds=1)  # missing gamma=-0.8


def test_bad_mcmc_flagged_not_silenced():
    fits = pd.DataFrame({"rhat_max": [1.05], "ess_bulk_min": [300],
                         "ess_tail_min": [350], "divergences": [2]})
    flags = diagnostics(fits, config().get("diagnostic_limits", {
        "max_rhat": 1.01, "min_bulk_ess": 400, "min_tail_ess": 400,
    }))
    assert len(flags) == 1 and bool(flags.iloc[0].flag_divergences)


def test_hamburg_batch_manifest_checksum_and_location(tmp_path, monkeypatch):
    import json
    from rhb.experiments import m1g_cross_city_sampling as mod
    root = tmp_path / "resilient-housing-bayes"
    root.mkdir()
    data_root = tmp_path / "data"
    batch_dir = data_root / "runs/ham_m1_citywide/terrain_batches"
    batch_dir.mkdir(parents=True)
    batch = batch_dir / "terrain_000000_000001.parquet"
    batch.write_bytes(b"sample-bytes")
    rel = batch.relative_to(data_root).as_posix()
    manifest = batch_dir / "extraction_run_manifest.json"
    manifest.write_text(json.dumps({"outputs": [{"path": rel, "sha256": mod.sha256(batch), "count": 2}]}))
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod.pd, "read_parquet", lambda path, columns: pd.DataFrame({
        "bldg_id": [1, 2], "T50": [0.5, -0.5], "terrain_measured": [True, True],
    })[columns])
    table, hashes = mod.verified_ham_batches(batch_dir, manifest, expected=1, expected_rows=2)
    assert len(table) == 2 and len(hashes["ham_terrain_manifest_sha256"]) == 64
    batch.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        mod.verified_ham_batches(batch_dir, manifest, expected=1, expected_rows=2)


def test_don_join_is_exact_and_no_silent_population_loss(tmp_path, monkeypatch):
    from rhb.experiments import m1g_cross_city_sampling as mod
    root = tmp_path / "resilient-housing-bayes"
    (root / "outputs/phase3/DON").mkdir(parents=True)
    data = tmp_path / "data/runs/don_m1_citywide"
    data.mkdir(parents=True)
    features_path = root / "outputs/phase3/DON/phase3_features_scaled.parquet"
    terrain_path = data / "terrain_features_m1_5m.parquet"
    features_path.write_bytes(b"f")
    terrain_path.write_bytes(b"t")
    cfg = {"cities": {"DON": {"expected_buildings": 2,
                             "features_path": "outputs/phase3/DON/phase3_features_scaled.parquet",
                             "terrain_path": "../data/runs/don_m1_citywide/terrain_features_m1_5m.parquet"}},
           "terrain_transform_scale_m": 0.5}
    features = pd.DataFrame({"bldg_id": [1, 2], "E_hat_v0": [0.2, 0.3],
                             "H_pluvial_v1_logrel": [0.01, 0.02]})
    terrain = pd.DataFrame({"bldg_id": [2, 1], "T50": [0.5, -0.5],
                            "terrain_measured": [True, True]})
    def fake_read(path, columns):
        return (features if path == features_path else terrain)[columns].copy()
    monkeypatch.setattr(mod, "ROOT", root)
    monkeypatch.setattr(mod.pd, "read_parquet", fake_read)
    joined, hashes = mod.load_universe("DON", cfg)
    assert len(joined) == 2 and joined.bldg_id.tolist() == ["1", "2"]
    np.testing.assert_allclose(joined["T"].to_numpy(),
                               np.arcsinh(np.array([-0.5, 0.5]) / 0.5))
    assert len(hashes["terrain_sha256"]) == 64
    terrain["bldg_id"] = [2, 3]
    with pytest.raises(ValueError, match="exact Phase3/terrain"):
        mod.load_universe("DON", cfg)


def test_gamma_coverage_false_not_counted_as_true():
    from rhb.pipelines.summarize_m1g import read_coverage
    assert read_coverage("False") is False
    assert read_coverage("True") is True
    assert read_coverage(False) is False
    assert np.isnan(read_coverage(np.nan))
    with pytest.raises(ValueError):
        read_coverage("maybe")
