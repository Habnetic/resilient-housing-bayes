import numpy as np

from rhb.decision.phase3_decision_metrics import (
    compute_topk_membership,
    summarize_decision_metrics,
)


def test_topk_membership_known_example() -> None:
    p_draws = np.array(
        [
            [0.9, 0.8, 0.1],
            [0.8, 0.9, 0.1],
            [0.7, 0.6, 0.2],
        ]
    )

    topk_prob, rank_std = compute_topk_membership(p_draws, k=1)

    np.testing.assert_allclose(
        topk_prob,
        np.array([2 / 3, 1 / 3, 0.0]),
    )
    assert rank_std[2] == 0.0


def test_topk_membership_mass_equals_k() -> None:
    p_draws = np.array(
        [
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [0.5, 0.6, 0.7, 0.8, 0.9],
            [0.8, 0.9, 0.5, 0.7, 0.6],
            [0.6, 0.5, 0.9, 0.8, 0.7],
        ]
    )

    k = 2
    topk_prob, _ = compute_topk_membership(p_draws, k=k)

    np.testing.assert_allclose(topk_prob.sum(), k)


def test_k_equals_number_of_assets_selects_every_asset() -> None:
    p_draws = np.array(
        [
            [0.9, 0.2, 0.5],
            [0.1, 0.8, 0.4],
        ]
    )

    topk_prob, _ = compute_topk_membership(p_draws, k=3)

    np.testing.assert_allclose(topk_prob, np.ones(3))


def test_rank_std_zero_when_ranking_is_constant() -> None:
    p_draws = np.array(
        [
            [0.9, 0.6, 0.2],
            [0.8, 0.5, 0.1],
            [0.7, 0.4, 0.0],
        ]
    )

    _, rank_std = compute_topk_membership(p_draws, k=1)

    np.testing.assert_allclose(rank_std, np.zeros(3))


def test_summarize_decision_metrics() -> None:
    topk_prob = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    rank_std = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    summary = summarize_decision_metrics(
        topk_prob=topk_prob,
        rank_std=rank_std,
        k=2,
        borderline_low=0.2,
        borderline_high=0.8,
    )

    assert summary["k"] == 2
    assert summary["n_assets"] == 5

    assert summary["borderline_count"] == 1
    assert summary["borderline_share"] == 0.2

    assert summary["stable_inclusion_count"] == 2
    assert summary["stable_inclusion_share"] == 0.4

    assert summary["stable_exclusion_count"] == 2
    assert summary["stable_exclusion_share"] == 0.4

    assert summary["rank_std_mean"] == 2.0
    assert summary["rank_std_p50"] == 2.0