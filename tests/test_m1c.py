import numpy as np
import pandas as pd
import pytest

from rhb.experiments.m1c_simulation import (
    assign_split,
    simulate_case,
)


@pytest.fixture
def example_data():
    return pd.DataFrame({
        "bldg_id": np.arange(100),
        "E_hat_v0": np.zeros(100),
        "H_pluvial_v1_logrel": np.zeros(100),
        "T": np.linspace(-2, 2, 100),
    })


def test_zero_topographic_effect(example_data):
    """Gamma=0 must eliminate T from the generator."""

    result = simulate_case(
        example_data,
        gamma_true=0.0,
        seed=101,
    )

    expected = 1 / (1 + np.exp(3))

    assert np.allclose(result["p_true"], expected)


def test_negative_topographic_effect(example_data):
    """Negative gamma reduces risk as T increases."""

    result = simulate_case(
        example_data,
        gamma_true=-0.8,
        seed=101,
    )

    probabilities = result["p_true"].to_numpy()

    assert np.all(np.diff(probabilities) < 0)


def test_simulation_reproducibility(example_data):
    """The same seed must reproduce identical outcomes."""

    a = simulate_case(example_data, -0.8, 101)
    b = simulate_case(example_data, -0.8, 101)

    assert np.array_equal(a["Y_m1c"], b["Y_m1c"])


def test_fixed_train_test_split(example_data):
    """Splits must be deterministic and non-overlapping."""

    a = assign_split(example_data, 80, 42)
    b = assign_split(example_data, 80, 42)

    assert (a["split"] == "train").sum() == 80
    assert (a["split"] == "test").sum() == 20

    assert a["split"].equals(b["split"])


def test_paired_randomness(example_data):
    """Identical random uniforms across effect scenarios."""

    zero = simulate_case(example_data, 0.0, 101)
    negative = simulate_case(example_data, -0.8, 101)

    # For positive T, negative gamma reduces event probability.
    positive = example_data["T"] > 0

    assert np.all(
        negative.loc[positive, "Y_m1c"].to_numpy()
        <= zero.loc[positive, "Y_m1c"].to_numpy()
    )