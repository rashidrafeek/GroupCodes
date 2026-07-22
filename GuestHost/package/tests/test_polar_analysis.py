import numpy as np
import pytest

import guesthost as gh


def test_connected_correlation_is_offset_invariant():
    values = np.array([0.0, 1, 0, -1, 0, 1, 0, -1])[:, None]
    corr = gh.connected_correlation(values, max_lag=3, block_length=2)
    shifted = gh.connected_correlation(values + 20, max_lag=3)
    np.testing.assert_allclose(corr["correlation"], shifted["correlation"])
    np.testing.assert_array_equal(corr["pair_counts"], [8, 7, 6, 5])
    metrics = gh.relaxation_metrics([1.0, 0.5, 0.2, -0.1], np.arange(4.0))
    assert metrics["resolved_1e"] and metrics["reached_zero"]
    assert 1 < metrics["one_over_e_time"] < 2


def test_known_antipolar_chain_and_structure_factor():
    phi = np.zeros((8, 2, 2, 8))
    phi[:, :, :, 1::2] = 180
    series = gh.polar_domain_series(phi, 2, 2)
    assert series["xi_q0"].shape == (8, 4, 8)
    with pytest.raises(ValueError):
        gh.polar_domain_series(
            phi, 2, 3, origin_mode="strict_nonoverlapping"
        )
    np.testing.assert_array_equal(gh.polar_bond_labels(phi, 2), -1)
    np.testing.assert_array_equal(gh.polar_domain_states(phi, 2, 3)["states"], -1)
    spatial = gh.spatial_correlation(
        phi, 2, field="orientation", connected=False
    )
    np.testing.assert_allclose(
        spatial["correlation"], [1, -1, 1, -1, 1], atol=1e-12
    )
    sf = gh.chain_structure_factor(
        phi, 2, field="orientation", connected=False
    )
    assert sf["indices"][np.argmax(sf["intensity"])] == 4
    assert gh.second_moment_correlation_length(sf)["resolved"]


def test_periodic_runs_and_state_survival():
    assert gh.periodic_label_runs([1, 1, -1, -1, 1]) == [
        {"label": -1, "length": 2},
        {"label": 1, "length": 3},
    ]
    states = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int8)[:, None]
    runs = gh.state_dwell_runs(states, np.arange(8.0))
    assert [run["duration"] for run in runs] == [2.0, 3.0, 1.0]
    assert [run["censored"] for run in runs] == [True, False, True]
    assert [run["left_censored"] for run in runs] == [True, False, False]
    assert [run["right_censored"] for run in runs] == [False, False, True]
    survival = gh.state_survival(runs, -1)
    np.testing.assert_allclose(survival["times"], [3.0])
    np.testing.assert_allclose(survival["survival"], [0.0])
    left_excluded = gh.state_survival(runs, 1)
    np.testing.assert_allclose(left_excluded["times"], [1.0])
    np.testing.assert_allclose(left_excluded["survival"], [1.0])
    assert not gh.survival_metrics(left_excluded)["median_resolved"]
    summary = gh.survival_metrics({
        "times": np.array([1.0, 2.0, 3.0]),
        "survival": np.array([2 / 3, 1 / 3, 1 / 3]),
    })
    assert summary["median_time"] == 2.0
    assert summary["restricted_mean_time"] == pytest.approx(2.0)
    assert summary["restriction_time"] == 3.0
