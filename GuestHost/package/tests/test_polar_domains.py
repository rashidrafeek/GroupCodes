import numpy as np

import guesthost as gh


def test_polar_domain_origins():
    np.testing.assert_array_equal(gh.polar_domain_origins(8, 3), np.arange(8))
    np.testing.assert_array_equal(
        gh.polar_domain_origins(8, 3, nonoverlapping=True), [0, 3, 6]
    )


def test_polar_domain_order_from_phi_known_chains():
    phi = np.zeros((2, 2, 4))
    phi[0, 0] = [10.0, 10.0, 10.0, 10.0]
    phi[0, 1] = [0.0, 180.0, 0.0, 180.0]
    result = gh.polar_domain_order_from_phi(phi, 2, 4)

    np.testing.assert_allclose(result["xi_q0"][0, 0], 1.0)
    np.testing.assert_allclose(result["S_q0"][0, 0], 1.0)
    np.testing.assert_allclose(result["xi_qpi"][1, 0], 1.0)
    np.testing.assert_allclose(result["S_qpi"][1, 0], 1.0)
    assert result["xi_q0"].shape == (4, 4)


def test_autocorrelation_multiple_series():
    values = np.array([[1.0, 2.0], [1.0, 1.0], [1.0, 0.0]])
    corr = gh.autocorrelation(values, max_lag=2)
    np.testing.assert_allclose(corr[:, 0], 1.0)
    np.testing.assert_allclose(corr[0], 1.0)
    np.testing.assert_allclose(corr[:, 1], [1.0, 0.6, 0.0])


def test_domain_order_python_reference_values():
    phi = np.array(
        [
            [[0.0, 30.0, 80.0, -170.0]],
            [[15.0, -20.0, 160.0, 175.0]],
        ]
    )
    result = gh.polar_domain_order_from_phi(
        phi, dir_coup=2, domain_size=3, nonoverlapping=True
    )
    np.testing.assert_array_equal(result["origins"], [0, 3])
    np.testing.assert_allclose(
        result["xi_q0"],
        [[0.679891185654455, -0.30816678978503],
         [0.333333333333333, -0.301872821034283]],
    )
