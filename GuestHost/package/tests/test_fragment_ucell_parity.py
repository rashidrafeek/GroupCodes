from pathlib import Path
import sys

import numpy as np
from ase.io import read
import guesthost as gh

try:
    import perovskite_analysis as pa
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perovskite_analysis" / "src"))
    import perovskite_analysis as pa


def _build_guesthost_lattice(frame_index=1):
    traj_path = Path(__file__).parent / "structures" / "mpb_trajectory.xyz"
    trj = gh.Trajectory(traj_path, order=True)
    all_inds = [320, 384, 385, 386, 256, 387, 388, 389, 0, 64, 65, 66]
    ma_inds = [0, 1, 2, 3, 4, 5, 6, 7]
    host_inds = [8, 9, 10, 11]
    lattices = trj.create_lattice(
        64,
        [ma_inds, host_inds],
        [gh.MethylAmmonium, gh.Host],
        (4, 4, 4),
        gh.HPLattice,
        ordering="type",
        unit_order=all_inds,
    )
    return lattices[frame_index], trj.atoms_list[frame_index]


def _perovskite_frame(frame_index=1):
    pkg_root = Path(pa.__file__).resolve().parents[2]
    traj_path = pkg_root / "tests" / "data" / "mpb_trajectory.xyz"
    if not traj_path.exists():
        traj_path = Path(__file__).parent / "structures" / "mpb_trajectory.xyz"
    return read(traj_path, index=frame_index)


def _pa_ucell_array(system, func, *args, **kwargs):
    vals = [func(system, udata, *args, **kwargs) for udata in pa.UNITCELL_INDEXDATA_MPB_4x4x4]
    return np.array(vals, dtype=object).reshape((4, 4, 4))


def test_fragment_theta_phi_matches_perovskite_analysis():
    lattice, _ = _build_guesthost_lattice(frame_index=1)
    system = _perovskite_frame(frame_index=1)

    gh_theta, gh_phi = lattice.all_ucell_theta_phi(dir=0)
    pa_vals = _pa_ucell_array(system, pa.ucell_theta_phi, dir=0)
    pa_theta = np.vectorize(lambda x: x["theta"])(pa_vals)
    pa_phi = np.vectorize(lambda x: x["phi"])(pa_vals)

    np.testing.assert_allclose(gh_theta, pa_theta, atol=1e-6)
    np.testing.assert_allclose(gh_phi, pa_phi, atol=1e-6)


def test_fragment_localcellparameters_match_perovskite_analysis():
    lattice, _ = _build_guesthost_lattice(frame_index=1)
    system = _perovskite_frame(frame_index=1)

    gh_vals = np.empty((4, 4, 4), dtype=object)
    for ind in np.ndindex(4, 4, 4):
        gh_vals[ind] = lattice.ucell_localcellparameters(ind)

    pa_vals = _pa_ucell_array(system, pa.ucell_localcellparameters)
    for key in ("a", "b", "c", "alpha", "beta", "gamma", "vol"):
        gh_arr = np.vectorize(lambda x: x[key])(gh_vals)
        pa_arr = np.vectorize(lambda x: x[key])(pa_vals)
        np.testing.assert_allclose(gh_arr, pa_arr, atol=1e-6)


def test_fragment_eta_ma_matches_perovskite_analysis():
    lattice, _ = _build_guesthost_lattice(frame_index=1)
    system = _perovskite_frame(frame_index=1)

    gh_eta = lattice.all_ucell_η_MA(dir=0)
    pa_eta = np.array(
        [pa.ucell_eta_MA(system, udata, dir=0) for udata in pa.UNITCELL_INDEXDATA_MPB_4x4x4]
    ).reshape((4, 4, 4, 2))

    np.testing.assert_allclose(gh_eta, pa_eta, atol=1e-6)


def test_fragment_eta_lat_matches_perovskite_analysis():
    lattice, _ = _build_guesthost_lattice(frame_index=1)
    system = _perovskite_frame(frame_index=1)

    gh_eta = lattice.all_ucell_η_lat(dir_coup=0)
    pa_eta = np.array(
        [pa.ucell_eta_lat(system, udata, dir_coup=0) for udata in pa.UNITCELL_INDEXDATA_MPB_4x4x4]
    ).reshape((4, 4, 4, 2))

    np.testing.assert_allclose(gh_eta, pa_eta, atol=1e-6)
