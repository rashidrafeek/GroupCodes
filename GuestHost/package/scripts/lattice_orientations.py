from pathlib import Path

import numpy as np
import guesthost as gh


def _resolve_test_paths():
    pkg_root = Path(gh.__file__).resolve().parent.parent.parent
    tests_root = pkg_root / "tests"
    data_path = tests_root / "structures" / "mpb_trajectory.xyz"
    ref_dir = tests_root / "reference_data"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test structure at {data_path}")
    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing reference data at {ref_dir}")
    return data_path, ref_dir


def main():
    data_path, ref_dir = _resolve_test_paths()

    trj = gh.Trajectory(str(data_path), order=True)
    n = 4
    all_inds = [320, 384, 385, 386, 256, 387, 388, 389, 0, 64, 65, 66]
    ma_inds = [0, 1, 2, 3, 4, 5, 6, 7]
    host_inds = [8, 9, 10, 11]

    lattices = trj.create_lattice(
        64,
        [ma_inds, host_inds],
        [gh.MethylAmmonium, gh.Host],
        (n, n, n),
        gh.HPLattice,
        ordering="type",
        unit_order=all_inds,
    )
    lattraj = gh.LatticeTrajectory(lattices)

    ax = 0
    dat = lattraj.compute_for_all(gh.HPLattice.all_ma_orientations, ax)

    t_dat = []
    p_dat = []
    for val in dat[1:]:
        t, p = val
        t_dat.append(t.flatten())
        p_dat.append(p.flatten())
    t_dat = np.array(t_dat)
    p_dat = np.array(p_dat)
    print(p_dat)

    ref_t = np.loadtxt(ref_dir / "reference_thetas.dat")
    ref_p = np.loadtxt(ref_dir / "reference_phis.dat")

    np.testing.assert_allclose(t_dat, ref_t, atol=1e-6, err_msg="Theta data does not match reference.")
    np.testing.assert_allclose(p_dat, ref_p, atol=1e-6, err_msg="Phi data does not match reference.")
    print("guesthost lattice orientation check: OK")


if __name__ == "__main__":
    main()
