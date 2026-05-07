from pathlib import Path

import guesthost as gh


def _resolve_test_path() -> Path:
    pkg_root = Path(gh.__file__).resolve().parents[2]
    data_path = pkg_root / "tests" / "structures" / "mpb_trajectory.xyz"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test data at {data_path}")
    return data_path


def main() -> None:
    data_path = _resolve_test_path()

    trj = gh.Trajectory(str(data_path), order=True)
    n = 4

    all_inds = [320, 384, 385, 386, 256, 387, 388, 389, 0, 16, 4, 1]
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

    frame_index = 1
    lattice = lattices[frame_index]

    ax = 0
    unitcell_index = (0, 0, 0)
    theta, phi = lattice.ma_orientation(unitcell_index, ax)
    print(theta, phi)


if __name__ == "__main__":
    main()
