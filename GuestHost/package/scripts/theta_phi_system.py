from pathlib import Path
import guesthost as gh

pkg_root = Path(gh.__file__).resolve().parent.parent
data_path = pkg_root / ".." / "tests" / "structures" / "mpb_trajectory.xyz"
if not data_path.exists():
    raise FileNotFoundError(f"Missing test data at {data_path}")

trj = gh.Trajectory(str(data_path), order=True)
n = 4

# all_inds = [320, 384, 385, 386, 256, 387, 388, 389, 0, 64, 65, 66]
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

lattice = lattices[1]
system = trj.atoms_list[1]

c_ind, n_ind = all_inds[0], all_inds[4]
pb_axis = (all_inds[8], [all_inds[9], all_inds[10], all_inds[11]])

theta, phi = lattice.ucell_theta_phi(system, c_ind, n_ind, pb_axis, dir=0)
print(theta, phi)
