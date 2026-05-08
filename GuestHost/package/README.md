# guesthost

`guesthost` provides tools for analyzing guest-host atomic systems, with helper
classes for molecular fragments, host fragments, unit-cell motifs, lattices, and
lattice trajectories.

## Installation

From the package root:

```bash
pip install -e .
```

## Data Model

The main objects are:

- `Trajectory`: reads ASE-supported trajectory files and stores frames, atomic
  coordinates, cells, symbols, and frame counts.
- `MethylAmmonium`: stores the C, N, H-C, H-N coordinates and the MA `CN`
  orientation vector for one molecule.
- `Host`: stores the central host atom and three in-cell host-environment atoms.
- `Motif`: groups the fragments belonging to one unit cell. For hybrid
  perovskite systems, `motif.ma` accesses the MA fragment and `motif.host`
  accesses the host fragment.
- `HPLattice`: stores motifs on a 3D lattice and provides unit-cell and
  lattice-level analysis methods.
- `LatticeTrajectory`: stores one lattice object per trajectory frame and
  provides trajectory-level compute helpers.

## Basic Usage

```python
from pathlib import Path
import guesthost as gh

pkg_root = Path(gh.__file__).resolve().parents[2]
data_path = pkg_root / "tests" / "structures" / "mpb_trajectory.xyz"

trj = gh.Trajectory(str(data_path), order=True)

lattices = trj.create_hplattice(
    unitcell_data=gh.UNITCELL_INDEXDATA_MPB_4x4x4,
    supercell_size=(4, 4, 4),
)

lattice = lattices[1]
ind = (0, 0, 0)

theta, phi = lattice.ucell_theta_phi(ind, dir=0)
params = lattice.ucell_localcellparameters(ind)
eta_ma = lattice.ucell_eta_MA(ind, dir=0)
eta_lat = lattice.ucell_eta_lat(ind, dir_coup=0)
```

Unit-cell data can also be inferred from a reference structure:

```python
lattices = trj.create_hplattice(
    reference=trj.atoms_list[0],
    supercell_size=(4, 4, 4),
)
```

Packaged MAPbBr3 reference structures and unit-cell data are available as
constants:

```python
system_4 = gh.MPB_SYS_4x4x4
unitcells_4 = gh.UNITCELL_INDEXDATA_MPB_4x4x4

system_8 = gh.MPB_SYS_8x8x8
unitcells_8 = gh.UNITCELL_INDEXDATA_MPB_8x8x8
```

## Unit-Cell Methods

`HPLattice` methods operate on unit-cell indices of the form `(ix, iy, iz)`.

```python
ind = (0, 0, 0)

theta, phi = lattice.ucell_theta_phi(ind, dir=0)
vecs = lattice.ucell_localcellvecs(ind)
params = lattice.ucell_localcellparameters(ind)
angle = lattice.ucell_pb_br_pb_angle(ind, dir=0)
angle_hostrelative = lattice.ucell_pb_br_pb_angle_hostrelative(ind, dir=0, dir_coup=1)
eta_ma = lattice.ucell_eta_MA(ind, dir=0)
eta_lat = lattice.ucell_eta_lat(ind, dir_coup=0)
```

Unicode method names are also available for compatibility:

```python
eta_ma = lattice.ucell_η_MA(ind, dir=0)
eta_lat = lattice.ucell_η_lat(ind, dir_coup=0)
```

## Lattice-Wide Methods

```python
theta, phi = lattice.all_ucell_theta_phi(dir=0)
eta_ma = lattice.all_ucell_eta_MA(dir=0)
eta_lat = lattice.all_ucell_eta_lat(dir_coup=0)
```

For trajectory-level analysis:

```python
lattraj = gh.LatticeTrajectory(lattices)
theta_phi_per_frame = lattraj.compute_for_all(
    lambda lat: lat.ucell_theta_phi((0, 0, 0), dir=0)
)
```

## Testing

From the package root:

```bash
python -m pytest tests
```

The tests include reference-data checks for lattice construction, orientations,
and torsions.
