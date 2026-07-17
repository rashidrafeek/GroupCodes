from importlib.resources import files

import numpy as np

import guesthost as gh


def test_reference_pb_axis_neighbours_match_motif_grid():
    """Canonical motif-grid neighbours must match each unit cell’s pb_axis data."""
    path = files("guesthost").joinpath("data/structures/mpb_cubic_8x8x8.xyz")
    lattice = gh.Trajectory(str(path)).create_hplattice(
        unitcell_data=gh.UNITCELL_INDEXDATA_MPB_8x8x8,
        supercell_size=(8, 8, 8),
    )[0]

    for index in np.ndindex(8, 8, 8):
        motif = lattice.get_motif(index)
        origin, axes = motif.unitcell_data["pb_axis"]
        assert int(motif.host.indices[0]) == int(origin)
        for direction, pb_index in enumerate(axes):
            neighbour = lattice.get_motif(lattice.shift_index(index, direction))
            assert int(neighbour.host.indices[0]) == int(pb_index)
