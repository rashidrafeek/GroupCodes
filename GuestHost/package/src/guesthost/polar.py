import numpy as np

from guesthost.lattice import reference_pb_coordinates
from guesthost.layerwise import dirs_perp


def _wrap180(x):
    x = np.asarray(x, dtype=float)
    x = np.where(x <= -180.0, x + 360.0, x)
    x = np.where(x >= 180.0, x - 360.0, x)
    return x


def _polar_op_from_phi(phi_arr, dir_coup):
    phi_prev = np.roll(phi_arr, shift=-1, axis=dir_coup)
    dphi = _wrap180(phi_prev - phi_arr)

    z = np.exp(1j * np.pi * dphi / 180.0)

    axis_len = phi_arr.shape[dir_coup]
    phase_shape = [1, 1, 1]
    phase_shape[dir_coup] = axis_len
    phase_pi = np.exp(-1j * np.pi * np.arange(axis_len)).reshape(phase_shape)
    phase_pi2 = np.exp(-1j * (np.pi / 2.0) * np.arange(axis_len)).reshape(phase_shape)

    z_pi = z * phase_pi
    z_pi_by2 = z * phase_pi2

    perp = dirs_perp[dir_coup]
    z_chain = z.mean(axis=perp)
    z_chain_pi = z_pi.mean(axis=perp)
    z_chain_pi_by2 = z_pi_by2.mean(axis=perp)

    return {
        "dphi": dphi,
        "z": z,
        "z_pi": z_pi,
        "z_pi_by2": z_pi_by2,
        "z_chain": z_chain,
        "z_chain_pi": z_chain_pi,
        "z_chain_pi_by2": z_chain_pi_by2,
        "S_chain": np.abs(z_chain) ** 2,
        "S_chain_pi": np.abs(z_chain_pi) ** 2,
        "S_chain_pi_by2": np.abs(z_chain_pi_by2) ** 2,
    }


def polar_phi_grid(lattice, dir_coup, cellshape=None):
    """Return MA phi on the canonical physical unit-cell grid."""
    if cellshape is None:
        cellshape = (lattice.nx, lattice.ny, lattice.nz)
    from guesthost.constants import (
        MPB_SYS_4x4x4, MPB_SYS_8x8x8,
        UNITCELL_INDEXDATA_MPB_4x4x4, UNITCELL_INDEXDATA_MPB_8x8x8,
    )
    references = {
        64: (MPB_SYS_4x4x4, UNITCELL_INDEXDATA_MPB_4x4x4),
        512: (MPB_SYS_8x8x8, UNITCELL_INDEXDATA_MPB_8x8x8),
    }
    phi = np.empty(cellshape, dtype=float)
    reference = references.get(int(np.prod(cellshape)))
    coordinate_by_pb = (
        reference_pb_coordinates(reference[0], reference[1], cellshape)
        if reference is not None else {}
    )
    for ind in np.ndindex(cellshape):
        motif = lattice.get_motif(ind)
        pb_index = int(motif.unitcell_data["pb_axis"][0])
        coords = coordinate_by_pb.get(pb_index, ind)
        phi[coords] = lattice.ucell_theta_phi(ind, dir=dir_coup)[1]
    return phi


def theta_phi_grid(lattice, dir_coup, cellshape=None):
    """Return MA theta and phi on the canonical fixed Pb-index grid.

    The returned arrays have shape ``cellshape``. Their coordinates are taken
    from the packaged reference Pb map, rather than from each instantaneous
    frame, so they can be stored and compared across a trajectory.
    """
    if cellshape is None:
        cellshape = (lattice.nx, lattice.ny, lattice.nz)
    from guesthost.constants import (
        MPB_SYS_4x4x4, MPB_SYS_8x8x8,
        UNITCELL_INDEXDATA_MPB_4x4x4, UNITCELL_INDEXDATA_MPB_8x8x8,
    )
    references = {
        64: (MPB_SYS_4x4x4, UNITCELL_INDEXDATA_MPB_4x4x4),
        512: (MPB_SYS_8x8x8, UNITCELL_INDEXDATA_MPB_8x8x8),
    }
    theta = np.empty(cellshape, dtype=float)
    phi = np.empty(cellshape, dtype=float)
    reference = references.get(int(np.prod(cellshape)))
    coordinate_by_pb = (
        reference_pb_coordinates(reference[0], reference[1], cellshape)
        if reference is not None else {}
    )
    for ind in np.ndindex(cellshape):
        motif = lattice.get_motif(ind)
        pb_index = int(motif.unitcell_data["pb_axis"][0])
        coords = coordinate_by_pb.get(pb_index, ind)
        theta_value, phi_value = lattice.ucell_theta_phi(ind, dir=dir_coup)
        theta[coords] = theta_value
        phi[coords] = phi_value
    return {"theta": theta, "phi": phi}


def polar_order_parameter(lattice, dir_coup, cellshape=None):
    if cellshape is None:
        cellshape = (lattice.nx, lattice.ny, lattice.nz)

    return _polar_op_from_phi(polar_phi_grid(lattice, dir_coup, cellshape), dir_coup)

def polar_domain_origins(chain_length, domain_size, nonoverlapping=False):
    """Return zero-based periodic domain origins.

    Set ``nonoverlapping=True`` to advance origins by ``domain_size``; otherwise
    every cell is used as a periodic origin.
    """
    if not 1 <= domain_size <= chain_length:
        raise ValueError("domain_size must be between 1 and chain_length")
    step = domain_size if nonoverlapping else 1
    return np.arange(0, chain_length, step, dtype=int)


def polar_domain_order_from_phi(phi, dir_coup, domain_size, nonoverlapping=False):
    """Compute q=0 and q=pi order fields from a fixed 3D MA-phi grid.

    The returned arrays are indexed by transverse chain and periodic origin.
    """
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 3 or dir_coup not in range(3):
        raise ValueError("phi must be 3D and dir_coup must be 0, 1, or 2")
    chain_length = phi.shape[dir_coup]
    origins = polar_domain_origins(chain_length, domain_size, nonoverlapping)
    chains = np.moveaxis(phi, dir_coup, -1).reshape(-1, chain_length)
    offsets = np.arange(domain_size)
    indices = (origins[:, None] + offsets) % chain_length
    domains = chains[:, indices]
    dphi = _wrap180(domains[..., :1] - domains)
    z = np.exp(1j * np.deg2rad(dphi))
    z_q0 = z.mean(axis=-1)
    z_qpi = (z * np.exp(-1j * np.pi * offsets)).mean(axis=-1)
    return {
        "origins": origins,
        "dphi": dphi,
        "z_q0": z_q0,
        "z_qpi": z_qpi,
        "xi_q0": z_q0.real,
        "xi_qpi": z_qpi.real,
        "S_q0": np.abs(z_q0) ** 2,
        "S_qpi": np.abs(z_qpi) ** 2,
    }


def polar_domain_order(lattice, dir_coup, domain_size, nonoverlapping=False):
    """Compute polar-domain order from an HPLattice using canonical Pb indexing."""
    phi = polar_phi_grid(lattice, dir_coup)
    return polar_domain_order_from_phi(phi, dir_coup, domain_size, nonoverlapping)


def polar_domain_order_trajectory(trajectory, dir_coup, domain_size, nonoverlapping=False):
    """Compute polar-domain order fields for all trajectory frames.

    Time is the first axis of every returned field.
    """
    frame_results = [
        polar_domain_order(frame, dir_coup, domain_size, nonoverlapping)
        for frame in trajectory.lattices
    ]
    keys = ("dphi", "z_q0", "z_qpi", "xi_q0", "xi_qpi", "S_q0", "S_qpi")
    result = {key: np.stack([frame[key] for frame in frame_results]) for key in keys}
    result["origins"] = frame_results[0]["origins"]
    return result


def autocorrelation(
    values, max_lag=None, normalize=True, demean=False, normalization="pair_count"
):
    """Compute optionally normalized lag autocorrelations along axis zero.

    Every remaining index is normalized independently by its zero-lag value.
    ``normalization="pair_count"`` divides lag ``k`` by its ``N-k`` available
    pairs. ``normalization="legacy_total"`` divides every lag by ``N`` and
    reproduces the original Julia ``StatsBase.autocor`` analysis.
    """
    values = np.asarray(values)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("values must contain at least one time sample")
    if max_lag is None:
        max_lag = values.shape[0] - 1
    if not 0 <= max_lag < values.shape[0]:
        raise ValueError("max_lag must be smaller than the time-series length")
    if normalization not in ("pair_count", "legacy_total"):
        raise ValueError("normalization must be 'pair_count' or 'legacy_total'")
    work = values - values.mean(axis=0) if demean else values
    corr = np.stack([
        np.mean(np.conjugate(work[: values.shape[0] - lag]) * work[lag:], axis=0)
        for lag in range(max_lag + 1)
    ])
    if normalization == "legacy_total":
        scale = (values.shape[0] - np.arange(max_lag + 1)) / values.shape[0]
        corr *= scale[(slice(None),) + (None,) * (values.ndim - 1)]
    if normalize:
        corr = corr / corr[0]
    return np.real_if_close(corr)


def polar_domain_autocorrelation(
    trajectory, dir_coup, domain_size, max_lag=None, nonoverlapping=False,
    demean=False, normalization="pair_count"
):
    """Average domain-order ACFs using ``pair_count`` or ``legacy_total`` normalization."""
    order = polar_domain_order_trajectory(
        trajectory, dir_coup, domain_size, nonoverlapping
    )
    result = {}
    for key in ("xi_q0", "xi_qpi", "S_q0", "S_qpi"):
        per_domain = autocorrelation(
            order[key], max_lag=max_lag, demean=demean, normalization=normalization
        )
        result[key] = per_domain.mean(axis=(1, 2))
    return result
