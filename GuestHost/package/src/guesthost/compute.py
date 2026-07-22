import inspect
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import h5py
import numpy as np

from guesthost.lattice import HPLattice, LatticeTrajectory
from guesthost.layerwise import compute_layerwise


def _call_ucell(func: Callable, lattice: HPLattice, ind, kwargs: Dict[str, Any]):
    if inspect.ismethod(func) and func.__self__ is lattice:
        return func(ind, **kwargs)
    return func(lattice, ind, **kwargs)


def compute(lattice: HPLattice, func: Callable, **kwargs):
    """Apply a unit-cell function to every motif in one HPLattice."""
    return [
        _call_ucell(func, lattice, ind, kwargs)
        for ind in np.ndindex(lattice.nx, lattice.ny, lattice.nz)
    ]


def compute_trajectory(trajectory: LatticeTrajectory, func: Callable, **kwargs):
    """Apply a unit-cell function to every HPLattice in a LatticeTrajectory."""
    return [compute(lattice, func, **kwargs) for lattice in trajectory.lattices]


def _format_key(func: Callable, kwargs: Dict[str, Any], layerwise: bool) -> str:
    base = func.__name__ if hasattr(func, "__name__") else str(func)
    if kwargs:
        args_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        base = f"{base}({args_str})"
    if layerwise:
        base = f"layerwise:{base}"
    return base


def compute_functions(
    obj: Union[HPLattice, LatticeTrajectory],
    funcs: Iterable[Union[Callable, Tuple]],
) -> Dict[str, Any]:
    """
    Compute a bundle of unit-cell or layerwise functions.

    Entries may be ``func``, ``(func, kwargs)``, or
    ``(func, kwargs, layerwise_bool)``.
    """
    results = {}
    for entry in funcs:
        if isinstance(entry, tuple):
            if len(entry) == 3:
                func, kwargs, layerwise = entry
            elif len(entry) == 2:
                func, kwargs = entry
                layerwise = False
            else:
                func = entry[0]
                kwargs = {}
                layerwise = False
        else:
            func = entry
            kwargs = {}
            layerwise = False

        key = _format_key(func, kwargs, layerwise)
        if layerwise:
            results[key] = compute_layerwise(obj, func, **kwargs)
        elif isinstance(obj, LatticeTrajectory):
            results[key] = compute_trajectory(obj, func, **kwargs)
        else:
            results[key] = compute(obj, func, **kwargs)
    return results


def compute_default_functions(obj: Union[HPLattice, LatticeTrajectory], cellshape=None):
    """Compute a standard set of unit-cell and layerwise analysis functions."""
    from guesthost.layerwise import (
        global_xi,
        global_xi_MA,
        global_xi_coupling,
        layerwise_eta_Lat,
    )

    funcs = []
    for direction in range(3):
        funcs.append((HPLattice.ucell_theta_phi, {"dir": direction}, False))
        funcs.append((HPLattice.ucell_eta_lat, {"dir_coup": direction}, False))
        funcs.append((HPLattice.ucell_eta_lat_alongaxis, {"dir_coup": direction}, False))
    funcs.append((HPLattice.ucell_localcellparameters, {}, False))

    for direction in range(3):
        cell_kwargs = {"cellshape": cellshape} if cellshape is not None else {}
        funcs.append((layerwise_eta_Lat, {"dir_coup": direction, **cell_kwargs}, True))
        funcs.append((global_xi, {"dir_coup": direction, **cell_kwargs}, True))
        funcs.append((global_xi_MA, {"dir": direction, **cell_kwargs}, True))
        funcs.append((global_xi_coupling, {"dir": direction, **cell_kwargs}, True))

    return compute_functions(obj, funcs)


def _to_numpy(data: Any):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (int, float, complex, np.number)):
        return np.array(data)
    if isinstance(data, dict):
        vals = [_to_numpy(data[k]) for k in sorted(data.keys())]
        try:
            return np.stack(vals)
        except Exception:
            return np.array(vals, dtype=object)
    if isinstance(data, (list, tuple)):
        converted = [_to_numpy(x) for x in data]
        try:
            return np.stack(converted)
        except Exception:
            try:
                return np.array(converted)
            except Exception as exc:
                raise ValueError(f"Cannot convert data to array: {exc}") from exc
    raise ValueError(f"Unsupported data type for save_results: {type(data)}")


def _write_dataset_or_group(parent, key, data):
    if isinstance(data, dict):
        group = parent.create_group(key)
        for subkey, value in data.items():
            _write_dataset_or_group(group, subkey, value)
        return
    parent.create_dataset(key, data=_to_numpy(data))


def save_results(filename, results: Dict[str, Any], trajectory=None, **kwargs):
    """Save computed results to an HDF5 file."""
    with h5py.File(filename, "w", **kwargs) as handle:
        if trajectory is not None:
            if hasattr(trajectory, "times"):
                handle.create_dataset("time", data=np.array(trajectory.times))
            if hasattr(trajectory, "steps"):
                handle.create_dataset("steps", data=np.array(trajectory.steps))
        for key, data in results.items():
            _write_dataset_or_group(handle, key, data)


def _read_node(node):
    if isinstance(node, h5py.Dataset):
        if node.shape == ():
            return node[()]
        return node[:]
    return {key: _read_node(node[key]) for key in node.keys()}


def load_results(filename):
    """Load HDF5 results written by save_results."""
    with h5py.File(filename, "r") as handle:
        return {key: _read_node(handle[key]) for key in handle.keys()}



def save_theta_phi(filename, theta, phi, times_ps, source_steps, metadata=None):
    """Write frame-first fixed-grid theta/phi arrays to HDF5.

    ``theta`` and ``phi`` must have shape ``(frame, x, y, z)``.  The selected
    frame time in ps and original dump timestep are stored alongside the grids.
    Scalar ``metadata`` entries are written as file attributes.
    """
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    times_ps = np.asarray(times_ps)
    source_steps = np.asarray(source_steps)
    if theta.shape != phi.shape or theta.ndim != 4:
        raise ValueError("theta and phi must have identical (frame, x, y, z) shapes")
    if len(times_ps) != theta.shape[0] or len(source_steps) != theta.shape[0]:
        raise ValueError("time and source-step arrays must match the frame count")
    with h5py.File(filename, "w") as handle:
        handle.create_dataset("theta", data=theta)
        handle.create_dataset("phi", data=phi)
        handle.create_dataset("time_ps", data=times_ps)
        handle.create_dataset("source_step", data=source_steps)
        handle.attrs["format"] = "guesthost_theta_phi_v1"
        handle.attrs["grid_order"] = "frame,x,y,z"
        for key, value in (metadata or {}).items():
            handle.attrs[str(key)] = value
    return filename


def load_theta_phi(filename):
    """Load a theta/phi HDF5 file written by :func:`save_theta_phi`."""
    with h5py.File(filename, "r") as handle:
        required = ("theta", "phi", "time_ps", "source_step")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"not a guesthost theta/phi HDF5 file; missing {missing}")
        return {
            "theta": handle["theta"][:],
            "phi": handle["phi"][:],
            "times_ps": handle["time_ps"][:],
            "source_steps": handle["source_step"][:],
            "metadata": dict(handle.attrs.items()),
        }


def load_phi(filename):
    """Load only phi and provenance from a theta/phi HDF5 file."""
    with h5py.File(filename, "r") as handle:
        required = ("phi", "time_ps", "source_step")
        missing = [key for key in required if key not in handle]
        if missing:
            raise ValueError(f"not a guesthost theta/phi HDF5 file; missing {missing}")
        return {
            "phi": handle["phi"][:],
            "times_ps": handle["time_ps"][:],
            "source_steps": handle["source_step"][:],
            "metadata": dict(handle.attrs.items()),
        }
