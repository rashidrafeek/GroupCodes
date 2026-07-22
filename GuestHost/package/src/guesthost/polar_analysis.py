"""Reusable temporal and spatial estimators for polar-domain analysis."""

import numpy as np

from guesthost.polar import _wrap180, polar_domain_order_from_phi


def _domain_origins(chain_length, domain_size, origin_mode):
    if origin_mode == "all":
        return np.arange(chain_length)
    if origin_mode == "strided":
        return np.arange(0, chain_length, domain_size)
    if origin_mode != "strict_nonoverlapping":
        raise ValueError(
            "origin_mode must be all, strided, or strict_nonoverlapping"
        )
    if chain_length % domain_size:
        raise ValueError(
            "strict nonoverlapping domains require domain_size to divide chain_length"
        )
    return np.arange(0, chain_length, domain_size)


def polar_domain_series(phi_frames, dir_coup, domain_size, origin_mode="all"):
    """Compute domain-order arrays from frame-first theta/phi data."""
    phi_frames = np.asarray(phi_frames, dtype=float)
    if phi_frames.ndim != 4 or dir_coup not in range(3):
        raise ValueError("phi_frames must be 4D and dir_coup must be 0, 1, or 2")
    origins = _domain_origins(phi_frames.shape[dir_coup + 1], domain_size, origin_mode)
    frames = [
        polar_domain_order_from_phi(frame, dir_coup, domain_size)
        for frame in phi_frames
    ]
    result = {"origins": origins}
    for key in ("xi_q0", "xi_qpi", "S_q0", "S_qpi"):
        result[key] = np.stack([frame[key][:, origins] for frame in frames])
    return result


def _block_sem(samples, block_length):
    if block_length is None:
        return np.nan
    if block_length < 1:
        raise ValueError("block_length must be positive")
    nblocks = len(samples) // block_length
    if nblocks < 2:
        return np.nan
    blocks = np.array([
        np.mean(samples[i * block_length : (i + 1) * block_length])
        for i in range(nblocks)
    ])
    return blocks.std(ddof=1) / np.sqrt(nblocks)


def connected_correlation(
    values, max_lag=None, aggregation="variance_weighted", block_length=None
):
    """Compute a connected pair-count correlation along axis zero."""
    values = np.asarray(values)
    ntime = values.shape[0]
    max_lag = ntime - 1 if max_lag is None else int(max_lag)
    if aggregation not in ("variance_weighted", "equal_series"):
        raise ValueError("aggregation must be variance_weighted or equal_series")
    if not 0 <= max_lag < ntime:
        raise ValueError("max_lag must be smaller than the time-series length")
    matrix = values.reshape(ntime, -1)
    work = matrix - matrix.mean(axis=0, keepdims=True)
    variances = np.mean(np.abs(work) ** 2, axis=0)
    valid = variances > np.finfo(float).eps
    if not np.any(valid):
        raise ValueError("all time series have zero variance")
    denominator = variances[valid].mean() if aggregation == "variance_weighted" else 1.0
    corr, error, pairs = [], [], []
    for lag in range(max_lag + 1):
        products = np.real(
            np.conjugate(work[: ntime - lag, valid]) * work[lag:, valid]
        )
        origin_values = (
            products.mean(axis=1) / denominator
            if aggregation == "variance_weighted"
            else (products / variances[valid]).mean(axis=1)
        )
        corr.append(origin_values.mean())
        error.append(_block_sem(origin_values, block_length))
        pairs.append(len(origin_values))
    norm = corr[0]
    corr = np.asarray(corr) / norm
    error = np.asarray(error) / abs(norm)
    corr[0] = 1.0
    return {
        "correlation": corr,
        "standard_error": error,
        "pair_counts": np.asarray(pairs),
        "nseries": int(valid.sum()),
        "aggregation": aggregation,
    }


def relaxation_metrics(correlation, lag_times):
    """Return first-1/e and positive-integral correlation times."""
    correlation = np.asarray(correlation, dtype=float)
    lag_times = np.asarray(lag_times, dtype=float)
    if correlation.shape != lag_times.shape:
        raise ValueError("correlation and lag_times must match")
    target = np.exp(-1)
    found = np.flatnonzero(correlation <= target)
    one_e = None
    if len(found):
        i = int(found[0])
        if i == 0:
            one_e = lag_times[0]
        else:
            f = (correlation[i - 1] - target) / (
                correlation[i - 1] - correlation[i]
            )
            one_e = lag_times[i - 1] + f * (lag_times[i] - lag_times[i - 1])
    zeros = np.flatnonzero(correlation <= 0)
    stop = int(zeros[0]) if len(zeros) else len(correlation)
    integral = float(np.trapz(correlation[:stop], lag_times[:stop])) if stop > 1 else 0.0
    if len(zeros) and stop > 0:
        f = correlation[stop - 1] / (correlation[stop - 1] - correlation[stop])
        zero_time = lag_times[stop - 1] + f * (
            lag_times[stop] - lag_times[stop - 1]
        )
        integral += 0.5 * correlation[stop - 1] * (
            zero_time - lag_times[stop - 1]
        )
    return {
        "one_over_e_time": one_e,
        "positive_integral_time": integral,
        "resolved_1e": one_e is not None,
        "reached_zero": bool(len(zeros)),
    }


def polar_bond_labels(phi_frames, dir_coup, tolerance_deg=45.0):
    """Classify forward bonds as polar (1), anti-polar (-1), or mixed (0)."""
    if not 0 <= tolerance_deg < 90:
        raise ValueError("tolerance_deg must be in [0, 90)")
    phi_frames = np.asarray(phi_frames, dtype=float)
    delta = np.abs(
        _wrap180(np.roll(phi_frames, -1, axis=dir_coup + 1) - phi_frames)
    )
    labels = np.zeros(delta.shape, dtype=np.int8)
    labels[delta <= tolerance_deg] = 1
    labels[delta >= 180 - tolerance_deg] = -1
    return labels


def polar_domain_states(
    phi_frames, dir_coup, domain_size, tolerance_deg=45.0, origin_mode="all"
):
    """Classify molecular windows as polar (1), anti-polar (-1), or mixed (0)."""
    labels = polar_bond_labels(phi_frames, dir_coup, tolerance_deg)
    chain_length = labels.shape[dir_coup + 1]
    origins = _domain_origins(chain_length, domain_size, origin_mode)
    chains = np.moveaxis(labels, dir_coup + 1, -1).reshape(
        labels.shape[0], -1, chain_length
    )
    states = np.zeros((len(labels), chains.shape[1], len(origins)), dtype=np.int8)
    for oi, origin in enumerate(origins):
        inds = (origin + np.arange(domain_size - 1)) % chain_length
        bonds = chains[:, :, inds]
        states[:, :, oi] = np.where(
            np.all(bonds == 1, axis=-1), 1,
            np.where(np.all(bonds == -1, axis=-1), -1, 0),
        )
    return {"states": states, "origins": origins}


def state_dwell_runs(states, times):
    """Return dwell events with separate left/right edge-censoring flags."""
    states = np.asarray(states)
    times = np.asarray(times, dtype=float)
    if len(times) != states.shape[0] or len(times) < 2:
        raise ValueError("times must contain at least two matching frames")
    dt = np.median(np.diff(times))
    matrix = states.reshape(len(times), -1)
    runs = []
    for series in range(matrix.shape[1]):
        start = 0
        while start < len(times):
            state = int(matrix[start, series])
            stop = start
            while stop + 1 < len(times) and matrix[stop + 1, series] == state:
                stop += 1
            if state:
                left_censored = start == 0
                right_censored = stop == len(times) - 1
                runs.append({
                    "series": series, "state": state, "start_index": start,
                    "end_index": stop, "duration": (stop - start + 1) * dt,
                    "left_censored": left_censored,
                    "right_censored": right_censored,
                    "censored": left_censored or right_censored,
                })
            start = stop + 1
    return runs


def state_survival(runs, state):
    """Compute Kaplan-Meier survival, excluding unknown-age left-censored runs."""
    selected = [
        run for run in runs
        if run["state"] == state and not run["left_censored"]
    ]
    durations = sorted({run["duration"] for run in selected})
    survival, at_risk, events, censored = [], [], [], []
    current = 1.0
    for time in durations:
        risk = sum(run["duration"] >= time for run in selected)
        event = sum(
            run["duration"] == time and not run["right_censored"]
            for run in selected
        )
        censor = sum(
            run["duration"] == time and run["right_censored"]
            for run in selected
        )
        current *= 1 - event / risk
        survival.append(current); at_risk.append(risk)
        events.append(event); censored.append(censor)
    return {
        "times": np.asarray(durations), "survival": np.asarray(survival),
        "at_risk": np.asarray(at_risk), "events": np.asarray(events),
        "censored": np.asarray(censored),
    }


def survival_metrics(survival):
    """Return the KM median and restricted mean through the last duration."""
    times = np.asarray(survival["times"], dtype=float)
    values = np.asarray(survival["survival"], dtype=float)
    if times.shape != values.shape:
        raise ValueError("survival times and values must match")
    if not len(times):
        return {
            "median_time": None, "restricted_mean_time": None,
            "restriction_time": None, "median_resolved": False,
        }
    if np.any(np.diff(times) < 0):
        raise ValueError("survival times must be sorted")
    crossings = np.flatnonzero(values <= 0.5)
    median_time = None if not len(crossings) else float(times[crossings[0]])
    previous_times = np.r_[0.0, times[:-1]]
    previous_survival = np.r_[1.0, values[:-1]]
    restricted_mean = float(np.sum((times - previous_times) * previous_survival))
    return {
        "median_time": median_time,
        "restricted_mean_time": restricted_mean,
        "restriction_time": float(times[-1]),
        "median_resolved": median_time is not None,
    }


def _analysis_field(phi_frames, dir_coup, field):
    phi_frames = np.asarray(phi_frames, dtype=float)
    if field == "orientation":
        return np.exp(1j * np.deg2rad(phi_frames))
    delta = _wrap180(
        np.roll(phi_frames, -1, axis=dir_coup + 1) - phi_frames
    )
    if field == "bond_complex":
        return np.exp(1j * np.deg2rad(delta))
    if field == "bond_polarity":
        return np.cos(np.deg2rad(delta))
    raise ValueError("field must be orientation, bond_complex, or bond_polarity")


def spatial_correlation(
    phi_frames, dir_coup, field="orientation", connected=True,
    stagger=False, block_length=None,
):
    """Compute signed equal-time spatial correlations through half a chain."""
    q = _analysis_field(phi_frames, dir_coup, field)
    chain_length = q.shape[dir_coup + 1]
    if stagger:
        shape = [1] * 4
        shape[dir_coup + 1] = chain_length
        q = q * np.exp(-1j * np.pi * np.arange(chain_length)).reshape(shape)
    mean_q = q.mean()
    distances = np.arange(chain_length // 2 + 1)
    raw, error = [], []
    for distance in distances:
        product = np.real(
            np.conjugate(q) * np.roll(q, -distance, axis=dir_coup + 1)
        )
        frame_values = product.mean(axis=(1, 2, 3))
        if connected:
            frame_values = frame_values - abs(mean_q) ** 2
        raw.append(frame_values.mean())
        error.append(_block_sem(frame_values, block_length))
    raw = np.asarray(raw)
    error = np.asarray(error)
    if raw[0] <= np.finfo(float).eps:
        raise ValueError("zero-distance correlation is not positive")
    return {
        "distances": distances, "correlation": raw / raw[0],
        "standard_error": error / raw[0], "raw_correlation": raw,
        "mean_order": mean_q, "field": field, "connected": connected,
        "stagger": stagger,
    }


def chain_structure_factor(phi_frames, dir_coup, field="orientation", connected=True):
    """Return the full discrete one-dimensional structure factor."""
    qfield = _analysis_field(phi_frames, dir_coup, field)
    if connected:
        qfield = qfield - qfield.mean()
    chain_length = qfield.shape[dir_coup + 1]
    chains = np.moveaxis(qfield, dir_coup + 1, -1).reshape(-1, chain_length)
    indices = np.arange(chain_length)
    wavevectors = 2 * np.pi * indices / chain_length
    intensity = []
    for q in wavevectors:
        phase = np.exp(-1j * q * indices)
        amplitude = chains @ phase / np.sqrt(chain_length)
        intensity.append(np.mean(np.abs(amplitude) ** 2))
    return {
        "indices": indices, "wavevectors": wavevectors,
        "intensity": np.asarray(intensity), "field": field,
        "connected": connected,
    }


def periodic_label_runs(labels):
    """Return circular label/length runs."""
    labels = np.asarray(labels)
    if not len(labels):
        return []
    if np.all(labels == labels[0]):
        return [{"label": int(labels[0]), "length": len(labels)}]
    boundary = int(np.flatnonzero(labels != np.roll(labels, 1))[0])
    ordered = np.roll(labels, -boundary)
    runs, start = [], 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        runs.append({"label": int(ordered[start]), "length": stop - start + 1})
        start = stop + 1
    return runs


def bond_domain_run_lengths(phi_frames, dir_coup, tolerance_deg=45.0):
    """Count circular polar, anti-polar, and unclassified bond runs."""
    labels = polar_bond_labels(phi_frames, dir_coup, tolerance_deg)
    chain_length = labels.shape[dir_coup + 1]
    chains = np.moveaxis(labels, dir_coup + 1, -1).reshape(
        labels.shape[0], -1, chain_length
    )
    counts = {}
    for frame in chains:
        for chain in frame:
            for run in periodic_label_runs(chain):
                key = (run["label"], run["length"])
                counts[key] = counts.get(key, 0) + 1
    return {
        "counts": counts, "nframes": chains.shape[0],
        "nchains": chains.shape[1], "chain_length": chain_length,
    }


def second_moment_correlation_length(structure_factor):
    """Estimate a finite-box second-moment scale around the largest peak."""
    intensity = np.asarray(structure_factor["intensity"])
    n = len(intensity)
    peak = int(np.argmax(intensity))
    neighbor = (intensity[(peak - 1) % n] + intensity[(peak + 1) % n]) / 2
    ratio = intensity[peak] / neighbor
    resolved = bool(np.isfinite(ratio) and ratio > 1)
    length_cells = (
        np.sqrt(ratio - 1) / (2 * np.sin(np.pi / n)) if resolved else None
    )
    q = float(structure_factor["wavevectors"][peak])
    if q > np.pi:
        q -= 2 * np.pi
    return {
        "peak_index": int(structure_factor["indices"][peak]),
        "peak_wavevector": q, "peak_intensity": float(intensity[peak]),
        "neighbor_intensity": float(neighbor), "length_cells": length_cells,
        "resolved": resolved,
    }
