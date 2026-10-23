"""Population weighted spectra.

If Sn switches between coordination states more slowly than plain MD can
sample, a time average over one trajectory is not the equilibrium spectrum.
The equilibrium spectrum is the sum over states of (state population from the
free energy) x (state resolved spectrum from snapshots of that state).
"""
from __future__ import annotations

import numpy as np


def gaussian_lines(grid, centres, widths, weights):
    y = np.zeros_like(grid, dtype=float)
    for c, w, a in zip(centres, widths, weights):
        y += a * np.exp(-0.5 * ((grid - c) / w) ** 2) / (w * np.sqrt(2 * np.pi))
    return y


def weighted_spectrum(grid, state_shifts, populations, min_width=1.0):
    """Spectrum from per state shift samples and state populations.

    state_shifts: dict state -> array of shifts (ppm) from snapshots
    populations:  dict state -> probability (sums to 1 over the states used)
    Each state is broadened by the spread of its own snapshots (dynamic
    disorder), never narrower than min_width ppm.
    Returns (intensity, weighted mean shift).
    """
    names = [k for k in populations if k in state_shifts and
             len(state_shifts[k])]
    p = np.array([populations[k] for k in names], dtype=float)
    p = p / p.sum()
    centres = np.array([np.mean(state_shifts[k]) for k in names])
    widths = np.array([max(np.std(state_shifts[k]), min_width)
                       for k in names])
    y = gaussian_lines(grid, centres, widths, p)
    return y, float(np.sum(p * centres))


def bootstrap_weighted_mean(state_shifts, populations, n_boot=2000, seed=0):
    """95 % interval of the population weighted mean shift.

    Resamples snapshots within each state; populations are held fixed
    (propagate their own uncertainty separately, e.g. from metadynamics
    blocks).
    """
    rng = np.random.default_rng(seed)
    names = [k for k in populations if k in state_shifts]
    p = np.array([populations[k] for k in names], dtype=float)
    p = p / p.sum()
    vals = []
    for _ in range(n_boot):
        m = [np.mean(rng.choice(state_shifts[k], len(state_shifts[k])))
             for k in names]
        vals.append(float(np.sum(p * np.array(m))))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
