"""A one dimensional double well where the free energy is known exactly.

Used to validate the metadynamics engine before trusting it on a zeolite:
if the engine cannot recover F(x) = U(x) here, nothing it says about Sn
coordination can be believed.
"""
from __future__ import annotations

import numpy as np

from .metad import WellTemperedMetaD1D


def double_well(x, barrier=5.0, tilt=0.0):
    """U(x) = barrier (x^2 - 1)^2 + tilt x, in units of kT; and dU/dx."""
    u = barrier * (x**2 - 1.0) ** 2 + tilt * x
    du = 4.0 * barrier * x * (x**2 - 1.0) + tilt
    return u, du


def run_metad_double_well(n_steps=400_000, dt=0.005, friction=1.0,
                          barrier=5.0, tilt=0.0, sigma=0.1, height=0.3,
                          bias_factor=8.0, pace=200, seed=1):
    """Langevin (BAOAB, kT = m = 1) with well tempered metadynamics on x.

    Returns the metadynamics object and the trajectory of x (every pace).
    """
    rng = np.random.default_rng(seed)
    md = WellTemperedMetaD1D(sigma=sigma, height=height,
                             bias_factor=bias_factor, kT=1.0,
                             grid_min=-2.5, grid_max=2.5, n_bins=1001)
    c1 = np.exp(-friction * dt)
    c2 = np.sqrt(1.0 - c1**2)
    x, v = -1.0, 0.0

    def force(x):
        _, du = double_well(x, barrier, tilt)
        return -du - md.dbias(x)

    f = force(x)
    xs = []
    for step in range(1, n_steps + 1):
        v += 0.5 * dt * f
        x += 0.5 * dt * v
        v = c1 * v + c2 * rng.standard_normal()
        x += 0.5 * dt * v
        f = force(x)
        v += 0.5 * dt * f
        if step % pace == 0:
            md.add_hill(x)
            xs.append(x)
    return md, np.array(xs)


def exact_free_energy(grid, barrier=5.0, tilt=0.0):
    u, _ = double_well(grid, barrier, tilt)
    return u - u.min()
