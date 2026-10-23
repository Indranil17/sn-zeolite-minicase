"""Well tempered metadynamics on one collective variable.

Bias:      V(s,t) = sum_k h_k exp(-(s - s_k)^2 / (2 sigma^2))
Heights:   h_k = h0 * exp(-V(s_k, t_k) / (kB * dT)),   dT = (gamma - 1) T
Estimate:  F(s) = -(gamma / (gamma - 1)) * V(s, t)     (up to a constant)

The bias lives on a grid, as in PLUMED, so evaluating it costs the same at the
first and the ten thousandth hill. ``BiasedCalculator`` adds the bias to any
ASE calculator (MACE, DFT, a toy model).
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

KB_EV = 8.617333262e-5  # eV / K
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1 and 2


class WellTemperedMetaD1D:
    def __init__(self, sigma, height, bias_factor, kT, grid_min, grid_max,
                 n_bins=2001):
        if bias_factor <= 1:
            raise ValueError("bias_factor must be > 1")
        self.sigma, self.h0, self.gamma = float(sigma), float(height), \
            float(bias_factor)
        self.kT = float(kT)
        self.kdT = (self.gamma - 1.0) * self.kT     # k_B * Delta T
        self.grid = np.linspace(grid_min, grid_max, n_bins)
        self.V = np.zeros(n_bins)
        self.dV = np.zeros(n_bins)
        self.hills = []                              # (centre, height)

    def bias(self, s):
        return float(np.interp(s, self.grid, self.V))

    def dbias(self, s):
        if s < self.grid[0] or s > self.grid[-1]:
            return 0.0
        return float(np.interp(s, self.grid, self.dV))

    def add_hill(self, s):
        h = self.h0 * np.exp(-self.bias(s) / self.kdT)
        x = self.grid - s
        g = np.exp(-0.5 * (x / self.sigma) ** 2)
        self.V += h * g
        self.dV += -h * x / self.sigma**2 * g
        self.hills.append((float(s), float(h)))
        return h

    def free_energy(self):
        """F(s) on the grid, shifted so that its minimum is zero."""
        F = -(self.gamma / (self.gamma - 1.0)) * self.V
        return self.grid.copy(), F - F.min()

    @classmethod
    def from_hills(cls, hills, **kw):
        """Rebuild the bias from a list of (centre, height) pairs."""
        m = cls(**kw)
        for s, h in hills:
            x = m.grid - s
            g = np.exp(-0.5 * (x / m.sigma) ** 2)
            m.V += h * g
            m.dV += -h * x / m.sigma**2 * g
            m.hills.append((float(s), float(h)))
        return m


def basin_populations(s, F, basins, kT):
    """Probability of each basin, p_k = int_basin exp(-F/kT) ds / sum.

    basins: dict name -> (s_low, s_high)
    """
    w = np.exp(-(F - F.min()) / kT)
    out = {}
    for name, (lo, hi) in basins.items():
        m = (s >= lo) & (s < hi)
        out[name] = float(_trapz(w[m], s[m])) if m.sum() > 1 else 0.0
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


class BiasedCalculator(Calculator):
    """Wrap an ASE calculator and add a metadynamics bias on one CV."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base, cv, metad, **kw):
        super().__init__(**kw)
        self.base, self.cv, self.metad = base, cv, metad
        self.s = None
        self.v = 0.0

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, ["energy", "forces"], system_changes)
        e = float(self.base.results["energy"])
        f = np.array(self.base.results["forces"], dtype=float)
        s, grad = self.cv(atoms)
        self.s = s
        self.v = self.metad.bias(s)
        self.results["energy"] = e + self.v
        self.results["free_energy"] = e + self.v
        self.results["forces"] = f - self.metad.dbias(s) * grad
