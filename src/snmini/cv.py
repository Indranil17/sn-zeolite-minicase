"""Collective variable: coordination number of one atom by a group of atoms.

    s = sum_j f(r_j),   f(r) = (1 - (r/r0)**n) / (1 - (r/r0)**m)

This is the rational switching function used by PLUMED's COORDINATION
(defaults n = 6, m = 2n), so a PLUMED run with the same r0, n and m measures
exactly the same number (cross-check input in plumed/).
"""
from __future__ import annotations

import numpy as np


def switching(r, r0, n=6, m=12):
    """f(r) and df/dr, with the removable singularity at r = r0 handled."""
    x = np.asarray(r, dtype=float) / r0
    f = np.empty_like(x)
    df = np.empty_like(x)
    near = np.abs(x - 1.0) < 1e-4
    xs = np.where(near, 0.5, x)          # placeholder, overwritten below
    num, den = 1.0 - xs**n, 1.0 - xs**m
    f[:] = num / den
    dfdx = (-n * xs**(n - 1) * den + m * xs**(m - 1) * num) / den**2
    df[:] = dfdx / r0
    # first order Taylor expansion around x = 1, where num and den -> 0:
    # f = n/m + c (x - 1),  c = n (n - m) / (2 m)
    c = n * (n - m) / (2.0 * m)
    f[near] = n / m + c * (x[near] - 1.0)
    df[near] = c / r0
    return f, df


class CoordinationCV:
    """Coordination number of ``center`` by the atoms in ``group``.

    __call__(atoms) returns (s, grad) with grad = ds/dR, shape (N, 3).
    Distances use the minimum image convention.
    """

    def __init__(self, center, group, r0=2.6, n=6, m=12):
        self.center = int(center)
        self.group = np.asarray(group, dtype=int)
        self.r0, self.n, self.m = float(r0), int(n), int(m)

    def __call__(self, atoms):
        vec = atoms.get_distances(self.center, self.group, mic=True,
                                  vector=True)
        vec = np.atleast_2d(vec)
        r = np.linalg.norm(vec, axis=1)
        f, df = switching(r, self.r0, self.n, self.m)
        s = float(f.sum())
        grad = np.zeros((len(atoms), 3))
        unit = vec / r[:, None]
        g = df[:, None] * unit            # d f / d R_j  (vector from centre)
        grad[self.group] += g
        grad[self.center] -= g.sum(axis=0)
        return s, grad

    def plumed_line(self, label="cn"):
        """Equivalent PLUMED action (1 based atom indices, lengths in Angstrom
        when the input starts with UNITS LENGTH=A)."""
        grp = ",".join(str(i + 1) for i in self.group)
        return (f"{label}: COORDINATION GROUPA={self.center + 1} GROUPB={grp} "
                f"R_0={self.r0} NN={self.n} MM={self.m}")
