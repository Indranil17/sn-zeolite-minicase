"""Umbrella sampling on one collective variable, analysed with MBAR.

Window k adds the harmonic bias V_k(s) = k_f (s - s_k)^2 / 2 to the potential.
``HarmonicBias`` has the same ``bias`` and ``dbias`` methods as the
metadynamics bias, so ``BiasedCalculator`` (snmini.metad) takes it unchanged.

MBAR (Shirts and Chodera, J. Chem. Phys. 129, 124105, 2008) combines the samples
of all windows. With u_kn the reduced energy (in kT) of pooled sample n in
window k and N_k the number of samples drawn in window k, the reduced free
energies f_k solve
    exp(-f_i) = sum_n exp(-u_in) / sum_k N_k exp(f_k - u_kn).
They are found here by minimising the convex function
    L(f) = mean_n ln sum_k N_k exp(f_k - u_kn) - sum_k (N_k / N) f_k,
whose gradient vanishes exactly at that solution. The weight of sample n in the
unbiased ensemble is then proportional to 1 / sum_k N_k exp(f_k - u_kn).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


class HarmonicBias:
    """V(s) = 0.5 * k * (s - centre)^2, energies in the unit of k times s^2 (eV)."""

    def __init__(self, centre, k):
        self.centre, self.k = float(centre), float(k)

    def bias(self, s):
        return 0.5 * self.k * (s - self.centre) ** 2

    def dbias(self, s):
        return self.k * (s - self.centre)


def reduced_energies(samples, centres, k, kT):
    """u_kn = V_k(s_n) / kT for pooled CV samples (shape N) in every window."""
    s = np.asarray(samples, float)[None, :]
    c = np.asarray(centres, float)[:, None]
    return 0.5 * k * (s - c) ** 2 / kT


def mbar_f(u_kn, N_k, gtol=1e-10):
    """Reduced free energies f_k of the windows (f_0 = 0)."""
    u_kn = np.asarray(u_kn, float)
    N_k = np.asarray(N_k, float)
    K, N = u_kn.shape
    logN = np.log(N_k)[:, None]

    def objective(f):
        a = logN + f[:, None] - u_kn
        log_den = logsumexp(a, axis=0)
        value = np.mean(log_den) - np.dot(N_k, f) / N
        grad = np.exp(a - log_den[None, :]).sum(axis=1) / N - N_k / N
        return value, grad

    res = minimize(objective, np.zeros(K), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=20000, ftol=1e-15, gtol=gtol))
    return res.x - res.x[0]


def unbiased_weights(u_kn, N_k, f_k):
    """Normalised weights of the pooled samples in the unbiased ensemble."""
    logN = np.log(np.asarray(N_k, float))[:, None]
    log_den = logsumexp(logN + np.asarray(f_k)[:, None] - u_kn, axis=0)
    w = np.exp(-log_den - (-log_den).max())
    return w / w.sum()


def pmf(samples, weights, edges, kT):
    """Free energy in each bin, F = -kT ln(sum of weights), minimum set to zero.
    Bins without samples are NaN."""
    h, _ = np.histogram(samples, bins=edges, weights=weights)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(h)
    F[~np.isfinite(F)] = np.nan
    return F - np.nanmin(F)


def window_probability(samples, weights, lo, hi):
    """Probability of the CV window [lo, hi) in the unbiased ensemble."""
    s = np.asarray(samples)
    return float(np.sum(np.asarray(weights)[(s >= lo) & (s < hi)]))
