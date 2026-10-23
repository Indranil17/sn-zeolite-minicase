"""Error bars for correlated time series (MD observables, NMR shifts)."""
from __future__ import annotations

import numpy as np


def autocorrelation(x):
    """Normalised autocorrelation function rho(t), computed with FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def integrated_time(x, c=5.0):
    """Integrated autocorrelation time tau = 1 + 2 sum rho(t), in samples.

    Window chosen self consistently (Sokal): smallest M with M >= c * tau(M).
    Var(mean) = Var(x) * tau / N with this convention.
    """
    rho = autocorrelation(x)
    tau = 2.0 * np.cumsum(rho) - 1.0
    for m in range(1, len(tau)):
        if m >= c * tau[m]:
            return float(tau[m])
    return float(tau[-1])


def sem_autocorr(x):
    """Standard error of the mean corrected for autocorrelation."""
    x = np.asarray(x, dtype=float)
    return float(np.std(x, ddof=1) * np.sqrt(integrated_time(x) / len(x)))


def block_sem(x, n_blocks=10):
    """Standard error of the mean from n_blocks contiguous block averages."""
    x = np.asarray(x, dtype=float)
    size = len(x) // n_blocks
    means = x[: size * n_blocks].reshape(n_blocks, size).mean(axis=1)
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def flyvbjerg_petersen(x):
    """Blocking analysis (Flyvbjerg and Petersen 1989).

    Returns arrays (block_size, sem, sem_error). The plateau of sem against
    block size is the error bar to quote.
    """
    x = np.asarray(x, dtype=float)
    sizes, sems, errs = [], [], []
    size = 1
    while len(x) >= 4:
        n = len(x)
        var = np.var(x, ddof=0) / (n - 1)
        sem = np.sqrt(var)
        sizes.append(size)
        sems.append(sem)
        errs.append(sem / np.sqrt(2.0 * (n - 1)))
        if n % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
        size *= 2
    return np.array(sizes), np.array(sems), np.array(errs)


def bootstrap_ci(x, func=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval of func(x)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    vals = np.array([func(x[rng.integers(0, len(x), len(x))])
                     for _ in range(n_boot)])
    return float(np.quantile(vals, alpha / 2)), \
        float(np.quantile(vals, 1 - alpha / 2))
