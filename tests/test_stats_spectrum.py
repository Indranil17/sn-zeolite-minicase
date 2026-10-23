import numpy as np

from snmini import stats
from snmini.spectrum import bootstrap_weighted_mean, weighted_spectrum


def ar1(phi, n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_integrated_time_of_ar1():
    phi = 0.8
    tau = stats.integrated_time(ar1(phi, 200_000))
    assert abs(tau - (1 + phi) / (1 - phi)) / ((1 + phi) / (1 - phi)) < 0.1


def test_block_and_acf_errors_agree():
    x = ar1(0.9, 100_000, seed=1)
    a, b = stats.sem_autocorr(x), stats.block_sem(x, 20)
    assert 0.6 < a / b < 1.6


def test_blocking_plateau_exceeds_naive_error():
    x = ar1(0.9, 2**15, seed=2)
    _, sem, _ = stats.flyvbjerg_petersen(x)
    assert sem[-3] > 2 * sem[0]        # correlation inflates the error


def test_weighted_spectrum_mean_and_norm():
    grid = np.linspace(-800, -300, 5001)
    shifts = {"a": np.array([-440.0, -444.0]), "b": np.array([-690.0])}
    y, mean = weighted_spectrum(grid, shifts, {"a": 0.25, "b": 0.75})
    assert np.isclose(mean, 0.25 * -442.0 + 0.75 * -690.0)
    assert np.isclose(np.trapezoid(y, grid) if hasattr(np, "trapezoid")
                      else np.trapz(y, grid), 1.0, atol=1e-3)
    lo, hi = bootstrap_weighted_mean(shifts, {"a": 0.25, "b": 0.75})
    assert lo <= mean <= hi
