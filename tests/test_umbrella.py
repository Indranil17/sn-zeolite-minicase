"""Umbrella sampling: the harmonic bias and MBAR against exact answers."""
import numpy as np

from snmini.umbrella import (HarmonicBias, mbar_f, pmf, reduced_energies,
                             unbiased_weights, window_probability)


def test_harmonic_bias_derivative():
    b = HarmonicBias(1.2, 3.0)
    s, h = 0.9, 1e-6
    fd = (b.bias(s + h) - b.bias(s - h)) / (2 * h)
    assert abs(fd - b.dbias(s)) < 1e-8
    assert b.bias(1.2) == 0.0 and b.dbias(1.2) == 0.0


def test_mbar_two_gaussians_exact():
    # states k with potential k x^2 / 2: Z_k = sqrt(2 pi / k), so f_2 - f_1 = 0.5 ln(k_2 / k_1)
    rng = np.random.default_rng(1)
    ks, n = [1.0, 4.0], 20000
    x = np.concatenate([rng.normal(0, 1 / np.sqrt(k), n) for k in ks])
    u = np.array([0.5 * k * x ** 2 for k in ks])
    f = mbar_f(u, [n, n])
    assert abs((f[1] - f[0]) - 0.5 * np.log(ks[1] / ks[0])) < 0.03


def test_umbrella_mbar_recovers_double_well():
    # U(x) = 2 (x^2 - 1)^2 in units of kT, harmonic windows, Metropolis sampling
    rng = np.random.default_rng(7)
    U = lambda x: 2.0 * (x ** 2 - 1.0) ** 2
    centres = np.arange(-2.4, 2.41, 0.2)
    k = 10.0
    x = centres.copy()
    samples = []
    for step in range(24000):
        prop = x + rng.normal(0, 0.3, x.size)
        dU = (U(prop) + 0.5 * k * (prop - centres) ** 2) - (U(x) + 0.5 * k * (x - centres) ** 2)
        x = np.where(rng.random(x.size) < np.exp(-dU), prop, x)
        if step >= 2000 and step % 20 == 0:
            samples.append(x.copy())
    samples = np.array(samples)                            # (n, windows)
    n_k = [samples.shape[0]] * centres.size
    s = samples.T.reshape(-1)                              # pooled, window by window
    u_kn = reduced_energies(s, centres, k, 1.0)
    f = mbar_f(u_kn, n_k)
    w = unbiased_weights(u_kn, n_k, f)
    edges = np.linspace(-1.6, 1.6, 33)
    F = pmf(s, w, edges, 1.0)
    mid = 0.5 * (edges[1:] + edges[:-1])
    exact = U(mid) - U(mid).min()
    assert np.max(np.abs(F - exact)) < 0.25                # within 0.25 kT everywhere in the well region
    # probability of the right well against the exact value
    xs = np.linspace(-3, 3, 60001)
    trapz = getattr(np, "trapezoid", None) or np.trapz        # numpy 1 and 2
    pr = trapz(np.exp(-U(xs)) * (xs > 0), xs) / trapz(np.exp(-U(xs)), xs)
    assert abs(window_probability(s, w, 0.0, 3.0) - pr) < 0.03
