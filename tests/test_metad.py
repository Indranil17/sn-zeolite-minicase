import numpy as np

from ase import Atoms
from snmini.calc import ZeroCalculator
from snmini.cv import CoordinationCV
from snmini.metad import (BiasedCalculator, WellTemperedMetaD1D,
                          basin_populations)
from snmini.toy import exact_free_energy, run_metad_double_well


def test_double_well_free_energy_recovered():
    md, _ = run_metad_double_well(n_steps=400_000, tilt=1.0, seed=1)
    x, F = md.free_energy()
    Fe = exact_free_energy(x, tilt=1.0)
    m = (x > -1.5) & (x < 1.5)
    err = (F - F[m].min())[m] - (Fe - Fe[m].min())[m]
    assert np.sqrt(np.mean(err**2)) < 0.35          # kT
    b = {"left": (-2.5, 0.0), "right": (0.0, 2.5)}
    p, pe = basin_populations(x, F, b, 1.0), basin_populations(x, Fe, b, 1.0)
    assert abs(p["right"] - pe["right"]) < 0.06


def test_well_tempered_heights_decay():
    md = WellTemperedMetaD1D(0.1, 1.0, 5.0, 1.0, -2, 2)
    h = [md.add_hill(0.0) for _ in range(50)]
    assert h[0] == 1.0 and all(b < a for a, b in zip(h, h[1:]))


def test_rebuild_from_hills():
    md = WellTemperedMetaD1D(0.1, 0.5, 6.0, 1.0, -2, 2)
    for s in np.linspace(-1, 1, 30):
        md.add_hill(s)
    md2 = WellTemperedMetaD1D.from_hills(md.hills, sigma=0.1, height=0.5,
                                         bias_factor=6.0, kT=1.0,
                                         grid_min=-2, grid_max=2)
    assert np.allclose(md.V, md2.V) and np.allclose(md.dV, md2.dV)


def test_bias_forces_are_minus_gradient_of_bias_energy():
    rng = np.random.default_rng(3)
    a = Atoms("SnO4", positions=rng.uniform(0, 5, (5, 3)), cell=[5, 5, 5],
              pbc=True)
    cv = CoordinationCV(0, [1, 2, 3, 4], r0=2.4)
    md = WellTemperedMetaD1D(0.2, 0.05, 8.0, 0.03, -0.5, 4.5)
    for s in np.linspace(0, 3, 25):
        md.add_hill(s)
    a.calc = BiasedCalculator(ZeroCalculator(), cv, md)
    f = a.get_forces()
    num = np.zeros_like(f)
    for i in range(len(a)):
        for k in range(3):
            b = a.copy()
            b.calc = BiasedCalculator(ZeroCalculator(), cv, md)
            b.positions[i, k] += 1e-5
            ep = b.get_potential_energy()
            b.positions[i, k] -= 2e-5
            em = b.get_potential_energy()
            num[i, k] = -(ep - em) / 2e-5
    assert np.allclose(f, num, atol=1e-5)
