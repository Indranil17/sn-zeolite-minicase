import numpy as np

from ase import Atoms
from snmini.cv import CoordinationCV, switching


def test_switching_limit_and_derivative():
    f, df = switching(np.array([2.6]), 2.6, 6, 12)
    assert np.isclose(f[0], 0.5)                       # n / m
    h = 1e-5
    fp, _ = switching(np.array([2.6 + h]), 2.6)
    fm, _ = switching(np.array([2.6 - h]), 2.6)
    assert np.isclose(df[0], (fp[0] - fm[0]) / (2 * h), rtol=1e-6)


def test_switching_derivative_everywhere():
    r = np.linspace(1.0, 5.0, 41)
    _, df = switching(r, 2.6)
    h = 1e-6
    fp, _ = switching(r + h, 2.6)
    fm, _ = switching(r - h, 2.6)
    assert np.allclose(df, (fp - fm) / (2 * h), atol=1e-6)


def test_cv_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 6, size=(7, 3))
    a = Atoms("SnO6", positions=pos, cell=[6, 6, 6], pbc=True)
    cv = CoordinationCV(0, [1, 2, 3, 4, 5, 6], r0=2.6)
    s, g = cv(a)
    num = np.zeros_like(g)
    for i in range(len(a)):
        for k in range(3):
            b = a.copy()
            b.positions[i, k] += 1e-6
            sp, _ = cv(b)
            b.positions[i, k] -= 2e-6
            sm, _ = cv(b)
            num[i, k] = (sp - sm) / 2e-6
    assert np.allclose(g, num, atol=1e-7)
    assert np.allclose(g.sum(axis=0), 0.0, atol=1e-10)   # translation invariant


def test_plumed_line_is_one_based():
    cv = CoordinationCV(0, [3, 5], r0=2.5)
    assert "GROUPA=1 GROUPB=4,6" in cv.plumed_line()
