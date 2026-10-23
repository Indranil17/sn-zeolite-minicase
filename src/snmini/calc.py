"""Calculator factory: the MACE foundation model, or a zero stand in.

The zero calculator lets every script run end to end on a laptop or in CI
(the plumbing is tested), while making it impossible to mistake its output for
physics: all energies are exactly zero.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n = len(self.atoms)
        self.results = dict(energy=0.0, free_energy=0.0,
                            forces=np.zeros((n, 3)), stress=np.zeros(6))


def get_calculator(kind="mace", model="medium", dispersion=True,
                   device="cuda", dtype="float64"):
    """kind = "mace" (foundation model through mace_mp) or "toy"."""
    if kind == "toy":
        return ZeroCalculator()
    if kind == "mace":
        from mace.calculators import mace_mp   # imported only when needed
        return mace_mp(model=model, dispersion=dispersion,
                       default_dtype=dtype, device=device)
    raise ValueError(f"unknown calculator kind: {kind}")


def thermalize(atoms, temperature_K, rng):
    """Maxwell-Boltzmann velocities with zero total momentum (any ASE >= 3.23)."""
    try:
        from ase.md.velocitydistribution import thermalize_momenta
        thermalize_momenta(atoms, temperature_K, rng=rng)
    except ImportError:
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K,
                                     rng=rng)
    from ase.md.velocitydistribution import Stationary
    Stationary(atoms)


def langevin(atoms, timestep_fs, temperature_K, rng, friction_per_fs=0.01):
    """Langevin thermostat (fixcm=False: ASE >= 3.28 warns that fixcm=True
    does not sample NVT exactly)."""
    from ase import units
    from ase.md.langevin import Langevin
    return Langevin(atoms, timestep=timestep_fs * units.fs,
                    temperature_K=temperature_K,
                    friction=friction_per_fs / units.fs, fixcm=False, rng=rng)
