# %% Time the MD of steps 02 and 03 before choosing a run length
# Runs N_STEPS of Langevin MD on the md_cell model, plain and with the
# metadynamics bias on the coordination number, and prints seconds per step.
# Run from the repository root:  python analysis/time_md.py   (GPU node)

from pathlib import Path
import time

import numpy as np
from ase.io import read

from snmini import structures as S
from snmini.calc import get_calculator, langevin, thermalize
from snmini.cv import CoordinationCV
from snmini.metad import KB_EV, BiasedCalculator, WellTemperedMetaD1D

# ----------------------------- CONFIG ------------------------------------
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
CALC, MODEL, DISPERSION, DEVICE, DTYPE = "mace", "medium", True, "cuda", "float32"
TEMPERATURE = 350.0        # K
TIMESTEP_FS = 0.5
N_STEPS = 1000
R0 = 2.6
# --------------------------------------------------------------------------

atoms = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn, ow = int(atoms.info["sn_index"]), S.water_oxygens(atoms)
kT = KB_EV * TEMPERATURE
base = get_calculator(CALC, MODEL, DISPERSION, DEVICE, DTYPE)
rng = np.random.default_rng(0)


def run(calc, label):
    atoms.calc = calc
    thermalize(atoms, TEMPERATURE, rng)
    dyn = langevin(atoms, TIMESTEP_FS, TEMPERATURE, rng)
    dyn.run(50)                                   # warm up (kernels, caches)
    t0 = time.perf_counter()
    dyn.run(N_STEPS)
    dt = (time.perf_counter() - t0) / N_STEPS
    print(f"{label}: {dt * 1000:.1f} ms/step, {len(atoms)} atoms, "
          f"{86400 / dt * TIMESTEP_FS * 1e-6:.2f} ns/day at {TIMESTEP_FS} fs")


run(base, "plain MD")
cv = CoordinationCV(sn, ow, r0=R0)
mtd = WellTemperedMetaD1D(sigma=0.07, height=0.8 * kT, bias_factor=10.0, kT=kT,
                          grid_min=-0.5, grid_max=len(ow) + 0.5)
run(BiasedCalculator(base, cv, mtd), "biased MD (CN bias)")
