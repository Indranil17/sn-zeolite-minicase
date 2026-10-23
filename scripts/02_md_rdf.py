# %% 02  Unbiased MD of the water loaded Sn cell: Sn-O(water) RDF and r0
# Why: the coordination number used as collective variable in 03 needs a
# cutoff r0. Taking r0 at the first minimum of the Sn-O(water) radial
# distribution function lets the data, not a guess, decide what "bound" means.

from pathlib import Path
import json

import numpy as np
from ase.io import read, write

from snmini import structures as S
from snmini.calc import get_calculator, langevin, thermalize
from snmini.cv import CoordinationCV
from snmini.stats import integrated_time, sem_autocorr

# ----------------------------- CONFIG ------------------------------------
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
CALC, MODEL, DISPERSION, DEVICE, DTYPE = "mace", "medium", True, "cuda", \
    "float32"              # float32 is fine and faster for dynamics
TEMPERATURE = 350.0        # K
TIMESTEP_FS = 0.5
STEPS = 200_000            # 100 ps
EQUIL_STEPS = 20_000       # discarded from analysis
SAVE_EVERY = 100           # steps between stored frames (50 fs)
R0_GUESS = 2.6             # Angstrom, only used for the CN time series
SEED = 2
OUTDIR = Path("outputs/02_md")
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
atoms = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn = int(atoms.info["sn_index"])
ow = S.water_oxygens(atoms)
print(f"{LABEL}: {atoms.get_chemical_formula()}, Sn {sn}, {len(ow)} waters")

atoms.calc = get_calculator(CALC, MODEL, DISPERSION, DEVICE, DTYPE)
rng = np.random.default_rng(SEED)
thermalize(atoms, TEMPERATURE, rng)
dyn = langevin(atoms, TIMESTEP_FS, TEMPERATURE, rng)

cv = CoordinationCV(sn, ow, r0=R0_GUESS)
frames, cn, dists, log = [], [], [], []


def record():
    step = dyn.get_number_of_steps()
    s, _ = cv(atoms)
    log.append((step * TIMESTEP_FS / 1000.0, s, atoms.get_temperature(),
                atoms.get_potential_energy()))
    if step >= EQUIL_STEPS:
        cn.append(s)
        dists.append(atoms.get_distances(sn, ow, mic=True))
        frames.append(atoms.copy())


dyn.attach(record, interval=SAVE_EVERY)
dyn.run(STEPS)

# radial distribution function Sn-O(water), normalised to the ideal gas
d = np.concatenate(dists)
edges = np.linspace(1.5, 6.0, 181)
hist, _ = np.histogram(d, bins=edges)
r = 0.5 * (edges[1:] + edges[:-1])
shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
rho = len(ow) / atoms.get_volume()
g = hist / (len(dists) * rho * shell)

# first minimum after the first peak (smoothed)
gs = np.convolve(g, np.ones(5) / 5, mode="same")
peak = np.argmax(np.where(r < 3.0, gs, -1))
win = np.where((r > r[peak]) & (r < 4.0))[0]
r0 = float(r[win[np.argmin(gs[win])]]) if len(win) else R0_GUESS

cn = np.array(cn)
summary = dict(mean_cn=float(cn.mean()), sem_cn=sem_autocorr(cn),
               tau_samples=integrated_time(cn),
               tau_ps=integrated_time(cn) * SAVE_EVERY * TIMESTEP_FS / 1000,
               r_first_peak=float(r[peak]), r0_recommended=r0,
               frames=len(frames))
print(json.dumps(summary, indent=1))

np.savetxt(OUTDIR / "rdf_sn_ow.dat", np.c_[r, g], header="r_A g(r)")
np.savetxt(OUTDIR / "log.dat", np.array(log),
           header="time_ps CN(r0_guess) T_K Epot_eV")
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=1))
write(OUTDIR / "md_frames.extxyz", frames, format="extxyz")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(r, g)
    ax[0].axvline(r0, ls="--", c="k")
    ax[0].set(xlabel="r(Sn-O$_w$) / Å", ylabel="g(r)",
              title=f"r0 = {r0:.2f} Å")
    t = np.array(log)[:, 0]
    ax[1].plot(t, np.array(log)[:, 1], lw=0.6)
    ax[1].set(xlabel="time / ps", ylabel=f"CN (r0 = {R0_GUESS} Å)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "rdf_cn.png", dpi=200)
except ImportError:
    pass
print(f"use R0 = {r0:.2f} A in 03_metad.py")
