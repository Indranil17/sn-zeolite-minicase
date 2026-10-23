# %% 03  Well tempered metadynamics on the Sn-O(water) coordination number
# Output in OUTDIR: HILLS and COLVAR (written and flushed as the run goes),
# frames.extxyz (EVERY stored frame, whatever its CN, with the CN, the bias and
# a flag for intact water), checkpoint.npz (positions and momenta, for RESTART),
# fes.dat, state_<window>.extxyz (intact frames of each CN window, for the GIPAW
# step), summary.json and metad.png.
# Nothing is overwritten: a run into an OUTDIR that has a HILLS file stops,
# unless RESTART = True, which continues from the checkpoint.
# The engine (snmini.metad) is validated first on an exactly solvable double
# well: run scripts/validate_metad_doublewell.py once before trusting this.
# What the free energy means: F(s) = -gamma/(gamma-1) V(s) holds where the bias
# went. Outside that CN range it is a flat extrapolation, so it is neither
# plotted nor used for populations.

from pathlib import Path
import json
import os

import numpy as np
from ase.io import read, write

from snmini import structures as S
from snmini.calc import get_calculator, langevin, thermalize
from snmini.cv import CoordinationCV
from snmini.metad import KB_EV, BiasedCalculator, WellTemperedMetaD1D

# ----------------------------- CONFIG ------------------------------------
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
CALC, MODEL, DISPERSION, DEVICE, DTYPE = "mace", "medium", True, "cuda", \
    "float32"
TEMPERATURE = 350.0        # K
TIMESTEP_FS = 0.5
STEPS = 450_000            # 225 ps; 1 ns is about 30 h on one GPU (55 ms/step)
R0 = 2.6                   # A; 02 recommends 2.51, both lie in the empty shell gap
SIGMA = 0.07               # Gaussian width in CN units
HEIGHT_KT = 1.0            # initial hill height in units of kT
BIAS_FACTOR = 10.0
PACE = 250                 # steps between hills (125 fs)
SAVE_EVERY = 200           # steps between stored frames (100 fs)
CHECKPOINT_EVERY = 5_000   # steps between checkpoints
N_TIMES = 10               # times at which dF is rebuilt from the hills
SEED = 3
OUTDIR = Path("outputs/03_metad")
RESTART = False            # True: continue from OUTDIR/checkpoint.npz and HILLS
# Windows for the free energy differences. 4 coordinate has no minimum (it is a
# shoulder), so only 5 and 6 coordinate are compared as windows; F at single CN
# values is reported for the shoulder.
DF_WINDOWS = {"5 coord": (0.65, 1.35), "6 coord": (1.35, 2.10)}
F_POINTS = [0.3, 0.4, 0.5]  # CN values at which F is reported, against the 5 coord minimum
# CN windows whose frames go to state_<name>.extxyz for GIPAW (only frames with
# all waters intact). In the 02 and 03 runs: 1.1 is one water at 2.15 A, 1.4 to
# 1.85 two waters, the second one soft (median 2.3 to 2.5 A); 0.2 to 0.55 is Sn without a bound water (99
# percent of those stored frames have no water within 2.8 A of Sn).
STATE_WINDOWS = {"4c": (0.20, 0.55), "5c": (0.90, 1.30), "6c": (1.40, 1.85)}
OH_CUT = 1.25              # A; an O-H distance above this counts as a broken water
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
hills_file, colvar_file = OUTDIR / "HILLS", OUTDIR / "COLVAR"
frames_file, ckpt_file = OUTDIR / "frames.extxyz", OUTDIR / "checkpoint.npz"
if hills_file.exists() and not RESTART:
    raise SystemExit(f"{hills_file} exists: use a new OUTDIR, or RESTART = True")

atoms = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn = int(atoms.info["sn_index"])
ow = S.water_oxygens(atoms)
h_all = [a.index for a in atoms if a.symbol == "H"]
kT = KB_EV * TEMPERATURE
cv = CoordinationCV(sn, ow, r0=R0)
kw = dict(sigma=SIGMA, height=HEIGHT_KT * kT, bias_factor=BIAS_FACTOR, kT=kT,
          grid_min=-0.5, grid_max=len(ow) + 0.5)

step0 = 0
if RESTART:                                 # rebuild the bias from the hills
    if not (hills_file.exists() and ckpt_file.exists()):
        raise SystemExit("RESTART needs HILLS and checkpoint.npz in OUTDIR")
    h0 = np.loadtxt(hills_file, comments="#", ndmin=2)
    mtd = WellTemperedMetaD1D.from_hills(list(zip(h0[:, 1], h0[:, 3])), **kw)
    ck = np.load(ckpt_file)
    atoms.set_positions(ck["positions"])
    atoms.set_momenta(ck["momenta"])
    step0 = int(ck["step"])
    print(f"restart at step {step0} with {len(h0)} hills")
else:
    mtd = WellTemperedMetaD1D(**kw)
base = get_calculator(CALC, MODEL, DISPERSION, DEVICE, DTYPE)
atoms.calc = BiasedCalculator(base, cv, mtd)
print(f"{LABEL}: {len(ow)} waters, kT = {kT:.4f} eV")
print("PLUMED equivalent:", cv.plumed_line())

rng = np.random.default_rng(SEED + step0)
if not RESTART:
    thermalize(atoms, TEMPERATURE, rng)
dyn = langevin(atoms, TIMESTEP_FS, TEMPERATURE, rng)
dyn.nsteps = step0                          # ASE counts steps from here
mode = "a" if RESTART else "w"
hills_fh, colvar_fh = open(hills_file, mode), open(colvar_file, mode)
if not RESTART:
    hills_fh.write("# time_ps  centre  sigma  height_eV\n")
    colvar_fh.write("# time_ps CN bias_eV\n")
last_done = step0 if RESTART else -1       # ASE calls the observer once before the first step


def save_checkpoint(step):
    tmp = OUTDIR / "checkpoint_tmp.npz"
    np.savez(tmp, positions=atoms.get_positions(), momenta=atoms.get_momenta(), step=step)
    os.replace(tmp, ckpt_file)             # replaces only this run's own file


def on_step():
    global last_done
    step = dyn.get_number_of_steps()
    if step <= last_done:
        return
    last_done = step
    t_ps = step * TIMESTEP_FS / 1000.0
    calc = atoms.calc
    if step % PACE == 0:
        h = mtd.add_hill(calc.s)
        hills_fh.write(f"{t_ps:.4f} {calc.s:.6f} {SIGMA} {h:.6e}\n")
        hills_fh.flush()
    if step % SAVE_EVERY == 0:
        colvar_fh.write(f"{t_ps:.4f} {calc.s:.6f} {calc.v:.8e}\n")
        colvar_fh.flush()
        intact = all(int(np.sum(atoms.get_distances(o, h_all, mic=True) < OH_CUT)) == 2
                     for o in ow)
        f = atoms.copy()
        f.info.update(cv=float(calc.s), bias_eV=float(calc.v), time_ps=t_ps,
                      step=step, intact=int(intact))
        write(frames_file, f, format="extxyz", append=True)
    if step % CHECKPOINT_EVERY == 0 and step > 0:
        save_checkpoint(step)


dyn.attach(on_step, interval=1)
atoms.get_forces()                         # sets calc.s before the first step
dyn.run(STEPS - step0)
hills_fh.close()
colvar_fh.close()
save_checkpoint(dyn.get_number_of_steps())

# ---------------- free energy where the bias went ------------------------
colvar = np.loadtxt(colvar_file, comments="#")
hills = np.loadtxt(hills_file, comments="#", ndmin=2)
lo_v, hi_v = hills[:, 1].min() - 3 * SIGMA, hills[:, 1].max() + 3 * SIGMA
EV_KJ = 96.485
trapz = getattr(np, "trapezoid", None) or np.trapz


def free_energy(n_hills):
    m = WellTemperedMetaD1D.from_hills(list(zip(hills[:n_hills, 1], hills[:n_hills, 3])), **kw)
    return m.grid, -(BIAS_FACTOR / (BIAS_FACTOR - 1.0)) * m.V


def window_dF(s, F):
    vis = (s >= lo_v) & (s <= hi_v)
    w = {n: trapz(np.exp(-(F[(s >= a) & (s < b)] - F[vis].min()) / kT), s[(s >= a) & (s < b)])
         for n, (a, b) in DF_WINDOWS.items()}
    return float(-kT * np.log(w["6 coord"] / w["5 coord"]) * EV_KJ)


s_grid, F_end = free_energy(len(hills))
vis = (s_grid >= lo_v) & (s_grid <= hi_v)
ref = (s_grid > 0.8) & (s_grid < 1.8)
F_rel = (F_end - F_end[ref].min()) * EV_KJ
dF_time = []
for k in range(1, N_TIMES + 1):
    n = int(round(len(hills) * k / N_TIMES))
    s, Fk = free_energy(n)
    dF_time.append((float(hills[n - 1, 0]), window_dF(s, Fk)))
F_points = {f"{p:.2f}": float(np.interp(p, s_grid, F_rel)) for p in F_POINTS}
edges = [DF_WINDOWS["6 coord"][0]]
state = np.digitize(colvar[:, 1], edges)           # 0: below 1.35, 1: above
crossings = int(np.sum(state[1:] != state[:-1]))

frames = read(frames_file, index=":")
per_window, n_total = {}, {}
for name, (a, b) in STATE_WINDOWS.items():
    inside = [f for f in frames if a <= f.info["cv"] < b]
    ok = [f for f in inside if f.info["intact"]]
    for f in ok:
        f.info["weight"] = 1.0
    if ok:
        write(OUTDIR / f"state_{name}.extxyz", ok, format="extxyz")
    per_window[name] = dict(window=[a, b], frames=len(inside), intact=len(ok))

summary = dict(kT_eV=kT, r0=R0, steps=STEPS, pace=PACE, sigma=SIGMA, height_kT=HEIGHT_KT,
               bias_factor=BIAS_FACTOR, seed=SEED, start=str(START),
               visited_cn_range=[float(lo_v), float(hi_v)], hills=len(hills),
               frames=len(frames), dF_6c_minus_5c_kJmol_end=dF_time[-1][1],
               dF_6c_minus_5c_vs_time_kJmol=dF_time,
               F_at_cn_kJmol_vs_5c_minimum=F_points, basin_changes_5c_6c=crossings,
               windows=per_window)
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=1))
np.savetxt(OUTDIR / "fes.dat", np.c_[s_grid, F_end], header="CN F_eV")
print(json.dumps(summary, indent=1))

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(s_grid[vis], F_rel[vis])
    for b in (0.65, 1.35, 2.10):
        ax[0].axvline(b, color="grey", ls=":", lw=0.8)
    ax[0].set(xlabel="CN(Sn-O$_w$)", ylabel="F / kJ mol$^{-1}$",
              title=f"F where the bias went ({lo_v:.2f} to {hi_v:.2f})")
    ax[1].plot(colvar[:, 0], colvar[:, 1], lw=0.4)
    ax[1].set(xlabel="time / ps", ylabel="CN")
    tt = np.array(dF_time)
    ax[2].plot(tt[:, 0], tt[:, 1], "o-")
    ax[2].set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title="6 coordinate minus 5 coordinate")
    fig.tight_layout()
    fig.savefig(OUTDIR / "metad.png", dpi=200)
except ImportError:
    pass
