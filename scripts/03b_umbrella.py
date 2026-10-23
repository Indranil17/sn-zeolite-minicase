# %% 03b  One umbrella sampling window on the Sn-O(water) coordination number
# An independent estimate of F(CN) to compare with the metadynamics: the same
# variable and potential, a harmonic restraint around CENTRE, many windows, then
# MBAR (03c_umbrella_mbar.py). One window per run (set CENTRE and OUTDIR, one job
# per window). The start structure is the stored metadynamics frame whose CN is
# closest to CENTRE (frames with all waters intact preferred), so windows started
# from different replicas or from different coordination states test whether the
# result depends on where the run began.
# Output in OUTDIR: COLVAR (time_ps, CN, every SAVE_EVERY steps, flushed as it
# goes), frames.extxyz (every FRAME_EVERY steps), summary.json.

from pathlib import Path
import json

import numpy as np
from ase.io import read, write

from snmini import structures as S
from snmini.calc import get_calculator, langevin, thermalize
from snmini.cv import CoordinationCV
from snmini.metad import KB_EV, BiasedCalculator
from snmini.umbrella import HarmonicBias

# ----------------------------- CONFIG ------------------------------------
START = Path("outputs/03_metad_n5a/frames.extxyz")   # frames with cv (and intact) in the info
REFERENCE = Path("outputs/01_screen/relaxed.extxyz")  # where the water oxygens are defined
LABEL = "md_cell"
CALC, MODEL, DISPERSION, DEVICE, DTYPE = "mace", "medium", True, "cuda", \
    "float32"
CENTRE = 1.10              # CN at the window centre
K_FORCE = 2.0              # eV per CN^2 (width about sqrt(kT / k) = 0.12 CN before the natural spread)
R0 = 2.6                   # A, as in 03
TEMPERATURE = 350.0        # K
TIMESTEP_FS = 0.5
STEPS = 100_000            # 50 ps in total, the first 10 ps are dropped in the analysis
SAVE_EVERY = 20            # steps between CN samples (10 fs)
FRAME_EVERY = 1_000        # steps between stored frames
SEED = 1
OUTDIR = Path("outputs/03b_umb_n5a/w_1.10")
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
if (OUTDIR / "COLVAR").exists():
    raise SystemExit(f"{OUTDIR / 'COLVAR'} exists: use a new OUTDIR")
ref = [a for a in read(REFERENCE, index=":") if a.info.get("label") == LABEL][0]
sn, ow = int(ref.info["sn_index"]), S.water_oxygens(ref)
h_all = [a.index for a in ref if a.symbol == "H"]
frames = read(START, index=":")
ok = [f for f in frames if f.info.get("intact", 1)]
pool = ok if ok else frames
atoms = min(pool, key=lambda f: abs(f.info["cv"] - CENTRE)).copy()
start_cn = float(atoms.info["cv"])
atoms.info = {k: v for k, v in atoms.info.items() if k in ("label", "sn_index")}
print(f"start: frame with CN {start_cn:.3f} for a window at {CENTRE:.2f}, "
      f"{len(ok)} of {len(frames)} frames intact")

kT = KB_EV * TEMPERATURE
cv = CoordinationCV(sn, ow, r0=R0)
bias = HarmonicBias(CENTRE, K_FORCE)
base = get_calculator(CALC, MODEL, DISPERSION, DEVICE, DTYPE)
atoms.calc = BiasedCalculator(base, cv, bias)
rng = np.random.default_rng(SEED)
thermalize(atoms, TEMPERATURE, rng)
dyn = langevin(atoms, TIMESTEP_FS, TEMPERATURE, rng)
colvar = open(OUTDIR / "COLVAR", "w")
colvar.write("# time_ps CN\n")
cn_log = []


def on_step():
    step = dyn.get_number_of_steps()
    s = atoms.calc.s
    if step % SAVE_EVERY == 0:
        colvar.write(f"{step * TIMESTEP_FS / 1000.0:.4f} {s:.6f}\n")
        colvar.flush()
        cn_log.append(s)
    if step % FRAME_EVERY == 0:
        intact = all(int(np.sum(atoms.get_distances(o, h_all, mic=True) < 1.25)) == 2 for o in ow)
        f = atoms.copy()
        f.info.update(cv=float(s), step=step, intact=int(intact))
        write(OUTDIR / "frames.extxyz", f, format="extxyz", append=True)


dyn.attach(on_step, interval=1)
atoms.get_forces()                         # sets calc.s before the first step
dyn.run(STEPS)
colvar.close()
x = np.array(cn_log[len(cn_log) // 5:])    # drop the first fifth for the quick summary
summary = dict(centre=CENTRE, k_force_eV_per_CN2=K_FORCE, temperature=TEMPERATURE, steps=STEPS,
               start_file=str(START), start_cn=start_cn, seed=SEED,
               mean_cn=float(x.mean()), std_cn=float(x.std(ddof=1)), samples=len(cn_log))
(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
