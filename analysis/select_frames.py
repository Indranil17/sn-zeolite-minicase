# %% Pick frames of one CN window from the stored metadynamics frames
# Takes every stored frame whose CN lies in WINDOW, keeps those with all six
# waters intact (O-H distances below OH_CUT for exactly two H per water), and
# picks N of them evenly spaced through the list (replicas in order, time within
# a replica), so that each replica contributes in proportion and no frame is
# preferred by its structure. Evenly spaced picks keep an average unbiased;
# farthest point sampling would favour the tails and is not used for averages.
# Writes one extxyz file for step 04 (GIPAW) or for the PBE force check.
# Run from the repository root:  python analysis/select_frames.py

from pathlib import Path
import json

import numpy as np
from ase.io import read, write

# ----------------------------- CONFIG ------------------------------------
SOURCES = sorted(Path("outputs").glob("03_metad_r[0-9]*/state_*.extxyz"))
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
WINDOW = (0.90, 1.30)      # CN: 5 coordinate; 6 coordinate is (1.40, 1.85)
N = 20
OH_CUT = 1.25              # A
OUTFILE = Path("outputs/05_selected/state_5c_n20.extxyz")
# --------------------------------------------------------------------------

start = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
h_all = [a.index for a in start if a.symbol == "H"]
ow = [a.index for a in start if a.symbol == "O" and
      int(np.sum(start.get_distances(a.index, h_all, mic=True) < OH_CUT)) == 2]

pool, seen = [], set()
for f in SOURCES:
    for a in read(f, index=":"):
        key = (f.parent.name, round(float(a.info["time_ps"]), 4))
        if key in seen or not (WINDOW[0] <= a.info["cv"] < WINDOW[1]):
            continue
        seen.add(key)
        intact = all(int(np.sum(a.get_distances(o, h_all, mic=True) < OH_CUT)) == 2 for o in ow)
        if intact:
            a.info["replica"] = f.parent.name
            pool.append(a)
pool.sort(key=lambda a: (a.info["replica"], a.info["time_ps"]))
if len(pool) < N:
    raise SystemExit(f"only {len(pool)} intact frames in the window, need {N}")
idx = np.unique(np.round(np.linspace(0, len(pool) - 1, N + 2)[1:-1]).astype(int))
sel = [pool[i] for i in idx]
for a in sel:
    a.info["weight"] = 1.0
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
write(OUTFILE, sel, format="extxyz")
per_rep = {r: sum(a.info["replica"] == r for a in sel) for r in sorted({a.info["replica"] for a in sel})}
print(f"{len(pool)} intact frames in CN {WINDOW}; wrote {len(sel)} to {OUTFILE}; per replica {per_rep}; "
      f"CN of picks {min(a.info['cv'] for a in sel):.2f} to {max(a.info['cv'] for a in sel):.2f}")
