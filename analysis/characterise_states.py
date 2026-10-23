# %% What do the frames of each CN window look like? (new replicas, all frames stored)
# For the 4, 5 and 6 coordinate windows of step 03 (frames.extxyz of every
# replica): number of frames, share with all waters intact, distances from Sn
# to the nearest three water oxygens, and the number of water oxygens within
# 2.8 A (a hard cutoff coordination number, independent of the switching
# function). Also checks that fes.dat equals the free energy rebuilt from HILLS
# up to a constant (the new 03 saves F without shifting its minimum to zero).
# Run from the repository root:  python analysis/characterise_states.py

from pathlib import Path
import json

import numpy as np
from ase.io import read

from snmini import structures as S

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_n*"))
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
WINDOWS = {"4c": (0.20, 0.55), "5c": (0.90, 1.30), "6c": (1.40, 1.85)}
HARD_CUT = 2.8             # A
SIGMA, GAMMA = 0.07, 10.0
OUTFILE = Path("outputs/03_combined/characterise_states.json")
# --------------------------------------------------------------------------

ref = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn, ow = int(ref.info["sn_index"]), S.water_oxygens(ref)
grid = np.linspace(-0.5, 6.5, 2001)
out = {}
pool = {w: [] for w in WINDOWS}
for rep in REPLICAS:
    frames = read(rep / "frames.extxyz", index=":")
    for name, (lo, hi) in WINDOWS.items():
        for a in frames:
            if lo <= a.info["cv"] < hi:
                d = np.sort(a.get_distances(sn, ow, mic=True))
                pool[name].append((rep.name, int(a.info["intact"]), d[:3], int(np.sum(d < HARD_CUT))))
    h = np.loadtxt(rep / "HILLS", comments="#")
    V = np.zeros_like(grid)
    for c, e in zip(h[:, 1], h[:, 3]):
        V += e * np.exp(-0.5 * ((grid - c) / SIGMA) ** 2)
    s, F = np.loadtxt(rep / "fes.dat", unpack=True)
    F_h = -(GAMMA / (GAMMA - 1)) * V
    diff = (F - F.min()) - (F_h - F_h.min())
    out[rep.name] = dict(fes_vs_HILLS_max_abs_diff_after_shift_kJmol=float(np.abs(diff).max() * 96.485))
    print(f"{rep.name}: fes.dat against HILLS after removing the constant: {out[rep.name]['fes_vs_HILLS_max_abs_diff_after_shift_kJmol']:.5f} kJ/mol")

print(f"\n{'window':6s} {'frames':>6s} {'intact':>7s} {'d1':>10s} {'d2':>10s} {'d3':>10s}   water O within {HARD_CUT} A (0 / 1 / 2 / 3+)")
for name, rows in pool.items():
    n = len(rows)
    intact = np.mean([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    k = np.array([r[3] for r in rows])
    cnt = [float(np.mean(k == 0)), float(np.mean(k == 1)), float(np.mean(k == 2)), float(np.mean(k >= 3))]
    out[name] = dict(frames=n, intact_fraction=float(intact),
                     d_mean=[float(x) for x in d.mean(axis=0)], d_std=[float(x) for x in d.std(axis=0)],
                     hard_cn_fractions_0_1_2_3plus=cnt,
                     frames_per_replica={r: int(sum(1 for x in rows if x[0] == r)) for r in sorted({x[0] for x in rows})})
    print(f"{name:6s} {n:6d} {intact:7.2f} " + " ".join(f"{d[:, i].mean():5.2f}+-{d[:, i].std():.2f}" for i in range(3))
          + "   " + " / ".join(f"{c:.2f}" for c in cnt))
OUTFILE.write_text(json.dumps(out, indent=1))
print(f"wrote {OUTFILE}")
