# %% Analyse the unbiased MD (step 02) with CRISP
# CRISP (github.com/Indranil17/CRISP) is a trajectory analysis package by the
# author of this repository. Its error_analysis.py and h_bond.py are used
# unchanged (clone CRISP into external/CRISP), as an independent cross check of
# the statistics in 02_md_rdf.py:
#   * statistical error of CN(t), the potential energy and the temperature, by
#     the integrated autocorrelation time and by block averaging (CRISP);
#   * hydrogen bonds per frame (CRISP count_hydrogen_bonds), split by donor
#     and acceptor type, and for the Sn bound water on its own.
# Run from the repository root with the snmini_crisp environment:
#   python analysis/crisp/crisp_md_analysis.py

from pathlib import Path
import json
import sys

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path("external/CRISP").resolve()))   # CRISP modules (github.com/Indranil17/CRISP), not bundled here
from error_analysis import autocorrelation_analysis, block_analysis   # CRISP
from h_bond import count_hydrogen_bonds                                # CRISP

# ----------------------------- CONFIG ------------------------------------
LOG = Path("outputs/02_md/log.dat")          # time_ps CN T_K Epot_eV
FRAMES = Path("outputs/02_md/md_frames.extxyz")
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
DISCARD_PS = 10.0          # equilibration of 02 (20 000 steps at 0.5 fs)
ANGLE_CUT = 120.0          # deg, CRISP default
HB_CUT = 2.4               # A, H...acceptor, CRISP default
BOND_CUT = 1.6             # A, covalent O-H, CRISP default
OUTFILE = Path("outputs/crisp_md02.json")
# --------------------------------------------------------------------------

# ---- 1. statistical error of the time series
t, cn, temp, epot = np.loadtxt(LOG, unpack=True)
keep = t >= DISCARD_PS
report = {}
for name, x in [("CN", cn), ("T_K", temp), ("Epot_eV", epot)]:
    x = x[keep]
    acf = autocorrelation_analysis(x)
    blk = block_analysis(x, convergence_tol=1e-3)
    blk_tight = block_analysis(x, convergence_tol=1e-5)
    report[name] = {"n": len(x), "mean": float(acf["mean"]), "std": float(acf["std"]),
                    "acf_error": float(acf["acf_err"]),
                    "tau_int_frames": float(acf["tau_int"]),
                    "block_error_tol_1e-3": float(blk["block_err"]),
                    "block_error_tol_1e-5": float(blk_tight["block_err"])}
    print(f"{name:8s} mean {acf['mean']:.4f}  std {acf['std']:.4f}  "
          f"error: ACF {acf['acf_err']:.4f}, block {blk['block_err']:.4f} "
          f"(tight tol {blk_tight['block_err']:.4f})  tau_int {acf['tau_int']:.1f} frames")

# ---- 2. hydrogen bonds
start = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn = int(start.info["sn_index"])
water_o = [a.index for a in start if a.symbol == "O" and
           int(np.sum(start.get_distances(a.index, [h.index for h in start if h.symbol == "H"],
                                          mic=True) < 1.25)) == 2]
bound_o = min(water_o, key=lambda o: start.get_distance(sn, o, mic=True))
frames = read(FRAMES, index=":")
tot, wf, ww, bound_don_fw, bound_don_w, bound_acc = [], [], [], [], [], []
for a in frames:
    d, n = count_hydrogen_bonds(a, acceptor_atoms=["O"], angle_cutoff=ANGLE_CUT,
                                h_bond_cutoff=HB_CUT, bond_cutoff=BOND_CUT)
    n_wf = n_ww = bd_fw = bd_w = ba = 0
    for h, bonds in d.items():
        if not bonds:
            continue
        donor = int(bonds[0][0])
        for b in bonds[1:]:
            acc = int(b[0])
            if donor in water_o:
                n_ww += acc in water_o
                n_wf += acc not in water_o
            if donor == bound_o:
                bd_w += acc in water_o
                bd_fw += acc not in water_o
            if acc == bound_o:
                ba += 1
    tot.append(n)
    wf.append(n_wf)
    ww.append(n_ww)
    bound_don_fw.append(bd_fw)
    bound_don_w.append(bd_w)
    bound_acc.append(ba)

print(f"\nSn bound water: O index {bound_o}, {len(water_o)} waters, {len(frames)} frames")
for name, x in [("all H bonds", tot), ("water -> framework O", wf), ("water -> water", ww),
                ("bound water -> framework O", bound_don_fw),
                ("bound water -> other water", bound_don_w),
                ("bound water accepts", bound_acc)]:
    x = np.array(x, float)
    acf = autocorrelation_analysis(x)
    report["hb " + name] = dict(mean=float(x.mean()), std=float(x.std(ddof=1)),
                                acf_error=float(acf["acf_err"]),
                                tau_int_frames=float(acf["tau_int"]))
    print(f"{name:28s} per frame: {x.mean():.2f} +- {acf['acf_err']:.2f} (std {x.std(ddof=1):.2f})")

OUTFILE.write_text(json.dumps(dict(software="CRISP, github.com/Indranil17/CRISP",
                                   params=dict(angle_cut=ANGLE_CUT, hb_cut=HB_CUT,
                                               bond_cut=BOND_CUT, discard_ps=DISCARD_PS),
                                   sn_bound_water_O=bound_o, results=report), indent=1))
print(f"wrote {OUTFILE}")
