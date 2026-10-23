# %% MACE-MP-0 against PBE on frames of the metadynamics runs: energies and forces
# Why: the 3.4 kJ/mol agreement of the first DFT check was for six relative
# energies at relaxed geometries, with no forces. Free energy sampling visits
# thermal, biased configurations, so the model is checked on those: PBE single
# points (Quantum ESPRESSO pw.x, forces on) at frames of each CN window, and MACE
# without D3 at the same frames (MACE-MP-0 learned PBE; D3 is an add on that PBE
# does not have here). Thresholds for the verdict (force RMSE under 50
# meV/A excellent, under 100 acceptable, over 200 flag; energy RMSE under 2, 5,
# 10 meV/atom for the spread inside a state).
# Three phases, set PHASE, run from the repository root:
#   "write"    inputs under OUTDIR/batch_XX/, five frames per batch, one job.sh each
#   "mace"     MACE energies and forces at the same frames (CPU is enough)
#   "compare"  read pw.x output, print the table, write summary.json
# Needs the frame files from analysis/select_frames.py.

from pathlib import Path
import json
import math

import numpy as np
from ase.io import read, write

# ----------------------------- CONFIG ------------------------------------
PHASE = "write"            # "write", "mace" or "compare"
FRAME_FILES = {"5c": Path("outputs/05_selected/state_5c_n30.extxyz"),
               "6c": Path("outputs/05_selected/state_6c_n30.extxyz")}
OUTDIR = Path("outputs/05_force_check")
BATCH = 5                  # frames per pw.x job
QE_PSEUDO_DIR = str(Path("pseudo_src").resolve())
QE_PSEUDOS = {"Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF", "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
              "O": "O.pbe-n-kjpaw_psl.0.1.UPF", "H": "H.pbe-kjpaw_psl.1.0.0.UPF"}
QE_ECUTWFC, QE_ECUTRHO = 60, 480      # Ry, as the first DFT check
K_DENSITY = 12.0           # 2x2x2 for the 9.3 A cell (0.17 ppm from 3x3x3 for shifts)
CALC, MODEL, DEVICE, DTYPE = "mace", "medium", "cpu", "float64"
EV_KJ = 96.485
# --------------------------------------------------------------------------

frames = []                                  # (state, index, atoms)
for state, f in FRAME_FILES.items():
    for k, a in enumerate(read(f, index=":")):
        frames.append((state, k, a))
batches = [frames[i:i + BATCH] for i in range(0, len(frames), BATCH)]


def folder(b, state, k):
    return OUTDIR / f"batch_{b:02d}" / f"{state}_{k:02d}"


if PHASE == "write":
    from snmini import qe
    for b, batch in enumerate(batches):
        for state, k, a in batch:
            d = folder(b, state, k)
            if (d / "scf.out").exists():             # never touch a started run
                continue
            d.mkdir(parents=True, exist_ok=True)
            n = [int(max(1, math.ceil(K_DENSITY / L))) for L in a.cell.lengths()]
            used = {el: QE_PSEUDOS[el] for el in sorted(set(a.get_chemical_symbols()))}
            qe.write_pw_scf(a, d, f"{state}{k:02d}", QE_PSEUDO_DIR, used, QE_ECUTWFC, QE_ECUTRHO, n)
            write(d / "geometry.extxyz", a, format="extxyz")
        (OUTDIR / f"batch_{b:02d}" / "job.sh").write_text(
            "#!/bin/bash\nNP=${PBS_NCPUS:-1}\nfor d in */; do\n  ( cd $d && "
            "mpirun -np $NP pw.x -nd 1 -in scf.in > scf.out ) || exit 1\ndone\n")
    print(f"{len(frames)} frames in {len(batches)} batches under {OUTDIR}")

elif PHASE == "mace":
    from snmini.calc import get_calculator
    calc = get_calculator(CALC, MODEL, False, DEVICE, DTYPE)     # no D3
    E, F = [], []
    for state, k, a in frames:
        a = a.copy()
        a.calc = calc
        E.append(float(a.get_potential_energy()))
        F.append(a.get_forces())
    np.savez(OUTDIR / "mace_noD3.npz", E=np.array(E), F=np.array(F),
             state=np.array([s for s, _, _ in frames]), k=np.array([k for _, k, _ in frames]))
    print(f"MACE energies and forces for {len(frames)} frames written")

elif PHASE == "compare":
    m = np.load(OUTDIR / "mace_noD3.npz")
    Ep, Fp, Em, Fm, st = [], [], [], [], []
    for i, (state, k, a) in enumerate(frames):
        b = i // BATCH
        out = folder(b, state, k) / "scf.out"
        if not out.exists():
            raise SystemExit(f"missing {out}")
        r = read(out, format="espresso-out")
        Ep.append(r.get_potential_energy())
        Fp.append(r.get_forces())
        Em.append(m["E"][i])
        Fm.append(m["F"][i])
        st.append(state)
    Ep, Em, st = np.array(Ep), np.array(Em), np.array(st)
    Fp, Fm = np.array(Fp), np.array(Fm)
    syms = np.array(frames[0][2].get_chemical_symbols())
    n_at = len(syms)
    res = {}
    print(f"{'subset':10s} {'frames':>6s} {'F RMSE':>10s} {'F MAE':>9s} {'F max':>9s} {'E spread RMSE':>14s}   verdict")
    for name, mask in [("all", np.ones(len(st), bool))] + [(s, st == s) for s in FRAME_FILES]:
        d = (Fm[mask] - Fp[mask]) * 1000                              # meV/A
        dE = (Em[mask] - Em[mask].mean()) - (Ep[mask] - Ep[mask].mean())   # eV, mean removed
        f_rmse = float(np.sqrt(np.mean(d ** 2)))
        e_rmse = float(np.sqrt(np.mean(dE ** 2)) * 1000 / n_at)       # meV/atom
        verdict = ("PASS" if f_rmse < 100 and e_rmse < 5 else
                   "WARN" if f_rmse < 200 and e_rmse < 10 else "FAIL")
        res[name] = dict(frames=int(mask.sum()), force_rmse_meV_A=f_rmse,
                         force_mae_meV_A=float(np.mean(np.abs(d))), force_max_meV_A=float(np.max(np.abs(d))),
                         energy_spread_rmse_meV_atom=e_rmse, verdict=verdict,
                         force_rmse_by_element={el: float(np.sqrt(np.mean(d[:, syms == el] ** 2)))
                                                for el in sorted(set(syms))})
        print(f"{name:10s} {mask.sum():6d} {f_rmse:10.1f} {res[name]['force_mae_meV_A']:9.1f} "
              f"{res[name]['force_max_meV_A']:9.1f} {e_rmse:14.2f}   {verdict}")
    print("force RMSE by element (all frames):", {k: round(v, 1) for k, v in res["all"]["force_rmse_by_element"].items()})
    if len(set(st)) == 2:
        s1, s2 = sorted(set(st))
        dm = (Em[st == s2].mean() - Em[st == s1].mean()) * EV_KJ
        dp = (Ep[st == s2].mean() - Ep[st == s1].mean()) * EV_KJ
        res["mean_energy_difference_kJmol"] = {f"{s2} minus {s1}": dict(mace=float(dm), pbe=float(dp))}
        print(f"mean energy {s2} minus {s1}: MACE {dm:.1f}, PBE {dp:.1f} kJ/mol (ensemble averages, not a free energy)")
    (OUTDIR / "summary.json").write_text(json.dumps(res, indent=1))
else:
    raise ValueError(f"unknown PHASE {PHASE!r}")
