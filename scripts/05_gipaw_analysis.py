# %% 05  Read GIPAW results, reference 119Sn shifts to SnO2, average per state
# Reads gipaw.x output (Quantum ESPRESSO, the default), CASTEP .magres or a
# VASP OUTCAR, depending on the code recorded in the manifest written by 04.
# Also counts how many labels of each nucleus one calculation yields: the
# reason 119Sn is the data poor nucleus for any machine learned shift model.

from pathlib import Path
from collections import Counter
import json
import warnings

import numpy as np
from ase.geometry import get_distances
from ase.io import read

from snmini import nmr, qe
from snmini.stats import bootstrap_ci

# ----------------------------- CONFIG ------------------------------------
MANIFEST = Path("outputs/04_gipaw/manifest.json")
DELTA_REF = nmr.SNO2_DELTA_PPM      # SnO2, -603.8 ppm vs SnMe4
OUTDIR = Path("outputs/05_gipaw")
# --------------------------------------------------------------------------

warnings.filterwarnings("ignore", message=".*CASTEP.*")


def sn_shielding(folder, code, sn_index=None):
    """Isotropic shielding (and tensor if available) of the Sn atom."""
    folder = Path(folder)
    if code == "qe":
        out = folder / "nmr.out"
        if not out.exists():
            return None
        try:
            recs = qe.read_gipaw_out(out)
        except ValueError as err:              # unfinished or crashed run
            print(err)
            return None
        labels = Counter(r["element"] for r in recs)
        if sn_index is None:                   # reference cell
            return dict(iso=float(np.mean([r["iso"] for r in recs
                                           if r["element"] == "Sn"])),
                        tensor=None, labels=labels)
        rec = [r for r in recs if r["atom"] == sn_index]
        if not rec or rec[0]["element"] != "Sn":
            raise ValueError(f"atom {sn_index} in {folder} is not Sn")
        return dict(iso=rec[0]["iso"], tensor=rec[0]["tensor"], labels=labels)
    if code == "castep":
        files = list(folder.glob("*.magres"))
        if not files:
            return None
        recs = nmr.read_magres(files[0])
        mag = read(files[0], format="magres")
        if sn_index is None:                       # reference cell
            tens = [r["tensor"] for r in recs if r["element"] == "Sn"]
            return dict(iso=float(np.mean([nmr.iso(t) for t in tens])),
                        tensor=None, labels=Counter(
                            r["element"] for r in recs))
        snap = read(folder / "snapshot.extxyz")
        _, d = get_distances(snap.positions[[sn_index]], mag.positions,
                             cell=snap.cell, pbc=True)
        j = int(np.argmin(d[0]))                   # match by position
        if recs[j]["element"] != "Sn":
            raise ValueError(f"atom matched in {folder} is not Sn")
        return dict(iso=nmr.iso(recs[j]["tensor"]), tensor=recs[j]["tensor"],
                    labels=Counter(r["element"] for r in recs))
    outcar = folder / "OUTCAR"
    if not outcar.exists():
        return None
    order = json.loads((folder / "order.json").read_text()) \
        if (folder / "order.json").exists() else None
    rows = nmr.read_vasp_outcar(outcar)
    snap = read(folder / "POSCAR")
    syms = snap.get_chemical_symbols()
    if sn_index is None:
        return dict(iso=float(np.mean([r["iso"] for r in rows
                                       if syms[r["atom"]] == "Sn"])),
                    tensor=None, labels=Counter(syms))
    j = order.index(sn_index)
    return dict(iso=rows[j]["iso"], tensor=None, labels=Counter(syms))


man = json.loads(MANIFEST.read_text())
code = man["code"]
ref = [c for c in man["calculations"] if c["state"] == "reference"][0]
r = sn_shielding(ref["folder"], code)
if r is None:
    raise SystemExit("reference SnO2 not finished yet")
sigma_ref = r["iso"]
print(f"sigma_iso(SnO2) = {sigma_ref:.2f} ppm -> delta_ref {DELTA_REF} ppm")

per_state, budget = {}, Counter()
for c in man["calculations"]:
    if c["state"] == "reference":
        continue
    res = sn_shielding(c["folder"], code, c["sn_index"])
    if res is None:
        print("missing:", c["folder"])
        continue
    delta = float(nmr.shift_from_shielding(res["iso"], sigma_ref, DELTA_REF))
    entry = dict(folder=c["folder"], delta_iso=delta, weight=c["weight"],
                 cv=c["cv"])
    if res["tensor"] is not None:
        di, span, skew = nmr.tensor_summary(res["tensor"], sigma_ref,
                                            DELTA_REF)
        entry.update(span=span, skew=skew)
    per_state.setdefault(c["state"], []).append(entry)
    budget.update(res["labels"])

summary = {}
for state, rows in per_state.items():
    d = np.array([x["delta_iso"] for x in rows])
    w = np.array([x["weight"] for x in rows])
    w = w / w.sum()
    lo, hi = bootstrap_ci(d)
    summary[state] = dict(n=len(d), mean=float(d.mean()),
                          weighted_mean=float(np.sum(w * d)),
                          std=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                          ci95=[lo, hi],
                          shifts=d.tolist())
    print(f"{state:>10s}: n={len(d):3d}  <delta> = {d.mean():8.1f} ppm "
          f"(95% CI {lo:.1f} to {hi:.1f}), spread {summary[state]['std']:.1f}")

n_calc = sum(len(v) for v in per_state.values())
print("\nlabels per calculation (why 119Sn is data poor):")
for el, n in sorted(budget.items()):
    print(f"  {el:>2s}: {n / max(n_calc, 1):6.1f}")
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "results.json").write_text(json.dumps(
    dict(sigma_ref=sigma_ref, delta_ref=DELTA_REF, states=summary,
         per_snapshot=per_state,
         labels_per_calc={k: v / max(n_calc, 1) for k, v in budget.items()}),
    indent=1))
