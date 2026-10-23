# %% 06  Population weighted 119Sn spectrum
# Joins the two halves: populations of the Sn coordination states from the
# free energy (03) and the shifts of each state from GIPAW snapshots (05).
# The experimental bands are for Sn-Beta, not Sn-CHA: compare trends (how far
# hydration moves the shift), not absolute positions.

from pathlib import Path
import json

import numpy as np

from snmini.spectrum import bootstrap_weighted_mean, weighted_spectrum

# ----------------------------- CONFIG ------------------------------------
METAD = Path("outputs/03_metad/summary.json")
GIPAW = Path("outputs/05_gipaw/results.json")
GRID = np.linspace(-800.0, -300.0, 2001)      # ppm
OUTDIR = Path("outputs/06_spectrum")
# Resonances of Sn-Beta (ppm vs SnMe4), for orientation only:
LIT = {
    "hydrated, octahedral (Bermejo-Deval 2012, PNAS and ACS Catal.; Qi 2018)": (-700.0, -685.0),
    "dehydrated, tetrahedral (Bermejo-Deval 2012, PNAS and ACS Catal.; Qi 2018)": (-446.0, -420.0),
}
# --------------------------------------------------------------------------

pops = json.loads(METAD.read_text())["populations"]
states = json.loads(GIPAW.read_text())["states"]
shifts = {k: np.array(v["shifts"]) for k, v in states.items()}
missing = [k for k in pops if k not in shifts]
if missing:
    print("no GIPAW data yet for:", missing, "(left out, populations renormalised)")
y, mean = weighted_spectrum(GRID, shifts, pops)
lo, hi = bootstrap_weighted_mean(shifts, pops)
print(f"population weighted <delta(119Sn)> = {mean:.1f} ppm "
      f"(95% CI {lo:.1f} to {hi:.1f})")
for k in shifts:
    print(f"  {k:>10s}: p = {pops.get(k, 0):.3f}, <delta> = "
          f"{shifts[k].mean():.1f} ppm")

OUTDIR.mkdir(parents=True, exist_ok=True)
np.savetxt(OUTDIR / "spectrum.dat", np.c_[GRID, y], header="delta_ppm I")
(OUTDIR / "summary.json").write_text(json.dumps(
    dict(weighted_mean=mean, ci95=[lo, hi], populations=pops,
         state_means={k: float(v.mean()) for k, v in shifts.items()}),
    indent=1))
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for (name, (a, b)), c in zip(LIT.items(), ["#cfe3f5", "#f5dccf"]):
        ax.axvspan(a, b, color=c, label=name)
    ax.plot(GRID, y / y.max(), c="k", label="this model, weighted")
    for k, v in shifts.items():
        ax.plot(v, np.full(len(v), -0.05), "|", ms=10, label=f"{k} snapshots")
    ax.invert_xaxis()
    ax.set(xlabel="δ($^{119}$Sn) / ppm", ylabel="intensity (norm.)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "spectrum.png", dpi=200)
except ImportError:
    pass
