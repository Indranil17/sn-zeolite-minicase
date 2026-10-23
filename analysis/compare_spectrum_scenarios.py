# %% One figure: the weighted 119Sn spectrum for each population scenario
# The state shifts are defined (5 coordinate from 20 snapshots, 6 coordinate from 4
# snapshots of replica 1); the weighting between them is not, because replica 1
# disagrees with the other six on the free energy of the second water. This cell
# draws the weighted spectrum for each population scenario of the results log
# (docs/run_log.md, section 9): the populations come from
# docs/results/data/all_replicas_summary.json, the snapshot shifts from the other
# files in docs/results/data/. Each state is a Gaussian as wide as the spread of its
# snapshots (snmini.spectrum.weighted_spectrum), weighted by its population, with
# the snapshots as marks and the ranges of Sn-Beta resonances for orientation.
# Run from the repository root:  python analysis/compare_spectrum_scenarios.py
# (needs the snmini package: pip install -e .)

from pathlib import Path
import json

import numpy as np

from snmini.spectrum import weighted_spectrum

# ----------------------------- CONFIG ------------------------------------
DATA = Path("docs/results/data")
OUTFILE = Path("docs/results/spectrum_scenarios.png")
GRID = np.linspace(-800.0, -400.0, 2001)               # ppm
LIT = {"hydrated Sn-Beta, octahedral": (-700.0, -685.0),
       "dehydrated Sn-Beta, tetrahedral": (-446.0, -420.0)}   # resonances, orientation only
STYLES = {                                              # scenario key in the summary file: (colour, line style)
    "replica 1 alone": ("#e34948", "-"),
    "all 7, mean": ("k", "-"),
    "without replica 1, mean": ("#2a78d6", "-"),
    "without replica 1, MACE -> PBE +3.3": ("#1baf7a", "--"),
    "without replica 1, MACE -> PBE +9.7": ("#eda100", ":"),
}
# --------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scen = json.loads((DATA / "all_replicas_summary.json").read_text())["weighted_shift"]
d5 = np.loadtxt(DATA / "delta_119Sn_5c_20_snapshots.csv", delimiter=",", skiprows=2, usecols=1)
d6 = np.array(json.loads((DATA / "snapshot_shifts_4each.json").read_text())["states"]["cn1.48"]["shifts"])
shifts = {"5c": d5, "6c": d6}

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for name, (col, ls) in STYLES.items():
    p6 = scen[name]["p6"]
    y, _ = weighted_spectrum(GRID, shifts, {"5c": 1.0 - p6, "6c": p6})
    ax.plot(GRID, y, color=col, ls=ls, lw=1.6,      # legend: the numbers of the results log table
            label=f"{name}: p(6c) = {p6:.4f}, weighted {scen[name]['weighted_ppm']:.1f} ppm")
for (name, (lo, hi)), col in zip(LIT.items(), ("tab:cyan", "tab:red")):
    ax.axvspan(lo, hi, alpha=0.18, color=col, label=name)
ax.plot(d5, np.full(len(d5), -0.0010), "|", ms=9, color="tab:purple", label="5 coordinate snapshots (20)")
ax.plot(d6, np.full(len(d6), -0.0010), "|", ms=9, color="tab:brown", label="6 coordinate snapshots (4, replica 1)")
ax.set_ylim(-0.0025, None)
ax.invert_xaxis()
ax.set(xlabel=r"$\delta(^{119}$Sn) / ppm", ylabel="intensity (each scaled to its area)")
ax.legend(fontsize=6.2, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)  # below the axes,
fig.tight_layout()                                                                              # off the peaks
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTFILE, dpi=200, bbox_inches="tight")
print(f"wrote {OUTFILE}")
