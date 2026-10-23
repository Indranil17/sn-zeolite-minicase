# %% Redraw the three panels of metad.png for runs made with the first 03 script
# The first version of 03_metad.py drew the third panel (dF against time) only
# when it had found two basins, so it was empty for two of the three replicas,
# and it plotted F over the whole CN grid, most of which the bias never visited.
# This cell writes metad_v2.png next to each old metad.png (the old file is not
# touched): F only where the bias went, the CN trace, and the free energy of the
# 6 coordinate state relative to the 5 coordinate one against time, from the bias
# (solid) and from the reweighted histogram with c(t) (dashed, Tiwary and
# Parrinello 2015). Both come from check_metad_replicas.py, run that first.
# Run from the repository root:  python analysis/redraw_metad_panels.py

from pathlib import Path
import json

import numpy as np

# ----------------------------- CONFIG ------------------------------------
CHECK = Path("outputs/03_combined/check_metad_replicas.json")
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
WINDOW_EDGES = (0.65, 1.35, 2.10)         # CN windows: 5 coordinate 0.65 to 1.35, 6 coordinate 1.35 to 2.10
EV_KJ = 96.485
# --------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

check = json.loads(CHECK.read_text())
for rep in REPLICAS:
    info = check[rep.name]
    lo, hi = info["visited_CN_range"]
    s, F = np.loadtxt(rep / "fes.dat", unpack=True)
    col = np.loadtxt(rep / "COLVAR", comments="#")
    vis = (s >= lo) & (s <= hi)
    F_rel = (F - F[(s > 0.8) & (s < 1.8)].min()) * EV_KJ
    cum = info["cumulative_estimates"]
    t = [c["time_ps"] for c in cum]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(s[vis], F_rel[vis])
    for b in WINDOW_EDGES:
        ax[0].axvline(b, color="grey", ls=":", lw=0.8)
    ax[0].set(xlabel="CN(Sn-O$_w$)", ylabel="F / kJ mol$^{-1}$",
              title=f"F where the bias went ({lo:.2f} to {hi:.2f})")
    ax[1].plot(col[:, 0], col[:, 1], lw=0.4)
    ax[1].set(xlabel="time / ps", ylabel="CN")
    ax[2].plot(t, [c["bias"]["6 coord"] for c in cum], "o-", label="from the bias")
    ax[2].plot(t, [c["reweighted"]["6 coord"] for c in cum], "s--", label="reweighted frames")
    ax[2].axhline(0, color="grey", lw=0.6)
    ax[2].set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title="6 coordinate minus 5 coordinate")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(rep / "metad_v2.png", dpi=200)
    plt.close(fig)
    print(f"wrote {rep / 'metad_v2.png'}")
