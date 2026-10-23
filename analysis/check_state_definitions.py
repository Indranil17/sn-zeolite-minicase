# %% Does dF(6 coordinate - 5 coordinate) depend on how the two states are defined?
# The CN windows used so far (5 coordinate 0.65 to 1.35, 6 coordinate 1.35 to 2.10)
# are a choice. Two checks that could break the +14 kJ/mol:
#  1. move the window edges (the boundary between the states and the upper edge);
#  2. define the states by geometry instead of by CN: the number of water oxygens
#     within HARD_CUT of Sn (1 water = 5 coordinate, 2 waters = 6 coordinate).
# Frames are reweighted with the time dependent offset c(t) (Tiwary and
# Parrinello 2015), second half of each run. Uses the four replicas that stored
# every frame (frames.extxyz). Run from the repository root:
#   python analysis/check_state_definitions.py

from pathlib import Path
import json

import numpy as np
from ase.io import read

from snmini import structures as S
from snmini.metad import KB_EV

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_n*"))
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
TEMPERATURE = 350.0
SIGMA, GAMMA = 0.07, 10.0
T_FROM = 112.5                              # ps, second half
HARD_CUT = 2.8                              # A
BOUNDARIES = [1.25, 1.30, 1.35, 1.40, 1.45]  # CN between 5 and 6 coordinate
UPPER = [1.9, 2.1, 2.3]                     # upper CN edge of the 6 coordinate window
LOWER = 0.65                                # lower CN edge of the 5 coordinate window
GRID = np.linspace(-0.5, 6.5, 2001)
OUTFILE = Path("outputs/03_combined/check_state_definitions.json")
# --------------------------------------------------------------------------

kT = KB_EV * TEMPERATURE
EV_KJ = 96.485
trapz = getattr(np, "trapezoid", None) or np.trapz
ref = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
sn, ow = int(ref.info["sn_index"]), S.water_oxygens(ref)

out = {}
for rep in REPLICAS:
    h = np.loadtxt(rep / "HILLS", comments="#")
    fr = read(rep / "frames.extxyz", index=":")
    t = np.array([a.info["time_ps"] for a in fr])
    cn = np.array([a.info["cv"] for a in fr])
    vb = np.array([a.info["bias_eV"] for a in fr])
    hard = np.array([int(np.sum(a.get_distances(sn, ow, mic=True) < HARD_CUT)) for a in fr])
    lo_v, hi_v = h[:, 1].min() - 3 * SIGMA, h[:, 1].max() + 3 * SIGMA
    win = (GRID >= lo_v) & (GRID <= hi_v)
    V = np.zeros_like(GRID)
    k = 0
    c = []
    for tt in t:
        while k < len(h) and h[k, 0] < tt - 1e-9:
            V += h[k, 3] * np.exp(-0.5 * ((GRID - h[k, 1]) / SIGMA) ** 2)
            k += 1
        num = trapz(np.exp(GAMMA / (GAMMA - 1) * V[win] / kT), GRID[win])
        den = trapz(np.exp(V[win] / ((GAMMA - 1) * kT)), GRID[win])
        c.append(kT * np.log(num / den))
    logw = (vb - np.array(c)) / kT
    m = t >= T_FROM
    w = np.exp(logw[m] - logw[m].max())
    cm, hm = cn[m], hard[m]

    def dF(mask6, mask5):
        return float(-kT * np.log(w[mask6].sum() / w[mask5].sum()) * EV_KJ)

    r = {"cn_windows": {}}
    for b in BOUNDARIES:
        for u in UPPER:
            r["cn_windows"][f"boundary {b:.2f}, upper {u:.1f}"] = dF((cm >= b) & (cm < u), (cm >= LOWER) & (cm < b))
    r["geometric_2_vs_1_water_within_2.8A"] = dF(hm == 2, hm == 1)
    r["geometric_2plus_vs_1_water"] = dF(hm >= 2, hm == 1)
    r["geometric_fraction_of_weight_0_1_2_3plus"] = [float(w[hm == i].sum() / w.sum()) if i < 3 else float(w[hm >= 3].sum() / w.sum())
                                                     for i in range(4)]
    out[rep.name] = r
    vals = list(r["cn_windows"].values())
    print(f"{rep.name}: CN window variants {min(vals):.1f} to {max(vals):.1f} kJ/mol (default 1.35 / 2.1: "
          f"{r['cn_windows']['boundary 1.35, upper 2.1']:.1f}); geometric 2 vs 1 water {r['geometric_2_vs_1_water_within_2.8A']:.1f}, "
          f"2+ vs 1 {r['geometric_2plus_vs_1_water']:.1f}; weight on 0/1/2/3+ waters "
          f"{np.round(r['geometric_fraction_of_weight_0_1_2_3plus'], 3)}")
allv = [v for r in out.values() for v in r["cn_windows"].values()]
geo = [r["geometric_2_vs_1_water_within_2.8A"] for r in out.values()]
print(f"all window variants of all replicas: {min(allv):.1f} to {max(allv):.1f}; geometric definition: mean {np.mean(geo):.1f}, range {min(geo):.1f} to {max(geo):.1f} kJ/mol")
OUTFILE.write_text(json.dumps(out, indent=1))
