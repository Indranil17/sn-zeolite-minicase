# %% How different are the metadynamics basins, and are the snapshots typical?
# Uses CRISP (github.com/Indranil17/CRISP), subsample.py
# create_repres(): a SOAP vector per frame, centred on Sn (rcut 6 A), and the
# farthest point sampling of fpsample that CRISP builds on.
# Why: GIPAW is run on a few snapshots per basin. Two things have to hold.
#   1. The basins differ structurally around Sn (otherwise the CN states are
#      not distinct states).
#   2. The snapshots are typical of their basin. They are taken evenly in time
#      (unbiased for a mean). Farthest point sampling is NOT used to choose
#      them: it favours the tails, which is right for a training set and wrong
#      for an average. Here it only measures how much structural variety a basin
#      has (how fast the farthest-point distance falls with the number of picks).
# Run from the repository root with the snmini_crisp environment, after
# analyse_metad_replicas.py:  python analysis/crisp/crisp_soap_diversity.py

from pathlib import Path
import json
import sys

import numpy as np
import fpsample
from ase.io import read

sys.path.insert(0, str(Path("external/CRISP").resolve()))   # CRISP modules (github.com/Indranil17/CRISP), not bundled here
from subsample import create_repres                                    # CRISP

# ----------------------------- CONFIG ------------------------------------
COMBINED = Path("outputs/03_combined/combined.json")
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
CORE_HALFWIDTH = 0.12
OH_CUT = 1.25
RCUT = 6.0                 # A, SOAP cutoff (CRISP default)
MAX_PER_BASIN = 150        # frames per basin used (evenly thinned) to keep SOAP cheap
OUTFILE = Path("outputs/03_combined/crisp_soap_diversity.json")
# --------------------------------------------------------------------------

comb = json.loads(COMBINED.read_text())
minima = {b["state"]: b["cn_min"] for b in comb["basins"]}
edges = comb["edges"]
start = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
h_all = [a.index for a in start if a.symbol == "H"]
ow = [a.index for a in start if a.symbol == "O" and
      int(np.sum(start.get_distances(a.index, h_all, mic=True) < OH_CUT)) == 2]

frames = {n: [] for n in minima}
for rep in REPLICAS:
    for f in sorted(rep.glob("state_*.extxyz")):
        for a in read(f, index=":"):
            k = int(np.digitize(a.info["cv"], edges[1:-1]))
            name = list(minima)[k]
            if abs(a.info["cv"] - minima[name]) > CORE_HALFWIDTH:
                continue
            if all(int(np.sum(a.get_distances(o, h_all, mic=True) < OH_CUT)) == 2 for o in ow):
                frames[name].append(a)

rep_vec, report = {}, {}
for name, fr in frames.items():
    if len(fr) < 5:
        print(f"{name}: only {len(fr)} intact frames, skipped")
        continue
    sel = np.unique(np.linspace(0, len(fr) - 1, min(MAX_PER_BASIN, len(fr))).astype(int))
    X = create_repres([fr[i] for i in sel], rcut=RCUT, ind="Sn")      # CRISP
    rep_vec[name] = X
    c = X.mean(axis=0)
    d_c = np.linalg.norm(X - c, axis=1)
    perm = fpsample.fps_sampling(X, min(20, len(X)), start_idx=0)
    fps_d = [float(np.min(np.linalg.norm(X[perm[:i]] - X[perm[i]], axis=1)))
             for i in range(1, len(perm))]
    # the evenly spaced GIPAW snapshots: where do they sit in the spread?
    pick = np.unique(np.linspace(0, len(sel) - 1, min(4, len(sel))).astype(int))
    pct = [float(np.mean(d_c < d_c[i]) * 100) for i in pick]
    report[name] = dict(frames=len(fr), used=len(sel), spread_mean_dist_to_centroid=float(d_c.mean()),
                        fps_min_dist_first_picks=fps_d[:5], fps_min_dist_last=fps_d[-1],
                        gipaw_snapshot_percentile_of_distance_to_centroid=pct)
    print(f"{name}: {len(fr)} intact frames; mean distance to centroid {d_c.mean():.3f}; "
          f"FPS distance after 2, 10, 19 picks: {fps_d[0]:.3f}, {fps_d[8]:.3f}, {fps_d[-1]:.3f}; "
          f"evenly spaced snapshots at percentiles {np.round(pct)}")

names = list(rep_vec)
print("\ndistance between basin centroids (SOAP, Sn centred) vs spread inside a basin:")
D = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        d = float(np.linalg.norm(rep_vec[a].mean(axis=0) - rep_vec[b].mean(axis=0)))
        D[f"{a} - {b}"] = d
        print(f"  {a} - {b}: {d:.3f}   (spread {a}: {report[a]['spread_mean_dist_to_centroid']:.3f}, "
              f"{b}: {report[b]['spread_mean_dist_to_centroid']:.3f})")
OUTFILE.write_text(json.dumps(dict(software="CRISP create_repres + fpsample", rcut=RCUT,
                                   basins=report, centroid_distances=D), indent=1))
print(f"wrote {OUTFILE}")
