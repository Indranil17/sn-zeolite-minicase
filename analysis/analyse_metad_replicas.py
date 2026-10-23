# %% Combine independent metadynamics replicas of step 03
# For the first three replicas (r1 to r3), made with the first version of 03.
# Reads outputs/03_metad_r*/ (fes.dat, HILLS, COLVAR, state_*.extxyz) and writes
# outputs/03_combined/: mean free energy F(CN) with the spread over replicas,
# basins and populations per replica (mean and spread), a convergence table,
# a figure, and pooled snapshot files for the GIPAW step in which every frame
# has all water molecules intact. Run from the repository root:
#   python analysis/analyse_metad_replicas.py
# What is deliberately different from the per replica summary of 03:
#  * populations are integrated only over the CN range that the bias visited
#    (HILLS centres +- 3 sigma); outside it the free energy from the bias is a
#    flat extrapolation, and a basin that reaches the grid edge would pick up
#    weight from CN values that were never sampled;
#  * snapshots are pooled over replicas and filtered for intact waters.

from pathlib import Path
import json

import numpy as np
from ase.io import read, write

from snmini import structures as S
from snmini.metad import KB_EV

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
START = Path("outputs/01_screen/relaxed.extxyz")
LABEL = "md_cell"
TEMPERATURE = 350.0        # K, as in 03
SIGMA = 0.07               # as in 03
SMOOTH = 9                 # grid points of the moving average used to find minima
MAX_BASIN_KT = 8.0         # basins with F above the lowest minimum by more are dropped
CORE_HALFWIDTH = 0.12      # CN window of a state snapshot (as in 03)
N_PER_STATE = 4            # snapshots per state written for GIPAW (pooled)
OH_CUT = 1.25              # A; an O-H distance above this counts as broken
OUTDIR = Path("outputs/03_combined")
# --------------------------------------------------------------------------

kT = KB_EV * TEMPERATURE
EV_TO_KJ = 96.485
OUTDIR.mkdir(parents=True, exist_ok=True)
start = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
ow = S.water_oxygens(start, oh_cut=OH_CUT)
h_all = [a.index for a in start if a.symbol == "H"]


def visited_range(rep):
    hills = np.loadtxt(rep / "HILLS", comments="#")
    return hills[:, 1].min() - 3 * SIGMA, hills[:, 1].max() + 3 * SIGMA


def find_basins(s, F, lo, hi):
    """Minima of the smoothed F inside [lo, hi]; borders at the maxima between."""
    Fs = np.convolve(F, np.ones(SMOOTH) / SMOOTH, mode="same")
    idx = np.where((s > lo) & (s < hi))[0][SMOOTH:-SMOOTH]
    mins = [i for i in idx[1:-1] if Fs[i] < Fs[i - 1] and Fs[i] <= Fs[i + 1]]
    mins = [i for i in mins if Fs[i] - Fs[idx].min() < MAX_BASIN_KT * kT]
    merged = []
    for i in mins:                                  # merge minima closer than 2 sigma
        if merged and s[i] - s[merged[-1]] < 2 * SIGMA:
            if Fs[i] < Fs[merged[-1]]:
                merged[-1] = i
            continue
        merged.append(i)
    edges = [lo]
    for a, b in zip(merged[:-1], merged[1:]):
        edges.append(s[a + int(np.argmax(Fs[a:b]))])
    edges.append(hi)
    return merged, edges


def populations(s, F, merged, edges):
    w = np.exp(-(F - F.min()) / kT)
    out = []
    for k in range(len(merged)):
        m = (s >= edges[k]) & (s < edges[k + 1])
        out.append(float(w[m].sum()))
    tot = sum(out)
    return [p / tot for p in out]


data = []
for rep in REPLICAS:
    s, F = np.loadtxt(rep / "fes.dat", unpack=True)
    lo, hi = visited_range(rep)
    summary = json.loads((rep / "summary.json").read_text())
    data.append(dict(rep=rep, s=s, F=F, lo=lo, hi=hi, summary=summary))
    print(f"{rep.name}: visited CN {lo:.2f} to {hi:.2f}; basins from 03: "
          f"{ {k: round(v, 2) for k, v in summary['minima'].items()} }")

# ---- consensus free energy: mean over replicas, each shifted to its minimum
common = np.array(data[0]["s"])
lo = max(d["lo"] for d in data)
hi = min(d["hi"] for d in data)
Fall = np.array([d["F"] - d["F"][(d["s"] > lo) & (d["s"] < hi)].min() for d in data])
Fmean, Fstd = Fall.mean(axis=0), Fall.std(axis=0, ddof=1) if len(data) > 1 else 0 * Fall[0]
merged, edges = find_basins(common, Fmean, lo, hi)
names = [f"cn{common[i]:.2f}" for i in merged]
print(f"\nconsensus window CN {lo:.2f} to {hi:.2f}; basins at CN {[round(common[i], 2) for i in merged]}")

rows = []
pops = []
for d, Fi in zip(data, Fall):
    p = populations(common, np.where((common >= lo) & (common <= hi), Fi, np.inf),
                    merged, edges)
    pops.append(p)
pops = np.array(pops)
for k, n in enumerate(names):
    rows.append(dict(state=n, cn_min=float(common[merged[k]]),
                     F_kJmol=float(Fmean[merged[k]] * EV_TO_KJ),
                     F_spread_kJmol=float(Fstd[merged[k]] * EV_TO_KJ),
                     population_mean=float(pops[:, k].mean()),
                     population_std=float(pops[:, k].std(ddof=1)) if len(data) > 1 else None,
                     population_per_replica=[float(x) for x in pops[:, k]]))
    print(f"{n}: F = {rows[-1]['F_kJmol']:.1f} +- {rows[-1]['F_spread_kJmol']:.1f} kJ/mol, "
          f"population {pops[:, k].mean():.3f} +- "
          f"{(pops[:, k].std(ddof=1) if len(data) > 1 else float('nan')):.3f}  per replica {np.round(pops[:, k], 3)}")

# ---- convergence: dF between the two most populated basins, per replica, vs time
conv = {}
order = np.argsort(-pops.mean(axis=0))[:2]
if len(order) == 2:
    a, b = order
    for d in data:
        c = d["summary"].get("convergence_dF_kJmol", [])
        conv[d["rep"].name] = c
    print(f"\nconvergence of dF({names[b]} - {names[a]}) from 03, per replica (time ps, dF kJ/mol): see convergence.json")

# ---- crossings: how often each replica moves between basins (from COLVAR)
for d in data:
    c = np.loadtxt(d["rep"] / "COLVAR", comments="#")
    state = np.digitize(c[:, 1], edges[1:-1])
    n_cross = int(np.sum(state[1:] != state[:-1]))
    d["crossings"] = n_cross
    visits = {names[k]: int(np.sum(state == k)) for k in range(len(names))}
    print(f"{d['rep'].name}: {n_cross} basin changes in {len(c)} stored frames; frames per basin {visits}")

# ---- pooled snapshots with intact waters
pool = {n: [] for n in names}
n_bad = {n: 0 for n in names}
for d in data:
    for f in sorted(d["rep"].glob("state_*.extxyz")):
        for a in read(f, index=":"):
            k = int(np.digitize(a.info["cv"], edges[1:-1]))
            if abs(a.info["cv"] - common[merged[k]]) > CORE_HALFWIDTH:
                continue
            intact = all(int(np.sum(a.get_distances(o, h_all, mic=True) < OH_CUT)) == 2
                          for o in ow)
            if intact:
                a.info["replica"] = d["rep"].name
                pool[names[k]].append(a)
            else:
                n_bad[names[k]] += 1
manifest = {}
for n in names:
    fr = pool[n]
    pick = np.unique(np.linspace(0, len(fr) - 1, min(N_PER_STATE, len(fr))).astype(int)) if fr else []
    sel = [fr[i] for i in pick]
    for a in sel:
        a.info["weight"] = 1.0
    if sel:
        write(OUTDIR / f"state_{n}.extxyz", sel, format="extxyz")
    manifest[n] = dict(intact_frames=len(fr), frames_with_broken_water=n_bad[n],
                       written=len(sel), replicas=sorted({a.info["replica"] for a in sel}))
    print(f"{n}: {len(fr)} intact frames in the window, {n_bad[n]} with a broken water, {len(sel)} written")

(OUTDIR / "combined.json").write_text(json.dumps(dict(
    window=[float(lo), float(hi)], temperature=TEMPERATURE, basins=rows,
    edges=[float(e) for e in edges], crossings={d["rep"].name: d["crossings"] for d in data},
    snapshots=manifest), indent=1))
(OUTDIR / "convergence.json").write_text(json.dumps(conv, indent=1))
np.savetxt(OUTDIR / "fes_mean.dat", np.c_[common, Fmean * EV_TO_KJ, Fstd * EV_TO_KJ],
           header="CN F_kJmol std_over_replicas_kJmol")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    m = (common >= lo) & (common <= hi)
    for d, Fi in zip(data, Fall):
        ax[0].plot(common[m], Fi[m] * EV_TO_KJ, lw=0.8, alpha=0.6, label=d["rep"].name[-2:])
    ax[0].plot(common[m], Fmean[m] * EV_TO_KJ, "k", lw=1.8, label="mean")
    ax[0].fill_between(common[m], (Fmean - Fstd)[m] * EV_TO_KJ, (Fmean + Fstd)[m] * EV_TO_KJ,
                       color="k", alpha=0.12)
    for k, n in enumerate(names):
        ax[0].annotate(f"{pops[:, k].mean():.2f}", (common[merged[k]], Fmean[merged[k]] * EV_TO_KJ),
                       textcoords="offset points", xytext=(0, 8), ha="center")
    ax[0].set(xlabel="CN(Sn-O$_w$)", ylabel="F / kJ mol$^{-1}$", title="free energy, replicas")
    ax[0].legend(fontsize=7)
    for d in data:
        c = np.loadtxt(d["rep"] / "COLVAR", comments="#")
        ax[1].plot(c[:, 0], c[:, 1], lw=0.4, label=d["rep"].name[-2:])
    ax[1].set(xlabel="time / ps", ylabel="CN", title="CN along the biased runs")
    for name, c in conv.items():
        if c:
            cc = np.array(c)
            ax[2].plot(cc[:, 0], cc[:, 1], "o-", ms=3, label=name[-2:])
    ax[2].set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title="two deepest basins")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / "metad_combined.png", dpi=200)
    print(f"wrote {OUTDIR / 'metad_combined.png'}")
except ImportError:
    pass
print(f"wrote {OUTDIR}/combined.json")
