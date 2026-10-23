# %% All seven metadynamics replicas on one page, and what they imply for the spectrum
# Reads outputs/03_combined/check_metad_replicas.json (first three replicas, r1 to
# r3) and check_metad_new.json (n5a, n5b started from the 5 coordinate state, n6a,
# n6b started from a 6 coordinate frame), both from check_metad_replicas.py.
# Prints dF(6c - 5c) and dF(4c - 5c) per replica (reweighted histogram with c(t),
# second half of the run, and the bias estimate at the end), their mean and
# spread with and without replica 1, the 6 coordinate population at 350 K, and
# the population weighted 119Sn shift from the state shifts of step 05.
# Run from the repository root:  python analysis/summarise_all_replicas.py

from pathlib import Path
import json

import numpy as np

# ----------------------------- CONFIG ------------------------------------
FILES = [Path("outputs/03_combined/check_metad_replicas.json"),
         Path("outputs/03_combined/check_metad_new.json")]
START_KIND = {"03_metad_r1": "5c", "03_metad_r2": "5c", "03_metad_r3": "5c",
              "03_metad_n5a": "5c", "03_metad_n5b": "5c", "03_metad_n6a": "6c", "03_metad_n6b": "6c"}
DELTA_5C, DELTA_6C = -578.5, -706.0     # ppm: 20 snapshots (2x2x2) and 4 snapshots (replica 1) of step 05
MACE_TO_PBE = (3.3, 9.7)                # kJ/mol, dF(6c-5c) shift from analysis/pbe_force_check.py
TEMPERATURE = 350.0
OUTFILE = Path("outputs/03_combined/all_replicas_summary.json")
# --------------------------------------------------------------------------

kT = 8.314462618e-3 * TEMPERATURE       # kJ/mol
rows = {}
for f in FILES:
    for rep, d in json.loads(f.read_text()).items():
        rows[rep] = dict(start=START_KIND[rep],
                         rew6=d["dF_reweighted_second_half"]["6 coord"], bias6=d["dF_bias_final"]["6 coord"],
                         rew4=d["dF_reweighted_second_half"]["4 coord"], bias4=d["dF_bias_final"]["4 coord"],
                         blocks=d["dF_6c_5c_reweighted_in_4_time_blocks"])
print(f"{'replica':14s} {'start':>5s} {'6c-5c reweighted':>17s} {'6c-5c bias':>11s} {'4c-5c reweighted':>17s} {'4c-5c bias':>11s}   6c-5c in 4 time blocks")
for rep, r in rows.items():
    print(f"{rep:14s} {r['start']:>5s} {r['rew6']:17.1f} {r['bias6']:11.1f} {r['rew4']:17.1f} {r['bias4']:11.1f}   "
          + " ".join(f"{b:6.1f}" for b in r["blocks"]))


def stats(keys, field):
    x = np.array([rows[k][field] for k in keys])
    return float(x.mean()), float(x.std(ddof=1)), float(x.std(ddof=1) / np.sqrt(len(x))), float(np.median(x))


allk = list(rows)
no_r1 = [k for k in allk if k != "03_metad_r1"]
res = {}
for label, keys in [("all 7", allk), ("without replica 1", no_r1),
                    ("started from 5c (5 replicas)", [k for k in allk if rows[k]["start"] == "5c"]),
                    ("started from 6c (2 replicas)", [k for k in allk if rows[k]["start"] == "6c"])]:
    m, s, se, med = stats(keys, "rew6")
    res[label] = dict(n=len(keys), mean=m, std=s, sem=se, median=med)
    print(f"dF(6c-5c), reweighted, {label:30s}: mean {m:5.1f}, spread (std) {s:4.1f}, standard error {se:3.1f}, median {med:5.1f} kJ/mol")
m4, s4, _, _ = stats(allk, "rew4")
print(f"dF(4c-5c), reweighted, all 7: mean {m4:.1f}, spread {s4:.1f} kJ/mol (window 0.05 to 0.65 of the CN; no minimum there)")

print("\nWeighted 119Sn shift (5c -578.5 ppm, 6c -706.0 ppm, 4c not computed because its population is negligible):")
table = {}
for label, dF in [("replica 1 alone", rows["03_metad_r1"]["rew6"]), ("all 7, mean", res["all 7"]["mean"]),
                  ("without replica 1, mean", res["without replica 1"]["mean"]),
                  ("without replica 1, MACE -> PBE +3.3", res["without replica 1"]["mean"] + MACE_TO_PBE[0]),
                  ("without replica 1, MACE -> PBE +9.7", res["without replica 1"]["mean"] + MACE_TO_PBE[1])]:
    p6 = 1.0 / (1.0 + np.exp(dF / kT))
    w = DELTA_5C * (1 - p6) + DELTA_6C * p6
    table[label] = dict(dF_kJmol=float(dF), p6=float(p6), weighted_ppm=float(w))
    print(f"  {label:38s}: dF(6c-5c) = {dF:5.1f} kJ/mol -> p(6c) = {p6:.4f} -> {w:7.1f} ppm")
OUTFILE.write_text(json.dumps(dict(per_replica=rows, dF6_stats=res, weighted_shift=table), indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
cols = {"5c": "tab:blue", "6c": "tab:red"}
for i, (rep, r) in enumerate(rows.items()):
    ax[0].plot([i], [r["rew6"]], "o", color=cols[r["start"]], ms=8)
    ax[0].plot([i], [r["bias6"]], "x", color="k", ms=6)
ax[0].axhline(res["without replica 1"]["mean"], color="grey", ls="--", lw=0.8)
ax[0].axhline(0, color="k", lw=0.5)
ax[0].set_xticks(range(len(rows)))
ax[0].set_xticklabels([k.replace("03_metad_", "") for k in rows], rotation=45)
ax[0].set(ylabel="dF(6 coord - 5 coord) / kJ mol$^{-1}$",
          title="seven replicas (blue: start 5 coord, red: start 6 coord; o reweighted, x bias)")
ax[0].title.set_fontsize(8)
for i, (k, t) in enumerate(table.items()):
    ax[1].barh(i, t["weighted_ppm"], color="tab:purple", alpha=0.7)
    ax[1].text(1.02, i, f"{t['weighted_ppm']:.0f}", va="center", ha="left", fontsize=8,   # value column right of
               transform=ax[1].get_yaxis_transform())                                   # the axes, off the band
ax[1].set_yticks(range(len(table)))
ax[1].set_yticklabels(list(table), fontsize=7)
ax[1].set_xlim(-720, -560)
ax[1].axvspan(-700, -685, color="tab:cyan", alpha=0.2)
ax[1].axvspan(-446, -420, color="tab:red", alpha=0.0)
ax[1].set(xlabel=r"population weighted $\delta(^{119}$Sn) / ppm", title="weighted shift by population scenario")
ax[1].title.set_fontsize(8)
fig.tight_layout()
fig.savefig(OUTFILE.parent / "all_replicas_summary.png", dpi=200)
print(f"wrote {OUTFILE}")
