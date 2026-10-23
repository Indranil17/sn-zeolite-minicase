# %% Overview figures of the README: system, Sn sites, collective variable, takeaways
# Writes four PNG files to docs/figures/:
#   system_overview.png      the pipeline, where each step runs and what it was checked against
#   sn_sites.png             builder geometries of the four site types (before relaxation)
#   collective_variable.png  the switching function and what CN counts in each state
#   takeaways.png            free energy of the three states per replica, 119Sn shifts
# Data: docs/results/data/ (all_replicas_summary.json, relaxed_site_shifts.json,
# snapshot_shifts_4each.json, delta_119Sn_5c_20_snapshots.csv). The site figure builds
# the models with snmini.structures from the IZA CIF (cached in data/ or downloaded).
# Run from the repository root:  python analysis/make_overview_figures.py

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Circle

# ----------------------------- CONFIG ------------------------------------
DATA = Path("docs/results/data")
OUT = Path("docs/figures")
CIF_CACHE = "data"
DRAW_SITES = True          # needs ase, spglib and the IZA CIF (cached or downloadable)
R0, N_EXP, M_EXP = 2.6, 6, 12            # switching function of 03_metad.py
STATE_WINDOWS = {"4 coordinate": (0.20, 0.55), "5 coordinate": (0.90, 1.30), "6 coordinate": (1.40, 1.85)}
PORE_CN = 0.35             # the waters that are not bound, together, from the MD (docs/run_log.md, section 5)
BAND_DEHYDRATED = (-446.0, -420.0)       # ppm, resonances of dehydrated Sn-Beta [1, 12, 2], orientation only
BAND_HYDRATED = (-700.0, -685.0)
SEED = 0
# --------------------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 200})
TEAL, AMBER, BLUE, GREY, RED, DARK = "#1f7a7a", "#c98a1b", "#3b6ea5", "#6b7783", "#b5483a", "#1f2a33"
C_LAP, C_GPU, C_CPU = "#dfe7ef", "#d3ebe9", "#f6e5c3"


def f_switch(r, r0=R0, n=N_EXP, m=M_EXP):
    x = np.asarray(r, float) / r0
    return (1.0 - x ** n) / (1.0 - x ** m)          # = 1 / (1 + x**6) for n = 6, m = 12


# ============================================================ 1. system overview
def box(ax, x, y, w, h, title, sub, color, check=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=color, ec=DARK, lw=1.0))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=DARK)
    ax.text(x + w / 2, y + h - 4.6, sub, ha="center", va="top", fontsize=7.6, color="#33414d", linespacing=1.25)
    if check:
        ax.text(x + w / 2, y + 1.3, "check: " + check, ha="center", va="bottom", fontsize=7.0, style="italic",
                color=RED, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(13, 6.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
w, h = 21, 15.5
xs = [1.5, 26.5, 51.5, 76.5]
top = [("00  Site library", "closed, 4 open, 1 and 2 waters,\nwater loaded cell\nASE, spglib, IZA CIF", C_LAP,
        "IZA coordination\nsequence reproduced"),
       ("01  Foundation screen", "relax, anneal, re-quench\nMACE-MP-0 + D3(BJ)\nhydrolysis, binding energies", C_GPU,
        "PBE single points (pw.x),\nMAE 3.4 kJ/mol, 6 numbers"),
       ("02  Unbiased MD", "100 ps at 350 K\nSn-O(water) RDF\nfixes the CV cutoff r0", C_GPU,
        "error of CN(t) cross checked\nwith a second code (CRISP)"),
       ("03  Metadynamics", "well tempered, on CN(Sn-O water)\n7 replicas of 225 ps\nF(CN), populations p_k", C_GPU,
        "exact double well, PBE forces\non 60 frames, 7 replicas")]
for x, (t_, s_, c_, k_) in zip(xs, top):
    box(ax, x, 31, w, h, t_, s_, c_, k_)
for a_, b_ in zip(xs[:-1], xs[1:]):
    arrow(ax, a_ + w + 0.3, 38.7, b_ - 0.3, 38.7)
bot = [("04a, 04  QE inputs", "GIPAW pseudopotentials,\npw.x and gipaw.x inputs\nSnO2 reference + snapshots", C_LAP,
        "QE writers and reader\nunit tested"),
       ("pw.x + gipaw.x", "PBE, 70/560 Ry, QE 7.5\nCPU nodes, -nd 1", C_CPU,
        "SnO2 cutoff and k grid\nconverged to under 1 ppm"),
       ("05  Shifts per state", "delta = sigma(SnO2) - sigma\n+ delta(SnO2), -603.8 ppm\n20 + 4 snapshots", C_LAP,
        "bootstrap interval,\nsampling error only"),
       ("06  Weighted shift", "sum of p_k <delta>_k\none line per state", C_LAP,
        "scenario for each\nreplica population")]
for x, (t_, s_, c_, k_) in zip(xs, bot):
    box(ax, x, 2.5, w, h, t_, s_, c_, k_)
for a_, b_ in zip(xs[:-1], xs[1:]):
    arrow(ax, a_ + w + 0.3, 10.2, b_ - 0.3, 10.2)
arrow(ax, xs[3] + 4, 30.5, xs[0] + 8, 18.5)                  # snapshots to 04
ax.text(40, 26.2, "snapshots with all waters intact\n(4, 5 and 6 coordinate)", fontsize=7.8, color="#33414d",
        ha="center", bbox=dict(fc="white", ec="none", pad=1.5))
arrow(ax, xs[3] + w - 4, 30.5, xs[3] + 6, 18.5)              # populations to 06 (right column)
ax.text(xs[3] + 7.5, 23.5, "populations from\nF(CN) at 350 K", fontsize=7.8, color="#33414d", ha="right",
        va="center")                                          # left of the arrow, below the long one
for x, c, t_ in ((1.5, C_LAP, "laptop"), (10.5, C_GPU, "one GPU"), (20.5, C_CPU, "CPU nodes")):
    ax.add_patch(FancyBboxPatch((x, 47.3), 2.6, 1.7, boxstyle="round,pad=0.1,rounding_size=0.4", fc=c, ec=DARK, lw=0.8))
    ax.text(x + 3.5, 48.15, t_, va="center", fontsize=8, color=DARK)
ax.text(99, 48.15, "snmini: Sn in CHA, from structure to weighted $^{119}$Sn shift", ha="right", va="center",
        fontsize=9, color=DARK, fontweight="bold")
fig.savefig(OUT / "system_overview.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============================================================ 3. collective variable
r = np.linspace(1.5, 5.0, 400)
fr = f_switch(r)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
a1.plot(r, fr, color=DARK, lw=2)
a1.axvline(R0, color=GREY, ls=":", lw=1)
a1.text(R0 + 0.04, 0.93, "r0 = 2.6 Å", color=GREY, fontsize=8)
for rr, lab, col, ty in ((2.15, "bound water", TEAL, 0.70), (2.53, "second water\n(soft, 2.4 to 2.5 Å)", AMBER, 0.50)):
    a1.plot([rr], [f_switch(rr)], "o", color=col, ms=7)      # label below the curve, left of its point
    a1.text(rr - 0.06, ty, f"{lab}\nf = {float(f_switch(rr)):.2f}", color=col, fontsize=8, ha="right", va="top")
a1.axvspan(3.5, 4.0, color=BLUE, alpha=0.18)
a1.text(4.06, 0.40, f"pore waters\n3.5 to 4 Å (band)\nf = {float(f_switch(3.5)):.2f} to {float(f_switch(4.0)):.2f}",
        color=BLUE, fontsize=8, ha="left")                    # beside the band, not on it
a1.set(xlabel="Sn-O(water) distance (Å)", ylabel="contribution f(r) to CN", ylim=(0, 1.08), xlim=(1.5, 5.0),
       title="Each water adds f(r) = (1 - (r/r0)^6) / (1 - (r/r0)^12)")
bound, second = float(f_switch(2.15)), float(f_switch(2.53))
states = ["4 coordinate", "5 coordinate", "6 coordinate"]
x = np.arange(3)
a2.bar(x, [0, bound, bound], color=TEAL, width=0.5, label="bound water (2.15 Å)")
a2.bar(x, [0, 0, second], bottom=[0, 0, bound], color=AMBER, width=0.5, label="second water (2.53 Å)")
a2.bar(x, [PORE_CN] * 3, bottom=[0, bound, bound + second], color=BLUE, alpha=0.45, width=0.5,
       label="other waters (about 0.35)")
for i, name in enumerate(states):
    lo, hi = STATE_WINDOWS[name]
    a2.fill_between([i - 0.32, i + 0.32], lo, hi, color=RED, alpha=0.12)
    a2.plot([i - 0.32, i + 0.32], [lo, lo], color=RED, lw=1)
    a2.plot([i - 0.32, i + 0.32], [hi, hi], color=RED, lw=1)
    a2.text(i, hi + 0.03, f"window {lo:.2f} to {hi:.2f}", color=RED, fontsize=7.5, ha="center", va="bottom")
a2.set_xticks(x, [s.replace(" ", "\n") for s in states])
a2.set(ylabel="CN", ylim=(0, 2.1), xlim=(-0.5, 2.5), title="CN is a sum over waters; states sit near 0.4, 1.1, 1.5 to 1.7")
a2.legend(loc="upper left", fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "collective_variable.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
tops = [PORE_CN, bound + PORE_CN, bound + second + PORE_CN]
print(f"f(2.15) = {bound:.3f}, f(2.40) = {float(f_switch(2.40)):.3f}, f(2.53) = {second:.3f}, "
      f"f(3.5) = {float(f_switch(3.5)):.3f}, f(4.0) = {float(f_switch(4.0)):.3f}")
for name, top_cn in zip(states, tops):
    lo, hi = STATE_WINDOWS[name]
    print(f"{name}: CN = {top_cn:.2f}, stored window {lo} to {hi}: {'inside' if lo <= top_cn <= hi else 'OUTSIDE'}")

# ============================================================ 4. takeaways
summ = json.loads((DATA / "all_replicas_summary.json").read_text())["per_replica"]
names = list(summ)
rew6 = np.array([summ[n]["rew6"] for n in names])
rew4 = np.array([summ[n]["rew4"] for n in names])
rng = np.random.default_rng(SEED)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
for k, vals, col in ((0, rew4, GREY), (1, np.zeros(len(names)), TEAL), (2, rew6, AMBER)):
    jitter = rng.uniform(-0.12, 0.12, len(vals))
    b1.scatter(np.full(len(vals), k) + jitter, vals, s=34, color=col, zorder=3, edgecolor="white", lw=0.6)
i1 = names.index("03_metad_r1")
b1.scatter([2 + jitter[i1]], [rew6[i1]], s=70, facecolor="none", edgecolor=RED, lw=1.6, zorder=4)  # ring on its dot
b1.annotate("replica 1\n(unexplained)", (2, rew6[i1]), xytext=(2.25, rew6[i1] - 2), fontsize=8, color=RED, va="top")
b1.axhline(0, color=GREY, lw=0.6, ls=":")
b1.set_xticks([0, 1, 2], ["4 coordinate", "5 coordinate", "6 coordinate"])
b1.set(ylabel="free energy relative to 5 coordinate (kJ/mol)", xlim=(-0.5, 3.0),
       title="Free energy of the states, one dot per replica")
b1.text(0, rew4.min() - 6, f"{rew4.min():.0f} to {rew4.max():.0f}", ha="center", fontsize=8, color=GREY)
six = np.delete(rew6, i1)
b1.text(2, six.max() + 5, f"{six.min():.1f} to {six.max():.1f}\nin six of seven", ha="center", fontsize=8, color=AMBER)
b1.text(1, 6, "reference", ha="center", fontsize=8, color=TEAL)
b1.set_ylim(-12, 62)

rel = json.loads((DATA / "relaxed_site_shifts.json").read_text())["states"]
snap4 = json.loads((DATA / "snapshot_shifts_4each.json").read_text())["states"]
d20 = np.loadtxt(DATA / "delta_119Sn_5c_20_snapshots.csv", delimiter=",", skiprows=2, usecols=1)
boot = np.random.default_rng(SEED).choice(d20, (20000, len(d20))).mean(axis=1)
ci = np.percentile(boot, [2.5, 97.5])
b2.axvspan(*BAND_DEHYDRATED, color=GREY, alpha=0.18)
b2.axvspan(*BAND_HYDRATED, color=GREY, alpha=0.18)
# band labels beside their bands (the x axis is inverted: ha="left" runs toward more negative shifts)
b2.text(BAND_DEHYDRATED[0] - 3, -1.75, "dehydrated\nSn-Beta", ha="left", fontsize=7.5, color=GREY, va="bottom")
b2.text(BAND_HYDRATED[1] + 3, -1.75, "hydrated\nSn-Beta", ha="right", fontsize=7.5, color=GREY, va="bottom")
rows = [("closed site, relaxed\n(1 geometry, PBE)", [rel["T1_closed"]["delta"]], None, DARK),
        ("closed + 1 water\n(1 geometry)", [rel["T1_closed_1w"]["delta"]], None, DARK),
        ("closed + 2 waters\n(1 geometry)", [rel["T1_closed_2w"]["delta"]], None, DARK),
        ("5 coordinate, MD\n(20 snapshots)", d20, ci, TEAL),
        ("6 coordinate, MD\n(4 snapshots, replica 1)", np.array(snap4["cn1.48"]["shifts"]), None, AMBER)]
for k, (lab, vals, interval, col) in enumerate(rows):
    y = len(rows) - 1 - k
    b2.scatter(vals, np.full(len(vals), y) + rng.uniform(-0.07, 0.07, len(vals)), s=16, color=col, alpha=0.55, zorder=2)
    b2.scatter([float(np.mean(vals))], [y], marker="D", s=46, color=col, zorder=3, edgecolor="white")
    if interval is not None:
        b2.plot(interval, [y - 0.22, y - 0.22], color=col, lw=2.2)
b2.set_yticks(range(len(rows)), [r_[0] for r_ in rows][::-1], fontsize=8)
b2.axvline(-579.4, color=TEAL, lw=1, ls="--")
b2.text(-583.0, -0.55, "weighted shift\nabout -579", color=TEAL, fontsize=7.5, ha="left", va="top")  # right of the line
b2.set(xlabel=r"$\delta(^{119}$Sn) (ppm, vs SnMe$_4$)", ylim=(-1.8, 4.4),
       title="$^{119}$Sn shifts: computed (points), reported for Sn-Beta (bands)\n"
             "CHA, PBE: the bands are for orientation only")
b2.invert_xaxis()
fig.tight_layout()
fig.savefig(OUT / "takeaways.png", bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"5 coordinate: mean {d20.mean():.1f}, bootstrap 95 percent {ci[0]:.1f} to {ci[1]:.1f}; six replicas "
      f"{six.min():.1f} to {six.max():.1f}; 4 coordinate {rew4.min():.1f} to {rew4.max():.1f}")

# ============================================================ 2. Sn sites (builder geometries)
if DRAW_SITES:
    from snmini import structures as S

    fw, source = S.load_framework("CHA", CIF_CACHE, True)
    fw = S.to_primitive(fw)
    si = [a.index for a in fw if a.symbol == "Si"]
    closed = S.substitute_sn(fw, si[0])
    sn = closed.info["sn_index"]
    bridges = S.bonded(closed, sn, ("O",), cutoff=2.2)
    op = S.make_open_site(closed, sn, S.symmetry_classes(closed, bridges)[0][0])
    models = [("closed site", closed), ("open site (hydrolysed)", op),
              ("closed + 1 water", S.add_bound_water(closed, 1)), ("closed + 2 waters", S.add_bound_water(closed, 2))]
    COL = {"Sn": "#c98a1b", "Si": "#8fa3b8", "O": "#c9453b", "H": "#f4f4f4"}
    RAD = {"Sn": 0.55, "Si": 0.40, "O": 0.34, "H": 0.20}

    def local_cluster(at):
        # Sn, its O within 2.7 A, the H on those O, and the Si bonded to those O; positions relative to Sn
        s_i = at.info["sn_index"]
        vec = at.get_distances(s_i, range(len(at)), mic=True, vector=True)
        d = np.linalg.norm(vec, axis=1)
        o_near = [i for i in range(len(at)) if at[i].symbol == "O" and d[i] < 2.7]
        keep = {s_i, *o_near}
        for i in range(len(at)):
            if at[i].symbol in ("H", "Si") and any(
                    np.linalg.norm(vec[i] - vec[o]) < (1.2 if at[i].symbol == "H" else 1.9) for o in o_near):
                keep.add(i)
        keep = sorted(keep)
        return keep, vec[keep], [at[i].symbol for i in keep], d

    def best_rotation(seed, n_try=4000):
        # the view that separates the atoms of all four models best: largest smallest 2D distance
        # between two atoms, in units of the sum of their drawn radii
        clusters = []
        for _, m in models:
            _, pos, sym, _ = local_cluster(m)
            rr = np.array([RAD[s] for s in sym])
            clusters.append((pos, rr[:, None] + rr[None]))
        rng_ = np.random.default_rng(seed)
        best, best_score = np.eye(3), -1.0
        for _ in range(n_try):
            q = rng_.normal(size=4)
            q /= np.linalg.norm(q)
            a, b, c, d_ = q
            R = np.array([[a*a + b*b - c*c - d_*d_, 2*(b*c - a*d_), 2*(b*d_ + a*c)],
                          [2*(b*c + a*d_), a*a - b*b + c*c - d_*d_, 2*(c*d_ - a*b)],
                          [2*(b*d_ - a*c), 2*(c*d_ + a*b), a*a - b*b - c*c + d_*d_]])
            score = min((np.linalg.norm((pos @ R.T)[:, None, :2] - (pos @ R.T)[None, :, :2], axis=2) / rsum
                         + np.eye(len(pos)) * 99).min() for pos, rsum in clusters)
            if score > best_score:
                best, best_score = R, score
        return best

    def place_label(axx, xy, away, tag, atoms, bonds, taken):
        # first spot around the O (starting on the side away from Sn) where the label box
        # touches no atom, no bond, no other label and stays inside the panel
        rend = fig.canvas.get_renderer()
        base = np.arctan2(away[1], away[0])
        for dist in (0.75, 1.0, 1.3):
            for k in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6):
                ang = base + k * np.pi / 6
                t = axx.text(xy[0] + dist * np.cos(ang), xy[1] + dist * np.sin(ang), tag, fontsize=7.6,
                             color=RED, ha="center", va="center")
                bb = t.get_window_extent(rend).transformed(axx.transData.inverted())
                hit = (any(np.hypot(np.clip(cx, bb.x0, bb.x1) - cx, np.clip(cy, bb.y0, bb.y1) - cy) < rad + 0.03
                           for cx, cy, rad in atoms)
                       or any(mpath.Path(seg).intersects_bbox(bb.padded(0.06)) for seg in bonds)
                       or any(bb.overlaps(o) for o in taken)
                       or bb.x0 < lo_xy[0] or bb.x1 > hi_xy[0] or bb.y0 < lo_xy[1] or bb.y1 > hi_xy[1])
                if not hit:
                    taken.append(bb)
                    return
                t.remove()
        print(f"no free spot for the label {tag}")

    view = best_rotation(SEED)
    all_xy = np.vstack([(local_cluster(m)[1] @ view.T)[:, :2] for _, m in models])
    lo_xy, hi_xy = all_xy.min(axis=0) - 0.7, all_xy.max(axis=0) + 0.7
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for axx, (title, at) in zip(axs, models):
        keep, vec, sym, d = local_cluster(at)
        pos = vec @ view.T
        n = len(keep)
        axx.set_aspect("equal")
        axx.axis("off")
        axx.set_xlim(lo_xy[0], hi_xy[0])
        axx.set_ylim(lo_xy[1], hi_xy[1])
        axx.apply_aspect()                                    # final data to pixel scale, for the label boxes
        bonds = []
        for i in range(n):
            for j in range(i + 1, n):
                dij = np.linalg.norm(vec[i] - vec[j])
                pair = {sym[i], sym[j]}
                seg = [pos[i, :2], pos[j, :2]]
                if pair == {"Sn", "O"} and dij < 2.7:
                    axx.plot(*np.transpose(seg), color=AMBER, lw=3.0, zorder=1, solid_capstyle="round")
                    bonds.append(seg)
                elif (pair == {"Si", "O"} and dij < 1.9) or (pair == {"O", "H"} and dij < 1.2):
                    axx.plot(*np.transpose(seg), color="#55606b", lw=1.8, zorder=1)
                    bonds.append(seg)
        for i in np.argsort(pos[:, 2]):
            axx.add_patch(Circle((pos[i, 0], pos[i, 1]), RAD[sym[i]], fc=COL[sym[i]], ec=DARK,
                                 lw=1.4 if sym[i] == "Sn" else 0.8, zorder=2 + 0.01 * pos[i, 2]))
        atoms = [(pos[i, 0], pos[i, 1], RAD[sym[i]]) for i in range(n)]
        s_i = sym.index("Sn")
        taken = []
        for i in range(n):                                    # label the O that carry H
            if sym[i] != "O":
                continue
            hs = [j for j in range(n) if sym[j] == "H" and np.linalg.norm(vec[i] - vec[j]) < 1.2]
            on_si = any(sym[j] == "Si" and np.linalg.norm(vec[i] - vec[j]) < 1.9 for j in range(n))
            tag = {2: "H$_2$O", 1: "Si-OH" if on_si else "Sn-OH"}.get(len(hs))
            if tag:
                place_label(axx, pos[i, :2], pos[i, :2] - pos[s_i, :2], tag, atoms, bonds, taken)
        near = sorted(round(float(d[i]), 2) for i in range(len(at)) if at[i].symbol == "O" and d[i] < 2.7)
        axx.set_title(title, fontsize=10, fontweight="bold", color=DARK)
        axx.text(0.5, -0.02, f"{len(near)} O within 2.7 Å of Sn\n" + ", ".join(f"{v:.2f}" for v in near) + " Å",
                 transform=axx.transAxes, ha="center", va="top", fontsize=7.6, color="#33414d")
    key = [Line2D([], [], ls="", marker="o", ms=9, mfc=COL[e], mec=DARK, mew=0.8) for e in COL]
    fig.legend(key, list(COL), loc="center right", bbox_to_anchor=(0.995, 1.03), ncol=4, frameon=False,
               fontsize=8.5, handletextpad=0.3, columnspacing=1.4)
    fig.text(0.02, 1.03, "Builder geometries before relaxation (Sn, its O, their H, the Si they bind)",
             va="center", fontsize=8.5, color="#33414d")
    fig.savefig(OUT / "sn_sites.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("site models (O count, Sn index):",
          [(t_, sum(a.symbol == "O" for a in m), m.info.get("sn_index")) for t_, m in models], source)
