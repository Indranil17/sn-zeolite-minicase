# %% Checks on the metadynamics replicas of step 03 (completeness and estimator)
# Are the files and figures of every replica complete, and is the free energy
# drawn correct? Four checks, each of which can fail:
#  1. completeness: number of hills, COLVAR frames, stored state frames, and which
#     CN ranges the stored frames cover (a CN range without stored frames is data
#     that cannot be used for GIPAW);
#  2. consistency: the final free energy fes.dat against the one rebuilt from HILLS,
#     and the bias stored in COLVAR against the bias rebuilt from HILLS at each frame;
#  3. range: which part of the CN axis the bias visited (outside it F is a flat
#     extrapolation, not data);
#  4. estimator: dF between the coordination states from the bias, F = -g/(g-1) V
#     (Barducci, Bussi, Parrinello 2008), against a second estimator from the same
#     frames, the reweighted histogram with the time dependent offset c(t)
#     (Tiwary and Parrinello, J. Phys. Chem. B 2015). Both share the CV and the
#     potential: this checks the algebra, not the choice of CV.
# Run from the repository root:  python analysis/check_metad_replicas.py

from pathlib import Path
import json

import numpy as np
from ase.io import read

from snmini.metad import KB_EV

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
TEMPERATURE = 350.0
SIGMA, GAMMA = 0.07, 10.0                   # as in 03
STEPS, PACE, SAVE_EVERY, DT_FS = 450_000, 250, 200, 0.5
WINDOWS = {"4 coord": (0.05, 0.65), "5 coord": (0.65, 1.35), "6 coord": (1.35, 2.10)}
GRID = np.linspace(-0.5, 6.5, 2001)         # grid of 03 (N_WATER = 6)
OUTFILE = Path("outputs/03_combined/check_metad_replicas.json")
# --------------------------------------------------------------------------

kT = KB_EV * TEMPERATURE
EV_KJ = 96.485
trapz = getattr(np, "trapezoid", None) or np.trapz


def window_dF(w_by_window):
    ref = w_by_window["5 coord"]
    return {n: float(-kT * np.log(v / ref) * EV_KJ) for n, v in w_by_window.items()
            if n != "5 coord" and v > 0 and ref > 0}


report = {}
for rep in REPLICAS:
    h = np.loadtxt(rep / "HILLS", comments="#")              # t_ps, centre, sigma, height_eV
    col = np.loadtxt(rep / "COLVAR", comments="#")           # t_ps, CN, bias_eV
    s_fes, F_fes = np.loadtxt(rep / "fes.dat", unpack=True)
    out = dict(hills=len(h), hills_expected=STEPS // PACE,
               colvar_frames=len(col), colvar_expected=STEPS // SAVE_EVERY + 1)
    # ---- 1. stored frames
    stored = []
    for f in sorted(rep.glob("state_*.extxyz")):
        stored += [a.info["cv"] for a in read(f, index=":")]
    stored = np.array(stored)
    out["state_files"] = [f.name for f in sorted(rep.glob("state_*.extxyz"))]
    out["stored_frames"] = int(len(stored))
    for name, (lo, hi) in WINDOWS.items():
        in_run = int(np.sum((col[:, 1] >= lo) & (col[:, 1] < hi)))
        in_store = int(np.sum((stored >= lo) & (stored < hi)))
        out[f"frames_in_{name.replace(' ', '_')}_COLVAR"] = in_run
        out[f"frames_in_{name.replace(' ', '_')}_stored"] = in_store
    # ---- 2. consistency of HILLS, COLVAR and fes.dat
    V = np.zeros_like(GRID)
    Vframe, k = [], 0
    times = col[:, 0]
    order = np.argsort(h[:, 0])
    hh = h[order]
    for t, s in zip(times, col[:, 1]):
        while k < len(hh) and hh[k, 0] < t - 1e-9:       # hills deposited before this frame
            V += hh[k, 3] * np.exp(-0.5 * ((GRID - hh[k, 1]) / SIGMA) ** 2)
            k += 1
        Vframe.append(np.interp(s, GRID, V))
    Vframe = np.array(Vframe)
    out["bias_COLVAR_vs_HILLS_max_abs_diff_kJmol"] = float(np.max(np.abs(Vframe - col[:, 2])) * EV_KJ)
    Vfin = np.zeros_like(GRID)
    for c, e in zip(h[:, 1], h[:, 3]):
        Vfin += e * np.exp(-0.5 * ((GRID - c) / SIGMA) ** 2)
    F_hills = -(GAMMA / (GAMMA - 1)) * Vfin
    F_hills -= F_hills.min()
    out["fes_dat_vs_HILLS_max_abs_diff_kJmol"] = float(np.max(np.abs(F_hills - F_fes)) * EV_KJ)
    # ---- 3. range of the bias
    lo_v, hi_v = float(h[:, 1].min() - 3 * SIGMA), float(h[:, 1].max() + 3 * SIGMA)
    out["visited_CN_range"] = [lo_v, hi_v]
    out["fraction_of_plotted_axis_with_data"] = float((hi_v - lo_v) / (GRID[-1] - GRID[0]))
    out["F_plateau_kJmol"] = float(F_fes.max() * EV_KJ)
    # ---- 4. estimators
    win = (GRID >= lo_v) & (GRID <= hi_v)
    Fb = -(GAMMA / (GAMMA - 1)) * Vfin
    wb = {n: float(trapz(np.exp(-(Fb[(GRID >= a) & (GRID < b)] - Fb[win].min()) / kT),
                         GRID[(GRID >= a) & (GRID < b)])) for n, (a, b) in WINDOWS.items()}
    out["dF_bias_final"] = window_dF(wb)
    # time dependent offset c(t) at every frame, by adding hills in time order
    V = np.zeros_like(GRID)
    k = 0
    cs = []
    for t in times:
        while k < len(hh) and hh[k, 0] < t - 1e-9:
            V += hh[k, 3] * np.exp(-0.5 * ((GRID - hh[k, 1]) / SIGMA) ** 2)
            k += 1
        num = trapz(np.exp(GAMMA / (GAMMA - 1) * V[win] / kT), GRID[win])
        den = trapz(np.exp(V[win] / ((GAMMA - 1) * kT)), GRID[win])
        cs.append(kT * np.log(num / den))
    cs = np.array(cs)
    logw = (col[:, 2] - cs) / kT
    for label, t0 in [("all", 0.0), ("from_56ps", 56.0), ("second_half", 112.5)]:
        m = times >= t0
        w = np.exp(logw[m] - logw[m].max())
        sel = {n: float(np.sum(w[(col[m, 1] >= a) & (col[m, 1] < b)])) for n, (a, b) in WINDOWS.items()}
        out[f"dF_reweighted_{label}"] = window_dF(sel)
    # 4 time blocks of the reweighted estimate
    blocks = []
    edges_t = np.linspace(0, times[-1] + 1e-6, 5)
    for a, b in zip(edges_t[:-1], edges_t[1:]):
        m = (times >= a) & (times < b)
        w = np.exp(logw[m] - logw[m].max())
        sel = {n: float(np.sum(w[(col[m, 1] >= lo) & (col[m, 1] < hi)])) for n, (lo, hi) in WINDOWS.items()}
        blocks.append(window_dF(sel).get("6 coord"))
    out["dF_6c_5c_reweighted_in_4_time_blocks"] = blocks
    out["effective_sample_fraction_second_half"] = float(
        (lambda w: w.sum() ** 2 / (w ** 2).sum() / len(w))(np.exp(logw[times >= 112.5] - logw[times >= 112.5].max())))
    # cumulative estimates at ten times: bias (hills deposited so far) and reweighted frames so far
    series = []
    for kk in range(1, 11):
        t_k = times[-1] * kk / 10
        Vk = np.zeros_like(GRID)
        for c_, e_ in zip(hh[hh[:, 0] <= t_k, 1], hh[hh[:, 0] <= t_k, 3]):
            Vk += e_ * np.exp(-0.5 * ((GRID - c_) / SIGMA) ** 2)
        Fk = -(GAMMA / (GAMMA - 1)) * Vk
        wk = {n: float(trapz(np.exp(-(Fk[(GRID >= a) & (GRID < b)] - Fk[win].min()) / kT),
                             GRID[(GRID >= a) & (GRID < b)])) for n, (a, b) in WINDOWS.items()}
        m = times <= t_k
        wr = np.exp(logw[m] - logw[m].max())
        sr = {n: float(np.sum(wr[(col[m, 1] >= a) & (col[m, 1] < b)])) for n, (a, b) in WINDOWS.items()}
        series.append(dict(time_ps=float(t_k), bias=window_dF(wk), reweighted=window_dF(sr)))
    out["cumulative_estimates"] = series
    out["_F_plot"] = (GRID[win], (F_fes - F_fes[(GRID > 0.8) & (GRID < 1.8)].min())[win] * EV_KJ)
    out["_cn_trace"] = (col[:, 0], col[:, 1])
    report[rep.name] = out
    print(f"\n=== {rep.name}")
    for k_, v in out.items():
        if not k_.startswith("_") and k_ != "cumulative_estimates":
            print(f"  {k_}: {v}")

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
OUTFILE.write_text(json.dumps({r: {k: v for k, v in d.items() if not k.startswith("_")}
                               for r, d in report.items()}, indent=1, default=float))
print(f"\nwrote {OUTFILE}")

# ---- figure: only the CN range that the bias visited, and both estimators against time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
colours = {r: plt.cm.tab10(i % 10) for i, r in enumerate(report)}
for r, d in report.items():
    g, Fp = d["_F_plot"]
    ax[0, 0].plot(g, Fp, color=colours[r], label=r[-2:])
    t, s = d["_cn_trace"]
    ax[0, 1].plot(t, s, color=colours[r], lw=0.4, label=r[-2:])
    cum = d["cumulative_estimates"]
    tt = [c["time_ps"] for c in cum]
    for row, key, title in [(1, "6 coord", "6 coordinate minus 5 coordinate"),
                            (1, "4 coord", "4 coordinate minus 5 coordinate")]:
        a = ax[1, 0] if key == "6 coord" else ax[1, 1]
        a.plot(tt, [c["bias"].get(key, np.nan) for c in cum], "-", color=colours[r], label=f"{r[-2:]} bias")
        a.plot(tt, [c["reweighted"].get(key, np.nan) for c in cum], "o--", ms=3, color=colours[r],
               label=f"{r[-2:]} reweighted")
        a.set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title=title)
for b in (0.65, 1.35, 2.10):
    ax[0, 0].axvline(b, color="grey", ls=":", lw=0.8)
ax[0, 0].set(xlabel="CN(Sn-O$_w$)", ylabel="F / kJ mol$^{-1}$",
             title="free energy, only where the bias went (no data outside)")
ax[0, 0].legend(fontsize=7, loc="lower left")
ax[0, 1].set(xlabel="time / ps", ylabel="CN", title="CN along the biased runs")
handles, labels = ax[1, 0].get_legend_handles_labels()       # one key for both bottom panels, below them
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=len(report), fontsize=7, frameon=False)
fig.savefig(OUTFILE.parent / "metad_replicas_checked.png", dpi=200)
print(f"wrote {OUTFILE.parent / 'metad_replicas_checked.png'}")
