# %% 03c  Combine the umbrella windows with MBAR: F(CN) and the 6 against 5 coordinate difference
# Reads every window of each family (a family is the set of windows started from
# one kind of structure, for example from the 5 coordinate runs and from the 6
# coordinate runs) and solves MBAR for each family on its own. If the families
# disagree, the result depends on the start structure, a sign of a slow degree of
# freedom that the CN does not describe. Errors: block bootstrap over each window
# (blocks of BLOCK samples), MBAR repeated for every resample.
# Run from the repository root after the windows of 03b have finished.

from pathlib import Path
import json

import numpy as np

from snmini.metad import KB_EV
from snmini.umbrella import (mbar_f, pmf, reduced_energies, unbiased_weights,
                             window_probability)

# ----------------------------- CONFIG ------------------------------------
FAMILIES = {"started from 5 coordinate": Path("outputs/03b_umb_n5a"),
            "started from 6 coordinate": Path("outputs/03b_umb_n6a")}
TEMPERATURE = 350.0
DISCARD_PS = 10.0          # equilibration of each window
STRIDE = 5                 # keep every 5th sample (50 fs)
BLOCK = 100                # samples per bootstrap block (5 ps)
N_BOOT = 30
DF_WINDOWS = {"5 coord": (0.65, 1.35), "6 coord": (1.35, 2.10)}
F_POINTS = [0.3, 0.4, 0.5]
BIN = 0.05
OUTDIR = Path("outputs/03_combined")
SEED = 0
# --------------------------------------------------------------------------

kT = KB_EV * TEMPERATURE
EV_KJ = 96.485
rng = np.random.default_rng(SEED)
OUTDIR.mkdir(parents=True, exist_ok=True)


def load(root):
    wins = []
    for d in sorted(root.glob("w_*")):
        if not (d / "summary.json").exists():
            print(f"skip (not finished): {d}")
            continue
        s = json.loads((d / "summary.json").read_text())
        t, cn = np.loadtxt(d / "COLVAR", comments="#", unpack=True)
        x = cn[t >= DISCARD_PS][::STRIDE]
        wins.append(dict(name=d.name, centre=s["centre"], k=s["k_force_eV_per_CN2"], x=x))
    return wins


def solve(wins, idx_by_win=None):
    """MBAR over windows; idx_by_win gives the samples of each window (for the bootstrap)."""
    xs = [w["x"] if idx_by_win is None else w["x"][idx_by_win[i]] for i, w in enumerate(wins)]
    s = np.concatenate(xs)
    n_k = [len(x) for x in xs]
    # every window may have its own force constant: u_kn is built window by window
    u = np.array([0.5 * w["k"] * (s - w["centre"]) ** 2 / kT for w in wins])
    f = mbar_f(u, n_k)
    return s, unbiased_weights(u, n_k, f), f


def df_of(s, w):
    p = {n: window_probability(s, w, lo, hi) for n, (lo, hi) in DF_WINDOWS.items()}
    return float(-kT * np.log(p["6 coord"] / p["5 coord"]) * EV_KJ)


results, curves = {}, {}
edges = np.arange(0.0, 2.6 + BIN, BIN)
mid = 0.5 * (edges[1:] + edges[:-1])
for name, root in FAMILIES.items():
    wins = load(root)
    if len(wins) < 3:
        print(f"{name}: only {len(wins)} finished windows, skipped")
        continue
    s, w, f = solve(wins)
    F = pmf(s, w, edges, kT) * EV_KJ
    dF = df_of(s, w)
    boots = []
    for _ in range(N_BOOT):
        idx = []
        for win in wins:
            n = len(win["x"])
            nb = max(1, n // BLOCK)
            starts = rng.integers(0, max(1, n - BLOCK + 1), nb)
            idx.append(np.concatenate([np.arange(a, a + BLOCK) for a in starts]))
        sb, wb, _ = solve(wins, idx)
        boots.append((df_of(sb, wb), pmf(sb, wb, edges, kT) * EV_KJ))
    dF_err = float(np.std([b[0] for b in boots], ddof=1))
    F_err = np.nanstd(np.array([b[1] for b in boots]), axis=0, ddof=1)
    ref = np.nanmin(F[(mid > 0.8) & (mid < 1.8)])
    pts = {f"{p:.2f}": float(np.interp(p, mid[~np.isnan(F)], (F - ref)[~np.isnan(F)])) for p in F_POINTS}
    results[name] = dict(windows=len(wins), samples=int(len(s)), dF_6c_minus_5c_kJmol=dF,
                         dF_error_kJmol=dF_err, F_at_cn_kJmol_vs_5c_minimum=pts,
                         window_centres=[w_["centre"] for w_ in wins],
                         mean_cn_per_window=[float(np.mean(w_["x"])) for w_ in wins])
    curves[name] = (F - ref, F_err)
    print(f"{name}: {len(wins)} windows, {len(s)} samples; dF(6c - 5c) = {dF:.1f} +- {dF_err:.1f} kJ/mol; "
          f"F at CN 0.3, 0.4, 0.5: {pts}")

(OUTDIR / "umbrella_mbar.json").write_text(json.dumps(results, indent=1))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, (F, err) in curves.items():
        ax.plot(mid, F, label=name)
        ax.fill_between(mid, F - err, F + err, alpha=0.2)
    ax.set(xlabel="CN(Sn-O$_w$)", ylabel="F / kJ mol$^{-1}$", xlim=(0.1, 2.4), ylim=(-5, 110),
           title="umbrella sampling and MBAR")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "umbrella_mbar.png", dpi=200)
except ImportError:
    pass
print(f"wrote {OUTDIR / 'umbrella_mbar.json'}")
