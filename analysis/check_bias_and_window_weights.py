# %% Two checks on the metadynamics replicas: stored bias against HILLS, equal weights in a CN window
# 1. Rebuild the bias at every COLVAR frame from HILLS with analytic Gaussians and
#    compare it with the stored bias; rebuild F = -gamma/(gamma-1) V on the grid of
#    fes.dat and compare. The engine itself uses a grid with linear interpolation,
#    so a small difference against analytic Gaussians is expected.
# 2. Frames taken with equal weights inside a CN window follow the tempered
#    distribution exp(-F/(gamma kT)), not exp(-F/kT) (the bias depends on CN only,
#    so at fixed CN they are Boltzmann distributed). Compute how far the mean CN
#    moves on the measured F(CN), and what that does to a shift whose correlation
#    with CN is R_CORR. Optional: the same on a double well with an exact answer.
# Run from the repository root:  python analysis/check_bias_and_window_weights.py
# Needs outputs/03_metad_r*/ (HILLS, COLVAR, fes.dat) and outputs/03_combined/fes_mean.dat
# (from analyse_metad_replicas.py), written by the first version of 03.

from pathlib import Path
import numpy as np

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
FES_MEAN = Path("outputs/03_combined/fes_mean.dat")   # CN, F (kJ/mol), spread
KT_EV = 0.030160666417        # 350 K
GAMMA = 10.0
SIGMA = 0.07                  # as in 03
WINDOWS = {"5 coordinate": (0.90, 1.30), "6 coordinate": (1.40, 1.85)}
SD_DELTA_PPM = 28.0           # spread of the 20 five coordinate shifts
R_CORR = (0.08, 0.3, 0.5)     # correlation between shift and CN to test
RUN_TOY = False               # True: also run the double well (about 1 min per seed)
# --------------------------------------------------------------------------

KJ = 96.485
KT_KJ = KT_EV * KJ

# ---- 1. stored bias against HILLS, fes.dat against -gamma/(gamma-1) V
for rep in REPLICAS:
    t_h, c_h, s_h, h_h = np.loadtxt(rep / "HILLS", comments="#").T
    col = np.loadtxt(rep / "COLVAR", comments="#")
    worst = 0.0
    for t, cn, b in col:
        d = []
        for mask in (t_h < t - 1e-9, t_h <= t + 1e-9):       # hill at the same time: either convention
            v = np.sum(h_h[mask] * np.exp(-0.5 * ((cn - c_h[mask]) / s_h[mask]) ** 2))
            d.append(abs(v - b))
        worst = max(worst, min(d))
    grid, F = np.loadtxt(rep / "fes.dat", comments="#", unpack=True)
    V = np.zeros_like(grid)
    for c, h in zip(c_h, h_h):
        V += h * np.exp(-0.5 * ((grid - c) / SIGMA) ** 2)
    Fr = -(GAMMA / (GAMMA - 1.0)) * V
    dF = np.abs((F - F.min()) - (Fr - Fr.min())).max()
    print(f"{rep.name}: {len(t_h)} hills, {len(col)} frames; bias against analytic Gaussians "
          f"{worst * KJ:.1e} kJ/mol; fes.dat against -g/(g-1) V {dF * KJ:.1e} kJ/mol")

# ---- 2. equal weights inside a CN window
cn, F, _ = np.loadtxt(FES_MEAN, unpack=True)
f_kt = F / KT_KJ
for name, (lo, hi) in WINDOWS.items():
    m = (cn >= lo) & (cn <= hi)
    s, f = cn[m], f_kt[m] - f_kt[m].min()
    w_true, w_samp = np.exp(-f), np.exp(-f / GAMMA)       # Boltzmann, tempered
    mean_true = np.sum(s * w_true) / w_true.sum()
    mean_eq = np.sum(s * w_samp) / w_samp.sum()
    sd_cn = np.sqrt(np.sum((s - mean_eq) ** 2 * w_samp) / w_samp.sum())
    print(f"{name} window {lo} to {hi}: F rises {f.max():.2f} kT; mean CN {mean_true:.4f} (Boltzmann), "
          f"{mean_eq:.4f} (equal weights), difference {mean_eq - mean_true:+.4f}")
    for r in R_CORR:
        print(f"    shift error for correlation {r}: {r * SD_DELTA_PPM / sd_cn * (mean_eq - mean_true):+.2f} ppm "
              f"(sd CN {sd_cn:.3f}, sd shift {SD_DELTA_PPM} ppm)")

if RUN_TOY:
    from snmini.metad import WellTemperedMetaD1D
    from snmini.toy import double_well
    barrier, tilt, gam, sig, h0, pace, dt, fric, n_steps = 5.0, 1.0, 8.0, 0.1, 0.3, 200, 0.005, 1.0, 1_200_000
    lo, hi = -1.35, -0.65
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        md = WellTemperedMetaD1D(sigma=sig, height=h0, bias_factor=gam, kT=1.0,
                                 grid_min=-2.5, grid_max=2.5, n_bins=1001)
        c1 = np.exp(-fric * dt); c2 = np.sqrt(1 - c1 ** 2)
        x, v = -1.0, 0.0
        f = -double_well(x, barrier, tilt)[1] - md.dbias(x)
        xs = []
        for step in range(1, n_steps + 1):                   # BAOAB Langevin, kT = m = 1
            v += 0.5 * dt * f; x += 0.5 * dt * v
            v = c1 * v + c2 * rng.standard_normal(); x += 0.5 * dt * v
            f = -double_well(x, barrier, tilt)[1] - md.dbias(x)
            v += 0.5 * dt * f
            if step % pace == 0:
                xs.append(x); md.add_hill(x)
        xs = np.array(xs)[len(xs) // 3:]
        u = double_well(md.grid, barrier, tilt)[0]
        mw = (md.grid >= lo) & (md.grid <= hi)
        exact = np.sum(md.grid[mw] * np.exp(-u[mw])) / np.sum(np.exp(-u[mw]))
        xw = xs[(xs >= lo) & (xs <= hi)]
        wf = np.exp(np.interp(xw, md.grid, md.V))
        print(f"toy seed {seed}: exact <x> {exact:+.4f}; equal weights {xw.mean() - exact:+.4f}; "
              f"exp(V_final/kT) weights {np.sum(xw * wf) / wf.sum() - exact:+.4f} (errors); {len(xw)} frames")
