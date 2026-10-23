# %% Validate the metadynamics engine where the answer is known
# A particle in a tilted double well, U(x) = 5 (x^2 - 1)^2 + 1.0 x (units kT).
# The exact free energy is U itself, so every error below is the engine's.
# Takes about a minute on a laptop.

from pathlib import Path
import json

import numpy as np

from snmini.metad import basin_populations
from snmini.toy import exact_free_energy, run_metad_double_well

# ----------------------------- CONFIG ------------------------------------
TILT = 1.0
LENGTHS = [100_000, 400_000, 1_600_000]       # MD steps
SEEDS = [1, 2, 3]
OUTDIR = Path("outputs/validation")
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
basins = {"left": (-2.5, 0.0), "right": (0.0, 2.5)}
rows, curves = [], {}
for n in LENGTHS:
    for seed in SEEDS:
        md, _ = run_metad_double_well(n_steps=n, tilt=TILT, seed=seed)
        x, F = md.free_energy()
        Fe = exact_free_energy(x, tilt=TILT)
        m = (x > -1.5) & (x < 1.5)
        err = (F - F[m].min())[m] - (Fe - Fe[m].min())[m]
        p = basin_populations(x, F, basins, 1.0)
        pe = basin_populations(x, Fe, basins, 1.0)
        rows.append(dict(steps=n, seed=seed, hills=len(md.hills),
                         rmse_kT=float(np.sqrt(np.mean(err**2))),
                         p_right=p["right"], p_right_exact=pe["right"]))
        curves[(n, seed)] = (x, F - F[m].min())
        print(rows[-1])

(OUTDIR / "doublewell.json").write_text(json.dumps(rows, indent=1))
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x, _ = curves[(LENGTHS[-1], SEEDS[0])]
    Fe = exact_free_energy(x, tilt=TILT)
    m = (x > -1.5) & (x < 1.5)
    ax[0].plot(x[m], (Fe - Fe[m].min())[m], "k", lw=2, label="exact")
    for seed in SEEDS:
        xs, Fs = curves[(LENGTHS[-1], seed)]
        ax[0].plot(xs[m], Fs[m], lw=1, label=f"WT-MetaD, seed {seed}")
    ax[0].set(xlabel="x", ylabel="F / kT", ylim=(-0.5, 9))
    ax[0].legend(fontsize=7, loc="upper center")       # above the barrier, clear of the curves
    for seed in SEEDS:
        r = [d["rmse_kT"] for d in rows if d["seed"] == seed]
        ax[1].loglog(LENGTHS, r, "o-", label=f"seed {seed}")
    ax[1].set(xlabel="MD steps", ylabel="RMSE of F / kT")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / "doublewell.png", dpi=200)
except ImportError:
    pass
