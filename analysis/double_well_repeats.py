# %% Repeat the double well check of the metadynamics engine with many seeds
# scripts/validate_metad_doublewell.py shows three seeds. A population or a free
# energy from one run has a seed to seed scatter, and three seeds cannot show it.
# This cell repeats the longest run (1.6 million steps) with SEEDS and reports,
# against the exact answer of the tilted double well:
#   * the RMSE of F(x) over -1.5 < x < 1.5, in kT;
#   * the error of the population of the right well (x > 0);
#   * the error of the free energy difference between the wells, ln of the odds
#     ratio, in kT and in kJ/mol at TEMPERATURE.
# The reference output is docs/validation_doublewell_repeats.json.
# Takes about 7 s per seed on a laptop. Run from the repository root:
#   python analysis/double_well_repeats.py

from pathlib import Path
import json

import numpy as np

from snmini.metad import basin_populations
from snmini.toy import exact_free_energy, run_metad_double_well

# ----------------------------- CONFIG ------------------------------------
TILT = 1.0
STEPS = 1_600_000
SEEDS = list(range(1, 16))
TEMPERATURE = 350.0        # K, only to express kT in kJ/mol
OUTFILE = Path("docs/validation_doublewell_repeats.json")
# --------------------------------------------------------------------------

KT_KJ = 8.314462618e-3 * TEMPERATURE
basins = {"left": (-2.5, 0.0), "right": (0.0, 2.5)}
rows = []
for seed in SEEDS:
    md, _ = run_metad_double_well(n_steps=STEPS, tilt=TILT, seed=seed)
    x, F = md.free_energy()
    Fe = exact_free_energy(x, tilt=TILT)
    m = (x > -1.5) & (x < 1.5)
    rmse = float(np.sqrt(np.mean(((F - F[m].min())[m] - (Fe - Fe[m].min())[m]) ** 2)))
    p = basin_populations(x, F, basins, 1.0)["right"]
    pe = basin_populations(x, Fe, basins, 1.0)["right"]
    d_ln_odds = float(np.log((p / (1 - p)) / (pe / (1 - pe))))
    rows.append(dict(seed=seed, rmse_kT=rmse, p_right=float(p), p_right_exact=float(pe),
                     dp=float(p - pe), d_free_energy_kT=d_ln_odds, d_free_energy_kJmol=d_ln_odds * KT_KJ))
    print(rows[-1])

dp = np.array([r["dp"] for r in rows])
dg = np.array([r["d_free_energy_kT"] for r in rows])
rm = np.array([r["rmse_kT"] for r in rows])
summary = dict(steps=STEPS, n_runs=len(rows), tilt=TILT,
               rmse_kT=dict(min=float(rm.min()), max=float(rm.max()), mean=float(rm.mean())),
               dp=dict(sd=float(dp.std(ddof=1)), largest_abs=float(np.abs(dp).max()),
                       share_within_0p01=float(np.mean(np.abs(dp) <= 0.01)),
                       share_within_0p02=float(np.mean(np.abs(dp) <= 0.02))),
               d_free_energy_kT=dict(sd=float(dg.std(ddof=1)), largest_abs=float(np.abs(dg).max())),
               d_free_energy_kJmol_at_T=dict(T=TEMPERATURE, sd=float(dg.std(ddof=1) * KT_KJ),
                                             largest_abs=float(np.abs(dg).max() * KT_KJ)))
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
OUTFILE.write_text(json.dumps(dict(summary=summary, runs=rows), indent=1))
print(json.dumps(summary, indent=1))
print(f"wrote {OUTFILE}")
