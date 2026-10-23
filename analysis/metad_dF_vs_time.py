# %% Free energy differences between the Sn coordination states against time
# Rebuilds the metadynamics bias of every replica from its HILLS file (the
# engine's own from_hills), at N_POINTS times, and reads off
#   dF(6 coordinate - 5 coordinate) and dF(4 coordinate - 5 coordinate)
# as F = -gamma/(gamma-1) V, integrated over CN windows. A converged run has a
# dF that stops drifting and replicas that agree. Also gives the average of
# the bias over the second half of each run (the usual way to reduce noise).
# Windows follow what the stored frames show (characterise_states.py): CN near 1.1 is Sn
# with one bound water (5 coordinate), CN 1.35 to 2.1 has two bound waters
# (6 coordinate), CN below about 0.65 has none (4 coordinate).
# Run from the repository root:  python analysis/metad_dF_vs_time.py

from pathlib import Path
import json

import numpy as np

from snmini.metad import KB_EV, WellTemperedMetaD1D

# ----------------------------- CONFIG ------------------------------------
REPLICAS = sorted(Path("outputs").glob("03_metad_r[0-9]*"))
TEMPERATURE = 350.0
SIGMA, BIAS_FACTOR, N_WATER = 0.07, 10.0, 6        # as in 03
WINDOWS = {"4 coord": (0.05, 0.65), "5 coord": (0.65, 1.35), "6 coord": (1.35, 2.10)}
N_POINTS = 10
OUTDIR = Path("outputs/03_combined")
# --------------------------------------------------------------------------

kT = KB_EV * TEMPERATURE
EV_TO_KJ = 96.485
OUTDIR.mkdir(parents=True, exist_ok=True)
trapz = getattr(np, "trapezoid", None) or np.trapz


def free_energy_of(hills):
    m = WellTemperedMetaD1D.from_hills(hills, sigma=SIGMA, height=1.0, bias_factor=BIAS_FACTOR,
                                       kT=kT, grid_min=-0.5, grid_max=N_WATER + 0.5)
    return m.grid, -(BIAS_FACTOR / (BIAS_FACTOR - 1.0)) * m.V


def dF(s, F):
    """Free energy of each window (-kT ln of its Boltzmann weight), as differences to 5 coord."""
    w = {}
    for name, (lo, hi) in WINDOWS.items():
        m = (s >= lo) & (s < hi)
        w[name] = trapz(np.exp(-(F[m] - F.min()) / kT), s[m])
    ref = w["5 coord"]
    return {n: float(-kT * np.log(v / ref) * EV_TO_KJ) for n, v in w.items() if n != "5 coord"}


out = {}
for rep in REPLICAS:
    h = np.loadtxt(rep / "HILLS", comments="#")           # time_ps centre sigma height_eV
    t_end = h[-1, 0]
    rows = []
    for k in range(1, N_POINTS + 1):
        n = int(round(len(h) * k / N_POINTS))
        s, F = free_energy_of([(c, e) for c, e in zip(h[:n, 1], h[:n, 3])])
        d = dF(s, F)
        rows.append(dict(time_ps=float(h[n - 1, 0]), **d))
    # time average of the bias over the second half: F averaged from the bias
    # reconstructed at 10 evenly spaced times between 50 and 100 percent
    Fs = []
    for k in range(5, 11):
        n = int(round(len(h) * k / 10))
        s, F = free_energy_of([(c, e) for c, e in zip(h[:n, 1], h[:n, 3])])
        Fs.append(F - F.min())
    avg = dF(s, np.mean(Fs, axis=0))
    out[rep.name] = dict(vs_time=rows, second_half_average=avg, t_end_ps=float(t_end))
    print(f"{rep.name}  dF 6c-5c (kJ/mol) vs time: "
          + " ".join(f"{r['6 coord']:6.1f}" for r in rows)
          + f"   | second half average: 6c-5c {avg['6 coord']:.1f}, 4c-5c {avg['4 coord']:.1f}")
    print(f"{'':10s}dF 4c-5c (kJ/mol) vs time: " + " ".join(f"{r['4 coord']:6.1f}" for r in rows))

avg6 = np.array([v["second_half_average"]["6 coord"] for v in out.values()])
avg4 = np.array([v["second_half_average"]["4 coord"] for v in out.values()])
print(f"\nsecond half averages over {len(out)} replicas: dF(6c-5c) = {avg6.mean():.1f} +- {avg6.std(ddof=1):.1f} "
      f"(range {avg6.min():.1f} to {avg6.max():.1f}) kJ/mol;  dF(4c-5c) = {avg4.mean():.1f} +- {avg4.std(ddof=1):.1f} kJ/mol")
out["summary"] = dict(dF_6c_5c_mean=float(avg6.mean()), dF_6c_5c_std=float(avg6.std(ddof=1)),
                      dF_6c_5c_per_replica=[float(x) for x in avg6],
                      dF_4c_5c_mean=float(avg4.mean()), dF_4c_5c_std=float(avg4.std(ddof=1)),
                      windows=WINDOWS)
(OUTDIR / "dF_vs_time.json").write_text(json.dumps(out, indent=1))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, v in out.items():
        if name == "summary":
            continue
        t = [r["time_ps"] for r in v["vs_time"]]
        ax[0].plot(t, [r["6 coord"] for r in v["vs_time"]], "o-", ms=3, label=name[-2:])
        ax[1].plot(t, [r["4 coord"] for r in v["vs_time"]], "o-", ms=3, label=name[-2:])
    ax[0].set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title="6 coordinate minus 5 coordinate")
    ax[1].set(xlabel="time / ps", ylabel="dF / kJ mol$^{-1}$", title="4 coordinate minus 5 coordinate")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTDIR / "dF_vs_time.png", dpi=200)
    print(f"wrote {OUTDIR / 'dF_vs_time.png'}")
except ImportError:
    pass
