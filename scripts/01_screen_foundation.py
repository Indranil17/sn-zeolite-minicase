# %% 01  Screen the Sn site library with a MACE foundation model
# Relax every model, then anneal and re-quench a few times so that open sites
# are not trapped in the geometry the builder guessed. Reports hydrolysis and
# hydration energies. These are foundation model numbers: the point of this
# step is ranking and triage, and the ranking is re-checked with DFT single
# points at the relaxed geometries (docs/runbook.md, step 2).

from pathlib import Path
import csv
import json

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE

from snmini import structures as S
from snmini.calc import get_calculator, langevin, thermalize

# ----------------------------- CONFIG ------------------------------------
MODELS = Path("outputs/00_structures/models.extxyz")
CALC = "mace"              # "mace", or "toy" for a no-physics smoke test
MODEL = "medium"           # any name your mace_mp accepts, e.g. "medium-mpa-0"
DISPERSION = True          # add D3(BJ) on top of the foundation model
DEVICE = "cuda"            # "cpu" works, slowly
DTYPE = "float64"          # float64 for relaxations and energy differences
FMAX = 0.02                # eV/A
MAX_STEPS = 2000
RELAX_CELL = True          # True for the idealised fallback framework
N_QUENCH = 3               # anneal + re-quench cycles per model (0: plain relax)
T_ANNEAL = 400.0           # K
ANNEAL_STEPS = 4000        # 2 ps at 0.5 fs
TIMESTEP_FS = 0.5
SEED = 1
OUTDIR = Path("outputs/01_screen")
# --------------------------------------------------------------------------


def relax(atoms, calc, cell):
    atoms.calc = calc
    obj = FrechetCellFilter(atoms) if cell else atoms
    FIRE(obj, logfile=None).run(fmax=FMAX, steps=MAX_STEPS)
    return atoms.get_potential_energy()


def anneal(atoms, calc, rng):
    atoms.calc = calc
    thermalize(atoms, T_ANNEAL, rng)
    langevin(atoms, TIMESTEP_FS, T_ANNEAL, rng).run(ANNEAL_STEPS)


OUTDIR.mkdir(parents=True, exist_ok=True)
calc = get_calculator(CALC, MODEL, DISPERSION, DEVICE, DTYPE)
rng = np.random.default_rng(SEED)
models = read(MODELS, index=":")
results, relaxed = {}, []
for m in models:
    label = m.info["label"]
    is_gas = label == "H2O_gas"
    # The state of a label is the Sn coordination of its first relaxation
    # (cycle 0). A later cycle where annealing changed it (a water left, an
    # open site closed) is a different state, so it may not replace the
    # minimum of this label; every cycle is recorded in energies.csv.
    ref_cn = None
    best_e, best, cycles = None, None, []
    trial = m.copy()
    trial.info.update(m.info)
    for cycle in range(N_QUENCH + 1):
        if cycle > 0 and not is_gas:
            anneal(trial, calc, rng)
        e = relax(trial, calc, cell=RELAX_CELL and not is_gas)
        cn = S.audit(trial)["sn_cn"] if "sn_index" in m.info else None
        if cycle == 0:
            ref_cn = cn
        cycles.append(dict(cycle=cycle, energy_eV=float(e), sn_cn=cn))
        if cn == ref_cn and (best_e is None or e < best_e):
            best_e, best = e, trial.copy()
        if is_gas:
            break
    state_ok = all(c["sn_cn"] == ref_cn for c in cycles)   # False: some cycle left the state
    best.info.update(m.info)
    best.info["energy_eV"] = float(best_e)
    row = dict(label=label, energy_eV=float(best_e),
               formula=best.get_chemical_formula(), state_ok=state_ok,
               cycles=json.dumps(cycles))
    if "sn_index" in m.info:
        rep = S.audit(best)
        row.update(sn_cn=rep["sn_cn"], min_heavy=rep["min_heavy"])
        if m.info.get("site") == "open":
            row["sn_silanol_O"] = float(best.get_distance(
                m.info["sn_index"], m.info["hydrolysed_o"], mic=True))
    results[label] = row
    relaxed.append(best)
    print(json.dumps(row, default=float))

# derived energies (eV): hydrolysis of each bridge, binding of n waters
e_w = results["H2O_gas"]["energy_eV"]
derived = []
for label, row in results.items():
    if "_open" in label:
        t = label.split("_")[0]
        derived.append(dict(label=label, quantity="E_hydrolysis",
                            eV=row["energy_eV"] - results[f"{t}_closed"]
                            ["energy_eV"] - e_w))
    if label.endswith("w") and "_closed_" in label:
        t, n = label.split("_")[0], int(label.split("_")[-1][:-1])
        derived.append(dict(label=label, quantity=f"E_bind_{n}w",
                            eV=(row["energy_eV"] - results[f"{t}_closed"]
                                ["energy_eV"] - n * e_w) / n))
for d in derived:
    d["kJ_per_mol"] = d["eV"] * 96.485
    print(f"{d['label']:>18s}  {d['quantity']:<13s} {d['eV']:8.3f} eV "
          f"{d['kJ_per_mol']:8.1f} kJ/mol")

write(OUTDIR / "relaxed.extxyz", relaxed, format="extxyz")
with open(OUTDIR / "energies.csv", "w", newline="") as fh:
    keys = sorted({k for r in results.values() for k in r})
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(results.values())
(OUTDIR / "derived.json").write_text(json.dumps(
    dict(model=MODEL, dispersion=DISPERSION, calc=CALC, derived=derived),
    indent=1))
print(f"wrote {OUTDIR}/relaxed.extxyz, energies.csv, derived.json")
if CALC == "toy":
    print("toy calculator: numbers above are meaningless (smoke test only)")
