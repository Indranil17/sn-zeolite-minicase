# %% Check the foundation model screen (step 01) against DFT single points
# Three phases, chosen in CONFIG, run from the repository root:
#   "write"   DFT single point inputs at the geometries that 01 relaxed:
#             Quantum ESPRESSO pw.x by default (free, open source); CASTEP or
#             VASP only if you hold a licence. No MACE needed.
#   "mace"    MACE energies at the same geometries WITHOUT D3 (GPU or CPU).
#   "compare" read the DFT energies once the jobs have finished and rebuild
#             the relative energies of 01 for both methods, side by side.
# Why without D3: MACE-MP-0 learned PBE energies (Materials Project; the +U
# correction there concerns some transition metal oxides, not Sn, Si, O or H),
# and 01 adds D3(BJ) on top. The D3 term is the same add-on for DFT and for
# MACE, so PBE against MACE alone tests what the foundation model learned.
# Runs as one Jupyter cell or as `python dft_check.py` (snmini installed).

from pathlib import Path
import json
import math
import warnings

import numpy as np
from ase.io import read, write

# ----------------------------- CONFIG ------------------------------------
PHASE = "write"            # "write", "mace" or "compare"
CODE = "qe"                # "qe" (default), or the licensed "castep" / "vasp"
RELAXED = Path("outputs/01_screen/relaxed.extxyz")
DERIVED_01 = Path("outputs/01_screen/derived.json")    # MACE + D3, from 01
SKIP = ["md_cell"]         # not needed for the relative energies, and large
CALC, MODEL, DEVICE, DTYPE = "mace", "medium", "cuda", "float64"   # "cpu" on a CPU node
# The pslibrary files downloaded for step 04a, used as they are (GIPAW data is
# not needed for energies). Absolute path as the jobs see it: run on the cluster.
QE_PSEUDO_DIR = str(Path("pseudo_src").resolve())
QE_PSEUDOS = {"Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
              "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
              "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
              "H": "H.pbe-kjpaw_psl.1.0.0.UPF"}
QE_ECUTWFC, QE_ECUTRHO = 60, 480                 # Ry; check the relative energies
VASP_ENCUT = 520           # eV, the Materials Project setting behind MACE-MP-0
CASTEP_CUTOFF = 700        # eV, on the fly pseudopotentials need more; converge
K_DENSITY = 20.0           # k points per direction = ceil(K_DENSITY / a_i)
FLAG_KJ = 10.0             # flag MACE vs DFT differences above this (kJ/mol)
OUTDIR = Path("outputs/01b_dft_check")
# --------------------------------------------------------------------------

EV_TO_KJ = 96.485
# ASE warns that it cannot validate CASTEP keywords without a CASTEP binary
warnings.filterwarnings("ignore", message=".*CASTEP.*")


def kgrid(atoms, label):
    if label == "H2O_gas":                 # molecule in a box: Gamma only
        return [1, 1, 1]
    return [max(1, math.ceil(K_DENSITY / L)) for L in atoms.cell.lengths()]


def write_qe(atoms, folder, label):
    from snmini import qe
    used = {el: QE_PSEUDOS[el] for el in set(atoms.get_chemical_symbols())}
    qe.write_pw_scf(atoms, folder, label, QE_PSEUDO_DIR, used, QE_ECUTWFC,
                    QE_ECUTRHO, kgrid(atoms, label))
    (folder / "job.sh").write_text(
        "#!/bin/bash\nNP=${PBS_NCPUS:-1}\n"
        "mpirun -np $NP pw.x -in scf.in > scf.out || exit 1\n")


def write_vasp(atoms, folder, label):
    order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
    write(folder / "POSCAR", atoms[order], format="vasp", direct=True)
    k = kgrid(atoms, label)
    (folder / "KPOINTS").write_text(f"auto\n0\nGamma\n{k[0]} {k[1]} {k[2]}\n")
    (folder / "INCAR").write_text(
        f"SYSTEM = {label}\n"
        "PREC = Accurate\n"
        f"ENCUT = {VASP_ENCUT}\n"
        "EDIFF = 1E-6\n"
        "ISMEAR = 0 ; SIGMA = 0.05\n"
        "GGA = PE\n"
        "LASPH = .TRUE.\n"
        "LREAL = .FALSE.\n"
        "NSW = 0 ; IBRION = -1\n"
        "LWAVE = .FALSE. ; LCHARG = .FALSE.\n")
    species = list(dict.fromkeys(atoms[order].get_chemical_symbols()))
    (folder / "job.sh").write_text(
        "#!/bin/bash\n# adapt to your scheduler\n"
        f"# POTCAR: concatenate in this order: {' '.join(species)}"
        + (" (Sn_d for Sn)" if "Sn" in species else "") + "\n"
        "mpirun vasp_std\n")


def write_castep(atoms, folder, label):
    write(folder / "single.cell", atoms, format="castep-cell")
    k = kgrid(atoms, label)
    with open(folder / "single.cell", "a") as fh:
        fh.write(f"\nKPOINTS_MP_GRID : {k[0]} {k[1]} {k[2]}\n")
    (folder / "single.param").write_text(
        "TASK            : SINGLEPOINT\n"
        "XC_FUNCTIONAL   : PBE\n"
        f"CUT_OFF_ENERGY  : {CASTEP_CUTOFF} eV\n"
        "FIX_OCCUPANCY   : TRUE\n"
        "ELEC_ENERGY_TOL : 1.0e-7 eV\n"
        "OPT_STRATEGY    : SPEED\n")
    (folder / "job.sh").write_text(
        "#!/bin/bash\n# adapt to your scheduler\nmpirun castep.mpi single\n")


def output_file(folder):
    """The DFT output file of a folder if the job has started, else None."""
    name = {"qe": "scf.out", "vasp": "OUTCAR", "castep": "single.castep"}[CODE]
    f = Path(folder) / name
    return f if f.exists() else None


def dft_energy(folder):
    """Total energy in eV from a finished single point, or None."""
    f = output_file(folder)
    if f is None:
        return None
    try:
        fmt = {"qe": "espresso-out", "vasp": "vasp-out",
               "castep": "castep-castep"}[CODE]
        return float(read(f, format=fmt).get_potential_energy())
    except Exception as err:               # unfinished or crashed run
        print(f"could not read an energy from {f} ({type(err).__name__}: {err})")
        return None


def derived(energies):
    """Same definitions as 01_screen_foundation.py (eV)."""
    e_w = energies["H2O_gas"]
    out = {}
    for label, e in energies.items():
        if "_open" in label:
            t = label.split("_")[0]
            out[f"{label} E_hydrolysis"] = e - energies[f"{t}_closed"] - e_w
        if label.endswith("w") and "_closed_" in label:
            t, n = label.split("_")[0], int(label.split("_")[-1][:-1])
            out[f"{label} E_bind_{n}w"] = (e - energies[f"{t}_closed"] - n * e_w) / n
    return out


models = [m for m in read(RELAXED, index=":") if m.info["label"] not in SKIP]
OUTDIR.mkdir(parents=True, exist_ok=True)

if PHASE == "write":
    writer = {"qe": write_qe, "vasp": write_vasp, "castep": write_castep}[CODE]
    n = 0
    for m in models:
        folder = OUTDIR / CODE / m.info["label"]
        if output_file(folder) is not None:    # never touch a started run
            print(f"kept, has output already: {folder}")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        writer(m, folder, m.info["label"])
        write(folder / "geometry.extxyz", m, format="extxyz")
        n += 1
    print(f"{n} single points written under {OUTDIR / CODE}")

elif PHASE == "mace":
    from snmini.calc import get_calculator
    calc = get_calculator(CALC, MODEL, False, DEVICE, DTYPE)   # no D3 here
    energies = {}
    for m in models:
        m.calc = calc
        energies[m.info["label"]] = float(m.get_potential_energy())
        print(f"{m.info['label']:>14s} {energies[m.info['label']]:14.6f} eV")
    (OUTDIR / "mace_noD3.json").write_text(json.dumps(
        dict(model=MODEL, calc=CALC, energies=energies), indent=1))
    print(f"wrote {OUTDIR / 'mace_noD3.json'}")

elif PHASE == "compare":
    e_dft = {m.info["label"]: dft_energy(OUTDIR / CODE / m.info["label"])
             for m in models}
    missing = [k for k, v in e_dft.items() if v is None]
    if missing:
        raise SystemExit(f"DFT not finished for: {missing}")
    e_mace = json.loads((OUTDIR / "mace_noD3.json").read_text())["energies"]
    d_dft, d_mace = derived(e_dft), derived(e_mace)
    d_d3 = {}
    if DERIVED_01.exists():                # MACE + D3 numbers from 01
        for d in json.loads(DERIVED_01.read_text())["derived"]:
            d_d3[f"{d['label']} {d['quantity']}"] = d["eV"]
    rows = []
    print(f"{'quantity':<32s}{'MACE+D3':>10s}{'MACE':>10s}{'PBE':>10s}"
          f"{'MACE-PBE':>10s}   kJ/mol")
    for key in d_dft:
        diff = (d_mace[key] - d_dft[key]) * EV_TO_KJ
        row = dict(quantity=key, mace_d3=d_d3.get(key, float("nan")) * EV_TO_KJ,
                   mace=d_mace[key] * EV_TO_KJ, pbe=d_dft[key] * EV_TO_KJ,
                   mace_minus_pbe=diff, flag=abs(diff) > FLAG_KJ)
        rows.append(row)
        print(f"{key:<32s}{row['mace_d3']:10.1f}{row['mace']:10.1f}"
              f"{row['pbe']:10.1f}{diff:10.1f}" + ("   <-- check" if row["flag"] else ""))
    mae = float(np.mean([abs(r["mace_minus_pbe"]) for r in rows]))
    print(f"mean absolute MACE-PBE difference: {mae:.1f} kJ/mol")
    (OUTDIR / "comparison.json").write_text(json.dumps(
        dict(code=CODE, model=MODEL, mae_kJ_per_mol=mae, rows=rows), indent=1))
    print(f"wrote {OUTDIR / 'comparison.json'}")

else:
    raise ValueError(f"unknown PHASE {PHASE!r}")
