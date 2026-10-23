# %% 04  Write GIPAW NMR inputs: SnO2 reference + snapshots of each Sn state
# Default: Quantum ESPRESSO with QE-GIPAW (free, open source): pw.x SCF, then
# gipaw.x for the shielding tensors. CASTEP and VASP (licensed) remain as
# options. Nothing is run here; each calculation gets its own folder and a
# job.sh that hpc/cpu_gipaw.pbs runs.

from pathlib import Path
import json
import math
import warnings

import numpy as np
from ase.io import read, write
from ase.spacegroup import crystal

from snmini import qe

# ----------------------------- CONFIG ------------------------------------
CODE = "qe"                # "qe" (default), or the licensed "castep" / "vasp"
STATE_FILES = sorted(Path("outputs/03_metad").glob("state_*.extxyz"))
N_PER_STATE = 10           # snapshots per state, evenly spaced in time
K_DENSITY = 20.0           # k points per direction = ceil(K_DENSITY / a_i)
OUTDIR = Path("outputs/04_gipaw")
# Quantum ESPRESSO: pseudopotentials WITH GIPAW data (04a_gipaw_pseudos.py).
# The path is written into every input as an absolute path, so run this
# script where the jobs will run (the cluster), or type the cluster path.
QE_PSEUDO_DIR = Path("pseudo_gipaw").resolve()
# the pslibrary 1.0.0 files from the QE library already carry GIPAW data (04a checks)
QE_PSEUDOS = {"Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
              "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
              "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
              "H": "H.pbe-kjpaw_psl.1.0.0.UPF"}
QE_ECUTWFC = 70            # Ry; NMR needs more than the UPF header suggests
QE_ECUTRHO = 560           # Ry; 8 x ecutwfc for PAW. Converge both on SnO2
# CASTEP or VASP, only if you hold a licence:
CUTOFF_EV = 700            # eV; converge (the SnO2 reference study used 870 eV)
CASTEP_ZORA = True         # scalar relativistic ZORA, on the fly pseudopotentials
VASP_LZORA = False         # ZORA shielding in VASP needs version 6.6.0 or later
VASP_POTCARS = "Sn_d, Si, O, H (your licensed set; never commit POTCARs)"
# --------------------------------------------------------------------------

# ASE warns that it cannot validate CASTEP keywords without a CASTEP binary
warnings.filterwarnings("ignore", message=".*CASTEP.*")


def sno2():
    """Cassiterite (rutile type SnO2), experimental cell, P4_2/mnm."""
    return crystal(["Sn", "O"], basis=[(0, 0, 0), (0.307, 0.307, 0)],
                   spacegroup=136, cellpar=[4.737, 4.737, 3.186, 90, 90, 90])


def kgrid(atoms):
    return [max(1, math.ceil(K_DENSITY / L)) for L in atoms.cell.lengths()]


def check_qe_pseudos():
    """Stop if a pseudopotential is missing or lacks GIPAW data. Run this
    script where the jobs run, so that a wrong file name fails here and not
    in every pw.x job."""
    missing = [n for n in QE_PSEUDOS.values() if not (QE_PSEUDO_DIR / n).exists()]
    if missing:
        raise SystemExit(f"not found in {QE_PSEUDO_DIR}: {missing}; copy the "
                         "QE_PSEUDOS dictionary printed by 04a_gipaw_pseudos.py")
    bad = [n for n in QE_PSEUDOS.values()
           if not qe.upf_gipaw_ready(QE_PSEUDO_DIR / n)]
    if bad:
        raise SystemExit(f"no GIPAW data in {bad}: run 04a_gipaw_pseudos.py "
                         "and use the regenerated files")


def write_qe(atoms, folder, name):
    used = {el: QE_PSEUDOS[el] for el in sorted(set(atoms.get_chemical_symbols()))}
    qe.write_pw_scf(atoms, folder, name, QE_PSEUDO_DIR, used,
                    QE_ECUTWFC, QE_ECUTRHO, kgrid(atoms))
    qe.write_gipaw_nmr(folder, name)
    # One MPI rank per core the scheduler gave us (PBS_NCPUS), 1 if run by hand.
    # -nd 1: serial dense linear algebra. With the default (ScaLAPACK or ELPA)
    # gipaw.x stopped with "cdiaghg: problems computing cholesky" on the 36 to
    # 42 atom cells, in every parallel setting tried; -nd 1 ran through.
    (folder / "job.sh").write_text(
        "#!/bin/bash\n# pw.x and gipaw.x must come from the same QE version\n"
        "NP=${PBS_NCPUS:-1}\n"
        "mpirun -np $NP pw.x -nd 1 -in scf.in > scf.out || exit 1\n"
        "mpirun -np $NP gipaw.x -nd 1 -in nmr.in > nmr.out || exit 1\n")


def started(folder):
    """True if a job has already run in this folder (never rewrite it)."""
    return any((folder / f).exists()
               for f in ("scf.out", "nmr.out", "SnO2.castep", "snap.castep",
                         "OUTCAR"))


def write_castep(atoms, folder, name):
    write(folder / f"{name}.cell", atoms, format="castep-cell")
    k = kgrid(atoms)
    with open(folder / f"{name}.cell", "a") as fh:
        fh.write(f"\nKPOINTS_MP_GRID : {k[0]} {k[1]} {k[2]}\n")
    rel = "RELATIVISTIC_TREATMENT : ZORA\n" if CASTEP_ZORA else ""
    (folder / f"{name}.param").write_text(
        "TASK            : MAGRES\n"
        "MAGRES_TASK     : SHIELDING\n"
        "XC_FUNCTIONAL   : PBE\n"
        f"CUT_OFF_ENERGY  : {CUTOFF_EV} eV\n"
        "FIX_OCCUPANCY   : TRUE\n"
        "ELEC_ENERGY_TOL : 1.0e-9 eV\n"
        "OPT_STRATEGY    : SPEED\n" + rel)
    (folder / "job.sh").write_text(
        "#!/bin/bash\n# adapt to your scheduler, see hpc/\n"
        f"mpirun castep.mpi {name}\n")


def write_vasp(atoms, folder, name):
    order = np.argsort(atoms.get_chemical_symbols(), kind="stable")
    write(folder / "POSCAR", atoms[order], format="vasp", direct=True)
    (folder / "order.json").write_text(json.dumps(order.tolist()))
    k = kgrid(atoms)
    (folder / "KPOINTS").write_text(f"auto\n0\nGamma\n{k[0]} {k[1]} {k[2]}\n")
    (folder / "INCAR").write_text(
        "PREC = Accurate\n"
        f"ENCUT = {CUTOFF_EV}\n"
        "EDIFF = 1E-8\n"
        "ISMEAR = 0 ; SIGMA = 0.01\n"
        "LASPH = .TRUE.\n"
        "LREAL = .FALSE.\n"
        "LCHIMAG = .TRUE.\n"
        "LNMR_SYM_RED = .TRUE.\n"
        "NLSPLINE = .TRUE.\n"
        + ("LZORA = .TRUE.\n" if VASP_LZORA else "")
        + "LWAVE = .FALSE. ; LCHARG = .FALSE.\n")
    (folder / "job.sh").write_text(
        "#!/bin/bash\n# adapt to your scheduler, see hpc/\n"
        f"# POTCAR: {VASP_POTCARS}\nmpirun vasp_std\n")


writer = {"qe": write_qe, "castep": write_castep, "vasp": write_vasp}[CODE]
if CODE == "qe":
    check_qe_pseudos()
OUTDIR.mkdir(parents=True, exist_ok=True)
manifest = []

ref = OUTDIR / "reference_SnO2"
ref.mkdir(exist_ok=True)
if started(ref):
    print(f"kept, has output already: {ref}")
else:
    writer(sno2(), ref, "SnO2")
manifest.append(dict(state="reference", folder=str(ref), weight=1.0))

for f in STATE_FILES:
    state = f.stem.replace("state_", "")
    frames = read(f, index=":")
    pick = np.unique(np.linspace(0, len(frames) - 1,
                                 min(N_PER_STATE, len(frames))).astype(int))
    for k in pick:
        a = frames[k]
        folder = OUTDIR / state / f"frame_{k:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        if started(folder):
            print(f"kept, has output already: {folder}")
        else:
            writer(a, folder, "snap")
            write(folder / "snapshot.extxyz", a, format="extxyz")
        manifest.append(dict(state=state, folder=str(folder),
                             weight=float(a.info.get("weight", 1.0)),
                             sn_index=int(a.info["sn_index"]),
                             cv=float(a.info.get("cv", np.nan))))
    print(f"{state}: {len(pick)} snapshots")

settings = dict(code=CODE, k_density=K_DENSITY)
if CODE == "qe":
    settings.update(ecutwfc_Ry=QE_ECUTWFC, ecutrho_Ry=QE_ECUTRHO,
                    pseudopotentials=QE_PSEUDOS)
else:
    settings.update(cutoff_eV=CUTOFF_EV,
                    zora=CASTEP_ZORA if CODE == "castep" else VASP_LZORA)
(OUTDIR / "manifest.json").write_text(json.dumps(
    dict(**settings, calculations=manifest), indent=1))
print(f"{len(manifest)} calculations written under {OUTDIR}")
print("First converge the cutoff and k points on SnO2 and one snapshot "
      "(target: 119Sn shielding stable to about 1 ppm).")
