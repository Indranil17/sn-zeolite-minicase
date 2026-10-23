# %% 04a  Pseudopotentials with GIPAW data, for Quantum ESPRESSO (free)
# gipaw.x needs pseudopotentials that carry reconstruction data. This script
# checks each file; for a pslibrary file without that data it writes the
# ld1.x input that regenerates it as a PAW dataset with GIPAW data (the input
# stored inside the file, plus lgipaw_reconstruction and use_paw_as_gipaw),
# and runs ld1.x if it is on the PATH. Nothing existing is overwritten.
#
# Source files: QE pseudopotential library, pslibrary 1.0.0, PAW, scalar
# relativistic, for example
#   https://pseudopotentials.quantum-espresso.org/upf_files/Sn.pbe-dn-kjpaw_psl.1.0.0.UPF
# Use the ld1.x of the QE version you build QE-GIPAW against.

from pathlib import Path
import shutil
import subprocess

from snmini import qe

# ----------------------------- CONFIG ------------------------------------
SOURCE_DIR = Path("pseudo_src")          # the downloaded UPF files
SOURCES = {"Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
           "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
           "O": "O.pbe-n-kjpaw_psl.0.1.UPF",     # SSSP; carries GIPAW data already
           "H": "H.pbe-kjpaw_psl.1.0.0.UPF"}
OUTDIR = Path("pseudo_gipaw")            # becomes QE_PSEUDO_DIR in 04
LD1 = "ld1.x"
RUN_LD1 = True                           # False: only write the ld1.x inputs
# --------------------------------------------------------------------------

OUTDIR.mkdir(exist_ok=True)
ld1 = shutil.which(LD1) if RUN_LD1 else None
final = {}
for el, name in SOURCES.items():
    src = SOURCE_DIR / name
    if not src.exists():
        print(f"{el}: {src} is missing; download it first")
        continue
    if qe.upf_gipaw_ready(src):                       # use as it is
        note = "already in"
        if not (OUTDIR / name).exists():
            shutil.copy2(src, OUTDIR / name)
            note = "copied to"
        final[el] = name
        print(f"{el}: {name} has GIPAW data; {note} {OUTDIR}")
        continue

    new = name.replace("rrkjus", "kjpaw").replace(".UPF", "_gipaw.UPF")
    if (OUTDIR / new).exists() and qe.upf_gipaw_ready(OUTDIR / new):
        final[el] = new
        print(f"{el}: {new} exists already, with GIPAW data; kept")
        continue
    try:
        text = qe.ld1_input_with_gipaw(src, new)
    except ValueError as err:                         # not a pslibrary file
        print(f"{el}: cannot regenerate ({err}); use a pslibrary PAW file")
        continue
    inp = OUTDIR / new.replace(".UPF", ".in")
    inp.write_text(text)
    if ld1 is None:
        print(f"{el}: wrote {inp}; run:  cd {OUTDIR} && {LD1} < {inp.name} "
              f"> {inp.stem}.out")
        continue
    with open(inp) as fin, open(OUTDIR / f"{inp.stem}.out", "w") as fout:
        run = subprocess.run([ld1], stdin=fin, stdout=fout,
                             stderr=subprocess.STDOUT, cwd=OUTDIR)
    if run.returncode == 0 and (OUTDIR / new).exists() \
            and qe.upf_gipaw_ready(OUTDIR / new):
        final[el] = new
        print(f"{el}: {new} written, GIPAW data present")
    else:
        print(f"{el}: ld1.x failed (exit {run.returncode}); "
              f"see {OUTDIR / (inp.stem + '.out')}")

print("\nQE_PSEUDOS for 04_gipaw_inputs.py:")
print(final)
if len(final) < len(SOURCES):
    print("Some elements are still missing a GIPAW ready file (see above).")
