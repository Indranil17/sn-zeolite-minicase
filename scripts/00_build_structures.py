# %% 00  Build the Sn site library: closed, open and hydrated Sn sites
# Runs as one Jupyter cell or as `python scripts/00_build_structures.py`.
# Geometry only. Every structure written here is a starting guess that
# 01_screen_foundation.py relaxes (and anneals) before any energy is used.

from pathlib import Path
import json

from ase.io import write
from snmini import structures as S

# ----------------------------- CONFIG ------------------------------------
FRAMEWORK = "CHA"          # IZA code; CHA has one T site, a complete miniature
CACHE_DIR = "data"         # IZA CIF is cached here (or put CHA.cif here yourself)
ALLOW_DOWNLOAD = True      # False: use the cached CIF or the idealised fallback
CELL = "primitive"         # "primitive" (CHA: 36 atoms) or "conventional" (108)
BOUND_WATERS = [1, 2]      # hydrated closed sites: 5 and 6 coordinate Sn
FREE_WATERS_MD = 4         # extra pore water in the cell used for dynamics
SEED = 0
OUTDIR = Path("outputs/00_structures")
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
fw, source = S.load_framework(FRAMEWORK, CACHE_DIR, ALLOW_DOWNLOAD)
print(f"framework {FRAMEWORK}: {fw.get_chemical_formula()}  [{source}]")
if CELL == "primitive":
    fw = S.to_primitive(fw)
si = [a.index for a in fw if a.symbol == "Si"]
print("cell:", fw.get_chemical_formula(), fw.cell.cellpar().round(3))
print("coordination sequence of T1:", S.coordination_sequence(fw, si[0], 5))

models = []
t_classes = S.symmetry_classes(fw, si)
print(f"{len(t_classes)} symmetry distinct T site(s)")
for ti, tcls in enumerate(t_classes, start=1):
    closed = S.substitute_sn(fw, tcls[0])
    closed.info.update(label=f"T{ti}_closed", t_site=ti, source=source)
    models.append(closed)
    sn = closed.info["sn_index"]

    # open sites: one per symmetry distinct Sn-O-Si bridge
    bridges = S.bonded(closed, sn, ("O",), cutoff=2.2)
    for bi, bcls in enumerate(S.symmetry_classes(closed, bridges), start=1):
        op = S.make_open_site(closed, sn, bcls[0])
        op.info.update(label=f"T{ti}_open{bi}", t_site=ti, source=source,
                       multiplicity=len(bcls))
        models.append(op)

    # hydrated closed sites
    for n in BOUND_WATERS:
        hyd = S.add_bound_water(closed, n)
        hyd.info.update(label=f"T{ti}_closed_{n}w", t_site=ti, source=source)
        models.append(hyd)

# one water loaded cell for dynamics (first T site, two bound waters + pore water)
md = S.add_free_waters(S.add_bound_water(models[0], 2), FREE_WATERS_MD,
                       seed=SEED)
md.info.update(label="md_cell", t_site=1, source=source)
models.append(md)

# reference molecule for hydrolysis and hydration energies
from ase.build import molecule
h2o = molecule("H2O")
h2o.center(vacuum=7.5)
h2o.pbc = True
h2o.info.update(label="H2O_gas")
models.append(h2o)

rows = []
for m in models:
    rep = S.audit(m) if "sn_index" in m.info else dict(
        formula=m.get_chemical_formula(), n_atoms=len(m))
    rep["label"] = m.info["label"]
    rows.append(rep)
    print(json.dumps(rep, default=float))
write(OUTDIR / "models.extxyz", models, format="extxyz")
(OUTDIR / "audit.json").write_text(json.dumps(rows, indent=1, default=float))
print(f"wrote {len(models)} models to {OUTDIR/'models.extxyz'}")
if "fallback" in source:
    print("NOTE: idealised fallback framework (linear T-O-T bridges). "
          "Relax with cell (01_screen_foundation.py, RELAX_CELL=True).")
