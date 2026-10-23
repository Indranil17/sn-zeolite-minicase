# %% Export the studied systems as CIF files (and two trajectories) for visualisation
# Reads outputs/ (extxyz files of steps 00 to 04), writes structures/ and structures/index.csv.
#   1_built/          the starting models of step 00 (builder geometries, before relaxation)
#   2_relaxed/        the same models after relaxation, anneal and re-quench with MACE-MP-0 + D3 (step 01)
#   3_dynamics/       the water loaded cell: the start of the unbiased MD, one frame of each coordination
#                     state from the metadynamics replicas, and two trajectories (extxyz)
#   4_gipaw_snapshots/ the snapshots on which the 119Sn shifts were computed (shift in index.csv)
# Every structure is moved so that Sn sits at the centre of the cell, and written in P1 with labels:
# Sn, Si, O (framework), Oh/Hh (hydroxyl or silanol), Owb/Hwb (water within 2.8 A of Sn), Owp/Hwp (other water).
# The shifts are read from docs/results/data/ (same order as the snapshot folders).
# Run from the repository root:  python analysis/export_structures.py

from pathlib import Path
import csv
import json

import numpy as np
from ase import Atoms
from ase.io import read, write

# ----------------------------- CONFIG ------------------------------------
OUTPUTS = Path("outputs")
DATA = Path("docs/results/data")
OUT = Path("structures")
R0 = 2.6                    # A, switching function of the CV
BOUND_CUT = 2.8             # A, a water O closer than this to Sn is "bound"
SN_CN_CUT = 2.45            # A, the neighbour count of step 01
STEP_TRAJ = 10              # every 10th stored metadynamics frame (1 ps)
STEP_MD = 30                # every 30th stored MD frame
INCLUDE_FRAMEWORK = True    # the empty CHA cell from the IZA CIF (needs network or data/CHA.cif)
# --------------------------------------------------------------------------

for sub in ("1_built", "2_relaxed", "3_dynamics", "4_gipaw_snapshots"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
index = []


def sn_of(at):
    s = at.info.get("sn_index")
    return int(np.ravel(s)[0]) if s is not None else None


def recentre(at):
    """Copy with only cell, positions and symbols; Sn at the cell centre; atoms wrapped."""
    new = Atoms(at.get_chemical_symbols(), positions=at.get_positions(), cell=at.cell, pbc=True)
    s = sn_of(at)
    if s is not None:
        new.translate(new.cell.sum(axis=0) / 2.0 - new.positions[s])
    new.wrap()
    return new, s


def analyse(at, s):
    """Sn neighbours, water classification, CN of the water oxygens."""
    sym = np.array(at.get_chemical_symbols())
    n_h = {}
    h_idx = np.where(sym == "H")[0]
    for i in np.where(sym == "O")[0]:
        n_h[i] = int(np.sum(at.get_distances(i, h_idx, mic=True) < 1.25)) if len(h_idx) else 0
    info = dict(n_atoms=len(at), formula=at.get_chemical_formula())
    if s is None:
        return info, n_h, {}, set()
    d_sn = at.get_distances(s, range(len(at)), mic=True)
    water = [i for i, k in n_h.items() if k == 2]
    near = sorted(round(float(d_sn[i]), 2) for i in np.where(sym == "O")[0] if d_sn[i] < 2.7)
    d_w = sorted(float(d_sn[i]) for i in water)
    info.update(sn_neighbours_2p45=int(sum(d_sn[i] < SN_CN_CUT for i in np.where(sym == "O")[0])),
                sn_O_within_2p7=" ".join(f"{v:.2f}" for v in near),
                water_O_to_Sn=" ".join(f"{v:.2f}" for v in d_w[:3]),
                n_waters=len(water),
                CN=round(float(sum(1 / (1 + (d / R0) ** 6) for d in d_w)), 3) if water else "")
    return info, n_h, d_sn, {i for i in water if d_sn[i] < BOUND_CUT}


def label_atoms(at, s, n_h, d_sn, bound):
    sym = at.get_chemical_symbols()
    owner = {}                                           # H index -> O index
    for h in [i for i, x in enumerate(sym) if x == "H"]:
        o_all = [o for o in n_h if True]
        d = at.get_distances(h, o_all, mic=True)
        owner[h] = o_all[int(np.argmin(d))]
    count = {}
    labels = []
    for i, x in enumerate(sym):
        if x == "Sn":
            tag = "Sn"
        elif x == "Si":
            tag = "Si"
        elif x == "O":
            tag = "Owb" if i in bound else ("Owp" if n_h.get(i) == 2 else ("Oh" if n_h.get(i, 0) >= 1 else "O"))
        else:
            o = owner.get(i)
            tag = "Hwb" if o in bound else ("Hwp" if n_h.get(o) == 2 else "Hh")
        count[tag] = count.get(tag, 0) + 1
        labels.append(f"{tag}{count[tag]}")
    return labels


def write_cif(at, path, name, comment):
    at, s = recentre(at)
    info, n_h, d_sn, bound = analyse(at, s)
    labels = label_atoms(at, s, n_h, d_sn, bound)
    a, b, c, al, be, ga = at.cell.cellpar()
    frac = at.get_scaled_positions(wrap=True)
    lines = [f"data_{name}", f"# {comment}", f"_cell_length_a {a:.5f}", f"_cell_length_b {b:.5f}", f"_cell_length_c {c:.5f}",
             f"_cell_angle_alpha {al:.4f}", f"_cell_angle_beta {be:.4f}", f"_cell_angle_gamma {ga:.4f}",
             "_symmetry_space_group_name_H-M 'P 1'", "_symmetry_Int_Tables_number 1",
             f"_chemical_formula_sum '{at.get_chemical_formula(mode='hill', empirical=False)}'",
             "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'", "loop_", "_atom_site_label", "_atom_site_type_symbol",
             "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"]
    for lab, x, f in zip(labels, at.get_chemical_symbols(), frac):
        lines.append(f"{lab} {x} {f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
    Path(path).write_text("\n".join(lines) + "\n")
    return info


def add(path, desc, info, **extra):
    row = dict(file=str(path.relative_to(OUT)), description=desc)
    row.update(info)
    row.update(extra)
    index.append(row)


# ---- 1 and 2: step 00 models and the relaxed models of step 01
shifts_rel = json.loads((DATA / "relaxed_site_shifts.json").read_text())["states"]
for sub, src, tag in (("1_built", "00_structures/models.extxyz", "builder geometry"),
                      ("2_relaxed", "01_screen/relaxed.extxyz", "relaxed, MACE-MP-0 + D3")):
    for at in read(OUTPUTS / src, index=":"):
        lab = at.info["label"]
        info = write_cif(at, OUT / sub / f"{lab}.cif", f"{lab}_{sub[2:]}", f"Sn in CHA, {lab}, {tag}")
        add(OUT / sub / f"{lab}.cif", f"{lab}, {tag}", info,
            delta_119Sn_ppm=round(shifts_rel[lab]["delta"], 1) if (sub == "2_relaxed" and lab in shifts_rel) else "")
if INCLUDE_FRAMEWORK:
    try:
        from snmini import structures as S
        fw, _ = S.load_framework("CHA", "data", True)
        fw = S.to_primitive(fw)
        info = write_cif(fw, OUT / "1_built" / "CHA_framework.cif", "CHA_framework", "empty chabazite cell, pure silica, IZA structure")
        add(OUT / "1_built" / "CHA_framework.cif", "empty CHA cell, pure silica, IZA structure", info)
    except Exception as err:                                      # no network and no cached CIF
        print("framework skipped:", err)


# ---- 3: dynamics of the water loaded cell
def thin(frames, step, keys):
    out = []
    for at in frames[::step]:
        new, _ = recentre(at)
        new.info.update({k: float(np.ravel(at.info[k])[0]) for k in keys if k in at.info})
        out.append(new)
    return out


md = read(OUTPUTS / "02_md/md_frames.extxyz", index=":")
info = write_cif(md[0], OUT / "3_dynamics/md_cell_start.cif", "md_cell_start", "water loaded cell, first stored frame of the unbiased MD")
add(OUT / "3_dynamics/md_cell_start.cif", "water loaded cell (54 atoms), first frame of the unbiased MD", info)
info = write_cif(md[-1], OUT / "3_dynamics/md_cell_end_of_md.cif", "md_cell_end_of_md", "water loaded cell, last stored frame of the unbiased MD")
add(OUT / "3_dynamics/md_cell_end_of_md.cif", "water loaded cell, last frame of the unbiased MD (100 ps)", info)
write(OUT / "3_dynamics/md_unbiased.extxyz", thin(md, STEP_MD, ()), format="extxyz")
add(OUT / "3_dynamics/md_unbiased.extxyz", f"unbiased MD, every {STEP_MD}th stored frame, {len(md[::STEP_MD])} frames", {})

for rep, state, target, name in (("n5a", "4c", 0.40, "4_coordinate"), ("n5a", "5c", 1.10, "5_coordinate"), ("n6a", "6c", 1.55, "6_coordinate")):
    fr = [a for a in read(OUTPUTS / f"03_metad_{rep}/state_{state}.extxyz", index=":") if bool(np.ravel(a.info.get("intact", 1))[0])]
    best = min(fr, key=lambda a: abs(float(np.ravel(a.info["cv"])[0]) - target))
    p = OUT / f"3_dynamics/metad_example_{name}.cif"
    info = write_cif(best, p, f"metad_example_{name}", f"water loaded cell, metadynamics frame of the {name.replace('_', ' ')} state")
    add(p, f"metadynamics frame of the {name.replace('_', ' ')} state (CN {float(np.ravel(best.info['cv'])[0]):.2f}, "
           f"{float(np.ravel(best.info['time_ps'])[0]):.1f} ps)", info)
for rep, start in (("n5a", "5 coordinate"), ("n6a", "6 coordinate")):
    fr = read(OUTPUTS / f"03_metad_{rep}/frames.extxyz", index=":")
    write(OUT / f"3_dynamics/metad_{rep}_trajectory.extxyz", thin(fr, STEP_TRAJ, ("cv", "time_ps")), format="extxyz")
    add(OUT / f"3_dynamics/metad_{rep}_trajectory.extxyz",
        f"metadynamics run started from the {start} state, every {STEP_TRAJ}th stored frame (1 ps), CN and time in the frame header", {})

# ---- 4: snapshots of the GIPAW calculations
snap = json.loads((DATA / "snapshot_shifts_4each.json").read_text())["per_snapshot"]
d20 = np.loadtxt(DATA / "delta_119Sn_5c_20_snapshots.csv", delimiter=",", skiprows=2, usecols=1)
jobs = []
for i in range(20):
    jobs.append((OUTPUTS / f"04_snap5c_k2/5c_n20/frame_{i:05d}/snapshot.extxyz", f"5c_2x2x2_snapshot_{i + 1:02d}",
                 "5 coordinate snapshot, 2x2x2 k", float(d20[i])))
for i, s_ in enumerate(snap["cn1.10"]):
    jobs.append((OUTPUTS / f"04_snap/cn1.10/frame_{i:05d}/snapshot.extxyz", f"5c_3x3x3_snapshot_{i + 1}",
                 "5 coordinate snapshot, 3x3x3 k", s_["delta_iso"]))
for i, s_ in enumerate(snap["cn1.48"]):
    jobs.append((OUTPUTS / f"04_snap/cn1.48/frame_{i:05d}/snapshot.extxyz", f"6c_3x3x3_snapshot_{i + 1}",
                 "6 coordinate snapshot, 3x3x3 k", s_["delta_iso"]))
for path, name, desc, d in jobs:
    at = read(path)
    info = write_cif(at, OUT / f"4_gipaw_snapshots/{name}.cif", name, f"water loaded cell, {desc}")
    add(OUT / f"4_gipaw_snapshots/{name}.cif", desc, info, delta_119Sn_ppm=round(d, 1),
        replica=str(at.info.get("replica", "")).replace("03_metad_", ""))

cols = ["file", "description", "n_atoms", "formula", "sn_neighbours_2p45", "sn_O_within_2p7", "n_waters", "water_O_to_Sn", "CN",
        "delta_119Sn_ppm", "replica"]
with open(OUT / "index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(index)
print(f"wrote {len(index)} entries to {OUT / 'index.csv'}")
