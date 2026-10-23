# Structures: what was studied and what happened

CIF files (P1, periodic) of every system of the study, for viewing in VESTA,
Mercury, Jmol, Avogadro or ASE (`ase gui file.cif`). Two trajectories are extxyz
files (OVITO, VMD, `ase gui`). `index.csv` lists every file with its Sn
neighbours, water distances, CN and, where computed, the ¹¹⁹Sn shift.

All structures are moved so that Sn sits at the centre of the cell. Atom labels
tell the chemistry: `Sn`, `Si`, `O` (framework oxygen), `Oh` and `Hh` (hydroxyl or
silanol), `Owb` and `Hwb` (water within 2.8 Å of Sn), `Owp` and `Hwp` (other
water). In a viewer, please set the Sn–O bond cutoff to 2.7 Å to see the coordination.
The default can mislead: VESTA proposes 2.82 Å for Sn–O (Edit, Bonds), which is
longer than any Sn–O distance in the table below and can draw oxygen that is not
bound to Sn, and other viewers pick a cutoff from covalent radii. At 2.7 Å the
four framework bonds (1.93 to 2.02 Å) and the fifth and sixth neighbours (2.10 to
2.60 Å) are drawn. The Sn coordination number in the code counts only oxygen
closer than 2.45 Å, so a water at 2.50 Å is drawn as a bond but not counted;
`index.csv` has the distances. The cell is the primitive rhombohedral cell of chabazite (about 9.3 Å, 94°): 36
atoms for the Sn site models, 54 for the water loaded cell.

| folder | content |
|---|---|
| `1_built/` | the models of step 00 as built, before relaxation: `T1_closed`, `T1_open1` to `T1_open4`, `T1_closed_1w`, `T1_closed_2w`, `md_cell` (the cell used for the dynamics), a water molecule in a box, and the empty chabazite cell (`CHA_framework`) |
| `2_relaxed/` | the same models after relaxation, anneal and re-quench with MACE-MP-0 + D3 (step 01) |
| `3_dynamics/` | `md_cell_start` and `md_cell_end_of_md` (first and last stored frame of the 100 ps unbiased MD); `metad_example_4_coordinate`, `_5_coordinate`, `_6_coordinate` (one metadynamics frame of each state); `md_unbiased.extxyz` (61 frames); `metad_n5a_trajectory.extxyz` and `metad_n6a_trajectory.extxyz` (226 frames each, 1 ps apart, CN and time in the frame header; the run n5a starts from the 5 coordinate state, n6a from the 6 coordinate state) |
| `4_gipaw_snapshots/` | the 28 snapshots on which ¹¹⁹Sn shifts were computed: `5c_2x2x2_snapshot_01` to `_20`, `5c_3x3x3_snapshot_1` to `_4`, `6c_3x3x3_snapshot_1` to `_4` (shifts in `index.csv`) |

## What happened

**Sites after relaxation** (Sn–O distances in Å within 2.7 Å of Sn, from `index.csv`;
shifts are the PBE values for the relaxed geometry):

| model | as built | after relaxation | δ(¹¹⁹Sn) / ppm |
|---|---|---|---|
| closed | 1.95 × 4 | 1.93, 1.93, 1.95, 1.96 | −422.5 |
| open 1 | 1.95 × 4, 2.14 | 1.94, 1.97, 1.97, 2.00, 2.38 | −515.4 |
| open 2 | 1.95 × 4, 2.27 | 1.97, 1.98, 1.99, 2.02, 2.15 | −508.4 |
| open 3 | 1.95 × 4, 2.13 | 1.97, 1.99, 2.01, 2.02, 2.18 | −514.3 |
| open 4 | 1.95 × 4, 2.24 | 1.96, 1.96, 1.97, 2.08, 2.10 | −544.0 |
| closed + 1 water | 1.95 × 4, 2.25 | 1.95, 1.95, 1.96, 1.96, 2.50 | −519.8 |
| closed + 2 waters | 1.95 × 4, 2.25, 2.25 | 1.97, 1.97, 1.98, 1.99, 2.42, 2.60 | −631.4 |

* **The open sites did not open.** The silanol oxygen (the fifth Sn–O distance)
  stays 2.10 to 2.38 Å from Sn; an open site would have it beyond 3 Å. Open 1 is
  borderline at 2.38 Å. This is the result for MACE-MP-0 + D3 and a 2 ps anneal,
  not a statement about Sn-Beta.
* **Water on the closed site moves away.** With one water the Sn–O(water) distance
  grows from 2.25 to 2.50 Å; with two waters, from 2.25 and 2.25 to 2.42 and
  2.60 Å.
* **In the water loaded cell only one water stays bound.** `md_cell` starts with
  two bound waters at 2.25 Å. After relaxation one is at 2.12 Å and the others are
  at 3.62 Å or more. In the unbiased MD the nearest water is at 2.06 Å in the
  first and 2.23 Å in the last stored frame, with no exchange in 100 ps.
* **Metadynamics visits the other states.** The three example frames have the
  nearest water at 3.64 Å (4 coordinate, CN 0.40), 2.22 Å (5 coordinate, CN 1.10),
  and 2.18 and 2.63 Å (6 coordinate, CN 1.55). In the two trajectories the number
  of waters within 2.8 Å of Sn changes 58 and 61 times between consecutive frames
  of the file; the runs are biased, so the share of time in a state is not its
  population.
* **The second water of the 6 coordinate state is soft.** In the four 6 coordinate
  snapshots the second water is at 2.51, 2.31, 2.34 and 2.35 Å; in the 5
  coordinate snapshots the single water is at 2.05 to 2.28 Å.
* **The shift follows the water count.** The 5 coordinate snapshots give −578.5 ppm
  on average (20 snapshots, from −642.6 to −536.5), the 6 coordinate ones about
  −706 ppm (four snapshots, from −758.3 to −683.3). `index.csv` lists the value of
  each snapshot.

The geometries in `1_built/` are starting guesses (Sn–O 1.95 Å, Sn–O(water)
2.25 Å); only the geometries in `2_relaxed/` and `3_dynamics/` come from the
potential. `3_dynamics/` and `4_gipaw_snapshots/` come from MACE-MP-0 + D3 at
350 K. `analysis/export_structures.py` rewrites these files from the run outputs.
