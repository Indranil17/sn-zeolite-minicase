# Run log

Runs on a PBS Pro cluster: one NVIDIA L40S GPU per MD or metadynamics job, 8 to
16 CPU cores per Quantum ESPRESSO job. MACE-MP-0 medium with D3(BJ); Quantum
ESPRESSO 7.5 (`pw.x`, `gipaw.x`),
PBE, pslibrary 1.0.0 and SSSP PAW files. Every number in the README comes from
this log and from `docs/results/data/`. Each result carries its condition; read
them together.

## 1. Tests and engine check

* 34 tests pass (31 plus 3 for umbrella sampling), about 25 s; again on a second
  machine (Windows 11, Python 3.12, NumPy 2.5), 14 s.
* `validate_metad_doublewell.py`, 26 s: F(x) rmse 0.08 to 0.18 kT, right well
  population 0.122 to 0.136 (exact 0.130) at 1.6e6 steps (three seeds).
* Repeated with 15 seeds (`analysis/double_well_repeats.py`,
  `docs/validation_doublewell_repeats.json`), 1.6e6 steps each: F(x) rmse 0.09 to
  0.22 kT; error of the right well population SD 0.019, largest 0.041, within
  0.01 in 8 of 15 runs; error of the free energy difference between the wells SD
  0.16 kT (0.5 kJ/mol at 350 K), largest 0.32 kT (0.9 kJ/mol). The three seeds of
  the first run give other numbers on another machine (the dynamics is chaotic),
  so the population is not quoted more tightly than this scatter.

## 2. Step 00: site library

* Framework from the IZA CIF (R-3m, DLS optimised). Nine models: `T1_closed`,
  `T1_open1` to `T1_open4`, `T1_closed_1w`, `T1_closed_2w`, `md_cell`, `H2O_gas`.
* Audit: Sn coordination 4 (closed), 5 (the four open variants before relaxation,
  and one water), 6 (two waters and `md_cell`); shortest heavy atom distance
  1.61 A (Si–O) everywhere; shortest H contact 1.72 to 2.50 A.
* The CIF coordinates carry about 1e-3 A of rounding. With `symprec = 1e-3`
  spglib saw R3 for the primitive cell and split CHA into 4 T sites (30 models).
  With 2e-3 to 2e-2 it sees R-3m, one T site, four O sites. Class detection now
  uses 5e-3 (`structures.symmetry_classes`). The tests did not catch it because
  they use the idealised fallback framework.

## 3. Step 01: foundation model screen

MACE-MP-0 medium + D3(BJ), float64, fmax 0.02 eV/A, relaxed cell, 3 anneal
(400 K, 2 ps) and re-quench cycles per model, 81 min on an L40S. Every cycle is
recorded in `energies.csv`. A first attempt failed at the first MACE call: the
CUDA 13 build of PyTorch has no kernels for the GTX 1080 Ti (sm_61) the job
landed on. The job script now asks for `gpu_mem=20gb`.

| model | sn_cn (2.45 A) | Sn–O(silanol) (A) | note |
|---|---|---|---|
| T1_open1 | 5 | 2.38 | does not open by the 2.45 A count in any cycle, but borderline: Sn–O 1.94, 1.97, 1.97, 2.00, 2.38 A |
| T1_open2 | 5 | 2.15 | does not open; bridging O protonated, still bonded |
| T1_open3 | 5 | 2.18 | does not open; lowest of the four (anneal found 258 meV below cycle 0) |
| T1_open4 | 5 | 2.10 | does not open; same in all four cycles |
| T1_closed_1w | 4 | | water at 2.50 A from Sn, outside the 2.45 A count |
| T1_closed_2w | 5 | | waters at 2.42 and 2.60 A; later cycles lost one at the 2.45 A cut |

None of the four hydrolysed variants opens to a 4 coordinate Sn with a free
silanol in this potential: Sn has 5 neighbours within 2.45 A in all four cycles of
all four variants (checked on `energies.csv`), and the silanol oxygen of the lowest
cycle is 2.10 to 2.38 A away, below the 3 A used here as the criterion for an open site. This is
a statement about MACE-MP-0 + D3 and a 2 ps anneal, not about Sn-Beta. In
`md_cell` one of the two initial bound waters left Sn in every cycle (5 neighbours
after relaxation against 6 in the step 00 audit), and its energy differs by 1.4 eV
between cycles because the water network rearranges.

| quantity | kJ/mol |
|---|---|
| E_hydrolysis, open1 / open2 / open3 / open4 | +8.5 / −7.5 / −30.6 / −14.8 |
| E_bind per water, 1 water / 2 waters | −49.4 / −44.6 |

E(H2O, gas) = −14.170 eV. The spread of the four open variants (39 kJ/mol) and
the 258 meV between anneal cycles of open3 show that single geometries from four
cycles are not converged minima. Treat these as triage numbers.

## 4. DFT check of step 01

`pw.x` PBE, 60 / 480 Ry, K_DENSITY 20 (3x3x3 for the 36 to 42 atom cells), water
in a box at Gamma, single points at the MACE + D3 geometries; MACE without D3 at
the same geometries (float64).

| quantity | MACE + D3 | MACE | PBE | MACE − PBE |
|---|---|---|---|---|
| open1 E_hydrolysis | 8.5 | 39.4 | 43.9 | −4.5 |
| open2 E_hydrolysis | −7.5 | 24.0 | 23.0 | +1.0 |
| open3 E_hydrolysis | −30.6 | 6.3 | 13.3 | −7.0 |
| open4 E_hydrolysis | −14.8 | 11.5 | 12.0 | −0.5 |
| E_bind, 1 water | −49.4 | −18.7 | −21.7 | +3.1 |
| E_bind per water, 2 waters | −44.6 | −15.9 | −11.6 | −4.3 |

Mean absolute MACE − PBE: 3.4 kJ/mol; no value above the 10 kJ/mol flag. Limits:
MACE geometries, not PBE; PBE without dispersion (the D3 term, about −31 kJ/mol
on average, −26 to −37 over the six quantities, for hydrolysis and for water
binding, is the same add-on on both sides and was not tested against DFT-D3); one geometry per model; six numbers, so 3.4 is a mean of
six values, not a statistical error.

## 5. Step 02: unbiased MD

* 100 ps at 350 K, 54 atoms, MACE-MP-0 medium + D3, float32. Timing: 42 ms per
  step plain (1.02 ns/day at 0.5 fs), 55 ms per step with the CN bias (0.79
  ns/day). The step is CPU bound (the D3 neighbour list), the GPU is 20 to 25
  percent busy.
* First RDF peak 2.14 A, recommended r0 2.51 A. One water stays bound to Sn at
  2.14 A for the whole run; g(r) is zero between about 2.5 and 3.3 A; second shell
  near 3.9 A. No binding or unbinding event in 100 ps.
* Mean CN 1.12 ± 0.006 (sem, r0 = 2.6 A), τ = 0.49 ps. This τ is the fluctuation
  of the soft CN (the bound water distance, and the five waters that are not
  bound, mostly at 3.5 to 4 A, which together add about 0.35; the integral of g(r)
  is 4.97 waters beyond 3.3 A), not an exchange time.
* Broken water: 77 of 1801 frames (4.3 percent) in 55 episodes of 50 to 150 fs, a
  transient proton shuttle between the bound water and the pore water network.
  Frames for GIPAW are filtered to those with all six waters intact. Whether
  MACE-MP-0 over-transfers protons here is not settled.
* The metadynamics replicas used r0 = 2.6 A, started before step 02 finished. Both
  2.51 and 2.6 A lie in the empty region between the first and second shell, so
  they count the same waters (CN differs by about 0.04).

## 6. Step 03: metadynamics

Setup: `md_cell` (54 atoms, 6 waters), r0 = 2.6 A, σ = 0.07, initial hill 1.0 kT
(0.0302 eV), bias factor 10, a hill every 250 steps, 350 K, 450 000 steps of 0.5 fs
(225 ps, 1800 hills) per replica, 55 to 60 ms per step (6 h 57 min to 7 h 32 min).
One nanosecond would take about 30 h.

**First three replicas** (seeds 11 to 13, first version of the script). Free
energy in kJ/mol relative to each replica's own minimum between CN 0.8 and 1.8:

| CN | 0.3 | 0.4 | 0.5 | 0.7 | 1.1 | 1.3 | 1.5 | 1.65 | 1.8 | 2.0 | 2.2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 56.1 | 57.1 | 54.4 | 36.7 | 7.0 | 13.8 | 3.3 | 0.0 | 6.7 | 32.1 | 68.7 |
| r2 | 58.9 | 54.0 | 49.5 | 36.9 | 0.2 | 6.5 | 16.9 | 22.0 | 31.5 | 60.3 | 89.2 |
| r3 | 58.0 | 53.8 | 50.3 | 35.5 | 0.2 | 8.0 | 19.3 | 22.0 | 29.6 | 51.9 | 92.3 |

* In all three: the 5 coordinate state (one bound water, CN 1.1) is a minimum;
  F at CN 0.3 to 0.5 lies 49 to 59 kJ/mol above it; the barrier to the 6
  coordinate region is small (F at CN 1.3 is 6.5 to 8.0 kJ/mol above the 5
  coordinate minimum in all three, 6.5 to 13.8 above each replica's own lowest
  point); each replica changes state 144 to 182 times.
* The 6 coordinate region is a shoulder in r2 and r3 (F rises from the minimum at
  CN 1.1 to 1.8 without a second minimum) and a second minimum, at CN 1.65, only
  in r1.
* ΔF(6c − 5c), second half average of the bias estimate: r1 −5.6, r2 +15.3, r3
  +13.6 kJ/mol (the reweighted values of the table below are −4.4, +13.6, +11.6).
* Frames were stored only near the basins each replica detected. All 485 stored 6
  coordinate frames came from replica 1, and no 4 coordinate frames were stored.
* A check of the files found: an empty ΔF panel for replicas 2 and 3; 62 to 64
  percent of every CN axis drawn as a flat extrapolation at 103 to 110 kJ/mol
  where the bias never went; per replica population labels that were not results.
  The files themselves were consistent: HILLS has 1801 hills, COLVAR all 2251
  frames, the stored bias equals the bias rebuilt from HILLS to 6e-5 kJ/mol on the
  engine's grid (3e-3 to 8e-3 kJ/mol against analytic Gaussians, the grid
  interpolation), and `fes.dat` equals −γ/(γ − 1) V rebuilt from HILLS to 8e-5
  kJ/mol.
* The bias estimate F = −γ/(γ − 1) V and a second estimator, the reweighted
  histogram of the same frames with the offset c(t) (Tiwary and Parrinello), agree
  within 1 to 4 kJ/mol where a run is steady. Both use the same CV and potential:
  this checks the algebra, not the choice of CV.
  Replica 1 is not steady: its reweighted ΔF in four time blocks is +12.5, +0.2,
  −2.0, −9.9 kJ/mol.

**Script rewritten, four more replicas** (seeds 21 to 24; `n5a`, `n5b` started from
the 5 coordinate state, `n6a`, `n6b` from a 6 coordinate frame, CN 1.57, waters at
2.12 and 2.51 A). The new `03_metad.py` stores every frame, flushes HILLS and
COLVAR as it goes, writes a checkpoint, plots F only where the bias went, and
computes ΔF against time for every replica.

| replica | start | ΔF(6c − 5c), reweighted, second half (kJ/mol) | bias estimate at the end |
|---|---|---|---|
| r1 | 5c | −4.4 | −6.2 |
| r2 | 5c | +13.6 | +14.7 |
| r3 | 5c | +11.6 | +15.7 |
| n5a | 5c | +15.2 | +19.3 |
| n5b | 5c | +13.5 | +13.1 |
| n6a | 6c | +14.5 | +15.8 |
| n6b | 6c | +17.5 | +13.9 |

* All seven: mean 11.7, spread 7.3, standard error 2.8, median 13.6 kJ/mol.
  Without replica 1: mean 14.3, spread 2.0, standard error 0.8, median 14.1.
  No start state dependence is seen beyond the spread: the two 6 coordinate starts
  give +14.5 and +17.5, the four 5 coordinate starts without replica 1 give +11.6
  to +15.2 (difference of means +2.5 ± 1.7 kJ/mol, two against four replicas, so
  the test has little power).
* Replica 1 is the single outlier. Its reweighted ΔF drifted from +12.5 to −9.9
  and it spent a long time near CN 0.1 at 45 ps, with an F minimum there that no
  other replica shows. No frames of that episode were stored. Not explained.
* 4 coordinate state: ΔF(4c − 5c) over CN 0.05 to 0.65 is 35 to 51 kJ/mol (window
  integrated, reweighted second half 40.7, 50.6, 47.3, 36.9, 46.1, 41.2, 34.9 for
  the seven replicas, bias estimate 33 to 58; there is no minimum, only a
  shoulder). F at the single points CN 0.3 to 0.5 ranges from 38
  to 85 kJ/mol between the new replicas.
* State definition: ΔF(6c − 5c) moves between +7.5 and +18.9 kJ/mol with the CN
  window edges, and is +16.2 to +17.3 when states are defined by 1 or 2 water
  oxygens within 2.8 A of Sn.
* Structures of the stored frames of the new replicas (hard cutoff 2.8 A): at CN
  0.20 to 0.55 (540 frames) 99 percent have no water within 2.8 A (nearest
  3.45 ± 0.34 A); at CN 0.90 to 1.30 (3697 frames, 86 percent intact) exactly one
  water (2.16 ± 0.09 A, next 3.68 A); at CN 1.40 to 1.85 (2202 frames, 90 percent
  intact) two waters in 88 percent and one in 12 (2.13 and 2.53 ± 0.22 A), so the
  second water is soft.
* The README first mapped CN values to coordination as 0, 1, 2. The 6 coordinate
  state sits at CN 1.5 to 1.7 because a second bound water at 2.4 A counts about
  0.62 (r0 = 2.6 A, n = 6, m = 12). Corrected.
* Snapshot weights: the first version of the code weighted frames by exp(V/kT)
  with the bias at storage time, which is wrong (the bias keeps growing). Frames
  inside a narrow CN window now carry equal weight. At fixed CN they are
  Boltzmann distributed; across the window the CN distribution is the tempered one,
  exp(−F/(γ kT)). Checked on the measured F(CN): the mean CN moves by −0.008 (5
  coordinate window, F rises 3.6 kT) and +0.067 (6 coordinate window, 4.9 kT), a
  change of 0.2 ppm in the 5 coordinate shift for the observed correlation of 0.08
  between shift and CN, 1 ppm at most for 0.5 (6 coordinate window: 1 to 7 ppm for
  an assumed 28 ppm spread; `analysis/check_bias_and_window_weights.py`). A toy double well with an exact
  answer (U rising 3 kT across the window, three seeds) gives errors of 0.001 to
  0.004 in ⟨x⟩ with equal weights, 0.001 to 0.009 with exp(V_final/kT) weights,
  inside the noise of 1200 frames.

PLUMED line for the cross check: `cn: COORDINATION GROUPA=25
GROUPB=37,40,43,46,49,52 R_0=2.6 NN=6 MM=12`.

## 7. MACE-MP-0 against PBE on metadynamics frames

30 frames of the 5 coordinate state (three replicas) and 30 of the 6 coordinate
state (replica 1), `pw.x` PBE 60 / 480 Ry, 2x2x2 k, forces on; MACE without D3 at
the same frames (float64).

* Force RMSE (per component) 79.7 meV/A over all atoms (5 coordinate 79.3, 6
  coordinate 80.0); by element H 70, O 64, Si 105, Sn 201 meV/A. Energy spread
  inside a state 1.63 meV/atom. The thresholds used (force RMSE under 100 meV/A,
  energy spread under 5 meV/atom) are met.
* Sn is the weak point: the force vector error on Sn is 332 (5 coordinate) and 363
  (6 coordinate) meV/A rms, 19 percent of the PBE force on Sn (rms 1827 meV/A).
  The three largest single errors (577 to 662 meV/A) are all on Sn.
* The energy offset PBE − MACE differs between the states: 6c − 5c = +9.7 ± 1.8
  kJ/mol (standard errors from 30 frames each; per frame scatter 5.6 and 8.3
  kJ/mol). MACE stabilises the 6 coordinate state by about 10 kJ/mol in potential
  energy relative to PBE. Estimate of the effect on ΔF(6c − 5c): +9.7 kJ/mol to
  first order and +3.3 kJ/mol with the second cumulant. The scatter exceeds kT
  (2.9 kJ/mol), so these are indicative. Potential energies only, no D3 on the PBE
  side, 6 coordinate frames from one replica.

## 8. GIPAW

**Pseudopotentials** (from the Quantum ESPRESSO pseudopotential library; all four
already carry GIPAW data, so no `ld1.x` regeneration was needed): `Sn.pbe-dn-kjpaw_psl.1.0.0.UPF`,
`Si.pbe-n-kjpaw_psl.1.0.0.UPF`, `O.pbe-n-kjpaw_psl.0.1.UPF`,
`H.pbe-kjpaw_psl.1.0.0.UPF`.

**Convergence on SnO2** (cassiterite, experimental cell 4.737 x 4.737 x 3.186 A, 2
Sn + 4 O), `Total sigma` of Sn:

| ecutwfc / ecutrho (Ry) | k grid (K_DENSITY) | σ_iso(Sn) (ppm) | change (ppm) | σ_iso(O) (ppm) |
|---|---|---|---|---|
| 60 / 480 | 5x5x7 (20) | 3017.44 | | 122.59 |
| 70 / 560 | 5x5x7 (20) | 3016.53 | −0.91 | 122.34 |
| 80 / 640 | 5x5x7 (20) | 3015.97 | −0.56 | 122.25 |
| 90 / 720 | 5x5x7 (20) | 3016.01 | +0.04 | 122.25 |
| 70 / 280 | 5x5x7 (20) | 3016.52 | −0.01 vs 70 / 560 | 122.33 |
| 70 / 560 | 6x6x8 (25) | 3018.49 | +1.96 vs K 20 | 128.37 |
| 70 / 560 | 7x7x10 (30) | 3019.30 | +0.81 | 130.85 |
| 70 / 560 | 8x8x11 (35) | 3019.49 | +0.19 | 131.55 |
| 70 / 560 | 9x9x13 (40) | 3019.61 | +0.12 | 131.88 |
| 70 / 560 | 11x11x16 (50) | 3019.62 | +0.01 | 132.01 |

The cutoff is converged to under 1 ppm at 70 Ry. The k grid is the sensitive part
for the small SnO2 cell: 5x5x7 is 3 ppm low for Sn; K_DENSITY 35 is within 0.13
ppm of K 50. Reference used: 70 / 560 Ry, K_DENSITY 40, σ(SnO2) = 3019.61 ppm,
δ(SnO2) = −603.8 ppm.

**k points in the zeolite cell.** `T1_closed` at 3x3x3 (14 k) and 4x4x4 (36 k): all
36 atoms agree within 0.01 ppm (Sn exactly), total energy within 2e-6 Ry. At 2x2x2
(4 k) against 3x3x3: Sn −0.08 ppm, all atoms within 0.17 ppm (mean 0.054); 44 min
against 72 min.

**gipaw.x crash.** At ecutrho 280 Ry `gipaw.x` stopped with `cdiaghg: problems
computing cholesky` in 8 of 9 runs while the SCF was fine and the job reported exit
0. At 70 / 560 Ry with the default parallel diagonalisation all 8 zeolite runs
stopped the same way. Variants tried on `T1_closed`: `-nd 1` ran through; 8 cores,
`diagonalization = 'cg'`, `nbnd = 121` and `conv_thr = 1e-8` all failed; a QE 7.3
gcc build did not finish in 2 h. The cause is the parallel dense linear algebra of
this QE 7.5 build, not the physics settings. The written `job.sh` now uses
`-nd 1` and stops at the first failing step.

**Relaxed step 01 geometries** (one geometry per state, 70 / 560 Ry, 3x3x3 k, 12
cores, `gipaw.x` 1 h 10 to 1 h 20 per state). δ = σ(SnO2) − σ + δ(SnO2).

| state | Sn–O < 2.7 A | σ (ppm) | δ (ppm) | δ minus closed |
|---|---|---|---|---|
| T1_closed | 1.93, 1.93, 1.95, 1.96 | 2838.3 | −422.5 | 0 |
| T1_open1 | 1.94, 1.97, 1.97, 2.00, 2.38 | 2931.2 | −515.4 | −92.9 |
| T1_open2 | 1.97, 1.98, 1.99, 2.02, 2.15 | 2924.2 | −508.4 | −85.9 |
| T1_open3 | 1.97, 1.99, 2.01, 2.02, 2.18 | 2930.2 | −514.3 | −91.8 |
| T1_open4 | 1.96, 1.96, 1.97, 2.08, 2.10 | 2959.8 | −544.0 | −121.5 |
| T1_closed_1w | 1.95 x2, 1.96 x2, 2.50 | 2935.6 | −519.8 | −97.3 |
| T1_closed_2w | 1.97, 1.97, 1.98, 1.99, 2.42, 2.60 | 3047.2 | −631.4 | −208.9 |

The shift moves upfield with every added Sn–O contact: closed to two waters by 209
ppm, in the direction and of about the size of the dehydrated to hydrated change in
Sn-Beta (about 250 ppm). The four open variants spread over 36 ppm (−508 to −544).
The closed value (−422.5) lies in the range of dehydrated Sn-Beta resonances
(about −420 to −446). Bermejo-Deval et al. (PNAS 2012) report −420 and −443 and
assign them to open and closed sites; the ACS Catal. paper of the same group
reports −424 and −443; Qi 2018 reports closed sites at −422, −441 and −446 and
open sites at −429 and −443. The sources do not agree on which line is open and
which is closed. The value lies near the downfield edge; that is not a validation
(CHA, PBE, one geometry).

**Snapshots of the metadynamics basins** (70 / 560 Ry, `-nd 1`, 16 cores,
`gipaw.x` 52 min to 1 h 53 min per 54 atom frame; frames with all waters intact).

| state | n | ⟨δ⟩ (ppm) | 95% interval | spread (ppm) | notes |
|---|---|---|---|---|---|
| 5 coordinate (CN 1.1) | 4, 3x3x3 k | −576.5 | −599.1 to −552.7 | 25.4 | −577.5, −544.3, −606.3, −578.0 |
| 5 coordinate | 20, 2x2x2 k | −578.5 | −591 to −567 | 28.0 | median −578.8, range −642.6 to −536.5 |
| 6 coordinate (CN 1.5) | 4, 3x3x3 k | −706.0 | −740.9 to −685.9 | 35.1 | −693.7, −683.3, −688.8, −758.3 |

* The 20 snapshots come from the first three replicas (5, 7 and 8 frames). By
  replica: −583, −567, −585 ppm (one way ANOVA p = 0.45). The shift does not
  correlate with the CN inside the window (r = 0.03 with all 20 snapshots; an
  earlier value of 0.08 is superseded). Values are in
  `docs/results/data/delta_119Sn_5c_20_snapshots.csv`.
* All 6 coordinate snapshots come from replica 1, the replica whose free energy
  differs from the others. The mean is pulled by one frame at −758 ppm (median
  −691). The 6 coordinate shift is preliminary.
* One ¹¹⁹Sn tensor per calculation; the same run gives labels for 11 ²⁹Si, 30 ¹⁷O
  and 12 ¹H.
* SOAP check (CRISP, Sn centred, rcut 6 A): distance between the two basin
  centroids 7.31; mean distance to the own centroid 6.05 (5 coordinate) and 5.46
  (6 coordinate). Each state has about as much internal variety as the difference
  between them, so four snapshots sample a state coarsely.

## 9. Weighted spectrum

Populations of the 5 and 6 coordinate states from the free energy at 350 K, state
shifts −578.5 ppm (5c) and −706.0 ppm (6c); the 4 coordinate state is negligible.

| populations from | ΔF(6c − 5c) (kJ/mol) | p(6c) | weighted ⟨δ⟩ (ppm) |
|---|---|---|---|
| replica 1 alone | −4.4 | 0.82 | −683 |
| all seven replicas, mean | 11.7 | 0.018 | −581 |
| six replicas without replica 1 | 14.3 | 0.007 | −579.4 |
| ... plus MACE to PBE shift +3.3 | 17.6 | 0.002 | −578.8 |
| ... plus MACE to PBE shift +9.7 | 24.0 | 0.0003 | −578.5 |

The 6 coordinate state carries at most a few percent, so the weighted shift is set
by the 5 coordinate shift; a 50 ppm error in the 6 coordinate shift moves it by
0.4 ppm (p = 0.007) to 0.9 ppm (p = 0.018), and by 41 ppm if replica 1 alone were
right (p = 0.82). The weighted shift is the first moment of the spectrum: one line
at that position if exchange between the states is fast on the NMR time scale, the
centre of two lines weighted by p if it is slow; the 127.5 ppm between the states
is about 19 kHz at 9.4 T, and the exchange rate was not computed. Valid for this model: 6 waters per 12 T sites, 350 K, MACE-MP-0 +
D3, CHA, one loading. `docs/results/all_replicas_summary.png` shows the replicas,
`docs/results/spectrum_scenarios.png` the spectra.

## 10. Errors found and corrected

* Symmetry tolerance split CHA into four T sites (section 2).
* Snapshot weights used the growing bias (section 6).
* An anneal cycle could change the state behind a label (a water left, an open site
  closed). The guard now takes the coordination after the first relaxation as the
  state, and every cycle is kept.
* The cycle guard first compared against 4 + bound waters at a hard 2.45 A cutoff
  and flagged every model.
* `gipaw.x` Cholesky failures (section 8).
* A job landed on a GPU the CUDA 13 build does not support (section 3).
* Figures that hid missing data (section 6).
* The generated SnO2 manifest pointed at the K_DENSITY 20 reference, which was not
  run and is about 3 ppm low; step 05 was run with the converged reference.
* The first version of the README mapped CN to coordination wrongly (section 6).
* Found on 22 October, when every number of this log was recomputed from the data
  files and the raw HILLS, COLVAR and MD log: the double well population was quoted
  within 0.01 although the seed to seed scatter is 0.02 (section 1); the
  correlation of shift with CN is 0.03, not 0.08 (section 8); the barrier at CN
  1.3 is 6.5 to 8.0 kJ/mol above the 5 coordinate minimum, not 7 to 14 (section
  6); there are five waters that are not bound in the 5 coordinate state, not four
  (section 5); the spectrum figure showed the first three replica scenarios and
  was redrawn with the final five (section 9). The literature shift ranges were
  checked in the papers (PNAS 2012, ACS Catal. 2012, Qi 2018): dehydrated Sn-Beta
  at about −420 to −446 ppm and hydrated at −685 to −700 ppm stand.
