# Code walkthrough

A short tour, module by module, of what each piece does and why it is written
that way.

## `src/snmini/structures.py`

* `load_framework` reads the IZA CIF (cached in `data/`), or builds an
  idealised framework from the T site coordinates with O at T–T midpoints.
  Midpoints keep the full space group symmetry, which the next steps rely on.
* `to_primitive` and `symmetry_classes` use spglib: the first shrinks CHA to
  12 T atoms, the second groups atoms that symmetry makes identical, so only
  distinct T sites and distinct Sn–O–Si bridges are built.
* `coordination_sequence` walks the T–T graph shell by shell while tracking
  periodic images; matching the published IZA sequence proves the topology is
  right.
* `best_site` is the main geometry routine: it samples points on a sphere around
  an atom, rejects directions that would make unphysical bond angles, and keeps the
  point with the most free space (or, when breaking a bond, the point furthest
  from the atom being let go).
* `substitute_sn`, `make_open_site`, `add_bound_water`, `add_free_waters` build
  the three site types; `audit` reports coordination and the closest contacts.

## `src/snmini/cv.py`

`switching` returns f(r) and df/dr. At r = r0 numerator and denominator both
vanish, so a first order Taylor expansion is used within 1e-4 of r0.
`CoordinationCV.__call__` returns s and ds/dR for every atom; the centre atom
gets minus the sum of the group gradients, which keeps the CV translation
invariant (a test checks this).

## `src/snmini/metad.py`

`WellTemperedMetaD1D` stores the bias and its derivative on a grid, so adding a
hill is one vector operation and evaluating the bias is one interpolation.
`BiasedCalculator` wraps any ASE calculator: energy plus V(s), forces minus
dV/ds times ds/dR (chain rule). `from_hills` rebuilds the bias from a HILLS
file, which restart and the analysis scripts use. `basin_populations` integrates
exp(-F/kT) over each basin.

## `src/snmini/umbrella.py`

`HarmonicBias` has the same `bias` and `dbias` methods as the metadynamics bias,
so `BiasedCalculator` takes it unchanged. `mbar_f` solves MBAR by minimising a
convex function whose gradient vanishes at the MBAR solution; `unbiased_weights`,
`pmf` and `window_probability` turn the result into F(CN) and state
probabilities. The tests compare with exact Gaussian and double well answers.

## `src/snmini/toy.py`

A BAOAB Langevin integrator for one particle in a tilted double well, used
only to validate the metadynamics engine against the exact answer.

## `src/snmini/nmr.py`, `stats.py`, `spectrum.py`

* `nmr`: shielding to shift (`δ = σ_ref − σ + δ_ref`), Herzfeld–Berger span
  and skew, readers for CASTEP `.magres` (through ASE) and the legacy VASP CSA
  table. The VASP reader stops with a clear message if it meets a layout it
  does not know, instead of guessing.
* `stats`: integrated autocorrelation time with a self consistent window,
  block averaging, the Flyvbjerg–Petersen blocking curve, bootstrap intervals.
* `spectrum`: sums one Gaussian per state, centred on the state mean, as wide
  as the spread of its snapshots, weighted by the state population.

## `src/snmini/qe.py`

Quantum ESPRESSO, the free route to GIPAW.
* `upf_header` and `upf_gipaw_ready` read the UPF header: GIPAW needs
  pseudopotentials with reconstruction data (`has_gipaw="T"`).
* `ld1_input_with_gipaw` takes the `ld1.x` input that pslibrary stores inside
  every UPF file and switches on `lpaw`, `lgipaw_reconstruction` and
  `use_paw_as_gipaw`, so `ld1.x` writes the same dataset again, with the data
  GIPAW needs.
* `write_pw_scf` and `write_gipaw_nmr` write the two inputs of one
  calculation; the k point grid is always written explicitly, and the Gamma
  point trick of `pw.x` (`K_POINTS gamma`) is avoided on purpose, as a
  precaution for the `gipaw.x` step that reads the result.
* `read_gipaw_out` reads the "Total NMR chemical shifts" block. Despite its
  title, it holds absolute shieldings σ; atom numbers follow the input order,
  so the Sn index from the snapshot is used directly.

## `scripts/`

Each script is one cell: CONFIG block first, then straight line code that
reads the previous step's output and writes its own under `outputs/`.

## `analysis/`

Checks made on the runs, in the same one cell style. `check_metad_replicas.py`
tests files and estimators of each replica, `check_bias_and_window_weights.py`
rebuilds the bias from HILLS and measures what equal weights do inside a CN
window, `check_state_definitions.py` moves
the state boundaries, `metad_dF_vs_time.py` and `summarise_all_replicas.py`
compare replicas, `characterise_states.py` looks at the stored structures,
`dft_check.py` and `pbe_force_check.py` compare MACE with PBE, `select_frames.py`
picks snapshots for GIPAW, `check_water_integrity.py` finds frames with a split
water, `make_overview_figures.py` redraws the figures of the README,
`export_structures.py` writes the CIF files of `structures/`. `analysis/crisp/`
needs a clone of CRISP.
