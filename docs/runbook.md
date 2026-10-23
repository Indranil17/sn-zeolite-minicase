# Runbook

Times are rough; measure on your own hardware with a short run first.

## 0. Environment (once)

```bash
conda env create -f environment.yml && conda activate snmini
pytest -q                                   # 34 tests, about 25 s
python scripts/validate_metad_doublewell.py # about a minute, writes outputs/validation/
```

On a GPU node install the CUDA build of PyTorch before `mace-torch`.
`torch-dftd` provides the D3(BJ) term behind `DISPERSION = True`; without it
`mace_mp` stops with an error.

Compute nodes often have no internet. Download the foundation model once on a
login node, into a cache directory the jobs can also see:

```bash
export XDG_CACHE_HOME=/path/on/shared/storage/.cache   # also export it in the job script
python -c "from mace.calculators import mace_mp; mace_mp(model='medium', device='cpu')"
```

## 1. Structures (laptop)

`python scripts/00_build_structures.py`

Check in `outputs/00_structures/audit.json`: Sn coordination 4 (closed),
5 (one water), 6 (two waters); `min_h_contact` above 1.5 Å everywhere. If the
IZA download is blocked, put `CHA.cif` from the IZA database into `data/`
yourself; without it the idealised fallback is used (relax with cell).

## 2. Foundation model screen (GPU)

`qsub -v STEP=01_screen_foundation hpc/gpu_dynamics.pbs`

Look at `energies.csv` (`sn_cn`, `sn_silanol_O`) and `derived.json` (energies). Before going on:
* Do the open sites relax to 4 coordinate Sn (`sn_cn` = 4, `sn_silanol_O`
  above about 3 Å)? If a variant stays 5 coordinate, note it: that is a result
  about this potential, not a bug.
* Run DFT single points at the relaxed geometries (plain PBE, the same
  pseudopotentials and cutoff for every model) and compare the hydrolysis and
  binding energies with MACE evaluated without D3 at the same geometries; the
  D3 term is the same add-on on both sides. Large disagreement means fine
  tune before dynamics.

## 3. Unbiased MD (GPU)

`qsub -v STEP=02_md_rdf hpc/gpu_dynamics.pbs` (100 ps)

Take `r0_recommended` from `outputs/02_md/summary.json` into `R0` of
`03_metad.py`. Look at `rdf_cn.png`: a clear first peak near 2.2 Å and a
minimum before 3.5 Å.

## 4. Metadynamics (GPU)

`qsub -v STEP=03_metad hpc/gpu_dynamics.pbs` (225 ps is 450 000 steps and takes about 7.5 h for 54 atoms at 55 ms per step; 1 ns would take about 30 h). The script stores every frame, flushes as it goes and writes a checkpoint; set `RESTART = True` to continue a run that was stopped.

Convergence is not a single number. Run several replicas from different start states (a 5 coordinate and a 6 coordinate frame), rebuild dF between the states from the HILLS files at several times (`analysis/metad_dF_vs_time.py`), compare the bias estimate with the reweighted histogram (`analysis/check_metad_replicas.py`), and test the state definition (`analysis/check_state_definitions.py`). The replicas have to agree, not only each one on its own: in the runs here six of seven did and one did not (`docs/run_log.md`, section 6).

## 5. GIPAW (CPU)

The default code is Quantum ESPRESSO with QE-GIPAW, both free and open source.
`pw.x` and `gipaw.x` must come from the same QE version: use a cluster module
that provides `gipaw.x`, or build QE and then `make gipaw` inside the QE
source tree (see the QE-GIPAW README).

Pseudopotentials need GIPAW reconstruction data (`has_gipaw="T"` in the UPF
header). Download the pslibrary 1.0.0 PAW files (scalar relativistic) for Sn,
Si and H, and O from SSSP or pslibrary, from
`https://pseudopotentials.quantum-espresso.org/upf_files/<name>`, into
`pseudo_src/`, then:

```bash
python scripts/04a_gipaw_pseudos.py      # checks, regenerates with ld1.x, fills pseudo_gipaw/
```

It copies files that already carry GIPAW data, and for the others writes the
`ld1.x` input stored in the file with `lpaw`, `lgipaw_reconstruction` and
`use_paw_as_gipaw` switched on, runs `ld1.x` and checks the result. Use the
`ld1.x` of the QE version you built QE-GIPAW with.

Then `python scripts/04_gipaw_inputs.py` on the cluster (the pseudopotential
path goes into every input as an absolute path), and submit each folder with
`hpc/cpu_gipaw.pbs`. Converge `QE_ECUTWFC` (with `QE_ECUTRHO` at 8 times it)
and the k points on SnO2 and one snapshot first (target: ¹¹⁹Sn shielding
stable to about 1 ppm). NMR needs a higher cutoff than the UPF header
suggests. In the runs here: 70 / 560 Ry; for SnO2 a K_DENSITY of 35 or more; 2x2x2
k for the 9.3 A zeolite cell (`docs/run_log.md`, section 8). A 54 atom frame took
52 min to 1 h 53 min on 16 cores; measure on your own hardware.

With Quantum ESPRESSO 7.5 `gipaw.x` stopped with `cdiaghg: problems computing
cholesky` unless run with `-nd 1` (serial dense linear algebra), although `pw.x`
finished and the job exited with status 0. `04_gipaw_inputs.py` writes `-nd 1`
into every `job.sh`; check `nmr.out` for the "Total NMR chemical shifts" block
before counting a job as done.

Sn is heavy enough that the relativistic treatment matters. Use exactly the
same pseudopotentials and settings for SnO2 and for every snapshot, so that
the referencing removes as much of the systematic error as it can, and read
the results as differences between states. CASTEP (scalar relativistic ZORA)
and VASP 6.6 (`LZORA`) inputs can still be written with `CODE = "castep"` or
`"vasp"` if you hold a licence.

## 6. Analysis (laptop)

```bash
python scripts/05_gipaw_analysis.py
python scripts/06_weighted_spectrum.py
```

The README Results table was filled from `outputs/05_gipaw/results.json`,
`outputs/06_spectrum/summary.json` and the files in `docs/results/data/`; the run
diary is `docs/run_log.md`.
