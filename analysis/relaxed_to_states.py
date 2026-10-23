# %% Turn the relaxed models of step 01 into state files for step 04
# 04_gipaw_inputs.py reads files named state_<name>.extxyz. This cell writes one
# such file per model of step 01, so GIPAW can be run on the relaxed closed,
# open and hydrated Sn sites before (or without) the metadynamics snapshots.
# Run from the repository root:  python analysis/relaxed_to_states.py
# (the geometry, sn_index and label are kept in the frame info by extxyz)

from pathlib import Path

from ase.io import read, write

# ----------------------------- CONFIG ------------------------------------
RELAXED = Path("outputs/01_screen/relaxed.extxyz")
LABELS = ["T1_closed", "T1_open1", "T1_open2", "T1_open3", "T1_open4",
          "T1_closed_1w", "T1_closed_2w"]
OUTDIR = Path("outputs/01_states")
# --------------------------------------------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
models = {m.info["label"]: m for m in read(RELAXED, index=":")}
for label in LABELS:
    m = models[label]
    m.info["weight"] = 1.0
    path = OUTDIR / f"state_{label}.extxyz"
    if path.exists():                       # never overwrite an earlier file
        print(f"kept: {path}")
        continue
    write(path, m, format="extxyz")
    print(f"{label}: Sn index {m.info['sn_index']}, {len(m)} atoms -> {path}")
