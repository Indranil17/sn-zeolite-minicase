# %% Check that the water molecules stay intact along a trajectory
# Why: 02 and 03 pick the "water oxygens" once, at the start, and the
# collective variable counts them around Sn. If a water splits on Sn
# (Sn-OH plus a proton on a framework oxygen), that oxygen is a hydroxide but
# is still counted, and a GIPAW shift taken from that frame belongs to another
# species. Foundation models can over-transfer protons, so check before using
# any frame. Runs as one Jupyter cell or as `python check_water_integrity.py`
# from the repository root (needs the snmini package installed).

from pathlib import Path
import json

import numpy as np
from ase.io import read

from snmini import structures as S

# ----------------------------- CONFIG ------------------------------------
START = Path("outputs/01_screen/relaxed.extxyz")   # where the waters are defined
LABEL = "md_cell"
FILES = [Path("outputs/02_md/md_frames.extxyz")] + \
    sorted(Path("outputs/03_metad").glob("state_*.extxyz"))
OH_CUT = 1.25              # A; an O-H distance above this counts as broken
OUTFILE = Path("outputs/water_integrity.json")
# --------------------------------------------------------------------------

start = [a for a in read(START, index=":") if a.info.get("label") == LABEL][0]
ow = S.water_oxygens(start, oh_cut=OH_CUT)
h_all = [a.index for a in start if a.symbol == "H"]
print(f"{LABEL}: {len(ow)} water oxygens defined at the start: {ow}")

report = {}
for f in FILES:
    if not f.exists():
        print(f"skip (not there yet): {f}")
        continue
    frames = read(f, index=":")
    broken_frames, events = 0, []
    for k, a in enumerate(frames):
        bad = []
        for o in ow:
            n_h = int(np.sum(a.get_distances(o, h_all, mic=True) < OH_CUT))
            if n_h != 2:                    # 1: hydroxide, 3: hydronium
                bad.append(dict(o=int(o), n_h=n_h))
        if bad:
            broken_frames += 1
            if len(events) < 20:            # keep the first few for inspection
                events.append(dict(frame=k, time_ps=a.info.get("time_ps"),
                                   cv=a.info.get("cv"), waters=bad))
    report[str(f)] = dict(frames=len(frames), frames_with_broken_water=broken_frames,
                          fraction=broken_frames / max(len(frames), 1),
                          first_events=events)
    print(f"{f}: {broken_frames} of {len(frames)} frames have a broken water")

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
OUTFILE.write_text(json.dumps(dict(label=LABEL, water_oxygens=[int(o) for o in ow],
                                   oh_cut=OH_CUT, files=report), indent=1,
                              default=float))
print(f"wrote {OUTFILE}")
