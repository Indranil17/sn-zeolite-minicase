"""Guards in scripts/04_gipaw_inputs.py, run in a scratch folder."""
import runpy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_gipaw_inputs.py"
NAMES = ["Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
         "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
         "O.pbe-n-kjpaw_psl.0.1.UPF",
         "H.pbe-kjpaw_psl.1.0.0.UPF"]
UPF_READY = '<UPF version="2.0.1">\n<PP_HEADER has_gipaw="T"/>\n</UPF>\n'


def test_missing_pseudopotential_stops(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pseudo_gipaw").mkdir()
    (tmp_path / "pseudo_gipaw" / NAMES[0]).write_text(UPF_READY)   # others absent
    with pytest.raises(SystemExit, match="not found"):
        runpy.run_path(str(SCRIPT))


def test_started_folder_is_not_rewritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdir = tmp_path / "pseudo_gipaw"
    pdir.mkdir()
    for n in NAMES:
        (pdir / n).write_text(UPF_READY)
    runpy.run_path(str(SCRIPT))
    ref = tmp_path / "outputs" / "04_gipaw" / "reference_SnO2"
    assert (ref / "scf.in").exists() and (ref / "job.sh").exists()
    (ref / "scf.out").write_text("finished\n")                     # a run started
    (ref / "scf.in").write_text("EDITED BY HAND\n")
    runpy.run_path(str(SCRIPT))
    assert (ref / "scf.in").read_text() == "EDITED BY HAND\n"
