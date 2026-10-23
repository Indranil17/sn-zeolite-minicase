"""Quantum ESPRESSO helpers: pseudopotential checks, inputs, gipaw.x reader."""
import numpy as np
from ase.build import bulk

from snmini import qe

UPF_NOT_READY = '''<UPF version="2.0.1">
  <PP_INFO>
    Generated using "atomic" code by A. Dal Corso  v.5.1
    <PP_INPUTFILE>
 &amp;input
   title='Si',
   zed=14.,
   rel=1,
   config='[Ne] 3s2 3p2 3d-1',
   iswitch=3,
   dft='PBE'
 /
 &amp;inputp
   lpaw=.false.,
   pseudotype=3,
   file_pseudopw='Si.pbe-n-rrkjus_psl.1.0.0.UPF',
   author='ADC',
   lloc=-1,
   rcloc=1.9,
   which_augfun='AE',
   rmatch_augfun_nc=.true.,
   tm=.true.
 /
2
3S  1  0  2.00  0.00  1.60  1.80  0.0
3P  2  1  2.00  0.00  1.60  1.80  0.0
    </PP_INPUTFILE>
  </PP_INFO>
  <PP_HEADER generated="x" element="Si" pseudo_type="USPP" is_paw="F"
     has_gipaw="F" paw_as_gipaw="F" wfc_cutoff="4.37E+001"/>
</UPF>
'''

UPF_READY = '''<UPF version="2.0.1">
  <PP_HEADER element=" O" pseudo_type="PAW" is_paw="T" has_gipaw="T"
     paw_as_gipaw="T"/>
  <PP_GIPAW gipaw_data_format="2">
  </PP_GIPAW>
</UPF>
'''

GIPAW_OUT = """
     Contributions to the NMR chemical shifts in ppm: --------------------------
     Atom  1  Sn  pos: (  0.000000  0.000000  0.000000)  sigma:     100.00
     Total NMR chemical shifts in ppm:     ---------------------------------------
     (adopting the Simpson convention for anisotropy and asymmetry)-----------

     Atom  1  Sn  pos: (  0.000000  0.000000  0.000000)  Total sigma:        2470.12
           2471.0000        0.0000        0.0000
              0.0000     2471.0000        0.0000
              0.0000        0.0000     2468.3600
     Sn   1  anisotropy:      -3.96  eta: 0.0000
     Atom  3  O   pos: (  0.307000  0.307000  0.000000)  Total sigma:          41.50
     O    3  anisotropy:      12.00  eta: 0.5000

     GIPAW        :      1.00s CPU      1.20s WALL

   JOB DONE.
"""


def test_upf_checks(tmp_path):
    a, b = tmp_path / "a.UPF", tmp_path / "b.UPF"
    a.write_text(UPF_NOT_READY)
    b.write_text(UPF_READY)
    assert qe.upf_header(a)["element"] == "Si"
    assert not qe.upf_gipaw_ready(a)
    assert qe.upf_gipaw_ready(b)


def test_ld1_input_becomes_paw_with_gipaw(tmp_path):
    a = tmp_path / "a.UPF"
    a.write_text(UPF_NOT_READY)
    txt = qe.ld1_input_with_gipaw(a, "Si_new.UPF")
    inputp = txt.split("&inputp")[1].split("\n /")[0]
    assert "&input\n" in txt and "&amp;" not in txt          # unescaped
    assert "lpaw=.true." in inputp and "lpaw=.false." not in inputp
    assert "use_paw_as_gipaw=.true." in inputp
    assert "lgipaw_reconstruction=.true." in inputp
    assert "file_pseudopw='Si_new.UPF'" in inputp
    assert "rrkjus" not in inputp                             # old file name gone
    assert "which_augfun='PSQ'" in inputp                     # AE is US only
    assert "3P  2  1  2.00" in txt                            # cards kept


def test_pw_and_gipaw_inputs(tmp_path):
    si = bulk("Si", "diamond", a=5.43)
    qe.write_pw_scf(si, tmp_path, "si", "./pseudo", {"Si": "Si.UPF"},
                    ecutwfc=60, ecutrho=480, kpts=(3, 3, 3))
    s = (tmp_path / "scf.in").read_text()
    assert "calculation      = 'scf'" in s and "prefix           = 'si'" in s
    assert "ecutwfc          = 60" in s and "ecutrho          = 480" in s
    assert "K_POINTS automatic\n3 3 3" in s
    assert "&FCP" not in s and "&RISM" not in s and "&CELL" not in s
    assert "Si 28.085 Si.UPF" in s or "Si.UPF" in s
    qe.write_gipaw_nmr(tmp_path, "si")
    n = (tmp_path / "nmr.in").read_text()
    assert "job = 'nmr'" in n and "prefix = 'si'" in n and n.strip().endswith("/")


def test_read_gipaw_out(tmp_path):
    f = tmp_path / "nmr.out"
    f.write_text(GIPAW_OUT)
    recs = qe.read_gipaw_out(f)
    assert [r["atom"] for r in recs] == [0, 2]               # 0 based
    assert recs[0]["element"] == "Sn" and recs[1]["element"] == "O"
    assert np.isclose(recs[0]["iso"], 2470.12)                # Total block, not Contributions
    assert recs[0]["tensor"].shape == (3, 3)
    assert np.isclose(np.trace(recs[0]["tensor"]) / 3, 2470.12, atol=0.01)
    assert recs[1]["tensor"] is None
