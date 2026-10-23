import numpy as np

from snmini import nmr

MAGRES = """#$magres-abinitio-v1.0
[atoms]
units lattice Angstrom
lattice 5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0
units atom Angstrom
atom Sn Sn 1 0.0 0.0 0.0
atom O O 1 2.0 0.0 0.0
[/atoms]
[magres]
units ms ppm
ms Sn 1 2470.0 0.0 0.0 0.0 2500.0 0.0 0.0 0.0 2530.0
ms O 1 200.0 0.0 0.0 0.0 200.0 0.0 0.0 0.0 200.0
[/magres]
"""

OUTCAR = """ ---------------------------------------------------------------------------------
  CSA tensor (J. Mason, Solid State Nucl. Magn. Reson. 2, 285 (1993))
 ---------------------------------------------------------------------------------
            EXCLUDING G=0 CONTRIBUTION                      INCLUDING G=0 CONTRIBUTION
         -----------------------------------------------   -----------------------------------------------
  ATOM    ISO_SHIFT        SPAN        SKEW               ISO_SHIFT        SPAN        SKEW
 ---------------------------------------------------------------------------------
  (absolute, valence only)
     1    4300.3335      0.0000      0.0000             4299.1000      0.0000      0.0000
     2     219.2181     45.1000      0.8000              218.0000     45.1000      0.8000
 ---------------------------------------------------------------------------------
  (absolute, valence and core)
     1    2510.5000      0.0000      0.0000             2509.2500      0.0000      0.0000
     2     250.1000     45.1000      0.8000              248.9000     45.1000      0.8000
 ---------------------------------------------------------------------------------
"""


def test_shift_from_shielding_sign():
    # 100 ppm less shielding than SnO2 -> 100 ppm downfield of SnO2
    assert np.isclose(nmr.shift_from_shielding(2400.0, 2500.0, -603.8),
                      -503.8)


def test_herzfeld_berger_conventions():
    di, span, skew = nmr.herzfeld_berger([10.0, -20.0, 40.0])
    assert np.isclose(di, 10.0) and np.isclose(span, 60.0)
    assert np.isclose(skew, 3 * (10.0 - 10.0) / 60.0)
    _, _, skew_axial = nmr.herzfeld_berger([40.0, 40.0, -20.0])
    assert np.isclose(skew_axial, 1.0)


def test_tensor_summary_from_shielding():
    sig = np.diag([2470.0, 2500.0, 2530.0])
    di, span, skew = nmr.tensor_summary(sig, 2500.0, 0.0)
    assert np.isclose(di, 0.0) and np.isclose(span, 60.0)


def test_read_magres(tmp_path):
    f = tmp_path / "x.magres"
    f.write_text(MAGRES)
    recs = nmr.read_magres(f)
    assert [r["element"] for r in recs] == ["Sn", "O"]
    assert np.isclose(nmr.iso(recs[0]["tensor"]), 2500.0)


def test_read_vasp_outcar_legacy_table(tmp_path):
    f = tmp_path / "OUTCAR"
    f.write_text("header\n" + OUTCAR + "trailer\n")
    rows = nmr.read_vasp_outcar(f, n_atoms=2)
    assert rows[0]["atom"] == 0 and np.isclose(rows[0]["iso"], 2509.25)
    assert np.isclose(rows[1]["span"], 45.1)


def test_linear_calibration_recovers_line():
    s = np.array([2400.0, 2500.0, 2600.0])
    a, b, rmse = nmr.linear_calibration(s, -1.0 * s + 1896.2)
    assert np.isclose(a, -1.0) and np.isclose(b, 1896.2) and rmse < 1e-9
