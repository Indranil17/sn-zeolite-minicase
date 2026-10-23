import numpy as np
import pytest

from snmini import structures as S


@pytest.fixture(scope="module")
def cha():
    fw, src = S.load_framework("CHA", cache_dir="/nonexistent",
                               allow_download=False)
    assert "fallback" in src
    return fw


@pytest.fixture(scope="module")
def prim(cha):
    return S.to_primitive(cha)


def test_fallback_topology(cha):
    syms = cha.get_chemical_symbols()
    assert syms.count("Si") == 36 and syms.count("O") == 72
    si = [a.index for a in cha if a.symbol == "Si"]
    o = [a.index for a in cha if a.symbol == "O"]
    assert {len(S.bonded(cha, i, ("O",), 1.9)) for i in si} == {4}
    assert {len(S.bonded(cha, i, ("Si",), 1.9)) for i in o} == {2}


def test_coordination_sequence_matches_iza(cha, prim):
    # IZA coordination sequence of CHA: 4 9 17 29 45 64 85 110 140 173
    for cell in (cha, prim):
        si = [a.index for a in cell if a.symbol == "Si"]
        assert S.coordination_sequence(cell, si[0], 6) == [4, 9, 17, 29, 45, 64]


def test_primitive_cell(prim):
    syms = prim.get_chemical_symbols()
    assert syms.count("Si") == 12 and syms.count("O") == 24
    si = [a.index for a in prim if a.symbol == "Si"]
    assert len(S.symmetry_classes(prim, si)) == 1        # one T site in CHA


@pytest.fixture(scope="module")
def closed(prim):
    si = [a.index for a in prim if a.symbol == "Si"]
    return S.substitute_sn(prim, si[0])


def test_closed_site(closed):
    sn = closed.info["sn_index"]
    assert closed[sn].symbol == "Sn"
    d = [closed.get_distance(sn, j, mic=True)
         for j in S.bonded(closed, sn, ("O",), 2.2)]
    assert len(d) == 4 and np.allclose(d, S.SN_O, atol=1e-6)
    assert S.audit(closed)["min_heavy"] > 1.4


def test_open_sites_add_one_water(closed):
    sn = closed.info["sn_index"]
    bridges = S.bonded(closed, sn, ("O",), 2.2)
    classes = S.symmetry_classes(closed, bridges)
    assert len(classes) == 4            # four distinct oxygens around T in CHA
    for c in classes:
        op = S.make_open_site(closed, sn, c[0])
        n0, n1 = closed.get_chemical_symbols(), op.get_chemical_symbols()
        assert n1.count("O") - n0.count("O") == 1
        assert n1.count("H") - n0.count("H") == 2
        rep = S.audit(op)
        assert rep["min_h_contact"] > 1.5
        assert rep["min_heavy"] > 1.2


def test_hydration_raises_coordination(closed):
    one = S.add_bound_water(closed, 1)
    two = S.add_bound_water(closed, 2)
    assert S.audit(one)["sn_cn"] == 5
    assert S.audit(two)["sn_cn"] == 6
    assert len(S.water_oxygens(two)) == 2
    assert S.audit(two)["min_h_contact"] > 1.5


def test_free_water(closed):
    m = S.add_free_waters(S.add_bound_water(closed, 2), 3, seed=0)
    assert len(S.water_oxygens(m)) == 5
    assert S.audit(m)["min_h_contact"] > 1.5
