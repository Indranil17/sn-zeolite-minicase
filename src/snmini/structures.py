"""Build Sn substituted zeolite models: closed, open and hydrated Sn sites.

Everything here is geometry: no energies are computed. Structures built by these
functions are starting points and must be relaxed (foundation model, then DFT)
before any number is read from them.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.geometry import find_mic, get_distances
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

IZA_CIF_URL = "https://europe.iza-structure.org/IZA-SC/cif/{code}.cif"

# Idealised T site of CHA from the IZA Database of Zeolite Structures
# (R -3 m, hexagonal setting, a = 13.6750 A, c = 14.7670 A, T1 at 36i).
# Oxygen is placed at T-T midpoints when this fallback is used, so the
# structure must be relaxed before use.
FALLBACKS = {
    "CHA": dict(spacegroup=166,
                cellpar=[13.675, 13.675, 14.767, 90.0, 90.0, 120.0],
                t_sites=[(0.9997, 0.2264, 0.1051)]),
}

SN_O = 1.95      # typical Sn-O(framework) bond, Angstrom
SI_O = 1.62      # typical Si-O bond
O_H = 0.97       # hydroxyl / water O-H
SN_OW = 2.25     # Sn-O(water) for a coordinated water molecule
HOH = np.deg2rad(104.5)


# --------------------------------------------------------------------------
# frameworks
# --------------------------------------------------------------------------
def framework_from_t_sites(spacegroup, cellpar, t_sites, tt_cutoff=3.4):
    """Pure silica framework from T site coordinates, with O at T-T midpoints."""
    t = crystal(["Si"] * len(t_sites), basis=t_sites,
                spacegroup=spacegroup, cellpar=cellpar)
    i, j, D = neighbor_list("ijD", t, tt_cutoff)
    keep = i < j                      # each T-T link once
    mids = t.positions[i[keep]] + 0.5 * D[keep]
    # midpoints keep the full space group symmetry (linear T-O-T, T-O
    # about 1.55 A); relaxation bends the bridges
    o = Atoms("O" * len(mids), positions=mids, cell=t.cell, pbc=True)
    fw = t + o
    fw.wrap()
    return fw


def load_framework(code="CHA", cache_dir="data", allow_download=True):
    """Return a pure silica framework.

    Order of preference: a cached IZA CIF in ``cache_dir``, a fresh download
    from the IZA database, then the idealised fallback (CHA only).
    The second return value says which source was used.
    """
    cache = Path(cache_dir) / f"{code}.cif"
    if cache.exists():
        return _clean_cif(read(cache)), f"IZA CIF (cached: {cache})"
    if allow_download:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(IZA_CIF_URL.format(code=code),
                                         headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                cache.write_bytes(r.read())
            return _clean_cif(read(cache)), "IZA CIF (downloaded)"
        except Exception as err:  # network blocked, site down, ...
            if cache.exists():
                cache.unlink()
            print(f"IZA download failed ({err}); trying the fallback.")
    if code in FALLBACKS:
        return framework_from_t_sites(**FALLBACKS[code]), \
            "idealised fallback (O at T-T midpoints)"
    raise RuntimeError(f"No CIF for {code}: put {code}.cif into {cache_dir}/")


def _clean_cif(atoms):
    """IZA CIFs list T atoms as Si; make sure only Si and O are present."""
    syms = set(atoms.get_chemical_symbols())
    if not syms <= {"Si", "O"}:
        raise ValueError(f"unexpected elements in framework CIF: {syms}")
    atoms.pbc = True
    return atoms


def to_primitive(atoms, symprec=1e-3):
    """Smallest periodic cell of the framework (spglib, positions not idealised)."""
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    lat, pos, num = spglib.standardize_cell(cell, to_primitive=True,
                                            no_idealize=True, symprec=symprec)
    return Atoms(numbers=num, scaled_positions=pos, cell=lat, pbc=True)


def _equivalent_atoms(atoms, symprec=5e-3):
    ds = spglib.get_symmetry_dataset(
        (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers),
        symprec=symprec)
    return np.asarray(ds.equivalent_atoms)


def symmetry_classes(atoms, indices, symprec=5e-3):
    """Group atom indices into symmetry equivalence classes.

    Returns a list of lists; the first member of each list is the representative.
    symprec is in Angstrom: the IZA CIF coordinates are rounded to about 1e-3,
    so 1e-3 splits CHA into four T sites (R3); 5e-3 recovers R-3m (one T site).
    """
    eq = _equivalent_atoms(atoms, symprec)
    classes = {}
    for idx in indices:
        classes.setdefault(int(eq[idx]), []).append(int(idx))
    return list(classes.values())


def coordination_sequence(atoms, start, shells=5, tt_cutoff=3.4):
    """Coordination sequence N1..Nk of T site ``start`` on the T-T graph.

    Periodic images are tracked explicitly, so the result is exact for any cell.
    Used as a topology check against the published IZA values.
    """
    t_idx = [a.index for a in atoms if a.symbol != "O" and a.symbol != "H"]
    t_atoms = atoms[t_idx]
    i, j, S = neighbor_list("ijS", t_atoms, tt_cutoff)
    nbrs = {}
    for a, b, s in zip(i, j, S):
        nbrs.setdefault(a, []).append((b, tuple(s)))
    s0 = t_idx.index(start)
    seen = {(s0, (0, 0, 0))}
    frontier = [(s0, (0, 0, 0))]
    out = []
    for _ in range(shells):
        new = []
        for node, img in frontier:
            for b, s in nbrs[node]:
                key = (b, (img[0] + s[0], img[1] + s[1], img[2] + s[2]))
                if key not in seen:
                    seen.add(key)
                    new.append(key)
        out.append(len(new))
        frontier = new
    return out


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def fibonacci_sphere(n):
    """n nearly uniform unit vectors."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                 np.cos(phi)]


def bonded(atoms, i, symbols=None, cutoff=2.4):
    """Indices bonded to atom i (minimum image), optionally filtered by symbol."""
    d = atoms.get_distances(i, range(len(atoms)), mic=True)
    out = [j for j in np.argsort(d) if j != i and d[j] < cutoff]
    if symbols is not None:
        out = [j for j in out if atoms[j].symbol in symbols]
    return out


def _unit(v):
    return v / np.linalg.norm(v)


def best_site(atoms, center, radius, avoid_dirs=(), max_cos=1.0,
              exclude=(), n=2000, far_from=None, min_clear=2.2):
    """Point at ``radius`` from atom ``center`` that is furthest from all others.

    avoid_dirs: unit vectors from the centre to existing bonds; candidates with
    cos(angle) > max_cos to any of them are rejected (keeps bond angles sane).
    exclude: further atom indices ignored when measuring free space (the centre
    atom itself is always ignored).
    far_from: optional atom index; among candidates with clearance >= min_clear,
    take the one furthest from this atom (used to break a bond cleanly).
    Returns (position, clearance in Angstrom).
    """
    all_dirs = fibonacci_sphere(n)
    dirs = all_dirs
    limit = max_cos
    while True:                       # soften the angle limit if nothing fits
        dirs = all_dirs
        for b in avoid_dirs:
            dirs = dirs[dirs @ b <= limit]
        if len(dirs) or limit >= 0.99:
            break
        limit = min(limit + 0.05, 0.99)
    if len(dirs) == 0:
        raise RuntimeError("no direction satisfies the angle constraint")
    cand = atoms.positions[center] + radius * dirs
    skip = set(exclude) | {int(center)}
    keep = [k for k in range(len(atoms)) if k not in skip]
    _, d = get_distances(cand, atoms.positions[keep], cell=atoms.cell,
                         pbc=atoms.pbc)
    clearance = d.min(axis=1)
    k = int(np.argmax(clearance))
    if far_from is not None:
        ok = np.where(clearance >= min_clear)[0]
        if len(ok):
            _, dt = get_distances(cand[ok], atoms.positions[[far_from]],
                                  cell=atoms.cell, pbc=atoms.pbc)
            k = int(ok[np.argmax(dt[:, 0])])
    return cand[k], float(clearance[k])


def _bond_dirs(atoms, i, partners):
    D = atoms.get_distances(i, partners, mic=True, vector=True)
    return [_unit(v) for v in np.atleast_2d(D)]


def expand_first_shell(atoms, center, length, other_length=SI_O):
    """Set centre-O bonds to ``length`` while keeping each O at
    ``other_length`` from its second T neighbour (closest point on the circle
    where the two bond spheres intersect)."""
    a = atoms.copy()
    c = a.positions[center]
    for j in bonded(a, center, ("O",), cutoff=2.2):
        o = c + a.get_distance(center, j, mic=True, vector=True)
        t2 = [k for k in bonded(a, j, ("Si", "Sn"), cutoff=2.2) if k != center]
        if not t2:
            a.positions[j] = c + length * _unit(o - c)
            continue
        t = o + a.get_distance(j, t2[0], mic=True, vector=True)
        d = np.linalg.norm(t - c)
        u = (t - c) / d
        x = (d**2 + length**2 - other_length**2) / (2 * d)
        rho2 = length**2 - x**2
        if rho2 <= 0:                       # spheres do not meet: stay on line
            a.positions[j] = c + length * u
            continue
        centre = c + x * u
        radial = (o - centre) - np.dot(o - centre, u) * u
        if np.linalg.norm(radial) < 1e-6:
            radial = np.cross(u, [0.31, 0.53, 0.79])
        a.positions[j] = centre + np.sqrt(rho2) * _unit(radial)
    return a


# --------------------------------------------------------------------------
# Sn sites
# --------------------------------------------------------------------------
def substitute_sn(framework, t_index):
    """Closed Sn site: Sn(OSi)4, with Sn-O stretched to a sensible length."""
    a = framework.copy()
    if a[t_index].symbol != "Si":
        raise ValueError("t_index must point to a Si atom")
    a[t_index].symbol = "Sn"
    a = expand_first_shell(a, t_index, SN_O, SI_O)
    a.info.update(site="closed", sn_index=int(t_index))
    return a


def make_open_site(closed, sn, o_bridge):
    """Open Sn site: hydrolyse the Sn-O-Si bond through ``o_bridge``.

    Result: (SiO)3Sn-OH plus a silanol HO-Si on the former neighbour.
    Composition grows by exactly one H2O. Geometry is a starting guess.
    """
    a = closed.copy()
    si = [j for j in bonded(a, o_bridge, ("Si",), cutoff=1.9)]
    if len(si) != 1:
        raise ValueError("o_bridge is not a Sn-O-Si bridge")
    si = si[0]
    # 1) keep the bridging O on Si, rotate it away from Sn -> silanol oxygen
    others = [j for j in bonded(a, si, ("O",), cutoff=1.9) if j != o_bridge]
    pos, _ = best_site(a, si, SI_O,
                       avoid_dirs=_bond_dirs(a, si, others),
                       max_cos=np.cos(np.deg2rad(85)), exclude=[o_bridge, sn],
                       far_from=sn, min_clear=2.3)
    a.positions[o_bridge] = pos
    # 2) new hydroxyl oxygen on Sn
    sn_o = [j for j in bonded(a, sn, ("O",), cutoff=2.2) if j != o_bridge]
    pos_on, _ = best_site(a, sn, SN_O,
                          avoid_dirs=_bond_dirs(a, sn, sn_o),
                          max_cos=np.cos(np.deg2rad(85)),
                          far_from=o_bridge, min_clear=2.3)
    a += Atoms("O", positions=[pos_on])
    o_sn = len(a) - 1
    # 3) protons, pointing away from the T atom they belong to
    for o, t in ((o_bridge, si), (o_sn, sn)):
        b = _bond_dirs(a, o, [t])
        pos_h, _ = best_site(a, o, O_H, avoid_dirs=b,
                             max_cos=np.cos(np.deg2rad(105)))
        a += Atoms("H", positions=[pos_h])
    a.info.update(site="open", sn_index=int(sn), hydrolysed_o=int(o_bridge),
                  silanol_si=int(si),
                  sn_silanol_o=float(a.get_distance(sn, o_bridge, mic=True)))
    a.wrap()
    return a


def _place_water(a, o_pos, away_from=None, n_rot=36):
    """Add one H2O with oxygen at o_pos; protons point away from ``away_from``."""
    if away_from is None:
        bis = fibonacci_sphere(64)
        cand = [(b, None) for b in bis]
    else:
        v, _ = find_mic(np.array([o_pos - away_from]), a.cell, a.pbc)
        cand = [(_unit(v[0]), None)]
    best, best_clear = None, -1.0
    for bis_dir, _ in cand:
        p0 = _unit(np.cross(bis_dir, [0.31, 0.53, 0.79]))
        q0 = np.cross(bis_dir, p0)
        for ang in np.linspace(0, np.pi, n_rot, endpoint=False):
            p = np.cos(ang) * p0 + np.sin(ang) * q0
            h1 = o_pos + O_H * (np.cos(HOH / 2) * bis_dir + np.sin(HOH / 2) * p)
            h2 = o_pos + O_H * (np.cos(HOH / 2) * bis_dir - np.sin(HOH / 2) * p)
            _, d = get_distances(np.array([h1, h2]), a.positions,
                                 cell=a.cell, pbc=a.pbc)
            c = d.min()
            if c > best_clear:
                best, best_clear = (h1, h2), c
    a += Atoms("OH2", positions=[o_pos, best[0], best[1]])
    return len(a) - 3


def add_bound_water(model, n=1):
    """Coordinate n water molecules to Sn (5 or 6 coordinate Sn)."""
    a = model.copy()
    sn = a.info["sn_index"]
    for _ in range(n):
        ligands = bonded(a, sn, ("O",), cutoff=2.6)
        pos, _ = best_site(a, sn, SN_OW,
                           avoid_dirs=_bond_dirs(a, sn, ligands),
                           max_cos=np.cos(np.deg2rad(70)))
        _place_water(a, pos, away_from=a.positions[sn])
    a.info["bound_water"] = a.info.get("bound_water", 0) + n
    return a


def add_free_waters(model, n, min_clearance=2.5, spacing=0.4, seed=0):
    """Fill pores with n water molecules at random free positions."""
    rng = np.random.default_rng(seed)
    a = model.copy()
    for _ in range(n):
        grid = np.stack(np.meshgrid(*[np.arange(0, 1, spacing / L)
                                      for L in a.cell.lengths()],
                                    indexing="ij"), -1).reshape(-1, 3)
        cart = a.cell.cartesian_positions(grid)
        _, d = get_distances(cart, a.positions, cell=a.cell, pbc=a.pbc)
        free = cart[d.min(axis=1) >= min_clearance]
        if len(free) == 0:
            raise RuntimeError("no free space left for another water")
        _place_water(a, free[rng.integers(len(free))])
    a.info["free_water"] = a.info.get("free_water", 0) + n
    a.wrap()
    return a


def water_oxygens(atoms, oh_cut=1.25):
    """Indices of O atoms carrying exactly two H (water molecules)."""
    out = []
    for o in [a.index for a in atoms if a.symbol == "O"]:
        h = bonded(atoms, o, ("H",), cutoff=oh_cut)
        if len(h) == 2:
            out.append(o)
    return out


def audit(atoms):
    """Quick sanity report for a built model."""
    sn = atoms.info.get("sn_index")
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    sym = np.array(atoms.get_chemical_symbols())
    heavy = sym != "H"
    hh = d[np.ix_(heavy, heavy)]
    # closest approach of H to anything other than its own O
    h_idx = np.where(~heavy)[0]
    h_min = np.inf
    for h in h_idx:
        own_o = int(np.argmin(d[h]))
        same = {own_o} | {int(k) for k in h_idx if d[own_o, k] < 1.25}
        others = [k for k in range(len(atoms)) if k not in same and k != h]
        h_min = min(h_min, d[h, others].min())
    rep = dict(formula=atoms.get_chemical_formula(),
               n_atoms=len(atoms),
               min_heavy=float(hh.min()),
               min_h_contact=float(h_min) if len(h_idx) else None,
               water=len(water_oxygens(atoms)))
    if sn is not None:
        rep["sn_cn"] = len(bonded(atoms, sn, ("O",), cutoff=2.45))
    return rep
