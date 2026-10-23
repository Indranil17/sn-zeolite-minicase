"""Shielding tensors to chemical shifts, conventions, and output readers.

Sign convention (same as CASTEP and VASP): sigma is the absolute shielding.
    delta = sigma_ref - sigma + delta_ref
For 119Sn the reference used here is SnO2 at -603.8 ppm against SnMe4
(Aliev, Bartok and Yates, Solid State Nucl. Magn. Reson. 89, 1-10, 2018,
doi:10.1016/j.ssnmr.2017.11.002). Change it in the scripts if a different
reference compound is used.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase.io import read

SNO2_DELTA_PPM = -603.8


def iso(tensor):
    return float(np.trace(np.asarray(tensor)) / 3.0)


def principal(tensor):
    """Eigenvalues of the symmetric part, ascending."""
    t = np.asarray(tensor, dtype=float)
    return np.linalg.eigvalsh(0.5 * (t + t.T))


def shift_from_shielding(sigma, sigma_ref, delta_ref):
    """Isotropic shift (or principal components) from absolute shielding."""
    return sigma_ref - np.asarray(sigma) + delta_ref


def herzfeld_berger(shift_principal):
    """(delta_iso, span, skew) from principal shift components.

    Ordering d11 >= d22 >= d33; span = d11 - d33; skew = 3 (d22 - iso) / span.
    """
    d33, d22, d11 = np.sort(np.asarray(shift_principal, dtype=float))
    di = (d11 + d22 + d33) / 3.0
    span = d11 - d33
    skew = 3.0 * (d22 - di) / span if span > 1e-9 else 0.0
    return float(di), float(span), float(skew)


def tensor_summary(sigma_tensor, sigma_ref, delta_ref):
    """Shift tensor parameters of one site from its shielding tensor."""
    shifts = shift_from_shielding(principal(sigma_tensor), sigma_ref,
                                  delta_ref)
    return herzfeld_berger(shifts)


def linear_calibration(sigma_calc, delta_exp):
    """Fit delta_exp = a * sigma_calc + b. Ideal slope a = -1."""
    a, b = np.polyfit(np.asarray(sigma_calc), np.asarray(delta_exp), 1)
    pred = a * np.asarray(sigma_calc) + b
    rmse = float(np.sqrt(np.mean((pred - np.asarray(delta_exp)) ** 2)))
    return float(a), float(b), rmse


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------
def read_magres(path):
    """Shielding tensors from a CASTEP .magres file.

    Returns a list of dicts: element, atom (0 based index), tensor (3x3, ppm).
    """
    atoms = read(path, format="magres")
    ms = atoms.get_array("ms")
    return [dict(element=s, atom=i, tensor=np.array(t))
            for i, (s, t) in enumerate(zip(atoms.get_chemical_symbols(), ms))]


_ROW = re.compile(r"^\s*(\d+)((?:\s+-?\d+\.\d+){6})\s*$")


def read_vasp_outcar(path, n_atoms=None):
    """Isotropic shieldings from a VASP OUTCAR (LCHIMAG = .TRUE.).

    Reads the legacy CSA table, block "(absolute, valence and core)", and takes
    the INCLUDING G=0 columns (iso, span, skew), as recommended for solids.
    Newer VASP versions print a per ion layout instead; the function then stops
    with a message rather than guessing.
    Returns a list of dicts: atom (0 based), iso, span, skew (shielding, ppm).
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    starts = [k for k, ln in enumerate(lines)
              if "(absolute, valence and core)" in ln]
    if not starts:
        if any("iso_shield" in ln for ln in lines):
            raise NotImplementedError(
                "This OUTCAR uses the per ion NMR layout of newer VASP "
                "versions. Extend read_vasp_outcar for it, or use the Quantum ESPRESSO route.")
        raise ValueError("no NMR shielding table found in OUTCAR")
    out = []
    for ln in lines[starts[-1] + 1:]:
        if set(ln.strip()) <= {"-"} and ln.strip():
            if out:
                break
            continue
        m = _ROW.match(ln)
        if m:
            vals = [float(v) for v in m.group(2).split()]
            out.append(dict(atom=int(m.group(1)) - 1, iso=vals[3],
                            span=vals[4], skew=vals[5]))
        elif out:
            break
    if n_atoms is not None and len(out) != n_atoms:
        raise ValueError(f"parsed {len(out)} rows, expected {n_atoms}")
    return out
