"""Quantum ESPRESSO helpers (free, open source): pw.x and gipaw.x inputs,
a reader for the NMR block of gipaw.x output, and pseudopotential checks.

GIPAW needs pseudopotentials that carry reconstruction data. A UPF v2 file
says so in its header (has_gipaw="T"). pslibrary files store the ld1.x input
they were made from, so a file without GIPAW data can be regenerated as a PAW
dataset with `lgipaw_reconstruction=.true.` and `use_paw_as_gipaw=.true.`
(the two flags the GIPAW tutorials add to the pslibrary inputs).
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
from ase.io import write


# --------------------------------------------------------------------------
# pseudopotentials
# --------------------------------------------------------------------------
def upf_header(path):
    """Attributes of <PP_HEADER .../> in a UPF v2 file (lower case keys)."""
    text = Path(path).read_text(errors="replace")
    m = re.search(r"<PP_HEADER\b(.*?)/?>", text, re.S)
    if not m:
        return {}
    return {k.lower(): v.strip()
            for k, v in re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(1))}


def _is_true(value):
    return str(value).strip().strip(".").lower() in {"t", "true"}


def upf_gipaw_ready(path):
    """True if the pseudopotential carries GIPAW reconstruction data."""
    if _is_true(upf_header(path).get("has_gipaw", "F")):
        return True
    return "<PP_GIPAW" in Path(path).read_text(errors="replace")


def ld1_input_with_gipaw(path, new_name):
    """The ld1.x input stored in a pslibrary UPF, changed so that it writes a
    PAW dataset with GIPAW data to `new_name`.

    Changes inside &inputp: lpaw=.true., lgipaw_reconstruction=.true.,
    use_paw_as_gipaw=.true., file_pseudopw=new_name; which_augfun 'AE'
    (allowed for ultrasoft only) becomes 'PSQ'. Everything else
    (configuration, radii, reference energies) is kept.
    """
    text = Path(path).read_text(errors="replace")
    m = re.search(r"<PP_INPUTFILE>(.*?)</PP_INPUTFILE>", text, re.S)
    if not m:
        raise ValueError(f"{path}: no PP_INPUTFILE; regenerate from the "
                         "original ld1.x input instead")
    inp = html.unescape(m.group(1)).strip("\n")
    block = re.search(r"(&inputp\b)(.*?)(\n\s*/)", inp, re.S | re.I)
    if not block:
        raise ValueError(f"{path}: no &inputp namelist in the stored input")
    drop = {"lpaw", "use_paw_as_gipaw", "lgipaw_reconstruction",
            "file_pseudopw"}
    kept = []
    for line in block.group(2).splitlines():
        key = line.split("=")[0].strip().lower()
        if key in drop:
            continue
        if key == "which_augfun" and "'ae'" in line.lower():
            line = line.split("=")[0] + "='PSQ',"
        kept.append(line)
    body = "\n".join(kept).rstrip()
    if body and not body.endswith(","):
        body += ","
    body += ("\n   lpaw=.true.,\n   lgipaw_reconstruction=.true.,"
             "\n   use_paw_as_gipaw=.true.,\n"
             f"   file_pseudopw='{new_name}'")
    return inp[:block.start(2)] + body + inp[block.end(2):] + "\n"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def write_pw_scf(atoms, folder, prefix, pseudo_dir, pseudopotentials,
                 ecutwfc, ecutrho, kpts, conv_thr=1e-10):
    """pw.x SCF input <folder>/scf.in. kpts is always an explicit grid; the
    Gamma point trick of pw.x is avoided as a precaution, since gipaw.x
    reads the result."""
    data = dict(
        control=dict(calculation="scf", prefix=prefix, outdir="./tmp",
                     pseudo_dir=str(pseudo_dir), tprnfor=True),
        system=dict(ecutwfc=ecutwfc, ecutrho=ecutrho, occupations="fixed"),
        electrons=dict(conv_thr=conv_thr),
    )
    path = Path(folder) / "scf.in"
    write(path, atoms, format="espresso-in", input_data=data,
          pseudopotentials=pseudopotentials, kpts=tuple(int(k) for k in kpts))
    # ASE writes every namelist; older pw.x versions do not know &FCP and
    # &RISM, and none of the empty ones is needed for an SCF
    text = re.sub(r"&(IONS|CELL|FCP|RISM)\s*\n/\n", "", path.read_text())
    path.write_text(text)


def write_gipaw_nmr(folder, prefix, q_gipaw=0.01):
    """gipaw.x input <folder>/nmr.in for NMR shielding tensors."""
    (Path(folder) / "nmr.in").write_text(
        "&inputgipaw\n"
        "    job = 'nmr'\n"
        f"    prefix = '{prefix}'\n"
        "    tmp_dir = './tmp/'\n"
        f"    q_gipaw = {q_gipaw}\n"
        "    use_nmr_macroscopic_shape = .true.\n"
        "    verbosity = 'medium'\n"
        "/\n")


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
_ATOM = re.compile(r"Atom\s*(\d+)\s+([A-Za-z]+)\s+pos:.*?Total sigma:\s*(-?\d+\.\d*)")
_ROW = re.compile(r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def read_gipaw_out(path):
    """Shielding from the "Total NMR chemical shifts" block of gipaw.x.

    The values in that block are absolute shieldings sigma (ppm). Returns a
    list of dicts: atom (0 based, input order), element, iso, tensor (3x3 or
    None when gipaw.x did not print it).
    """
    lines = Path(path).read_text(errors="replace").splitlines()
    starts = [k for k, ln in enumerate(lines)
              if "Total NMR chemical shifts" in ln]
    if not starts:
        raise ValueError(f"{path}: no 'Total NMR chemical shifts' block; "
                         "did gipaw.x finish?")
    out = []
    k = starts[-1] + 1
    while k < len(lines):
        m = _ATOM.search(lines[k])
        if m:
            rows = [_ROW.match(lines[j]) for j in range(k + 1, min(k + 4, len(lines)))]
            tensor = None
            if len(rows) == 3 and all(rows):
                tensor = np.array([[float(x) for x in r.groups()] for r in rows])
            out.append(dict(atom=int(m.group(1)) - 1, element=m.group(2),
                            iso=float(m.group(3)), tensor=tensor))
        elif out and ("Contributions" in lines[k] or "JOB DONE" in lines[k]):
            break
        k += 1
    if not out:
        raise ValueError(f"{path}: NMR block found but no atom lines parsed")
    return out
