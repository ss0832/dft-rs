# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Transcribe the D4 parameter tables from the PRIMARY paper's supplementary
# material (Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019), si.pdf):
#
#   Table A1  EEQ parameters EN_i, J_ii, kappa_i, a_i, up to radon (Z = 86)
#   Table A4  BJ-damping parameters (DFT-D4-ATM) per density functional
#
# This reads the paper, not any implementation: `dftd4` is LGPL-3.0 and is
# used in this tree only as a numerical oracle.
import re
import sys

from pypdf import PdfReader

SRC = "d4si/si.pdf"


def pages(reader, lo, hi):
    return "\n".join((reader.pages[i].extract_text() or "") for i in range(lo - 1, hi))


def eeq(reader):
    """Table A1: four floats per atomic number."""
    txt = pages(reader, 5, 7)
    rows = {}
    num = r"-?\d+\.\d+"
    for mt in re.finditer(rf"(?m)^\s*(\d{{1,3}})\s+({num})\s+({num})\s+({num})\s+({num})\s*$", txt):
        z = int(mt.group(1))
        if 1 <= z <= 86:
            rows[z] = tuple(float(mt.group(k)) for k in range(2, 6))
    return rows


def damping(reader):
    """Table A4: functional name followed by s6, s8, a1, a2."""
    txt = pages(reader, 15, 16)
    out = {}
    num = r"-?\d+\.\d+"
    for mt in re.finditer(rf"(?m)^\s*([A-Za-z0-9()\-+*/]+)\s+({num})\s+({num})\s+({num})\s+({num})\s*$", txt):
        name = mt.group(1)
        out[name] = tuple(float(mt.group(k)) for k in range(2, 6))
    return out


def main():
    r = PdfReader(SRC)
    e = eeq(r)
    d = damping(r)
    print(f"# EEQ rows: {len(e)}   damping rows: {len(d)}", file=sys.stderr)
    print("# PROVENANCE: Caldeweyher, Ehlert, Hansen, Neugebauer, Spicher,")
    print("# Bannwarth & Grimme, J. Chem. Phys. 150, 154122 (2019),")
    print("# supplementary material, Table A1. Transcribed from the paper.")
    print("# z,en,jii,kappa,a")
    for z in sorted(e):
        v = e[z]
        print(f"{z},{v[0]:.8f},{v[1]:.8f},{v[2]:.8f},{v[3]:.8f}")
    print("# --- Table A4: DFT-D4-ATM BJ damping ---")
    print("# functional,s6,s8,a1,a2")
    for k in sorted(d):
        v = d[k]
        print(f"{k},{v[0]:.8f},{v[1]:.8f},{v[2]:.8f},{v[3]:.8f}")


if __name__ == "__main__":
    main()
