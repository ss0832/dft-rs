# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Oracle of the standard molecular quadrature grid (Treutler-Ahlrichs M4
# radial map, Lebedev angular rules, five-region pruning, Becke partition
# with the Treutler size adjustment): PySCF's default grids, used strictly
# as a numerical oracle, for H2, LiH, water, phosphine and ferrocene at
# several levels. Per atom it records the number of points before the
# partition; per molecule the partitioned integral of a smooth test
# function and the PW92 LDA (Slater + PW92) STO-3G RKS energy on that grid.
#
# Run (WSL): <venv>/bin/python gen_grid_oracle.py

import json
import os

import numpy
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid

from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

SYMBOLS = {1: "H", 3: "Li", 6: "C", 8: "O", 15: "P", 26: "Fe"}
LEVELS = [1, 3, 5, 9]
ENERGY_LEVELS = [3]


def test_function(coords, centres):
    """sum_A exp(-0.7 |r - R_A|^2) (1 + 0.3 x y + 0.1 z)."""
    total = numpy.zeros(len(coords))
    for c in centres:
        d2 = ((coords - c) ** 2).sum(axis=1)
        total += numpy.exp(-0.7 * d2)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    return total * (1.0 + 0.3 * x * y + 0.1 * z)


def build(numbers, positions):
    elements = sorted({SYMBOLS[z] for z in numbers})
    basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), set(elements))
    geom = "; ".join(f"{SYMBOLS[z]} {x} {y} {w}" for z, (x, y, w) in zip(numbers, positions))
    return gto.M(atom=geom, basis=basis, unit="Bohr", charge=0, spin=0, cart=False, verbose=0)


def main():
    with open(os.path.join(OUT_DIR, "multicomponent_hf.json"), encoding="utf-8") as fh:
        cases = json.load(fh)
    out = {
        "provenance": f"PySCF {pyscf.__version__} default grids (values only)",
        "test_function": "sum_A exp(-0.7 |r - R_A|^2) (1 + 0.3 x y + 0.1 z)",
        "functional": "lda_x,lda_c_pw",
        "basis": "sto-3g",
        "molecules": {},
    }
    for case in cases:
        name = case["name"]
        numbers, positions = case["numbers"], case["positions"]
        mol = build(numbers, positions)
        centres = numpy.asarray(positions, dtype=float)
        entry = {"numbers": numbers, "positions": positions, "levels": {}}
        for level in LEVELS:
            grids = gen_grid.Grids(mol)
            grids.level = level
            atomic = grids.gen_atomic_grids(mol, level=level, prune=grids.prune,
                                            radi_method=grids.radi_method)
            counts = [int(atomic[mol.atom_symbol(i)][0].shape[0]) for i in range(mol.natm)]
            grids.build(with_non0tab=False)
            integral = float(numpy.dot(grids.weights, test_function(grids.coords, centres)))
            record = {"atom_points": counts, "points": int(grids.weights.size),
                      "integral": integral}
            if level in ENERGY_LEVELS:
                mf = dft.RKS(mol)
                mf.xc = out["functional"]
                mf.grids.level = level
                mf.conv_tol = 1e-11
                mf.max_cycle = 300
                mf.kernel()
                if not mf.converged:
                    mf = mf.newton()
                    mf.kernel()
                record["energy"] = float(mf.e_tot)
                record["converged"] = bool(mf.converged)
            entry["levels"][str(level)] = record
            print(name, level, record, flush=True)
        out["molecules"][name] = entry
    path = os.path.join(OUT_DIR, "standard_grids.json")
    with open(path, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
