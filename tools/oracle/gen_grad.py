# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 5 oracle generator: frozen PySCF analytic nuclear gradients.
# PySCF DFT gradients include grid response only approximately at default
# settings; grids.level 9 keeps that residual < 1e-6 Ha/Bohr.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_grad.py

import os
import pyscf
import pyscf.grad  # registers mf.Gradients()
from pyscf import dft, gto, scf

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

CASES = [
    ("h2o", "sto-3g", "hf", None),
    ("h2o", "def2-svp", "hf", None),
    ("nh3", "def2-svp", "hf", None),
    ("h2o", "def2-svp", "svwn5", "slater,vwn"),
    ("h2o", "def2-svp", "pbe", "pbe,pbe"),
    ("h2o", "def2-svp", "b3lyp", "b3lypg"),
    ("ch3_doublet", "sto-3g", "hf", None),
]


def main():
    for mol_name, basis_name, func_name, pyscf_xc in CASES:
        charge, mult, atoms = MOLECULES[mol_name]
        elements = sorted({sym for sym, *_ in atoms})
        basis = parse_g94(os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements))
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom, basis=basis, unit="Bohr", charge=charge, spin=mult - 1, cart=False
        )
        if pyscf_xc is None:
            mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
            mf.xc = pyscf_xc
            mf.grids.level = 9
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        g = mf.Gradients()
        if pyscf_xc is not None:
            g.grid_response = True
        grad = g.kernel()
        case = f"{mol_name}_{basis_name.replace('-', '')}_{func_name}"
        with open(os.path.join(OUT_DIR, f"{case}_grad.csv"), "w", newline="\n") as fh:
            fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} gradient oracle\n")
            fh.write("# geometry unit: Bohr; gradient Hartree/Bohr\n")
            fh.write(f"basis,{basis_name}\n")
            fh.write(f"functional,{func_name}\n")
            fh.write(f"charge,{charge}\n")
            fh.write(f"mult,{mult}\n")
            for sym, x, y, z in atoms:
                fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
            for a in range(len(atoms)):
                fh.write(f"grad,{a},{grad[a][0]:.12e},{grad[a][1]:.12e},{grad[a][2]:.12e}\n")
        print(f"{case}: |g| = {abs(grad).max():.8f}")


if __name__ == "__main__":
    main()
