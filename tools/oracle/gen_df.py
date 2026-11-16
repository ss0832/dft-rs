# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 3 oracle generator: density-fitted (RI-J) RKS energies with the SAME
# def2-universal-JFIT auxiliary basis dft-rs bundles — the fit equations are
# identical on both sides, so agreement is grid-limited only.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_df.py

import os
import pyscf
from pyscf import gto, dft

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

CASES = [
    ("h2o", "def2-svp", "svwn5", "slater,vwn"),
    ("nh3", "def2-svp", "svwn5", "slater,vwn"),
]


def main():
    for mol_name, basis_name, func_name, pyscf_xc in CASES:
        charge, mult, atoms = MOLECULES[mol_name]
        elements = sorted({sym for sym, *_ in atoms})
        basis = parse_g94(os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements))
        auxbasis = parse_g94(
            os.path.join(DATA_DIR, "def2-universal-jfit.g94"), set(elements)
        )
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom, basis=basis, unit="Bohr", charge=charge, spin=mult - 1, cart=False
        )
        mf = dft.RKS(mol).density_fit(auxbasis=auxbasis)
        mf.xc = pyscf_xc
        mf.grids.level = 9
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        case = f"{mol_name}_{basis_name.replace('-', '')}_{func_name}_rij"
        with open(os.path.join(OUT_DIR, f"{case}_dft.csv"), "w", newline="\n") as fh:
            fh.write(
                f"# PROVENANCE: PySCF {pyscf.__version__} density-fitted DFT oracle "
                "(def2-universal-JFIT, values only)\n"
            )
            fh.write("# geometry unit: Bohr (exact literals); pyscf grid level 9\n")
            fh.write(f"basis,{basis_name}\n")
            fh.write(f"functional,{func_name}\n")
            fh.write("density_fit,def2-universal-jfit\n")
            fh.write(f"charge,{charge}\n")
            fh.write(f"mult,{mult}\n")
            for sym, x, y, z in atoms:
                fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
            fh.write(f"e_tot,{mf.e_tot:.14e}\n")
        print(f"{case}: E = {mf.e_tot:.10f}")


if __name__ == "__main__":
    main()
