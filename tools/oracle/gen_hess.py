# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 6 oracle generator: PySCF-optimized geometry + harmonic frequencies
# (analytic RHF/RKS Hessian, mass-weighted, TR-projected).
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_hess.py

import os
import numpy as np
import pyscf
from pyscf import dft, gto, scf
from pyscf.hessian import thermo

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES

CASES = [
    ("h2o", "sto-3g", "hf", None),
    ("h2o", "def2-svp", "b3lyp", "b3lypg"),
]


def main():
    for mol_name, basis_name, func_name, pyscf_xc in CASES:
        charge, mult, atoms = MOLECULES[mol_name]
        elements = sorted({sym for sym, *_ in atoms})
        basis = parse_g94(os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements))
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=mult - 1)

        def make_mf(m):
            if pyscf_xc is None:
                mf = scf.RHF(m)
            else:
                mf = dft.RKS(m)
                mf.xc = pyscf_xc
                mf.grids.level = 9
            mf.conv_tol = 1e-12
            return mf

        from pyscf.geomopt.geometric_solver import optimize as geo_opt
        mol_eq = geo_opt(make_mf(mol))
        mf = make_mf(mol_eq)
        mf.kernel()
        assert mf.converged
        h = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mol_eq, h)
        freqs = np.asarray(freq_info["freq_wavenumber"], dtype=complex)
        freqs_real = [(-abs(f.imag) if abs(f.imag) > 1 else f.real) for f in freqs]

        case = f"{mol_name}_{basis_name.replace('-', '')}_{func_name}"
        with open(os.path.join(OUT_DIR, f"{case}_freq.csv"), "w", newline="\n") as fh:
            fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} frequency oracle\n")
            fh.write("# optimized geometry (Bohr) + harmonic wavenumbers (cm^-1)\n")
            fh.write(f"basis,{basis_name}\n")
            fh.write(f"functional,{func_name}\n")
            fh.write(f"charge,{charge}\n")
            fh.write(f"mult,{mult}\n")
            coords = mol_eq.atom_coords()
            for a in range(mol_eq.natm):
                fh.write(
                    f"atom,{mol_eq.atom_charge(a)},{coords[a][0]:.12f},"
                    f"{coords[a][1]:.12f},{coords[a][2]:.12f}\n"
                )
            for f in freqs_real:
                fh.write(f"freq,{f:.6f}\n")
        print(f"{case}: {['%.1f' % f for f in freqs_real]}")


if __name__ == "__main__":
    main()
