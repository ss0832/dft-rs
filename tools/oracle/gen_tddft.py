# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# TDA excitation-energy oracle: frozen PySCF TDA (CIS for HF) energies for
# singlet/triplet closed-shell and spin-resolved open-shell references.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_tddft.py

import os
import pyscf
from pyscf import dft, gto, scf

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

# (mol, basis, dft-rs name, pyscf xc or None for HF, spin mode)
CASES = [
    ("h2o", "sto-3g", "hf", None, "singlet"),
    ("h2o", "sto-3g", "hf", None, "triplet"),
    ("h2o", "sto-3g", "svwn5", "slater,vwn", "singlet"),
    ("h2o", "sto-3g", "svwn5", "slater,vwn", "triplet"),
    ("h2o", "sto-3g", "b3lyp", "b3lypg", "singlet"),
    ("h2o", "sto-3g", "m06l", "m06_l", "singlet"),
    ("ch3_doublet", "sto-3g", "svwn5", "slater,vwn", "spinresolved"),
]

NSTATES = 4


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mol_name, basis_name, func, xc, spin in CASES:
        charge, mult, atoms = MOLECULES[mol_name]
        elements = sorted({sym for sym, *_ in atoms})
        basis = parse_g94(os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements))
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom, basis=basis, unit="Bohr", charge=charge, spin=mult - 1, cart=False
        )
        if xc is None:
            mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
            mf.xc = xc
            mf.grids.level = 9
        mf.conv_tol = 1e-12
        mf.kernel()
        assert mf.converged
        for method in ("tda", "rpa"):
            if method == "tda":
                td = mf.TDA()
            elif xc is None:
                td = mf.TDHF()
            else:
                td = mf.TDDFT()
            td.nstates = NSTATES + 2
            td.singlet = spin != "triplet"
            td.conv_tol = 1e-9
            td.kernel()
            energies = td.e[:NSTATES]
            oscs = td.oscillator_strength()[:NSTATES]
            case = f"{mol_name}_{basis_name.replace('-', '')}_{func}_{spin}"
            with open(
                os.path.join(OUT_DIR, f"{case}_{method}.csv"), "w", newline="\n"
            ) as fh:
                fh.write(
                    f"# PROVENANCE: PySCF {pyscf.__version__} {method.upper()} oracle "
                    "(excitation energies, Hartree)\n"
                )
                fh.write("# geometry unit: Bohr (exact literals); grid level 9\n")
                fh.write(f"basis,{basis_name}\n")
                fh.write(f"functional,{func}\n")
                fh.write(f"spin,{spin}\n")
                fh.write(f"charge,{charge}\n")
                fh.write(f"mult,{mult}\n")
                for sym, x, y, z in atoms:
                    fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
                for e, f in zip(energies, oscs):
                    fh.write(f"excitation,{e:.14e},{f:.10e}\n")
            print(f"{case} [{method}]: " + "  ".join(f"{e:.8f}" for e in energies))


if __name__ == "__main__":
    main()
