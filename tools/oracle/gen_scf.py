# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 2 oracle generator: frozen PySCF RHF/UHF total energies (+ <S^2>)
# for the dft-rs SCF gate. Geometries are exact Bohr literals embedded in
# the output CSVs, which the Rust test parses back — a single source of
# truth, no unit conversion in the comparison.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_scf.py

import os
import pyscf
from pyscf import gto, scf

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR

Z = {"H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9}

MOLECULES = {
    "h2": (0, 1, [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.4)]),
    "lih": (0, 1, [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 3.0139)]),
    "h2o": (
        0,
        1,
        [
            ("O", 0.0, 0.0, 0.2217),
            ("H", 0.0, 1.4309, -0.8867),
            ("H", 0.0, -1.4309, -0.8867),
        ],
    ),
    "nh3": (
        0,
        1,
        [
            ("N", 0.0, 0.0, 0.14),
            ("H", 0.0, 1.772, -0.652),
            ("H", 1.5345, -0.886, -0.652),
            ("H", -1.5345, -0.886, -0.652),
        ],
    ),
    "ch4": (
        0,
        1,
        [
            ("C", 0.0, 0.0, 0.0),
            ("H", 1.186, 1.186, 1.186),
            ("H", 1.186, -1.186, -1.186),
            ("H", -1.186, 1.186, -1.186),
            ("H", -1.186, -1.186, 1.186),
        ],
    ),
    "hf": (0, 1, [("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 1.7325)]),
    "co": (0, 1, [("C", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 2.1316)]),
    "n2": (0, 1, [("N", 0.0, 0.0, 0.0), ("N", 0.0, 0.0, 2.0743)]),
    "h2o2": (
        0,
        1,
        [
            ("O", 0.0, 1.3712, 0.0),
            ("O", 0.0, -1.3712, 0.0),
            ("H", 1.75, 1.8, 0.6),
            ("H", -1.75, -1.8, 0.6),
        ],
    ),
    "c2h4": (
        0,
        1,
        [
            ("C", 0.0, 0.0, 1.2635),
            ("C", 0.0, 0.0, -1.2635),
            ("H", 0.0, 1.7482, 2.3286),
            ("H", 0.0, -1.7482, 2.3286),
            ("H", 0.0, 1.7482, -2.3286),
            ("H", 0.0, -1.7482, -2.3286),
        ],
    ),
    # Open shell (UHF)
    "o2_triplet": (0, 3, [("O", 0.0, 0.0, 0.0), ("O", 0.0, 0.0, 2.2819)]),
    "ch3_doublet": (
        0,
        2,
        [
            ("C", 0.0, 0.0, 0.0),
            ("H", 2.0395, 0.0, 0.0),
            ("H", -1.0197, 1.7662, 0.0),
            ("H", -1.0197, -1.7662, 0.0),
        ],
    ),
}

BASES = [("sto-3g", "sto-3g.g94"), ("def2-svp", "def2-svp.g94")]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for mol_name, (charge, mult, atoms) in MOLECULES.items():
        for basis_name, basis_file in BASES:
            elements = sorted({sym for sym, *_ in atoms})
            basis = parse_g94(os.path.join(DATA_DIR, basis_file), set(elements))
            geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
            mol = gto.M(
                atom=geom,
                basis=basis,
                unit="Bohr",
                charge=charge,
                spin=mult - 1,
                cart=False,
            )
            if mult == 1:
                mf = scf.RHF(mol)
            else:
                mf = scf.UHF(mol)
            mf.conv_tol = 1e-12
            mf.max_cycle = 200
            mf.kernel()
            assert mf.converged, f"{mol_name}/{basis_name} did not converge"
            case = f"{mol_name}_{basis_name.replace('-', '')}"
            path = os.path.join(OUT_DIR, f"{case}_scf.csv")
            with open(path, "w", newline="\n") as fh:
                fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} SCF oracle\n")
                fh.write("# geometry unit: Bohr (exact literals)\n")
                fh.write(f"basis,{basis_name}\n")
                fh.write(f"charge,{charge}\n")
                fh.write(f"mult,{mult}\n")
                fh.write(f"method,{'rhf' if mult == 1 else 'uhf'}\n")
                for sym, x, y, z in atoms:
                    fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
                fh.write(f"e_tot,{mf.e_tot:.14e}\n")
                fh.write(f"e_nuc,{mol.energy_nuc():.14e}\n")
                if mult != 1:
                    s2 = mf.spin_square()[0]
                    fh.write(f"s2,{s2:.10e}\n")
            print(f"{case}: E = {mf.e_tot:.10f}")


if __name__ == "__main__":
    main()
