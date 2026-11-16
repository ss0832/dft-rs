# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Organometallic oracle: Ni(CO)4 (Td, 9 atoms) with LANL2DZ (Ni ECP,
# C/O all-electron) - frozen energy, analytic gradient, and Hessian
# (HF; B3LYP energy+gradient). Basis AND ECP parsed from the bundled g94.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_organometallic.py

import os
import numpy as np
import pyscf
from pyscf import dft, gto, scf
import pyscf.grad  # noqa: F401  (registers mf.Gradients)
import pyscf.hessian  # noqa: F401  (registers mf.Hessian)

from gen_ecp import parse_g94_with_ecp
from gen_integrals import DATA_DIR, OUT_DIR

Z = {"Ni": 28, "C": 6, "O": 8}

# Td geometry (Bohr): Ni-C = 3.473, C-O = 2.156 along (±1,±1,±1)/sqrt(3).
U3 = 1.0 / np.sqrt(3.0)
DIRS = [(U3, U3, U3), (U3, -U3, -U3), (-U3, U3, -U3), (-U3, -U3, U3)]
D_NIC = 3.473
D_CO = 2.156


def geometry():
    atoms = [("Ni", 0.0, 0.0, 0.0)]
    for d in DIRS:
        atoms.append(("C", D_NIC * d[0], D_NIC * d[1], D_NIC * d[2]))
    for d in DIRS:
        r = D_NIC + D_CO
        atoms.append(("O", r * d[0], r * d[1], r * d[2]))
    return atoms


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    basis, ecps = parse_g94_with_ecp(
        os.path.join(DATA_DIR, "lanl2dz.g94"), {"Ni", "C", "O"}
    )
    atoms = geometry()
    geom = "; ".join(f"{s} {x:.16e} {y:.16e} {z:.16e}" for s, x, y, z in atoms)
    mol = gto.M(
        atom=geom,
        basis={s: basis[s] for s in {"Ni", "C", "O"}},
        ecp={"Ni": ecps["Ni"]},
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
    )
    for func, xc in [("hf", None), ("b3lyp", "b3lypg")]:
        if xc is None:
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = xc
            mf.grids.level = 8
        mf.conv_tol = 1e-12
        mf.max_cycle = 300
        mf.kernel()
        assert mf.converged, func
        grad = mf.Gradients().kernel()
        hess = None
        if xc is None:
            hess = mf.Hessian().kernel()  # (N, N, 3, 3)
        case = f"nico4_lanl2dz_{func}"
        with open(os.path.join(OUT_DIR, f"{case}_om.csv"), "w", newline="\n") as fh:
            fh.write(
                f"# PROVENANCE: PySCF {pyscf.__version__} organometallic oracle "
                "(Ni(CO)4, basis+ECP from dft-rs lanl2dz.g94)\n"
            )
            fh.write("# geometry unit: Bohr (exact literals)\n")
            fh.write("basis,lanl2dz\n")
            fh.write(f"functional,{func}\n")
            fh.write("charge,0\n")
            fh.write("mult,1\n")
            for s, x, y, z in atoms:
                fh.write(f"atom,{Z[s]},{x:.16e},{y:.16e},{z:.16e}\n")
            fh.write(f"e_tot,{mf.e_tot:.14e}\n")
            for a in range(len(atoms)):
                fh.write(
                    f"grad,{a},{grad[a][0]:.12e},{grad[a][1]:.12e},{grad[a][2]:.12e}\n"
                )
            if hess is not None:
                n = len(atoms)
                for a in range(n):
                    for x in range(3):
                        row = [hess[a][b][x][y] for b in range(n) for y in range(3)]
                        fh.write(
                            "hess," + str(3 * a + x) + ","
                            + ",".join(f"{v:.10e}" for v in row) + "\n"
                        )
        print(f"{case}: E = {mf.e_tot:.10f}  |grad|max = {np.abs(grad).max():.3e}")


if __name__ == "__main__":
    main()
