# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Transition-metal ECP sweep (user order 2026-08-04):
# 1) The full ECP integral matrix <mu|U_M|nu> for EVERY 3d metal Sc..Zn
#    plus Pd, on an M-H probe system, against PySCF's `ECPscalar` - a
#    state-free gate that exercises every semi-local channel (lmax,
#    r-powers, exponents) per element.
# 2) Closed-shell SCF energy gates: TiCl4 (d0), CuH (d10), ZnH2, Pd atom
#    (d10 1S) - written in the *_ecp.csv format the p45 gate discovers.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_ecp_metals.py

import os
import numpy as np
import pyscf
from pyscf import gto, scf

from gen_ecp import parse_g94_with_ecp
from gen_integrals import DATA_DIR, OUT_DIR, dftrs_ao_permutation

METALS = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Pd"]
Z = {
    "H": 1, "Cl": 17, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Pd": 46,
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = set(METALS) | {"H", "Cl"}
    basis, ecps = parse_g94_with_ecp(os.path.join(DATA_DIR, "lanl2dz.g94"), wanted)
    for m in METALS:
        assert m in ecps, f"no ECP parsed for {m}"

    # ---- 1) ECP matrices on M-H probes ----
    for m in METALS:
        nval = Z[m] - ecps[m][0] + 1  # valence electrons incl. the H
        spin = nval % 2
        mol = gto.M(
            atom=f"{m} 0 0 0; H 0 0 3.0",
            basis={m: basis[m], "H": basis["H"]},
            ecp={m: ecps[m]},
            unit="Bohr",
            charge=0,
            spin=spin,
            cart=False,
        )
        mat = mol.intor("ECPscalar")
        perm = dftrs_ao_permutation(mol)
        mat = mat[np.ix_(perm, perm)]
        fname = f"{m.lower()}h_lanl2dz_ecpmat.csv"
        with open(os.path.join(OUT_DIR, fname), "w", newline="\n") as fh:
            fh.write(
                f"# PROVENANCE: PySCF {pyscf.__version__} ECPscalar matrix "
                f"({m}-H probe, basis+ECP from dft-rs lanl2dz.g94, dft-rs AO order)\n"
            )
            fh.write("basis,lanl2dz\n")
            fh.write(f"atom,{Z[m]},0.0,0.0,0.0\n")
            fh.write(f"atom,{Z['H']},0.0,0.0,3.0\n")
            for r in range(mat.shape[0]):
                fh.write(
                    "row," + str(r) + ","
                    + ",".join(f"{v:.14e}" for v in mat[r]) + "\n"
                )
        print(f"{m}-H ECP matrix: nao = {mat.shape[0]}, |max| = {np.abs(mat).max():.4f}")

    # ---- 2) Closed-shell SCF energy gates ----
    td = 1.0 / np.sqrt(3.0)
    cases = [
        ("ticl4", [("Ti", 0, 0, 0)]
         + [("Cl", 4.101 * sx * td, 4.101 * sy * td, 4.101 * sz * td)
            for sx, sy, sz in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]),
        ("cuh", [("Cu", 0, 0, 0), ("H", 0, 0, 2.765)]),
        ("znh2", [("Zn", 0, 0, 0), ("H", 0, 0, 2.929), ("H", 0, 0, -2.929)]),
        ("pd_atom", [("Pd", 0, 0, 0)]),
    ]
    for tag, atoms in cases:
        syms = {s for s, *_ in atoms}
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom,
            basis={s: basis[s] for s in syms},
            ecp={s: ecps[s] for s in syms if s in ecps},
            unit="Bohr",
            charge=0,
            spin=0,
            cart=False,
        )
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.max_cycle = 300
        mf.kernel()
        assert mf.converged, tag
        with open(
            os.path.join(OUT_DIR, f"{tag}_lanl2dz_hf_ecp.csv"), "w", newline="\n"
        ) as fh:
            fh.write(
                f"# PROVENANCE: PySCF {pyscf.__version__} ECP oracle "
                "(closed-shell transition-metal case, values only)\n"
            )
            fh.write("basis,lanl2dz\n")
            fh.write("functional,hf\n")
            fh.write("charge,0\n")
            fh.write("mult,1\n")
            for s, x, y, z in atoms:
                fh.write(f"atom,{Z[s]},{x},{y},{z}\n")
            fh.write(f"e_tot,{mf.e_tot:.14e}\n")
        print(f"{tag}: E = {mf.e_tot:.10f}")


if __name__ == "__main__":
    main()
