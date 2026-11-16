# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# C-PCM solvation oracle: HF/STO-3G water in water (eps = 78.3553).
import os

import pyscf
from pyscf import gto, scf
from pyscf.solvent import pcm

from gen_integrals import parse_g94, DATA_DIR

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "oracle_data")

ATOMS = [
    ("O", 0.0, 0.05, 0.2217),
    ("H", 0.03, 1.4309, -0.8867),
    ("H", 0.0, -1.4309, -0.8867),
]
basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), {"O", "H"})
geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in ATOMS)
mol = gto.M(atom=geom, basis=basis, unit="Bohr", cart=False)

mf_gas = scf.RHF(mol)
mf_gas.conv_tol = 1e-12
mf_gas.kernel()

solvent = pcm.PCM(mol)
solvent.method = "C-PCM"
solvent.eps = 78.3553
mf = scf.RHF(mol).PCM(solvent)
mf.conv_tol = 1e-12
mf.kernel()

e_gas = mf_gas.e_tot
e_sol = mf.e_tot
print("E(gas)   =", e_gas)
print("E(solv)  =", e_sol)
print("dG_elst  =", e_sol - e_gas, "Hartree =", (e_sol - e_gas) * 627.5094740631, "kcal/mol")

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "pcm_water_cpcm.csv"), "w", newline="\n") as fh:
    fh.write(
        "# PROVENANCE: PySCF %s C-PCM oracle (values only); HF/STO-3G water,\n"
        % pyscf.__version__
    )
    fh.write("# eps = 78.3553, PySCF default cavity (Lebedev 302, modified Bondi x 1.2)\n")
    fh.write("e_gas,%.12f\n" % e_gas)
    fh.write("e_solvated,%.12f\n" % e_sol)
    fh.write("dg_elst,%.12f\n" % (e_sol - e_gas))
