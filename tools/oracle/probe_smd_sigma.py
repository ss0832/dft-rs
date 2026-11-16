# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Measure SMD's ISOLATED atomic surface tensions directly: a single atom in
# a cavity has no pairwise partner, so sigma_Z = G_CDS / (4 pi (R_Z+0.4)^2).
# Repeated per solvent this also exposes the descriptor scaling.
import numpy as np
from pyscf import gto
from pyscf.solvent import smd

HARTREE2KCAL = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
SOLVENTS = ["water"]

print("sigma_Z (cal/(mol A^2)) from a single isolated atom")
print("%-4s %10s" % ("Z", "area"), "".join("%12s" % s[:11] for s in SOLVENTS))
for z, sym in SYM.items():
    area = 4 * np.pi * (RAD[z] + 0.4) ** 2
    mol = gto.M(atom=f"{sym} 0 0 0", basis="sto-3g", spin=None)
    vals = []
    for solv in SOLVENTS:
        obj = smd.SMD(mol)
        obj.solvent = solv
        try:
            g, _ = smd.get_cds_legacy(obj)
            vals.append(g * HARTREE2KCAL * 1000.0 / area)
        except Exception:
            vals.append(float("nan"))
    print("%-4s %10.3f" % (sym, area), "".join("%12.3f" % v for v in vals))
