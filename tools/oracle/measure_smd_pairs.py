# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Measure every aqueous pairwise surface-tension term on a fine distance
# grid and write the raw curves to tools/oracle/smd_pair_curves.csv so the
# functional form T(R) can be identified offline.
#
#   excess(R) = G_CDS(R) - sum_Z sigma_Z A_Z(R)
#             = sigma_ab T(R) A_a(R) + sigma_ba T(R) A_b(R)
import os

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
# Only the pairs measured to CARRY a term are worth scanning; the rest are
# exact already (measure_smd_water_pairs.py).
PAIRS = [(1, 6), (6, 6), (6, 7), (6, 8), (7, 8), (8, 15)]
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
SIG = {1: 48.690, 6: 129.740, 7: 0.0, 8: 0.0, 9: 38.180, 14: 0.0, 15: 0.0,
       16: -9.100, 17: 9.820, 35: -8.720}
OUT = os.path.join(os.path.dirname(__file__), "smd_pair_curves.csv")


def sasa(c, zs, n=40000):
    g = np.pi * (3 - np.sqrt(5))
    k = np.arange(n)
    z = 1 - (2 * k + 1) / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    d = np.stack([r * np.cos(g * k), r * np.sin(g * k), z], 1)
    w = 4 * np.pi / n
    rad = np.array([RAD[i] + 0.4 for i in zs])
    out = np.zeros(len(zs))
    for i in range(len(zs)):
        p = c[i] + d * rad[i]
        ok = np.ones(n, bool)
        for j in range(len(zs)):
            if j != i:
                ok &= np.linalg.norm(p - c[j], axis=1) >= rad[j]
        out[i] = ok.sum() * w * rad[i] ** 2
    return out


grid = [round(x, 2) for x in np.arange(0.7, 6.05, 0.1)]
rows = []
for za, zb in PAIRS:
    for R in grid:
        mol = gto.M(atom=f"{SYM[za]} 0 0 0; {SYM[zb]} 0 0 {R}", basis="sto-3g",
                    spin=None, verbose=0)
        c = mol.atom_coords(unit="B") * 0.52917721092
        a = sasa(c, [za, zb])
        obj = smd.SMD(mol)
        obj.solvent = "water"
        try:
            g, _ = smd.get_cds_legacy(obj)
        except Exception:
            continue
        excess = g * K * 1000.0 - (SIG[za] * a[0] + SIG[zb] * a[1])
        rows.append((SYM[za], SYM[zb], R, a[0], a[1], excess))

with open(OUT, "w", newline="\n") as fh:
    fh.write("# PROVENANCE: SMD reference implementation (MNSOL via PySCF) used\n")
    fh.write("#   as an ORACLE. Aqueous pairwise excess over the atomic tension sum.\n")
    fh.write("# Columns: elem_a,elem_b,R_angstrom,area_a,area_b,excess_cal_per_mol\n")
    for r in rows:
        fh.write("%s,%s,%.2f,%.4f,%.4f,%.4f\n" % r)
print("wrote", OUT, len(rows), "points")
