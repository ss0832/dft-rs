# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Heteronuclear pair terms: with sigma_Z known from the isolated-atom scan,
# a diatomic Z-Z' at distance R gives ONE equation for the two pair terms
# sigma_{ZZ'} T(R) and sigma_{Z'Z} T(R). Scanning R maps their sum.
import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}
# isolated tensions in water, measured (cal/(mol A^2))
SIG_W = {1: 48.690, 6: 113.018, 7: 0.0, 8: 0.0, 16: -5.273, 17: 5.874}


def sasa(coords, zs, n=20000):
    g = np.pi * (3 - np.sqrt(5))
    k = np.arange(n)
    z = 1 - (2 * k + 1) / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    d = np.stack([r * np.cos(g * k), r * np.sin(g * k), z], 1)
    w = 4 * np.pi / n
    rad = np.array([RAD[i] + 0.4 for i in zs])
    out = np.zeros(len(zs))
    for i in range(len(zs)):
        p = coords[i] + d * rad[i]
        ok = np.ones(n, bool)
        for j in range(len(zs)):
            if j != i:
                ok &= np.linalg.norm(p - coords[j], axis=1) >= rad[j]
        out[i] = ok.sum() * w * rad[i] ** 2
    return out


for za, zb in [(8, 1), (7, 1), (6, 1), (6, 8), (6, 7), (16, 1)]:
    print(f"--- {SYM[za]}-{SYM[zb]} (water) : excess over sum(sigma_Z A_Z) ---")
    for R in [0.9, 1.1, 1.3, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.5]:
        mol = gto.M(atom=f"{SYM[za]} 0 0 0; {SYM[zb]} 0 0 {R}", basis="sto-3g",
                    spin=None, charge=0, verbose=0)
        c = mol.atom_coords(unit="B") * 0.52917721092
        a = sasa(c, [za, zb])
        obj = smd.SMD(mol)
        obj.solvent = "water"
        try:
            g, _ = smd.get_cds_legacy(obj)
        except Exception as e:
            print("   skip", R, e)
            continue
        base = SIG_W[za] * a[0] + SIG_W[zb] * a[1]
        excess = g * K * 1000.0 - base
        print("  R=%4.1f A=(%7.2f,%7.2f) excess=%+10.1f cal/mol" % (R, a[0], a[1], excess))
