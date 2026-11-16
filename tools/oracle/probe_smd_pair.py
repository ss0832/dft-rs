# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Determine SMD's pairwise switching function EXPERIMENTALLY. The CDS term
# is purely geometric, so any (even unphysical) geometry is fair game: for a
# HOMONUCLEAR diatomic both atoms share one tension, hence
#   sigma(R) = G_CDS(R) / (A_1(R) + A_2(R)) = sigma_Z + sigma_ZZ * T(R),
# and scanning R maps T(R) directly.
import numpy as np
from pyscf import gto
from pyscf.solvent import smd

BOHR = 0.52917721092
HARTREE2KCAL = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
R_SOLV = 0.4


def sasa(coords, zs, n=20000):
    golden = np.pi * (3 - np.sqrt(5))
    k = np.arange(n)
    z = 1 - (2 * k + 1) / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    phi = golden * k
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    w = 4 * np.pi / n
    radii = np.array([RAD[zi] + R_SOLV for zi in zs])
    out = np.zeros(len(zs))
    for ia in range(len(zs)):
        pts = coords[ia] + dirs * radii[ia]
        ok = np.ones(n, dtype=bool)
        for jb in range(len(zs)):
            if jb != ia:
                ok &= np.linalg.norm(pts - coords[jb], axis=1) >= radii[jb]
        out[ia] = ok.sum() * w * radii[ia] ** 2
    return out


for sym, z in [("C", 6), ("N", 7), ("O", 8), ("H", 1)]:
    print(f"=== {sym}2 homonuclear scan, solvent = water ===")
    for R in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 5.0, 6.0, 8.0]:
        mol = gto.M(atom=f"{sym} 0 0 0; {sym} 0 0 {R}", basis="sto-3g", unit="Angstrom",
                    spin=None if z % 2 == 0 else 0)
        coords = mol.atom_coords(unit="B") * BOHR
        areas = sasa(coords, [z, z])
        obj = smd.SMD(mol)
        obj.solvent = "water"
        g, _ = smd.get_cds_legacy(obj)
        sigma = g * HARTREE2KCAL * 1000.0 / areas.sum()  # cal/(mol A^2)
        print("  R=%4.1f  A_tot=%8.3f  G=%+8.4f kcal/mol  sigma=%+9.3f" %
              (R, areas.sum(), g * HARTREE2KCAL, sigma))
