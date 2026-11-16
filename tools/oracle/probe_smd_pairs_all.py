# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Which element pairs carry a distance-dependent aqueous tension term?
# Excess over sum_Z sigma_Z A_Z at a short and a long separation.
import itertools

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
SIG = {1: 48.690, 6: 113.018, 7: 0.0, 8: 0.0, 9: 29.428, 15: 0.0,
       16: -5.273, 17: 5.874, 35: -3.687}


def sasa(c, zs, n=20000):
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


zs_all = sorted(SYM)
print("pair   excess@1.5A   excess@3.0A   (cal/mol)   -> term?")
free = []
for za, zb in itertools.combinations_with_replacement(zs_all, 2):
    vals = []
    for R in (1.5, 3.0):
        mol = gto.M(atom=f"{SYM[za]} 0 0 0; {SYM[zb]} 0 0 {R}", basis="sto-3g",
                    spin=None, verbose=0)
        c = mol.atom_coords(unit="B") * 0.52917721092
        a = sasa(c, [za, zb])
        obj = smd.SMD(mol)
        obj.solvent = "water"
        try:
            g, _ = smd.get_cds_legacy(obj)
        except Exception:
            vals.append(float("nan"))
            continue
        vals.append(g * K * 1000.0 - (SIG[za] * a[0] + SIG[zb] * a[1]))
    tag = "TERM" if max(abs(v) for v in vals if v == v) > 1.0 else "free"
    if tag == "free":
        free.append((za, zb))
    print("%-6s %+12.1f %+12.1f    %s" % (f"{SYM[za]}-{SYM[zb]}", vals[0], vals[1], tag))
print("\nterm-free pairs:", [(SYM[a], SYM[b]) for a, b in free])
print("rust:", " | ".join(f"({a}, {b})" for a, b in free))
