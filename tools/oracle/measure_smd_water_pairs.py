# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Which element pairs carry a distance-dependent SMD surface-tension term
# IN WATER? Same construction as measure_smd_nonaq_pairs.py, but with the
# aqueous atomic tensions.
#
# This is a RE-determination: the first pass divided by areas built from
# the intrinsic Coulomb radii, so both the tensions and the excess were
# wrong for every element whose two radius sets differ (C, F, Si, S, Cl,
# Br). The pairs it declared free happened to involve only H, O, N and P,
# where the two sets coincide, so that conclusion survived - but nothing
# else from it can be trusted.
import itertools

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set).
RAD = {1: 1.20, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
       11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
       19: 2.75, 20: 2.31, 30: 2.10, 32: 2.11, 33: 1.85, 34: 1.90, 35: 1.85,
       50: 2.17, 51: 2.06, 52: 2.06, 53: 1.98}
SYM = {1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
       11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
       19: "K", 20: "Ca", 30: "Zn", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
       50: "Sn", 51: "Sb", 52: "Te", 53: "I"}
SIG = {1: 48.690, 6: 129.740, 7: 0.0, 8: 0.0, 9: 38.180, 16: -9.100,
       17: 9.820, 35: -8.720}
SCAN = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.6, 4.4]


def sasa(coords, zs, n=40000):
    g = np.pi * (3 - np.sqrt(5))
    k = np.arange(n)
    zc = 1 - (2 * k + 1) / n
    r = np.sqrt(np.maximum(1 - zc * zc, 0))
    d = np.stack([r * np.cos(g * k), r * np.sin(g * k), zc], 1)
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


def main():
    print("# za,zb,max_excess_cal_per_mol,verdict")
    for za, zb in itertools.combinations_with_replacement(sorted(SYM), 2):
        worst = 0.0
        ok = True
        for R in SCAN:
            spec = f"{SYM[za]} 0 0 0; {SYM[zb]} 0 0 {R}"
            try:
                mol = gto.M(atom=spec, basis="sto-3g", spin=None, verbose=0)
                obj = smd.SMD(mol)
                obj.solvent = "water"
                g, _ = smd.get_cds_legacy(obj)
            except Exception:
                ok = False
                continue
            c = mol.atom_coords(unit="B") * 0.52917721092
            a = sasa(c, [za, zb])
            pred = SIG.get(za, 0.0) * a[0] + SIG.get(zb, 0.0) * a[1]
            worst = max(worst, abs(g * K * 1000.0 - pred))
        verdict = "free" if (ok and worst < 2.0) else "pair-term"
        print(f"{za},{zb},{worst:.3f},{verdict}")


if __name__ == "__main__":
    main()
