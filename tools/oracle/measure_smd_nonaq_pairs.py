# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Which element pairs carry a distance-dependent SMD surface-tension term
# OUTSIDE water?
#
# With the CDS radii and the atomic tensions both measured
# (`measure_smd_cds_radii.py`), the atomic model predicts
#   G_CDS = sum_A tau_{Z_A}(solvent) * SASA_A
# exactly for any pair that has no pairwise term. Scanning a diatomic and
# comparing against that prediction therefore isolates the pair term: an
# excess identically zero across the whole scan means "free", anything else
# means the pair must be refused until its shape is determined.
#
# Two solvents with very different descriptors are used so that a pair term
# cannot hide behind a coincidental cancellation in one of them.
import itertools

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
RAD = {1: 1.20, 3: 1.82, 4: 1.53, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
       11: 2.27, 12: 1.73, 13: 1.84, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75,
       19: 2.75, 20: 2.31, 30: 2.10, 32: 2.11, 33: 1.85, 34: 1.90, 35: 1.85,
       50: 2.17, 51: 2.06, 52: 2.06, 53: 1.98}
SYM = {1: "H", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
       11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl",
       19: "K", 20: "Ca", 30: "Zn", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
       50: "Sn", 51: "Sb", 52: "Te", 53: "I"}
TENS = {6: (58.0993, 48.0994, 32.8696), 7: (32.6183, 0.0, 0.0),
        8: (-17.5600, 193.0605, -43.7901), 14: (-18.0401, 0.0, 0.0),
        16: (-33.1703, 0.0, 0.0), 17: (-24.3100, 0.0, 0.0),
        35: (-35.4207, 0.0, 0.0)}
SOLVENTS = ["benzene", "methanol"]
SCAN = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.6, 4.4]


def tau(z, name):
    d = [float(x) for x in smd.solvent_db[name]]
    n, alpha, beta, gamma, _eps, phi, psi = d[0], d[2], d[3], d[4], d[5], d[6], d[7]
    tn, ta, tb = TENS.get(z, (0.0, 0.0, 0.0))
    return tn * n + ta * alpha + tb * beta + 0.35 * gamma - 4.19 * phi ** 2 - 6.68 * psi ** 2


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
        for name in SOLVENTS:
            for R in SCAN:
                spec = f"{SYM[za]} 0 0 0; {SYM[zb]} 0 0 {R}"
                try:
                    mol = gto.M(atom=spec, basis="sto-3g", spin=None, verbose=0)
                    obj = smd.SMD(mol)
                    obj.solvent = name
                    g, _ = smd.get_cds_legacy(obj)
                except Exception:
                    ok = False
                    continue
                c = mol.atom_coords(unit="B") * 0.52917721092
                a = sasa(c, [za, zb])
                pred = tau(za, name) * a[0] + tau(zb, name) * a[1]
                worst = max(worst, abs(g * K * 1000.0 - pred))
        # The Monte-Carlo-free Fibonacci SASA is good to a few tenths of a
        # cal/mol at 40k points; anything under 2 cal/mol is "no term".
        verdict = "free" if (ok and worst < 2.0) else "pair-term"
        print(f"{za},{zb},{worst:.3f},{verdict}")


if __name__ == "__main__":
    main()
