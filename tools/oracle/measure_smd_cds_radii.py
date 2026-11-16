# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Measure the radii the SMD CDS term uses for its solvent-accessible
# surface, and check whether the atomic surface tensions really are linear
# in the tabulated solvent descriptors.
#
# RADII, geometrically: put a probe H atom at distance d from atom Z. The
# two expanded spheres (R + 0.4 A) start to occlude each other exactly at
# d = R_Z + R_H + 0.8, so G_CDS(d) is constant beyond that point and bends
# below it. Bisecting on "does G_CDS differ from its asymptote" reads the
# radius off without assuming any tension value.
#
# LINEARITY: for each element, least-squares fit G_CDS/area against ALL the
# raw descriptor columns. A vanishing residual proves the model is linear in
# those columns (and identifies which refractive-index column is the live
# one); a large residual means the model is something else, and is reported
# rather than papered over.
import sys

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

HARTREE2KCAL = 627.5094740631
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
       14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}
PROBE_SOLVENT = "benzene"


def cds(atom_spec, solvent):
    mol = gto.M(atom=atom_spec, basis="sto-3g", spin=None, verbose=0)
    obj = smd.SMD(mol)
    obj.solvent = solvent
    g, _ = smd.get_cds_legacy(obj)
    return g * HARTREE2KCAL


def contact_distance(z, tol=1e-4):
    """Bisect the separation at which the probe first feels atom Z."""
    sym = SYM[z]
    far = cds(f"{sym} 0 0 0; H 0 0 40.0", PROBE_SOLVENT)
    touching = lambda d: abs(cds(f"{sym} 0 0 0; H 0 0 {d}", PROBE_SOLVENT) - far) > 1e-9
    lo, hi = 0.5, 20.0
    assert touching(lo) and not touching(hi), f"bracket failed for {sym}"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if touching(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    # --- radii -------------------------------------------------------
    # The probe is hydrogen, so the contact distance is R_Z + R_H + 0.8.
    # Solve for R_H first from the homonuclear H2 case (2 R_H + 0.8).
    d_hh = contact_distance(1)
    r_h = 0.5 * (d_hh - 0.8)
    print(f"# R_H = {r_h:.4f} (from H...H contact at {d_hh:.4f})")
    print("# z,r_cds")
    radii = {1: r_h}
    for z in sorted(SYM):
        if z == 1:
            continue
        d = contact_distance(z)
        radii[z] = d - r_h - 0.8
    for z in sorted(radii):
        print(f"{z},{radii[z]:.4f}")

    # --- linearity ---------------------------------------------------
    rows = []
    for name, d in smd.solvent_db.items():
        if name == "water":
            continue
        rows.append((name, [float(x) for x in d]))
    n_desc = len(rows[0][1])
    print(f"# descriptor columns available: {n_desc}", file=sys.stderr)
    print("# z,resid_max,resid_rms," + ",".join(f"c{k}" for k in range(n_desc + 2)))
    for z in sorted(SYM):
        area = 4 * np.pi * (radii[z] + 0.4) ** 2
        A, y = [], []
        for name, d in rows:
            try:
                y.append(cds(f"{SYM[z]} 0 0 0", name) * 1000.0 / area)
            except Exception:
                continue
            # raw columns, plus the squares of the last two (SMD squares
            # the aromaticity and halogenicity descriptors).
            A.append(d + [d[-2] ** 2, d[-1] ** 2])
        A = np.asarray(A)
        y = np.asarray(y)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = A @ sol - y
        print(f"{z},{np.abs(r).max():.6f},{np.sqrt((r ** 2).mean()):.6f},"
              + ",".join(f"{v:.6f}" for v in sol))


if __name__ == "__main__":
    main()
