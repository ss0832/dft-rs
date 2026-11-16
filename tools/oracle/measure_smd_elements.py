# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Which elements does the SMD CDS term actually cover, and what are their
# radii and tensions?
#
# Rather than assuming the published element list, ask the reference: for
# each Z, (1) bisect the hydrogen-probe contact distance to read the CDS
# radius off geometrically, and (2) regress the isolated-atom G_CDS/area
# against the solvent descriptors over every non-aqueous solvent, plus read
# the aqueous value directly. An element the reference does not know either
# raises or returns an identically zero surface tension with the shared
# molecular block still present — both of which this reports rather than
# guesses.
import sys

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

K = 627.5094740631
PROBE_SOLVENT = "benzene"
R_H = 1.20

# Everything through Xe plus the common heavier main-group elements; the
# script keeps whatever the reference accepts.
CANDIDATES = list(range(54, 89))  # Xe..Ra; 1..53 already measured
SKIP = set()  # probe the noble gases too; they are closed shell and legal


def sym(z):
    return gto.mole._std_symbol(z)


def cds(spec, solvent):
    # The CDS term is purely geometric -- it never touches the wavefunction --
    # so the basis only has to EXIST. STO-3G stops at xenon, which would
    # otherwise cap the sweep there; one s function per element removes that
    # limit without changing any answer.
    syms = {t.split()[0] for t in spec.split(";")}
    dummy = gto.basis.parse("X    S\n      1.0                    1.0\n")
    mol = gto.M(atom=spec, basis={s: dummy for s in syms}, spin=None, verbose=0)
    obj = smd.SMD(mol)
    obj.solvent = solvent
    g, _ = smd.get_cds_legacy(obj)
    return g * K


def contact_distance(z, tol=1e-4):
    s = sym(z)
    far = cds(f"{s} 0 0 0; H 0 0 40.0", PROBE_SOLVENT)
    touching = lambda d: abs(cds(f"{s} 0 0 0; H 0 0 {d}", PROBE_SOLVENT) - far) > 1e-9
    lo, hi = 0.4, 20.0
    if not touching(lo) or touching(hi):
        return None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if touching(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    rows = []
    for name, d in smd.solvent_db.items():
        if name == "water":
            continue
        v = [float(x) for x in d]
        rows.append((name, (v[0], v[2], v[3], v[4], v[6] ** 2, v[7] ** 2)))

    print("# z,symbol,r_cds,sigma_water,t_n,t_alpha,t_beta,resid_max")
    for z in CANDIDATES:
        if z in SKIP:
            continue
        s = sym(z)
        try:
            dcon = contact_distance(z)
        except Exception as exc:
            print(f"# {s}: probe failed ({exc})", file=sys.stderr)
            continue
        if dcon is None:
            print(f"# {s}: no contact bracket", file=sys.stderr)
            continue
        r = dcon - R_H - 0.8
        area = 4 * np.pi * (r + 0.4) ** 2
        try:
            sig_w = cds(f"{s} 0 0 0", "water") * 1000.0 / area
        except Exception:
            sig_w = float("nan")
        A, y = [], []
        for name, dd in rows:
            try:
                y.append(cds(f"{s} 0 0 0", name) * 1000.0 / area)
            except Exception:
                continue
            A.append(dd)
        if len(y) < 10:
            print(f"# {s}: too few solvents", file=sys.stderr)
            continue
        A = np.asarray(A)
        y = np.asarray(y)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = np.abs(A @ sol - y).max()
        print(f"{z},{s},{r:.4f},{sig_w:.4f},{sol[0]:.4f},{sol[1]:.4f},"
              f"{sol[2]:.4f},{resid:.6f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
