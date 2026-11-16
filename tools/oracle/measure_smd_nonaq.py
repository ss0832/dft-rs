# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Measure the NON-AQUEOUS SMD atomic surface tensions by regression rather
# than transcription.
#
# For a single isolated atom there is no pairwise partner, so
#   G_CDS / A_Z = sigma_Z + sigma_M
# and SMD builds both sides from the solvent descriptors linearly:
#   sigma_Z    = t_Z^n n + t_Z^a alpha + t_Z^b beta
#   sigma_M    = t^g gamma/gamma_0 + t^p phi^2 + t^q psi^2
# so the whole model is linear in the unknowns and can be recovered from
# enough (element, solvent) measurements by least squares. Water is
# EXCLUDED: it uses the separate aqueous column, already measured.
#
# Output: one CSV row per element with its three coefficients, plus the
# three shared molecular coefficients, and the residual of the fit.
import sys

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

HARTREE2KCAL = 627.5094740631
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
       14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br"}


def descriptors(name):
    """(n, alpha, beta, gamma, phi^2, psi^2) from the reference table.

    Layout confirmed against known solvents: field 1 is the refractive
    index, 2/3 the Abraham acidity/basicity, 4 the macroscopic surface
    tension, 5 the dielectric constant, 6/7 the aromaticity and
    electronegative halogenicity (unsquared; SMD uses their squares).
    """
    d = smd.solvent_db.get(name)
    if d is None:
        return None
    n, a, b, g, _eps, p, q = (float(x) for x in d[1:8])
    return n, a, b, g, p * p, q * q


def measured(z, name):
    mol = gto.M(atom=f"{SYM[z]} 0 0 0", basis="sto-3g", spin=None, verbose=0)
    obj = smd.SMD(mol)
    obj.solvent = name
    g, _ = smd.get_cds_legacy(obj)
    area = 4 * np.pi * (RAD[z] + 0.4) ** 2
    return g * HARTREE2KCAL * 1000.0 / area


def main():
    rows = []
    for name in smd.solvent_db:
        if name == "water":
            continue  # separate aqueous column, already measured
        d = descriptors(name)
        if d is not None:
            rows.append((name, d))
    print(f"# solvents used: {len(rows)}", file=sys.stderr)

    # The molecular term multiplies the SAME total area as the atomic ones,
    # so sigma_M can be folded into every atomic tension without changing
    # any energy: tau_Z = sigma_Z + sigma_M. That leaves ONE independent
    # linear model per element over the six descriptors, and no assumption
    # about which descriptors SMD routes to which block.
    zs = sorted(SYM)
    print("# z,t_n,t_alpha,t_beta,t_gamma,t_phi2,t_psi2,maxresid,rms")
    for z in zs:
        A, y = [], []
        for name, d in rows:
            try:
                y.append(measured(z, name))
            except Exception:
                continue
            A.append(d)
        A = np.asarray(A)
        y = np.asarray(y)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = A @ sol - y
        print(",".join(f"{v:.6f}" for v in sol)
              + f",{np.abs(r).max():.6f},{np.sqrt((r ** 2).mean()):.6f}",
              file=sys.stderr)
        print(f"{z}," + ",".join(f"{v:.6f}" for v in sol)
              + f",{np.abs(r).max():.6f},{np.sqrt((r ** 2).mean()):.6f}")


if __name__ == "__main__":
    main()
