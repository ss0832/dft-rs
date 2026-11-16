# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover the SMD surface-tension parameterization by least squares against
# the SMD reference implementation, the same methodology the project uses
# for the B97/M06 series coefficients (G_CDS is LINEAR in the tensions).
#
# Model under test:
#   G_CDS = sum_A sigma_A * SASA_A  +  sigma_M * sum_A SASA_A
#   sigma_A     = atomic tension of element Z(A) (+ pairwise terms)
#   SASA_A      = solvent-accessible area with radius R_Z + 0.4 Angstrom
# Stage 1 (this script): is an ATOMIC-ONLY model enough? Fit one tension per
# (element, solvent) independently -- the most flexible atomic model -- and
# look at the residual. A large residual proves the pairwise sigma_ZZ'
# terms are needed (and by how much).

import numpy as np
from pyscf import gto
from pyscf.solvent import smd

BOHR = 0.52917721092
HARTREE2KCAL = 627.5094740631

# SMD intrinsic Coulomb radii (Angstrom) + 0.4 solvent radius.
# CDS radii MEASURED by measure_smd_cds_radii.py (Bondi set); the SMD
# intrinsic Coulomb radii belong to the ELECTROSTATICS and are wrong here.
RAD = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
R_SOLV = 0.4

MOLS = {
    "water": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    "methane": "C 0 0 0; H 0.6276 0.6276 0.6276; H 0.6276 -0.6276 -0.6276; "
    "H -0.6276 0.6276 -0.6276; H -0.6276 -0.6276 0.6276",
    "methanol": "C -0.0503 0.6685 0; O -0.0503 -0.7585 0; H -1.0807 1.0417 0; "
    "H 0.465 1.0417 0.8924; H 0.465 1.0417 -0.8924; H 0.8544 -1.0417 0",
    "ammonia": "N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; H -0.81 -0.47 -0.27",
    "ethane": "C 0 0 0.765; C 0 0 -0.765; H 0 1.017 1.16; H 0.881 -0.509 1.16; "
    "H -0.881 -0.509 1.16; H 0 -1.017 -1.16; H 0.881 0.509 -1.16; H -0.881 0.509 -1.16",
    "hydrogen fluoride": "F 0 0 0; H 0 0 0.917",
    "hydrogen sulfide": "S 0 0 0; H 0 0.96 0.94; H 0 -0.96 0.94",
    "chloromethane": "C 0 0 0; Cl 0 0 1.781; H 1.028 0 -0.363; "
    "H -0.514 0.89 -0.363; H -0.514 -0.89 -0.363",
    "formaldehyde": "C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588",
    "ethanol": "C 1.168 -0.4 0; C 0 0.554 0; O -1.168 -0.244 0; H 2.096 0.174 0; "
    "H 1.14 -1.03 0.89; H 1.14 -1.03 -0.89; H 0.03 1.19 0.89; H 0.03 1.19 -0.89; "
    "H -1.93 0.34 0",
}
SOLVENTS = [
    "water", "methanol", "ethanol", "acetone", "acetonitrile",
    "dimethylsulfoxide", "benzene", "toluene", "chloroform",
    "carbon tetrachloride", "1,4-dioxane", "cyclohexane", "n-hexane", "pyridine",
]


def sasa(coords_ang, zs, n=6000):
    """Per-atom solvent-accessible area (Angstrom^2), golden-spiral sampling."""
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
        pts = coords_ang[ia] + dirs * radii[ia]
        exposed = np.ones(n, dtype=bool)
        for jb in range(len(zs)):
            if jb == ia:
                continue
            d = np.linalg.norm(pts - coords_ang[jb], axis=1)
            exposed &= d >= radii[jb]
        out[ia] = exposed.sum() * w * radii[ia] ** 2
    return out


rows = []
for mname, geom in MOLS.items():
    mol = gto.M(atom=geom, basis="sto-3g", unit="Angstrom")
    zs = list(mol.atom_charges())
    coords = mol.atom_coords(unit="B") * BOHR
    areas = sasa(coords, zs)
    for solv in SOLVENTS:
        obj = smd.SMD(mol)
        obj.solvent = solv
        try:
            g, _ = smd.get_cds_legacy(obj)
        except Exception:
            continue
        rows.append((mname, solv, zs, areas, g * HARTREE2KCAL))

print("data rows:", len(rows))
elements = sorted({z for _, _, zs, _, _ in rows for z in zs})
print("elements:", elements)

# Stage 1: independent tension per (element, solvent) + a molecular term.
for solv in SOLVENTS:
    sub = [r for r in rows if r[1] == solv]
    if len(sub) < 3:
        continue
    els = sorted({z for _, _, zs, _, _ in sub for z in zs})
    a = np.zeros((len(sub), len(els) + 1))
    b = np.zeros(len(sub))
    for i, (_, _, zs, areas, g) in enumerate(sub):
        for z, ar in zip(zs, areas):
            a[i, els.index(z)] += ar
        a[i, -1] = areas.sum()
        b[i] = g
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    resid = a @ sol - b
    print(
        "%-22s n=%2d  max|resid| = %7.4f kcal/mol   rms = %7.4f"
        % (solv, len(sub), np.abs(resid).max(), np.sqrt((resid**2).mean()))
    )
    if solv == "water":
        for z, s in zip(els, sol[:-1]):
            print("      sigma(Z=%2d) = %9.3f cal/(mol A^2)" % (z, s))
        print("      sigma_M      = %9.3f" % sol[-1])
