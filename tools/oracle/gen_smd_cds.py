# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Frozen SMD G_CDS reference values across solvents and molecules, so the
# full (pairwise + molecular-term) SMD surface-tension implementation can
# be gated numerically. Requires a PySCF built with the SMD interface.
import os

import pyscf
from pyscf import gto
from pyscf.solvent import smd

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "oracle_data")

MOLECULES = {
    # name: geometry in ANGSTROM (SMD's CDS term is geometry-only)
    "water": [("O", 0.0, 0.0, 0.0), ("H", 0.0, 0.757, 0.587), ("H", 0.0, -0.757, 0.587)],
    "methane": [
        ("C", 0.0, 0.0, 0.0),
        ("H", 0.6276, 0.6276, 0.6276),
        ("H", 0.6276, -0.6276, -0.6276),
        ("H", -0.6276, 0.6276, -0.6276),
        ("H", -0.6276, -0.6276, 0.6276),
    ],
    "methanol": [
        ("C", -0.0503, 0.6685, 0.0),
        ("O", -0.0503, -0.7585, 0.0),
        ("H", -1.0807, 1.0417, 0.0),
        ("H", 0.4650, 1.0417, 0.8924),
        ("H", 0.4650, 1.0417, -0.8924),
        ("H", 0.8544, -1.0417, 0.0),
    ],
}
SOLVENTS = [
    "water",
    "methanol",
    "ethanol",
    "acetone",
    "acetonitrile",
    "dimethylsulfoxide",
    "benzene",
    "toluene",
    "chloroform",
    "carbon tetrachloride",
    "1,4-dioxane",
    "cyclohexane",
]

rows = []
for mname, atoms in MOLECULES.items():
    geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    mol = gto.M(atom=geom, basis="sto-3g", unit="Angstrom")
    for solv in SOLVENTS:
        obj = smd.SMD(mol)
        obj.solvent = solv
        try:
            g, _ = smd.get_cds_legacy(obj)
        except Exception as e:  # noqa: BLE001
            print("skip", mname, solv, e)
            continue
        rows.append((mname, solv, g))
        print("%-9s %-22s %+.10f Ha  %+8.4f kcal/mol" % (mname, solv, g, g * 627.5094740631))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "smd_cds.csv"), "w", newline="\n") as fh:
    fh.write(
        "# PROVENANCE: SMD reference implementation (MNSOL interface shipped\n"
        "#   with PySCF %s) evaluated as an ORACLE only - values, not code.\n"
        % pyscf.__version__
    )
    fh.write("# Geometries in Angstrom, see tools/oracle/gen_smd_cds.py.\n")
    fh.write("# Columns: molecule,solvent,g_cds_hartree\n")
    for m, s, g in rows:
        fh.write("%s,%s,%.12f\n" % (m, s, g))
print("wrote", len(rows), "rows")
