# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Emit the bundled solvent-descriptor table (src/data/solvents.csv).
#
# The values are published physical constants of the Minnesota Solvent
# Descriptor Database (Winget, Dolney, Giesen, Cramer & Truhlar), the same
# table the SMD paper (J. Phys. Chem. B 113, 6378 (2009)) builds on. They
# are routed here through PySCF's Apache-2.0 distribution of that database,
# mirroring how the project already sources its D3 parameters via MOPAC.
import os

from pyscf.solvent import smd

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "data")
os.makedirs(OUT, exist_ok=True)

db = smd.solvent_db
rows = []
for name in sorted(db.keys()):
    key = name.strip()
    if not key:
        continue
    n, _n2, alpha, beta, gamma, eps, phi2, psi2 = db[name]
    if eps <= 0.0:
        continue
    rows.append((key.lower(), eps, n, alpha, beta, gamma, phi2, psi2))

path = os.path.join(OUT, "solvents.csv")
with open(path, "w", newline="\n", encoding="utf-8") as fh:
    fh.write("# PROVENANCE: Minnesota Solvent Descriptor Database\n")
    fh.write("#   (Winget, Dolney, Giesen, Cramer, Truhlar), the descriptor set\n")
    fh.write("#   used by SMD (Marenich, Cramer, Truhlar, J. Phys. Chem. B 113,\n")
    fh.write("#   6378 (2009)). Measured/derived physical constants, routed via\n")
    fh.write("#   PySCF's Apache-2.0 distribution of the same table (the project\n")
    fh.write("#   sources its D3 parameters the same way, via MOPAC).\n")
    fh.write("# Retrieved: 2026-08-04.\n")
    fh.write("# Columns: name,eps,refractive_index,abraham_alpha,abraham_beta,\n")
    fh.write("#          macroscopic_surface_tension_cal_per_mol_A2,phi2,psi2\n")
    fh.write("# gamma/phi2/psi2 = -1 marks 'water' (its own SMD parameter set).\n")
    for r in rows:
        fh.write("%s,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n" % r)
print("wrote", path, len(rows), "solvents")
