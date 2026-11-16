# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Can PySCF run wb97m_v WITHOUT its automatic VV10 term? (nlc=0 probe)
import os
from pyscf import gto, dft

from gen_integrals import parse_g94, DATA_DIR
from gen_scf import MOLECULES

charge, mult, atoms = MOLECULES["h2o"]
basis = parse_g94(os.path.join(DATA_DIR, "def2-svp.g94"), {"O", "H"})
geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=0)

mf = dft.RKS(mol)
mf.xc = "wb97m_v"
mf.nlc = 0
mf.grids.level = 9
mf.conv_tol = 1e-12
e = mf.kernel()
print("converged:", mf.converged, "E_semilocal:", format(e, ".12f"))
