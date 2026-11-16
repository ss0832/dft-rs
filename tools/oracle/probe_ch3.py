# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: CH3/B3LYP UKS solution landscape.
import os
from pyscf import gto, dft

from gen_integrals import parse_g94, DATA_DIR
from gen_scf import MOLECULES

charge, mult, atoms = MOLECULES["ch3_doublet"]
basis = parse_g94(os.path.join(DATA_DIR, "def2-svp.g94"), {"C", "H"})
geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=1, cart=False)

mf = dft.UKS(mol)
mf.xc = "b3lypg"
mf.grids.level = 9
mf.conv_tol = 1e-12
mf.kernel()
print("default guess:", mf.e_tot, "S2 =", mf.spin_square()[0], "conv", mf.converged)

# Stability analysis: follow internal instabilities if any.
try:
    mo1, _, stable, _ = mf.stability(return_status=True)
    if not stable:
        dm1 = mf.make_rdm1(mo1, mf.mo_occ)
        mf2 = dft.UKS(mol)
        mf2.xc = "b3lypg"
        mf2.grids.level = 9
        mf2.conv_tol = 1e-12
        mf2.kernel(dm0=dm1)
        print("after stability:", mf2.e_tot, "S2 =", mf2.spin_square()[0])
    else:
        print("solution is internally stable")
except Exception as e:
    print("stability check failed:", e)
