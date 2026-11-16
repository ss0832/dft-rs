# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Diagnose how this PySCF build enables VV10.
import os
from pyscf import dft, gto
from gen_integrals import parse_g94, DATA_DIR
from gen_scf import MOLECULES

charge, mult, atoms = MOLECULES["h2o"]
basis = parse_g94(os.path.join(DATA_DIR, "def2-svp.g94"), {"O", "H"})
geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
mol = gto.M(atom=geom, basis=basis, unit="Bohr")

mf = dft.RKS(mol)
mf.xc = "b88,lyp"
mf.nlc = "VV10"
print("nlc attr:", repr(mf.nlc))
print("do_nlc:", mf.do_nlc() if hasattr(mf, "do_nlc") else "n/a")
import inspect
src = inspect.getsource(type(mf).do_nlc) if hasattr(type(mf), "do_nlc") else ""
print(src[:400])
