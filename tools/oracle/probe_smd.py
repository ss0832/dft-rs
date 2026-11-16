# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Is a usable SMD-CDS oracle available in this environment?
from pyscf import gto
from pyscf.solvent import smd

print("libsolvent:", smd.libsolvent)
print("has oracle:", smd.libsolvent is not None)
try:

    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g")
    obj = smd.SMD(mol)
    obj.solvent = "water"
    g, _ = smd.get_cds_legacy(obj)
    print("water G_CDS (Hartree):", g)
except Exception as e:
    print("oracle unavailable:", type(e).__name__, e)
