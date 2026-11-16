# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# SAPT electrostatics oracle: E_elst^(1) for the HF/STO-3G water dimer,
# assembled from PySCF integrals on an independent stack (monomer SCFs in
# monomer bases; dimer-basis cross integrals).
import numpy as np
import pyscf
from pyscf import gto, scf
import os

from gen_integrals import parse_g94, DATA_DIR

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "oracle_data")

A = [("O", 0.0, 0.0, 0.0), ("H", 0.0, 1.43, 1.108), ("H", 0.0, -1.43, 1.108)]
B = [("O", 0.0, 0.0, 5.4), ("H", 0.0, 1.43, 6.5), ("H", 0.0, -1.43, 6.5)]

basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), {"O", "H"})


def build(atoms):
    geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    return gto.M(atom=geom, basis=basis, unit="Bohr", cart=False)


mol_a = build(A)
mol_b = build(B)
mol_d = build(A + B)

mf_a = scf.RHF(mol_a)
mf_a.conv_tol = 1e-12
mf_a.kernel()
mf_b = scf.RHF(mol_b)
mf_b.conv_tol = 1e-12
mf_b.kernel()

nao_a = mol_a.nao
nao_d = mol_d.nao
p_a = np.zeros((nao_d, nao_d))
p_a[:nao_a, :nao_a] = mf_a.make_rdm1()
p_b = np.zeros((nao_d, nao_d))
p_b[nao_a:, nao_a:] = mf_b.make_rdm1()

# Nuclear attraction split by nucleus set.
charges = mol_d.atom_charges()
coords = mol_d.atom_coords()
v_a = np.zeros((nao_d, nao_d))
v_b = np.zeros((nao_d, nao_d))
for k in range(mol_d.natm):
    with mol_d.with_rinv_origin(coords[k]):
        vk = -charges[k] * mol_d.intor("int1e_rinv")
    if k < mol_a.natm:
        v_a += vk
    else:
        v_b += vk

j_b = scf.hf.get_jk(mol_d, p_b, hermi=1)[0]
e_nn = 0.0
for i in range(mol_a.natm):
    for j in range(mol_a.natm, mol_d.natm):
        e_nn += charges[i] * charges[j] / np.linalg.norm(coords[i] - coords[j])

e_elst = np.einsum("mn,mn->", p_a, v_b) + np.einsum("mn,mn->", p_b, v_a) + np.einsum(
    "mn,mn->", p_a, j_b
) + e_nn

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "sapt_water_dimer_elst.csv"), "w", newline="\n") as fh:
    fh.write("# PROVENANCE: PySCF %s SAPT electrostatics oracle (values only)\n" % pyscf.__version__)
    fh.write("# HF/STO-3G water dimer, Bohr literals as in tests/p12_sapt.rs\n")
    fh.write("e_monomer_a,%.12f\n" % mf_a.e_tot)
    fh.write("e_monomer_b,%.12f\n" % mf_b.e_tot)
    fh.write("e_elst,%.12f\n" % e_elst)
print("E_elst =", e_elst)
