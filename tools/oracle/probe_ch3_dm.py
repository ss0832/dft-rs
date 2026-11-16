# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: converge PySCF CH3/B3LYP starting from the dft-rs
# converged density (read from a wavefunction checkpoint).
import os
import re
from pathlib import Path
import numpy as np
from pyscf import gto, dft

from gen_integrals import parse_g94, dftrs_ao_permutation, DATA_DIR
from gen_scf import MOLECULES

CKPT = os.environ.get(
    "DFT_RS_CH3_CHECKPOINT",
    str(Path(__file__).resolve().parents[2] / "target" / "ch3_b3lyp.wfn.txt"),
)


def read_matrix(lines, name, n):
    m = np.zeros((n, n))
    idx = lines.index(f"[{name}] rows={n} cols={n}")
    for line in lines[idx + 1 : idx + 1 + n * n]:
        i, j, v = line.split()
        m[int(i), int(j)] = float(v)
    return m


lines = open(CKPT).read().splitlines()
nao = int(re.search(r"nao_sph=(\d+)", "\n".join(lines)).group(1))
pa = read_matrix(lines, "density_alpha", nao)
pb = read_matrix(lines, "density_beta", nao)

charge, mult, atoms = MOLECULES["ch3_doublet"]
basis = parse_g94(os.path.join(DATA_DIR, "def2-svp.g94"), {"C", "H"})
geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=1, cart=False)
perm = dftrs_ao_permutation(mol)  # A_dftrs = A_pyscf[perm][:, perm]
inv = np.argsort(perm)
pa_p = pa[np.ix_(inv, inv)]
pb_p = pb[np.ix_(inv, inv)]
print("Tr(Pa S) =", np.trace(pa_p @ mol.intor("int1e_ovlp")))

mf = dft.UKS(mol)
mf.xc = "b3lypg"
mf.grids.level = 9
mf.conv_tol = 1e-12
# Single-shot energy of the dft-rs density (no SCF):
mf.grids.build()
e_dm = mf.energy_tot(dm=np.array([pa_p, pb_p]))
print("PySCF energy of dft-rs density:", e_dm)
e = mf.kernel(dm0=(pa_p, pb_p))
print("from dft-rs density:", e, "S2 =", mf.spin_square()[0], "conv:", mf.converged)
