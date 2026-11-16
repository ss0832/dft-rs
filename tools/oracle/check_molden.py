# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Molden round-trip verification: load a dft-rs Molden file with PySCF and
# check MO orthonormality against the overlap of the reconstructed molecule
# (validates basis block, AO ordering, and coefficient conventions).
#
# Usage (WSL): python check_molden.py /path/to/file.molden

import sys
import numpy as np
from pyscf.tools import molden

def main(path):
    mol, mo_energy, mo_coeff, mo_occ, irrep, spins = molden.load(path)
    mol.build()
    s = mol.intor("int1e_ovlp")
    c = mo_coeff if mo_coeff.ndim == 2 else mo_coeff[0]
    ortho = c.T @ s @ c
    dev = np.abs(ortho - np.eye(ortho.shape[0])).max()
    nelec = (np.asarray(mo_occ) > 0).sum() if np.asarray(mo_occ).ndim == 1 else 0
    print(f"nao={mol.nao} nmo={c.shape[1]} max|C^T S C - I| = {dev:.3e}")
    print(f"occupied MOs: {nelec}, first MO energies: {np.round(mo_energy[:3], 6)}")
    assert dev < 1e-7, "MO orthonormality broken -> Molden conventions wrong"
    print("MOLDEN ROUND-TRIP OK")

if __name__ == "__main__":
    main(sys.argv[1])
