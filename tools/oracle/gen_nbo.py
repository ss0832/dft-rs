# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Oracle generator for natural atomic orbitals and natural population
# analysis: frozen PySCF values (numerical oracle only) from its NAO routine
# (component-averaged atomic natural orbitals, symmetric orthogonalization
# of the core, occupancy-weighted orthogonalization of the valence set,
# symmetric orthogonalization of the Rydberg set, restored natural
# character) for Hartree-Fock references on basis files of the package.
# The molecules contain H, B-Ne and Al-Ar only, whose natural minimal basis
# (core and occupied valence subshells) is the same in both programs.
#
# Run (WSL): <environment with pyscf>/bin/python gen_nbo.py [output dir]
import json
import os
import sys

import numpy
from pyscf import gto, scf
from pyscf.lo import nao

from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

# Bohr.
WATER = [("O", 0.0, 0.0, 0.23), ("H", 0.0, 1.4309, -0.8867), ("H", 0.12, -1.4809, -0.9067)]
AMMONIA = [("N", 0.0, 0.0, 0.14), ("H", 0.0, 1.77, -0.52), ("H", 1.53, -0.88, -0.52),
           ("H", -1.53, -0.88, -0.48)]
FORMALDEHYDE = [("C", 0.0, 0.0, -1.0), ("O", 0.0, 0.0, 1.28), ("H", 0.0, 1.78, -2.1),
                ("H", 0.05, -1.77, -2.1)]
ETHYLENE = [("C", 0.0, 0.0, 1.26), ("C", 0.0, 0.0, -1.26), ("H", 0.0, 1.75, 2.33),
            ("H", 0.0, -1.75, 2.33), ("H", 0.0, 1.75, -2.33), ("H", 0.0, -1.75, -2.33)]
PHOSPHINE = [("P", 0.0, 0.0, 0.2), ("H", 0.0, 2.25, -0.9), ("H", 1.95, -1.12, -0.9),
             ("H", -1.95, -1.12, -0.9)]
FILES = {"sto-3g": "sto-3g.g94", "6-31g*": "6-31g_d.g94"}
CASES = [
    ("water", WATER, "sto-3g", 0, 1),
    ("water", WATER, "6-31g*", 0, 1),
    ("ammonia", AMMONIA, "6-31g*", 0, 1),
    ("formaldehyde", FORMALDEHYDE, "6-31g*", 0, 1),
    ("ethylene", ETHYLENE, "sto-3g", 0, 1),
    ("phosphine", PHOSPHINE, "6-31g*", 0, 1),
    ("water", WATER, "6-31g*", 1, 2),
]


def molecule(atoms, basis, charge, multiplicity):
    data = parse_g94(os.path.join(DATA_DIR, FILES[basis]), {a[0] for a in atoms})
    return gto.M(atom="; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms),
                 basis=data, unit="Bohr", cart=False, charge=charge,
                 spin=multiplicity - 1, verbose=0)


def case(name, atoms, basis, charge, multiplicity):
    mol = molecule(atoms, basis, charge, multiplicity)
    mf = scf.RHF(mol) if multiplicity == 1 else scf.UHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    assert mf.converged
    dm = mf.make_rdm1()
    spin = numpy.asarray([dm / 2, dm / 2]) if multiplicity == 1 else numpy.asarray(dm)
    s = mol.intor_symmetric("int1e_ovlp")
    c = nao.nao(mol, mf, s)
    metric = c.T @ s @ c
    assert numpy.abs(metric - numpy.eye(len(metric))).max() < 1e-9
    d = [c.T @ s @ p @ s @ c for p in spin]
    occupations = numpy.diag(d[0] + d[1])
    spin_density = numpy.diag(d[0] - d[1])
    owner = [mol.bas_atom(b) for b in range(mol.nbas)
             for _ in range(mol.bas_nctr(b) * (2 * mol.bas_angular(b) + 1))]
    natoms = mol.natm
    population = numpy.zeros(natoms)
    spin_population = numpy.zeros(natoms)
    for mu, atom in enumerate(owner):
        population[atom] += occupations[mu]
        spin_population[atom] += spin_density[mu]
    wiberg = numpy.zeros((natoms, natoms))
    for mu, a in enumerate(owner):
        for nu, b in enumerate(owner):
            if a != b:
                wiberg[a, b] += 2 * (d[0][mu, nu] ** 2 + d[1][mu, nu] ** 2)
    per_atom = [sorted((float(occupations[mu]) for mu, a in enumerate(owner) if a == atom),
                       reverse=True) for atom in range(natoms)]
    return {
        "molecule": name,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "numbers": [int(z) for z in mol.atom_charges()],
        "positions": [[x, y, z] for _, x, y, z in atoms],
        "energy": float(mf.e_tot),
        "natural_charges": [float(z - p) for z, p in zip(mol.atom_charges(), population)],
        "natural_spin_populations": [float(v) for v in spin_population],
        "atom_occupancies": per_atom,
        "wiberg_bond_indices": wiberg.tolist(),
    }


def main():
    scf.hf.MUTE_CHKFILE = True
    target = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    out = {
        "source": "PySCF lo.nao natural atomic orbitals of RHF/UHF references (numerical oracle)",
        "cases": [case(*c) for c in CASES],
    }
    with open(os.path.join(target, "nbo_oracle.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
