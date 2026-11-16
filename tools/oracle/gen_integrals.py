# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 1 oracle generator: frozen PySCF one-electron matrices and ERI
# samples for the dft-rs integral engine.
#
# Reproducibility contract:
# - The basis definition is read from dft-rs's own bundled .g94 files and
#   fed to PySCF in its internal format, so shell order and contraction
#   data are identical by construction.
# - AO ordering is permuted from PySCF's convention (p shells: x,y,z;
#   l >= 2: m = -l..+l) into the dft-rs convention (m = 0,+1,-1,+2,-2,...
#   for every l; p = z,x,y in m terms).
# - Output CSVs carry a provenance header. Regeneration is manual and its
#   diffs are reviewed like code changes.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_integrals.py

import os
import numpy as np
from pyscf import gto
import pyscf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "src", "basis", "data")
OUT_DIR = os.path.join(ROOT, "tests", "oracle_data")

SHELL_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}


def parse_g94(path, wanted_elements):
    """Minimal Gaussian94 parser mirroring dft-rs basis/g94.rs (orbital
    blocks only; ECP sections are not needed for these molecules)."""
    basis = {}
    with open(path) as fh:
        lines = [l.strip() for l in fh]
    lines = [l for l in lines if l and not l.startswith("!")]
    pos = 0
    while pos < len(lines) and lines[pos].startswith("****"):
        pos += 1
    while pos < len(lines):
        header = lines[pos].split()
        sym = header[0].capitalize()
        pos += 1
        if pos < len(lines) and lines[pos].split()[0].upper().endswith("-ECP"):
            # BSE places the ECP section after all orbital blocks; the
            # oracle molecules use all-electron elements only, so stop here.
            break
        shells = []
        while pos < len(lines) and not lines[pos].startswith("****"):
            typ, nprim, _scale = lines[pos].split()
            nprim = int(nprim)
            pos += 1
            rows = []
            for _ in range(nprim):
                toks = [t.replace("D", "E").replace("d", "E") for t in lines[pos].split()]
                rows.append([float(t) for t in toks])
                pos += 1
            if typ.upper() == "SP":
                shells.append([0] + [[r[0], r[1]] for r in rows])
                shells.append([1] + [[r[0], r[2]] for r in rows])
            else:
                l = SHELL_L[typ.upper()]
                shells.append([l] + [[r[0], r[1]] for r in rows])
        if sym in wanted_elements:
            basis[sym] = shells
        while pos < len(lines) and lines[pos].startswith("****"):
            pos += 1
    return basis


def dftrs_ao_permutation(mol, source_basis=None):
    """Map spherical AOs to original input-shell and canonical component order.

    The reference engine sorts shells by angular momentum. The native parser
    preserves input order, including interleaved S/P shells. Match contraction
    columns by exponents and signed coefficient ratios before ordering m.
    An explicit basis dictionary is required: an alias loses source ordering.
    """
    basis = mol.basis if source_basis is None else source_basis
    if not isinstance(basis, dict) or mol.cart:
        raise ValueError("AO mapping requires an explicit spherical source basis")
    perm, used = [], set()
    offsets = mol.ao_loc_nr()
    for atom in range(mol.natm):
        shells = basis.get(mol.atom_symbol(atom), basis.get(mol.atom_pure_symbol(atom)))
        if shells is None:
            raise ValueError("Source basis does not identify an atom's shell order")
        for shell in shells:
            l = shell[0]
            if len(shell) < 2 or not isinstance(shell[1], (list, tuple, np.ndarray)):
                raise ValueError("Unsupported source shell format")
            rows = np.asarray(shell[1:], dtype=float)
            order_exp = np.argsort(-rows[:, 0], kind="stable")
            exponents = rows[order_exp, 0]
            for column in range(1, rows.shape[1]):
                coefficients = rows[order_exp, column]
                norm = np.linalg.norm(coefficients)
                if norm == 0:
                    raise ValueError("Zero source contraction")
                coefficients = coefficients / norm
                matches = []
                for bas_id in range(mol.nbas):
                    if mol.bas_atom(bas_id) != atom or mol.bas_angular(bas_id) != l:
                        continue
                    other_exp = mol.bas_exp(bas_id)
                    if len(other_exp) != len(exponents) or not np.allclose(
                        other_exp, exponents, rtol=1e-13, atol=0
                    ):
                        continue
                    for ctr, other in enumerate(mol.bas_ctr_coeff(bas_id).T):
                        if (bas_id, ctr) not in used and np.allclose(
                            other / np.linalg.norm(other), coefficients,
                            rtol=1e-12, atol=1e-14,
                        ):
                            matches.append((bas_id, ctr))
                if not matches:
                    raise ValueError("Source contraction does not match reference basis")
                bas_id, ctr = matches[0]
                used.add((bas_id, ctr))
                off = offsets[bas_id] + ctr * (2 * l + 1)
                perm.extend(off + k for k in _component_order(l))
    if len(used) != sum(mol.bas_nctr(i) for i in range(mol.nbas)) or sorted(perm) != list(range(mol.nao)):
        raise ValueError("AO mapping is not a complete bijection")
    return np.asarray(perm, dtype=int)


def _component_order(l):
    if l == 0:
        return [0]
    if l == 1:
        return [2, 0, 1]
    ms = [0]
    for m in range(1, l + 1):
        ms += [m, -m]
    return [m + l for m in ms]


def write_matrix(path, name, mat, meta):
    with open(path, "w", newline="\n") as fh:
        fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} oracle, {name}\n")
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        fh.write("# i,j,value\n")
        n = mat.shape[0]
        for i in range(n):
            for j in range(n):
                fh.write(f"{i},{j},{mat[i, j]:.16e}\n")


def write_eri_samples(path, name, eri_full, nao, meta, n_samples=250):
    rng = np.random.default_rng(20260803)
    with open(path, "w", newline="\n") as fh:
        fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} oracle, {name}\n")
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        fh.write("# i,j,k,l,value\n")
        seen = set()
        while len(seen) < n_samples:
            i, j, k, l = (int(x) for x in rng.integers(0, nao, 4))
            if (i, j, k, l) in seen:
                continue
            seen.add((i, j, k, l))
            fh.write(f"{i},{j},{k},{l},{eri_full[i, j, k, l]:.16e}\n")


# Geometries are exact BOHR literals shared verbatim with the Rust test
# (tests/p1_oracle.rs) so no Å↔Bohr conversion constant enters the
# comparison (PySCF pins CODATA-2010 Bohr, dft-rs CODATA-2018).
CASES = [
    (
        "h2o_sto3g",
        "sto-3g.g94",
        ["O", "H"],
        """O 0.0 0.0 0.2217
           H 0.0 1.4309 -0.8867
           H 0.0 -1.4309 -0.8867""",
    ),
    (
        "h2o_def2svp",
        "def2-svp.g94",
        ["O", "H"],
        """O 0.0 0.0 0.2217
           H 0.0 1.4309 -0.8867
           H 0.0 -1.4309 -0.8867""",
    ),
    (
        "ch4_sto3g",
        "sto-3g.g94",
        ["C", "H"],
        """C 0.0 0.0 0.0
           H 1.1860 1.1860 1.1860
           H 1.1860 -1.1860 -1.1860
           H -1.1860 1.1860 -1.1860
           H -1.1860 -1.1860 1.1860""",
    ),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, basis_file, elements, geom in CASES:
        basis = parse_g94(os.path.join(DATA_DIR, basis_file), set(elements))
        mol = gto.M(atom=geom, basis=basis, unit="Bohr", cart=False)
        perm = dftrs_ao_permutation(mol)
        meta = {
            "molecule": " / ".join(geom.split()),
            "basis_file": basis_file,
            "unit": "Bohr geometry literals, matrices in Hartree/Bohr a.u.",
            "ao_order": "dft-rs canonical (m = 0,+1,-1,...; p = z,x,y)",
        }
        s = mol.intor("int1e_ovlp")[np.ix_(perm, perm)]
        t = mol.intor("int1e_kin")[np.ix_(perm, perm)]
        v = mol.intor("int1e_nuc")[np.ix_(perm, perm)]
        write_matrix(os.path.join(OUT_DIR, f"{name}_s.csv"), name, s, meta)
        write_matrix(os.path.join(OUT_DIR, f"{name}_t.csv"), name, t, meta)
        write_matrix(os.path.join(OUT_DIR, f"{name}_v.csv"), name, v, meta)
        # Dipole about the origin (PySCF int1e_r has common origin (0,0,0)).
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            r = mol.intor("int1e_r")  # (3, nao, nao)
        for axis, label in enumerate("xyz"):
            write_matrix(
                os.path.join(OUT_DIR, f"{name}_dip{label}.csv"),
                name,
                r[axis][np.ix_(perm, perm)],
                meta,
            )
        nao = mol.nao
        eri = mol.intor("int2e").reshape(nao, nao, nao, nao)
        eri = eri[np.ix_(perm, perm, perm, perm)]
        write_eri_samples(
            os.path.join(OUT_DIR, f"{name}_eri.csv"), name, eri, nao, meta
        )
        print(f"{name}: nao={nao} written")
    # Environment freeze record.
    with open(os.path.join(OUT_DIR, "env.md"), "w", newline="\n") as fh:
        fh.write("# Oracle environment\n\n")
        fh.write(f"- PySCF {pyscf.__version__}, NumPy {np.__version__}\n")
        fh.write("- Generated by tools/oracle/gen_integrals.py (WSL Ubuntu)\n")
        fh.write("- Regeneration is manual; diffs are reviewed like code.\n")


if __name__ == "__main__":
    main()
