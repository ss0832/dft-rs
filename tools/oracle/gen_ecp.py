# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 4.5 oracle: frozen PySCF HF/DFT energies with LANL2DZ ECPs.
# Both the orbital basis AND the ECP are parsed from dft-rs's bundled
# lanl2dz.g94 and fed to PySCF in its internal format, so every number is
# shared by construction (values-only reproducibility contract).
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_ecp.py

import os
import pyscf
from pyscf import gto, dft, scf

from gen_integrals import DATA_DIR, OUT_DIR, SHELL_L

Z = {"H": 1, "Na": 11, "Cl": 17, "Cu": 29}


def parse_g94_with_ecp(path, wanted):
    """Full Gaussian94 parser: orbital blocks + interleaved ECP sections."""
    basis, ecps = {}, {}
    with open(path) as fh:
        lines = [l.strip() for l in fh]
    lines = [l for l in lines if l and not l.startswith("!")]
    pos = 0
    while pos < len(lines) and lines[pos].startswith("****"):
        pos += 1
    while pos < len(lines):
        header = lines[pos].split()
        if header[0].upper().endswith("-ECP"):
            # ECP section: "<EL>-ECP  <lmax>  <ncore>".
            sym = header[0][:-4].capitalize()
            lmax = int(header[1])
            ncore = int(header[2])
            pos += 1
            channels = []
            for _ in range(lmax + 1):
                pos += 1  # title line ("d potential", "s-d potential", ...)
                nterms = int(lines[pos])
                pos += 1
                terms = []
                for _ in range(nterms):
                    toks = lines[pos].replace("D", "E").replace("d", "E").split()
                    terms.append((int(toks[0]), float(toks[1]), float(toks[2])))
                    pos += 1
                channels.append(terms)
            if sym in wanted:
                # PySCF internal format: [ncore, [[l, chans], ...]] with
                # chans[power_n] = [[exp, coef], ...] and the local channel
                # tagged l = -1. The G94 block lists the local channel first.
                def to_chans(terms):
                    maxn = max(t[0] for t in terms)
                    chans = [[] for _ in range(maxn + 1)]
                    for n, e, c in terms:
                        chans[n].append([e, c])
                    return chans

                entries = [[-1, to_chans(channels[0])]]
                for l, terms in enumerate(channels[1:]):
                    entries.append([l, to_chans(terms)])
                ecps[sym] = [ncore, entries]
            continue
        sym = header[0].capitalize()
        pos += 1
        shells = []
        while pos < len(lines) and not lines[pos].startswith("****"):
            if lines[pos].split()[0].upper().endswith("-ECP"):
                break  # this element's ECP section follows its shells
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
                shells.append([SHELL_L[typ.upper()]] + [[r[0], r[1]] for r in rows])
        # The ECP part of the file repeats the bare element header before
        # the "<EL>-ECP" line - do not clobber parsed shells with it.
        if sym in wanted and shells:
            basis[sym] = shells
        while pos < len(lines) and lines[pos].startswith("****"):
            pos += 1
    return basis, ecps


# (tag, atoms [(sym, x, y, z) Bohr], charge, mult, functional, pyscf xc)
CASES = [
    ("nah", [("Na", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 3.566)], 0, 1, "hf", None),
    ("cl2", [("Cl", 0.0, 0.0, 0.0), ("Cl", 0.0, 0.0, 3.756)], 0, 1, "hf", None),
    ("nah", [("Na", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 3.566)], 0, 1, "b3lyp", "b3lypg"),
    ("cl2", [("Cl", 0.0, 0.0, 0.0), ("Cl", 0.0, 0.0, 3.756)], 0, 1, "pbe", "pbe,pbe"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = {"H", "Na", "Cl"}
    basis, ecps = parse_g94_with_ecp(os.path.join(DATA_DIR, "lanl2dz.g94"), wanted)
    assert "Na" in ecps and "Cl" in ecps and "H" not in ecps
    for tag, atoms, charge, mult, func, xc in CASES:
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom,
            basis={s: basis[s] for s in {a[0] for a in atoms}},
            ecp={s: ecps[s] for s in {a[0] for a in atoms} if s in ecps},
            unit="Bohr",
            charge=charge,
            spin=mult - 1,
            cart=False,
        )
        if xc is None:
            mf = scf.RHF(mol) if mult == 1 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
            mf.xc = xc
            mf.grids.level = 9
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, f"{tag}/{func} not converged"
        case = f"{tag}_lanl2dz_{func}"
        with open(os.path.join(OUT_DIR, f"{case}_ecp.csv"), "w", newline="\n") as fh:
            fh.write(
                f"# PROVENANCE: PySCF {pyscf.__version__} ECP oracle "
                "(basis+ECP from dft-rs lanl2dz.g94, values only)\n"
            )
            fh.write("# geometry unit: Bohr (exact literals)\n")
            fh.write("basis,lanl2dz\n")
            fh.write(f"functional,{func}\n")
            fh.write(f"charge,{charge}\n")
            fh.write(f"mult,{mult}\n")
            for sym, x, y, z in atoms:
                fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
            fh.write(f"e_tot,{mf.e_tot:.14e}\n")
        print(f"{case}: E = {mf.e_tot:.10f}  (nelec = {mol.nelectron})")


if __name__ == "__main__":
    main()
