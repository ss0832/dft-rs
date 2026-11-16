# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent numerical PKZB exchange/correlation and dense local kernels.

Perdew, Kurth, Zupan and Blaha, PRL 82, 2544 (1999), with erratum.
Optional PySCF/libxc is used only when freezing numerical fixtures.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.dft import libxc

from gen_functional_points import POINTS, spin_inputs
from gen_integrals import DATA_DIR, OUT_DIR, dftrs_ao_permutation, parse_g94
from gen_scf import MOLECULES, Z


def point_value(code, point):
    output = libxc.eval_xc(code, spin_inputs(point, True), spin=1, deriv=2)
    potentials = [*output[1][0][0], *output[1][1][0], *output[1][3][0]]
    result = {"energy": float(output[0][0] * (point[0] + point[1])),
              "potential": [float(v) for v in potentials]}
    # Empty spin channels have singular mixed density partials. Only active
    # spin-channel first partials are compared at these boundary points.
    if point[0] * point[1] > 0:
        kernel = np.zeros((7, 7))
        groups = [
            (0, [(0, 0), (0, 1), (1, 1)]),
            (1, [(i, j) for i in range(2) for j in range(2, 5)]),
            (2, [(i, j) for i in range(2, 5) for j in range(i, 5)]),
            (4, [(5, 5), (5, 6), (6, 6)]),
            (6, [(i, j) for i in range(2) for j in range(5, 7)]),
            (9, [(i, j) for i in range(2, 5) for j in range(5, 7)]),
        ]
        for slot, pairs in groups:
            for value, (i, j) in zip(output[2][slot][0], pairs):
                kernel[i, j] = kernel[j, i] = value
        result["kernel"] = kernel.tolist()
    return result


def main():
    lib.num_threads(2)
    points = [*POINTS,
              [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    result = {"provenance": f"PySCF {pyscf.__version__}, libxc {libxc.__version__}",
              "publication": "doi:10.1103/PhysRevLett.82.2544; erratum82,5179",
              "variables": "rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b",
              "units": "atomic units; energy per volume Hartree/Bohr^3; tau includes one-half",
              "convention": "original PKZB, Libxc identifiers 213/239; reference FHC input convention",
              "points": points,
              "components": {}}
    for name, code in [("exchange", "MGGA_X_PKZB,"), ("correlation", ",MGGA_C_PKZB")]:
        result["components"][name] = [point_value(code, p) for p in points]
    result["scf"] = []
    for name in ["h2o", "ch3_doublet"]:
        charge, multiplicity, atoms = MOLECULES[name]
        basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                    spin=multiplicity - 1, cart=False, verbose=0, max_memory=256)
        mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
        mf._chkfile.close()
        mf.chkfile = None
        mf.xc = "MGGA_X_PKZB,MGGA_C_PKZB"
        mf.max_memory = 256
        mf.grids.level = 9
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, name
        density = mf.make_rdm1()
        spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
        total = spins.sum(axis=0)
        core = float(np.einsum("ij,ji", mf.get_hcore(), total))
        hartree = float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))
        exchange = float(mf._numint.nr_uks(mol, mf.grids, "MGGA_X_PKZB,", spins, max_memory=256)[1])
        correlation = float(mf._numint.nr_uks(mol, mf.grids, ",MGGA_C_PKZB", spins, max_memory=256)[1])
        permutation = dftrs_ao_permutation(mol)
        defect = mol.energy_nuc() + core + hartree + exchange + correlation - mf.e_tot
        assert abs(defect) < 2e-11
        result["scf"].append({"molecule": name, "numbers": [Z[a[0]] for a in atoms],
                              "positions": [list(a[1:]) for a in atoms], "charge": charge,
                              "multiplicity": multiplicity, "total": float(mf.e_tot),
                              "core": core, "coulomb": hartree, "exchange": exchange,
                              "correlation": correlation, "reconstruction_error": float(defect),
                              "density": spins[:, permutation][:, :, permutation].tolist()})
        for split in [False, True]:
            suffix = "_split" if split else ""
            csv = Path(OUT_DIR) / f"{name}_sto3g_pkzb{suffix}_dft.csv"
            with csv.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"# PROVENANCE: {result['provenance']}; original PKZB213/239\n")
                handle.write("# spherical STO-3G, grid9; no frozen core or extra correction\n")
                handle.write("basis,sto-3g\n")
                handle.write("x,pkzb\nc,pkzb\n" if split else "functional,pkzb\n")
                handle.write(f"charge,{charge}\nmult,{multiplicity}\n")
                for symbol, x, y, z in atoms:
                    handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
                handle.write(f"e_tot,{mf.e_tot:.16e}\n")
        print(name, f"{mf.e_tot:.14f}", "term reconstruction", defect, flush=True)
    path = Path(OUT_DIR) / "pkzb.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path.name, len(points), "points per component")


if __name__ == "__main__":
    main()
