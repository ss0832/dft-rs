# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent TASK/mTASK exchange and explicit PW92 recipes.

PRResearch 1, 033082 (2019); PRMaterials 5, 063803 (2021).
The optional numerical reference is used only to freeze values.
"""

import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib

from gen_functional_points import POINTS
from gen_integrals import DATA_DIR, OUT_DIR, dftrs_ao_permutation, parse_g94
from gen_pkzb import point_value
from gen_scf import MOLECULES, Z

MODELS = {"task": "MGGA_X_TASK", "mtask": "MGGA_X_MTASK"}


def main():
    lib.num_threads(2)
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    result = {"provenance": f"PySCF {pyscf.__version__}, libxc {dft.libxc.__version__}",
              "publications": ["doi:10.1103/PhysRevResearch.1.033082", "doi:10.1103/PhysRevMaterials.5.063803"],
              "identifiers": {"task": 707, "mtask": 724},
              "convention": "published rounded TASK/mTASK exchange parameters; explicit LDA_C_PW original PW92 correlation",
              "variables": "rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b",
              "units": "atomic units; energy per volume Hartree/Bohr^3; half-weighted tau",
              "points": points, "components": {}, "combined": {}, "scf": []}
    for model, exchange_code in MODELS.items():
        result["components"][model] = [point_value(exchange_code + ",", p) for p in points]
        result["combined"][model] = [point_value(exchange_code + ",LDA_C_PW", p) for p in points]
        for name in ["h2o", "ch3_doublet"]:
            charge, multiplicity, atoms = MOLECULES[name]
            basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
            mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                        spin=multiplicity - 1, cart=False, verbose=0, max_memory=256)
            mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
            mf._chkfile.close()
            mf.chkfile = None
            mf.xc = exchange_code + ",LDA_C_PW"
            mf.grids.level = 9
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-10
            mf.max_cycle = 200
            mf.kernel()
            assert mf.converged, (model, name)
            density = mf.make_rdm1()
            spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
            total = spins.sum(axis=0)
            core = float(np.einsum("ij,ji", mf.get_hcore(), total))
            hartree = float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))
            exchange = float(mf._numint.nr_uks(mol, mf.grids, exchange_code + ",", spins, max_memory=256)[1])
            correlation = float(mf._numint.nr_uks(mol, mf.grids, ",LDA_C_PW", spins, max_memory=256)[1])
            defect = mol.energy_nuc() + core + hartree + exchange + correlation - mf.e_tot
            assert abs(defect) < 2e-11
            permutation = dftrs_ao_permutation(mol)
            result["scf"].append({"functional": model, "molecule": name,
                "numbers": [Z[a[0]] for a in atoms], "positions": [list(a[1:]) for a in atoms],
                "charge": charge, "multiplicity": multiplicity, "total": float(mf.e_tot),
                "core": core, "coulomb": hartree, "exchange": exchange, "correlation": correlation,
                "reconstruction_error": float(defect), "density": spins[:, permutation][:, :, permutation].tolist()})
            for split in [False, True]:
                suffix = "_split" if split else ""
                path = Path(OUT_DIR) / f"{name}_sto3g_{model}{suffix}_dft.csv"
                with path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(f"# PROVENANCE: {result['provenance']}; {model} exchange + original PW92\n")
                    handle.write("basis,sto-3g\n")
                    handle.write(f"x,{model}\nc,pw92\n" if split else f"functional,{model}\n")
                    handle.write(f"charge,{charge}\nmult,{multiplicity}\n")
                    for symbol, x, y, z in atoms:
                        handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
                    handle.write(f"e_tot,{mf.e_tot:.16e}\n")
            print(model, name, f"{mf.e_tot:.14f}", "term reconstruction", defect, flush=True)
    path = Path(OUT_DIR) / "task.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
