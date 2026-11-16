# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent N12-SX local component values and complete GGA kernels.

Peverati and Truhlar, PCCP 14, 16187 (2012), equations 1-9/Table 2.
The local fitted term has coefficient one; separate HF is 25% short range.
"""

import json
import argparse
from pathlib import Path

import pyscf
from pyscf import lib, dft, gto
import numpy as np
from pyscf.dft import libxc

from gen_edf2 import point_value
from gen_functional_points import POINTS
from gen_integrals import DATA_DIR, parse_g94, dftrs_ao_permutation
from gen_scf import MOLECULES, Z


def write_census_cases(result):
    output = Path(__file__).resolve().parents[2] / "tests/oracle_data"
    for case in result.get("scf", []):
        lines = ["# PROVENANCE: " + result["provenance"],
                 "# spherical STO-3G; grid9; N12-SX screened HF.25/0, omega.11",
                 "basis,sto-3g", "functional,n12-sx", f"charge,{case['charge']}",
                 f"mult,{case['multiplicity']}"]
        lines += [f"atom,{z}," + ",".join(map(str, r))
                  for z, r in zip(case["numbers"], case["positions"])]
        lines.append(f"e_tot,{case['total']:.16e}")
        (output / f"{case['molecule']}_sto3g_n12-sx_dft.csv").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", action="store_true")
    args = parser.parse_args()
    lib.num_threads(1)
    points = [*POINTS, [.31, .22, .041, -.017, .029, .37, .21],
              [.31, .22, 0., 0., 0., .37, .21]]
    result = {
        "publication": "doi:10.1039/C2CP42576A equations1-9/Table2",
        "provenance": f"PySCF {pyscf.__version__}; numerical library {libxc.__version__}",
        "convention": "energy per volume; canonical rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb; fitted local term unscaled; SR HF .25, LR HF0, omega .11",
        "points": points,
        "components": {label: [point_value(code, p) for p in points]
                       for label, code in [("exchange", "HYB_GGA_X_N12_SX,"),
                                           ("correlation", ",GGA_C_N12_SX")]},
    }
    if args.molecules:
        result["scf"] = []
        for label in ["h2o", "ch3_doublet"]:
            charge, multiplicity, atoms = MOLECULES[label]
            basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
            mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                        spin=multiplicity-1, cart=False, verbose=0, max_memory=256)
            mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
            mf._chkfile.close()
            mf.chkfile = None
            mf.xc = "HYB_GGA_X_N12_SX,GGA_C_N12_SX"
            mf.grids.level = 9
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-10
            mf.max_cycle = 300
            mf.kernel()
            assert mf.converged, label
            dm = mf.make_rdm1()
            spins = np.array([dm*.5, dm*.5]) if dm.ndim == 2 else dm
            total = spins.sum(axis=0)
            terms = {
                "one_electron": float(np.einsum("ij,ji", mf.get_hcore(), total)),
                "hartree": float(.5*np.einsum("ij,ji", mf.get_j(dm=total), total)),
                "full_exchange": float(-.5*np.einsum("sij,sji", mf.get_k(dm=spins), spins)),
                "long_range_exchange": float(-.5*np.einsum("sij,sji", mf.get_k(dm=spins, omega=.11), spins)),
                "local_exchange": float(mf._numint.nr_uks(mol, mf.grids, "HYB_GGA_X_N12_SX,", spins, max_memory=256)[1]),
                "correlation": float(mf._numint.nr_uks(mol, mf.grids, ",GGA_C_N12_SX", spins, max_memory=256)[1]),
                "nuclear_repulsion": float(mol.energy_nuc()),
            }
            reconstructed = sum(terms[k] for k in ["one_electron", "hartree", "local_exchange", "correlation", "nuclear_repulsion"])+.25*(terms["full_exchange"]-terms["long_range_exchange"])
            assert abs(reconstructed-mf.e_tot)<1e-9
            order = dftrs_ao_permutation(mol)
            result["scf"].append({"molecule": label, "numbers": [Z[a[0]] for a in atoms],
                                  "positions": [list(a[1:]) for a in atoms], "charge": charge,
                                  "multiplicity": multiplicity, "total": float(mf.e_tot),
                                  "converged": True, "terms": terms,
                                  "density": [s[np.ix_(order, order)].tolist() for s in spins]})
    Path(__file__).resolve().parents[2].joinpath("tests/oracle_data/n12_sx.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.molecules:
        write_census_cases(result)


if __name__ == "__main__":
    main()
