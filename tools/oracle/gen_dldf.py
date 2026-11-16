# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent dlDF local kernels and optional RKS/UKS term references.

Pernal et al., PRL 103, 263201 (2009), canonical rounded XC parameters.
This is dlDF, without the separately fitted intermolecular dispersion model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.dft import libxc

from gen_functional_points import POINTS
from gen_integrals import DATA_DIR, dftrs_ao_permutation, parse_g94
from gen_pkzb import point_value
from gen_scf import MOLECULES, Z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", action="store_true")
    args = parser.parse_args()
    lib.num_threads(1)
    points = [*POINTS, [.31, .22, .041, -.017, .029, .37, .21],
              [.31, .22, 0., 0., 0., .37, .21]]
    result = {
        "publication": "doi:10.1103/PhysRevLett.103.263201",
        "provenance": f"PySCF {pyscf.__version__}; numerical library {libxc.__version__}",
        "definition": "canonical rounded IDs36/37; HF0.6144129; local exchange already scaled; PW92-MOD correlation with corrected Fermi factor eta1e-10; no dispersion",
        "convention": "energy per volume; rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b; tau includes one-half",
        "points": points,
        "components": {label: [point_value(code, p) for p in points]
                       for label, code in [("exchange", "HYB_MGGA_X_DLDF,"),
                                           ("correlation", ",MGGA_C_DLDF")]},
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
            mf.xc = "HYB_MGGA_X_DLDF,MGGA_C_DLDF"
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
                "local_exchange": float(mf._numint.nr_uks(mol, mf.grids, "HYB_MGGA_X_DLDF,", spins, max_memory=256)[1]),
                "correlation": float(mf._numint.nr_uks(mol, mf.grids, ",MGGA_C_DLDF", spins, max_memory=256)[1]),
                "nuclear_repulsion": float(mol.energy_nuc()),
            }
            reconstructed = sum(v for k, v in terms.items() if k != "full_exchange") + .6144129*terms["full_exchange"]
            assert abs(reconstructed-mf.e_tot) < 1e-9
            order = dftrs_ao_permutation(mol, basis)
            result["scf"].append({"molecule": label, "numbers": [Z[a[0]] for a in atoms],
                                  "positions": [list(a[1:]) for a in atoms], "charge": charge,
                                  "multiplicity": multiplicity, "total": float(mf.e_tot),
                                  "converged": True, "terms": terms,
                                  "density": [s[np.ix_(order, order)].tolist() for s in spins]})
    output = Path(__file__).resolve().parents[2] / "tests/oracle_data"
    (output / "dldf.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for case in result.get("scf", []):
        lines = ["# PROVENANCE: "+result["provenance"], "# spherical STO-3G; grid9; dlDF HF0.6144129; no dispersion",
                 "basis,sto-3g", "functional,dldf", f"charge,{case['charge']}", f"mult,{case['multiplicity']}"]
        lines += [f"atom,{z},"+",".join(map(str, r)) for z, r in zip(case["numbers"], case["positions"])]
        lines.append(f"e_tot,{case['total']:.16e}")
        (output / f"{case['molecule']}_sto3g_dldf_dft.csv").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
