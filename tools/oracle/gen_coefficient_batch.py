# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent B98, kinetic-HCTH and BMK recipe fixtures, using existing components."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyscf
from pyscf import dft, gto, lib
from pyscf.dft import libxc

from gen_edf2 import point_value as gga_value
from gen_functional_points import POINTS
from gen_integrals import DATA_DIR, dftrs_ao_permutation, parse_g94
from gen_pkzb import point_value as mgga_value
from gen_scf import MOLECULES, Z

MODELS = {
    "b98": {"code": "HYB_GGA_XC_SB98_2C", "hf": .219847, "family": "gga",
            "parts": [("integrated", "HYB_GGA_XC_SB98_2C", "gga")],
            "definition": "Schmider-Becke1998 fit2c; published B98 mapping; fitted terms unscaled; original PW92"},
    "thcth": {"code": "MGGA_X_TAU_HCTH,GGA_C_TAU_HCTH", "hf": 0., "family": "mgga",
              "parts": [("exchange", "MGGA_X_TAU_HCTH,", "mgga"),
                        ("correlation", ",GGA_C_TAU_HCTH", "gga")],
              "definition": "Boese-Handy2002 kinetic exchange and original-PW92 fitted correlation; no HF"},
    "thcthhyb": {"code": "HYB_MGGA_X_TAU_HCTH,GGA_C_HYB_TAU_HCTH", "hf": .15, "family": "mgga",
                 "parts": [("exchange", "HYB_MGGA_X_TAU_HCTH,", "mgga"),
                           ("correlation", ",GGA_C_HYB_TAU_HCTH", "gga")],
                 "definition": "Boese-Handy2002 refitted hybrid coefficients, fitted local X included whole plus0.15HF"},
    "bmk": {"code": "279,280", "hf": .42, "family": "mgga",
            "parts": [("exchange", "279,", "mgga"), ("correlation", ",280", "gga")],
            "definition": "Boese-Martin2004 published TableII kinetic1=4.73936; fitted local exchange unscaled plus0.42HF; original PW92"},
}


def point(code, p, family):
    return (mgga_value if family == "mgga" else gga_value)(code, p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecules", action="store_true")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS))
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.models and args.output_directory is None:
        parser.error("Selected models require a separate output directory to preserve existing fixtures")
    lib.num_threads(1)
    points = [*POINTS, [.31, .22, .041, -.017, .029, .37, .21],
              [.31, .22, 0., 0., 0., .37, .21]]
    result = {"provenance": f"PySCF {pyscf.__version__}; numerical library {libxc.__version__}",
              "convention": "energy per volume; rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b; half-weighted tau; no dispersion/PT2",
              "publications": ["doi:10.1063/1.476438", "doi:10.1063/1.1476309"],
              "points": points, "models": {}}
    output = args.output_directory or Path(__file__).resolve().parents[2] / "tests/oracle_data"
    output.mkdir(parents=True, exist_ok=True)
    selected = args.models or list(MODELS)
    if "bmk" in selected:
        result["publications"].append("doi:10.1063/1.1774975")
    for name in selected:
        model = MODELS[name]
        assert abs(libxc.hybrid_coeff(model["code"])-model["hf"]) < 1e-14
        entry = dict(hf=model["hf"], definition=model["definition"], family=model["family"],
                     values=[point(model["code"], p, model["family"]) for p in points],
                     components={label: [point(code, p, family) for p in points]
                                 for label, code, family in model["parts"]}, scf=[])
        if args.molecules:
            for label in ["h2o", "ch3_doublet"]:
                charge, multiplicity, atoms = MOLECULES[label]
                basis = parse_g94(str(Path(DATA_DIR)/"sto-3g.g94"), {a[0] for a in atoms})
                mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                            spin=multiplicity-1, cart=False, verbose=0, max_memory=256)
                mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
                mf._chkfile.close()
                mf.chkfile = None
                mf.xc = model["code"]
                mf.grids.level = 9
                mf.conv_tol = 1e-13
                mf.conv_tol_grad = 1e-10
                mf.max_cycle = 300
                mf.kernel()
                assert mf.converged, (name, label)
                dm = mf.make_rdm1()
                spins = np.array([dm*.5, dm*.5]) if dm.ndim == 2 else dm
                total = spins.sum(axis=0)
                terms = dict(one_electron=float(np.einsum("ij,ji", mf.get_hcore(), total)),
                             hartree=float(.5*np.einsum("ij,ji", mf.get_j(dm=total), total)),
                             full_exchange=float(-.5*np.einsum("sij,sji", mf.get_k(dm=spins), spins)),
                             nuclear_repulsion=float(mol.energy_nuc()))
                for part, code, _ in model["parts"]:
                    terms[part] = float(mf._numint.nr_uks(mol, mf.grids, code, spins, max_memory=256)[1])
                reconstructed = sum(v for k, v in terms.items() if k != "full_exchange")+model["hf"]*terms["full_exchange"]
                assert abs(reconstructed-mf.e_tot) < 1e-9
                order = dftrs_ao_permutation(mol, basis)
                case = dict(molecule=label, numbers=[Z[a[0]] for a in atoms], positions=[list(a[1:]) for a in atoms],
                            charge=charge, multiplicity=multiplicity, total=float(mf.e_tot), converged=True,
                            terms=terms, density=[s[np.ix_(order, order)].tolist() for s in spins])
                entry["scf"].append(case)
                lines = ["# PROVENANCE: "+result["provenance"], "# spherical STO-3G; grid9; no dispersion/PT2",
                         "basis,sto-3g", f"functional,{name}", f"charge,{charge}", f"mult,{multiplicity}"]
                lines += [f"atom,{z},"+",".join(map(str, r)) for z, r in zip(case["numbers"], case["positions"])]
                lines.append(f"e_tot,{case['total']:.16e}")
                (output/f"{label}_sto3g_{name}_dft.csv").write_text("\n".join(lines)+"\n", encoding="utf-8")
        result["models"][name] = entry
        (output/"coefficient_batch.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(name, "reference complete", flush=True)


if __name__ == "__main__":
    main()
