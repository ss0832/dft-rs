# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Numerical WC04/WP04 oracles in the explicit VWN-RPA convention.

Wiitala, Hoye and Cramer, JCTC 2, 1085 (2006), doi:10.1021/ct6001016.
The paper does not identify the local VWN fit. The named libxc 7.0 entries
611/612 explicitly select VWN-RPA; no new fitted variant is inferred here.
The oracle is optional, used only to freeze numerical fixtures.
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

PARTS = [("lda_exchange", "LDA_X,"), ("b88_exchange", "GGA_X_B88,"),
         ("vwn_rpa", ",LDA_C_VWN_RPA"), ("lyp", ",GGA_C_LYP")]
MODELS = {
    "wc04": ("HYB_GGA_XC_WC04", 0.7400, [-0.9998, 0.9999, 0.9998, 0.0001]),
    "wp04": ("HYB_GGA_XC_WP04", 0.1189, [0.0385, 0.9614, 0.9998, 0.0001]),
}


def point_value(code, point, second=False):
    inputs = spin_inputs(point, False)
    lda = "GGA" not in code
    if lda:
        inputs = tuple(spin[0] for spin in inputs)
    output = libxc.eval_xc(code, inputs, spin=1, deriv=2 if second else 1)
    potential = [*output[1][0][0].tolist(), *([0.0] * 3 if lda else output[1][1][0].tolist())]
    result = {"energy": float(output[0][0] * (point[0] + point[1])), "potential": potential}
    if second and point[0] * point[1] > 0:
        kernel = np.zeros((5, 5))
        for value, (i, j) in zip(output[2][0][0], [(0, 0), (0, 1), (1, 1)]):
            kernel[i, j] = kernel[j, i] = value
        for value, (i, j) in zip(output[2][1][0], [(i, j) for i in range(2) for j in range(2, 5)]):
            kernel[i, j] = kernel[j, i] = value
        for value, (i, j) in zip(output[2][2][0], [(i, j) for i in range(2, 5) for j in range(i, 5)]):
            kernel[i, j] = kernel[j, i] = value
        result["kernel"] = kernel.tolist()
    return result


def main():
    lib.num_threads(2)
    result = {"provenance": f"PySCF {pyscf.__version__}, libxc {libxc.__version__}",
              "publication": "doi:10.1021/ct6001016", "convention": "named libxc 7.0 WC04/WP04; VWN-RPA",
              "variables": "rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b",
              "units": "atomic units; energy Hartree, geometry Bohr, density energy Hartree/Bohr^3",
              "density_ao_order": "canonical spherical m=0,+1,-1,...; p=z,x,y",
              "points": POINTS, "components": {}, "functionals": {}}
    for label, code in PARTS:
        result["components"][label] = [point_value(code, p) for p in POINTS]
    for name, (code, exact_fraction, weights) in MODELS.items():
        assert libxc.hybrid_coeff(code) == exact_fraction
        model = {"oracle_identifier": code, "exact_exchange": exact_fraction,
                 "weights": dict(zip([p[0] for p in PARTS], weights)),
                 "values": [point_value(code, p, True) for p in POINTS], "scf": []}
        for index, point in enumerate(POINTS):
            combined = model["values"][index]
            energy = sum(w * result["components"][label][index]["energy"] for (label, _), w in zip(PARTS, weights))
            assert abs(energy - combined["energy"]) < 1e-12
            for k in range(5):
                if point[1] == 0.0 and k in (1, 3, 4):
                    continue
                potential = sum(w * result["components"][label][index]["potential"][k] for (label, _), w in zip(PARTS, weights))
                assert abs(potential - combined["potential"][k]) < 1e-11
        for molecule in ["h2o", "ch3_doublet"]:
            charge, multiplicity, atoms = MOLECULES[molecule]
            basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
            mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge, spin=multiplicity - 1,
                        cart=False, verbose=0, max_memory=256)
            mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
            mf._chkfile.close()
            mf.chkfile = None
            mf.max_memory = 256
            mf.xc = code
            mf.grids.level = 9
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-10
            mf.max_cycle = 300
            mf.kernel()
            assert mf.converged, (name, molecule)
            density = mf.make_rdm1()
            spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
            total = spins.sum(axis=0)
            core = float(np.einsum("ij,ji", mf.get_hcore(), total))
            hartree = float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))
            exact = float(-0.5 * np.einsum("sij,sji", mf.get_k(dm=spins), spins))
            terms = {label: float(mf._numint.nr_uks(mol, mf.grids, part, spins, max_memory=256)[1]) for label, part in PARTS}
            reconstruction = mol.energy_nuc() + core + hartree + exact_fraction * exact + sum(w * terms[label] for (label, _), w in zip(PARTS, weights))
            assert abs(reconstruction - mf.e_tot) < 2e-11
            permutation = dftrs_ao_permutation(mol)
            model["scf"].append({"molecule": molecule, "basis": "sto-3g", "charge": charge,
                                 "multiplicity": multiplicity, "numbers": [Z[a[0]] for a in atoms],
                                 "positions": [list(a[1:]) for a in atoms],
                                 "density": spins[:, permutation][:, :, permutation].tolist(),
                                 "total": float(mf.e_tot), "nuclear": float(mol.energy_nuc()),
                                 "core": core, "coulomb": hartree, "hf_exchange": exact, **terms,
                                 "reconstruction_error": reconstruction - mf.e_tot})
            path = Path(OUT_DIR) / f"{molecule}_sto3g_{name}_dft.csv"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"# PROVENANCE: {result['provenance']}; {code}; VWN-RPA\n")
                handle.write("# spherical STO-3G, grid9, zero temperature; no frozen core/extra corrections\n")
                handle.write(f"basis,sto-3g\nfunctional,{name}\ncharge,{charge}\nmult,{multiplicity}\n")
                for symbol, x, y, z in atoms:
                    handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
                handle.write(f"e_tot,{mf.e_tot:.16e}\n")
            print(name, molecule, f"{mf.e_tot:.14f}", "reconstruction", reconstruction - mf.e_tot, flush=True)
        result["functionals"][name] = model
    (Path(OUT_DIR) / "nmr_hybrids.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
