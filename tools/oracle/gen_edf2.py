# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent EDF2 coefficient, component, kernel and energy-term fixtures.

Lin, George and Gill, Aust. J. Chem. 57, 365 (2004), equation (2).
Published weights are retained without normalizing their sums.
The optional numerical evaluator is not a runtime dependency.
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
from gga_boundary_reference import correct_edf2_boundary

PARTS = [("lda_exchange", "LDA_X,"), ("b88_exchange", "GGA_X_B88,"),
         ("modified_b88", "REFERENCE_EDF2_B88"), ("vwn5", ",LDA_C_VWN"),
         ("lyp", ",GGA_C_LYP"), ("modified_lyp", "REFERENCE_EDF2_LYP")]
WEIGHTS = [0.2811, 0.6227, -0.0551, 0.3029, 0.5998, -0.0053]
CODE = "HYB_GGA_XC_EDF2"


def point_value(code, point):
    lda = libxc.xc_type(code) == "LDA"
    inputs = spin_inputs(point, False)
    if lda:
        inputs = tuple(spin[0] for spin in inputs)
    output = libxc.eval_xc(code, inputs, spin=1, deriv=2)
    potential = [*output[1][0][0].tolist(), *([0.0] * 3 if lda else output[1][1][0].tolist())]
    result = {"energy": float(output[0][0] * (point[0] + point[1])), "potential": potential}
    if point[0] * point[1] > 0:
        kernel = np.zeros((5, 5))
        groups = [(0, [(0, 0), (0, 1), (1, 1)])]
        if not lda:
            groups += [(1, [(i, j) for i in range(2) for j in range(2, 5)]),
                       (2, [(i, j) for i in range(2, 5) for j in range(i, 5)])]
        for slot, pairs in groups:
            for value, (i, j) in zip(output[2][slot][0], pairs):
                kernel[i, j] = kernel[j, i] = value
        result["kernel"] = kernel.tolist()
    correct_edf2_boundary(code, point, result)
    return result


def main():
    lib.num_threads(2)
    libxc.register_custom_functional_("REFERENCE_EDF2_B88", "GGA_X_B88", ext_params={106: [0.0035, 6.0]})
    libxc.register_custom_functional_("REFERENCE_EDF2_LYP", "GGA_C_LYP", ext_params={131: [0.055, 0.158, 0.25, 0.3505]})
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    result = {"provenance": f"PySCF {pyscf.__version__}, libxc {libxc.__version__}",
        "publication": "doi:10.1071/CH03263, equation2", "identifier": 476,
        "convention": "VWN5, full B88 components, independent modified B88/LYP parameters; no normalization",
        "variables": "rho_a,rho_b,sigma_aa,sigma_ab,sigma_bb,tau_a,tau_b",
        "units": "atomic units; total energies Hartree; energy density Hartree/Bohr^3; geometry Bohr",
        "density_ao_order": "canonical spherical m=0,+1,-1,...; p=z,x,y",
        "points": points, "components": {}, "functionals": {}}
    for label, code in PARTS:
        result["components"][label] = [point_value(code, p) for p in points]
    assert libxc.hybrid_coeff(CODE) == 0.1695
    model = {"oracle_identifier": CODE, "exact_exchange": 0.1695,
             "weights": dict(zip([p[0] for p in PARTS], WEIGHTS)),
             "values": [point_value(CODE, p) for p in points], "scf": []}
    for index, point in enumerate(points):
        target = model["values"][index]
        for key in ["energy", "potential", "kernel"]:
            if key not in target:
                continue
            reconstructed = sum(w * np.asarray(result["components"][label][index][key]) for (label, _), w in zip(PARTS, WEIGHTS))
            assert np.max(np.abs(reconstructed - np.asarray(target[key]))) < 1e-10, (index, key)
    for name in ["h2o", "ch3_doublet"]:
        charge, multiplicity, atoms = MOLECULES[name]
        basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge, spin=multiplicity - 1,
                    cart=False, verbose=0, max_memory=256)
        mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
        mf._chkfile.close()
        mf.chkfile = None
        mf.max_memory = 256
        mf.xc = CODE
        mf.grids.level = 9
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 300
        mf.kernel()
        assert mf.converged, name
        density = mf.make_rdm1()
        spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
        total = spins.sum(axis=0)
        core = float(np.einsum("ij,ji", mf.get_hcore(), total))
        hartree = float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))
        exact = float(-0.5 * np.einsum("sij,sji", mf.get_k(dm=spins), spins))
        terms = {label: float(mf._numint.nr_uks(mol, mf.grids, code, spins, max_memory=256)[1]) for label, code in PARTS}
        defect = mol.energy_nuc() + core + hartree + 0.1695 * exact + sum(w * terms[label] for (label, _), w in zip(PARTS, WEIGHTS)) - mf.e_tot
        assert abs(defect) < 2e-11
        permutation = dftrs_ao_permutation(mol)
        model["scf"].append({"molecule": name, "basis": "sto-3g", "charge": charge,
            "multiplicity": multiplicity, "numbers": [Z[a[0]] for a in atoms],
            "positions": [list(a[1:]) for a in atoms], "density": spins[:, permutation][:, :, permutation].tolist(),
            "total": float(mf.e_tot), "nuclear": float(mol.energy_nuc()), "core": core,
            "coulomb": hartree, "hf_exchange": exact, **terms, "reconstruction_error": float(defect)})
        path = Path(OUT_DIR) / f"{name}_sto3g_edf2_dft.csv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"# PROVENANCE: {result['provenance']}; EDF2 identifier476; VWN5\n")
            handle.write("# spherical STO-3G, grid9, integer populations, no extra corrections\n")
            handle.write(f"basis,sto-3g\nfunctional,edf2\ncharge,{charge}\nmult,{multiplicity}\n")
            for symbol, x, y, z in atoms:
                handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
            handle.write(f"e_tot,{mf.e_tot:.16e}\n")
        print(name, f"{mf.e_tot:.14f}", "term reconstruction", defect, flush=True)
    result["functionals"]["edf2"] = model
    (Path(OUT_DIR) / "edf2.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
