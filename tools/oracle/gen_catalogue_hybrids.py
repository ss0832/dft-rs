# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Freeze numerical BLYP35 oracles using independently evaluated components.

BLYP35 is 0.35 HF exchange + 0.65 B88 exchange + full LYP correlation.
Renz et al., JACS 131, 16292 (2009); doi:10.1021/ja9070859.
The oracle dependencies are used only to produce numerical fixtures.
Run from any directory with the existing oracle Python environment.
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


def main():
    lib.num_threads(2)
    output = Path(OUT_DIR)
    code = "0.35*HF + 0.65*GGA_X_B88, GGA_C_LYP"
    result = {
        "provenance": f"PySCF {pyscf.__version__}, libxc {libxc.__version__}",
        "definition": code,
        "units": "Hartree; energy density Hartree/Bohr^3; geometry Bohr",
        "density_ao_order": "canonical spherical m=0,+1,-1,...; p=z,x,y",
        "points": POINTS,
        "components": {},
        "scf": [],
    }
    for label, component in [("exchange", "GGA_X_B88,"), ("correlation", ",GGA_C_LYP")]:
        rows = []
        for point in POINTS:
            energy, potential = libxc.eval_xc(
                component, spin_inputs(point, False), spin=1, deriv=1
            )[:2]
            rows.append({
                "energy": float(energy[0] * (point[0] + point[1])),
                "potential": [*potential[0][0].tolist(), *potential[1][0].tolist()],
            })
        result["components"][label] = rows
    for molecule in ["h2o", "ch3_doublet"]:
        charge, multiplicity, atoms = MOLECULES[molecule]
        basis = parse_g94(str(Path(DATA_DIR) / "sto-3g.g94"), {a[0] for a in atoms})
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", charge=charge,
                    spin=multiplicity - 1, cart=False, verbose=0)
        mf = dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)
        mf._chkfile.close()
        mf.xc = code
        mf.chkfile = None
        mf.grids.level = 9
        mf.conv_tol = 1e-13
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, molecule
        density = mf.make_rdm1()
        spins = np.array([density * 0.5, density * 0.5]) if density.ndim == 2 else density
        total = spins.sum(axis=0)
        core = float(np.einsum("ij,ji", mf.get_hcore(), total))
        coulomb = float(0.5 * np.einsum("ij,ji", mf.get_j(dm=total), total))
        exchange = float(-0.5 * np.einsum("sij,sji", mf.get_k(dm=spins), spins))
        terms = {}
        for label, component in [("exchange", "GGA_X_B88,"), ("correlation", ",GGA_C_LYP")]:
            terms[label] = float(mf._numint.nr_uks(mol, mf.grids, component, spins)[1])
        reconstructed = mol.energy_nuc() + core + coulomb + 0.35 * exchange + 0.65 * terms["exchange"] + terms["correlation"]
        assert abs(reconstructed - mf.e_tot) < 2e-11
        permutation = dftrs_ao_permutation(mol)
        result["scf"].append({"molecule": molecule, "basis": "sto-3g",
                              "numbers": [Z[a[0]] for a in atoms],
                              "positions": [list(a[1:]) for a in atoms],
                              "charge": charge, "multiplicity": multiplicity,
                              "density": spins[:, permutation][:, :, permutation].tolist(),
                              "total": float(mf.e_tot), "nuclear": float(mol.energy_nuc()),
                              "core": core, "coulomb": coulomb, "hf_exchange": exchange,
                              **terms, "reconstruction_error": reconstructed - mf.e_tot})
        path = output / f"{molecule}_sto3g_blyp35_dft.csv"
        with path.open("w", newline="\n") as handle:
            handle.write(f"# PROVENANCE: {result['provenance']}; {code}\n")
            handle.write("# spherical STO-3G; grid level 9; no frozen core, NLC or dispersion\n")
            handle.write(f"basis,sto-3g\nfunctional,blyp35\ncharge,{charge}\nmult,{multiplicity}\n")
            for symbol, x, y, z in atoms:
                handle.write(f"atom,{Z[symbol]},{x},{y},{z}\n")
            handle.write(f"e_tot,{mf.e_tot:.16e}\n")
        print(f"{molecule}: {mf.e_tot:.14f}; reconstruction {reconstructed - mf.e_tot:.3e}", flush=True)
    (output / "catalogue_hybrids.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
