# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent original omegaB2PLYP/omegaB2GP-PLYP ground-state terms.

Casanova-Paez, Dardis and Goerigk, JCTC 15, 4735 (2019),
DOI10.1021/acs.jctc.9b00013; author follow-up arXiv:2006.07821.
Full-Coulomb PT2 is separate from the range-separated orbital parent.
"""
import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, mp
from pyscf.dft import libxc

import gen_double_hybrids as dh
from gen_functional_points import POINTS, spin_inputs
from gen_integrals import OUT_DIR
from gga_boundary_reference import correct_wb2_boundary

RECIPES = {"wb2plyp": (0.53, 0.30, 0.73, 0.27),
           "wb2gp-plyp": (0.65, 0.27, 0.64, 0.36)}


def point_value(code, point, omega):
    out = libxc.eval_xc(code, spin_inputs(point, False), spin=1, deriv=2, omega=omega)
    result = {"energy": float(out[0][0] * (point[0] + point[1])),
              "potential": [*out[1][0][0].tolist(), *out[1][1][0].tolist(), 0.0, 0.0]}
    if point[0] * point[1] > 0:
        kernel = np.zeros((7, 7))
        pairs = [[(0, 0), (0, 1), (1, 1)],
                 [(i, j) for i in range(2) for j in range(2, 5)],
                 [(i, j) for i in range(2, 5) for j in range(i, 5)]]
        for slot, group in enumerate(pairs):
            for value, (i, j) in zip(out[2][slot][0], group):
                kernel[i, j] = kernel[j, i] = value
        result["kernel"] = kernel.tolist()
    correct_wb2_boundary(code, point, omega, result)
    return result


def run_case(model, numbers, positions, charge, spin, newton_first=False):
    ax, omega, c, pt2 = RECIPES[model]
    # Numerical integration evaluates only the local piece, but its parser
    # needs explicit range metadata to pass omega to the screened component.
    sr_code = f"LR_HF({omega}) + GGA_X_ITYH,"
    elements = {dh.SYMBOLS[z] for z in numbers}
    basis = dh.parse_g94(str(Path(dh.DATA_DIR) / "sto-3g.g94"), elements)
    mol = gto.M(atom=[(dh.SYMBOLS[z], p) for z, p in zip(numbers, positions)],
                basis=basis, unit="Bohr", charge=charge, spin=spin, cart=False,
                verbose=0, max_memory=256)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf._chkfile.close()
    mf.chkfile = None
    mf.xc = f"{ax}*HF + {1-ax}*LR_HF({omega}) + {1-ax}*GGA_X_ITYH, {c}*GGA_C_LYP"
    mf.omega = omega
    mf.grids.level = 9
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    if 26 in numbers:
        mf.verbose = 4
    if newton_first:
        mf = mf.newton()
    mf.kernel()
    attempts = [{"solver": "Newton" if newton_first else "RKS/UKS",
                 "converged": bool(mf.converged), "energy": float(mf.e_tot)}]
    if not mf.converged and not newton_first:
        mf = mf.newton()
        mf.kernel()
        attempts.append({"solver": "Newton", "converged": bool(mf.converged), "energy": float(mf.e_tot)})
    assert mf.converged, model
    density = mf.make_rdm1()
    total = density.sum(axis=0) if spin else density
    j, k = mf.get_jk(mol, density)
    klr = mf.get_k(mol, density, omega=omega)
    factor = -0.5 if spin else -0.25
    exchange = factor * np.einsum("sij,sji" if spin else "ij,ji", density, ax*k + (1-ax)*klr)
    hartree = 0.5 * np.einsum("ij,ji", total, j.sum(axis=0) if spin else j)
    one = float(np.einsum("ij,ji", total, mf.get_hcore()))
    evaluator = mf._numint.nr_uks if spin else mf._numint.nr_rks
    parts = [w * float(evaluator(mol, mf.grids, code, density, max_memory=256)[1])
             for w, code in [(1-ax, sr_code), (c, ",GGA_C_LYP")]]
    defect = mol.energy_nuc() + one + hartree + exchange + sum(parts) - mf.e_tot
    print(model, "term reconstruction", mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=spin),
          "E", mf.e_tot, "one", one, "J", hartree, "K", exchange, "parts", parts,
          "defect", defect, flush=True)
    assert abs(defect) < 2e-10, defect
    pt = mp.UMP2(mf) if spin else mp.MP2(mf)
    pt.kernel()
    return {"model": model, "charge": charge, "spin": spin,
            "parent": float(mf.e_tot), "target": float(mf.e_tot), "one": one,
            "hartree": float(hartree), "exchange": float(exchange), "parts": parts,
            "same_spin": float(pt.e_corr_ss), "opposite_spin": float(pt.e_corr_os),
            "dispersion": 0.0, "energy": float(mf.e_tot + pt2*pt.e_corr),
            "reconstruction_error": float(defect), "converged": bool(mf.converged),
            "reference_scf_attempts": attempts}


def main():
    lib.num_threads(2)
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    data = {"provenance": f"PySCF {dh.pyscf.__version__}, libxc {libxc.__version__}; independent full/LR exchange and full-Coulomb MP2",
            "publication": "doi:10.1021/acs.jctc.9b00013; arXiv:2006.07821",
            "units": "atomic units; local energy per volume; sigma spin invariants",
            "points": points, "recipes": {}, "molecules": {}, "cases": []}
    path = Path(OUT_DIR) / "wb2plyp.json"
    molecules = dh.molecules()
    assert {"h2", "lih", "water", "phosphine", "ferrocene"} <= set(molecules)
    for model, (ax, omega, c, pt2) in RECIPES.items():
        data["recipes"][model] = {"exact_exchange": ax, "long_range_exchange": 1-ax,
            "omega": omega, "same_spin": pt2, "opposite_spin": pt2,
            "components": {label: [point_value(code, p, omega) for p in points]
                for label, code in [("exchange", "GGA_X_ITYH,"), ("correlation", ",GGA_C_LYP"),
                                    ("parent", f"{1-ax}*GGA_X_ITYH,{c}*GGA_C_LYP")]}}
        for name, (numbers, positions, charge, spin) in molecules.items():
            row = run_case(model, numbers, positions, charge, spin)
            row["molecule"] = name
            data["molecules"][name] = {"numbers": numbers, "positions": positions, "charge": charge, "spin": spin}
            data["cases"].append(row)
            path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding="utf-8")
            print(model, name, row["energy"], row["reconstruction_error"], flush=True)


if __name__ == "__main__":
    main()
