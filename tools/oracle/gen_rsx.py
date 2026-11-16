# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Original RSX-QIDH/RSX-0DH independent point and full-Coulomb PT2 oracles.

DOI10.1021/acs.jctc.8b00261 and DOI10.1063/1.5097164. SR-PBE is ITYH-PBE,
not the HJS hole. Native point density thresholds are explicitly converged
at the fully polarized endpoint; no pre-existing fixture is corrected.
"""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, mp
from pyscf.dft import libxc

import gen_double_hybrids as dh
from gen_functional_points import POINTS
from gen_integrals import OUT_DIR

MODELS = {"rsx-qidh": (3.0 ** (-1.0 / 3.0), 0.27, 1.0 / 3.0),
          "rsx-0dh": (0.5, 0.33, 0.125)}


def point_value(code, omega, point, threshold=1e-20):
    api = ctypes.CDLL(libxc._itrf._name)
    pointer = ctypes.POINTER(ctypes.c_double)
    api.xc_func_alloc.restype = ctypes.c_void_p
    api.xc_func_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    api.xc_func_end.argtypes = [ctypes.c_void_p]
    api.xc_func_free.argtypes = [ctypes.c_void_p]
    api.xc_func_set_dens_threshold.argtypes = [ctypes.c_void_p, ctypes.c_double]
    api.xc_func_set_ext_params_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    api.xc_gga_exc_vxc_fxc.argtypes = [ctypes.c_void_p, ctypes.c_size_t] + [pointer] * 8
    handle = api.xc_func_alloc()
    assert api.xc_func_init(handle, libxc.XC_CODES[code], 2) == 0
    try:
        api.xc_func_set_dens_threshold(handle, threshold)
        if omega:
            api.xc_func_set_ext_params_name(handle, b"_omega", omega)
        rho = (ctypes.c_double * 2)(*point[:2])
        sigma = (ctypes.c_double * 3)(*point[2:5])
        output = [(ctypes.c_double * n)() for n in [1, 2, 3, 3, 6, 6]]
        api.xc_gga_exc_vxc_fxc(handle, 1, rho, sigma, *output)
        result = {"energy": sum(point[:2]) * output[0][0],
                  "potential": list(output[1]) + list(output[2]) + [0.0, 0.0]}
        if point[0] * point[1] > 0:
            matrix = np.zeros((7, 7))
            pairs = [[(0, 0), (0, 1), (1, 1)],
                     [(i, j) for i in range(2) for j in range(2, 5)],
                     [(i, j) for i in range(2, 5) for j in range(i, 5)]]
            for values, indices in zip(output[3:], pairs):
                for value, (i, j) in zip(values, indices):
                    matrix[i, j] = matrix[j, i] = value
            result["kernel"] = matrix.tolist()
        return result
    finally:
        api.xc_func_end(handle)
        api.xc_func_free(handle)


def run_case(model, numbers, positions, charge, spin):
    ax, omega, pt2 = MODELS[model]
    basis = dh.parse_g94(str(Path(dh.DATA_DIR) / "sto-3g.g94"), {dh.SYMBOLS[z] for z in numbers})
    mol = gto.M(atom=[(dh.SYMBOLS[z], p) for z, p in zip(numbers, positions)],
                basis=basis, unit="Bohr", charge=charge, spin=spin, cart=False,
                verbose=0, max_memory=256)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf._chkfile.close()
    mf.chkfile = None
    mf.xc = f"{ax}*HF + {1-ax}*LR_HF({omega}) + {1-ax}*GGA_X_ITYH_PBE,{1-pt2}*GGA_C_PBE"
    mf.omega = omega
    mf.grids.level = 9
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    if 26 in numbers:
        mf.verbose = 4
    mf.kernel()
    attempts = [{"method": "RKS/UKS", "converged": bool(mf.converged), "energy": float(mf.e_tot)}]
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
        attempts.append({"method": "Newton", "converged": bool(mf.converged), "energy": float(mf.e_tot)})
    assert mf.converged, (model, attempts)
    density = mf.make_rdm1()
    total = density.sum(axis=0) if spin else density
    j, k = mf.get_jk(mol, density)
    klr = mf.get_k(mol, density, omega=omega)
    factor = -0.5 if spin else -0.25
    exchange = factor * np.einsum("sij,sji" if spin else "ij,ji", density, ax*k + (1-ax)*klr)
    hartree = 0.5 * np.einsum("ij,ji", total, j.sum(axis=0) if spin else j)
    one = float(np.einsum("ij,ji", total, mf.get_hcore()))
    evaluator = mf._numint.nr_uks if spin else mf._numint.nr_rks
    parts = [weight * float(evaluator(mol, mf.grids, code, density, max_memory=256)[1])
             for weight, code in [(1-ax, f"LR_HF({omega}) + GGA_X_ITYH_PBE,"),
                                  (1-pt2, ",GGA_C_PBE")]]
    defect = mol.energy_nuc() + one + hartree + exchange + sum(parts) - mf.e_tot
    assert abs(defect) < 2e-10, defect
    perturbation = mp.UMP2(mf) if spin else mp.MP2(mf)
    perturbation.kernel()
    return {"model": model, "charge": charge, "spin": spin, "converged": bool(mf.converged),
            "parent": float(mf.e_tot), "target": float(mf.e_tot), "one": one,
            "hartree": float(hartree), "exchange": float(exchange), "parts": parts,
            "same_spin": float(perturbation.e_corr_ss), "opposite_spin": float(perturbation.e_corr_os),
            "dispersion": 0.0, "energy": float(mf.e_tot + pt2*perturbation.e_corr),
            "reconstruction_error": float(defect), "reference_scf_attempts": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--molecules", nargs="+")
    args = parser.parse_args()
    lib.num_threads(1)
    path = Path(OUT_DIR) / "rsx.json"
    data = json.loads(path.read_text()) if path.exists() else {"models": {}, "cases": [], "molecules": {}}
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    data.update(points=points, provenance=f"PySCF {dh.pyscf.__version__}, Libxc {libxc.__version__}",
                publications=["doi:10.1021/acs.jctc.8b00261", "doi:10.1063/1.5097164"],
                convention="ITYH SR-PBE + PBE-C, full LR HF, unscaled full-Coulomb SS/OS PT2; no dispersion",
                point_density_threshold=1e-20, point_threshold_check=1e-30,
                units="atomic units; local energy per volume; half-weighted tau")
    for model in args.models:
        ax, omega, pt2 = MODELS[model]
        components = {label: [point_value(code, w, p) for p in points]
                      for label, code, w in [("exchange", "GGA_X_ITYH_PBE", omega), ("correlation", "GGA_C_PBE", 0.0)]}
        values = []
        threshold_defect = 0.0
        for index, point in enumerate(points):
            for label, code, w in [("exchange", "GGA_X_ITYH_PBE", omega), ("correlation", "GGA_C_PBE", 0.0)]:
                other = point_value(code, w, point, 1e-30)
                threshold_defect = max(threshold_defect, abs(other["energy"]-components[label][index]["energy"]))
            row = {}
            for key in components["exchange"][index]:
                value = ((1-ax)*np.asarray(components["exchange"][index][key])
                         + (1-pt2)*np.asarray(components["correlation"][index][key]))
                row[key] = float(value) if value.ndim == 0 else value.tolist()
            values.append(row)
        assert threshold_defect < 1e-14
        data["models"][model] = {"ax": ax, "omega": omega, "pt2": pt2, "components": components,
                                  "values": values, "threshold_energy_defect": threshold_defect}
        path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        if args.points_only:
            continue
        for name, (numbers, positions, charge, spin) in dh.molecules().items():
            if args.molecules and name not in args.molecules:
                continue
            row = run_case(model, numbers, positions, charge, spin)
            row["molecule"] = name
            data["cases"] = [c for c in data["cases"] if (c["model"], c["molecule"]) != (model, name)]
            data["cases"].append(row)
            data["molecules"][name] = {"numbers": numbers, "positions": positions, "charge": charge, "spin": spin}
            path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
            print(model, name, row["energy"], "reconstruction", row["reconstruction_error"], flush=True)


if __name__ == "__main__":
    main()
