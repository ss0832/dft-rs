# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""DSD-PBEB95-D3(BJ): independent local, spin-PT2 and dispersion values.

Kozuch and Martin, JCC 34, 2327 (2013), DOI10.1002/jcc.23391;
published BJ recipe tabulated by Goerigk and Grimme, WIREs Comput.
Mol. Sci. 4, 576 (2014), equation9 and table4.
The dispersion-free parameter set is a different fit.
"""
import json
from pathlib import Path

import numpy as np
from pyscf import dft, gto, lib, mp
from pyscf.dft import libxc

import gen_double_hybrids as dh
from gen_edf2 import point_value as gga_point
from gen_functional_points import POINTS
from gen_integrals import OUT_DIR
from gen_pkzb import point_value as mgga_point

RECIPE = {
    "parent": "0.66*HF + 0.34*GGA_X_PBE, 0.55*MGGA_C_BC95",
    "exact_exchange": 0.66,
    "parts": [(0.34, "GGA_X_PBE"), (0.55, "MGGA_C_BC95")],
    "same_spin": 0.09,
    "opposite_spin": 0.46,
    "d3bj": {"s6": 0.61, "s8": 0.0, "a1": 0.0, "a2": 6.2},
}


def run_case(numbers, positions, charge, spin, newton_first=False):
    elements = {dh.SYMBOLS[z] for z in numbers}
    basis = dh.parse_g94(str(Path(dh.DATA_DIR) / "sto-3g.g94"), elements)
    mol = gto.M(atom=[(dh.SYMBOLS[z], position) for z, position in zip(numbers, positions)],
                basis=basis, unit="Bohr", charge=charge, spin=spin, cart=False,
                verbose=0, max_memory=256)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf._chkfile.close()
    mf.chkfile = None
    mf.xc = RECIPE["parent"]
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
    assert mf.converged
    density = mf.make_rdm1()
    total = density.sum(axis=0) if spin else density
    j, k = mf.get_jk(mol, density)
    if spin:
        exchange = -0.5 * np.einsum("sij,sji", density, k)
        hartree = 0.5 * np.einsum("ij,ji", total, j.sum(axis=0))
        pt = mp.UMP2(mf)
    else:
        exchange = -0.25 * np.einsum("ij,ji", density, k)
        hartree = 0.5 * np.einsum("ij,ji", total, j)
        pt = mp.MP2(mf)
    pt.kernel()
    one = float(np.einsum("ij,ji", total, mf.get_hcore()))
    evaluator = mf._numint.nr_uks if spin else mf._numint.nr_rks
    parts = [weight * float(evaluator(mol, mf.grids, code, density, max_memory=256)[1])
             for weight, code in RECIPE["parts"]]
    exact = RECIPE["exact_exchange"] * float(exchange)
    dispersion = dh.d3bj(mol, RECIPE["d3bj"])
    reconstruction = mol.energy_nuc() + one + float(hartree) + exact + sum(parts) - mf.e_tot
    assert abs(reconstruction) < 2e-10
    energy = (mf.e_tot + RECIPE["same_spin"] * pt.e_corr_ss
              + RECIPE["opposite_spin"] * pt.e_corr_os + dispersion)
    return {"model": "dsd-pbeb95", "charge": charge, "spin": spin,
            "parent": float(mf.e_tot), "target": float(mf.e_tot),
            "one": one, "hartree": float(hartree), "exchange": exact, "parts": parts,
            "same_spin": float(pt.e_corr_ss), "opposite_spin": float(pt.e_corr_os),
            "dispersion": dispersion, "energy": float(energy), "converged": bool(mf.converged),
            "reconstruction_error": float(reconstruction), "reference_scf_attempts": attempts}


def point_value(code, point):
    if libxc.xc_type(code) == "MGGA":
        return mgga_point(code, point)
    result = gga_point(code, point)
    result["potential"] += [0.0, 0.0]
    if "kernel" in result:
        result["kernel"] = [row + [0.0, 0.0] for row in result["kernel"]]
        result["kernel"] += [[0.0] * 7, [0.0] * 7]
    return result


def main():
    lib.num_threads(2)
    points = [*POINTS, [0.31, 0.22, 0.041, -0.017, 0.029, 0.37, 0.21],
              [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]]
    data = {
        "provenance": f"PySCF {dh.pyscf.__version__}, libxc {libxc.__version__}; separate PySCF MP2 spin terms and D3(BJ)",
        "definition": "published DSD-PBEB95-D3(BJ), not dispersion-free or PBE-hole exchange fit",
        "publications": ["doi:10.1002/jcc.23391", "doi:10.1002/wcms.1193 equation9 table4"],
        "units": "atomic units; energies Hartree, local energy per volume; half-weighted tau",
        "recipe": RECIPE, "points": points, "components": {}, "molecules": {}, "cases": [],
    }
    for label, code in [("exchange", "GGA_X_PBE,"), ("correlation", ",MGGA_C_BC95"),
                        ("parent", "0.34*GGA_X_PBE,0.55*MGGA_C_BC95")]:
        data["components"][label] = [point_value(code, point) for point in points]
    molecules = {key: value for key, value in dh.molecules().items() if key != "ferrocene"}
    path = Path(OUT_DIR) / "dsd_pbeb95.json"
    for name, (numbers, positions, charge, spin) in molecules.items():
        row = run_case(numbers, positions, charge, spin)
        assert row["converged"], name
        row["molecule"] = name
        data["molecules"][name] = {"numbers": numbers, "positions": positions, "charge": charge, "spin": spin}
        data["cases"].append(row)
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(name, row["energy"], "SS", row["same_spin"], "OS", row["opposite_spin"], "D3", row["dispersion"], flush=True)


if __name__ == "__main__":
    main()
