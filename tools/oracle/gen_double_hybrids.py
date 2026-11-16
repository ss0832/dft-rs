# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Term-by-term energy oracle of the named double hybrids: PySCF RKS/UKS
# parents (libxc evaluates the semilocal parts; used strictly as a
# numerical oracle), separate unscaled same-spin and opposite-spin MP2
# correlation energies on those orbitals, the exact-exchange, Hartree and
# one-electron energies, every weighted semilocal part, the D3(BJ) energy
# where the recipe has one, and, for xDH recipes, the target functional on
# the parent orbitals. H2, LiH, water, phosphine and ferrocene in STO-3G
# (geometries of tests/oracle_data/multicomponent_hf.json) and the LiH
# cation doublet, grid level 9.
#
# Run (WSL): <venv>/bin/python gen_double_hybrids.py [model ...]
# Existing entries of other models are kept.

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pyscf
from pyscf import dft, gto, lib, mp

from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

lib.num_threads(2)

SYMBOLS = {1: "H", 3: "Li", 6: "C", 8: "O", 15: "P", 26: "Fe"}
QIDH = 3.0 ** (-1.0 / 3.0)
PBE02 = 2.0 ** (-1.0 / 3.0)


def adiabatic(x, c, a_x, pt2):
    """Exchange/correlation pair with a_x exact exchange and pt2 PT2."""
    return {
        "parent": f"{a_x!r}*HF + {1 - a_x!r}*{x}, {1 - pt2!r}*{c}",
        "exact_exchange": a_x,
        "parts": [(1 - a_x, x), (1 - pt2, c)],
        "same_spin": pt2,
        "opposite_spin": pt2,
    }


def b2_type(x, a_x, c, pt2):
    return {
        "parent": f"{a_x!r}*HF + {1 - a_x!r}*{x}, {c!r}*GGA_C_LYP",
        "exact_exchange": a_x,
        "parts": [(1 - a_x, x), (c, "GGA_C_LYP")],
        "same_spin": pt2,
        "opposite_spin": pt2,
    }


MODELS = {
    "b2gp-plyp": b2_type("GGA_X_B88", 0.65, 0.64, 0.36),
    "b2k-plyp": b2_type("GGA_X_B88", 0.72, 0.58, 0.42),
    "b2t-plyp": b2_type("GGA_X_B88", 0.60, 0.69, 0.31),
    "mpw2plyp": b2_type("GGA_X_MPW91", 0.55, 0.75, 0.25),
    "mpw2k-plyp": b2_type("GGA_X_MPW91", 0.72, 0.58, 0.42),
    "pbe0-dh": adiabatic("GGA_X_PBE", "GGA_C_PBE", 0.5, 0.125),
    "pbe-qidh": adiabatic("GGA_X_PBE", "GGA_C_PBE", QIDH, 1.0 / 3.0),
    "pbe0-2": adiabatic("GGA_X_PBE", "GGA_C_PBE", PBE02, 0.5),
    "tpss-qidh": adiabatic("MGGA_X_TPSS", "MGGA_C_TPSS", QIDH, 1.0 / 3.0),
    "scan0-dh": adiabatic("MGGA_X_SCAN", "MGGA_C_SCAN", 0.5, 0.125),
    "scan-qidh": adiabatic("MGGA_X_SCAN", "MGGA_C_SCAN", QIDH, 1.0 / 3.0),
    "r2scan0-dh": adiabatic("MGGA_X_R2SCAN", "MGGA_C_R2SCAN", 0.5, 0.125),
    "r2scan-qidh": adiabatic("MGGA_X_R2SCAN", "MGGA_C_R2SCAN", QIDH, 1.0 / 3.0),
    "revdsd-pbep86": {
        "parent": "0.69*HF + 0.31*GGA_X_PBE, 0.4210*GGA_C_P86",
        "exact_exchange": 0.69,
        "parts": [(0.31, "GGA_X_PBE"), (0.4210, "GGA_C_P86")],
        "same_spin": 0.0636,
        "opposite_spin": 0.5922,
        "d3bj": {"s6": 0.4377, "s8": 0.0, "a1": 0.0, "a2": 5.5},
    },
    "xyg3": {
        "parent": "B3LYPG",
        "target": {
            "exact_exchange": 0.8033,
            "parts": [(-0.0140, "LDA_X"), (0.2107, "GGA_X_B88"), (0.6789, "GGA_C_LYP")],
        },
        "same_spin": 0.3211,
        "opposite_spin": 0.3211,
    },
}

# The open-shell case of tests/p132_double_hybrid_energy.rs.
EXTRA = {"lithium_hydride_cation": ([3, 1], [[0.0, 0.0, 0.0], [0.2, 0.1, 3.0]], 1, 1)}


def molecules():
    with open(os.path.join(OUT_DIR, "multicomponent_hf.json"), encoding="utf-8") as fh:
        cases = json.load(fh)
    out = {c["name"]: (c["numbers"], c["positions"], 0, 0) for c in cases}
    out.update(EXTRA)
    return out


def d3bj(mol, params):
    from dftd3.interface import DispersionModel, RationalDampingParam

    p = RationalDampingParam(s6=params["s6"], s8=params["s8"], a1=params["a1"],
                             a2=params["a2"], s9=0.0)
    model = DispersionModel(mol.atom_charges(), mol.atom_coords())
    return float(model.get_dispersion(p, grad=False)["energy"])


def run(model, recipe, numbers, positions, charge, spin):
    elements = sorted({SYMBOLS[z] for z in numbers})
    basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), set(elements))
    geom = "; ".join(f"{SYMBOLS[z]} {x} {y} {w}" for z, (x, y, w) in zip(numbers, positions))
    mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=spin,
                cart=False, verbose=0)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf.xc = recipe["parent"]
    mf.grids.level = 9
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    dm = mf.make_rdm1()
    hcore = mf.get_hcore()
    if spin:
        total = dm[0] + dm[1]
        j, k = mf.get_jk(mol, dm)
        exchange = -0.5 * (np.einsum("ij,ji", dm[0], k[0]) + np.einsum("ij,ji", dm[1], k[1]))
        hartree = 0.5 * np.einsum("ij,ji", total, j[0] + j[1])
        pt = mp.UMP2(mf)
    else:
        total = dm
        j, k = mf.get_jk(mol, dm)
        exchange = -0.25 * np.einsum("ij,ji", dm, k)
        hartree = 0.5 * np.einsum("ij,ji", dm, j)
        pt = mp.MP2(mf)
    pt.kernel()
    one = float(np.einsum("ij,ji", total, hcore))

    def part_energy(code):
        if spin:
            return float(mf._numint.nr_uks(mol, mf.grids, code, dm)[1])
        return float(mf._numint.nr_rks(mol, mf.grids, code, dm)[1])

    energy_terms = recipe.get("target", recipe)
    parts = [w * part_energy(code) for w, code in energy_terms["parts"]]
    exact = energy_terms["exact_exchange"] * float(exchange)
    dispersion = d3bj(mol, recipe["d3bj"]) if "d3bj" in recipe else 0.0
    if "target" in recipe:
        target = mol.energy_nuc() + one + float(hartree) + exact + sum(parts)
    else:
        target = mf.e_tot
    energy = (target + recipe["same_spin"] * pt.e_corr_ss
              + recipe["opposite_spin"] * pt.e_corr_os + dispersion)
    return {
        "model": model,
        "charge": charge,
        "spin": spin,
        "parent": float(mf.e_tot),
        "target": float(target),
        "one": one,
        "hartree": float(hartree),
        "exchange": exact,
        "parts": parts,
        "same_spin": float(pt.e_corr_ss),
        "opposite_spin": float(pt.e_corr_os),
        "dispersion": dispersion,
        "energy": float(energy),
        "converged": bool(mf.converged),
    }


def main():
    only = set(sys.argv[1:])
    path = os.path.join(OUT_DIR, "double_hybrid_molecules.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"cases": []}
    data["provenance"] = (f"PySCF {pyscf.__version__} RKS/UKS parents (libxc-backed, values "
                          "only), unscaled MP2 spin components, grid level 9, STO-3G")
    mols = molecules()
    data["molecules"] = {k: {"numbers": v[0], "positions": v[1], "charge": v[2], "spin": v[3]}
                         for k, v in mols.items()}
    kept = [c for c in data["cases"] if only and c["model"] not in only]
    for model, recipe in MODELS.items():
        if only and model not in only:
            continue
        for name, (numbers, positions, charge, spin) in mols.items():
            row = run(model, recipe, numbers, positions, charge, spin)
            row["molecule"] = name
            print(f"{model} {name}: E = {row['energy']:.10f} converged {row['converged']}",
                  flush=True)
            kept.append(row)
            data["cases"] = kept
            with open(path, "w", newline="\n") as fh:
                json.dump(data, fh, indent=1)
                fh.write("\n")


if __name__ == "__main__":
    main()
