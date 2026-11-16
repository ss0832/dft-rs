# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Five-molecule energy oracle of the functionals added to the registry:
# PySCF RKS energies (libxc evaluates the functional; used strictly as a
# numerical oracle) for H2, LiH, water, phosphine and ferrocene in STO-3G at
# grid level 9, geometries in Bohr shared with
# tests/oracle_data/multicomponent_hf.json.
#
# Run (WSL): <venv>/bin/python gen_functional_molecules.py [name ...]
# Existing entries of other functionals are kept, so the file grows batch by
# batch.

import json
import os
import sys

import pyscf
from pyscf import dft, gto

from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

SYMBOLS = {1: "H", 3: "Li", 6: "C", 8: "O", 15: "P", 26: "Fe"}

# (dft-rs specification, pyscf xc string, optional radial points per atom)
# The specification is a combined name or an [exchange, correlation] pair.
FUNCTIONALS = [
    ("pbesol", "gga_x_pbe_sol,gga_c_pbe_sol", None),
    ("rpbe", "gga_x_rpbe,gga_c_pbe", None),
    ("pw91", "gga_x_pw91,gga_c_pw91", None),
    ("scan", "mgga_x_scan,mgga_c_scan", 200),
    ("lrc-wpbeh", "HYB_GGA_XC_LRC_WPBEH", None),
    ("spz81", "lda_x,lda_c_pz", None),
    ("xalpha07", "1.05*lda_x,", None),
    (["slater", "wigner"], "lda_x,lda_c_wigner", None),
    (["slater", "hl"], "lda_x,lda_c_hl", None),
    (["slater", "gl"], "lda_x,lda_c_gl", None),
    ("bpw91", "gga_x_b88,gga_c_pw91", None),
    ("olyp", "gga_x_optx,gga_c_lyp", None),
    ("opbe", "gga_x_optx,gga_c_pbe", None),
    ("opw91", "gga_x_optx,gga_c_pw91", None),
    ("mpwlyp", "gga_x_mpw91,gga_c_lyp", None),
    ("mpwpw91", "gga_x_mpw91,gga_c_pw91", None),
    ("mpwpbe", "gga_x_mpw91,gga_c_pbe", None),
    ("g96lyp", "gga_x_g96,gga_c_lyp", None),
    ("xlyp", "gga_xc_xlyp", None),
    ("kt1", "gga_xc_kt1", None),
    ("kt2", "gga_xc_kt2", None),
    ("kt3", "gga_xc_kt3", None),
    ("sogga", "gga_x_sogga,gga_c_pbe", None),
    ("xpbe", "gga_x_xpbe,gga_c_xpbe", None),
    ("apbe", "gga_x_apbe,gga_c_apbe", None),
    ("pbeint", "gga_x_pbeint,gga_c_pbeint", None),
    ("wc", "gga_x_wc,gga_c_pbe", None),
    ("mpbe", "gga_x_mpbe,gga_c_pbe", None),
    ("b97-d", "gga_xc_b97_d", None),
    (["pw86", "pbe"], "gga_x_pw86,gga_c_pbe", None),
    (["lg93", "pbe"], "gga_x_lg93,gga_c_pbe", None),
    (["ft97", "pbe"], "gga_x_ft97_b,gga_c_pbe", None),
    ("am05", "gga_x_am05,gga_c_am05", None),
    ("n12", "gga_x_n12,gga_c_n12", None),
    ("gam", "gga_x_gam,gga_c_gam", None),
    ("sogga11", "gga_x_sogga11,gga_c_sogga11", None),
    ("pbeloc", "gga_x_pbe,gga_c_pbeloc", None),
    ("bop", "gga_x_b88,gga_c_op_b88", None),
    ("edf1", "gga_xc_edf1", None),
    (["pbe", "vpbe"], "gga_x_pbe,gga_c_pbe_vwn", None),
    ("rscan", "mgga_x_rscan,mgga_c_rscan", None),
    ("revtpss", "mgga_x_revtpss,mgga_c_revtpss", None),
    ("modtpss", "mgga_x_modtpss,mgga_c_tpss", None),
    ("bb95", "gga_x_b88,mgga_c_bc95", None),
    ("b1b95", "HYB_MGGA_XC_B88B95", None),
    ("bb1k", "HYB_MGGA_XC_BB1K", None),
    ("mpw1b95", "HYB_MGGA_XC_MPW1B95", None),
    ("pw6b95", "HYB_MGGA_XC_PW6B95", None),
    ("pwb6k", "HYB_MGGA_XC_PWB6K", None),
    ("b3lyp*", "HYB_GGA_XC_B3LYPs", None),
    ("x3lyp", "HYB_GGA_XC_X3LYP", None),
    ("b3p86", "HYB_GGA_XC_B3P86", None),
    ("mpw3pw91", "HYB_GGA_XC_MPW3PW", None),
    ("b1lyp", "HYB_GGA_XC_B1LYP", None),
    ("b1pw91", "HYB_GGA_XC_B1PW91", None),
    ("bhandh", "HYB_GGA_XC_BHANDH", None),
    ("bhandhlyp", "HYB_GGA_XC_BHANDHLYP", None),
    ("pbe0-1/3", "HYB_GGA_XC_PBE0_13", None),
    ("mpw1pw91", "HYB_GGA_XC_MPW1PW", None),
    ("mpw1lyp", "HYB_GGA_XC_MPW1LYP", None),
    ("mpw1pbe", "HYB_GGA_XC_MPW1PBE", None),
    ("mpw1k", "HYB_GGA_XC_MPW1K", None),
    ("b97-1", "HYB_GGA_XC_B97_1", None),
    ("b97-2", "HYB_GGA_XC_B97_2", None),
    ("b97-3", "HYB_GGA_XC_B97_3", None),
    ("b97-k", "HYB_GGA_XC_B97_K", None),
    ("tpss0", "HYB_MGGA_XC_TPSS0", None),
    ("revtpssh", "HYB_MGGA_XC_REVTPSSH", None),
    ("scan0", "HYB_MGGA_X_SCAN0,MGGA_C_SCAN", None),
    ("r2scan0", "HYB_MGGA_XC_R2SCAN0", None),
    ("r2scanh", "HYB_MGGA_XC_R2SCANH", None),
    ("r2scan50", "HYB_MGGA_XC_R2SCAN50", None),
    ("sogga11-x", "HYB_GGA_X_SOGGA11_X,GGA_C_SOGGA11_X", None),
    ("m05", "HYB_MGGA_X_M05,MGGA_C_M05", None),
    ("m05-2x", "HYB_MGGA_X_M05_2X,MGGA_C_M05_2X", None),
    ("m06-hf", "HYB_MGGA_X_M06_HF,MGGA_C_M06_HF", None),
    ("apf", "HYB_GGA_XC_APF", None),
    ("rcam-b3lyp", "HYB_GGA_XC_RCAM_B3LYP", None),
    ("lc-bop", "HYB_GGA_XC_LC_BOP", None),
    ("vsxc", "MGGA_X_GVT4,MGGA_C_VSXC", None),
    ("mn12-l", "MGGA_X_MN12_L,MGGA_C_MN12_L", None),
    ("mn12-sx", "HYB_MGGA_X_MN12_SX,MGGA_C_MN12_SX", None),
    ("mn15-l", "MGGA_X_MN15_L,MGGA_C_MN15_L", None),
    ("mn15", "HYB_MGGA_X_MN15,MGGA_C_MN15", None),
    ("m08-hx", "HYB_MGGA_X_M08_HX,MGGA_C_M08_HX", None),
    ("m08-so", "HYB_MGGA_X_M08_SO,MGGA_C_M08_SO", None),
    ("m11", "HYB_MGGA_X_M11,MGGA_C_M11", None),
    ("revm11", "HYB_MGGA_X_REVM11,MGGA_C_REVM11", None),
    ("m11-l", "MGGA_X_M11_L,MGGA_C_M11_L", None),
]


def label(spec):
    return spec if isinstance(spec, str) else "/".join(spec)


def molecules():
    with open(os.path.join(OUT_DIR, "multicomponent_hf.json"), encoding="utf-8") as fh:
        cases = json.load(fh)
    return {c["name"]: (c["numbers"], c["positions"]) for c in cases}


def energy(numbers, positions, xc, radial):
    elements = sorted({SYMBOLS[z] for z in numbers})
    basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), set(elements))
    geom = "; ".join(f"{SYMBOLS[z]} {x} {y} {w}" for z, (x, y, w) in zip(numbers, positions))
    mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=0, spin=0, cart=False, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 9
    if radial is not None:
        mf.grids.atom_grid = {s: (radial, 590) for s in elements}
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    return mf.e_tot, bool(mf.converged)


def main():
    only = set(sys.argv[1:])
    path = os.path.join(OUT_DIR, "functional_molecules.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {"provenance": "", "basis": "sto-3g", "grid_level": 9, "molecules": {}, "cases": []}
    data["provenance"] = f"PySCF {pyscf.__version__} RKS (libxc-backed, values only), grid level 9"
    mols = molecules()
    data["molecules"] = {k: {"numbers": v[0], "positions": v[1]} for k, v in mols.items()}
    done = {label(c["spec"]): c for c in data["cases"]}
    for spec, xc, radial in FUNCTIONALS:
        name = label(spec)
        if only and name not in only:
            continue
        energies, converged = {}, {}
        for mol_name, (numbers, positions) in mols.items():
            e, ok = energy(numbers, positions, xc, radial)
            energies[mol_name] = e
            converged[mol_name] = ok
            print(f"{name} {mol_name}: E = {e:.10f} converged {ok}", flush=True)
        done[name] = {"spec": spec, "pyscf_xc": xc, "radial": radial,
                      "energies": energies, "converged": converged}
        data["cases"] = list(done.values())
        with open(path, "w", newline="\n") as fh:
            json.dump(data, fh, indent=1)
            fh.write("\n")


if __name__ == "__main__":
    main()
