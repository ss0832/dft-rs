# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Pointwise oracle of the functional-extensibility set: energy per volume
# and first derivatives (rho, sigma, tau per spin) of every new exchange
# and correlation component, and of the semilocal part of LRC-wPBEh, at
# fixed (rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b) points.
# PySCF evaluates libxc here strictly as a numerical oracle.
#
# Run (WSL): <venv>/bin/python gen_functional_points.py

import json
import os

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf.dft import libxc

from gen_integrals import OUT_DIR

K_FACTOR_C = 0.3 * (6.0 * np.pi**2) ** (2.0 / 3.0)


def alpha_point(rho, sigma, alpha):
    """Spin-unpolarized point whose per-spin iso-orbital indicator is alpha."""
    tau_w = sigma / (8.0 * rho)
    tau_unif = K_FACTOR_C * rho ** (5.0 / 3.0)
    tau = tau_w + alpha * tau_unif
    return [rho, rho, sigma, sigma, sigma, tau, tau]


POINTS = [
    [0.3, 0.1, 0.0425, 0.031, 0.0233, 0.35, 0.12],
    [1.2, 0.9, 0.29, 0.055, 0.1725, 2.2, 1.6],
    [0.5, 0.4, 0.1, 0.05, 0.08, 3.0, 2.5],
    [0.5, 0.4, 0.1, 0.05, 0.08, 0.01, 0.008],
    [0.4, 0.0, 0.09, 0.0, 0.0, 0.5, 0.0],
    [0.02, 0.015, 1e-4, 5e-5, 6e-5, 0.004, 0.003],
    alpha_point(0.25, 0.03, 0.0),
    alpha_point(0.25, 0.03, 1.0),
    alpha_point(0.25, 0.03, 5.0),
]

# (table, dft-rs name, libxc code, family)
COMPONENTS = [
    ("x", "pbesol", "gga_x_pbe_sol,", "gga"),
    ("c", "pbesol", ",gga_c_pbe_sol", "gga"),
    ("x", "rpbe", "gga_x_rpbe,", "gga"),
    ("x", "pw91", "gga_x_pw91,", "gga"),
    ("x", "scan", "mgga_x_scan,", "mgga"),
    ("c", "scan", ",mgga_c_scan", "mgga"),
    ("xc", "lrc-wpbeh", "HYB_GGA_XC_LRC_WPBEH", "gga"),
    ("x", "xalpha07", "1.05*lda_x,", "lda"),
    ("c", "pz81", ",lda_c_pz", "lda"),
    ("c", "wigner", ",lda_c_wigner", "lda"),
    ("c", "hl", ",lda_c_hl", "lda"),
    ("c", "gl", ",lda_c_gl", "lda"),
    ("x", "pw86", "gga_x_pw86,", "gga"),
    ("x", "mpw91", "gga_x_mpw91,", "gga"),
    ("x", "g96", "gga_x_g96,", "gga"),
    ("x", "lg93", "gga_x_lg93,", "gga"),
    ("x", "ft97", "gga_x_ft97_b,", "gga"),
    ("x", "xpbe", "gga_x_xpbe,", "gga"),
    ("c", "xpbe", ",gga_c_xpbe", "gga"),
    ("x", "sogga", "gga_x_sogga,", "gga"),
    ("x", "mpbe", "gga_x_mpbe,", "gga"),
    ("x", "apbe", "gga_x_apbe,", "gga"),
    ("c", "apbe", ",gga_c_apbe", "gga"),
    ("x", "pbeint", "gga_x_pbeint,", "gga"),
    ("c", "pbeint", ",gga_c_pbeint", "gga"),
    ("x", "wc", "gga_x_wc,", "gga"),
    ("x", "kt1", "gga_x_kt1,", "gga"),
    ("xc", "xlyp", "gga_xc_xlyp", "gga"),
    ("xc", "kt2", "gga_xc_kt2", "gga"),
    ("xc", "kt3", "gga_xc_kt3", "gga"),
    ("xc", "b97-d", "gga_xc_b97_d", "gga"),
    ("x", "am05", "gga_x_am05,", "gga"),
    ("c", "am05", ",gga_c_am05", "gga"),
    ("x", "n12", "gga_x_n12,", "gga"),
    ("c", "n12", ",gga_c_n12", "gga"),
    ("x", "gam", "gga_x_gam,", "gga"),
    ("c", "gam", ",gga_c_gam", "gga"),
    ("x", "sogga11", "gga_x_sogga11,", "gga"),
    ("c", "sogga11", ",gga_c_sogga11", "gga"),
    ("c", "pbeloc", ",gga_c_pbeloc", "gga"),
    ("c", "vpbe", ",gga_c_pbe_vwn", "gga"),
    ("c", "op", ",gga_c_op_b88", "gga"),
    ("xc", "edf1", "gga_xc_edf1", "gga"),
    ("c", "p86", ",gga_c_p86", "gga"),
    ("c", "vwn3", ",lda_c_vwn_rpa", "lda"),
    ("x", "rscan", "mgga_x_rscan,", "mgga"),
    ("c", "rscan", ",mgga_c_rscan", "mgga"),
    ("x", "revtpss", "mgga_x_revtpss,", "mgga"),
    ("c", "revtpss", ",mgga_c_revtpss", "mgga"),
    ("x", "modtpss", "mgga_x_modtpss,", "mgga"),
    ("c", "b95", ",mgga_c_bc95", "mgga"),
    ("xc", "b1b95", "HYB_MGGA_XC_B88B95", "mgga"),
    ("xc", "bb1k", "HYB_MGGA_XC_BB1K", "mgga"),
    ("xc", "mpw1b95", "HYB_MGGA_XC_MPW1B95", "mgga"),
    ("xc", "pw6b95", "HYB_MGGA_XC_PW6B95", "mgga"),
    ("xc", "pwb6k", "HYB_MGGA_XC_PWB6K", "mgga"),
    ("xc", "b3lyp*", "HYB_GGA_XC_B3LYPs", "gga"),
    ("xc", "x3lyp", "HYB_GGA_XC_X3LYP", "gga"),
    ("xc", "b3p86", "HYB_GGA_XC_B3P86", "gga"),
    ("xc", "mpw3pw91", "HYB_GGA_XC_MPW3PW", "gga"),
    ("xc", "b1lyp", "HYB_GGA_XC_B1LYP", "gga"),
    ("xc", "b1pw91", "HYB_GGA_XC_B1PW91", "gga"),
    ("xc", "bhandh", "HYB_GGA_XC_BHANDH", "gga"),
    ("xc", "bhandhlyp", "HYB_GGA_XC_BHANDHLYP", "gga"),
    ("xc", "pbe0-1/3", "HYB_GGA_XC_PBE0_13", "gga"),
    ("xc", "mpw1pw91", "HYB_GGA_XC_MPW1PW", "gga"),
    ("xc", "mpw1lyp", "HYB_GGA_XC_MPW1LYP", "gga"),
    ("xc", "mpw1pbe", "HYB_GGA_XC_MPW1PBE", "gga"),
    ("xc", "mpw1k", "HYB_GGA_XC_MPW1K", "gga"),
    ("xc", "b97-1", "HYB_GGA_XC_B97_1", "gga"),
    ("xc", "b97-2", "HYB_GGA_XC_B97_2", "gga"),
    ("xc", "b97-3", "HYB_GGA_XC_B97_3", "gga"),
    ("xc", "b97-k", "HYB_GGA_XC_B97_K", "gga"),
    ("xc", "tpss0", "HYB_MGGA_XC_TPSS0", "mgga"),
    ("xc", "revtpssh", "HYB_MGGA_XC_REVTPSSH", "mgga"),
    ("xc", "scan0", "HYB_MGGA_X_SCAN0,MGGA_C_SCAN", "mgga"),
    ("xc", "r2scan0", "HYB_MGGA_XC_R2SCAN0", "mgga"),
    ("xc", "r2scanh", "HYB_MGGA_XC_R2SCANH", "mgga"),
    ("xc", "r2scan50", "HYB_MGGA_XC_R2SCAN50", "mgga"),
    ("xc", "sogga11-x", "HYB_GGA_X_SOGGA11_X,GGA_C_SOGGA11_X", "gga"),
    ("xc", "m05", "HYB_MGGA_X_M05,MGGA_C_M05", "mgga"),
    ("xc", "m05-2x", "HYB_MGGA_X_M05_2X,MGGA_C_M05_2X", "mgga"),
    ("xc", "m06-hf", "HYB_MGGA_X_M06_HF,MGGA_C_M06_HF", "mgga"),
    ("xc", "apf", "HYB_GGA_XC_APF", "gga"),
    ("xc", "rcam-b3lyp", "HYB_GGA_XC_RCAM_B3LYP", "gga"),
    ("xc", "lc-bop", "HYB_GGA_XC_LC_BOP", "gga"),
    ("xc", "vsxc", "MGGA_X_GVT4,MGGA_C_VSXC", "mgga"),
    ("xc", "mn12-l", "MGGA_X_MN12_L,MGGA_C_MN12_L", "mgga"),
    ("xc", "mn12-sx", "HYB_MGGA_X_MN12_SX,MGGA_C_MN12_SX", "mgga"),
    ("xc", "mn15-l", "MGGA_X_MN15_L,MGGA_C_MN15_L", "mgga"),
    ("xc", "mn15", "HYB_MGGA_X_MN15,MGGA_C_MN15", "mgga"),
    ("xc", "m08-hx", "HYB_MGGA_X_M08_HX,MGGA_C_M08_HX", "mgga"),
    ("xc", "m08-so", "HYB_MGGA_X_M08_SO,MGGA_C_M08_SO", "mgga"),
    ("xc", "m11", "HYB_MGGA_X_M11,MGGA_C_M11", "mgga"),
    ("xc", "revm11", "HYB_MGGA_X_REVM11,MGGA_C_REVM11", "mgga"),
    ("xc", "m11-l", "MGGA_X_M11_L,MGGA_C_M11_L", "mgga"),
]


def spin_inputs(p, mgga):
    saa, sab, sbb = p[2], p[3], p[4]
    ga = np.array([np.sqrt(saa), 0.0, 0.0])
    if saa > 0:
        gbx = sab / np.sqrt(saa)
        gby = np.sqrt(max(sbb - gbx * gbx, 0.0))
    else:
        gbx, gby = np.sqrt(sbb), 0.0
    if mgga:
        va = np.vstack([[p[0]], [ga[0]], [ga[1]], [ga[2]], [0.0], [p[5]]])
        vb = np.vstack([[p[1]], [gbx], [gby], [0.0], [0.0], [p[6]]])
    else:
        va = np.vstack([[p[0]], [ga[0]], [ga[1]], [ga[2]]])
        vb = np.vstack([[p[1]], [gbx], [gby], [0.0]])
    return va, vb


def main():
    out = {
        "provenance": f"PySCF {pyscf_version} libxc point values (energy per volume, first derivatives)",
        "variables": "rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b",
        "points": POINTS,
        "components": [],
    }
    for table, name, code, family in COMPONENTS:
        rows = []
        for p in POINTS:
            va, vb = spin_inputs(p, family == "mgga")
            if family == "lda":
                va, vb = va[0], vb[0]
            exc, vxc = libxc.eval_xc(code, (va, vb), spin=1, deriv=1)[:2]
            rho = p[0] + p[1]
            vrho = np.asarray(vxc[0])[0].tolist()
            vsigma = np.asarray(vxc[1])[0].tolist() if family != "lda" else [0.0, 0.0, 0.0]
            vtau = np.asarray(vxc[3])[0].tolist() if family == "mgga" else [0.0, 0.0]
            rows.append({"e": float(exc[0] * rho), "vrho": vrho, "vsigma": vsigma, "vtau": vtau})
        out["components"].append(
            {"table": table, "name": name, "libxc": code, "family": family, "values": rows}
        )
    path = os.path.join(OUT_DIR, "functional_points.json")
    with open(path, "w", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
