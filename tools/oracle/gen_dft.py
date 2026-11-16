# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Phase 3 oracle generator: frozen PySCF RKS/UKS LDA energies.
# PySCF evaluates functionals through libxc — used here strictly as a
# NUMERICAL oracle (energy values only). Grid level 9 ≈ grid-converged.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_dft.py

import os
import pyscf
from pyscf import gto, dft

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

# (case tag, molecule, basis file, dft-rs functional name, pyscf xc string)
CASES = [
    ("h2o", "sto-3g", "svwn5", "slater,vwn"),
    ("h2o", "def2-svp", "svwn5", "slater,vwn"),
    ("h2o", "sto-3g", "svwn3", "slater,vwn_rpa"),
    ("h2o", "def2-svp", "svwn3", "slater,vwn_rpa"),
    ("h2o", "sto-3g", "spw92", "slater,pw"),
    ("h2o", "def2-svp", "spw92", "slater,pw"),
    ("nh3", "def2-svp", "svwn5", "slater,vwn"),
    ("o2_triplet", "def2-svp", "svwn5", "slater,vwn"),
    ("ch3_doublet", "sto-3g", "svwn5", "slater,vwn"),
    # GGA
    ("h2o", "sto-3g", "pbe", "pbe,pbe"),
    ("h2o", "def2-svp", "pbe", "pbe,pbe"),
    ("h2o", "def2-svp", "blyp", "b88,lyp"),
    ("nh3", "def2-svp", "blyp", "b88,lyp"),
    ("o2_triplet", "def2-svp", "pbe", "pbe,pbe"),
    # Hybrids
    ("h2o", "sto-3g", "b3lyp", "b3lypg"),
    ("h2o", "def2-svp", "b3lyp", "b3lypg"),
    ("h2o", "def2-svp", "b3lyp5", "b3lyp5"),
    ("h2o", "def2-svp", "o3lyp", "o3lyp"),
    ("h2o", "def2-svp", "pbe0", "pbe0"),
    ("ch3_doublet", "def2-svp", "b3lyp", "b3lypg"),
    # Range-separated hybrids
    ("h2o", "sto-3g", "cam-b3lyp", "camb3lyp"),
    ("h2o", "def2-svp", "cam-b3lyp", "camb3lyp"),
    ("ch3_doublet", "def2-svp", "cam-b3lyp", "camb3lyp"),
    # LC-BLYP: no libxc alias in this build and PySCF's compound-string RSH
    # path errors out — covered instead by internal Rust gates
    # (ω→0 → BLYP; ω→∞ LR-HF → full HF) plus the CAM-B3LYP oracle, which
    # exercises the same ITYH + K_erf machinery.
    # B97 family (XC-integrated)
    ("h2o", "def2-svp", "b97", "b97"),
    ("h2o", "def2-svp", "hcth93", "hcth93"),
    ("h2o", "def2-svp", "hcth120", "hcth120"),
    ("h2o", "def2-svp", "hcth147", "hcth147"),
    ("h2o", "def2-svp", "hcth407", "hcth407"),
    ("ch3_doublet", "def2-svp", "hcth407", "hcth407"),
    # ωB97 family (semilocal + RSH exchange; no dispersion/NLC terms here).
    # wb97x_d/wb97x_v SCF is unsupported by this PySCF build — those two are
    # pinned pointwise against libxc in the Rust unit tests instead.
    ("h2o", "def2-svp", "wb97", "wb97"),
    ("h2o", "def2-svp", "wb97x", "wb97x"),
    # ωB97M-V (semilocal part only — no VV10 here; NLC gated separately)
    ("h2o", "def2-svp", "wb97mv", "wb97m_v"),
    # revPBE + B97M-V (semilocal part)
    ("h2o", "sto-3g", "revpbe", "gga_x_pbe_r,gga_c_pbe"),
    ("h2o", "sto-3g", "b97mv", "b97m_v"),
    # TPSS family
    ("h2o", "sto-3g", "tpss", "mgga_x_tpss,mgga_c_tpss"),
    ("h2o", "sto-3g", "tpssh", "HYB_MGGA_XC_TPSSH"),
    ("ch3_doublet", "sto-3g", "tpss", "mgga_x_tpss,mgga_c_tpss"),
    # M06 family (meta-GGA)
    ("h2o", "def2-svp", "m06l", "m06_l"),
    ("h2o", "def2-svp", "m06", "m06"),
    ("h2o", "def2-svp", "m062x", "m06_2x"),
    ("ch3_doublet", "def2-svp", "m062x", "m06_2x"),
    ("h2o", "sto-3g", "revm06l", "mgga_x_revm06_l,mgga_c_revm06_l"),
    ("h2o", "sto-3g", "revm06", "hyb_mgga_x_revm06,mgga_c_revm06"),
    # Legacy correlation (BP86 / B3PW91); open-shell cases exercise the
    # P86 / PW91-C ferro branches inside a real SCF
    ("h2o", "sto-3g", "bp86", "gga_x_b88,gga_c_p86"),
    ("h2o", "def2-svp", "bp86", "gga_x_b88,gga_c_p86"),
    ("ch3_doublet", "sto-3g", "bp86", "gga_x_b88,gga_c_p86"),
    ("h2o", "def2-svp", "b3pw91", "HYB_GGA_XC_B3PW91"),
    ("ch3_doublet", "sto-3g", "b3pw91", "HYB_GGA_XC_B3PW91"),
    # LC-ωPBE with the HJS 2008 SR hole (ω = 0.4, full LR-HF)
    ("h2o", "def2-svp", "lc-wpbe-hjs", "HYB_GGA_XC_LC_WPBE_WHS"),
    ("ch3_doublet", "sto-3g", "lc-wpbe-hjs", "HYB_GGA_XC_LC_WPBE_WHS"),
    # HSE06 screened hybrid + canonical LC-ωPBE (HSE/wPBEh hole)
    ("h2o", "def2-svp", "hse06", "HSE06"),
    ("ch3_doublet", "sto-3g", "hse06", "HSE06"),
    ("h2o", "def2-svp", "lc-wpbe", "HYB_GGA_XC_LC_WPBE"),
    ("ch3_doublet", "sto-3g", "lc-wpbe", "HYB_GGA_XC_LC_WPBE"),
    # M06-SX screened Minnesota hybrid
    ("h2o", "sto-3g", "m06sx", "hyb_mgga_x_m06_sx,mgga_c_m06_sx"),
    # r2SCAN
    ("h2o", "sto-3g", "r2scan", "mgga_x_r2scan,mgga_c_r2scan"),
    ("ch3_doublet", "sto-3g", "r2scan", "mgga_x_r2scan,mgga_c_r2scan"),
    # Functional-extensibility set: PBEsol, RPBE, PW91, SCAN, LRC-wPBEh
    ("h2o", "sto-3g", "pbesol", "gga_x_pbe_sol,gga_c_pbe_sol"),
    ("ch3_doublet", "sto-3g", "pbesol", "gga_x_pbe_sol,gga_c_pbe_sol"),
    ("h2o", "sto-3g", "rpbe", "gga_x_rpbe,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "rpbe", "gga_x_rpbe,gga_c_pbe"),
    ("h2o", "sto-3g", "pw91", "gga_x_pw91,gga_c_pw91"),
    ("ch3_doublet", "sto-3g", "pw91", "gga_x_pw91,gga_c_pw91"),
    ("h2o", "sto-3g", "scan", "mgga_x_scan,mgga_c_scan"),
    ("ch3_doublet", "sto-3g", "scan", "mgga_x_scan,mgga_c_scan"),
    ("h2o", "def2-svp", "lrc-wpbeh", "HYB_GGA_XC_LRC_WPBEH"),
    ("ch3_doublet", "sto-3g", "lrc-wpbeh", "HYB_GGA_XC_LRC_WPBEH"),
    # Catalogue batch 1 (LDA)
    ("h2o", "sto-3g", "spz81", "lda_x,lda_c_pz"),
    ("ch3_doublet", "sto-3g", "spz81", "lda_x,lda_c_pz"),
    ("h2o", "sto-3g", "xalpha07", "1.05*lda_x,"),
    # Catalogue batch 2a (GGA)
    ("h2o", "sto-3g", "bpw91", "gga_x_b88,gga_c_pw91"),
    ("ch3_doublet", "sto-3g", "bpw91", "gga_x_b88,gga_c_pw91"),
    ("h2o", "sto-3g", "olyp", "gga_x_optx,gga_c_lyp"),
    ("ch3_doublet", "sto-3g", "olyp", "gga_x_optx,gga_c_lyp"),
    ("h2o", "sto-3g", "opbe", "gga_x_optx,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "opbe", "gga_x_optx,gga_c_pbe"),
    ("h2o", "sto-3g", "opw91", "gga_x_optx,gga_c_pw91"),
    ("ch3_doublet", "sto-3g", "opw91", "gga_x_optx,gga_c_pw91"),
    ("h2o", "sto-3g", "mpwlyp", "gga_x_mpw91,gga_c_lyp"),
    ("ch3_doublet", "sto-3g", "mpwlyp", "gga_x_mpw91,gga_c_lyp"),
    ("h2o", "sto-3g", "mpwpw91", "gga_x_mpw91,gga_c_pw91"),
    ("ch3_doublet", "sto-3g", "mpwpw91", "gga_x_mpw91,gga_c_pw91"),
    ("h2o", "sto-3g", "mpwpbe", "gga_x_mpw91,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "mpwpbe", "gga_x_mpw91,gga_c_pbe"),
    ("h2o", "sto-3g", "g96lyp", "gga_x_g96,gga_c_lyp"),
    ("ch3_doublet", "sto-3g", "g96lyp", "gga_x_g96,gga_c_lyp"),
    ("h2o", "sto-3g", "xlyp", "gga_xc_xlyp"),
    ("ch3_doublet", "sto-3g", "xlyp", "gga_xc_xlyp"),
    ("h2o", "sto-3g", "kt1", "gga_xc_kt1"),
    ("ch3_doublet", "sto-3g", "kt1", "gga_xc_kt1"),
    ("h2o", "sto-3g", "kt2", "gga_xc_kt2"),
    ("ch3_doublet", "sto-3g", "kt2", "gga_xc_kt2"),
    ("h2o", "sto-3g", "kt3", "gga_xc_kt3"),
    ("ch3_doublet", "sto-3g", "kt3", "gga_xc_kt3"),
    ("h2o", "sto-3g", "sogga", "gga_x_sogga,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "sogga", "gga_x_sogga,gga_c_pbe"),
    ("h2o", "sto-3g", "xpbe", "gga_x_xpbe,gga_c_xpbe"),
    ("ch3_doublet", "sto-3g", "xpbe", "gga_x_xpbe,gga_c_xpbe"),
    ("h2o", "sto-3g", "apbe", "gga_x_apbe,gga_c_apbe"),
    ("ch3_doublet", "sto-3g", "apbe", "gga_x_apbe,gga_c_apbe"),
    ("h2o", "sto-3g", "pbeint", "gga_x_pbeint,gga_c_pbeint"),
    ("ch3_doublet", "sto-3g", "pbeint", "gga_x_pbeint,gga_c_pbeint"),
    ("h2o", "sto-3g", "wc", "gga_x_wc,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "wc", "gga_x_wc,gga_c_pbe"),
    ("h2o", "sto-3g", "mpbe", "gga_x_mpbe,gga_c_pbe"),
    ("ch3_doublet", "sto-3g", "mpbe", "gga_x_mpbe,gga_c_pbe"),
    ("h2o", "sto-3g", "b97-d", "gga_xc_b97_d"),
    ("ch3_doublet", "sto-3g", "b97-d", "gga_xc_b97_d"),
    # Catalogue batch 2b (GGA)
    ("h2o", "sto-3g", "am05", "gga_x_am05,gga_c_am05"),
    ("ch3_doublet", "sto-3g", "am05", "gga_x_am05,gga_c_am05"),
    ("h2o", "sto-3g", "n12", "gga_x_n12,gga_c_n12"),
    ("ch3_doublet", "sto-3g", "n12", "gga_x_n12,gga_c_n12"),
    ("h2o", "sto-3g", "gam", "gga_x_gam,gga_c_gam"),
    ("ch3_doublet", "sto-3g", "gam", "gga_x_gam,gga_c_gam"),
    ("h2o", "sto-3g", "sogga11", "gga_x_sogga11,gga_c_sogga11"),
    ("ch3_doublet", "sto-3g", "sogga11", "gga_x_sogga11,gga_c_sogga11"),
    ("h2o", "sto-3g", "pbeloc", "gga_x_pbe,gga_c_pbeloc"),
    ("ch3_doublet", "sto-3g", "pbeloc", "gga_x_pbe,gga_c_pbeloc"),
    ("h2o", "sto-3g", "bop", "gga_x_b88,gga_c_op_b88"),
    ("ch3_doublet", "sto-3g", "bop", "gga_x_b88,gga_c_op_b88"),
    ("h2o", "sto-3g", "edf1", "gga_xc_edf1"),
    ("ch3_doublet", "sto-3g", "edf1", "gga_xc_edf1"),
    ("h2o", "sto-3g", "rscan", "mgga_x_rscan,mgga_c_rscan"),
    ("ch3_doublet", "sto-3g", "rscan", "mgga_x_rscan,mgga_c_rscan"),
    ("h2o", "sto-3g", "revtpss", "mgga_x_revtpss,mgga_c_revtpss"),
    ("ch3_doublet", "sto-3g", "revtpss", "mgga_x_revtpss,mgga_c_revtpss"),
    ("h2o", "sto-3g", "modtpss", "mgga_x_modtpss,mgga_c_tpss"),
    ("ch3_doublet", "sto-3g", "modtpss", "mgga_x_modtpss,mgga_c_tpss"),
    ("h2o", "sto-3g", "bb95", "gga_x_b88,mgga_c_bc95"),
    ("ch3_doublet", "sto-3g", "bb95", "gga_x_b88,mgga_c_bc95"),
    ("h2o", "sto-3g", "b1b95", "HYB_MGGA_XC_B88B95"),
    ("ch3_doublet", "sto-3g", "b1b95", "HYB_MGGA_XC_B88B95"),
    ("h2o", "sto-3g", "bb1k", "HYB_MGGA_XC_BB1K"),
    ("ch3_doublet", "sto-3g", "bb1k", "HYB_MGGA_XC_BB1K"),
    ("h2o", "sto-3g", "mpw1b95", "HYB_MGGA_XC_MPW1B95"),
    ("ch3_doublet", "sto-3g", "mpw1b95", "HYB_MGGA_XC_MPW1B95"),
    ("h2o", "sto-3g", "pw6b95", "HYB_MGGA_XC_PW6B95"),
    ("ch3_doublet", "sto-3g", "pw6b95", "HYB_MGGA_XC_PW6B95"),
    ("h2o", "sto-3g", "pwb6k", "HYB_MGGA_XC_PWB6K"),
    ("ch3_doublet", "sto-3g", "pwb6k", "HYB_MGGA_XC_PWB6K"),
    ("h2o", "sto-3g", "b3lyp*", "HYB_GGA_XC_B3LYPs"),
    ("ch3_doublet", "sto-3g", "b3lyp*", "HYB_GGA_XC_B3LYPs"),
    ("h2o", "sto-3g", "x3lyp", "HYB_GGA_XC_X3LYP"),
    ("ch3_doublet", "sto-3g", "x3lyp", "HYB_GGA_XC_X3LYP"),
    ("h2o", "sto-3g", "b3p86", "HYB_GGA_XC_B3P86"),
    ("ch3_doublet", "sto-3g", "b3p86", "HYB_GGA_XC_B3P86"),
    ("h2o", "sto-3g", "mpw3pw91", "HYB_GGA_XC_MPW3PW"),
    ("ch3_doublet", "sto-3g", "mpw3pw91", "HYB_GGA_XC_MPW3PW"),
    ("h2o", "sto-3g", "b1lyp", "HYB_GGA_XC_B1LYP"),
    ("ch3_doublet", "sto-3g", "b1lyp", "HYB_GGA_XC_B1LYP"),
    ("h2o", "sto-3g", "b1pw91", "HYB_GGA_XC_B1PW91"),
    ("ch3_doublet", "sto-3g", "b1pw91", "HYB_GGA_XC_B1PW91"),
    ("h2o", "sto-3g", "bhandh", "HYB_GGA_XC_BHANDH"),
    ("ch3_doublet", "sto-3g", "bhandh", "HYB_GGA_XC_BHANDH"),
    ("h2o", "sto-3g", "bhandhlyp", "HYB_GGA_XC_BHANDHLYP"),
    ("ch3_doublet", "sto-3g", "bhandhlyp", "HYB_GGA_XC_BHANDHLYP"),
    ("h2o", "sto-3g", "pbe0-1/3", "HYB_GGA_XC_PBE0_13"),
    ("ch3_doublet", "sto-3g", "pbe0-1/3", "HYB_GGA_XC_PBE0_13"),
    ("h2o", "sto-3g", "mpw1pw91", "HYB_GGA_XC_MPW1PW"),
    ("ch3_doublet", "sto-3g", "mpw1pw91", "HYB_GGA_XC_MPW1PW"),
    ("h2o", "sto-3g", "mpw1lyp", "HYB_GGA_XC_MPW1LYP"),
    ("ch3_doublet", "sto-3g", "mpw1lyp", "HYB_GGA_XC_MPW1LYP"),
    ("h2o", "sto-3g", "mpw1pbe", "HYB_GGA_XC_MPW1PBE"),
    ("ch3_doublet", "sto-3g", "mpw1pbe", "HYB_GGA_XC_MPW1PBE"),
    ("h2o", "sto-3g", "mpw1k", "HYB_GGA_XC_MPW1K"),
    ("ch3_doublet", "sto-3g", "mpw1k", "HYB_GGA_XC_MPW1K"),
    ("h2o", "sto-3g", "b97-1", "HYB_GGA_XC_B97_1"),
    ("ch3_doublet", "sto-3g", "b97-1", "HYB_GGA_XC_B97_1"),
    ("h2o", "sto-3g", "b97-2", "HYB_GGA_XC_B97_2"),
    ("ch3_doublet", "sto-3g", "b97-2", "HYB_GGA_XC_B97_2"),
    ("h2o", "sto-3g", "b97-3", "HYB_GGA_XC_B97_3"),
    ("ch3_doublet", "sto-3g", "b97-3", "HYB_GGA_XC_B97_3"),
    ("h2o", "sto-3g", "b97-k", "HYB_GGA_XC_B97_K"),
    ("ch3_doublet", "sto-3g", "b97-k", "HYB_GGA_XC_B97_K"),
    ("h2o", "sto-3g", "tpss0", "HYB_MGGA_XC_TPSS0"),
    ("ch3_doublet", "sto-3g", "tpss0", "HYB_MGGA_XC_TPSS0"),
    ("h2o", "sto-3g", "revtpssh", "HYB_MGGA_XC_REVTPSSH"),
    ("ch3_doublet", "sto-3g", "revtpssh", "HYB_MGGA_XC_REVTPSSH"),
    ("h2o", "sto-3g", "scan0", "HYB_MGGA_X_SCAN0,MGGA_C_SCAN"),
    ("ch3_doublet", "sto-3g", "scan0", "HYB_MGGA_X_SCAN0,MGGA_C_SCAN"),
    ("h2o", "sto-3g", "r2scan0", "HYB_MGGA_XC_R2SCAN0"),
    ("ch3_doublet", "sto-3g", "r2scan0", "HYB_MGGA_XC_R2SCAN0"),
    ("h2o", "sto-3g", "r2scanh", "HYB_MGGA_XC_R2SCANH"),
    ("ch3_doublet", "sto-3g", "r2scanh", "HYB_MGGA_XC_R2SCANH"),
    ("h2o", "sto-3g", "r2scan50", "HYB_MGGA_XC_R2SCAN50"),
    ("ch3_doublet", "sto-3g", "r2scan50", "HYB_MGGA_XC_R2SCAN50"),
    ("h2o", "sto-3g", "sogga11-x", "HYB_GGA_X_SOGGA11_X,GGA_C_SOGGA11_X"),
    ("ch3_doublet", "sto-3g", "sogga11-x", "HYB_GGA_X_SOGGA11_X,GGA_C_SOGGA11_X"),
    ("h2o", "sto-3g", "m05", "HYB_MGGA_X_M05,MGGA_C_M05"),
    ("ch3_doublet", "sto-3g", "m05", "HYB_MGGA_X_M05,MGGA_C_M05"),
    ("h2o", "sto-3g", "m05-2x", "HYB_MGGA_X_M05_2X,MGGA_C_M05_2X"),
    ("ch3_doublet", "sto-3g", "m05-2x", "HYB_MGGA_X_M05_2X,MGGA_C_M05_2X"),
    ("h2o", "sto-3g", "m06-hf", "HYB_MGGA_X_M06_HF,MGGA_C_M06_HF"),
    ("ch3_doublet", "sto-3g", "m06-hf", "HYB_MGGA_X_M06_HF,MGGA_C_M06_HF"),
    ("h2o", "sto-3g", "apf", "HYB_GGA_XC_APF"),
    ("ch3_doublet", "sto-3g", "apf", "HYB_GGA_XC_APF"),
    ("h2o", "sto-3g", "rcam-b3lyp", "HYB_GGA_XC_RCAM_B3LYP"),
    ("ch3_doublet", "sto-3g", "rcam-b3lyp", "HYB_GGA_XC_RCAM_B3LYP"),
    ("h2o", "sto-3g", "lc-bop", "HYB_GGA_XC_LC_BOP"),
    ("ch3_doublet", "sto-3g", "lc-bop", "HYB_GGA_XC_LC_BOP"),
    ("h2o", "sto-3g", "vsxc", "MGGA_X_GVT4,MGGA_C_VSXC"),
    ("ch3_doublet", "sto-3g", "vsxc", "MGGA_X_GVT4,MGGA_C_VSXC"),
    ("h2o", "sto-3g", "mn12-l", "MGGA_X_MN12_L,MGGA_C_MN12_L"),
    ("ch3_doublet", "sto-3g", "mn12-l", "MGGA_X_MN12_L,MGGA_C_MN12_L"),
    ("h2o", "sto-3g", "mn12-sx", "HYB_MGGA_X_MN12_SX,MGGA_C_MN12_SX"),
    ("ch3_doublet", "sto-3g", "mn12-sx", "HYB_MGGA_X_MN12_SX,MGGA_C_MN12_SX"),
    ("h2o", "sto-3g", "mn15-l", "MGGA_X_MN15_L,MGGA_C_MN15_L"),
    ("ch3_doublet", "sto-3g", "mn15-l", "MGGA_X_MN15_L,MGGA_C_MN15_L"),
    ("h2o", "sto-3g", "mn15", "HYB_MGGA_X_MN15,MGGA_C_MN15"),
    ("ch3_doublet", "sto-3g", "mn15", "HYB_MGGA_X_MN15,MGGA_C_MN15"),
    ("h2o", "sto-3g", "m08-hx", "HYB_MGGA_X_M08_HX,MGGA_C_M08_HX"),
    ("ch3_doublet", "sto-3g", "m08-hx", "HYB_MGGA_X_M08_HX,MGGA_C_M08_HX"),
    ("h2o", "sto-3g", "m08-so", "HYB_MGGA_X_M08_SO,MGGA_C_M08_SO"),
    ("ch3_doublet", "sto-3g", "m08-so", "HYB_MGGA_X_M08_SO,MGGA_C_M08_SO"),
    ("h2o", "sto-3g", "m11", "HYB_MGGA_X_M11,MGGA_C_M11"),
    ("ch3_doublet", "sto-3g", "m11", "HYB_MGGA_X_M11,MGGA_C_M11"),
    ("h2o", "sto-3g", "revm11", "HYB_MGGA_X_REVM11,MGGA_C_REVM11"),
    ("ch3_doublet", "sto-3g", "revm11", "HYB_MGGA_X_REVM11,MGGA_C_REVM11"),
    ("h2o", "sto-3g", "m11-l", "MGGA_X_M11_L,MGGA_C_M11_L"),
    ("ch3_doublet", "sto-3g", "m11-l", "MGGA_X_M11_L,MGGA_C_M11_L"),
]


def main():
    import sys

    only = set(sys.argv[1:])  # optional functional-name filter (frozen-CSV hygiene)
    os.makedirs(OUT_DIR, exist_ok=True)
    for mol_name, basis_name, func_name, pyscf_xc in CASES:
        if only and func_name not in only:
            continue
        charge, mult, atoms = MOLECULES[mol_name]
        elements = sorted({sym for sym, *_ in atoms})
        basis = parse_g94(
            os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements)
        )
        geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
        mol = gto.M(
            atom=geom,
            basis=basis,
            unit="Bohr",
            charge=charge,
            spin=mult - 1,
            cart=False,
        )
        mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
        mf.xc = pyscf_xc
        if "ityh" in pyscf_xc:
            mf.omega = 0.33  # ω for the RSH components of a custom string
        if pyscf_xc.endswith("_v"):
            mf.nlc = 0  # semilocal part only (VV10 totals are gated separately)
        mf.grids.level = 9
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged, f"{mol_name}/{basis_name}/{func_name} not converged"
        # File names carry no path or wildcard characters (pbe0-1/3, b3lyp*).
        safe = func_name.replace("*", "star").replace("/", "_")
        case = f"{mol_name}_{basis_name.replace('-', '')}_{safe}"
        with open(os.path.join(OUT_DIR, f"{case}_dft.csv"), "w", newline="\n") as fh:
            fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} DFT oracle (libxc-backed, values only)\n")
            fh.write("# geometry unit: Bohr (exact literals); pyscf grid level 9\n")
            fh.write(f"basis,{basis_name}\n")
            fh.write(f"functional,{func_name}\n")
            fh.write(f"charge,{charge}\n")
            fh.write(f"mult,{mult}\n")
            for sym, x, y, z in atoms:
                fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
            fh.write(f"e_tot,{mf.e_tot:.14e}\n")
        print(f"{case}: E = {mf.e_tot:.10f}")


if __name__ == "__main__":
    main()
