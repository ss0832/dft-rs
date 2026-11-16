# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# VV10 oracle: PySCF self-consistent BLYP+VV10 total energy (and the plain
# BLYP total on the same settings, so the NLC contribution is isolated).
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_nlc.py

import os
import pyscf
from pyscf import dft, gto

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

def main():
    charge, mult, atoms = MOLECULES["h2o"]
    basis = parse_g94(os.path.join(DATA_DIR, "def2-svp.g94"), {"O", "H"})
    geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    mol = gto.M(atom=geom, basis=basis, unit="Bohr", charge=charge, spin=0)

    mf0 = dft.RKS(mol)
    mf0.xc = "b88,lyp"
    mf0.grids.level = 9
    mf0.conv_tol = 1e-12
    e_blyp = mf0.kernel()
    assert mf0.converged

    mf = dft.RKS(mol)
    mf.xc = "b88,lyp"
    mf.nlc = "vv10"
    mf.grids.level = 9
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-12
    e_vv10 = mf.kernel()
    assert mf.converged

    with open(os.path.join(OUT_DIR, "h2o_def2svp_blyp_vv10_nlc.csv"), "w", newline="\n") as fh:
        fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} VV10 oracle (b=5.9, C=0.0093)\n")
        fh.write("# self-consistent BLYP+VV10 vs plain BLYP, level-9 XC grid, level-3 NLC grid\n")
        fh.write("basis,def2-svp\n")
        fh.write("functional,blyp\n")
        for sym, x, y, z in atoms:
            fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
        fh.write(f"e_blyp,{e_blyp:.14e}\n")
        fh.write(f"e_blyp_vv10,{e_vv10:.14e}\n")
        fh.write(f"e_nlc_sc,{e_vv10 - e_blyp:.14e}\n")
    print(f"BLYP {e_blyp:.10f}  BLYP+VV10 {e_vv10:.10f}  ΔNLC {e_vv10 - e_blyp:.10f}")

    # ωB97M-V: the same protocol with the functional's own VV10 parameters
    # (b = 6.0, C = 0.01; libxc supplies them via the NLC flag).
    mf0 = dft.RKS(mol)
    mf0.xc = "wb97m_v"
    mf0.nlc = 0
    mf0.grids.level = 9
    mf0.conv_tol = 1e-12
    e_sl = mf0.kernel()
    assert mf0.converged

    mf = dft.RKS(mol)
    mf.xc = "wb97m_v"
    mf.grids.level = 9
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-12
    e_tot = mf.kernel()
    assert mf.converged

    with open(os.path.join(OUT_DIR, "h2o_def2svp_wb97mv_vv10_nlc.csv"), "w", newline="\n") as fh:
        fh.write(f"# PROVENANCE: PySCF {pyscf.__version__} VV10 oracle (b=6.0, C=0.01)\n")
        fh.write("# self-consistent wB97M-V vs its semilocal part, level-9 XC grid, level-3 NLC grid\n")
        fh.write("basis,def2-svp\n")
        fh.write("functional,wb97mv\n")
        for sym, x, y, z in atoms:
            fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
        fh.write(f"e_semilocal,{e_sl:.14e}\n")
        fh.write(f"e_total,{e_tot:.14e}\n")
        fh.write(f"e_nlc_sc,{e_tot - e_sl:.14e}\n")
    print(f"wB97M-V sl {e_sl:.10f}  total {e_tot:.10f}  ΔNLC {e_tot - e_sl:.10f}")


if __name__ == "__main__":
    main()
