# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Oracle generator for complete-active-space SCF and multiconfiguration
# pair-density functional theory: frozen PySCF values (numerical oracle only,
# energies), on the STO-3G basis file of the package and PySCF grid level 9.
#
# Run (WSL): <environment with pyscf and its mcpdft module>/bin/python gen_mcpdft.py
import json
import os
import sys

import pyscf
from pyscf import gto, mcpdft, scf

from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

WATER = [("O", 0.0, 0.0, 0.23), ("H", 0.0, 1.4309, -0.8867), ("H", 0.12, -1.4809, -0.9067)]
LITHIUM_HYDRIDE = [("Li", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 3.015)]
HYDROGEN = [("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 1.4)]


def molecule(atoms):
    basis = parse_g94(os.path.join(DATA_DIR, "sto-3g.g94"), {a[0] for a in atoms})
    return gto.M(atom="; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms),
                 basis=basis, unit="Bohr", cart=False, verbose=0)


def reference(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def tight(mc):
    # The oracle optimizer converges reliably to this orbital gradient; the
    # PDFT energies follow the orbitals linearly, so compare at 1e-5 Hartree.
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = 200
    return mc


def state_terms(mc):
    """Independent nuclear, one-electron, Coulomb and unscaled on-top terms."""
    nuclear, one, coulomb, on_top, _ = mc.get_energy_decomposition(split_x_c=False)
    fraction, correlation_fraction = mc.otfnal._numint.hybrid_coeff(mc.otfnal.otxc)
    assert abs(fraction - correlation_fraction) < 1e-12
    return {"state_nuclear_repulsion": [float(nuclear)],
            "state_one_electron": [float(one)],
            "state_coulomb": [float(coulomb)],
            "state_on_top": [float(on_top) / (1 - fraction)]}


def main():
    # Energy-only oracles need no checkpoint files or implicit cleanup.
    scf.hf.MUTE_CHKFILE = True
    out = {"provenance": f"PySCF {pyscf.__version__} CASSCF and MC-PDFT oracle (values only), "
                         "grid level 9, package STO-3G basis, geometry in Bohr, orbital gradient 1e-6",
           "cases": []}
    mol = molecule(WATER)
    mf = reference(mol)
    for otxc in ["tPBE", "ftPBE", "tPBE0"]:
        name = otxc.lower()
        # Full valence: only O 1s in the core, so the solution is unique.
        mc = tight(mcpdft.CASSCF(mf, otxc, 6, 8, grids_level=9))
        mc.kernel()
        assert mc.converged
        out["cases"].append({"molecule": "water", "cas": [8, 6], "roots": 1,
                             "on_top_functional": name, "method": "state-specific",
                             "cas_energies": [float(mc.e_mcscf)], "pdft_energies": [float(mc.e_tot)],
                             "state_terms": state_terms(mc)})
    mol = molecule(LITHIUM_HYDRIDE)
    mf = reference(mol)
    for method, label in [(None, "state-specific"), ("cms", "cms-pdft"), ("lin", "l-pdft"), ("xms", "xms-pdft")]:
        base = mcpdft.CASSCF(mf, "tPBE", 2, 2, grids_level=9)
        mc = base.state_average([0.5, 0.5]) if method is None else base.multi_state([0.5, 0.5], method)
        mc.fix_spin_(ss=0)
        tight(mc)
        mc.kernel()
        assert mc.converged
        out["cases"].append({"molecule": "lithium hydride", "cas": [2, 2], "roots": 2,
                             "on_top_functional": "tpbe", "method": label,
                             "cas_energies": [float(e) for e in mc.e_mcscf],
                             "pdft_energies": sorted(float(e) for e in mc.e_states)})
    target = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    mol = molecule(HYDROGEN)
    mf = reference(mol)
    for otxc in ["tPBE", "ftPBE"]:
        mc = tight(mcpdft.CASSCF(mf, otxc, 2, 2, grids_level=9))
        mc.kernel()
        assert mc.converged
        out["cases"].append({"molecule": "hydrogen", "cas": [2, 2], "roots": 1,
                             "on_top_functional": otxc.lower(), "method": "state-specific",
                             "cas_energies": [float(mc.e_mcscf)], "pdft_energies": [float(mc.e_tot)],
                             "state_terms": state_terms(mc)})
    os.makedirs(target, exist_ok=True)
    with open(os.path.join(target, "mcpdft_oracle.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
