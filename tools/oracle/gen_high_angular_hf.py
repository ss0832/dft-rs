# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Regenerate C1 g/h/i HF term oracles with independently integrated cores.

The optional scalar ECP is always integrated by direct radial/angular
quadrature, including g and h. No analytic ECP recurrence is used. This is
an explicit development oracle; it is not part of production calculations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
from pyscf import __version__, gto, lib, scf

from gen_high_angular_ecp import potentials, quadrature
from gen_integrals import parse_g94


ROOT = Path(__file__).resolve().parents[2]


class QuadratureRHF(scf.hf.RHF):
    """The usual HF solver with an independently integrated fixed core."""

    def get_hcore(self, mol=None):
        return self._quadrature_core


class QuadratureUHF(scf.uhf.UHF):
    """The spin-resolved HF solver with the same independent core."""

    def get_hcore(self, mol=None):
        return self._quadrature_core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tests/oracle_data/high_angular_hf_terms.json")
    parser.add_argument("--logs", type=Path, required=True,
                        help="Explicit external directory for temporary SCF logs")
    args = parser.parse_args()
    if args.logs.resolve().is_relative_to(ROOT):
        parser.error("temporary oracle logs must be outside the distributable source")
    args.logs.mkdir(parents=True, exist_ok=True)
    lib.num_threads(1)
    result = {
        "provenance": f"PySCF {__version__} RHF/UHF; scalar cores from independent direct quadrature",
        "units": {"energy": "hartree", "positions": "bohr"},
        "scf": {"conv_tol": 1e-14, "conv_tol_grad": 1e-12, "max_cycle": 500},
        "core_quadrature": {"radial_points": 240, "polar_points": 48,
                            "azimuthal_points": 96, "rmax_bohr": 10},
        "cases": [],
    }
    for label in ["g", "h", "i"]:
        basis_path = ROOT / f"tests/data/c1_{label}_model.g94"
        for ecp in [False, True]:
            centre = np.array([0.17, -0.11, 0.23])
            positions = np.array([centre, [1.77, 0.18, -0.53],
                                  [-0.31, 1.74, -0.45], [-1.20, -1.31, -0.61]])
            if ecp:
                positions[1:] = centre + 1.4 * (positions[1:] - centre)
            elements = ["P" if ecp else "N", "He", "H", "H"]
            for unrestricted in [False, True]:
                tag = f'{label}_{"ecp" if ecp else "ae"}_{"uhf" if unrestricted else "rhf"}'
                electrons = 3 if unrestricted else 2
                with (args.logs / f"{tag}.log").open("w", encoding="utf-8") as log:
                    molecule = gto.M(atom=list(zip(elements, positions.tolist())),
                                     basis=parse_g94(str(basis_path), set(elements)),
                                     ecp={"P": gto.basis.load_ecp("lanl2dz", "P")} if ecp else {},
                                     charge=(9 if ecp else 11) - electrons,
                                     spin=int(unrestricted), unit="Bohr", verbose=0)
                    molecule.stdout = log
                    core = molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")
                    if ecp:
                        core += quadrature(molecule, potentials(basis_path), 240, 48)
                    method = QuadratureUHF(molecule) if unrestricted else QuadratureRHF(molecule)
                    method._quadrature_core = core
                    method._chkfile.close()
                    method._chkfile = None
                    method.chkfile = None
                    method.stdout = log
                    method.verbose = 4
                    method.init_guess = "1e"
                    method.conv_tol = 1e-14
                    method.conv_tol_grad = 1e-12
                    method.max_cycle = 500
                    energy = method.kernel()
                    if not method.converged:
                        raise RuntimeError(f"{tag} independent reference is not converged")
                    density = method.make_rdm1()
                    spins = density if unrestricted else np.array([density/2, density/2])
                    total = spins.sum(axis=0)
                    coulomb, _ = method.get_jk(molecule, total)
                    _, exchange = method.get_jk(molecule, spins)
                    one = float(np.einsum("ij,ji", total, core))
                    hartree = float(0.5 * np.einsum("ij,ji", total, coulomb))
                    exact_exchange = float(-0.5 * np.einsum("sij,sji", spins, exchange))
                    nuclear = float(molecule.energy_nuc())
                    if abs(nuclear + one + hartree + exact_exchange - energy) >= 1e-11:
                        raise RuntimeError(f"{tag} independent energy terms do not sum")
                    result["cases"].append({
                        "case": tag, "shell": label, "scalar_ecp": ecp,
                        "unrestricted": unrestricted, "electrons": electrons,
                        "positions": positions.tolist(), "nao": molecule.nao,
                        "basis_sha256": hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                        "energy": float(energy), "nuclear_repulsion": nuclear,
                        "one_electron": one, "hartree": hartree,
                        "exchange": exact_exchange,
                    })
                    print(tag, "energy", float(energy), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
