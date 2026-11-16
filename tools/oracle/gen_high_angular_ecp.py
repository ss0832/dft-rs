# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Regenerate independent complete g/h/i scalar-core quadrature matrices.

Run explicitly with the existing development oracle environment. Production
does not depend on this script or its numerical-oracle packages. Only Gaussian
AO values are obtained from PySCF; the local and semilocal core integrals use
direct radial/angular quadrature of their definition, not its ECP recurrence.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
from numpy.polynomial.legendre import leggauss
from pyscf import __version__, gto, lib

from gen_integrals import dftrs_ao_permutation, parse_g94


ROOT = Path(__file__).resolve().parents[2]
LEVELS = [(160, 24), (200, 36), (240, 48)]


def potentials(path):
    """Read the fixture's local d and projected s-d/p-d radial terms."""
    lines = path.read_text(encoding="utf-8").splitlines()
    channels = []
    for name in ["d potential", "s-d potential", "p-d potential"]:
        start = lines.index(name)
        count = int(lines[start + 1])
        channels.append([tuple(map(float, line.split()))
                         for line in lines[start + 2:start + 2 + count]])
    return channels


def quadrature(molecule, channels, radial_count, polar_count):
    """Integrate U_local plus sum_lm |lm> delta_U_l <lm|, in Hartree."""
    z, wz = leggauss(polar_count)
    phi = np.arange(2 * polar_count) * np.pi / polar_count
    directions = np.stack(np.broadcast_arrays(
        np.sqrt(1 - z*z)[:, None] * np.cos(phi),
        np.sqrt(1 - z*z)[:, None] * np.sin(phi),
        z[:, None]), axis=-1).reshape(-1, 3)
    angular_weights = np.repeat(wz, 2 * polar_count) * np.pi / polar_count
    # Any orthonormal real basis spans the same l=1 projector.
    harmonics = [np.full((len(directions), 1), 1 / np.sqrt(4*np.pi)),
                 np.sqrt(3 / (4*np.pi)) * directions]
    nodes, weights = leggauss(radial_count)
    radius, radial_weights = 5 * (nodes + 1), 5 * weights
    answer = np.zeros((molecule.nao, molecule.nao))
    centre = molecule.atom_coords()[0]
    for r, weight in zip(radius, radial_weights):
        aos = molecule.eval_gto("GTOval_sph", centre + r * directions)
        radial = [sum(c * r**(n-2) * np.exp(-a*r*r) for n, a, c in terms)
                  for terms in channels]
        answer += weight*r*r * radial[0] * (aos.T @ (angular_weights[:, None] * aos))
        for l, harmonics_l in enumerate(harmonics):
            projection = aos.T @ (angular_weights[:, None] * harmonics_l)
            answer += weight*r*r * radial[l+1] * (projection @ projection.T)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tests/oracle_data/high_angular_ecp.json")
    args = parser.parse_args()
    lib.num_threads(1)
    centre = np.array([0.17, -0.11, 0.23])
    positions = np.array([centre, [1.77, 0.18, -0.53],
                          [-0.31, 1.74, -0.45], [-1.20, -1.31, -0.61]])
    positions[1:] = centre + 1.4 * (positions[1:] - centre)
    result = {
        "definition": "Direct local-core 3D quadrature and spherical-harmonic s/p projections",
        "provenance": f"PySCF {__version__} AO values; independent radial/angular ECP quadrature",
        "units": {"matrix": "hartree", "positions": "bohr"},
        "rmax_bohr": 10,
        "levels": LEVELS,
        "positions": positions.tolist(),
        "cases": [],
        "notes": "Full library spherical AO order. Invalid analytic ECP references are recorded, never adopted.",
    }
    for label in ["g", "h", "i"]:
        path = ROOT / f"tests/data/c1_{label}_model.g94"
        molecule = gto.M(atom=list(zip(["P", "He", "H", "H"], positions.tolist())),
                         basis=parse_g94(str(path), {"P", "He", "H"}),
                         ecp={"P": gto.basis.load_ecp("lanl2dz", "P")},
                         charge=7, spin=0, unit="Bohr", verbose=0)
        matrices = [quadrature(molecule, potentials(path), nr, na) for nr, na in LEVELS]
        defects = [float(np.max(abs(b-a))) for a, b in zip(matrices, matrices[1:])]
        reference = molecule.intor("ECPscalar")
        maximum = float(np.max(abs(reference)))
        valid_reference = np.isfinite(reference).all() and maximum < 1e6
        independent_defect = float(np.max(abs(matrices[-1]-reference))) if valid_reference else None
        permutation = dftrs_ao_permutation(molecule)
        result["cases"].append({
            "shell": label,
            "refinement_max_defects": defects,
            "analytic_oracle_defect": independent_defect,
            "analytic_oracle_maximum": maximum,
            "quadrature_maximum": float(np.max(abs(matrices[-1]))),
            "matrix": matrices[-1][np.ix_(permutation, permutation)].tolist(),
        })
        print(label, "refinement", defects, "independent analytic defect", independent_defect)
        if defects[-1] >= 2e-11 or (valid_reference and independent_defect >= 2e-11):
            raise RuntimeError(f"{label} quadrature did not pass the independent refinement gate")
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
