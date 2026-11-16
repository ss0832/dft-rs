# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Reference data of the official Skala implementation for the oracle test.

Usage::

    python tools/oracle/skala_reference.py OUTPUT.oracle [--functional skala-1.1]

Needs the official ``skala`` package with ``torch`` and ``pyscf``. The script
converges a PBE density of the water cation (a doublet, so that the two spin
channels differ), evaluates the features and the exchange-correlation energy
density of the functional on the grid of the official PySCF integration, and
writes them in the tensor container of the model files:

    points [n, 3], centers [atoms, 3], atom_of [n], weights [n],
    atomic_weights [n], rho [2, n], gradient [2, n, 3], tau [2, n],
    energy_density [n], energy [1]

and, from a self-consistent calculation of the neutral molecule with the
functional, the grid of that calculation and its results:

    scf_points [m, 3], scf_atom_of [m], scf_weights [m],
    scf_atomic_weights [m], scf_energy [1], scf_exchange_correlation [1],
    scf_dipole [3], scf_gradient [atoms, 3]

The test ``official_checkpoint_reproduces_the_reference_energy_density`` of
``tests/p100_skala.rs`` reads the file named by ``DFT_RS_SKALA_ORACLE``
together with the converted model named by ``DFT_RS_SKALA_MODEL``. Neither
file is part of the package.
"""
import argparse
import struct

import numpy as np

MAGIC = b"DFTRS-SKALA\0"
VERSION = 1


def write(path, tensors):
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", VERSION, len(tensors)))
        for name in sorted(tensors):
            array = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float64))
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<I", array.ndim))
            stream.write(struct.pack(f"<{array.ndim}Q", *array.shape))
            stream.write(array.astype("<f8").tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="oracle file to write")
    parser.add_argument("--functional", default="skala-1.1")
    parser.add_argument("--basis", default="def2-svp")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--scf-grid-level", type=int, default=3)
    arguments = parser.parse_args()

    import torch
    from pyscf import dft, gto
    from skala.functional import load_functional
    from skala.pyscf import SkalaKS
    from skala.pyscf.features import generate_features

    mol = gto.M(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        unit="Bohr",
        basis=arguments.basis,
        charge=1,
        spin=1,
        verbose=0,
    )
    functional = load_functional(arguments.functional)
    ks = SkalaKS(mol, xc=functional, with_dftd3=False)
    ks.grids.level = arguments.grid_level
    ks.grids.build()
    reference = dft.UKS(mol, xc="pbe")
    reference.kernel()
    density_matrix = torch.tensor(np.asarray(reference.make_rdm1()))
    features = generate_features(mol, density_matrix, ks.grids, set(functional.features))
    energy_density = functional.get_exc_density(features).detach().double().numpy()
    energy = float(functional.get_exc(features).detach())
    sizes = features["atomic_grid_sizes"].numpy()
    tensors = dict(
        points=features["grid_coords"].numpy(),
        centers=features["coarse_0_atomic_coords"].numpy(),
        atom_of=np.repeat(np.arange(len(sizes)), sizes).astype(np.float64),
        weights=features["grid_weights"].numpy(),
        atomic_weights=features["atomic_grid_weights"].numpy(),
        rho=features["density"].numpy(),
        gradient=np.transpose(features["grad"].numpy(), (0, 2, 1)),
        tau=features["kin"].numpy(),
        energy_density=energy_density,
        energy=[energy],
    )
    # Self-consistent reference of the neutral molecule; its grid is written
    # point by point so that the test solves the same discretized equations.
    neutral = gto.M(
        atom="O 0 0 0; H 0 1.43 1.11; H 0 -1.43 1.11",
        unit="Bohr",
        basis=arguments.basis,
        verbose=0,
    )
    scf = SkalaKS(neutral, xc=functional, with_dftd3=False)
    scf.grids.level = arguments.scf_grid_level
    # The reference evaluates the functional in single precision, so its
    # energy fluctuates by a few 1e-9 Hartree between iterations.
    scf.conv_tol = 1e-8
    scf_energy = float(scf.kernel())
    assert scf.converged
    scf_density = np.asarray(scf.make_rdm1())
    if scf_density.ndim == 2:
        scf_density = np.stack([0.5 * scf_density, 0.5 * scf_density])
    scf_features = generate_features(
        neutral, torch.tensor(scf_density), scf.grids, set(functional.features)
    )
    scf_sizes = scf_features["atomic_grid_sizes"].numpy()
    tensors.update(
        scf_points=scf_features["grid_coords"].numpy(),
        scf_atom_of=np.repeat(np.arange(len(scf_sizes)), scf_sizes).astype(np.float64),
        scf_weights=scf_features["grid_weights"].numpy(),
        scf_atomic_weights=scf_features["atomic_grid_weights"].numpy(),
        scf_energy=[scf_energy],
        scf_exchange_correlation=[float(functional.get_exc(scf_features).detach())],
        scf_dipole=np.asarray(scf.dip_moment(unit="AU", verbose=0), dtype=np.float64),
        scf_gradient=np.asarray(scf.nuc_grad_method().kernel(), dtype=np.float64),
    )
    points = tensors["points"].shape[0]
    assert tensors["rho"].shape == (2, points) and tensors["gradient"].shape == (2, points, 3)
    write(arguments.output, tensors)
    print(
        f"wrote {arguments.output}: {points} grid points, {len(sizes)} atoms, "
        f"exchange-correlation energy {energy:.10f} Hartree, "
        f"electrons {float((tensors['rho'].sum(axis=0) * tensors['weights']).sum()):.8f}; "
        f"self-consistent energy {scf_energy:.10f} Hartree on {len(tensors['scf_weights'])} points"
    )


if __name__ == "__main__":
    main()
