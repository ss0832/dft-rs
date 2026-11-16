# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Isotopic LiH spectra assembled from four-component electronic response."""

from concurrent.futures import ThreadPoolExecutor

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import numpy as np
import pytest
from ase import Atoms

from test_dirac_london import lithium_hydride_basis
from test_fixed_populations import cli_report


PROTON_G = 5.5856946893


def spin_matrices(twice_spin):
    spin = twice_spin / 2
    m = np.arange(-spin, spin + 1)
    plus = np.diag(np.sqrt(spin * (spin + 1) - m[:-1] * (m[:-1] + 1)), -1)
    return np.diag(m), plus


def dense_hydrogen_lines(shieldings, coupling_hz, g_factors, reference, spectrometer):
    """Diagonalize the full product space independently of its Iz blocks."""
    li_z, li_plus = spin_matrices(3)
    h_z, h_plus = spin_matrices(1)
    carrier = spectrometer * np.asarray(g_factors) / PROTON_G * 1e6
    larmor = carrier * (1 - 1e-6 * np.asarray(shieldings))
    hamiltonian = -larmor[0] * np.kron(li_z, np.eye(2)) - larmor[1] * np.kron(np.eye(4), h_z)
    hamiltonian += coupling_hz * (
        np.kron(li_z, h_z)
        + 0.5 * (np.kron(li_plus, h_plus.T) + np.kron(li_plus.T, h_plus))
    )
    energies, vectors = np.linalg.eigh(hamiltonian)
    detect = vectors.T @ np.kron(np.eye(4), h_plus) @ vectors
    # The squared raising norm of I=1/2 over the full product space is 4.
    return sorted(
        (reference + (energies[j] - energies[i] - carrier[1]) / (carrier[1] * 1e-6), w / 4)
        for (i, j), w in np.ndenumerate(detect**2)
        if energies[j] > energies[i] and w > 4e-10
    )


@pytest.mark.parametrize("spin_model", ["first-order", "exact"])
def test_lih_four_component_nmr_spectrum_routes(tmp_path, spin_model):
    numbers = [3, 1]
    positions = np.array([[0.13, -0.22, 0.14], [2.91, 0.48, -0.82]])
    atoms = Atoms(numbers, positions=positions * Bohr)
    electronic = dict(relativistic_method="dirac", xc="hf",
                      basis=str(lithium_hydride_basis(tmp_path)), cpu_threads=1,
                      grid_level=0, grid_weight_cutoff=0.0,
                      nuclear_rms_radii=[4.62e-5, 1.59e-5],
                      nuclear_magnetic_rms_radii=[4.62e-5, 1.59e-5],
                      gauge="giao", response_tolerance=1e-10,
                      conv_tol=1e-13, conv_tol_grad=1e-10)
    factors = [2.170951, PROTON_G]
    reference = 30.0
    spectrometer = 400.0
    controls = dict(electronic, nuclear_g=factors, nuclear_spins=[3, 1],
                    nmr_couplings=True, nmr_spin_model=spin_model,
                    nmr_reference_elements=[1], nmr_reference_shieldings=[reference],
                    nmr_spectrometer_mhz=spectrometer, nmr_points=101)
    converted = dict(controls, **{
        name: np.asarray(controls[name]) * Bohr
        for name in ("nuclear_rms_radii", "nuclear_magnetic_rms_radii")
    })
    converted.update(conv_tol=controls["conv_tol"] * Hartree,
                     conv_tol_grad=controls["conv_tol_grad"] * Hartree)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [
            pool.submit(dft_rs.nmr_spectrum, numbers, positions, **controls),
            pool.submit(dft_rs.DftRs(units="atomic").get_nmr_spectrum, atoms, **controls),
            pool.submit(dft_rs.DftRs().get_nmr_spectrum, atoms, **converted),
            pool.submit(cli_report, tmp_path, "nmr-spectrum", atoms, controls, spinor=True),
        ]
        native, atomic, physical, cli = [task.result() for task in tasks]
    for value in (native, atomic, physical, cli):
        assert value["converged"] and value["response_converged"]
        assert value["coupling_pairs"] == [[0, 1]]
        assert value["coupling_response_residual"] < 1e-10
        assert "fermi_contact_tensors" not in value
        for name in ("isotropic_shieldings", "coupling_constants_hz", "spectrum_axis",
                     "spectrum_intensity", "spectrum_line_positions", "spectrum_line_weights"):
            np.testing.assert_allclose(value[name], native[name], rtol=0.0, atol=1e-8)
    couplings = dft_rs.spin_spin_couplings(
        numbers, positions, nuclear_g=factors,
        **{key: value for key, value in electronic.items() if key != "gauge"},
    )
    np.testing.assert_allclose(native["reduced_couplings"],
                               couplings["reduced_couplings"], rtol=0., atol=1e-12)
    np.testing.assert_allclose(native["coupling_constants_hz"],
                               couplings["coupling_constants_hz"], rtol=0., atol=1e-8)
    actual = sorted(zip(native["spectrum_line_positions"][0],
                        native["spectrum_line_weights"][0], strict=True))
    j_hz, = native["coupling_constants_hz"]
    if spin_model == "exact":
        assert native["spin_hilbert_dimension"] == 8
        assert native["spin_hamiltonian_residual"] < 1e-6
        expected = dense_hydrogen_lines(native["isotropic_shieldings"], j_hz,
                                        factors, reference, spectrometer)
        np.testing.assert_allclose([p for p, w in actual if w > 1e-10],
                                   [p for p, _ in expected], rtol=0., atol=2e-8)
        np.testing.assert_allclose([w for _, w in actual if w > 1e-10],
                                   [w for _, w in expected], rtol=0., atol=1e-12)
    else:
        shift = reference - native["isotropic_shieldings"][1]
        expected = [(shift + m * j_hz / spectrometer, 0.25) for m in (-1.5, -0.5, 0.5, 1.5)]
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-10)
    assert abs(sum(w for _, w in actual) - 1) < 1e-12
    print(f"4c LiH {spin_model} J={j_hz:.15g} Hz, ppm/weights={actual}; "
          f"coupled residual={native['coupling_response_residual']:.6e}")
