# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Exact scalar spin spectra share the existing molecular response providers."""
import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr


@pytest.mark.parametrize("xc,reference", [("hf", "rks"), ("svwn5", "rks"), ("pbe", "rks"), ("tpss", "rks"), ("pbe0", "rks"), ("b2plyp", "rks")])
def test_exact_equivalent_protons_share_native_ase_and_cli(tmp_path, xc, reference):
    from test_fixed_populations import cli_report

    positions = [[0.13, -0.21, -0.71], [0.13, -0.21, 0.69]]
    atoms = Atoms("H2", positions=np.asarray(positions) * Bohr)
    controls = dict(xc=xc, reference=reference, basis="sto-3g", dispersion="none",
                    grid_level=3, grid_weight_cutoff=0., conv_tol=1e-13,
                    conv_tol_grad=1e-11, conv_tol_dp=1e-11, max_iter=500,
                    nmr_reference_elements=[1], nmr_reference_shieldings=[30.],
                    nuclear_g=[5.5856946893, 5.5856946893], nuclear_spins=[1, 1],
                    nmr_couplings=True, nmr_spin_model="exact", nmr_points=101,
                    response_tolerance=1e-11)
    native = dft_rs.nmr_spectrum([1, 1], positions, **controls)
    adapted = DftRs(units="atomic").get_nmr_spectrum(atoms, **controls)
    physical = DftRs().get_nmr_spectrum(atoms, **controls)
    cli = cli_report(tmp_path, "nmr-spectrum", atoms, controls)
    for report in [native, adapted, physical, cli]:
        assert report["converged"] and report["response_converged"]
        assert report["spectrum_spin_model"] == "exact"
        assert report["spin_hilbert_dimension"] == 4
        assert report["units"]["spin_hamiltonian_residual"] == "Hz"
        assert report["spin_hamiltonian_residual"] < 1e-8
        assert max(abs(v) for v in report["spin_transition_sum_rule_defects"]) < 1e-12
        assert abs(sum(report["spectrum_line_weights"][0]) - 2.) < 1e-12
        shift = np.mean(report["chemical_shifts"])
        axis = np.asarray(report["spectrum_axis"][0])
        half = report["spectrum_broadening"] / 2.
        expected = 2. * half / (np.pi * ((axis - shift)**2 + half**2))
        np.testing.assert_allclose(report["spectrum_intensity"][0], expected, rtol=0., atol=2e-8)
    for other in [adapted, physical, cli]:
        for name in ["spectrum_axis", "spectrum_intensity", "spectrum_line_positions", "spectrum_line_weights"]:
            np.testing.assert_allclose(other[name], native[name], rtol=0., atol=1e-8)
    print("exact NMR public", xc, reference, "eigen residual", native["spin_hamiltonian_residual"],
          "sum rule", native["spin_transition_sum_rule_defects"], flush=True)


def test_unobserved_isotope_and_display_references_do_not_change_the_spin_hamiltonian():
    options = dict(xc="hf", basis="sto-3g", dispersion="none", conv_tol=1e-13,
                   conv_tol_grad=1e-11, conv_tol_dp=1e-11, max_iter=500,
                   nmr_couplings=True, nmr_spin_model="exact", nuclear_g=[2.170951, 5.5856946893],
                   nuclear_spins=[3, 1], nmr_points=51, response_tolerance=1e-11)
    positions = [[0., 0., 0.], [.11, -.17, 3.02]]
    hydrogen = dft_rs.nmr_spectrum([3, 1], positions, nmr_reference_elements=[1],
                                 nmr_reference_shieldings=[30.], **options)
    both = dft_rs.nmr_spectrum([3, 1], positions, nmr_reference_elements=[1, 3],
                              nmr_reference_shieldings=[30., 10000.], **options)
    shifted = dft_rs.nmr_spectrum([3, 1], positions, nmr_reference_elements=[1],
                                 nmr_reference_shieldings=[31.], **options)
    for result in [hydrogen, both, shifted]:
        assert result["converged"] and result["spin_hamiltonian_converged"]
    for key in ["spectrum_line_weights", "spectrum_line_positions", "spectrum_intensity"]:
        np.testing.assert_array_equal(hydrogen[key][0], both[key][0])
    np.testing.assert_array_equal(hydrogen["spectrum_line_weights"][0], shifted["spectrum_line_weights"][0])
    np.testing.assert_allclose(np.asarray(shifted["spectrum_line_positions"][0]) -
                               np.asarray(hydrogen["spectrum_line_positions"][0]), 1., atol=1e-12, rtol=0.)
