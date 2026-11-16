# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Mass response shares the multicomponent driver and explicit electron-mass units."""
import copy

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.ase_calculator import _to_ase_units
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_fixed_populations import cli_report


def test_electron_mass_labels_convert_energy_and_length_once():
    report = {
        "mass_response": {"gradient": [2.], "hessian": [3.],
                          "response_number_defects": [[4.]],
                          "position_response_residuals": [5.]},
        "units": {"mass_response.gradient": "hartree/m_e (m_e is the electron mass)",
                  "mass_response.hessian": "hartree/m_e^2",
                  "mass_response.response_number_defects": "1/m_e",
                  "mass_response.position_response_residuals": "bohr/m_e"},
    }
    original = copy.deepcopy(report)
    converted = _to_ase_units(report)
    assert report == original
    assert converted["mass_response"]["gradient"] == [2.*Hartree]
    assert converted["mass_response"]["hessian"] == [3.*Hartree]
    assert converted["mass_response"]["response_number_defects"] == [[4.]]
    assert converted["mass_response"]["position_response_residuals"] == [5.*Bohr]
    assert converted["units"]["mass_response.gradient"].startswith("eV/m_e ")
    assert converted["units"]["mass_response.hessian"] == "eV/m_e^2"


def test_complete_mass_report_native_ase_and_canonical_cli(tmp_path):
    positions = [[.13, -.21, .37], [1.61, .32, -.18], [-.47, 1.43, .61]]
    options = dict(xc="hf", basis="sto-3g", charge=1, dispersion="none",
                   quantum_atoms=[0, 1], quantum_masses=[1836.15267343, 3672.30534686],
                   quantum_exponents=[4., 8.], quantum_lmax=1,
                   interspecies_correlation="none", quantum_position_constraint=True,
                   quantum_mass_response=True, species_tolerance=1e-12,
                   conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500,
                   response_tolerance=1e-12, grid_weight_cutoff=0., cpu_threads=1, memory_mb=512)
    native = dft_rs.multicomponent([1, 1, 1], positions, **options)
    atoms = Atoms("H3", positions=np.asarray(positions)*Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
    cli = cli_report(tmp_path, "multicomponent", atoms, options)
    physical_options = dict(options, quantum_exponents=np.asarray(options["quantum_exponents"])/Bohr**2,
                            conv_tol=options["conv_tol"]*Hartree,
                            conv_tol_grad=options["conv_tol_grad"]*Hartree)
    physical = dft_rs.DftRs().get_multicomponent(atoms, **physical_options)
    expected = native["mass_response"]
    assert native["converged"] and expected["converged"]
    assert expected["hessian_shape"] == [2, 2]
    assert "fixed Gaussian bases" in expected["convention"]
    assert not native["center_of_mass_removed"]
    np.testing.assert_allclose(expected["gradient"], [-4.8822583635569264e-6, -1.2310670365356188e-6],
                               atol=2e-16, rtol=0.)
    np.testing.assert_allclose(expected["hessian"], [5.1896452296480283e-9, 2.5671336761369467e-14,
                                                   2.5671336761374194e-14, 6.5767920950982357e-10],
                               atol=1e-19, rtol=0.)
    for other in [atomic, cli]:
        assert other["converged"] and other["mass_response"]["converged"]
        np.testing.assert_allclose(other["mass_response"]["gradient"], expected["gradient"], atol=2e-16, rtol=0.)
        np.testing.assert_allclose(other["mass_response"]["hessian"], expected["hessian"], atol=1e-19, rtol=0.)
    for key in ["gradient", "hessian", "raw_asymmetry", "kinetic_energy_defects"]:
        np.testing.assert_allclose(np.asarray(physical["mass_response"][key])/Hartree,
                                   expected[key], atol=2e-16, rtol=0.)
    assert physical["quantum"][0]["mass"] == options["quantum_masses"][0]
    assert physical["units"]["mass_response.gradient"].startswith("eV/m_e ")
    assert physical["units"]["mass_response.hessian"] == "eV/m_e^2"
    assert max(expected["response_residuals"]) < 1e-12
    np.testing.assert_allclose(expected["reference_number_defects"], 0., atol=1e-12, rtol=0.)
    np.testing.assert_allclose(expected["response_number_defects"], 0., atol=1e-12, rtol=0.)
    disabled = dft_rs.multicomponent([1, 1, 1], positions, **dict(options, quantum_mass_response=False))
    assert disabled["mass_response"] is None
    assert disabled["energy"] == pytest.approx(native["energy"], abs=1e-12)
