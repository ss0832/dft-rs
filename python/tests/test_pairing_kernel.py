# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Off-diagonal pairing components share values, units and diagnostics."""
import json
import os
import subprocess

import numpy as np
import pytest
from dft_rs.units import HARTREE_IN_EV as Hartree
import dft_rs


INPUTS = dict(band_energies=[-0.016, 0.007], chemical_potential=0.002,
              electronic_temperature=1500.0, phonon_energies=[0.025, 0.018],
              pair_couplings_squared=[[0.0001, 0.0002, 0.0002, 0.00015],
                                      [0.00003, 0.00005, 0.00005, 0.00002]],
              screened_pair_coulomb=[[0.01, 0.002], [0.002, 0.008]],
              band_energy_direction=[0.004, -0.003], response_order=4)


def test_pairing_kernel_native_cli_ase_components_and_units():
    direct = dft_rs.scdft_pairing_kernel(**INPUTS)
    generic = dft_rs.run_feature("scdft-pairing-kernel", **INPUTS)
    atomic = dft_rs.DftRs.get_scdft_pairing_kernel(units="atomic", **INPUTS)
    converted_inputs = dict(INPUTS)
    for name in ("band_energies", "chemical_potential", "phonon_energies",
                 "screened_pair_coulomb", "band_energy_direction"):
        converted_inputs[name] = np.asarray(INPUTS[name])*Hartree
    converted_inputs["pair_couplings_squared"] = np.asarray(INPUTS["pair_couplings_squared"])*Hartree**2
    ase = dft_rs.DftRs.get_scdft_pairing_kernel(**converted_inputs)
    argv = [os.environ["DFT_RS_CLI"], "scdft-pairing-kernel", "--format", "json"]
    for name, value in INPUTS.items():
        argv += ["--"+name.replace("_", "-"), json.dumps(value)]
    process = subprocess.run(argv, text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    cli = json.loads(process.stdout)
    assert direct["converged"] and direct["response_order"] == 4
    for report in (generic, atomic, cli):
        assert report["model"] == direct["model"]
        assert report["units"] == direct["units"]
        for name in ("pairing_kernel", "pairing_kernel_coefficients",
                     "phonon_pairing_kernel", "coulomb_pairing_kernel",
                     "quadrature_absolute_errors", "quadrature_relative_errors", "pairing_asymmetry"):
            np.testing.assert_allclose(report[name], direct[name], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(direct["pairing_kernel"],
                               np.asarray(direct["phonon_pairing_kernel"])
                               +np.asarray(direct["coulomb_pairing_kernel"]), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(direct["coulomb_pairing_kernel"], INPUTS["screened_pair_coulomb"])
    for name in ("pairing_kernel", "pairing_kernel_coefficients", "phonon_pairing_kernel",
                 "coulomb_pairing_kernel", "quadrature_absolute_errors", "pairing_asymmetry"):
        np.testing.assert_allclose(ase[name], np.asarray(direct[name])*Hartree, rtol=1e-10, atol=1e-12)
    assert max(direct["quadrature_relative_errors"]) < 1e-10
    assert max(direct["pairing_asymmetry"]) < 1e-12
    assert not any(unit == "unspecified" for unit in direct["units"].values())
    print("pairing raw quadrature absolute", direct["quadrature_absolute_errors"],
          "relative", direct["quadrature_relative_errors"], "asymmetry", direct["pairing_asymmetry"])


def test_pairing_kernel_fermi_surface_and_resonance_have_no_denominator_floor():
    controls = dict(INPUTS, band_energies=[0.0, 0.025], chemical_potential=0.0,
                    band_energy_direction=[0.002, 0.003], electronic_temperature=2000.0)
    report = dft_rs.scdft_pairing_kernel(**controls)
    assert report["converged"]
    assert np.isfinite(report["pairing_kernel_coefficients"]).all()
    with pytest.raises(Exception, match="positive"):
        dft_rs.scdft_pairing_kernel(**dict(INPUTS, phonon_energies=[0.0, 0.018]))


def test_phonon_mass_parameter_paths_and_units_on_every_surface():
    controls = dict(INPUTS, band_weights=[0.6, 0.4], band_weight_direction=[0.02, -0.02],
                    chemical_potential_direction=0.0003, temperature_direction=60.0,
                    phonon_energy_direction=[0.001, -0.0004],
                    pair_couplings_squared_direction=[[0.00001]*4, [-0.000002]*4],
                    screened_pair_coulomb_direction=[[0.001, 0.0002], [0.0002, -0.0003]])
    direct = dft_rs.scdft_pairing_kernel(**controls)
    atomic = dft_rs.DftRs.get_scdft_pairing_kernel(units="atomic", **controls)
    converted = dict(controls)
    for key in ("band_energies", "chemical_potential", "phonon_energies", "screened_pair_coulomb",
                "band_energy_direction", "chemical_potential_direction", "phonon_energy_direction",
                "screened_pair_coulomb_direction"):
        converted[key] = np.asarray(controls[key])*Hartree
    for key in ("pair_couplings_squared", "pair_couplings_squared_direction"):
        converted[key] = np.asarray(controls[key])*Hartree**2
    adapted = dft_rs.DftRs.get_scdft_pairing_kernel(**converted)
    argv = [os.environ["DFT_RS_CLI"], "scdft-pairing-kernel", "--format", "json"]
    for name, value in controls.items():
        argv += ["--"+name.replace("_", "-"), json.dumps(value)]
    process = subprocess.run(argv, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    for report in (atomic, json.loads(process.stdout), adapted):
        assert report["converged"] and "particle-hole-symmetric" in report["mass_model"]
        for key in ("mass_renormalization", "mass_renormalization_coefficients",
                    "mass_quadrature_absolute_errors", "mass_quadrature_relative_errors"):
            assert report["units"][key] == "dimensionless"
            np.testing.assert_allclose(report[key], direct[key], rtol=1e-9, atol=1e-12)
    assert np.asarray(direct["mass_renormalization"]).min() > 0
    assert np.max(direct["mass_quadrature_relative_errors"]) < 1e-10
    np.testing.assert_allclose(adapted["pairing_kernel_coefficients"],
                               np.asarray(direct["pairing_kernel_coefficients"])*Hartree, rtol=1e-9, atol=1e-12)
    print("phonon mass public raw", direct["mass_renormalization_coefficients"],
          direct["mass_quadrature_absolute_errors"], direct["mass_quadrature_relative_errors"])
    with pytest.raises(Exception, match="positive band weight"):
        dft_rs.scdft_pairing_kernel(**dict(controls, band_weights=[0.0, 1.0]))
