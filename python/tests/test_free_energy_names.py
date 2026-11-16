# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Explicit thermodynamic potentials agree on the common public surfaces."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


def test_mermin_name_canonical_ensemble_and_compatibility(tmp_path):
    numbers, positions = [1, 1], [[0, 0, 0], [0, 0, 3.0]]
    options = dict(xc="svwn5", basis="6-31g", grid_level=4,
                   electronic_temperature=6000.0, conv_tol=1e-13,
                   conv_tol_dp=1e-11, max_iter=400)
    direct = dft_rs.mermin_free_energy(numbers, positions, **options)
    compatibility = dft_rs.free_energy(numbers, positions, **options)
    generic = dft_rs.run_feature("mermin-free-energy", numbers, positions, **options)
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    atomic = dft_rs.DftRs(units="atomic", **options).get_mermin_free_energy(atoms)
    converted_options = dict(options, conv_tol=options["conv_tol"]*Hartree)
    converted = dft_rs.DftRs(**converted_options).get_mermin_free_energy(atoms)
    path = tmp_path/"mermin.xyz"
    path.write_text(f"2\nMermin reference\nH 0 0 0\nH 0 0 {3.0*Bohr:.16f}\n")
    command = [os.environ["DFT_RS_CLI"], "mermin-free-energy", str(path), "--format", "json"]
    for name, value in options.items():
        command.extend(["--"+name.replace("_", "-"), str(value)])
    cli = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "[scf]" in cli.stderr
    for result in (direct, generic, atomic, compatibility, json.loads(cli.stdout)):
        assert result["free_energy_kind"].startswith("mermin")
        assert "gibbs_free_energy" not in result and result["converged"]
        assert result["mermin_free_energy"] == result["free_energy"]
        np.testing.assert_allclose(result["mermin_free_energy"], direct["mermin_free_energy"], atol=1e-10)
        np.testing.assert_allclose(result["internal_energy"]-result["mermin_free_energy"],
                                   result["kt"]*result["entropy"], atol=1e-14)
    np.testing.assert_allclose(converted["mermin_free_energy"], direct["mermin_free_energy"]*Hartree, atol=1e-9)
    assert converted["units"]["mermin_free_energy"] == "eV"


def test_water_gibbs_name_rrho_and_pressure_convention(tmp_path):
    numbers = [8, 1, 1]
    positions = [[0, 0, 0.1173], [0, 1.4332, -0.9314], [0, -1.4332, -0.9314]]
    controls = dict(xc="hf", basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-12,
                    temperature=298.15, pressure=101325.0, symmetry_number=2)
    direct = dft_rs.gibbs_free_energy(numbers, positions, **controls)
    harmonic = dft_rs.frequencies_report(numbers, positions, **controls)
    atoms = Atoms("OH2", positions=np.asarray(positions)*Bohr)
    reference = {k:v for k,v in controls.items() if k not in ("temperature", "pressure", "symmetry_number")}
    thermo_controls = {k:controls[k] for k in ("temperature", "pressure", "symmetry_number")}
    atomic = dft_rs.DftRs(units="atomic", **reference).get_gibbs_free_energy(atoms, **thermo_controls)
    reference["conv_tol"] *= Hartree
    converted = dft_rs.DftRs(**reference).get_gibbs_free_energy(atoms, **thermo_controls)
    path = tmp_path/"gibbs.xyz"
    path.write_text("3\nGibbs reference\n"+"".join(
        symbol+" "+" ".join(f"{x*Bohr:.16f}" for x in row)+"\n"
        for symbol,row in zip(("O","H","H"), positions)))
    command = [os.environ["DFT_RS_CLI"], "gibbs-free-energy", str(path), "--format", "json"]
    for name,value in controls.items():
        command.extend(["--"+name.replace("_", "-"), str(value)])
    cli = subprocess.run(command, check=True, capture_output=True, text=True)
    for result in (direct, harmonic, atomic, json.loads(cli.stdout)):
        assert result["free_energy_kind"].startswith("gibbs") and result["converged"]
        assert "mermin_free_energy" not in result and "free_energy" not in result
        np.testing.assert_allclose(result["gibbs_free_energy"], direct["gibbs_free_energy"], atol=1e-9)
        thermo = result["thermochemistry"]
        assert result["gibbs_free_energy"] == thermo["gibbs_free_energy"]
        np.testing.assert_allclose(thermo["gibbs_free_energy"],
            thermo["enthalpy"]-thermo["temperature_kelvin"]*thermo["entropy"], atol=1e-14)
        # Independent ideal-gas pV contribution per molecule, using SI constants.
        kb = 1.380649e-23/4.3597447222071e-18
        np.testing.assert_allclose(thermo["enthalpy"]-thermo["internal_energy"],
                                   kb*controls["temperature"], rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(converted["gibbs_free_energy"], direct["gibbs_free_energy"]*Hartree, atol=1e-8)


@pytest.mark.parametrize("electronic_temperature", [0.0, 8000.0])
def test_mermin_can_also_return_gibbs_on_the_same_reference(tmp_path, electronic_temperature):
    numbers, positions = [1, 1], [[0, 0, 0], [0, 0, 2.0]]
    options = dict(xc="svwn5", basis="6-31g", grid_level=4, conv_tol=1e-13,
                   conv_tol_dp=1e-11, electronic_temperature=electronic_temperature)
    controls = dict(pressure=101325.0, symmetry_number=2)
    direct = dft_rs.mermin_free_energy(numbers, positions, gibbs=True, **controls, **options)
    separate = dft_rs.gibbs_free_energy(numbers, positions, **controls, **options)
    atoms = Atoms("H2", positions=np.asarray(positions)*Bohr)
    adapted = dft_rs.DftRs(units="atomic", **options).get_mermin_free_energy(atoms, gibbs=True, **controls)
    path = tmp_path/"combined.xyz"
    path.write_text(f"2\nMermin and Gibbs reference\nH 0 0 0\nH 0 0 {2*Bohr:.16f}\n")
    argv = [os.environ["DFT_RS_CLI"], "mermin-free-energy", str(path), "--gibbs", "--format", "json"]
    for name, value in dict(options, **controls).items():
        argv.extend(["--"+name.replace("_", "-"), str(value)])
    cli = subprocess.run(argv, check=True, capture_output=True, text=True)
    assert cli.stderr.count("[scf] converged") == 1, "the Gibbs option must reuse one electronic reference"
    for result in (direct, adapted, json.loads(cli.stdout)):
        assert result["converged"]
        np.testing.assert_allclose(result["gibbs_free_energy"], separate["gibbs_free_energy"], atol=1e-10)
        thermo = result["thermochemistry"]
        temperature = electronic_temperature or 298.15
        assert thermo["temperature_kelvin"] == temperature
        np.testing.assert_allclose(thermo["gibbs_free_energy"],
            thermo["enthalpy"]-temperature*thermo["entropy"], atol=1e-14)
        # A_e already includes the electronic entropy; add only the RRHO part.
        kb = 3.166811563455608e-6
        rrho_entropy = thermo["entropy"]-kb*result["entropy"]
        expected = result["mermin_free_energy"]+thermo["enthalpy_correction"]-temperature*rrho_entropy
        np.testing.assert_allclose(result["gibbs_free_energy"], expected, atol=1e-13)
        if electronic_temperature:
            assert max(result["ensemble_response_residuals"]) < 1e-10
            assert np.max(result["ensemble_number_defects"]) < 1e-10
    with pytest.raises(Exception, match="equal electronic and nuclear"):
        dft_rs.mermin_free_energy(numbers, positions, gibbs=True, temperature=300.0,
                                  **dict(options, electronic_temperature=8000.0))
