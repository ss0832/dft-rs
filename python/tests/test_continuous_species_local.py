# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Actual native/ASE/CLI checks for local electronic and steric residuals."""
import json
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

NUMBERS = [8, 1, 1]
POSITIONS = [[0.13, -0.21, 0.17], [1.34, 0.82, -0.23], [-0.74, 1.13, 0.91]]
METHOD = dict(xc="pbe", basis="sto-3g", dispersion="none", guess="core", grid_level=1,
              grid_weight_cutoff=0, conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11)
CONTROLS = dict(fixed_electron_count=10, charge_directions=[[0, 0, 0], [0, 0, 0], [0.17, -0.31, 0.14]],
                charge_coordinates=[0.23, 0.07, -0.19], charge_bounds=[[-0.5, 0.5], [0, 0.5], [-0.5, 0.5]],
                conserve_total_charge=False, charge_origin=[0.11, 0.07, -0.09],
                local_potential_atoms=[0], local_potential_exponents=[0.63],
                local_potential_reference_amplitudes=[0.12], local_potential_directions=[[1, 0, 0.03]],
                steric_reference_strengths=[0.18], steric_directions=[[0, 1, 0.02]],
                steric_environment_exponents=[0.47, 0.81, 0.59],
                steric_environment_hardness=[0.31, 0.73, 0.23], steric_pair_mask=[[1, 1, 0]],
                local_frame_anchors=[0, 1, 2], local_frame_offsets=[[0.27, -0.19, 0.33]],
                charge_nuclear_gradient=True, charge_dynamic_response=True, imaginary=True,
                frequency=0.1, charge_probe_points=[[2.31, -1.47, 0.86], [-1.83, 0.67, 2.14]],
                fragment_internal_energy=-1.234)


@pytest.mark.parametrize("open_shell", [False, True])
def test_local_native_ase_cli_units_and_classical_constant(tmp_path, open_shell):
    controls = dict(CONTROLS)
    if open_shell:
        controls.update(fixed_electron_count=9, fixed_spin_electron_counts=[5, 4])
    native = dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD, **controls)
    assert native["fragment_internal_energy"] == -1.234
    assert native["local_potential_amplitudes"] == pytest.approx([0.3443])
    assert native["steric_strengths"] == pytest.approx([0.2462])
    unshifted = dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD,
                                        **dict(controls, fragment_internal_energy=0.0))
    huge = dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD,
                                   **dict(controls, fragment_internal_energy=1e80))
    for value in (native, huge):
        assert value["energy_without_internal_reference"] == unshifted["energy"]
        assert value["classical_energy_without_internal_reference"] == unshifted["classical_energy"]
        assert value["scf_energy_convention"] == unshifted["scf_energy_convention"]
        assert "excluded from trial energies" in value["scf_energy_convention"]
        for key in ("electronic_energy", "dipole", "polarizability", "forces",
                    "parameter_energy_gradient", "parameter_energy_hessian"):
            np.testing.assert_array_equal(value[key], unshifted[key])
    assert native["energy"] - unshifted["energy"] == pytest.approx(-1.234, abs=1e-10)
    assert native["classical_energy"] - unshifted["classical_energy"] == pytest.approx(-1.234, abs=1e-12)
    for key in ("electronic_energy", "dipole", "polarizability", "forces"):
        np.testing.assert_allclose(native[key], unshifted[key], atol=1e-9, rtol=1e-9)
    assert max(native["dynamic_response_residuals"]) <= 1e-10
    assert np.max(np.abs(native["dynamic_polarizability_imag"])) < 1e-12
    atoms = Atoms(numbers=NUMBERS, positions=np.array(POSITIONS)*Bohr)
    external = dict(controls)
    for name in ("charge_origin", "local_frame_offsets", "charge_probe_points"):
        external[name] = (np.asarray(controls[name])*Bohr).tolist()
    for name in ("local_potential_reference_amplitudes", "local_potential_directions",
                 "steric_reference_strengths", "steric_directions"):
        external[name] = (np.asarray(controls[name])*Hartree).tolist()
    external["fragment_internal_energy"] *= Hartree
    external["frequency"] *= Hartree
    for name in ("local_potential_exponents", "steric_environment_exponents"):
        external[name] = (np.asarray(controls[name])/Bohr**2).tolist()
    external["steric_environment_hardness"] = (np.asarray(controls["steric_environment_hardness"])/Bohr**3).tolist()
    calc = dft_rs.DftRs(**dict(METHOD, conv_tol=METHOD["conv_tol"]*Hartree,
                             conv_tol_grad=METHOD["conv_tol_grad"]*Hartree))
    atomic = calc.get_continuous_species(atoms, units="atomic", **controls)
    adapted = calc.get_continuous_species(atoms, **external)
    groups = {
        1.0: ["physical_total_charge"],
        Hartree: ["energy", "classical_energy", "electronic_energy", "parameter_energy_gradient",
                  "parameter_energy_hessian", "fragment_internal_energy",
                  "energy_without_internal_reference", "classical_energy_without_internal_reference"],
        Bohr: ["dipole", "parameter_dipole_jacobian"],
        Bohr**3: ["polarizability", "dynamic_polarizability_real", "dynamic_polarizability_imag"],
        Hartree/Bohr: ["gradient", "forces"],
        Bohr**-3: ["probe_electron_density"],
    }
    for factor, keys in groups.items():
        for key in keys:
            np.testing.assert_allclose(atomic[key], native[key], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(adapted[key], np.asarray(native[key])*factor, atol=1e-7, rtol=1e-8)
    np.testing.assert_allclose(adapted["probe_electrostatic_potential"],
                               np.asarray(native["probe_electrostatic_potential"])*Hartree, atol=1e-8, rtol=1e-8)
    path = tmp_path / "local_model.xyz"
    path.write_text("3\nClamped asymmetric molecule\n" + "".join(
        f"{symbol} {x*Bohr:.17g} {y*Bohr:.17g} {z*Bohr:.17g}\n"
        for symbol, (x, y, z) in zip(["O", "H", "H"], POSITIONS)), encoding="utf-8")
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for actual public route acceptance"
    arguments = [executable, "continuous-species", str(path), "--format", "json"]
    for key, value in dict(METHOD, **controls).items():
        arguments.extend(["--" + key.replace("_", "-"), json.dumps(value)
                          if isinstance(value, (list, bool)) else str(value)])
    process = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    energies = re.findall(r"\[scf\] converged.*\(E = ([+\-\d.eE]+)\)", process.stderr)
    # SCF logs describe the Hamiltonian without its state-common scalar.
    assert energies and float(energies[-1]) == pytest.approx(
        report["energy_without_internal_reference"], abs=1e-10)
    assert report["scf_energy_convention"] == native["scf_energy_convention"]
    assert report["energy"] == pytest.approx(
        report["energy_without_internal_reference"] + report["fragment_internal_energy"], abs=1e-12)
    for keys in groups.values():
        for key in keys:
            np.testing.assert_allclose(report[key], native[key], atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("override", [
    {"local_potential_exponents": [-0.1]},
    {"local_potential_directions": [[1, 0]]},
    {"steric_pair_mask": [[1, 0.5, 0]]},
    {"steric_environment_hardness": [-0.1, 0.2, 0.3]},
    {"steric_reference_strengths": [-1]},
    {"local_frame_anchors": [0, 0, 2]},
    {"charge_probe_points": [POSITIONS[0]]},
])
def test_local_model_rejections(override):
    with pytest.raises((ValueError, RuntimeError)):
        dft_rs.continuous_species(NUMBERS, POSITIONS, **METHOD, **dict(CONTROLS, **override))
