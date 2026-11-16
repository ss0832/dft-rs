# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""PT2 electric/magnetic transition residues on every public surface."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM


def request(unrestricted, xc):
    positions = [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                 [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
    if unrestricted:
        positions.append([1.41, 1.83, 1.59])
    controls = dict(xc=xc, basis="sto-3g", reference="uks" if unrestricted else "rks",
                    multiplicity=2 if unrestricted else 1, dispersion="none", guess="core",
                    grid_level=0, grid_weight_cutoff=0., conv_tol=1e-13,
                    conv_tol_dp=1e-13, conv_tol_grad=1e-13,
                    nstates=2, td_tolerance=1e-10, broadening_ev=0.2, points=101)
    if xc != "b2plyp":
        controls.update(pt2_same_spin=0.031, pt2_opposite_spin=0.047)
    return [1] * len(positions), positions, controls


DIAGNOSTICS = ["pt2_parent_excitation_energies", "pt2_excitation_energy_corrections",
               "pt2_transition_singles_correction_norms", "pt2_transition_doubles_norms",
               "pt2_transition_pole_residuals", "pt2_transition_normalization_defects",
               "pt2_transition_smallest_double_denominators"]


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("xc", ["hf", "tpss", "pbe0", "b2plyp"])
def test_correlated_transition_vectors_and_residuals_on_native_driver_and_ase(unrestricted, xc):
    numbers, positions, controls = request(unrestricted, xc)
    native = dft_rs.ecd_spectrum(numbers, positions, **controls)
    direct = dft_rs.run_feature("ecd-spectrum", numbers, positions, **controls)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions) * Bohr)
    atomic = DftRs(units="atomic").get_ecd_spectrum(atoms, **controls)
    physical = DftRs(units="ase").get_ecd_spectrum(atoms, **controls)
    assert native["converged"] and native["response_converged"]
    assert "property gradients" in native["model"]
    for key in DIAGNOSTICS + ["molar_absorptivity", "ecd_length", "ecd_velocity"]:
        for report in [direct, atomic]:
            np.testing.assert_allclose(report[key], native[key], rtol=2e-9, atol=2e-9)
    for row, adapted, converted in zip(native["states"], atomic["states"], physical["states"]):
        for key in ["transition_dipole", "transition_momentum", "transition_angular_momentum",
                    "energy", "oscillator_strength", "rotatory_strength", "rotatory_strength_velocity"]:
            np.testing.assert_allclose(adapted[key], row[key], rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(converted["transition_dipole"], np.asarray(row["transition_dipole"]) * Bohr,
                                   rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(converted["transition_momentum"], np.asarray(row["transition_momentum"]) / Bohr,
                                   rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(converted["transition_angular_momentum"], row["transition_angular_momentum"],
                                   rtol=2e-9, atol=2e-9)
        assert converted["energy"] == pytest.approx(row["energy"] * Hartree, rel=2e-9, abs=2e-9)
    np.testing.assert_allclose(physical["pt2_excitation_energy_corrections"],
                               np.asarray(native["pt2_excitation_energy_corrections"]) * Hartree,
                               rtol=2e-9, atol=2e-9)
    assert max(native["pt2_transition_pole_residuals"]) < 1e-10
    assert max(native["pt2_transition_normalization_defects"]) < 1e-10


@pytest.mark.parametrize("unrestricted", [False, True])
def test_named_correlated_ecd_on_the_canonical_cli(tmp_path, unrestricted):
    numbers, positions, controls = request(unrestricted, "b2plyp")
    native = dft_rs.ecd_spectrum(numbers, positions, **controls)
    geometry = tmp_path / "geometry.xyz"
    geometry.write_text(str(len(numbers)) + "\ncorrelated transition fixture\n" + "\n".join(
        "H " + " ".join(format(x * BOHR_IN_ANGSTROM, ".17g") for x in position)
        for position in positions) + "\n", encoding="utf-8")
    executable = os.environ.get("DFT_RS_CLI", "dft-rs")
    command = [executable, "ecd-spectrum", str(geometry), "--format", "json"]
    for key, value in controls.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "iter" in completed.stderr and "residual" in completed.stderr
    report = json.loads(completed.stdout)
    for key in DIAGNOSTICS:
        np.testing.assert_allclose(report[key], native[key], rtol=2e-9, atol=2e-9)
    for a, b in zip(report["states"], native["states"]):
        for key in ["energy", "transition_dipole", "transition_momentum", "transition_angular_momentum",
                    "rotatory_strength", "rotatory_strength_velocity"]:
            np.testing.assert_allclose(a[key], b[key], rtol=2e-9, atol=2e-9)
