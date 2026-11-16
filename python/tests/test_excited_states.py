# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Excited-state calculations agree on native Python and ASE."""
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

import dft_rs
from dft_rs import DftRs
from test_fixed_populations import cli_report

WATER_NUMBERS = [8, 1, 1]
WATER = [[0.0, 0.0, 0.23], [0.0, 1.4309, -0.8867], [0.12, -1.4809, -0.9067]]
TIGHT = dict(basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-12)


def water_atoms():
    return Atoms("OH2", positions=np.asarray(WATER) * Bohr)


def calculator():
    return DftRs(units="atomic", basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-12)


def test_excitations_agree_on_native_and_ase():
    native = dft_rs.excitations(WATER_NUMBERS, WATER, nstates=3, **TIGHT)
    adapted = calculator().get_excitations(water_atoms(), nstates=3, units="atomic")
    assert native["feature"] == "excitations" and native["converged"]
    assert native["td_method"] == "tda" and native["td_spin"] == "singlet"
    assert len(native["states"]) == 3
    for a, b in zip(native["states"], adapted["states"]):
        assert abs(a["energy"] - b["energy"]) < 1e-9
        assert abs(a["oscillator_strength"] - b["oscillator_strength"]) < 1e-8
        assert a["energy"] > 0.0 and "rotatory_strength_velocity" in a


@pytest.mark.parametrize(("reference", "charge", "multiplicity"), [
    ("rks", 0, 1),
    ("uks", 1, 2),
])
def test_molecular_natural_transition_orbitals(tmp_path, reference, charge, multiplicity):
    controls = dict(TIGHT, xc="wb97x-d", reference=reference, charge=charge,
                    multiplicity=multiplicity, grid_level=1, grid_weight_cutoff=0.,
                    cpu_threads=1, nstates=2, amplitudes=True, td_nto=True,
                    td_difference_density=True)
    native = dft_rs.excitations(WATER_NUMBERS, WATER, **controls)
    ase = DftRs(units="atomic").get_excitations(water_atoms(), **controls)
    cli = cli_report(tmp_path, "excitations", water_atoms(), controls)
    for i, state in enumerate(native["states"]):
        transferred = 0.
        for s, channel in enumerate(state["natural_transition_orbitals"]):
            amplitude = np.asarray(state["amplitudes"][s])
            scale = np.sqrt(2.) if reference == "rks" else 1.
            singular = np.linalg.svd(amplitude * scale, compute_uv=False)
            singular = singular[singular > 1e-11]
            np.testing.assert_allclose(channel["singular_values"], singular, rtol=0., atol=1e-11)
            assert channel["reconstruction_residual"] < 1e-11
            np.testing.assert_allclose(channel["transition_weight"], np.sum((amplitude * scale)**2),
                                       rtol=0., atol=1e-12)
            nd = state["natural_difference_orbitals"][s]
            occupation = np.linalg.eigvalsh(amplitude.T @ amplitude)[::-1]
            occupation = occupation[:len(singular)]
            np.testing.assert_allclose(nd["attachment_occupations"], occupation, rtol=0., atol=1e-11)
            np.testing.assert_allclose(nd["detachment_occupations"], -occupation, rtol=0., atol=1e-11)
            transferred += nd["attachment_electrons"]
            for report in (ase, cli):
                other = report["states"][i]["natural_transition_orbitals"][s]
                np.testing.assert_allclose(other["weights"], channel["weights"], rtol=0., atol=1e-9)
                other_nd = report["states"][i]["natural_difference_orbitals"][s]
                np.testing.assert_allclose(other_nd["attachment_occupations"], occupation, rtol=0., atol=1e-9)
            print(f"{reference} water charge {charge} state {i} {channel['channel']}: "
                  f"transition weights {channel['weights']}, residual {channel['reconstruction_residual']:.3e}")
        assert abs(transferred - 1.) < 1e-10


def test_spectrum_integrates_to_the_oscillator_strengths():
    report = dft_rs.ecd_spectrum(WATER_NUMBERS, WATER, nstates=4, broadening_ev=0.3,
                                 points=2001, **TIGHT)
    adapted = calculator().get_ecd_spectrum(water_atoms(), nstates=4, broadening_ev=0.3,
                                            points=2001, units="atomic")
    assert len(report["energies_ev"]) == 2001
    total = report["oscillator_strength_sum"]
    assert abs(report["absorption_integral_oscillator_strength"] - total) < 1e-5 * total
    np.testing.assert_allclose(adapted["molar_absorptivity"], report["molar_absorptivity"],
                               rtol=1e-6, atol=1e-6)


def test_excited_state_gradient_matches_on_native_and_ase():
    native = dft_rs.td_state_gradient(WATER_NUMBERS, WATER, root=1, **TIGHT)
    adapted = calculator().get_td_state_gradient(water_atoms(), root=1, units="atomic")
    assert np.asarray(native["gradient"]).shape == (3, 3)
    assert native["translation_residual"] < 1e-7
    assert abs(native["excited_state_energy"] - native["energy"] - native["excitation_energy"]) < 1e-12
    np.testing.assert_allclose(adapted["gradient"], native["gradient"], atol=1e-8)
