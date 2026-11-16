# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import math

import pytest

from dft_rs.native import (
    HARTREE_IN_EV,
    calculate,
    nmr,
    optimize,
    periodic,
    periodic_capabilities,
    periodic_scf,
    polarizability,
    raman,
    saddle,
    sapt,
    tddft,
    vibrations,
)


H2 = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
WATER = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]]
# Bohr, matching `tests/p12_sapt.rs`'s `water_a`/`water_b(0.0)` exactly, so the
# same monomer separation this project already gates in Rust is reused here.
SAPT_WATER_A = [[0.0, 0.0, 0.0], [0.0, 1.43, 1.108], [0.0, -1.43, 1.108]]
SAPT_WATER_B = [[0.0, 0.0, 5.4], [0.0, 1.43, 6.5], [0.0, -1.43, 6.5]]


def test_native_result_schema_and_charge_conservation():
    result = calculate([1, 1], H2, wavefunction=True)
    assert result["converged"]
    assert math.isfinite(result["energy"])
    assert abs(sum(result["mulliken"])) < 1e-9
    assert result["e_nlc"] == 0.0
    assert result["e_dispersion"] == 0.0
    wavefunction = result["wavefunction"]
    assert set(wavefunction) == {"nao", "mo_energy", "mo_coeff", "density", "overlap"}


def test_invalid_ghost_and_controls_raise_value_error():
    with pytest.raises(ValueError, match="Z=0"):
        calculate([0], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="grid_level"):
        calculate([1], [[0.0, 0.0, 0.0]], charge=1, grid_level=10)
    with pytest.raises(ValueError, match="conv_tol"):
        calculate([1], [[0.0, 0.0, 0.0]], multiplicity=2, conv_tol=0.0)


@pytest.mark.parametrize("name", ["Me", "Et", "tBu", "Ph", "CF3", "OMe"])
def test_pseudo_substituent_names_are_typed_refusals_before_scf(name):
    with pytest.raises(NotImplementedError, match="named pseudo-substituent"):
        calculate([6, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], pseudo_sub=f"1={name}")


@pytest.mark.parametrize("spec", ["1=bogus", "1=Me;1=Et", "0=Me", "2=Me"])
def test_invalid_pseudo_substituent_requests_remain_value_errors(spec):
    with pytest.raises(ValueError):
        calculate([6, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], pseudo_sub=spec)


def test_builtin_vv10_is_not_silently_omitted():
    plain = calculate(
        [1, 1],
        H2,
        xc="b97m-v",
        dispersion="none",
        grid_level=1,
        vv10_grid_level=1,
    )
    complete = calculate(
        [1, 1],
        H2,
        xc="b97m-v",
        dispersion="auto",
        grid_level=1,
        vv10_grid_level=1,
    )
    assert plain["e_nlc"] == 0.0
    assert abs(complete["e_nlc"]) > 1e-8
    assert abs(complete["energy"] - plain["energy"]) > 1e-8


def test_density_fitting_is_exposed_and_numerically_controlled():
    exact = calculate([1, 1], H2, basis="def2-svp")
    fitted = calculate(
        [1, 1],
        H2,
        basis="def2-svp",
        density_fit="def2-universal-jfit",
    )
    assert fitted["converged"]
    assert abs(fitted["energy"] - exact["energy"]) < 2e-3


def test_natural_population_analysis_conserves_electrons_and_charge():
    result = calculate([1, 1], H2, nbo=True)
    analysis = result["nbo"]
    assert analysis["method"].startswith("natural population analysis")
    assert analysis["lewis_structures"] == [["BD(1) H1-H2"]]
    assert sum(analysis["atomic_populations"]) == pytest.approx(2.0, abs=1e-10)
    assert sum(analysis["atomic_charges"]) == pytest.approx(0.0, abs=1e-10)
    assert analysis["atomic_populations"][0] == pytest.approx(
        analysis["atomic_populations"][1], abs=1e-10
    )
    assert analysis["wiberg_bond_indices"][0][1] > 0.5


def test_periodic_capabilities_expose_staged_physics():
    capabilities = periodic_capabilities()
    assert capabilities["finite_k"]
    assert capabilities["fermi_dirac_occupations"]
    assert capabilities["periodic_ecp_energy"]
    assert capabilities["periodic_ecp_forces"]
    assert capabilities["fixed_cell_optimization"]
    assert not capabilities["partial_boundaries"]
    assert not capabilities["vv10"]
    assert not capabilities["chg"]


def test_periodic_optimization_controls_are_not_silently_ignored():
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    with pytest.raises(ValueError, match="require relax=True"):
        periodic_scf(
            [2],
            [[0.0, 0.0, 0.0]],
            cell,
            xc="hf",
            opt_max_steps=1,
        )
    with pytest.raises(ValueError, match="opt_max_steps"):
        periodic_scf(
            [2],
            [[0.0, 0.0, 0.0]],
            cell,
            xc="hf",
            relax=True,
            opt_max_steps=0,
        )
    with pytest.raises(ValueError, match="adiis_until_error"):
        periodic_scf(
            [2],
            [[0.0, 0.0, 0.0]],
            cell,
            xc="hf",
            adiis_until_error=float("nan"),
        )


def test_periodic_gamma_density_is_shape_checked_before_hartree_build():
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    with pytest.raises(ValueError, match="density must be a finite 1x1"):
        periodic(
            [1],
            [[0.0, 0.0, 0.0]],
            cell,
            multiplicity=2,
            density=[[1.0, 0.0], [0.0, 1.0]],
        )


def test_periodic_gamma_density_count_matches_bloch_overlap():
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    result = periodic(
        [1],
        [[0.0, 0.0, 0.0]],
        cell,
        multiplicity=2,
        overlap_cutoff=12.0,
        density=[[0.75]],
    )
    s_gamma = result["overlap"][0]["real"][0][0]
    assert result["density_electron_count"] == pytest.approx(0.75 * s_gamma, abs=1e-12)
    assert len(result["hartree"]) == 1


def test_periodic_ecp_core_derivative_is_analytic_and_explicit():
    cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]]
    result = periodic(
        [11, 1],
        [[0.0, 0.0, 0.0], [0.3, -0.2, 2.4]],
        cell,
        basis="lanl2dz",
        overlap_cutoff=6.0,
        derivatives=True,
    )
    derivative = result["derivatives"][0]
    assert "ecp" in derivative
    assert any(
        abs(value) > 1e-10
        for atom in derivative["ecp"]
        for axis in atom
        for row in axis["real"]
        for value in row
    )
    for atom in range(2):
        for axis in range(3):
            for part in ("real", "imag"):
                core = derivative["core_hamiltonian"][atom][axis][part]
                kinetic = derivative["kinetic"][atom][axis][part]
                nuclear = derivative["nuclear"][atom][axis][part]
                ecp = derivative["ecp"][atom][axis][part]
                for i in range(result["nao"]):
                    for j in range(result["nao"]):
                        assert core[i][j] == pytest.approx(
                            kinetic[i][j] + nuclear[i][j] + ecp[i][j], abs=1e-13
                        )


@pytest.mark.parametrize("xc", ["wb97x-v", "wb97m-v", "b97m-v", "wb97x-d"])
def test_periodic_nonlocal_functionals_are_rejected(xc):
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    with pytest.raises(NotImplementedError, match=r"nonlocal .*term is not yet periodic"):
        periodic_scf(
            [1],
            [[0.0, 0.0, 0.0]],
            cell,
            multiplicity=2,
            xc=xc,
        )


# --- external field ---------------------------------------------------


def test_external_field_matches_the_conjugacy_identity():
    """`dE/dE_z = -mu_z` at zero field, exactly as scope.md's gate states.

    Water's C2v geometry here puts its dipole entirely along z, so a
    forward-difference derivative of the field-perturbed energy should
    reproduce minus the ZERO-field dipole to first order. This is the same
    identity the Rust test suite gates the polarizability CPKS solve
    against; here it is checking that the BINDING actually threads
    `field=` into `ScfOptions::external_field` rather than dropping it.
    """
    delta = 1e-3
    base = calculate([8, 1, 1], WATER, basis="sto-3g")
    perturbed = calculate([8, 1, 1], WATER, basis="sto-3g", field=[0.0, 0.0, delta])
    assert base["dipole"][2] != 0.0
    d_energy_d_field = (perturbed["energy"] - base["energy"]) / delta
    assert d_energy_d_field == pytest.approx(-base["dipole"][2], abs=2e-2)


def test_external_field_composes_with_the_analytic_gradient():
    result = calculate(
        [8, 1, 1], WATER, basis="sto-3g", field=[0.0, 0.0, 1e-3], gradient=True
    )
    assert result["converged"]
    grad = result["gradient"]
    assert len(grad) == 3
    assert all(math.isfinite(v) for row in grad for v in row)


# --- continuum solvation -----------------------------------------------


def test_solvent_pcm_stabilizes_and_energy_accounting_is_consistent():
    """The PCM reaction field is self-consistent, so it is already folded
    into `e_elec`/`energy`; `e_pcm` is reported only as a diagnostic and
    must not be double-added."""
    gas = calculate([8, 1, 1], WATER, basis="sto-3g")
    solvated = calculate([8, 1, 1], WATER, basis="sto-3g", solvent="water")
    assert solvated["e_pcm"] < 0.0  # C-PCM's induced-charge energy is <= 0
    assert solvated["energy"] < gas["energy"]
    assert solvated["e_smd_cds"] == 0.0
    assert solvated["energy"] == pytest.approx(
        solvated["e_elec"] + solvated["e_nuc"] + solvated["e_dispersion"] + solvated["e_smd_cds"],
        abs=1e-9,
    )


def test_solvent_smd_prefix_adds_the_non_electrostatic_term_on_top():
    """Unlike `e_pcm`, SMD's non-electrostatic term is NOT self-consistent
    and must be added on top of the SCF total by the caller, exactly as the
    CLI does it (`src/bin/dft_rs.rs`)."""
    result = calculate([8, 1, 1], WATER, basis="sto-3g", solvent="smd:water")
    assert result["e_pcm"] < 0.0
    assert result["e_smd_cds"] != 0.0
    assert result["energy"] == pytest.approx(
        result["e_elec"] + result["e_nuc"] + result["e_dispersion"] + result["e_smd_cds"],
        abs=1e-9,
    )


def test_solvent_custom_with_eps_needs_no_named_table_entry():
    result = calculate([8, 1, 1], WATER, basis="sto-3g", solvent="custom", eps=78.3553)
    assert result["e_pcm"] < 0.0


def test_unknown_solvent_and_custom_without_eps_raise_value_error():
    with pytest.raises(ValueError, match="unknown solvent"):
        calculate([1, 1], H2, solvent="not-a-real-solvent")
    with pytest.raises(ValueError, match="custom"):
        calculate([1, 1], H2, solvent="custom")


# --- polarizability ------------------------------------------------------


def test_polarizability_tensor_is_symmetric_and_isotropic_matches_trace():
    result = polarizability([1, 1], H2, basis="sto-3g", xc="hf")
    alpha = result["alpha"]
    assert len(alpha) == 3 and all(len(row) == 3 for row in alpha)
    for i in range(3):
        for j in range(3):
            assert alpha[i][j] == pytest.approx(alpha[j][i], abs=1e-8)
    assert result["isotropic"] == pytest.approx(
        (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0, abs=1e-10
    )
    assert result["isotropic"] > 0.0
    assert math.isfinite(result["energy"])


def test_remaining_molecular_entry_points_are_wired_and_validate_controls():
    optimized = optimize([1, 1], H2, max_steps=1, gtol=1e3)
    assert optimized["converged"]
    assert optimized["iterations"] == 0
    assert len(optimized["positions"]) == 2
    assert len(optimized["trajectory"]) == 1

    with pytest.raises(ValueError, match="hessian_step"):
        raman([1, 1], H2, hessian_step=0.0)
    with pytest.raises(ValueError, match="hessian_step"):
        saddle([1, 1], H2, hessian_step=0.0)
    with pytest.raises(ValueError, match="saddle method"):
        saddle([1, 1], H2, method="invented")


def test_vibrations_and_saddle_public_signatures_expose_new_controls():
    import inspect

    vibration_parameters = inspect.signature(vibrations).parameters
    assert {"temperature", "pressure", "symmetry_number"} <= set(vibration_parameters)
    saddle_parameters = inspect.signature(saddle).parameters
    assert saddle_parameters["method"].default == "prfo"


def test_nmr_binding_returns_full_per_nucleus_tensors_and_typed_refusals():
    result = nmr([1, 1], H2, basis="sto-3g", xc="hf")
    assert result["gauge"] == "Giao"
    assert len(result["shieldings"]) == 2
    for entry in result["shieldings"]:
        assert len(entry["tensor"]) == 3
        assert all(len(row) == 3 for row in entry["tensor"])
        assert math.isfinite(entry["isotropic"])

    with pytest.raises(ValueError, match="magnetic gauge must be"):
        nmr([1, 1], H2, gauge="invented")
    atom = nmr([1], [[0.0, 0.0, 0.0]], multiplicity=2)
    assert atom["unrestricted"] and atom["converged"]
    assert atom["shape"] == [1, 3, 3]
    assert all(value == 0 for row in atom["shieldings"][0]["paramagnetic"] for value in row)


# --- TDDFT / ECD -----------------------------------------------------------


def test_tddft_tda_states_are_ordered_with_non_negative_oscillator_strength():
    result = tddft([1, 1], H2, basis="sto-3g", xc="hf", nstates=3, method="tda")
    assert result["method"] == "tda"
    states = result["states"]
    assert 0 < len(states) <= 3
    energies = [s["energy"] for s in states]
    assert energies == sorted(energies)
    for s in states:
        assert s["oscillator_strength"] >= -1e-10
        assert s["energy_ev"] == pytest.approx(s["energy"] * HARTREE_IN_EV, rel=1e-9)
        assert s["tamm_dancoff"] is True
        # H2 has an inversion centre, hence is achiral: both ECD gauges
        # must vanish exactly (up to numerical noise) for every state.
        assert abs(s["rotatory_strength_length"]) < 1e-6
        assert abs(s["rotatory_strength_velocity"]) < 1e-6


def test_tddft_rejects_unknown_method_and_spin():
    with pytest.raises(ValueError, match="unknown method"):
        tddft([1, 1], H2, method="bogus")
    with pytest.raises(ValueError, match="unknown spin"):
        tddft([1, 1], H2, spin="bogus")


# --- SAPT ------------------------------------------------------------------


def test_sapt_components_sum_to_the_reported_interaction_total():
    result = sapt(
        [8, 1, 1],
        SAPT_WATER_A,
        [8, 1, 1],
        SAPT_WATER_B,
        basis="sto-3g",
        xc="hf",
    )
    for key in (
        "e_monomer_a",
        "e_monomer_b",
        "e_elst",
        "e_exch",
        "e_ind_a",
        "e_ind_b",
        "e_exch_ind",
        "e_disp_unc",
        "e_disp_sos_uncoupled",
        "e_disp_coupled",
        "e_exch_disp",
        "e_int_supermolecular",
        "delta",
        "interaction_total",
    ):
        assert math.isfinite(result[key]), key
    assert result["interaction_total"] == pytest.approx(
        result["e_elst"]
        + result["e_exch"]
        + result["e_ind_a"]
        + result["e_ind_b"]
        + result["e_exch_ind"]
        + result["e_disp_coupled"]
        + result["e_exch_disp"],
        rel=1e-9,
    )
    # First-order exchange (Pauli repulsion) is always destabilizing, and
    # second-order dispersion always stabilizing -- true of any SAPT
    # decomposition regardless of the particular dimer geometry.
    assert result["e_exch"] > 0.0
    assert result["e_disp_coupled"] < 0.0


def test_sapt_rejects_ecp_basis_before_returning_incomplete_components():
    with pytest.raises(NotImplementedError, match="ECP"):
        sapt(
            [11],
            [[0.0, 0.0, 0.0]],
            [1],
            [[0.0, 0.0, 6.0]],
            basis="lanl2dz",
            multiplicity_a=2,
            multiplicity_b=2,
        )


def test_sapt_refuses_a_monomer_whose_response_has_blown_up():
    """Mirrors `tests/p12_sapt.rs`'s
    `sapt_refuses_a_monomer_whose_response_has_blown_up`: a triplet oxygen
    atom beside a water has a nearly singular orbital Hessian (partially
    filled degenerate shell, symmetry-broken SCF at a saddle point), so
    SAPT must raise rather than report the runaway rotation as an energy.
    """
    ang = 1.889_726_124_6
    water = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.957 * ang],
        [0.927 * ang, 0.0, -0.240 * ang],
    ]
    o_triplet = [[0.0, 0.0, 2.900 * ang]]
    with pytest.raises(ValueError, match="orbital rotation|non-positive curvature"):
        sapt(
            [8, 1, 1],
            water,
            [8],
            o_triplet,
            basis="sto-3g",
            multiplicity_b=3,
        )
