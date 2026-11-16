# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Experimental quantum-nuclear SCF uses one driver and one unit boundary."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree

import dft_rs
from test_fixed_populations import cli_report


POSITIONS = [[0., 0., -.7], [0., 0., .7]]
ORACLES = json.loads((Path(__file__).resolve().parents[2]/
                     "tests/oracle_data/multicomponent_hf.json").read_text(encoding="utf-8"))

@pytest.mark.parametrize("electric", [False, True, "quadrupole"])
@pytest.mark.parametrize("interaction", ["coulomb", "gaunt", "breit"])
def test_joint_dirac_coulomb_all_quantum_public_routes(tmp_path, electric, interaction):
    basis = tmp_path / "finite.g94"
    basis.write_text("H 0\nS 1 1.0\n0.84 1.0\n****\nHe 0\nS 1 1.0\n0.93 1.0\n****\n", encoding="ascii")
    numbers = [2, 1]
    positions = np.array([[0.12, -0.27, 0.09], [1.41, 0.31, -0.46]])
    atoms = Atoms(numbers, positions=positions * Bohr)
    options = dict(
        species_electronic_model="dirac-coulomb", basis=str(basis), charge=1, speed_of_light=7.4,
        nuclear_rms_radii=[0.12, 0.081], quantum_atoms=[0, 1], quantum_masses=[17.0, 19.0],
        quantum_exponents=[4.0, 8.0], quantum_lmax=0, cpu_threads=1,
    )
    if interaction != "coulomb":
        options["dirac_interaction"] = interaction
    if electric:
        options.update(field=[0.012, -0.008, 0.005], species_polarizability=True,
                       frequency=0.04, frequency_damping=0.004)
    if electric == "quadrupole":
        options.update(species_quadrupole_polarizability=True,
                       species_field_gradient=[0.003, -0.001, 0.002, -0.003, 0.004],
                       origin=[0.12, 0.21, -0.08])
    ase_options = dict(options, nuclear_rms_radii=np.asarray(options["nuclear_rms_radii"])*Bohr,
                       quantum_exponents=np.asarray(options["quantum_exponents"])/Bohr**2)
    if electric:
        ase_options.update(field=np.asarray(options["field"])*Hartree/Bohr,
                           frequency=options["frequency"]*Hartree,
                           frequency_damping=options["frequency_damping"]*Hartree)
    if electric == "quadrupole":
        ase_options.update(
            species_field_gradient=np.asarray(options["species_field_gradient"])*Hartree/Bohr**2,
            origin=np.asarray(options["origin"])*Bohr,
        )
    direct = dft_rs.multicomponent(numbers, positions, **options)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
    physical = dft_rs.DftRs().get_multicomponent(atoms, **ase_options)
    xyz = tmp_path / "helium_hydride.xyz"
    xyz.write_text(
        "2\nTwo quantum nuclei\n" + "\n".join(
            f"{number} {x:.17g} {y:.17g} {z:.17g}"
            for number, (x, y, z) in zip(numbers, atoms.positions, strict=True)
        ) + "\n", encoding="ascii",
    )
    command = [os.environ["DFT_RS_CLI"], "multicomponent", str(xyz), "--format", "json"]
    for key, value in options.items():
        command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, check=True, encoding="utf-8", capture_output=True)
    assert any("[joint-no-pair] iteration" in line and "energy" in line and "residual" in line
               for line in process.stderr.splitlines())
    rejected = subprocess.run([*command, "--species-derivative-order", "1"],
                              encoding="utf-8", capture_output=True)
    assert rejected.returncode != 0 and "species_derivative_order" in rejected.stderr
    for report in (direct, atomic, json.loads(process.stdout)):
        assert report["converged"] and report["projector_reference"] == "free-particle"
        assert report["iterations"] > 1
        assert report["classical_nucleus_count"] == 0
        assert report["units"] == direct["units"]
        assert report["energy"] == pytest.approx(direct["energy"], abs=1e-11, rel=0)
        assert report["energy_components"] == pytest.approx(direct["energy_components"], abs=1e-11, rel=0)
        if interaction != "coulomb":
            assert report["dirac_interaction"] == interaction
            assert abs(report["energy_components"]["current_exchange"]) > 1e-6
            assert report["current_hermiticity_defect"] < 1e-12
            assert report["units"]["current_hermiticity_defect"] == "hartree"
        for key in ("electronic_density_real", "electronic_density_imaginary"):
            np.testing.assert_allclose(report[key], direct[key], rtol=0, atol=1e-11)
        for actual, expected in zip(report["quantum"], direct["quantum"], strict=True):
            assert actual["orbital_residual"] < 1e-10
            for key in ("density", "fock", "orbital_residual", "density_residual", "particle_number_error"):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-11)
    assert physical["energy"]/Hartree == pytest.approx(direct["energy"], abs=1e-11, rel=0)
    for actual, expected in zip(physical["quantum"], direct["quantum"], strict=True):
        assert actual["rms_radius"] / Bohr == pytest.approx(expected["rms_radius"], abs=1e-13)
        np.testing.assert_allclose(np.asarray(actual["fock"]) / Hartree, expected["fock"], rtol=0, atol=1e-11)
    if electric:
        for report in (atomic, json.loads(process.stdout)):
            np.testing.assert_allclose(report["dipole"], direct["dipole"], rtol=0, atol=1e-11)
            for key in ("tensor_real", "tensor_imag", "response_residuals", "reference_residual"):
                np.testing.assert_allclose(report["polarizability"][key], direct["polarizability"][key],
                                           rtol=0, atol=1e-10)
        np.testing.assert_allclose(np.asarray(physical["dipole"])/Bohr, direct["dipole"], atol=1e-11)
        for key in ("tensor_real", "tensor_imag"):
            np.testing.assert_allclose(np.asarray(physical["polarizability"][key])/Bohr**3,
                                       direct["polarizability"][key], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(direct["dipole_origin"], options.get("origin", np.zeros(3)))
        assert direct["units"]["dipole_origin"] == "bohr"
        assert "exp(-i z t)" in direct["polarizability"]["axis_convention"]
        assert direct["polarizability"]["frequency_real"] == options["frequency"]
        assert direct["polarizability"]["frequency_imag"] == options["frequency_damping"]
        assert physical["polarizability"]["frequency_real"]/Hartree == pytest.approx(options["frequency"])
        assert direct["polarizability"]["tensor_imag"][0][0] > 0
        assert max(direct["polarizability"]["response_residuals"]) <= 1e-10
    if electric == "quadrupole":
        joint = direct["electric_quadrupole"]
        assert joint["origin_convention"] == "explicit"
        assert len(joint["response_residuals"]) == 8
        assert max(joint["response_residuals"]) <= 1e-10
        assert "polarizability" not in joint
        keys_and_factors = {
            "origin": Bohr,
            "moment": Bohr**2,
            "polarizability_real": Bohr**4/Hartree,
            "polarizability_imag": Bohr**4/Hartree,
            "dipole_from_field_gradient_real": Bohr**3/Hartree,
            "dipole_from_field_gradient_imag": Bohr**3/Hartree,
            "field_conjugate_moment_from_field_real": Bohr**3/Hartree,
            "field_conjugate_moment_from_field_imag": Bohr**3/Hartree,
            "response_residuals": 1.0,
            "reference_residual": Hartree,
        }
        for key, factor in keys_and_factors.items():
            assert "electric_quadrupole."+key in direct["units"]
            for report in (atomic, json.loads(process.stdout)):
                np.testing.assert_allclose(report["electric_quadrupole"][key], joint[key],
                                           rtol=0, atol=1e-10)
            actual = np.asarray(physical["electric_quadrupole"][key])/factor
            np.testing.assert_allclose(actual, joint[key], rtol=0, atol=1e-10)
        np.testing.assert_allclose(physical["field_gradient"], ase_options["species_field_gradient"],
                                   rtol=0, atol=1e-12)
        assert np.shape(joint["polarizability_real"]) == (5, 5)
        assert np.shape(joint["dipole_from_field_gradient_real"]) == (3, 5)
        assert np.shape(joint["field_conjugate_moment_from_field_real"]) == (5, 3)
    print(f"Installed joint Dirac {interaction.capitalize()}", json.dumps(direct, sort_keys=True, allow_nan=False))


@pytest.mark.parametrize("xc,unrestricted", [
    ("hf", False), ("svwn5", True), ("pbe", False),
    ("tpss", True), ("pbe0", False), ("cam-b3lyp", True),
])
def test_simultaneous_quantum_nuclei_share_all_routes(tmp_path, xc, unrestricted):
    options = dict(xc=xc, unrestricted=unrestricted, basis="sto-3g", dispersion="none",
                   quantum_exponents=[4., 8.], quantum_lmax=0,
                   interspecies_correlation="none", grid_level=2,
                   conv_tol=1e-13, conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    report = dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[1], **options)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, quantum_atoms=[1], **options)
    cli = cli_report(tmp_path, "multicomponent", atoms, dict(options, quantum_atoms=[1]))
    for other in [report, atomic, cli]:
        assert other["converged"] and other["experimental"]
        assert not other["center_of_mass_removed"]
        assert other["electronic_orbital_residual"] < 1e-10
        assert other["electronic_density_residual"] < 1e-10
        assert other["quantum"][0]["density_residual"] < 1e-10
        assert other["quantum"][0]["particle_number"] == pytest.approx(1., abs=1e-12)
        assert other["energy"] == pytest.approx(report["energy"], abs=1e-10)
        assert sum(other["energy_components"].values()) == pytest.approx(other["energy"], abs=1e-11)
    physical = dft_rs.DftRs().get_multicomponent(atoms, quantum_atoms=[1], **dict(
        options, quantum_exponents=np.asarray(options["quantum_exponents"])/Bohr**2,
        conv_tol=options["conv_tol"]*Hartree, conv_tol_grad=options["conv_tol_grad"]*Hartree))
    assert physical["energy"]/Hartree == pytest.approx(report["energy"], abs=1e-10)
    np.testing.assert_allclose(np.asarray(physical["quantum"][0]["position"])/Bohr,
                               report["quantum"][0]["position"], atol=1e-10, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["quantum"][0]["position_variance"])/Bohr**2,
                               report["quantum"][0]["position_variance"], atol=1e-10, rtol=0.)
    assert physical["quantum"][0]["mass"] == report["quantum"][0]["mass"]
    print("multicomponent public", xc, unrestricted, report["energy"], report["iterations"],
          report["electronic_orbital_residual"], report["quantum"][0]["orbital_residual"])


def test_all_quantum_and_added_positron_models_are_explicit():
    options = dict(xc="hf", dispersion="none", quantum_exponents=[4., 8.], quantum_lmax=0,
                   interspecies_correlation="none", conv_tol=1e-13,
                   conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    report = dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[0, 1], **options)
    assert report["converged"] and len(report["quantum"]) == 2
    assert any("center-of-mass" in warning for warning in report["warnings"])
    assert report["energy_components"]["classical"] == 0.
    assert report["energy_components"]["quantum_classical"] == 0.
    positron = dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[],
                                    positron_positions=[[0., 0., 0.]], **options)
    assert positron["converged"]
    assert positron["quantum"][0]["mass"] == 1.
    assert positron["quantum"][0]["correlation_channel"] is None
    assert positron["quantum"][0]["replaced_atom"] is None
    assert positron["energy_components"]["interspecies_correlation"] == 0.
    assert any("annihilation" in warning for warning in positron["warnings"])


def test_quantum_particle_invalid_combinations_are_rejected():
    options = dict(xc="hf", dispersion="none", interspecies_correlation="none")
    for extra in [dict(quantum_atoms=[1, 1]), dict(quantum_atoms=[2]),
                  dict(quantum_atoms=[1], quantum_masses=[-1.]),
                  dict(quantum_atoms=[1], quantum_lmax=-1),
                  dict(quantum_atoms=[1], quantum_exponents=[0.]),
                  dict(quantum_atoms=[1], species_mixing=0.),
                  dict(quantum_atoms=[])]:
        with pytest.raises(ValueError):
            dft_rs.multicomponent([1, 1], POSITIONS, **options, **extra)
    for extra in [dict(quantum_atoms=[1], quantum_lmax=4),
                  dict(quantum_atoms=[1], species_derivative_order=3)]:
        with pytest.raises(NotImplementedError, match="not implemented"):
            dft_rs.multicomponent([1, 1], POSITIONS, **options, **extra)
    with pytest.raises(NotImplementedError, match="jointly variational"):
        dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[1],
                             electronic_temperature=300., **options)


@pytest.mark.parametrize("case", ORACLES, ids=lambda c: c["name"])
def test_five_quantum_nuclear_energies_match_independent_scf(case):
    report = dft_rs.multicomponent(case["numbers"], case["positions"],
        quantum_atoms=[case["quantum_atom"]], quantum_exponents=case["quantum_exponents"],
        quantum_lmax=case["quantum_lmax"], xc="hf", dispersion="none",
        interspecies_correlation="none", species_tolerance=1e-12,
        conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    assert_independent_quantum_energy(report, case)


def assert_independent_quantum_energy(report, case):
    """Check a computed report against an independently optimized reference."""
    assert report["converged"]
    errors = {key: abs(value-case["energy_components"][key])
              for key, value in report["energy_components"].items()}
    print("independent quantum HF", case["name"], "energy", report["energy"],
          "energy_error", abs(report["energy"]-case["energy"]), "term_errors", errors,
          "electronic_residual", report["electronic_orbital_residual"],
          "quantum_map", report["quantum"][0]["density_residual"])
    assert abs(report["energy"]-case["energy"]) < 2e-9
    assert max(errors.values()) < 2e-8


def test_one_basis_quantum_space_still_converges_electronic_epc():
    report = dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[1],
        quantum_exponents=[8.], quantum_lmax=0, xc="hf", dispersion="none",
        interspecies_correlation="epc17", conv_tol=1e-13,
        conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    assert report["converged"]
    assert report["energy_components"]["interspecies_correlation"] < 0.
    assert report["electronic_density_residual"] < 1e-10


@pytest.mark.parametrize("constrained", [False, True])
def test_coupled_geometry_hessian_and_polarizability_share_all_routes(tmp_path, constrained):
    options = dict(xc="hf", dispersion="none", quantum_atoms=[1],
        quantum_exponents=[4., 8.], quantum_lmax=1, interspecies_correlation="none",
        quantum_position_constraint=constrained, species_derivative_order=2,
        species_polarizability=True, species_tolerance=1e-12,
        conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    if constrained:
        # Explicit common targets avoid interpreting the last digits of a
        # converted XYZ basis center as the requested expectation position.
        options["quantum_position_targets"] = [POSITIONS[1]]
    report = dft_rs.multicomponent([1, 1], POSITIONS, **options)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
    cli = cli_report(tmp_path, "multicomponent", atoms, options)
    for other in [report, atomic, cli]:
        assert other["converged"]
        d = other["geometry_derivatives"]
        assert d["converged"] and d["hessian_shape"] == [6, 6]
        assert d["directional_evaluations"] == 21
        assert max(d["response_residuals"]) < 1e-10
        np.testing.assert_allclose(d["gradient"], report["geometry_derivatives"]["gradient"], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(d["hessian"], report["geometry_derivatives"]["hessian"], atol=1e-8, rtol=0.)
        h = np.asarray(d["hessian"]).reshape(2, 3, 2, 3)
        assert np.max(np.abs(h.sum(axis=0))) < 1e-8
        assert np.max(np.abs(np.sum(d["gradient"], axis=0))) < 1e-10
        assert other["polarizability"]["converged"]
        assert other["polarizability"]["raw_asymmetry"] < 1e-9
        if constrained:
            assert other["position_constraints"]["residual"] < 1e-12
            np.testing.assert_allclose(other["quantum"][0]["position"], POSITIONS[1], atol=1e-12, rtol=0.)
    ase_options = dict(options, quantum_exponents=np.asarray(options["quantum_exponents"])/Bohr**2,
                       conv_tol=options["conv_tol"]*Hartree, conv_tol_grad=options["conv_tol_grad"]*Hartree)
    if constrained:
        ase_options["quantum_position_targets"] = np.asarray(options["quantum_position_targets"])*Bohr
    physical = dft_rs.DftRs().get_multicomponent(atoms, **ase_options)
    np.testing.assert_allclose(np.asarray(physical["geometry_derivatives"]["gradient"])*Bohr/Hartree,
                               report["geometry_derivatives"]["gradient"], atol=1e-9, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["geometry_derivatives"]["hessian"])*Bohr**2/Hartree,
                               report["geometry_derivatives"]["hessian"], atol=1e-8, rtol=0.)
    np.testing.assert_allclose(np.asarray(physical["polarizability"]["tensor"])/Bohr**3,
                               report["polarizability"]["tensor"], atol=1e-9, rtol=0.)
    if constrained:
        np.testing.assert_allclose(np.asarray(physical["position_constraints"]["multipliers"])*Bohr/Hartree,
                                   report["position_constraints"]["multipliers"], atol=1e-10, rtol=0.)


def test_first_order_and_directional_species_derivative_contracts():
    options = dict(xc="hf", dispersion="none", quantum_atoms=[1], quantum_exponents=[4., 8.],
                   interspecies_correlation="none", species_derivative_order=1)
    full = dft_rs.multicomponent([1, 1], POSITIONS, **options)["geometry_derivatives"]
    along = dft_rs.multicomponent([1, 1], POSITIONS,
        species_direction=[[0., 0., 1.], [0., 0., 0.]], **options)["geometry_derivatives"]
    assert full["directional_evaluations"] == 6 and "hessian" not in full
    assert "hessian" not in along
    assert full["gradient"][0][2] == pytest.approx(along["gradient"], abs=1e-10)


def test_all_quantum_constrained_isotope_pair_keeps_the_internal_stretch():
    proton = 1836.15267343
    report = dft_rs.multicomponent([1, 1], POSITIONS, xc="hf", dispersion="none",
        quantum_atoms=[0, 1], quantum_masses=[proton, 2*proton],
        quantum_exponents=[4., 8.], quantum_lmax=1, interspecies_correlation="none",
        quantum_position_constraint=True, quantum_position_targets=POSITIONS,
        species_derivative_order=2, species_direction=[[0., 0., -2/3], [0., 0., 1/3]],
        species_tolerance=1e-12, response_tolerance=1e-12,
        conv_tol=1e-14, conv_tol_grad=1e-12, conv_tol_dp=1e-12, max_iter=500)
    assert report["converged"] and report["position_constraints"]["residual"] < 1e-12
    assert report["energy_components"]["classical"] == 0.
    assert report["energy_components"]["quantum_classical"] == 0.
    assert report["energy_components"]["quantum_pair_coulomb"] > 0.
    assert not report["center_of_mass_removed"]
    assert report["geometry_derivatives"]["position_response_residual"] < 1e-10
    assert np.isfinite(report["geometry_derivatives"]["hessian"])


def test_structured_particles_do_not_inject_legacy_defaults(monkeypatch):
    captured = []

    def record(*args, **kwargs):
        captured.append((args, kwargs))
        return {"converged": True}

    monkeypatch.setattr(dft_rs.native, "run_feature", record)
    particles = [{"kind": "positron", "position": [0.1, -0.2, 0.3]}]
    dft_rs.multicomponent([1, 1], POSITIONS, particles=particles)
    assert captured[-1][1] == {"particles": particles}
    dft_rs.multicomponent([1, 1], POSITIONS, quantum_atoms=[])
    assert captured[-1][1] == {"quantum_atoms": []}


def test_structured_control_units_follow_registry_paths_without_mutation(monkeypatch):
    import copy
    from dft_rs import ase_calculator

    monkeypatch.setattr(ase_calculator, "_CONTROL_UNITS", {
        "particles": "text", "particles.position": "bohr",
        "particles.exponents": "1/bohr^2", "particles.mass": "atomic units of mass",
        "particles.charge": "e", "particles.kind": "text", "particles.atom": "integer",
        "field": "hartree/e/bohr", "gks_operator_real": "hartree",
    })
    controls = {"particles": [{"kind": "particle", "position": [0.3, -0.2, 0.7],
                              "exponents": [0.1, 0.5], "mass": 2., "charge": -1., "atom": 0}],
                "field": [0.03, -0.01, 0.02], "gks_operator_real": [[0.2, 0.1], [0.1, -0.3]]}
    original = copy.deepcopy(controls)
    actual = ase_calculator._controls_to_atomic(controls)
    assert controls == original
    p = actual["particles"][0]
    np.testing.assert_allclose(p["position"], np.asarray(original["particles"][0]["position"])/Bohr, rtol=3e-16, atol=0.)
    np.testing.assert_allclose(p["exponents"], np.asarray([0.1, 0.5])*Bohr**2, rtol=3e-16, atol=0.)
    assert (p["kind"], p["mass"], p["charge"], p["atom"]) == ("particle", 2., -1., 0)
    np.testing.assert_allclose(actual["field"], np.asarray(original["field"])*Bohr/Hartree, rtol=3e-16, atol=0.)
    np.testing.assert_allclose(actual["gks_operator_real"], np.asarray(original["gks_operator_real"])/Hartree, rtol=3e-16, atol=0.)


@pytest.mark.parametrize("correlation", ["none", "drummond-2011"])
def test_structured_positron_energy_annihilation_and_all_public_routes(tmp_path, correlation):
    controls = dict(xc="hf", dispersion="none", basis="sto-3g", grid_level=2,
                    grid_weight_cutoff=0., conv_tol=1e-13, conv_tol_grad=1e-11,
                    conv_tol_dp=1e-11, species_tolerance=1e-10, max_iter=500,
                    particles=[dict(kind="positron", position=[0.13, -0.08, 0.21],
                                    exponents=[0.15, 0.4], lmax=0, correlation=correlation)],
                    annihilation_enhancement="drummond-2011")
    native = dft_rs.multicomponent([1, 1], POSITIONS, **controls)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **controls)
    cli = cli_report(tmp_path, "multicomponent", atoms, controls)
    import copy
    ase_controls = copy.deepcopy(controls)
    ase_controls["particles"][0]["position"] = np.asarray(controls["particles"][0]["position"])*Bohr
    ase_controls["particles"][0]["exponents"] = np.asarray([0.15, 0.4])/Bohr**2
    ase_controls["conv_tol"] *= Hartree
    ase_controls["conv_tol_grad"] *= Hartree
    physical = dft_rs.DftRs().get_multicomponent(atoms, **ase_controls)
    assert native["converged"] and native["experimental"]
    for other in (atomic, cli, physical):
        assert other["converged"]
        value = other["energy"]/Hartree if other is physical else other["energy"]
        np.testing.assert_allclose(value, native["energy"], atol=1e-10, rtol=0.)
        assert other["quantum"][0]["mass"] == 1.
        rate = other["annihilation"]["positrons"][0]
        reference = native["annihilation"]["positrons"][0]
        for key in ("rate", "rate_per_ns", "lifetime_ps"):
            np.testing.assert_allclose(rate[key], reference[key], rtol=1e-10, atol=0.)
        contact = rate["contact_density"]*Bohr**3 if other is physical else rate["contact_density"]
        np.testing.assert_allclose(contact, reference["contact_density"], rtol=1e-10, atol=0.)
    rate = native["annihilation"]["positrons"][0]
    assert rate["rate"] > 0. and np.isfinite(rate["lifetime_ps"])
    np.testing.assert_allclose(rate["rate_per_ns"]*rate["lifetime_ps"], 1000., rtol=1e-14)
    assert native["quantum"][0]["correlation_channel"] == (None if correlation == "none" else 0)


def test_added_positron_geometry_public_routes(tmp_path):
    particle = dict(kind="positron", position=[0.13, -0.08, 0.21],
                    exponents=[0.15, 0.40], lmax=0, correlation="drummond-2011")
    controls = dict(xc="pbe", reference="rks", basis="sto-3g", dispersion="none",
                    grid_level=0, grid_weight_cutoff=0., cpu_threads=1,
                    species_derivative_order=2, species_tolerance=1e-10)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    native = dft_rs.multicomponent([1, 1], POSITIONS, particles=[particle], **controls)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, particles=[particle], **controls)
    cli = cli_report(tmp_path, "multicomponent", atoms, dict(controls, particles=[particle]))
    physical_particle = dict(particle, position=np.asarray(particle["position"])*Bohr,
                             exponents=np.asarray(particle["exponents"])/Bohr**2)
    physical = dft_rs.DftRs().get_multicomponent(atoms, particles=[physical_particle], **controls)
    reference = native["geometry_derivatives"]
    for report in (native, atomic, cli, physical):
        assert report["converged"]
        result = report["geometry_derivatives"]
        assert result["converged"]
        assert result["added_particle_coordinate_order"] == [0]
        assert result["hessian_shape"] == [9, 9]
        assert np.asarray(result["gradient"]).shape == (3, 3)
        for key, power in (("gradient", 1), ("hessian", 2)):
            scale = Bohr**power/Hartree if report is physical else 1.
            np.testing.assert_allclose(np.asarray(result[key])*scale, reference[key], rtol=0., atol=1e-8)
    response = np.asarray(reference["response_residuals"])
    assert np.isfinite(response).all() and (response < 1e-9).all()
    print(f"H2+positron energy {native['energy']:.16e}; Gaussian gradient {reference['gradient'][-1]}; "
          f"Gaussian Hessian {np.asarray(reference['hessian']).reshape(9, 9)[-3:, -3:]}; "
          f"maximum coupled residual {response.max():.6e}")


@pytest.mark.parametrize("constrained", [False, True])
def test_nonrelativistic_joint_field_gradient_public_routes(tmp_path, constrained):
    field = np.array([0.0006, -0.0004, 0.0003])
    gradient = np.array([0.0009, -0.0007, 0.0012, 0.0005, -0.0008])
    origin = np.array([0.11, -0.27, 0.19])
    particle = dict(kind="positron", position=[.18, -.13, .29],
                    exponents=[.22, .64], lmax=1, correlation="drummond-2011")
    controls = dict(xc="pbe", basis="sto-3g", grid_level=0, grid_weight_cutoff=0.,
                    cpu_threads=1, field=field.tolist(),
                    species_field_gradient=gradient.tolist(), origin=origin.tolist(),
                    particles=[particle], species_quadrupole_polarizability=True,
                    quantum_position_constraint=constrained, species_tolerance=1e-10)
    atoms = Atoms("H2", positions=np.asarray(POSITIONS)*Bohr)
    native = dft_rs.multicomponent([1, 1], POSITIONS, **controls)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **controls)
    cli = cli_report(tmp_path, "multicomponent", atoms, controls)
    physical_particle = dict(particle, position=np.asarray(particle["position"])*Bohr,
                             exponents=np.asarray(particle["exponents"])/Bohr**2)
    physical = dft_rs.DftRs().get_multicomponent(atoms, **dict(
        controls, origin=origin*Bohr, field=field*Hartree/Bohr,
        species_field_gradient=gradient*Hartree/Bohr**2, particles=[physical_particle]))
    for candidate in (native, atomic, cli, physical):
        assert candidate["converged"]
        factor = Hartree if candidate is physical else 1.
        length = Bohr if candidate is physical else 1.
        np.testing.assert_allclose(candidate["energy"]/factor, native["energy"], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(candidate["field_gradient_origin"], origin*length, atol=1e-13, rtol=0.)
        np.testing.assert_allclose(candidate["field_gradient"], gradient*factor/length**2,
                                   atol=1e-13, rtol=0.)
        for key, length_power, energy_power in (("moment", 2, 0), ("dipole_moment", 1, 0),
                                                ("polarizability", 4, -1)):
            scale = length**length_power*factor**energy_power
            np.testing.assert_allclose(np.asarray(candidate["quadrupole"][key])/scale,
                                       native["quadrupole"][key], atol=1e-8, rtol=0.)
    print(f"H2 with positron and finite electric field gradient, constrained={constrained}: "
          f"E={native['energy']:.15e}, Q={native['quadrupole']['moment']}, "
          f"C00={native['quadrupole']['polarizability'][0][0]:.12e}, "
          f"residual={max(native['quadrupole']['response_residuals']):.6e}")


def test_positron_point_oracle_reproduces_from_independent_decimal_generator():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-W", "error",
                             str(root/"tools/oracle/gen_positron_relaxation.py")],
                            capture_output=True, text=True, check=True)
    assert result.stderr == ""
    expected = (root/"tests/oracle_data/positron_relaxation.csv").read_text(encoding="utf-8")
    assert result.stdout == expected


def test_c1_epc17_gradient_explicit_reference_acceptance_all_routes(tmp_path):
    """A complete Cartesian FD witness supports this directional gradient, not a Hessian claim."""
    positions = [[.17, -.11, .23], [1.43, .81, -.62],
                 [-.63, 1.59, -.43], [-1.12, -1.36, -.71]]
    direction = [[.31, -.27, .19], [-.11, .17, .23],
                 [.13, .07, -.29], [-.17, .09, .11]]
    atoms = Atoms("NH3", positions=np.asarray(positions)*Bohr)
    options = dict(xc="hf", basis="sto-3g", dispersion="none", quantum_atoms=[1],
                   quantum_exponents=[4., 8.], quantum_lmax=1,
                   interspecies_correlation="epc17-1", species_derivative_order=1,
                   species_direction=direction, field=[.0013, -.0009, .0017],
                   species_tolerance=1e-13, species_max_iterations=500, species_mixing=.5,
                   conv_tol=1e-14, conv_tol_grad=1e-14, conv_tol_dp=1e-14, max_iter=500,
                   response_tolerance=1e-13, response_accept_residual=1e-11,
                   grid_level=2, grid_weight_cutoff=0., cpu_threads=1, memory_mb=512)
    reports = {
        "native": dft_rs.multicomponent([7, 1, 1, 1], positions, **options),
        "ASE_atomic": dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options),
        "CLI": cli_report(tmp_path, "multicomponent", atoms, options),
    }
    physical_options = dict(options, field=np.asarray(options["field"])*Hartree/Bohr,
                            quantum_exponents=np.asarray(options["quantum_exponents"])/Bohr**2,
                            conv_tol=options["conv_tol"]*Hartree,
                            conv_tol_grad=options["conv_tol_grad"]*Hartree)
    reports["ASE_physical"] = dft_rs.DftRs().get_multicomponent(atoms, **physical_options)
    for surface, report in reports.items():
        physical = surface == "ASE_physical"
        assert report["converged"] and report["experimental"]
        assert report["electronic_orbital_residual"] < 1e-11
        assert report["quantum"][0]["orbital_residual"] < 1e-11
        assert abs(report["quantum"][0]["particle_number"]-1.) < 1e-11
        derivative = report["geometry_derivatives"]
        assert derivative["converged"] and "hessian" not in derivative
        assert derivative["response_residual"] < 1e-11
        assert derivative["reference_residual"] < 1e-11
        energy = report["energy"]/Hartree if physical else report["energy"]
        gradient = derivative["gradient"]/(Hartree/Bohr) if physical else derivative["gradient"]
        np.testing.assert_allclose(energy, -55.401961000564256, atol=1e-10, rtol=0.)
        np.testing.assert_allclose(gradient, .10625710981573015, atol=1e-9, rtol=0.)
        for key in ("gradient", "reference_energy_error"):
            assert "geometry_derivatives."+key in report["units"]
