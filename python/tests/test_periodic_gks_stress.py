"""Exact-artifact periodic stress routing, raw tensors, units and exclusions."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from ase import Atoms
import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


def fixture(tmp_path):
    basis = tmp_path / "synthetic.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\nHe 0\nS 1 1.0\n1.1 1.0\n****\n"
                     "Li 0\nS 1 1.0\n4.0 1.0\nS 1 1.0\n0.65 1.0\n****\n")
    return [1, 2], [[0.13, -0.21, 0.07], [2.0, 0.45, -0.31]], [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]], basis


def parameters(basis, reference="gks", xc="hf"):
    return dict(reference=reference, xc=xc, basis=str(basis), dispersion="none",
                multiplicity=2, k_mesh=[1, 1, 1], overlap_cutoff=3.5,
                exchange_cutoff=3.5, xc_ao_cutoff=12.0, grid_level=4,
                grid_cutoff=6.0, grid_weight_cutoff=0.0, ewald_alpha=0.5,
                ewald_real_cutoff=12.0, ewald_reciprocal_cutoff=7.0,
                electrostatic_order=16, guess="core", conv_tol=1e-12,
                conv_tol_grad=1e-11, conv_tol_dp=1e-11, max_iter=300,
                damping=0.2, no_adiis=True, cpu_threads=1, memory_mb=512,
                nuclear_soc_scale=0.0, stress=True, forces=False)


def check(report, reference):
    assert report["reference_type"] == reference and report["converged"]
    assert "same-gauge" in report["stress_method"]
    tensor = np.asarray(report["strain_derivative"])
    assert tensor.shape == (3, 3) and np.isfinite(tensor).all()
    np.testing.assert_allclose(sum(np.asarray(v) for v in report["strain_components"].values()), tensor, rtol=0, atol=1e-12)
    # The report's geometry field may use the common structured cell schema;
    # the explicit input fixture is the independent volume reference.
    volume = abs(np.linalg.det(np.array([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]])))
    np.testing.assert_allclose(np.asarray(report["stress"]), tensor / volume, rtol=0, atol=1e-13)
    assert abs(report["pressure"] + np.trace(report["stress"]) / 3) < 1e-13
    assert report["units"]["stress"] == "hartree/bohr^3"
    assert report["units"]["strain_derivative"] == "hartree"
    diagnostics = report["stress_diagnostics"]
    for name, bound in [("orbital_residual", 1e-11), ("occupation_consistency", 1e-11),
                        ("number_error", 1e-11), ("reference_energy_defect", 1e-12),
                        ("reference_fock_defect", 1e-11), ("xc_reference_energy_defect", 1e-12),
                        ("coulomb_reference_energy_defect", 1e-10), ("coulomb_quadrature_defect", 1e-10)]:
        assert diagnostics[name] <= bound


def cli_command(executable, tmp_path, numbers, positions, cell, values, feature="periodic-energy", pbc=(True, True, True)):
    xyz = tmp_path / "cell.xyz"
    symbols = {1: "H", 2: "He", 3: "Li", 11: "Na"}
    xyz.write_text(f"{len(numbers)}\nPeriodic stress interface fixture\n" + "\n".join(symbols[z] + " " + " ".join(str(x * BOHR_IN_ANGSTROM) for x in row) for z, row in zip(numbers, positions)) + "\n")
    rows = ";".join(",".join(str(x * BOHR_IN_ANGSTROM) for x in row) for row in cell)
    mask = ",".join(str(int(axis)) for axis in pbc)
    command = [executable, feature, str(xyz), "--cell", rows, "--pbc", mask, "--format", "json"]
    for key, value in values.items():
        command.append("--" + key.replace("_", "-"))
        if key == "field":
            command.append(",".join(str(component) for component in value))
        elif not (isinstance(value, bool) and value):
            command.append(json.dumps(value) if isinstance(value, (list, bool)) else str(value))
    return command


@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_vv10_finite_k_stress_and_forces_use_public_routes(tmp_path, reference):
    numbers = [11, 1]
    positions = [[0.13, -0.21, 0.07], [2.21, -1.28, 1.61]]
    cell = [[9.4, 0.71, 0.33], [0.45, 10.8, 0.69], [0.24, -0.55, 11.2]]
    basis = Path(__file__).resolve().parents[2] / "tests/data/periodic_scalar_ecp.g94"
    controls = dict(xc="wb97x-v", reference=reference, basis=str(basis),
                    k_mesh=[3, 1, 1], overlap_cutoff=5.5, exchange_cutoff=5.5,
                    xc_ao_cutoff=9.0, vv10_cutoff=9.0, grid_level=0, grid_shell=1,
                    grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_dp=1e-11,
                    conv_tol_grad=1e-11)
    native = dft_rs.stress(numbers, positions, cell, forces=True, **controls)
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell) * BOHR_IN_ANGSTROM, pbc=True)
    ase = dft_rs.DftRs().get_feature_report("stress", atoms, forces=True, **controls)
    proc = subprocess.run(cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions,
                                      cell, dict(controls, forces=True), feature="stress"),
                          check=True, capture_output=True, text=True)
    cli = json.loads(proc.stdout)
    for report in (native, cli, ase):
        assert report["converged"] and report["reference_type"] == reference
        assert abs(report["energy_components"]["nonlocal_correlation"]) > 1e-8
        assert "analytic" in report["stress_provenance"]
    for key in ("energy", "forces", "stress", "orbital_residual", "strain_derivative"):
        np.testing.assert_allclose(cli[key], native[key], rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(ase["stress"], np.asarray(native["stress"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM**3)
    np.testing.assert_allclose(ase["forces"], np.asarray(native["forces"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM)
    np.testing.assert_allclose(np.sum(native["forces"], axis=0), 0, atol=1e-7)
    assert "[pbc-scf] iter" in proc.stderr and "residual" in proc.stderr


@pytest.mark.parametrize("reference, model", [
    ("rks", "charged"), ("uks", "charged"), ("rks", "scalar_ecp"), ("rks", "finite_bias"),
])
def test_double_hybrid_stress_routes(tmp_path, reference, model):
    numbers = [2, 1]
    positions = np.array([[0.17, 0.41, -0.19], [0.49, -0.12, 1.19]])
    cell = np.array([[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]])
    populations = (1, 1)
    if reference == "uks":
        populations = (2, 1)
        numbers = [1, 1, 1, 1]
        positions = np.array([[0.17, 0.41, -0.19], [0.49, -0.12, 1.19],
                              [1.71, 0.28, 0.33], [1.39, 1.62, 1.47]])
    controls = dict(
        xc="b2plyp", reference=reference, charge=1, basis="sto-3g", dispersion="none",
        forces=False, k_mesh=[1, 1, 1], overlap_cutoff=5.5, xc_ao_cutoff=16.0, grid_level=1,
        grid_shell=1, grid_weight_cutoff=0.0, conv_tol=1e-12, conv_tol_dp=1e-11,
        conv_tol_grad=1e-11, response_tolerance=1e-11,
    )
    if reference == "uks":
        controls.update(multiplicity=2, max_iter=500)
    if model == "scalar_ecp":
        numbers = [1, 11, 1]
        positions = np.array([[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]])
        basis = Path(__file__).resolve().parents[2] / "tests/data/periodic_scalar_ecp.g94"
        controls.update(basis=str(basis), forces=True)
    elif model == "finite_bias":
        populations = (2, 2)
        numbers = [3, 1]
        positions[1] = [1.09, -0.82, 2.45]
        basis = Path(__file__).resolve().parents[2] / "tests/data/periodic_lih.g94"
        controls.update(charge=0, basis=str(basis), field=[0.0013, -0.0021, 0.0017])
    native = dft_rs.stress(numbers, positions.tolist(), cell.tolist(), **controls)
    atoms = Atoms(numbers, positions=positions * BOHR_IN_ANGSTROM,
                  cell=cell * BOHR_IN_ANGSTROM, pbc=True)
    atomic = dft_rs.DftRs().get_periodic_energy_report(atoms, units="atomic",
                                                      stress=True, **controls)
    process = subprocess.run(
        cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                    controls, feature="stress"), text=True, capture_output=True, check=True,
    )
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    for report in (native, atomic, json.loads(process.stdout)):
        assert report["converged"] and report["double_hybrid_model"] == "b2plyp"
        assert report["reference_type"] == reference
        assert (report["n_alpha"], report["n_beta"]) == populations
        assert report["stress_provenance"] == "analytic"
        if model == "finite_bias":
            assert report["stress_field_convention"] == "fixed Cartesian Berry electric field"
            assert report["units"]["reference_field"] == "hartree/e/bohr"
            np.testing.assert_array_equal(report["reference_field"], controls["field"])
        else:
            assert report["charge_compensation"] == "uniform 3D Ewald background"
            for key in ("primitive_cell_charge", "born_von_karman_charge"):
                assert report[key] == 1 and report["units"][key] == "e"
        assert report["units"]["stress"].startswith("hartree/bohr^3")
        for key in ("energy", "stress", "strain_derivative", "stress_voigt",
                    "series_reference_residual", "amplitude_response_residuals",
                    "orbital_residual"):
            np.testing.assert_allclose(report[key], native[key], rtol=1e-9, atol=2e-9)
        if model == "scalar_ecp":
            assert report["units"]["forces"] == "hartree/bohr"
            np.testing.assert_allclose(report["forces"], native["forces"], rtol=0, atol=2e-9)
        terms = report["strain_derivative_terms"]
        assert terms.keys() == native["strain_derivative_terms"].keys()
        for name, value in terms.items():
            np.testing.assert_allclose(value, native["strain_derivative_terms"][name], rtol=1e-9, atol=2e-9)
        assert np.max(np.abs(terms["pt2_same_spin_beta"])) < 1e-15
        alpha_same_spin = np.max(np.abs(terms["pt2_same_spin_alpha"]))
        if reference == "uks":
            assert alpha_same_spin > 1e-12
        else:
            assert alpha_same_spin < 1e-15
        strain = np.asarray(report["strain_derivative"])
        np.testing.assert_allclose(sum(np.asarray(v) for v in report["strain_derivative_terms"].values()),
                                   strain, rtol=0, atol=1e-12)
        np.testing.assert_allclose(report["stress"], (strain + strain.T) / (2 * abs(np.linalg.det(cell))),
                                   rtol=0, atol=1e-12)
    physical_controls = dict(controls, overlap_cutoff=controls["overlap_cutoff"] * BOHR_IN_ANGSTROM,
                             xc_ao_cutoff=controls["xc_ao_cutoff"] * BOHR_IN_ANGSTROM,
                             conv_tol=controls["conv_tol"] * HARTREE_IN_EV,
                             conv_tol_grad=controls["conv_tol_grad"] * HARTREE_IN_EV)
    if model == "finite_bias":
        physical_controls["field"] = [v * HARTREE_IN_EV / BOHR_IN_ANGSTROM for v in controls["field"]]
    atoms.calc = dft_rs.DftRs(units="ase", **{k: v for k, v in physical_controls.items() if k != "forces"})
    np.testing.assert_allclose(atoms.get_stress(), np.asarray(native["stress_voigt"]) *
                               HARTREE_IN_EV / BOHR_IN_ANGSTROM**3, rtol=1e-9, atol=2e-8)
    keys = ("energy", "stress", "strain_derivative_terms", "series_reference_residual",
            "amplitude_response_residuals", "orbital_residual")
    print("Installed double-hybrid stress", json.dumps(dict(reference=reference, fixture=model, **{key: native[key] for key in keys})))


@pytest.mark.parametrize("reference,xc", [("gks", "hf"), ("ghf", "hf"), ("gks", "svwn5"), ("gks", "pbe")])
def test_actual_stress_native_ase_and_both_clis(reference, xc, tmp_path):
    numbers, positions, cell, basis = fixture(tmp_path)
    values = parameters(basis, reference, xc)
    native = dft_rs.native.run_feature("periodic-energy", numbers, positions, cell=cell, **values)
    reports = {"native": native}
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM, cell=np.asarray(cell) * BOHR_IN_ANGSTROM, pbc=True)
    calculator = dft_rs.DftRs(units="atomic", damping=0.2, use_adiis=False, adiis_until_error=None)
    reports["ASE"] = calculator.get_periodic_energy_report(atoms, **values)
    cli_labels = ["DFT_RS_CLI"]
    if os.environ.get("DFT_RS_CLI_ALIAS"):
        cli_labels.append("DFT_RS_CLI_ALIAS")
    direct = reference == "gks" and xc == "hf"
    direct_values = {key: value for key, value in values.items() if key != "stress"}
    if direct:
        reports["native-stress"] = dft_rs.stress(numbers, positions, cell, **direct_values)
        reports["ASE-stress"] = calculator.get_feature_report("stress", atoms, **direct_values)
    for label in cli_labels:
        process = subprocess.run(cli_command(os.environ[label], tmp_path, numbers, positions, cell, values), capture_output=True, text=True, check=True)
        assert any(key in process.stderr.lower() for key in ("energy", "e="))
        assert any(key in process.stderr.lower() for key in ("residual", "orbital", "density"))
        reports[label] = json.loads(process.stdout)
        if direct:
            process = subprocess.run(
                cli_command(os.environ[label], tmp_path, numbers, positions, cell, direct_values, "stress"),
                capture_output=True, text=True, check=True,
            )
            reports[f"{label}-stress"] = json.loads(process.stdout)
    raw_dir = Path(os.environ.get("DFT_RS_PUBLIC_RAW_DIR", tmp_path))
    for label, report in reports.items():
        check(report, reference)
        np.testing.assert_allclose(report["strain_derivative"], native["strain_derivative"], rtol=0, atol=1e-10)
        assert abs(report["energy"] - native["energy"]) < 1e-10
        (raw_dir / f"{reference}-{xc}-{label}.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("change", [dict(nuclear_soc_scale=1.0), dict(spin_field_real=[0.1, 0.0, 0.0]), dict(uniform_magnetic_field=[0.001, 0.0, 0.0]), dict(xc="b2plyp")])
def test_stress_does_not_substitute_unsupported_parent(change, tmp_path):
    numbers, positions, cell, _ = fixture(tmp_path)
    values = parameters(tmp_path / "deliberately-absent.g94")
    values.update(change)
    with pytest.raises(NotImplementedError):
        dft_rs.native.run_feature("periodic-energy", numbers, positions, cell=cell, **values)


@pytest.mark.parametrize("second_term", ["hf", "svwn5"])
def test_weighted_stress_uses_the_installed_cli_and_standard_ase_calculation(second_term, tmp_path):
    numbers, positions, cell, basis = fixture(tmp_path)
    values = parameters(basis)
    values.pop("xc")
    values.pop("stress")
    values["xc_terms"] = [{"xc": "hf", "weight": 0.65}, {"xc": second_term, "weight": 0.35}]
    native = dft_rs.stress(numbers, positions, cell, **values)
    check(native, "gks")
    process = subprocess.run(
        cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell, values, "stress"),
        capture_output=True, text=True, check=True,
    )
    assert "energy" in process.stderr.lower()
    assert "residual" in process.stderr.lower()
    cli = json.loads(process.stdout)
    check(cli, "gks")
    np.testing.assert_allclose(cli["strain_derivative"], native["strain_derivative"], rtol=0, atol=1e-10)
    assert abs(cli["energy"] - native["energy"]) < 1e-10
    assert abs(cli["orbital_residual"] - native["orbital_residual"]) < 1e-11

    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell) * BOHR_IN_ANGSTROM, pbc=True)
    constructor = {key: value for key, value in values.items() if key not in ("forces", "no_adiis")}
    atoms.calc = dft_rs.DftRs(units="atomic", use_adiis=False, adiis_until_error=None, **constructor)
    ase_report = atoms.calc.get_feature_report("stress", atoms, forces=False)
    check(ase_report, "gks")
    assert abs(ase_report["orbital_residual"] - native["orbital_residual"]) < 1e-11
    np.testing.assert_allclose(ase_report["strain_derivative"], native["strain_derivative"], rtol=0, atol=1e-10)
    assert abs(ase_report["energy"] - native["energy"]) < 1e-10
    np.testing.assert_allclose(atoms.get_stress(), np.asarray(native["stress_voigt"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM**3,
                               rtol=0, atol=1e-9)
    for name, report in (("native", native), ("CLI", cli), ("ASE", ase_report)):
        print(f"Weighted HF/{second_term} {name} energy={report['energy']:.17e} "
              f"orbital={report['orbital_residual']:.17e} "
              f"strain={json.dumps(report['strain_derivative'])}")



@pytest.mark.parametrize("pbc", [(True, False, False), (True, False, True)], ids=["wire", "slab"])
def test_dimensional_gks_stress_uses_active_measure_on_public_routes(pbc, tmp_path):
    check_dimensional_stress_routes("gks", pbc, tmp_path)


@pytest.mark.parametrize("pbc", [(True, False, False), (True, False, True)], ids=["wire", "slab"])
@pytest.mark.parametrize("reference", ["rks", "uks"])
def test_dimensional_collinear_stress_uses_active_measure_on_public_routes(reference, pbc, tmp_path):
    check_dimensional_forces_and_stress(reference, pbc, tmp_path)


@pytest.mark.parametrize("reference,pbc,xc", [
    ("rks", (True, False, False), "wb97x-v"),
    ("uks", (True, False, True), "wb97m-v"),
], ids=["wire", "slab"])
def test_dimensional_vv10_stress_and_forces_use_public_routes(reference, pbc, xc, tmp_path):
    check_dimensional_stress_routes(reference, pbc, tmp_path, vv10_xc=xc)


@pytest.mark.parametrize("pbc,xc,mesh", [
    ((True, False, False), "wb97x-v", [3, 1, 1]),
    ((True, False, True), "wb97m-v", [1, 1, 1]),
], ids=["wire", "slab"])
def test_dimensional_roks_vv10_on_public_routes(pbc, xc, mesh, tmp_path):
    check_dimensional_stress_routes("roks", pbc, tmp_path, mesh, vv10_xc=xc)


@pytest.mark.parametrize("pbc,xc", [
    ((True, False, False), "wb97x-v"),
    ((True, False, True), "wb97m-v"),
    ((True, True, True), "wb97x-v"),
], ids=["wire", "slab", "bulk"])
def test_periodic_gks_vv10_on_public_routes(pbc, xc, tmp_path):
    check_dimensional_stress_routes("gks", pbc, tmp_path, [2, 1, 1], vv10_xc=xc)


def test_periodic_d1_wire_on_public_routes(tmp_path):
    _, positions, cell, basis = fixture(tmp_path)
    numbers = [1, 3]
    pbc = (True, False, False)
    cell = np.asarray(cell)
    cell[0] *= 0.55
    values = dict(reference="rks", multiplicity=1, xc="pbe", basis=str(basis),
                  k_mesh=[2, 1, 1], grid_level=0, grid_weight_cutoff=0.0,
                  overlap_cutoff=3.8, exchange_cutoff=3.8, xc_ao_cutoff=12.0,
                  electrostatic_order=48, grid_tolerance=0.1, cpu_threads=1)
    correction = dict(dispersion="d1", disp_cutoff=6.4, dispersion_free_c6=[6.5, 35.0],
                      dispersion_free_radii=[2.8, 3.1], dispersion_energy_scale=0.81)
    reference = dft_rs.stress(numbers, positions, cell.tolist(), pbc=list(pbc), **values)
    native = dft_rs.stress(numbers, positions, cell.tolist(), pbc=list(pbc),
                           **values, **correction)
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=cell * BOHR_IN_ANGSTROM, pbc=pbc)
    ase = dft_rs.DftRs(units="atomic").get_periodic_stress_report(atoms, **values, **correction)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                          dict(values, **correction), feature="stress", pbc=pbc)
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "[pbc-scf]" in process.stderr and "residual" in process.stderr
    for report in (native, ase, json.loads(process.stdout)):
        assert report["converged"] and report["reference_type"] == "rks"
        assert report["dispersion_model"] == "D1" and report["dispersion_energy"] < 0.0
        assert report["units"]["dispersion_energy"] == "hartree per cell"
        assert report["units"]["stress"] == "hartree/bohr"
        for key in ("energy", "dispersion_energy", "forces", "stress", "orbital_residual"):
            np.testing.assert_allclose(report[key], native[key], rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(report["energy"] - reference["energy"],
                                   report["dispersion_energy"], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(report["forces"]) - reference["forces"],
                                   -np.asarray(report["gradient_components"]["dispersion"]),
                                   rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(report["strain_derivative"]) - reference["strain_derivative"],
                                   report["strain_components"]["dispersion"], rtol=0.0, atol=1e-12)
    print(f"Periodic D1 energy={native['energy']:.17e} dispersion={native['dispersion_energy']:.17e} "
          f"stress_xx={native['stress'][0][0]:.17e} residual={native['orbital_residual']:.17e}")


@pytest.mark.parametrize("reference,pbc,mesh", [
    ("roks", (True, True, True), [1, 1, 1]),
    ("roks", (True, True, True), [2, 1, 1]),
    ("roks", (True, False, False), [2, 1, 1]),
    ("roks", (True, False, True), [2, 1, 1]),
    ("rks", (True, False, False), [2, 1, 1]),
    ("rks", (True, False, True), [2, 1, 1]),
    ("uks", (True, False, False), [2, 1, 1]),
    ("uks", (True, False, True), [2, 1, 1]),
    ("gks", (True, False, False), [2, 1, 1]),
    ("gks", (True, False, True), [2, 1, 1]),
    ("gks", (True, True, True), [2, 1, 1]),
], ids=["roks-bulk-gamma", "roks-bulk-finite-k", "roks-wire", "roks-slab",
        "rks-wire", "rks-slab", "uks-wire", "uks-slab", "gks-wire", "gks-slab", "gks-bulk"])
def test_periodic_chg_on_public_routes(reference, pbc, mesh, tmp_path):
    _exercise_named_periodic_pair_dispersion("wb97x-d", reference, pbc, mesh, tmp_path)


@pytest.mark.parametrize("reference,pbc,mesh", [
    ("rks", (True, False, False), [2, 1, 1]),
    ("uks", (True, False, True), [2, 1, 1]),
    ("roks", (True, True, True), [1, 1, 1]),
    ("gks", (True, True, True), [2, 1, 1]),
], ids=["rks-wire", "uks-slab", "roks-bulk-gamma", "gks-bulk-finite-k"])
def test_periodic_wb97x_d3_on_public_routes(reference, pbc, mesh, tmp_path):
    _exercise_named_periodic_pair_dispersion("wb97x-d3", reference, pbc, mesh, tmp_path)


def _exercise_named_periodic_pair_dispersion(xc, reference, pbc, mesh, tmp_path):
    basis = tmp_path / "periodic-chg.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n"
                     "Li 0\nS 1 1.0\n4.0 1.0\nS 1 1.0\n0.65 1.0\n****\n"
                     "He 0\nS 1 1.0\n1.1 1.0\n****\n",
                     encoding="ascii")
    numbers = {"roks": [1, 3, 1], "rks": [1, 3, 2], "uks": [1, 2, 2], "gks": [1, 2, 2]}[reference]
    positions = [[0.13, -0.21, 0.07], [2.0, 0.45, -0.31], [-0.46, 1.29, 1.62]]
    cell = [[8.4, 0.7, 0.3], [0.4, 8.9, 0.6], [0.2, -0.5, 9.2]]
    values = dict(reference=reference, multiplicity=1 if reference == "rks" else 2,
                  xc=xc, basis=str(basis),
                  k_mesh=mesh, grid_level=0, grid_shell=1, grid_weight_cutoff=0.0,
                  overlap_cutoff=3.8, exchange_cutoff=3.8, xc_ao_cutoff=12.0,
                  disp_cutoff=5.4, cpu_threads=1, damping=0.0,
                  conv_tol=1e-13, conv_tol_dp=1e-11, conv_tol_grad=1e-10)
    if xc == "wb97x-d3":
        values["cn_cutoff"] = 5.4
    if reference == "gks" or not all(pbc):
        cell = [[value * (0.55 if active else 1.0) for value in row]
                for row, active in zip(cell, pbc)]
        values.pop("grid_shell")
        values["grid_tolerance"] = 0.1
    if reference == "gks":
        values.update(damping=0.2, nuclear_soc_scale=0.0, no_adiis=True,
                      ewald_alpha=0.5, ewald_real_cutoff=12.0, ewald_reciprocal_cutoff=7.0,
                      grid_cutoff=6.0, electrostatic_order=16 if all(pbc) else 48, max_iter=300)
    native = dft_rs.stress(numbers, positions, cell, pbc=list(pbc), **values)
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell) * BOHR_IN_ANGSTROM, pbc=pbc)
    calculator = dft_rs.DftRs(units="atomic", use_adiis=False, adiis_until_error=None)
    ase = calculator.get_periodic_stress_report(atoms, **values)
    args = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell,
                       values, feature="stress", pbc=pbc)
    process = subprocess.run(args, capture_output=True, text=True, check=True)
    assert "residual" in process.stderr
    log_prefix = {"roks": "[pbc-roks]", "gks": "[spinor-scf]"}.get(reference, "[pbc-scf]")
    assert log_prefix in process.stderr
    for report in (native, ase, json.loads(process.stdout)):
        assert report["converged"] and report["reference_type"] == reference
        assert report["dispersion_energy"] < 0.0
        assert report["units"] == native["units"]
        assert report["units"]["dispersion_energy"] == "hartree per cell"
        if reference == "gks" or not all(pbc):
            assert report["units"]["strain_components.dispersion"] == "hartree per cell"
            assert report["units"]["gradient_components.dispersion"] == "hartree/bohr"
            unit = {1: "hartree/bohr", 2: "hartree/bohr^2", 3: "hartree/bohr^3"}[sum(pbc)]
            assert report["units"]["stress"] == unit
            active = np.asarray(cell)[np.asarray(pbc)]
            projector = active.T @ np.linalg.solve(active @ active.T, active)
            pressure = -np.sum(projector * np.asarray(report["stress"])) / sum(pbc)
            np.testing.assert_allclose(report["pressure"], pressure, rtol=0.0, atol=1e-12)
        for key in ("energy", "dispersion_energy", "strain_derivative", "forces",
                    "stress", "pressure", "orbital_residual"):
            np.testing.assert_allclose(report[key], native[key], rtol=0.0, atol=1e-10)
        if reference == "roks":
            reconstructed = (report["electronic_energy"] + report["nuclear_energy"]
                             + report["dispersion_energy"])
        else:
            reconstructed = sum(report["energy_components"].values())
        np.testing.assert_allclose(report["energy"], reconstructed, rtol=0.0, atol=1e-12)
    if mesh == [1, 1, 1]:
        optimized = calculator.get_opt_fixed_cell(atoms, max_steps=1, gtol=1e-7, **values)
        assert optimized["objective"] < native["energy"]
        np.testing.assert_allclose(optimized["energy"], optimized["objective"],
                                   rtol=0.0, atol=1e-12)
        assert optimized["dispersion_energy"] < 0.0
    spectrum = calculator.get_dos(atoms, sigma=0.04, points=101, **values)
    np.testing.assert_allclose(spectrum["energy"], native["energy"], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(spectrum["dispersion_energy"], native["dispersion_energy"],
                               rtol=0.0, atol=1e-12)
    states = 6.0 if reference in ("uks", "gks") else 8.0
    assert abs(sum(channel["state_count"] for channel in spectrum["channels"]) - states) < 1e-12
    print(f"{reference.upper()} {xc} {mesh}: energy={native['energy']:.17e} correction="
          f"{native['dispersion_energy']:.17e} stress_xx={native['stress'][0][0]:.17e} "
          f"residual={native['orbital_residual']:.17e}")


@pytest.mark.parametrize("pbc", [(True, False, False), (True, False, True)], ids=["wire", "slab"])
@pytest.mark.parametrize("mesh", [[1, 1, 1], [3, 1, 1]], ids=["gamma", "mesh"])
def test_dimensional_roks_stress_uses_active_measure_on_public_routes(pbc, mesh, tmp_path):
    check_dimensional_forces_and_stress("roks", pbc, tmp_path, mesh)


@pytest.mark.parametrize("reference,pbc", [("uks", (True, False, False)), ("gks", (True, False, True))])
def test_dimensional_scalar_ecp_stress_on_public_routes(reference, pbc, tmp_path):
    check_dimensional_forces_and_stress(reference, pbc, tmp_path, scalar_ecp=True)


@pytest.mark.parametrize("reference,pbc", [("gks", (True, False, False)), ("roks", (True, False, True))])
def test_dimensional_dos_on_public_routes(reference, pbc, tmp_path):
    numbers, positions, cell, basis = fixture(tmp_path)
    values = parameters(basis, reference=reference)
    values.pop("stress")
    values.pop("forces")
    if reference == "roks":
        values.pop("nuclear_soc_scale")
        values["damping"] = 0.0
    values.update(k_mesh=[3, 1, 1], electrostatic_order=48, sigma=0.06, points=1201,
                  energy_window=[-8.0, 8.0], projected=True)
    native = dft_rs.dos(numbers, positions, cell, pbc=list(pbc), **values)
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=np.asarray(cell) * BOHR_IN_ANGSTROM, pbc=pbc)
    ase = dft_rs.DftRs(units="atomic").get_dos(atoms, **values)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell, values,
                          feature="dos", pbc=pbc)
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    for report in (native, ase, json.loads(process.stdout)):
        assert report["converged"] and report["reference_type"] == reference
        assert report["sampling_k_points"] == 3
        assert report["units"]["channels.state_count"] == "states per cell"
        assert abs(sum(c["state_count"] for c in report["channels"]) - 4.0) < 1e-12
        np.testing.assert_allclose(report["energies"], native["energies"], rtol=0, atol=1e-13)
        assert len(report["channels"]) == len(native["channels"])
        for current, expected in zip(report["channels"], native["channels"]):
            assert abs(current["integration_residual"]) < 1e-10
            assert current["projection_sum_residual"] < 1e-10
            np.testing.assert_allclose(current["dos"], expected["dos"], rtol=0, atol=1e-8)
            np.testing.assert_allclose(current["projected_atoms"], expected["projected_atoms"], rtol=0, atol=1e-8)
    print(f"Installed DOS {reference} energy={native['energy']:.17e} states=4")


def check_dimensional_forces_and_stress(reference, pbc, tmp_path, mesh=None, scalar_ecp=False):
    native, atoms, controls = check_dimensional_stress_routes(reference, pbc, tmp_path, mesh, scalar_ecp)
    constructor = {key: value for key, value in controls.items() if key not in ("no_adiis", "forces", "stress")}
    atoms.calc = dft_rs.DftRs(units="atomic", use_adiis=False, adiis_until_error=None, **constructor)
    combined = atoms.calc.get_periodic_energy_report(atoms, forces=True, stress=True)
    assert combined["reference_type"] == reference and combined["converged"]
    assert abs(combined["energy"] - native["energy"]) < 1e-10
    np.testing.assert_allclose(combined["strain_derivative"], native["strain_derivative"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(atoms.get_forces(), np.asarray(combined["forces"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM,
                               rtol=0, atol=1e-9)
    if scalar_ecp:
        ecp = np.asarray(combined["gradient_components"]["ecp"])
        assert np.max(np.abs(ecp)) > 1e-6
        np.testing.assert_allclose(ecp.sum(axis=0), 0.0, rtol=0, atol=1e-12)


def test_dimensional_fourier_field_stress_on_public_routes(tmp_path):
    pbc = [True, False, False]
    native, atoms, controls = check_dimensional_stress_routes("uks", pbc, tmp_path, fourier_field=True)
    dedicated = dict(controls, forces=True, stress=True, unrestricted=True)
    dedicated.pop("reference")
    both = dft_rs.periodic_field_scf(
        atoms.numbers, atoms.positions / BOHR_IN_ANGSTROM,
        atoms.cell.array / BOHR_IN_ANGSTROM, pbc=pbc, **dedicated,
    )
    combined = dft_rs.periodic_energy(
        atoms.numbers, atoms.positions / BOHR_IN_ANGSTROM,
        atoms.cell.array / BOHR_IN_ANGSTROM, pbc=pbc, **controls,
    )
    for report in (both, combined):
        assert report["converged"] and report["reference_type"] == "uks"
        assert abs(report["energy"] - native["energy"]) < 1e-10
        for key in ("stress", "forces", "strain_derivative"):
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-10)

    # A zero-wave scalar source shifts the energy by the particle count.
    # Its overlap derivative cancels the change of the Pulay contribution.
    scalar = dict(controls, field_wave_indices=[0, 0, 0], field_scalar=[0.025, 0.031],
                  field_vector_real=[0.0] * 3, field_vector_imag=[0.0] * 3)
    geometry = (atoms.numbers, atoms.positions / BOHR_IN_ANGSTROM,
                atoms.cell.array / BOHR_IN_ANGSTROM)
    shifted = dft_rs.stress(*geometry, pbc=pbc, **scalar)
    zero = dft_rs.stress(*geometry, pbc=pbc, **dict(scalar, field_scalar=[0.0, 0.0]))
    constant_energy = 2 * scalar["field_amplitude"] * scalar["field_scalar"][0] * sum(atoms.numbers)
    np.testing.assert_allclose(shifted["energy"] - zero["energy"], constant_energy, rtol=0, atol=1e-10)
    for key in ("forces", "strain_derivative", "stress"):
        np.testing.assert_allclose(shifted[key], zero[key], rtol=0, atol=1e-10)


def check_dimensional_stress_routes(reference, pbc, tmp_path, mesh=None, scalar_ecp=False,
                                    *, fourier_field=False, vv10_xc=None):
    numbers, positions, cell, basis = fixture(tmp_path)
    if reference == "rks":
        numbers = [1, 3]
    elif reference == "roks":
        numbers = [1, 3, 1]
        positions.append([-0.46, 1.29, 1.62])
    if scalar_ecp:
        numbers = [1, 11, 1]
        positions.append([-0.46, 1.29, 1.62])
        basis.write_text(basis.read_text() + "Na 0\nS 1 1.0\n0.7 1.0\n****\n"
                         "Na 0\nNa-ECP 1 10\np potential\n1\n2 0.65 0.75\n"
                         "s-p potential\n1\n2 0.91 -0.40\n")
    cell = np.asarray(cell)
    cell[np.asarray(pbc)] *= 0.55
    active = cell[np.asarray(pbc)]
    dimension = len(active)
    gram = active @ active.T
    measure = np.sqrt(np.linalg.det(gram))
    projector = active.T @ np.linalg.solve(gram, active)
    unit = {1: "hartree/bohr", 2: "hartree/bohr^2", 3: "hartree/bohr^3"}[dimension]
    controls = parameters(basis, reference=reference)
    if reference != "gks":
        controls.pop("nuclear_soc_scale")
    if reference == "rks":
        controls["multiplicity"] = 1
    controls.update(k_mesh=[2, 1, 1], overlap_cutoff=3.8, exchange_cutoff=3.8, electrostatic_order=48)
    if reference == "roks":
        controls.update(k_mesh=mesh or [1, 1, 1], damping=0.0)
    if fourier_field:
        controls.update(k_mesh=[3, 1, 1], forces=True, field_wave_indices=[1, 0, 0],
                        field_scalar=[0.065, -0.047], field_vector_real=[0.023, -0.039, 0.028],
                        field_vector_imag=[-0.017, 0.012, 0.031], field_amplitude=0.74)
    if vv10_xc:
        if reference != "roks":
            numbers.append(2)
            positions.append([-0.46, 1.29, 1.62])
        controls.update(xc=vv10_xc, vv10_cutoff=6.4, k_mesh=mesh or [3, 1, 1], forces=True,
                        grid_level=0, grid_tolerance=0.1, conv_tol=1e-13, conv_tol_grad=1e-10)
        if reference == "gks" and dimension == 3:
            controls["electrostatic_order"] = 16
    native = dft_rs.stress(numbers, positions, cell.tolist(), pbc=list(pbc), **controls)
    atoms = Atoms(numbers, positions=np.asarray(positions) * BOHR_IN_ANGSTROM,
                  cell=cell * BOHR_IN_ANGSTROM, pbc=pbc)
    calc = dft_rs.DftRs(units="atomic", damping=0.2, use_adiis=False, adiis_until_error=None)
    atomic = calc.get_periodic_stress_report(atoms, **controls)
    command = cli_command(os.environ["DFT_RS_CLI"], tmp_path, numbers, positions, cell, controls,
                          feature="stress", pbc=pbc)
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "energy" in process.stderr.lower() and "residual" in process.stderr.lower()
    cli = json.loads(process.stdout)
    for report in (native, atomic, cli):
        assert report["converged"] and report["reference_type"] == reference
        assert report["units"]["stress"] == report["units"]["pressure"] == unit
        assert ("stress_gpa" in report) == (dimension == 3)
        assert abs(report["stress_periodic_measure"] - measure) < 1e-12
        np.testing.assert_allclose(report["strain_derivative"], native["strain_derivative"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(report["stress"], np.asarray(report["strain_derivative"]) / measure,
                                   rtol=0, atol=1e-12)
        assert abs(report["pressure"] + np.sum(projector * report["stress"]) / dimension) < 1e-12
        assert abs(report["orbital_residual"] - native["orbital_residual"]) < 1e-11
        if fourier_field:
            assert report["stress_field_convention"] == "fixed reduced wave indices and fixed Cartesian Fourier amplitudes"
            for key in ("stress_diagnostics", "gradient_diagnostics"):
                assert report[key]["external_field_energy_defect"] < 1e-12
            for key in ("strain_components", "gradient_components"):
                assert np.max(np.abs(report[key]["external_field"])) > 1e-5
                np.testing.assert_allclose(report[key]["external_field"], native[key]["external_field"],
                                           rtol=0, atol=1e-10)
        if scalar_ecp:
            assert np.max(np.abs(report["strain_components"]["ecp"])) > 1e-6
            np.testing.assert_allclose(report["strain_components"]["ecp"], native["strain_components"]["ecp"],
                                       rtol=0, atol=1e-10)
        if vv10_xc:
            keys = ["energy", "forces"]
            if reference == "roks":
                assert report["units"]["nonlocal_correlation_energy"] == "hartree per cell"
                assert abs(report["nonlocal_correlation_energy"]) > 1e-8
                keys.append("nonlocal_correlation_energy")
            else:
                terms = report["energy_components"]
                assert abs(terms["nonlocal_correlation"]) > 1e-8
                assert abs(sum(terms.values()) - report["energy"]) < 1e-11
                keys.append("xc_grid_number_defect" if reference == "gks" else "grid_number_defect")
            if reference == "gks":
                assert report["units"]["nonlocal_correlation_energy"] == "hartree per cell"
                keys.append("nonlocal_correlation_energy")
                for key in ("gradient_components", "strain_components"):
                    assert np.max(np.abs(report[key]["nonlocal_correlation"])) > 1e-8
                    np.testing.assert_allclose(report[key]["nonlocal_correlation"],
                                               native[key]["nonlocal_correlation"], rtol=0, atol=1e-10)
            assert report["stress_diagnostics"]["xc_reference_energy_defect"] < 1e-11
            for key in keys:
                np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-10)
        if mesh == [3, 1, 1]:
            assert report["density_conjugation_defect"] < 1e-12
            assert report["fock_conjugation_defect"] < 1e-10
            np.testing.assert_allclose(report["density_imag"], native["density_imag"], rtol=0, atol=1e-10)
    physical_controls = dict(controls)
    for key in ("overlap_cutoff", "exchange_cutoff", "xc_ao_cutoff", "grid_cutoff", "ewald_real_cutoff"):
        physical_controls[key] *= BOHR_IN_ANGSTROM
    for key in ("ewald_alpha", "ewald_reciprocal_cutoff"):
        physical_controls[key] /= BOHR_IN_ANGSTROM
    for key in ("conv_tol", "conv_tol_grad"):
        physical_controls[key] *= HARTREE_IN_EV
    if fourier_field:
        physical_controls["field_scalar"] = np.asarray(physical_controls["field_scalar"]) * HARTREE_IN_EV
    if vv10_xc:
        physical_controls["vv10_cutoff"] *= BOHR_IN_ANGSTROM
    physical = calc.get_periodic_stress_report(atoms, units="ase", **physical_controls)
    scale = HARTREE_IN_EV / BOHR_IN_ANGSTROM**dimension
    np.testing.assert_allclose(physical["stress"], np.asarray(native["stress"]) * scale, rtol=0, atol=1e-9)
    if vv10_xc:
        np.testing.assert_allclose(physical["forces"], np.asarray(native["forces"]) * HARTREE_IN_EV / BOHR_IN_ANGSTROM,
                                   rtol=0, atol=1e-9)
    print(f"Installed dimensional {reference.upper()} stress dimension={dimension} {json.dumps(native, sort_keys=True)}")
    return native, atoms, controls
