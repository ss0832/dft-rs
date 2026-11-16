# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Intermolecular perturbation analysis through the native, generic, ASE and CLI routes."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


ENERGY_KEYS = ["electrostatic", "exchange", "induction", "exchange_induction", "dispersion",
               "exchange_dispersion", "interaction_energy", "interaction_energy_rpa",
               "supermolecular_interaction", "delta"]


def test_sapt_analysis_public_routes(tmp_path):
    numbers = [8, 1, 1, 8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.108], [0.0, -1.43, 1.108],
                          [0.0, 0.0, 5.4], [0.0, 1.43, 6.5], [0.0, -1.43, 6.5]])
    options = dict(xc="pbe0", basis="sto-3g", conv_tol=1e-12, dispersion="none",
                   fragments=[0, 0, 0, 1, 1, 1], asymptotic_shift_a=0.05,
                   asymptotic_shift_b=0.05, casimir_polder_points=12)
    native = dft_rs.sapt_analysis(numbers, positions, **options)
    generic = dft_rs.run_feature("sapt-analysis", numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_sapt_analysis(atoms, **options)
    electronvolt = dft_rs.DftRs().get_sapt_analysis(
        atoms, **dict(options, conv_tol=options["conv_tol"] * Hartree,
                      asymptotic_shift_a=0.05 * Hartree, asymptotic_shift_b=0.05 * Hartree))
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "dimer.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    symbols = {1: "H", 8: "O"}
    xyz.write_text(f"{len(numbers)}\nsapt acceptance\n" + "".join(
        f"{symbols[z]} " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n"
        for z, row in zip(numbers, positions)), encoding="utf-8")
    argv = [executable, "sapt-analysis", str(xyz), "--format", "json"]
    for key, value in options.items():
        rendered = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
        argv.extend(["--" + key.replace("_", "-"), rendered])
    executed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr
    cli = json.loads(executed.stdout)
    for other in [generic, atomic, cli]:
        for key in ENERGY_KEYS:
            np.testing.assert_allclose(other[key], native[key], atol=1e-9, rtol=0)
        for group, key in [("rpa", "casimir_polder"), ("charge_transfer", "energy"),
                           ("partition", "electrostatic")]:
            np.testing.assert_allclose(other[group][key], native[group][key], atol=1e-9, rtol=0)
        assert other["units"] == native["units"]
    for key in ENERGY_KEYS:
        np.testing.assert_allclose(electronvolt[key], np.asarray(native[key]) * Hartree,
                                   rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(electronvolt["fragments"][0]["asymptotic_shift"],
                               0.05 * Hartree, rtol=1e-12)
    assert native["rpa"]["casimir_polder_residual"] < 1e-8 * abs(native["rpa"]["dispersion"])
    assert native["partition"]["residual"] < 1e-9
    print("sapt-analysis: " + ", ".join(f"{k}={native[k]}" for k in ENERGY_KEYS))


@pytest.mark.parametrize(("variant", "reference_energy", "eta", "correlation"), [
    ("sapt0", 1.868986122105836e-3, None, False),
    ("sapt0-s2", 1.860374017185528e-3, None, False),
    ("ssapt0", 2.402226914188570e-3, None, False),
    ("sapt-30", 1.868844195738965e-3, None, False),
    ("sapt2+3", 2.491741734470445e-3, None, False),
    pytest.param("sapt0", 1.868986122105836e-3, 3.0, False, id="regularized"),
    pytest.param("sapt0", 1.868986122105836e-3, None, True, id="correlation"),
])
def test_sapt0_public_routes_and_units(tmp_path, variant, reference_energy, eta, correlation):
    numbers = [1, 1, 1, 1]
    positions = np.array([[0.04, 0.12, -0.06], [1.41, -0.31, 0.63],
                          [1.02, 0.77, 4.62], [1.67, 1.86, 5.15]])
    options = dict(xc="hf", basis="6-31g", conv_tol=1e-13, conv_tol_dp=1e-11,
                   fragments=[0, 0, 1, 1], sapt_variant=variant)
    if eta is not None:
        options["sapt_regularization_eta"] = eta
    if correlation:
        options.update(sapt_correlation_terms=True, sapt_relaxed_induction30=True)
    native = dft_rs.sapt_analysis(numbers, positions, **options)
    generic = dft_rs.run_feature("sapt-analysis", numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_sapt_analysis(atoms, **options)
    converted = dict(options, conv_tol=options["conv_tol"] * Hartree)
    if eta is not None:
        converted["sapt_regularization_eta"] = eta / Bohr**2
    electronvolt = dft_rs.DftRs().get_sapt_analysis(atoms, **converted)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "hydrogen-dimer.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("4\nNoncoplanar hydrogen dimer\n" + "".join(
        "H " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n" for row in positions
    ), encoding="utf-8")
    command = [executable, "sapt-analysis", str(xyz), "--format", "json",
               "--sapt-variant", variant, "--xc", "hf", "--basis", "6-31g",
               "--fragments", "0,0,1,1", "--conv-tol", "1e-13", "--conv-tol-dp", "1e-11"]
    if eta is not None:
        command.extend(["--sapt-regularization-eta", str(eta)])
    if correlation:
        command.extend(["--sapt-correlation-terms", "true", "--sapt-relaxed-induction30", "true"])
    executed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr
    cli = json.loads(executed.stdout)
    keys = [key for key in ENERGY_KEYS if key != "interaction_energy_rpa"]
    keys.extend(["exchange_induction_directions", "dispersion_uncoupled"])
    if variant == "ssapt0":
        keys.append("delta_hf")
        for other in [native, generic, atomic, cli, electronvolt]:
            np.testing.assert_allclose(other["exchange_scale"], 1.009959557257506, atol=1e-10, rtol=0)
            assert other["units"]["exchange_scale"] == "dimensionless"
        np.testing.assert_allclose(native["delta_hf"], 5.328187587708256e-4, atol=1e-11, rtol=0)
    raw_deviations = {}
    for title, other in [("Rust CLI", cli), ("generic Python", generic), ("atomic ASE", atomic)]:
        assert other["units"] == native["units"]
        assert other["sapt_variant"] == variant
        assert other["interaction_energy_rpa"] is None
        assert "rpa" not in other
        diffs = {}
        for key in keys:
            diffs[key] = np.max(np.abs(np.asarray(other[key]) - np.asarray(native[key])))
            np.testing.assert_allclose(other[key], native[key], atol=1e-11, rtol=0)
        raw_deviations[title] = max(diffs.values())
    for key in keys:
        np.testing.assert_allclose(electronvolt[key], np.asarray(native[key]) * Hartree,
                                   atol=1e-10, rtol=1e-8)
    assert native["units"]["exchange_induction_directions"] == "hartree"
    assert electronvolt["units"]["exchange_induction_directions"] == "eV"
    np.testing.assert_allclose(sum(native["exchange_induction_directions"]), native["exchange_induction"],
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(native["interaction_energy"], reference_energy, atol=1e-11, rtol=0)
    if variant in {"sapt-30", "sapt2+3"}:
        coefficients = {
            "induction": 1.388708275411512e-9,
            "dispersion": -1.455824630054733e-8,
            "induction_dispersion": -1.890504994794368e-8,
            "exchange_induction": -1.068075978232381e-8,
            "exchange_dispersion": -2.034251967583891e-8,
            "exchange_induction_dispersion": -1.172742412878337e-8,
            "exchange": -4.275070358694609e-8,
            "total": -7.482529156002558e-8,
        }
        for key, expected in coefficients.items():
            np.testing.assert_allclose(native["third_order"][key], expected, atol=1e-11, rtol=0)
            for title, other in [("Rust CLI", cli), ("generic Python", generic), ("atomic ASE", atomic)]:
                difference = abs(other["third_order"][key] - native["third_order"][key])
                raw_deviations[title] = max(raw_deviations[title], difference)
                np.testing.assert_allclose(other["third_order"][key], native["third_order"][key],
                                           atol=1e-11, rtol=0)
            np.testing.assert_allclose(electronvolt["third_order"][key], expected * Hartree,
                                       atol=1e-10, rtol=1e-8)
        if variant == "sapt-30":
            assert "partition" not in native and "charge_transfer" not in native
    if eta is not None:
        regularized = native["regularized_induction"]
        expected = {
            "induction": [-1.8563936110048981e-6, -1.998998094653041e-6],
            "exchange_induction": [1.7956868758417853e-6, 2.0952695500370797e-6],
            "charge_transfer": [-2.9266444749906288e-8, 1.064014496118134e-7],
            "total_charge_transfer": 7.713500486190711e-8,
        }
        assert regularized["eta"] == eta
        for key, value in expected.items():
            np.testing.assert_allclose(regularized[key], value, atol=1e-11, rtol=0)
            for title, other in [("Rust CLI", cli), ("generic Python", generic), ("atomic ASE", atomic)]:
                reference = np.asarray(regularized[key])
                measured = np.asarray(other["regularized_induction"][key])
                difference = np.max(np.abs(measured - reference))
                raw_deviations[title] = max(raw_deviations[title], difference)
                np.testing.assert_allclose(measured, reference, atol=1e-11, rtol=0)
            np.testing.assert_allclose(electronvolt["regularized_induction"][key],
                                       np.asarray(value) * Hartree, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(electronvolt["regularized_induction"]["eta"],
                                   eta / Bohr**2, rtol=1e-12)
        assert native["units"]["regularized_induction.eta"] == "bohr^-2"
        assert electronvolt["units"]["regularized_induction.eta"] == "angstrom^-2"
    if correlation or variant == "sapt2+3":
        groups = {
            "correlation_terms": ("electrostatics", "exchange", "relaxed_electrostatics",
                                  "second_order_polarization", "second_order_exchange",
                                  "truncated_induction22", "truncated_exchange_induction22"),
            "relaxed_induction30": ("induction", "delta_hf_order2"),
        }
        expected_electrostatic = np.array([[1.386867510487630e-5, 6.084214153311484e-6],
                                           [5.200091200046071e-6, 1.450318461850975e-6]])
        np.testing.assert_allclose(native["correlation_terms"]["relaxed_electrostatics"],
                                   expected_electrostatic, atol=1e-13, rtol=0)
        np.testing.assert_allclose(native["correlation_terms"]["second_order_polarization"][2],
                                   [-2.329475587868648e-6, 3.178920027731963e-6], atol=1e-13, rtol=0)
        if variant == "sapt2+3":
            terms = {
                "relaxed_exchange_induction30": -7.479392618882041e-8,
                "delta_hf_order3": 5.328737318035622e-4,
                "correlated_electrostatics": 2.660329892008483e-5,
                "correlated_exchange": 6.256092377268093e-5,
                "correlated_induction": -1.184082831409993e-9,
                "correlated_dispersion": 8.494444398633144e-7,
                "cubic_dispersion_and_mixed": -6.553324005311329e-8,
            }
            groups["sapt2_plus3"] = tuple(terms)
            for key, value in terms.items():
                np.testing.assert_allclose(native["sapt2_plus3"][key], value, atol=1e-12, rtol=0)
            np.testing.assert_allclose(
                native["delta_hf"] + native["sapt2_plus3"]["relaxed_exchange_induction30"]
                + native["relaxed_induction30"]["induction"],
                native["relaxed_induction30"]["delta_hf_order2"], atol=1e-15, rtol=0,
            )
        for group, fields in groups.items():
            for key in fields:
                reference = np.asarray(native[group][key])
                for title, other in [("Rust CLI", cli), ("generic Python", generic), ("atomic ASE", atomic)]:
                    measured = np.asarray(other[group][key])
                    deviation = np.max(np.abs(measured - reference))
                    raw_deviations[title] = max(raw_deviations[title], deviation)
                    np.testing.assert_allclose(measured, reference, atol=1e-11, rtol=0)
                np.testing.assert_allclose(electronvolt[group][key], reference * Hartree, atol=1e-11, rtol=0)
                assert native["units"][f"{group}.{key}"] == "hartree"
                assert electronvolt["units"][f"{group}.{key}"] == "eV"
        np.testing.assert_allclose(native["relaxed_induction30"]["delta_hf_order2"] + sum(
            [native["electrostatic"], native["exchange"], sum(native["induction"]),
             native["exchange_induction"]]), native["supermolecular_interaction"], atol=1e-15, rtol=0)
    else:
        assert "correlation_terms" not in native and "relaxed_induction30" not in native
    with pytest.raises(ValueError, match=r"SAPT\(0\) requires field-free integer Aufbau Hartree-Fock"):
        dft_rs.sapt_analysis(numbers, positions, **dict(options, xc="pbe0"))
    print(f"{variant} public maximum deviations (Hartree): {raw_deviations}")


@pytest.mark.parametrize(("xc", "total"), [
    ("pbe", 1.789458889069682e-3),
    ("wb97x-d", 1.7765013295757835e-3),
    # Dense SAPT oracle with the independently checked literal VV10 tail.
    ("wb97x-v", 1.790220363598613e-3),
    ("wb97m-v", 1.7861508798670308e-3),
])
def test_ks_fdds_and_regularization_public_parity(tmp_path, xc, total):
    numbers = [1, 1, 1, 1]
    positions = np.array([[0.04, 0.12, -0.06], [1.41, -0.31, 0.63],
                          [1.02, 0.77, 4.62], [1.67, 1.86, 5.15]])
    controls = dict(xc=xc, basis="6-31g", conv_tol=1e-13, conv_tol_dp=1e-11,
                    fragments=[0, 0, 1, 1], sapt_variant="sapt-ks-fdds",
                    sapt_partition=False, charge_transfer=False, grid_level=1,
                    grid_weight_cutoff=0.0, vv10_grid_level=0, casimir_polder_points=16,
                    sapt_regularization_eta=3.0)
    if xc == "pbe":
        controls.update(asymptotic_shift_a=0.08, asymptotic_shift_b=0.08)
    if xc == "wb97x-d":
        controls["dispersion"] = "none"
    native = dft_rs.sapt_analysis(numbers, positions, **controls)
    generic = dft_rs.run_feature("sapt-analysis", numbers, positions, **controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_sapt_analysis(atoms, **controls)
    ase_controls = dict(controls, conv_tol=controls["conv_tol"] * Hartree,
                        sapt_regularization_eta=3.0 / Bohr**2)
    if xc == "pbe":
        ase_controls.update(asymptotic_shift_a=0.08 * Hartree, asymptotic_shift_b=0.08 * Hartree)
    electronvolt = dft_rs.DftRs().get_sapt_analysis(atoms, **ase_controls)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "hydrogen-dimer.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("4\nNoncoplanar hydrogen dimer\n" + "".join(
        "H " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n" for row in positions
    ), encoding="utf-8")
    command = [executable, "sapt-analysis", str(xyz), "--format", "json"]
    for key, value in controls.items():
        if isinstance(value, list):
            rendered = ",".join(map(str, value))
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        command.extend(["--" + key.replace("_", "-"), rendered])
    executed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr
    cli = json.loads(executed.stdout)
    deviations = {}
    fields = [((), key) for key in ENERGY_KEYS]
    fields += [(("regularized_induction",), key) for key in (
        "induction", "exchange_induction", "charge_transfer", "total_charge_transfer",
    )]
    fields += [(("rpa",), key) for key in ("dispersion", "exchange_dispersion", "casimir_polder")]
    for path, key in fields:
        baseline = np.asarray(native[path[0]][key] if path else native[key])
        converted = electronvolt[path[0]][key] if path else electronvolt[key]
        np.testing.assert_allclose(converted, baseline * Hartree, atol=1e-10, rtol=0)
        for label, other in [("CLI", cli), ("generic", generic), ("ASE", atomic)]:
            measured = np.asarray(other[path[0]][key] if path else other[key])
            np.testing.assert_allclose(measured, baseline, atol=1e-11, rtol=0)
            deviations[label] = max(deviations.get(label, 0), np.max(np.abs(measured-baseline)))
    assert native["sapt_variant"] == "sapt-ks-fdds"
    np.testing.assert_allclose(native["interaction_energy"], total, atol=1e-11, rtol=0)
    np.testing.assert_allclose(native["rpa"]["casimir_polder"], native["dispersion"], atol=1e-12, rtol=0)
    assert native["rpa"]["casimir_polder_residual"] < 1e-11
    print(f"{xc} KS-FDDS public maximum deviations (Hartree): {deviations}")

def test_sapt2_plus3_water_respects_managed_memory(tmp_path):
    numbers = [8, 1, 1, 8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.0], [0.04, 0.10, 1.81], [1.72, 0.10, -0.45],
                          [0.12, -0.24, 5.35], [1.45, 0.66, 6.20], [-1.00, 1.04, 6.01]])
    controls = dict(xc="hf", basis="sto-3g", conv_tol=1e-13, conv_tol_dp=1e-11,
                    sapt_variant="sapt2+3", fragments=[0, 0, 0, 1, 1, 1],
                    incore_mb=128, memory_mb=128, cpu_threads=1,
                    sapt_partition=False, charge_transfer=False)
    native = dft_rs.sapt_analysis(numbers, positions, **controls)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_sapt_analysis(atoms, **controls)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the water memory acceptance"
    path = tmp_path / "water-dimer.xyz"
    scale = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    path.write_text("6\nWater dimer\n" + "".join(
        symbol + " " + " ".join(f"{r * scale:.17g}" for r in row) + "\n"
        for symbol, row in zip("OHHOHH", positions)
    ), encoding="utf-8")
    args = [executable, "sapt-analysis", str(path), "--format", "json"]
    for key, value in controls.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif isinstance(value, bool):
            value = str(value).lower()
        args.extend(["--" + key.replace("_", "-"), str(value)])
    completed = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "[scf]" in completed.stderr
    cli = json.loads(completed.stdout)
    groups = ["sapt2_plus3", "correlation_terms", "third_order", "relaxed_induction30"]
    for name, report in [("CLI", cli), ("ASE", atomic), ("native", native)]:
        resources = report["resources"]
        assert resources["memory_mb"] == 128
        assert resources["managed_peak_bytes"] <= 128 * 1024**2
        np.testing.assert_allclose(report["interaction_energy"], 1.2164009737768004e-3,
                                   rtol=0, atol=2e-11)
        differences = []
        for group in groups:
            assert report[group].keys() == native[group].keys()
            for key in report[group]:
                expected = np.asarray(native[group][key])
                actual = np.asarray(report[group][key])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-11)
                differences.append(np.max(np.abs(actual - expected)))
        print(f"Water SAPT2+3 {name}: {report['interaction_energy']:+.15e} Hartree, "
              f"term discrepancy {max(differences):.3e} Hartree, "
              f"managed peak {resources['managed_peak_bytes']} bytes")
