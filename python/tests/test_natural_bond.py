# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Natural bond orbital analysis through the native, generic, ASE and CLI routes."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs
from test_fixed_populations import cli_report


@pytest.mark.parametrize("xc", ["svwn5", "pbe", "wb97x-d", "wb97x-v"])
def test_gks_natural_bond_one_electron_public_routes(tmp_path, xc):
    points = np.array([[.12, -.08, -.64], [.37, .24, .74]])
    atoms = Atoms("H2", positions=points*Bohr)
    axis = [.36, .48, .8]
    controls = dict(xc=xc, basis="sto-3g", reference="gks", charge=1,
                    multiplicity=2, grid_level=0, grid_weight_cutoff=0.,
                    vv10_grid_level=0, cpu_threads=1, nbo_spin_density=True,
                    nbo_natural_orbitals=True, spin_axis=axis)
    unrotated = dft_rs.natural_bond([1, 1], points.tolist(), **dict(controls, spin_axis=[0., 0., 1.]))
    native = dft_rs.natural_bond([1, 1], points.tolist(), **controls)
    atomic = dft_rs.DftRs(units="atomic").get_natural_bond(atoms, **controls)
    cli = cli_report(tmp_path, "natural-bond", atoms, controls)
    for result in (native, atomic, cli):
        assert result["converged"] and result["reference_type"] == "gks"
        assert result["nbo_spinor_density_defect"] < 1e-12
        assert result["nbo_spinor_fock_defect"] < 1e-12
        np.testing.assert_allclose(result["nbo_spin_axis"], axis, atol=1e-11, rtol=0.)
        assert abs(result["energy"] - unrotated["energy"]) < 5e-12
        np.testing.assert_allclose(result["natural_spin_density"]["weights"], [0., 1.],
                                   atol=1e-12, rtol=0.)
        np.testing.assert_allclose(result["natural_charges"], native["natural_charges"],
                                   atol=1e-11, rtol=0.)
    print(f"H2+ GKS/{xc} E={native['energy']:.13e}; rotated energy defect "
          f"{native['energy'] - unrotated['energy']:.3e}; spin weights {native['natural_spin_density']['weights']}")


@pytest.mark.parametrize("charge,multiplicity,xc,reference", [
    (0, 1, "hf", None),
    (1, 2, "hf", None),
    (1, 2, "wb97x-d", "uks"),
    (1, 2, "wb97x-v", "roks"),
    (1, 2, "wb97x-d", "gks"),
])
def test_natural_bond_public_routes(tmp_path, charge, multiplicity, xc, reference):
    numbers = [8, 1, 1]
    positions = np.array([[0.0, 0.0, 0.23], [0.0, 1.4309, -0.8867], [0.12, -1.4809, -0.9067]])
    options = dict(xc=xc, basis="6-31g*" if reference is None else "sto-3g",
                   charge=charge, multiplicity=multiplicity,
                   conv_tol=1e-12, conv_tol_grad=1e-9, dispersion="auto",
                   nbo_spin_density=True, nbo_natural_orbitals=True,
                   nbo_spin_density_points=[[.34, -.19, .32], [-.81, .16, -.17]])
    if reference is not None:
        options.update(reference=reference, grid_level=1, vv10_grid_level=0,
                       grid_weight_cutoff=0., cpu_threads=1)
    if reference == "gks":
        options["grid_level"] = 0
        options["spin_axis"] = [.36, .48, .8]
    else:
        options["nbo_corresponding_orbitals"] = True
    native = dft_rs.natural_bond(numbers, positions, **options)
    generic = dft_rs.run_feature("natural-bond", numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = dft_rs.DftRs(units="atomic").get_natural_bond(atoms, **options)
    # The same physical tolerances in eV, so that both unit modes solve the
    # same problem and only the unit conversion is compared.
    physical_options = dict(options, conv_tol=options["conv_tol"] * Hartree,
                            nbo_spin_density_points=np.array(options["nbo_spin_density_points"]) * Bohr)
    electronvolt = dft_rs.DftRs().get_natural_bond(atoms, **physical_options)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "water.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("3\nnatural bond acceptance\n" + "".join(
        f"{symbol} " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n"
        for symbol, row in zip(("O", "H", "H"), positions)), encoding="utf-8")
    argv = [executable, "natural-bond", str(xyz), "--format", "json"]
    for key, value in options.items():
        rendered = json.dumps(value) if isinstance(value, (list, bool)) else str(value)
        argv.extend(["--" + key.replace("_", "-"), rendered])
    executed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr
    cli = json.loads(executed.stdout)
    keys = ["natural_charges", "natural_spin_populations", "nao_occupancies",
            "wiberg_bond_indices", "core_populations", "valence_populations",
            "rydberg_populations"]
    for other in [generic, atomic, cli, electronvolt]:
        if reference is not None:
            assert other["reference_type"] == reference
        for key in keys:
            np.testing.assert_allclose(other[key], native[key], atol=1e-9, rtol=0)
        assert other["natural_electron_configurations"] == native["natural_electron_configurations"]
        assert other["nao_labels"] == native["nao_labels"]
        assert len(other["channels"]) == len(native["channels"])
    for other in [generic, atomic, cli]:
        assert other["units"] == native["units"]
    expected = ["total"] if multiplicity == 1 else ["alpha", "beta"]
    assert [c["spin"] for c in native["channels"]] == expected
    natural = native["natural_spin_density"]
    orbitals = native["natural_orbitals"]
    assert abs(sum(orbitals["occupations"]) - (10 - charge)) < 1e-10
    assert orbitals["occupation_bound_defect"] < 1e-10
    assert orbitals["reconstruction_residual"] < 1e-12
    assert abs(sum(natural["weights"]) - (multiplicity - 1)) < 1e-10
    assert abs(natural["trace_residual"]) < 1e-12
    assert natural["reconstruction_residual"] < 1e-12
    for other in [generic, atomic, cli, electronvolt]:
        candidate = other["natural_spin_density"]
        physical = other is electronvolt
        np.testing.assert_allclose(candidate["weights"], natural["weights"], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(np.asarray(candidate["points"]) / (Bohr if physical else 1.),
                                   natural["points"], atol=1e-13, rtol=0.)
        np.testing.assert_allclose(np.asarray(candidate["values"]) * (Bohr**3 if physical else 1.),
                                   natural["values"], atol=1e-9, rtol=0.)
        candidate_orbitals = other["natural_orbitals"]
        np.testing.assert_allclose(candidate_orbitals["occupations"], orbitals["occupations"], atol=1e-9, rtol=0.)
        np.testing.assert_allclose(candidate_orbitals["effectively_unpaired_electrons"],
                                   orbitals["effectively_unpaired_electrons"], atol=1e-9, rtol=0.)
    if reference != "gks":
        corresponding = native["corresponding_orbitals"]
        assert corresponding["cross_overlap_residual"] < 1e-12
        assert abs(corresponding["spin_squared"] - native["s_squared"]) < 1e-11
        for other in (generic, atomic, cli, electronvolt):
            pairs = other["corresponding_orbitals"]
            np.testing.assert_allclose(pairs["paired_overlap"], corresponding["paired_overlap"],
                                       atol=1e-9, rtol=0.)
        print(f"{xc}/{reference}: corresponding overlaps {corresponding['paired_overlap']}; "
              f"cross overlap residual {corresponding['cross_overlap_residual']:.3e}")
    if multiplicity == 1:
        np.testing.assert_allclose(natural["weights"], 0., atol=1e-12, rtol=0.)
        np.testing.assert_allclose(natural["values"], 0., atol=1e-12, rtol=0.)
        assert orbitals["effectively_unpaired_electrons"] < 1e-12
    print(f"{xc}/{reference}: spin density point values {natural['values']}; "
          f"spin spectrum {natural['weights']}; effectively unpaired {orbitals['effectively_unpaired_electrons']:.12g}")
    if reference == "gks":
        np.testing.assert_allclose(native["nbo_spin_axis"], options["spin_axis"], atol=1e-8, rtol=0.)
        assert native["nbo_spinor_density_defect"] < 1e-9
        assert native["nbo_spinor_fock_defect"] < 1e-8
    for k, channel in enumerate(native["channels"]):
        labels = [b["label"] for b in channel["nbos"] if b["lewis"]]
        for other in [generic, atomic, cli, electronvolt]:
            assert [b["label"] for b in other["channels"][k]["nbos"] if b["lewis"]] == labels
            np.testing.assert_allclose(
                [b["occupancy"] for b in other["channels"][k]["nbos"]],
                [b["occupancy"] for b in channel["nbos"]], atol=1e-9, rtol=0)
        # Energies in eV on the ASE route.
        np.testing.assert_allclose(
            [b["energy"] for b in electronvolt["channels"][k]["nbos"]],
            np.asarray([b["energy"] for b in channel["nbos"]]) * Hartree, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(
            [p["stabilization"] for p in electronvolt["channels"][k]["perturbation"]],
            np.asarray([p["stabilization"] for p in channel["perturbation"]]) * Hartree,
            rtol=1e-8, atol=1e-10)
        assert channel["lewis_count"] == channel["expected_lewis_count"]
        assert channel["orthonormality_residual"] < 1e-12
        print(f"water charge {charge} {channel['spin']}: Lewis {labels}, "
              f"non-Lewis {channel['non_lewis_occupancy']:.6f}, "
              f"{len(channel['perturbation'])} donor-acceptor pairs")
    assert abs(sum(native["natural_spin_populations"]) - (multiplicity - 1)) < 1e-10


@pytest.mark.parametrize("feature,native_name,ase_name,extra,keys", [
    ("natural-shielding", "natural_shielding", "get_natural_shielding", {},
     ["decomposition_residual"]),
    ("natural-coupling", "natural_coupling", "get_natural_coupling", {"nuclei": [0, 1]},
     ["decomposition_residual"]),
    ("natural-energy-decomposition", "natural_energy_decomposition",
     "get_natural_energy_decomposition", {"fragments": [0, 0, 0, 1]},
     ["interaction_energy", "charge_transfer", "electrostatic", "exchange_correlation",
      "deformation"]),
    ("natural-bond", "natural_bond", "get_natural_bond", {"nbo_resonance": True},
     ["natural_charges", "resonance_bond_orders"]),
])
def test_natural_bond_extension_routes(tmp_path, feature, native_name, ase_name, extra, keys):
    if feature == "natural-energy-decomposition":
        numbers = [8, 1, 1, 10]
        positions = np.array([[0.0, 0.0, 0.23], [0.0, 1.4309, -0.8867],
                              [0.12, -1.4809, -0.9067], [0.0, 0.0, 6.5]])
    else:
        numbers = [8, 1, 1]
        positions = np.array([[0.0, 0.0, 0.23], [0.0, 1.4309, -0.8867], [0.12, -1.4809, -0.9067]])
    options = dict(xc="hf", basis="sto-3g", conv_tol=1e-12, conv_tol_grad=1e-9,
                   dispersion="none", **extra)
    native = getattr(dft_rs, native_name)(numbers, positions, **options)
    generic = dft_rs.run_feature(feature, numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = getattr(dft_rs.DftRs(units="atomic"), ase_name)(atoms, **options)
    electronvolt = getattr(dft_rs.DftRs(), ase_name)(
        atoms, **dict(options, conv_tol=options["conv_tol"] * Hartree))
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "molecule.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    symbols = {1: "H", 8: "O", 10: "Ne"}
    xyz.write_text(f"{len(numbers)}\nextension acceptance\n" + "".join(
        f"{symbols[z]} " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n"
        for z, row in zip(numbers, positions)), encoding="utf-8")
    argv = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        rendered = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
        argv.extend(["--" + key.replace("_", "-"), rendered.lower() if isinstance(value, bool) else rendered])
    executed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr
    cli = json.loads(executed.stdout)
    for other in [generic, atomic, cli]:
        for key in keys:
            np.testing.assert_allclose(other[key], native[key], atol=1e-9, rtol=0)
        assert other["units"] == native["units"]
    energy_keys = [k for k in keys if k in ("interaction_energy", "charge_transfer", "electrostatic",
                                             "exchange_correlation", "deformation")]
    for key in energy_keys:
        np.testing.assert_allclose(electronvolt[key], native[key] * Hartree, rtol=1e-8, atol=1e-10)
    print(f"{feature}: " + ", ".join(f"{k}={native[k]}" for k in keys if not isinstance(native[k], list)))
