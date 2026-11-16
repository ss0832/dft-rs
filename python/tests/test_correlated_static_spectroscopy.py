# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""One common static spectroscopy provider on native, ASE and CLI surfaces."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree


POSITIONS = np.array([[0.13, -0.17, 0.21], [1.68, 0.14, -0.09],
                      [-0.38, 1.73, 0.32], [0.26, -0.48, 1.84], [1.71, 1.58, 1.39]])
DIRECTION = np.array([[0.07, -0.03, 0.02], [-0.04, 0.08, 0.01],
                      [0.02, 0.01, -0.06], [0.03, -0.02, 0.04], [-0.01, 0.03, -0.02]])
OPTIONS = dict(basis="sto-3g", grid_level=0, grid_weight_cutoff=0.0,
               conv_tol=1e-14, conv_tol_grad=1e-13, conv_tol_dp=1e-13,
               response_tolerance=1e-12, max_iter=500, dispersion="none")


def test_unrestricted_taylor_reference_is_independent_of_orbital_gradient_controls():
    options = dict(basis="sto-3g", xc="hf", charge=0, multiplicity=2, grid_level=0,
                   conv_tol=1e-14, conv_tol_grad=1e-13, conv_tol_dp=1e-13,
                   max_iter=500, dispersion="none", response_tol=1e-12,
                   reference_field=[0.003, -0.005, 0.004])
    automatic = dft_rs.molecular_taylor([1]*5, POSITIONS.tolist(), DIRECTION.tolist(),
                                       [0.02, -0.03, 0.01], order=1, **options)
    explicit = dft_rs.molecular_taylor([1]*5, POSITIONS.tolist(), DIRECTION.tolist(),
                                      [0.02, -0.03, 0.01], order=1,
                                      unrestricted=True, **options)
    assert automatic["converged"] and explicit["converged"]
    for key in ("energy_coefficients", "dipole_coefficients", "density_coefficients",
                "overlap_coefficients", "orthonormal_fock_coefficients"):
        np.testing.assert_array_equal(explicit[key], automatic[key])


def test_generic_ase_control_conversion_uses_the_registered_getter_unit_contract():
    from dft_rs import ase_calculator as adapter

    physical = {"nuclear_direction": [[Bohr, -2*Bohr, 3*Bohr]], "electric_order": 3}
    converted = adapter._controls_to_atomic(physical, "static-spectroscopy")
    np.testing.assert_allclose(converted["nuclear_direction"], [[1, -2, 3]], rtol=0, atol=1e-15)
    # The unqualified name remains a dimensionless contraction pattern on
    # the periodic nuclear tensor API; only its typed getter adds path units.
    assert adapter._controls_to_atomic(physical)["nuclear_direction"] == physical["nuclear_direction"]
    assert converted["electric_order"] == 3
    assert physical["nuclear_direction"] == [[Bohr, -2*Bohr, 3*Bohr]]


def validate(report, rank):
    assert report["converged"] and report["provenance"] == "analytic"
    assert report["shape"] == [3] * rank
    assert len(report["tensor"]) == 3**rank
    assert report["permutation_comparisons"] == {1: 0, 2: 6, 3: 36}[rank]
    assert report["permutation_defect"] < 1e-10
    assert max(report["response_residuals"], default=0) < 1e-10
    assert max(report["amplitude_relative_residuals"], default=0) < 1e-10
    print("static correlated spectroscopy diagnostics", rank, {
        key: max(map(abs, report[key]), default=0) for key in
        ("response_residuals", "amplitude_response_residuals", "amplitude_relative_residuals",
         "projector_defects", "metric_defects", "orbital_frame_defects", "reference_defects")})


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_every_tensor_entry_on_common_native_and_atomic_and_physical_ase(unrestricted, rank):
    options = dict(OPTIONS, xc="hf", charge=int(not unrestricted),
                   multiplicity=2 if unrestricted else 1, unrestricted=unrestricted,
                   pt2_same_spin=0.17, pt2_opposite_spin=0.29, reference_field=[0.003, -0.005, 0.004])
    direct = dft_rs.static_spectroscopic_derivative([1]*5, POSITIONS.tolist(),
        DIRECTION.tolist(), electric_order=rank, **options)
    common = dft_rs.run_feature("static-spectroscopy", [1]*5, POSITIONS.tolist(),
        nuclear_direction=DIRECTION.tolist(), electric_order=rank, **options)
    validate(common, rank)
    atoms = Atoms("H5", positions=POSITIONS*Bohr)
    atomic = DftRs(units="atomic", **options).get_static_spectroscopic_derivative(atoms,
        nuclear_direction=DIRECTION.tolist(), electric_order=rank)
    for report in (direct, atomic):
        np.testing.assert_array_equal(report["tensor"], common["tensor"])
        assert report["unit_labels"] == common["units"]
    physical_options = dict(options, conv_tol=options["conv_tol"]*Hartree,
        conv_tol_grad=options["conv_tol_grad"]*Hartree,
        reference_field=np.asarray(options["reference_field"])*Hartree/Bohr)
    physical = DftRs(**physical_options).get_static_spectroscopic_derivative(atoms,
        nuclear_direction=(DIRECTION*Bohr).tolist(), electric_order=rank)
    np.testing.assert_allclose(physical["tensor"], np.asarray(common["tensor"])*Bohr**rank/Hartree**(rank-1),
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(physical["nuclear_direction"], DIRECTION*Bohr, rtol=0, atol=1e-14)
    generic = DftRs(**physical_options).get_feature_report("static-spectroscopy", atoms,
        nuclear_direction=(DIRECTION*Bohr).tolist(), electric_order=rank)
    np.testing.assert_allclose(generic["tensor"], physical["tensor"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(generic["nuclear_direction"], DIRECTION*Bohr, rtol=0, atol=1e-14)



@pytest.mark.parametrize("xc", ["pbe", "tpss", "pbe0", "b2plyp"])
def test_generic_and_named_compositions_use_the_same_static_spectroscopy_entry(xc):
    # These route checks do not extend the independent derivative acceptance
    # beyond the C1 full-component models measured in the Rust suite.
    options = dict(OPTIONS, xc=xc)
    if xc != "b2plyp":
        options.update(pt2_same_spin=0.17, pt2_opposite_spin=0.29)
    positions = [[0.1, -0.2, 0.3], [1.53, -0.2, 1.4], [-1.33, -0.2, 1.4]]
    direction = [[0.03, -0.02, 0.01], [-0.02, 0.04, -0.01], [0.01, -0.03, 0.02]]
    direct = dft_rs.static_spectroscopic_derivative([8, 1, 1], positions, direction,
        electric_order=3, **options)
    validate(direct, 3)
    assert direct["pt2_opposite_spin_coefficient"] > 0
    atoms = Atoms("OH2", positions=np.asarray(positions)*Bohr)
    adapted = DftRs(units="atomic", **options).get_static_spectroscopic_derivative(atoms,
        nuclear_direction=direction, electric_order=3)
    np.testing.assert_allclose(adapted["tensor"], direct["tensor"], rtol=0, atol=1e-10)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_canonical_cli_json_and_legacy_component_rows_and_default_scf_progress(tmp_path, rank):
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for public route acceptance"
    xyz = tmp_path / "static_spectroscopy.xyz"
    xyz.write_text("5\nC1 static correlated spectroscopy\n" + "".join(
        "H " + " ".join(f"{v:.17g}" for v in row) + "\n" for row in POSITIONS*Bohr), encoding="utf-8")
    options = dict(OPTIONS, xc="hf", charge=1, pt2_same_spin=0.17, pt2_opposite_spin=0.29)
    args = [executable, "static-spectroscopy", str(xyz), "--electric-order", str(rank),
            "--nuclear-direction", json.dumps(DIRECTION.tolist())]
    for key, value in options.items():
        args += ["--" + key.replace("_", "-"), str(value).lower()]
    structured = subprocess.run([*args, "--format", "json"], capture_output=True, text=True, check=False)
    assert structured.returncode == 0, structured.stderr
    assert "SCF" in structured.stderr and "E" in structured.stderr
    report = json.loads(structured.stdout)
    validate(report, rank)
    native = dft_rs.static_spectroscopic_derivative([1]*5, POSITIONS.tolist(), DIRECTION.tolist(),
        electric_order=rank, **options)
    np.testing.assert_allclose(report["tensor"], native["tensor"], rtol=0, atol=1e-9)
    plain = subprocess.run(args, capture_output=True, text=True, check=False)
    assert plain.returncode == 0, plain.stderr
    values = [float(line.split()[2]) for line in plain.stdout.splitlines() if line.startswith("tensor ")]
    assert len(values) == 3**rank
    np.testing.assert_allclose(values, report["tensor"], rtol=0, atol=1e-11)


def test_spectroscopy_rejects_inapplicable_reference_and_malformed_paths():
    for extra, error in [(dict(reference="roks"), NotImplementedError),
                         (dict(reference="gks"), NotImplementedError),
                         (dict(electric_order=4), ValueError),
                         (dict(nuclear_direction=[[0.0, 0.0, 1.0]]), ValueError),
                         (dict(unrecognized_control=1), ValueError)]:
        controls = dict(OPTIONS, xc="hf", charge=1, pt2_same_spin=0.17, pt2_opposite_spin=0.29,
                        nuclear_direction=DIRECTION.tolist(), electric_order=2)
        controls.update(extra)
        with pytest.raises(error):
            dft_rs.run_feature("static-spectroscopy", [1]*5, POSITIONS.tolist(), **controls)
