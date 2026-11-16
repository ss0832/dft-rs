# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Full complex hybrid density and dielectric routes, with no tensor copying."""
import os
import re
import subprocess

import numpy as np
import pytest
from ase import Atoms

from dft_rs import DftRs, periodic_density_response, periodic_dielectric
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr


POSITIONS = np.array([[0.1, -0.2, 0.3], [1.4, 0.31, -0.17],
                      [-0.23, 1.7, 0.21], [0.38, -0.13, 1.89]])
CELL = np.array([[8.7, 0.31, 0.13], [0.2, 9.2, 0.43], [0.11, 0.27, 9.7]])


def density_array(report):
    return np.asarray([[np.asarray(k["real"]) + 1j*np.asarray(k["imag"])
                        for k in spin] for spin in report["density"]])


@pytest.mark.parametrize("xc", ["hf", "pbe", "pbe0", "cam-b3lyp"])
@pytest.mark.parametrize("frequency", ["static", "retarded", "imaginary"])
@pytest.mark.parametrize("k_mesh", [(1, 1, 1), (2, 1, 1)])
def test_hybrid_finite_q_native_ase_and_canonical_cli(xc, frequency, k_mesh, tmp_path):
    basis = tmp_path / "primitive.g94"
    basis.write_text("H 0\nS 1 1.0\n0.91 1.0\n****\n")
    scf = dict(basis=str(basis), xc=xc, k_mesh=k_mesh, overlap_cutoff=4.0,
               exchange_cutoff=4.0, grid_level=0, grid_shell=1,
               grid_weight_cutoff=0.0, max_iter=500, conv_tol=1e-13,
               conv_tol_dp=1e-12, conv_tol_grad=1e-11)
    response = dict(q=(0.043, 0.017, -0.021), reciprocal_cutoff=1.0,
                    response_tol=1e-12,
                    omega=0.0 if frequency == "static" else 0.13,
                    damping=0.04 if frequency == "retarded" else 0.0,
                    imaginary=frequency == "imaginary")
    native = periodic_density_response([1]*4, POSITIONS, CELL, **scf, **response)
    dielectric = periodic_dielectric([1]*4, POSITIONS, CELL, **scf, **response)
    atoms = Atoms("H4", positions=POSITIONS*Bohr, cell=CELL*Bohr, pbc=True)
    calculator = DftRs(units="atomic", **scf)
    adapted = calculator.get_periodic_density_response(atoms, **response)
    adapted_dielectric = calculator.get_periodic_dielectric(atoms, **response)
    for report in [native, dielectric, adapted, adapted_dielectric]:
        assert report["converged"]
        assert max(report["residuals"]) <= response["response_tol"]
        assert "exchange" in report["model"]
        assert report["reference"]["particle_number_error"] < 1e-9
    np.testing.assert_allclose(density_array(native), density_array(adapted),
                               atol=1e-11, rtol=0, equal_nan=False)
    for key in ["screened_chi_real", "screened_chi_imag",
                "inverse_epsilon_real", "inverse_epsilon_imag"]:
        np.testing.assert_allclose(dielectric[key], adapted_dielectric[key],
                                   atol=1e-11, rtol=0, equal_nan=False)
    chi = complex(native["fourier_density_real"][0], native["fourier_density_imag"][0])
    expected_chi = complex(dielectric["screened_chi_real"][0][0],
                           dielectric["screened_chi_imag"][0][0])
    assert abs(chi-expected_chi) < 1e-11
    wave = np.asarray(dielectric["reciprocal_waves"][0])
    inverse = complex(dielectric["inverse_epsilon_real"][0][0],
                      dielectric["inverse_epsilon_imag"][0][0])
    assert abs(inverse-(1+4*np.pi*chi/np.dot(wave, wave))) < 1e-11

    xyz = tmp_path / "c1.xyz"
    xyz.write_text("4\nC1 periodic hybrid response\n" + "".join(
        "H " + " ".join(f"{v*Bohr:.17g}" for v in row) + "\n"
        for row in POSITIONS))
    command = [os.environ["DFT_RS_CLI"], "periodic-response", str(xyz),
               "--cell", ";".join(",".join(f"{v*Bohr:.17g}" for v in row) for row in CELL),
               "--basis", str(basis), "--xc", xc, "--overlap-cutoff", "4",
               "--k-mesh", ",".join(map(str, k_mesh)),
               "--exchange-cutoff", "4", "--grid-level", "0", "--grid-shell", "1",
               "--grid-weight-cutoff", "0", "--max-iter", "500",
               "--conv-tol", "1e-13", "--conv-tol-dp", "1e-12", "--conv-tol-grad", "1e-11",
               "--response-q", ",".join(map(str, response["q"])),
               "--response-omega", str(response["omega"]),
               "--response-damping", str(response["damping"]),
               "--response-tol", str(response["response_tol"]),
               "--reciprocal-cutoff", "1.0"]
    if response["imaginary"]:
        command.append("--imaginary-frequency")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert native["model"] in result.stdout
    assert "[pbc-scf] iter" in result.stderr
    assert "E=" in result.stderr and "residual=" in result.stderr
    rows = [line.split() for line in result.stdout.splitlines()]
    entries = [row for row in rows if row and row[0] == "density"]
    expected = density_array(native)
    assert np.isfinite(expected).all()
    assert len(entries) == expected.size
    seen = set()
    differences = []
    for row in entries:
        index = tuple(map(int, row[1:5]))
        assert index not in seen, "duplicate CLI AO density tensor entry"
        seen.add(index)
        differences.append(abs(expected[index]-complex(float(row[5]), float(row[6]))))
    assert seen == set(np.ndindex(expected.shape)), "incomplete CLI AO density tensor"
    assert np.isfinite(differences).all()
    cli_residuals = re.search(r"Response residuals:\s*\[([^]]+)\]", result.stdout)
    assert cli_residuals is not None
    cli_residuals = [float(value) for value in cli_residuals.group(1).split(",")]
    assert max(cli_residuals) <= response["response_tol"]
    print(xc, frequency, k_mesh, "complete_AO_rows", len(seen), "maximum_full_AO_map", max(differences),
          "native_response", native["residuals"], "CLI_response", cli_residuals,
          "native_reference", native["reference"])
    assert max(differences) < 1e-11
