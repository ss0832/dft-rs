# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""C1 molecular positron electric and mass response through the public driver."""

import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as BOHR, HARTREE_IN_EV as EH


def test_c1_ammonia_positron_quadrupole_and_damped_dipole_public_routes(tmp_path):
    numbers = [7, 1, 1, 1]
    positions = np.array([
        [0.13, -0.21, 0.09], [1.61, 0.77, -0.29],
        [-1.34, 0.61, 0.33], [0.31, -1.74, 0.58],
    ])
    positron = dict(kind="positron", position=[0.45, -0.17, 1.8],
                    exponents=[0.12, 0.38], lmax=1, correlation="drummond-2011")
    options = dict(xc="m06", reference="rks", basis="sto-3g", dispersion="none",
                   grid_level=1, grid_weight_cutoff=0.0, conv_tol=1e-13,
                   conv_tol_grad=1e-11, conv_tol_dp=1e-11, species_tolerance=1e-11,
                   species_max_iterations=500, cpu_threads=1, particles=[positron],
                   quantum_position_constraint=True, species_quadrupole_polarizability=True,
                   frequencies=[0.036, -0.036], frequency_damping=0.004)
    native = dft_rs.multicomponent(numbers, positions.tolist(), **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR)
    atomic = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
    source = tmp_path / "ammonia.xyz"
    source.write_text("4\nC1 ammonia\n" + "".join(
        f"{symbol} " + " ".join(f"{x:.17g}" for x in point) + "\n"
        for symbol, point in zip(atoms.get_chemical_symbols(), atoms.positions)
    ), encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "multicomponent", str(source), "--format", "json"]
    for name, value in options.items():
        command.extend(("--" + name.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    process = subprocess.run(command, check=True, capture_output=True, encoding="utf-8")
    assert "[scf] iter" in process.stderr and "SCF residual" in process.stderr
    cli = json.loads(process.stdout)
    for report in (native, atomic, cli):
        assert report["converged"]
        q = report["quadrupole"]
        chi = report["dynamic_polarizability"]
        assert report["units"]["quadrupole.polarizability"] == "e^2*bohr^4/hartree"
        assert report["units"]["dynamic_polarizability.tensor_imaginary"] == "bohr^3"
        np.testing.assert_allclose(q["polarizability"], np.transpose(q["polarizability"]),
                                   atol=1e-8, rtol=0)
        np.testing.assert_allclose(q["dipole_from_field_gradient"],
                                   np.transpose(q["field_conjugate_moment_from_field"]),
                                   atol=1e-8, rtol=0)
        np.testing.assert_allclose(chi["tensor_real"][0], chi["tensor_real"][1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(chi["tensor_imaginary"][0], -np.asarray(chi["tensor_imaginary"][1]),
                                   atol=1e-10, rtol=0)
        assert np.max(np.abs(chi["tensor_imaginary"][0])) > 1e-6
        for group, fields in (("quadrupole", ("moment", "polarizability",
                                             "dipole_from_field_gradient",
                                             "field_conjugate_moment_from_field",
                                             "dipole_position_response_residuals",
                                             "quadrupole_position_response_residuals")),
                              ("dynamic_polarizability", ("tensor_real", "tensor_imaginary",
                                                          "response_residuals"))):
            for key in fields:
                np.testing.assert_allclose(report[group][key], native[group][key],
                                           atol=1e-8, rtol=0)
    print("C1 NH3 plus positron static quadrupole", native["quadrupole"])
    print("C1 NH3 plus positron damped dipole", native["dynamic_polarizability"])


def test_c1_ammonia_positron_static_response_agrees_across_public_routes(tmp_path):
    numbers = [7, 1, 1, 1]
    positions = np.array([
        [0.13, -0.21, 0.09],
        [1.61, 0.77, -0.29],
        [-1.34, 0.61, 0.33],
        [0.31, -1.74, 0.58],
    ])
    positron = dict(kind="positron", position=[0.45, -0.17, 1.8],
                    exponents=[0.12, 0.38], lmax=1, correlation="drummond-2011")
    options = dict(xc="m06", reference="rks", basis="sto-3g", dispersion="none",
                   grid_level=1, grid_weight_cutoff=0.0, conv_tol=1e-13,
                   conv_tol_grad=1e-11, conv_tol_dp=1e-11, species_tolerance=1e-11,
                   species_max_iterations=500, cpu_threads=1, particles=[positron],
                   species_polarizability=True, quantum_mass_response=True)
    native = dft_rs.multicomponent(numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR)
    ase = dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
    source = tmp_path / "ammonia.xyz"
    source.write_text(
        "4\nC1 ammonia\n" + "".join(
            f"{symbol} " + " ".join(f"{x:.17g}" for x in point) + "\n"
            for symbol, point in zip(atoms.get_chemical_symbols(), atoms.positions)
        ),
        encoding="utf-8",
    )
    command = [os.environ["DFT_RS_CLI"], "multicomponent", str(source), "--format", "json"]
    for name, value in options.items():
        command.extend(["--" + name.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, check=True, capture_output=True, encoding="utf-8")
    assert "[scf] iter" in process.stderr and "SCF residual" in process.stderr
    cli = json.loads(process.stdout)
    alpha = native["polarizability"]
    mass = native["mass_response"]
    mass_keys = ("gradient", "hessian", "response_residuals", "reference_residuals",
                 "position_response_residuals")
    assert np.asarray(alpha["tensor"]).shape == (3, 3)
    assert np.asarray(mass["gradient"]).shape == (1,)
    assert np.asarray(mass["hessian"]).shape == (1,)
    for report in (native, ase, cli):
        assert report["converged"] and report["polarizability"]["converged"]
        assert report["units"]["polarizability.tensor"] == "bohr^3"
        np.testing.assert_allclose(report["polarizability"]["tensor"], alpha["tensor"],
                                   rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(report["polarizability"]["response_residuals"],
                                   alpha["response_residuals"], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(report["energy"], native["energy"], rtol=0.0, atol=1e-10)
        other = report["mass_response"]
        assert other["converged"] and other["hessian_shape"] == [1, 1]
        assert report["units"]["mass_response.gradient"] == "hartree/m_e (m_e is the electron mass)"
        assert report["units"]["mass_response.hessian"] == "hartree/m_e^2"
        assert report["units"]["mass_response.response_residuals"] == "dimensionless"
        for key in mass_keys:
            np.testing.assert_allclose(other[key], mass[key], rtol=0.0, atol=1e-9)
        assert max(abs(value) for value in other["response_residuals"]) < 1e-8
    converted = dict(options, conv_tol=options["conv_tol"] * EH,
                     conv_tol_grad=options["conv_tol_grad"] * EH, particles=[
                         dict(positron, position=(np.asarray(positron["position"]) * BOHR).tolist(),
                              exponents=(np.asarray(positron["exponents"]) / BOHR**2).tolist()),
                     ])
    physical = dft_rs.DftRs().get_multicomponent(atoms, **converted)
    np.testing.assert_allclose(physical["polarizability"]["tensor"],
                               np.asarray(alpha["tensor"]) * BOHR**3, rtol=0.0, atol=1e-9)
    assert physical["units"]["mass_response.gradient"] == "eV/m_e (m_e is the electron mass)"
    assert physical["units"]["mass_response.hessian"] == "eV/m_e^2"
    assert physical["units"]["mass_response.position_response_residuals"].startswith("angstrom/m_e ")
    for key, factor in zip(mass_keys, (EH, EH, 1.0, 1.0, BOHR)):
        np.testing.assert_allclose(physical["mass_response"][key],
                                   np.asarray(mass[key]) * factor, rtol=0.0, atol=1e-9)
    max_mass_differences = {
        key: max(np.max(np.abs(np.asarray(other["mass_response"][key]) - np.asarray(mass[key])))
                 for other in (ase, cli))
        for key in mass_keys
    }
    annihilation = native["annihilation"]["positrons"][0]
    print(f"M06 NH3+positron E={native['energy']:.16e} alpha={alpha['tensor']} "
          f"response_residuals={alpha['response_residuals']} "
          f"rate_per_ns={annihilation['rate_per_ns']:.16e} "
          f"lifetime_ps={annihilation['lifetime_ps']:.16e}")
    print(f"M06 NH3+positron mass_gradient={mass['gradient']} mass_hessian={mass['hessian']} "
          f"mass_response_residuals={mass['response_residuals']} "
          f"mass_reference_residuals={mass['reference_residuals']} "
          f"mass_position_residuals={mass['position_response_residuals']} "
          f"max_public_mass_differences={max_mass_differences}")


def test_positron_periodic_reports_missing_capability_across_public_routes(tmp_path):
    positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    cell = 8.0 * np.eye(3)
    positron = dict(kind="positron", position=[0.4, 0.3, 0.2],
                    exponents=[0.15, 0.4], correlation="drummond-2011")
    options = dict(xc="m06", basis="sto-3g", dispersion="none",
                   particles=[positron], species_polarizability=True)
    calculator = dft_rs.DftRs(units="atomic")
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text(f"2\nHydrogen\nH 0 0 {-0.7 * BOHR}\nH 0 0 {0.7 * BOHR}\n",
                   encoding="utf-8")
    reason = "periodic `multicomponent` is not implemented"
    for active in range(1, 4):
        pbc = [axis < active for axis in range(3)]
        with pytest.raises(NotImplementedError, match=reason) as native_error:
            dft_rs.multicomponent([1, 1], positions, cell=cell.tolist(),
                                  pbc=pbc, **options)

        atoms = Atoms("H2", positions=np.asarray(positions) * BOHR,
                      cell=cell * BOHR, pbc=pbc)
        if active == 3:
            with pytest.raises(NotImplementedError, match=reason) as ase_error:
                calculator.get_multicomponent(atoms, **options)
        else:
            with pytest.raises(ValueError, match="needs a fully periodic Atoms object") as ase_error:
                calculator.get_multicomponent(atoms, **options)

        command = [os.environ["DFT_RS_CLI"], "multicomponent", str(xyz),
                   "--format", "json", "--cell", str(8.0 * BOHR),
                   "--pbc", ",".join(str(int(axis)) for axis in pbc),
                   "--xc", "m06", "--basis", "sto-3g", "--dispersion", "none",
                   "--particles", json.dumps([positron]), "--species-polarizability", "true"]
        process = subprocess.run(command, check=False, capture_output=True, encoding="utf-8")
        assert process.returncode == 1
        assert reason in process.stderr
        assert not process.stdout
        print(f"periodic axes={pbc} native={native_error.value} "
              f"ASE={ase_error.value} CLI status={process.returncode} "
              f"stderr={process.stderr.strip()}")
