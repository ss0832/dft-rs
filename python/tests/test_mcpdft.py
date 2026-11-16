# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""CASSCF and MC-PDFT through the native, generic, ASE and CLI routes."""
import json
import os
import subprocess
import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
import dft_rs


@pytest.mark.parametrize("feature,native_name,ase_name,extra", [
    ("casscf", "casscf", "get_casscf", {}),
    ("mcpdft", "mcpdft", "get_mcpdft", {"on_top_functional": "ftpbe", "pdft_method": "cms-pdft"}),
    ("casscf", "casscf", "get_casscf", {"cas": [2, 2], "cas_roots": 1, "derivative_order": 2}),
    ("mcpdft", "mcpdft", "get_mcpdft", {"cas": [2, 2], "cas_roots": 1, "derivative_order": 2,
                                      "on_top_functional": "ftsvwn5"}),
])
def test_active_space_public_routes(tmp_path, feature, native_name, ase_name, extra):
    numbers = [3, 1]
    positions = np.array([[0., 0., 0.], [0., 0., 3.015]])
    options = dict(xc="hf", basis="sto-3g", cas=[2, 3], cas_roots=2, conv_tol=1e-12,
                   conv_tol_dp=1e-11, dispersion="none")
    options.update(extra)
    if options.get("derivative_order") == 2:
        options.update(casscf_gradient_tol=1e-11, casscf_energy_tol=1e-13, ci_tol=1e-13)
    native = getattr(dft_rs, native_name)(numbers, positions, **options)
    generic = dft_rs.run_feature(feature, numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=positions * Bohr)
    atomic = getattr(dft_rs.DftRs(units="atomic"), ase_name)(atoms, **options)
    # The same physical tolerances in eV, so that both unit modes solve the
    # same problem and only the unit conversion is compared.
    in_ev = dict(options, conv_tol=options["conv_tol"] * Hartree)
    for key in ["casscf_gradient_tol", "casscf_energy_tol", "ci_tol"]:
        if key in options:
            in_ev[key] = options[key] * Hartree
    electronvolt = getattr(dft_rs.DftRs(), ase_name)(atoms, **in_ev)
    executable = os.environ.get("DFT_RS_CLI")
    assert executable, "DFT_RS_CLI is required for the canonical CLI acceptance"
    xyz = tmp_path / "lithium-hydride.xyz"
    native_length = 1 / dft_rs.native.BOHR_PER_ANGSTROM
    xyz.write_text("2\nactive-space acceptance\n" + "".join(
        f"{symbol} " + " ".join(f"{v * native_length:.17g}" for v in row) + "\n"
        for symbol, row in zip(("Li", "H"), positions)), encoding="utf-8")
    argv = [executable, feature, str(xyz), "--format", "json"]
    for key, value in options.items():
        argv.extend(["--" + key.replace("_", "-"),
                     json.dumps(value) if isinstance(value, list) else str(value)])
    executed = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert "[scf]" in executed.stderr and "[casscf]" in executed.stderr
    cli = json.loads(executed.stdout)
    keys = ["cas_energies", "average_energy", "natural_occupations"]
    if feature == "mcpdft":
        keys += ["pdft_energies", "state_on_top", "state_coulomb", "effective_hamiltonian"]
    if options.get("derivative_order") == 2:
        keys += ["gradients", "hessians", "derivative_stationarity", "derivative_solve_residual"]
    for other in [generic, atomic, cli]:
        for key in keys:
            np.testing.assert_allclose(other[key], native[key], atol=1e-10, rtol=0)
        assert other["units"] == native["units"]
    energy_keys = ["cas_energies", "average_energy"] + (["pdft_energies"] if feature == "mcpdft" else [])
    for key in energy_keys:
        np.testing.assert_allclose(np.asarray(electronvolt[key]), np.asarray(native[key]) * Hartree,
                                   rtol=1e-12, atol=0)
    if options.get("derivative_order") == 2:
        for key, factor in [("gradients", Hartree / Bohr), ("hessians", Hartree / Bohr**2)]:
            np.testing.assert_allclose(electronvolt[key], np.asarray(native[key]) * factor,
                                       rtol=1e-10, atol=1e-10)
        assert native["derivative_solve_residual"] < 1e-10
        print(f"{feature} derivative stationarity={native['derivative_stationarity']:.12e} "
              f"solve_residual={native['derivative_solve_residual']:.12e}")
    assert native["orbitals_converged"]
    assert native["orbital_gradient"] < 1e-8
    assert max(native["ci_residuals"]) < 1e-8
    np.testing.assert_allclose(native["spin_squared"], np.zeros(options["cas_roots"]), atol=1e-8)
