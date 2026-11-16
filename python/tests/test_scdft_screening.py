# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Microscopic SCDFT Coulomb generation through the shared public driver."""

import json
import os
import subprocess

import numpy as np
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as BOHR, HARTREE_IN_EV as EH


def test_screened_coulomb_feeds_the_gamma_phonon_adapter_across_public_routes(tmp_path):
    numbers = [1, 1]
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
    cell = np.eye(3) * 9
    options = dict(
        xc="pbe", unrestricted=False, basis="sto-3g", grid_level=1, grid_weight_cutoff=0.0,
        grid_shell=1, overlap_cutoff=4.0, conv_tol=1e-11, conv_tol_grad=1e-10,
        conv_tol_dp=1e-10, cpu_threads=1, electronic_temperature=3000.0,
        eliashberg_t_min_k=10.0, scdft_pairing=True, scdft_screening="rpa-neutral",
        response_reciprocal_cutoff=0.95,
    )
    native = dft_rs.electron_phonon(numbers, positions, cell, **options)
    atoms = Atoms(numbers=numbers, positions=positions * BOHR, cell=cell * BOHR, pbc=True)
    calculator = dft_rs.DftRs(units="atomic")
    ase = calculator.get_electron_phonon(atoms, **options)
    physical_options = dict(
        options, overlap_cutoff=options["overlap_cutoff"] * BOHR,
        conv_tol=options["conv_tol"] * EH, conv_tol_grad=options["conv_tol_grad"] * EH,
        response_reciprocal_cutoff=options["response_reciprocal_cutoff"] / BOHR,
    )
    physical = dft_rs.DftRs().get_electron_phonon(atoms, **physical_options)
    path = tmp_path / "hydrogen.xyz"
    path.write_text(
        "2\nH2 periodic pairing\n" + "".join(
            "H " + " ".join(f"{coordinate:.17g}" for coordinate in point) + "\n"
            for point in atoms.positions
        ), encoding="utf-8",
    )
    command = [
        os.environ["DFT_RS_CLI"], "electron-phonon", str(path), "--cell",
        ",".join(f"{value:.17g}" for value in atoms.cell.array.flatten()),
        "--pbc", "1,1,1", "--format", "json",
    ]
    for key, value in options.items():
        command.extend(("--" + key.replace("_", "-"),
                        value if isinstance(value, str) else json.dumps(value)))
    process = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "[pbc-scf] iter" in process.stderr and "E=" in process.stderr
    assert "[pbc-scf] physical final check" in process.stderr and "residual=" in process.stderr
    cli = json.loads(process.stdout)
    inputs = native["scdft_inputs"]
    screened = np.asarray(inputs["screened_pair_coulomb"])
    assert screened.shape == (2, 2)
    assert np.min(screened) >= 0.0
    np.testing.assert_allclose(screened, screened.T, rtol=0.0, atol=1e-11)
    diag = native["scdft_screening_diagnostics"]
    assert diag["reciprocal_modes"] == [6]
    assert diag["gamma_head"] is None
    assert "fixed-number neutral cell" in diag["gamma_ensemble"]
    assert max(diag["inverse_residuals"]) < 1e-11
    assert diag["electron_number_residual"] < 1e-10
    for report in (ase, cli):
        assert report["scdft_screening"] == "rpa-neutral"
        assert report["units"]["scdft_inputs.screened_pair_coulomb"] == "hartree"
        np.testing.assert_allclose(report["scdft_inputs"]["screened_pair_coulomb"],
                                   screened, rtol=1e-8, atol=1e-11)
        np.testing.assert_allclose(report["scdft_pairing_kernel"], native["scdft_pairing_kernel"],
                                   rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(physical["scdft_inputs"]["screened_pair_coulomb"],
                               screened * EH, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        physical["scdft_screening_diagnostics"]["gamma_compressibility"],
        diag["gamma_compressibility"] / (EH * BOHR**3), rtol=1e-8, atol=1e-12,
    )
    component = dft_rs.scdft_pairing_kernel(**inputs)
    np.testing.assert_allclose(component["pairing_kernel"], native["scdft_pairing_kernel"],
                               rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(component["mass_renormalization"],
                               native["scdft_mass_renormalization"], rtol=1e-11, atol=1e-12)
    print("SCDFT H2 automatic RPA Coulomb", screened.tolist(), "diagnostics", diag)
