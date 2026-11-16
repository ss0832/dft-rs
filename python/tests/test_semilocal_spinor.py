# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Generic semilocal spinor SCF on all public surfaces."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("xc", ["pbe", "tpss", "r2scan", "m06l"])
def test_semilocal_uniform_spinor_native_generic_ase_and_cli(xc, dimension, tmp_path):
    cell = np.diag([40. if a < dimension else 0. for a in range(3)])
    axes = [a < dimension for a in range(3)]
    positions = np.zeros((1, 3))
    field = [.21, .17, .13 if dimension == 1 else 0.]
    settings = dict(xc=xc, basis="sto-3g", grid_level=6, k_mesh=[1, 1, 1],
                    overlap_cutoff=12., exchange_cutoff=6., grid_cutoff=12.,
                    ewald_alpha=.55, ewald_real_cutoff=16., ewald_reciprocal_cutoff=8.,
                    uniform_magnetic_field=field, magnetic_origin=[1.1, -.4, .9],
                    field_amplitude=.7, nuclear_soc_scale=0.,
                    electronic_temperature=.4 / 3.166811563455607e-6,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11, max_iter=200)
    native = dft_rs.periodic_spinor_scf([1], positions, cell, pbc=axes, **settings)
    generic = dft_rs.run_feature("periodic-spinor-scf", [1], positions, cell=cell, pbc=axes, **settings)
    atoms = Atoms("H", positions=positions * Bohr, cell=cell * Bohr, pbc=axes)
    ase = dft_rs.DftRs(units="atomic").get_periodic_spinor_scf(atoms, **settings, units="atomic")
    xyz = tmp_path / "semilocal-spinor.xyz"
    xyz.write_text("1\nsemilocal spinor reference\nH 0 0 0\n", encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], "periodic-spinor-scf", str(xyz), "--cell",
            ";".join(",".join(f"{v * .529177210903:.17g}" for v in row) for row in cell),
            "--pbc", ",".join(str(int(v)) for v in axes), "--format", "json"]
    for key, value in settings.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        args.extend(["--" + key.replace("_", "-"), str(value)])
    cli = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    for report in (native, generic, ase, cli):
        assert report["converged"] and report["backend_converged"]
        assert "global spin projection" in report["xc_convention"]
        diagnostics = report["xc_diagnostics"]
        assert diagnostics["converged"]
        assert diagnostics["angular_energy_defect"] <= 1e-11
        assert diagnostics["angular_operator_defect"] <= 1e-11
        assert diagnostics["hermiticity_defect"] <= 1e-11
        assert diagnostics["component_energy_defect"] <= 1e-11
        assert diagnostics["angular_refinement_history"]
        assert report["xc_grid_number_defect"] < 1e-5
        assert report["particle_number_error"] < 1e-12
        assert report["orbital_residual"] < 1e-11
        assert report["occupation_consistency"] < 1e-12
        assert report["units"] == native["units"] and report["model"] == native["model"]
        for key in ("energy", "free_energy", "field_energy_derivative", "magnetization",
                    "density_real", "density_imag", "fock_real", "fock_imag",
                    "xc_component_energies", "xc_grid_number_defect"):
            np.testing.assert_allclose(report[key], native[key], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(report["energy"], sum(report["energy_components"].values()), atol=1e-12, rtol=0)
        np.testing.assert_allclose(report["energy_components"]["semilocal"], sum(report["xc_component_energies"]), atol=1e-12, rtol=0)
    print(f"semilocal spinor public {xc} dimension={dimension}: E={native['energy']:.12e}, "
          f"F={native['free_energy']:.12e}, orbital={native['orbital_residual']:.3e}, "
          f"number={native['particle_number_error']:.3e}, grid={native['xc_grid_number_defect']:.3e}, "
          f"angular={native['xc_diagnostics']}")
