# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import json
import os
import subprocess

from ase import Atoms
import numpy as np
import pytest

from dft_rs import DftRs, molecular_medium_optics
from dft_rs.units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


@pytest.mark.parametrize("magnetic", [False, True], ids=["dielectric", "magnetic-kerr"])
def test_molecular_medium_optics_on_public_routes(tmp_path, magnetic):
    positions = np.array(
        [[0.13, -0.21, 0.19], [1.57, 0.16, -0.08], [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]]
        if magnetic else [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
    )
    count = len(positions)
    atoms = Atoms("H" * count, positions=positions * BOHR_IN_ANGSTROM)
    angle = 0.47
    c, s = np.cos(angle), np.sin(angle)
    controls = dict(
        xc="hf", basis="sto-3g", frequency=0.12, frequency_damping=0.008,
        medium_number_density=0.002, slab_thickness=1400.0,
        incidence_angle=0.38, incident_refractive_index=1.1, exit_refractive_index=1.3,
        medium_orientation=[[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
    )
    if magnetic:
        controls.update(
            reference="uks", charge=3, multiplicity=2, frequency=0.18, frequency_damping=0.028,
            medium_magnetic_field=[0.0, 0.0, 0.003], gauge_origin=[0.23, -0.50, 0.42],
            grid_weight_cutoff=0.0,
        )
    native = molecular_medium_optics([1] * count, positions.tolist(), **controls)
    calculator = DftRs()
    ase = calculator.get_molecular_medium_optics(atoms, units="atomic", **controls)
    xyz = tmp_path / "hydrogen.xyz"
    rows = ["H " + " ".join(format(v, ".17g") for v in position) for position in atoms.positions]
    xyz.write_text("\n".join([str(count), "Molecular medium optics", *rows, ""]), encoding="ascii")
    cli = os.environ.get("DFT_RS_CLI")
    assert cli, "DFT_RS_CLI is required for installed medium optics acceptance"
    args = [cli, "molecular-medium-optics", str(xyz), "--format", "json"]
    for name, value in controls.items():
        args.extend(["--" + name.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(args, check=True, capture_output=True, encoding="utf-8")
    assert "residual" in process.stderr
    cli_report = json.loads(process.stdout)
    keys = [
        "relative_permittivity_real", "relative_permittivity_imag",
        "normal_refractive_index_real", "normal_refractive_index_imag",
        "normal_absorption_coefficient", "reflection_jones_real", "reflection_jones_imag",
        "transmission_jones_real", "transmission_jones_imag", "reflectance", "transmittance",
        "absorptance", "local_field_residual", "slab_boundary_residual", "flux_balance_residual",
    ]
    if magnetic:
        keys.extend([
            "response_residual", "faraday_tensor_real", "faraday_tensor_imag",
            "magnetic_kerr_rotation", "magnetic_kerr_ellipticity",
            "reversed_field_flux_balance_residual", "reversed_field_slab_boundary_residual",
        ])
        assert np.max(np.abs(native["magnetic_kerr_rotation"])) > 1e-9
    else:
        keys.append("response_residuals")
    for report in (ase, cli_report):
        assert report["converged"] and native["converged"]
        for key in keys:
            assert report["units"][key] == native["units"][key]
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=2e-10)
    physical = dict(controls)
    for key in ("frequency", "frequency_damping"):
        physical[key] *= HARTREE_IN_EV
    physical["medium_number_density"] /= BOHR_IN_ANGSTROM**3
    physical["slab_thickness"] *= BOHR_IN_ANGSTROM
    if magnetic:
        physical["gauge_origin"] = (np.asarray(physical["gauge_origin"]) * BOHR_IN_ANGSTROM).tolist()
    ase = calculator.get_molecular_medium_optics(atoms, units="ase", **physical)
    np.testing.assert_allclose(ase["normal_absorption_coefficient"],
                               np.asarray(native["normal_absorption_coefficient"]) / BOHR_IN_ANGSTROM,
                               rtol=0, atol=2e-10)
    for key in ("reflectance", "transmittance", "absorptance"):
        np.testing.assert_allclose(ase[key], native[key], rtol=0, atol=2e-10)
    if magnetic:
        for key in ("magnetic_kerr_rotation", "magnetic_kerr_ellipticity"):
            np.testing.assert_allclose(ase[key], native[key], rtol=0, atol=2e-10)
    print("Installed magnetic molecular medium optics" if magnetic else "Installed molecular medium optics",
          json.dumps(native, sort_keys=True, allow_nan=False))
