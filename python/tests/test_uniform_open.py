# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Uniform open-boundary continuum fields on every public surface."""
import json
import os
import subprocess

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
import dft_rs


def second_moment():
    """Independent normalized STO-3G H radial Gaussian second moment."""
    exponents = np.array([3.42525091, .62391373, .16885540])
    coefficients = np.array([.15432897, .53532814, .44463454])
    weights = coefficients * (2 * exponents / np.pi) ** .75
    overlap = 0.0
    moment = 0.0
    for i, a in enumerate(exponents):
        for j, b in enumerate(exponents):
            value = weights[i] * weights[j] * (np.pi / (a + b)) ** 1.5
            overlap += value
            moment += value / (2 * (a + b))
    return moment / overlap


@pytest.mark.parametrize("axes", [[True, False, False], [False, True, False],
                                  [False, False, True], [True, True, False]])
@pytest.mark.parametrize("spinor", [True,False])
def test_uniform_continuum_field_native_generic_ase_cli_and_closed_oracle(axes, spinor, tmp_path):
    cell = np.array([[9., .4, .7], [.9, 9.5, .5], [1.2, .6, 10.]])
    dimension = sum(axes)
    seed = np.array([.021, -.017, .013])
    active = cell[np.array(axes)]
    if dimension == 1:
        t = active[0] / np.linalg.norm(active[0])
        field = seed
        parallel = t * t.dot(field)
        # Columns of the antisymmetric cross-product matrix.
        cross = np.column_stack([np.cross(parallel, e) for e in np.eye(3)])
        matrix = np.outer(t, np.cross(t, field)) + .5 * cross
    else:
        normal = np.cross(*active)
        normal /= np.linalg.norm(normal)
        field = seed - normal * normal.dot(seed)
        matrix = np.outer(np.cross(field, normal), normal)
    cell[np.logical_not(axes)] = 0.0
    positions = np.zeros((1, 3))
    settings = dict(xc="hf", basis="sto-3g", k_mesh=[1, 1, 1],
                    overlap_cutoff=3., exchange_cutoff=3., grid_cutoff=8.,
                    ewald_alpha=.55, ewald_real_cutoff=12., ewald_reciprocal_cutoff=7.,
                    uniform_magnetic_field=field.tolist(), magnetic_origin=[1.1, -.4, .9],
                    nuclear_soc_scale=0., field_amplitude=.4,
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11, max_iter=150)
    feature="periodic-spinor-scf" if spinor else "periodic-field-scf"
    if not spinor:
        settings.pop("nuclear_soc_scale")
        settings["multiplicity"]=2
    native = getattr(dft_rs,feature.replace("-","_"))([1], positions, cell, pbc=axes, **settings)
    generic = dft_rs.run_feature(feature, [1], positions, cell=cell, pbc=axes, **settings)
    atoms = Atoms("H", positions=positions * Bohr, cell=cell * Bohr, pbc=axes)
    ase = getattr(dft_rs.DftRs(units="atomic"),"get_"+feature.replace("-","_"))(atoms, **settings)
    xyz = tmp_path / "uniform-open.xyz"
    xyz.write_text("1\nuniform open reference\nH 0 0 0\n", encoding="utf-8")
    args = [os.environ["DFT_RS_CLI"], feature, str(xyz), "--cell",
            ";".join(",".join(f"{v * Bohr:.17g}" for v in row) for row in cell),
            "--pbc", ",".join(str(int(v)) for v in axes), "--format", "json"]
    for key, value in settings.items():
        if isinstance(value, list):
            value = ",".join(map(str, value))
        args.extend(["--" + key.replace("_", "-"), str(value)])
    cli = json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)
    for report in (native, generic, ase, cli):
        assert report["converged"] and report["dimension"] == dimension
        assert not report["orbital_conjugation_constraint"]
        assert report["particle_number_error"] < 1e-12
        assert report["orbital_residual"] < 1e-11
        assert report["raw_field_hermitian_residual"] < 1e-12
        assert report["units"] == native["units"] and report["model"] == native["model"]
        for key in ("energy", "free_energy", "field_energy_derivative",
                    "density_real", "density_imag", "fock_real", "fock_imag"):
            np.testing.assert_allclose(report[key], native[key], rtol=1e-10, atol=1e-11)
    contact = np.sum(matrix * matrix) * second_moment()
    zeeman = -.5 * np.linalg.norm(field) if spinor else 0.0
    expected_orbital = .5 * .4 ** 2 * contact
    orbital=native["energy_components"]["orbital_field" if spinor else "external_field"]
    np.testing.assert_allclose(orbital, expected_orbital, atol=1e-12, rtol=1e-7)
    if spinor:
        np.testing.assert_allclose(native["energy_components"]["spin_soc"], .4 * zeeman, atol=1e-12, rtol=0)
    np.testing.assert_allclose(native["field_energy_derivative"], .4 * contact + zeeman, atol=1e-12, rtol=1e-7)
    if spinor:
        np.testing.assert_allclose(native["magnetization"], -field / np.linalg.norm(field), atol=1e-11, rtol=0)
    np.testing.assert_allclose(native["energy"], sum(native["energy_components"].values()), atol=1e-13, rtol=0)
    print(f"uniform open public spinor={spinor} axes={axes}: orbital={orbital:.12e}, oracle defect={abs(orbital-expected_orbital):.3e}, "
          f"derivative={native['field_energy_derivative']:.12e}, number={native['particle_number_error']:.3e}, "
          f"orbital residual={native['orbital_residual']:.3e}")


def test_uniform_open_controls_reject_incompatible_fields_and_unused_origins():
    common = dict(xc="hf", basis="sto-3g", nuclear_soc_scale=0.)
    cell = np.diag([9., 9., 0.])
    with pytest.raises(ValueError, match="magnetic Bloch"):
        dft_rs.periodic_spinor_scf([1], [[0, 0, 0]], cell, pbc=[True, True, False],
                                 uniform_magnetic_field=[0., 0., .2], **common)
    with pytest.raises(ValueError, match="requires uniform"):
        dft_rs.periodic_spinor_scf([1], [[0, 0, 0]], cell, pbc=[True, True, False],
                                 magnetic_origin=[1., 0., 0.], **common)
    with pytest.raises(ValueError, match="Fourier orbital"):
        dft_rs.periodic_spinor_scf([1], [[0, 0, 0]], cell, pbc=[True, True, False],
                                 uniform_magnetic_field=[.2, 0., 0.], field_vector_real=[.1, 0., 0.], **common)
