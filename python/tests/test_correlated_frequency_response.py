# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Perturbative double-hybrid quasi-energy response on the common API."""

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_fixed_populations import cli_report


@pytest.mark.parametrize("reference", ["rks", "uks"])
@pytest.mark.parametrize("feature, property_name, rank", [
    ("dynamic-polarizability", "polarizability", 1),
    ("quadrupole-polarizability", "quadrupole_polarizability", 2),
    ("multipole-polarizability", "multipole_polarizability", 3),
])
def test_collinear_correlated_frequency_all_routes_and_channels(
    tmp_path, reference, feature, property_name, rank
):
    numbers = [1] * 4
    positions = np.array([
        [0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
        [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28],
    ])
    atoms = Atoms(numbers, positions=positions * Bohr)
    basis = tmp_path / "single_shell.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    options = dict(
        xc="hf", basis=str(basis), dispersion="none", reference=reference,
        charge=int(reference == "uks"), multiplicity=2 if reference == "uks" else 1,
        double_hybrid_target_xc="svwn5", pt2_same_spin=0.28, pt2_opposite_spin=0.49,
        grid_level=0, grid_weight_cutoff=0.0, conv_tol=1e-13, conv_tol_dp=1e-12,
        conv_tol_grad=1e-11, cpu_threads=1,
    )
    if rank > 1:
        options["origin"] = [0.23, -0.50, 0.42]
    if rank > 2:
        options["multipole_rank"] = rank
    options.update(frequency=0.038, frequency_damping=0.006)
    wrapper = feature.replace("-", "_")
    native_route = getattr(dft_rs, wrapper)
    native = native_route(numbers, positions, **options)
    negative = native_route(numbers, positions, **dict(options, frequency=-options["frequency"]))
    cli = cli_report(tmp_path, feature, atoms, options)
    getter = "get_" + wrapper
    atomic = getattr(dft_rs.DftRs(units="atomic"), getter)(atoms, **options)
    if rank > 1:
        physical_options = dict(
            options, origin=np.asarray(options["origin"])*Bohr,
            frequency=options["frequency"]*Hartree,
            frequency_damping=options["frequency_damping"]*Hartree,
            conv_tol=options["conv_tol"]*Hartree, conv_tol_grad=options["conv_tol_grad"]*Hartree,
        )
        physical = getattr(dft_rs.DftRs(), getter)(atoms, **physical_options)
    else:
        # The existing dedicated dipole getter preserves native units.
        physical = getattr(dft_rs.DftRs(), getter)(atoms, **options)
    factor = Bohr**(2*rank)/Hartree if rank > 1 else 1.0
    parent = f"parent_{property_name}"
    target = f"target_{property_name}_correction"
    pair = f"pt2_{property_name}_correction"
    ss_alpha = f"pt2_same_spin_alpha_{property_name}"
    os = f"pt2_opposite_spin_{property_name}"
    ss_beta = f"pt2_same_spin_beta_{property_name}"
    count = 2*rank + 1
    assert native["converged"] and native["shape"] == [count, count]
    assert "quasi-energy" in native["model"]
    for part, sign in (("real", 1), ("imag", -1)):
        def get(name):
            return np.asarray(native[name + "_" + part])

        for name in (property_name, parent, target, pair, ss_alpha, os, ss_beta):
            key = name + "_" + part
            assert native["units"][key] == f"e^2*bohr^{2*rank}/hartree"
            assert get(name).shape == (count, count)
            for actual in (cli, atomic):
                np.testing.assert_allclose(actual[key], get(name), atol=2e-8, rtol=0)
            np.testing.assert_allclose(np.asarray(physical[key])/factor, get(name), atol=2e-8, rtol=0)
            np.testing.assert_allclose(negative[key], sign*get(name), atol=2e-8, rtol=0)
        np.testing.assert_allclose(get(property_name), get(parent)+get(target)+get(pair),
                                   atol=2e-10, rtol=0)
        np.testing.assert_allclose(
            get(pair), options["pt2_same_spin"]*(get(ss_alpha)+get(ss_beta))
            + options["pt2_opposite_spin"]*get(os), atol=2e-10, rtol=0,
        )
    assert np.linalg.norm(native[pair + "_imag"]) > 1e-12
    diagnostics = native["frequency_pair_diagnostics"]
    assert len(diagnostics) == count**2
    for component in diagnostics:
        assert np.max(component["frame_inverse_defects"]) < 1e-8
        assert np.max(component["frame_projector_defects"]) < 1e-8
        for name in ("left_pair_relative_residuals", "right_pair_relative_residuals"):
            assert np.max(component[name]) < 1e-12
    print(f"{reference} damped {property_name}", native)
