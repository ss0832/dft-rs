# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""One response, mode convention and unit conversion on the installed APIs."""

import json

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr, HARTREE_IN_EV as Hartree
from test_fixed_populations import cli_report


@pytest.mark.parametrize("rank", [3, 4, 5])
def test_racah_multipole_polarizability_all_public_routes(tmp_path, rank):
    numbers = [8, 1, 1]
    positions = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08], [-0.19, 1.43, 0.37]])
    atoms = Atoms(numbers, positions=positions * Bohr)
    controls = dict(xc="hf", basis="sto-3g", dispersion="none",
                    multipole_rank=rank, origin=[0.11, -0.17, 0.26],
                    frequency=0.045, frequency_damping=0.005, cpu_threads=1)
    native = dft_rs.multipole_polarizability(numbers, positions, **controls)
    cli = cli_report(tmp_path, "multipole-polarizability", atoms, controls)
    atomic = dft_rs.DftRs(units="atomic").get_multipole_polarizability(atoms, **controls)
    physical = dft_rs.DftRs().get_multipole_polarizability(
        atoms, **dict(controls, origin=np.asarray(controls["origin"])*Bohr,
                      frequency=controls["frequency"]*Hartree,
                      frequency_damping=controls["frequency_damping"]*Hartree)
    )
    count = 2 * rank + 1
    assert native["multipole_rank"] == rank
    assert native["shape"] == [count, count]
    assert native["mode_order"] == [0] + [m for i in range(1, rank+1) for m in (i, -i)]
    assert np.shape(native["mode_cartesian_coefficients"]) == (count, (rank+1)*(rank+2)//2)
    assert "multipole_polarizability" not in native
    assert len(native["response_residuals"]) == count
    assert max(native["response_residuals"]) <= 1e-10
    assert native["multipole_polarizability_imag"][0][0] > 0
    for part in ("real", "imag"):
        key = "multipole_polarizability_" + part
        assert native["units"][key] == f"e^2*bohr^{2*rank}/hartree"
        assert physical["units"][key] == f"e^2*angstrom^{2*rank}/eV"
        for report in (cli, atomic):
            np.testing.assert_allclose(report[key], native[key], rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(physical[key])*Hartree/Bohr**(2*rank),
                                   native[key], rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(physical["origin"])/Bohr, native["origin"], atol=1e-14)
    negative = dft_rs.multipole_polarizability(
        numbers, positions, **dict(controls, frequency=-controls["frequency"])
    )
    static = dft_rs.multipole_polarizability(
        numbers, positions, **dict(controls, frequency=0.0, frequency_damping=0.0)
    )
    near_static = dft_rs.multipole_polarizability(
        numbers, positions, **dict(controls, frequency=1e-6, frequency_damping=0.0,
                                  imaginary_frequency=True)
    )
    np.testing.assert_allclose(negative["multipole_polarizability_real"],
                               native["multipole_polarizability_real"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(negative["multipole_polarizability_imag"],
                               -np.asarray(native["multipole_polarizability_imag"]), rtol=0, atol=1e-8)
    np.testing.assert_array_equal(static["multipole_polarizability"],
                                  static["multipole_polarizability_real"])
    np.testing.assert_allclose(near_static["multipole_polarizability_real"],
                               static["multipole_polarizability"], rtol=0, atol=1e-7)
    print(f"Installed Racah rank {rank} response", native)


def test_racah_multipole_rank_and_correlated_frequency_refusals():
    numbers, positions = [1, 1], [[0, 0, -0.7], [0, 0, 0.7]]
    with pytest.raises(ValueError, match="multipole_rank must be between 1 and 5"):
        dft_rs.multipole_polarizability(numbers, positions, xc="hf", multipole_rank=6)
    with pytest.raises(NotImplementedError, match="retarded parent, target and PT2"):
        dft_rs.multipole_polarizability(numbers, positions, xc="b2plyp",
                                        reference="gks", multipole_rank=3, frequency=0.03)


@pytest.mark.parametrize(("reference", "charge", "multiplicity"), [
    ("rks", 0, 1), ("uks", 1, 2), ("roks", 1, 2), ("gks", 1, 2),
])
def test_molecular_origin_tracks_translation_for_all_multipole_entries(
        tmp_path, reference, charge, multiplicity):
    # Four unequal noncoplanar positions remove all geometrical symmetries.
    # Charged references also exercise the non-invariant laboratory dipole.
    positions = np.array([[0.13, -0.21, 0.19], [1.57, 0.16, -0.08],
                          [-0.19, 1.93, 0.37], [0.42, 0.29, 2.28]])
    shift = np.array([2.31, -1.72, 0.93])
    numbers = [1] * len(positions)
    basis = tmp_path / "origin-fixture.g94"
    basis.write_text("H 0\nS 1 1.0\n0.8 1.0\n****\n", encoding="ascii")
    controls = dict(xc="hf", basis=str(basis), reference=reference, charge=charge,
                    multiplicity=multiplicity, grid_weight_cutoff=0, dispersion="none",
                    conv_tol=1e-13, conv_tol_dp=1e-12, conv_tol_grad=1e-11,
                    max_iter=300, cpu_threads=1)
    moments = [dft_rs.molecular_multipole(numbers, geometry, max_rank=5, **controls)
               for geometry in (positions, positions + shift)]
    for geometry, report in zip((positions, positions + shift), moments, strict=True):
        assert report["origin_convention"] == "center_of_mass"
        np.testing.assert_allclose(report["origin"], np.mean(geometry, axis=0),
                                   rtol=0, atol=1e-14)
    defects = {}
    for rank in range(6):
        for representation in ("cartesian", "traceless", "spherical"):
            before = np.asarray(moments[0]["ranks"][rank][representation])
            after = np.asarray(moments[1]["ranks"][rank][representation])
            np.testing.assert_allclose(after, before, rtol=2e-9, atol=2e-9)
            defects[f"moment_{rank}_{representation}"] = float(np.max(np.abs(after-before)))
    for rank in (3, 4, 5):
        for frequency, damping in ((0.0, 0.0), (0.045, 0.005)):
            arguments = dict(controls, multipole_rank=rank, frequency=frequency,
                             frequency_damping=damping)
            before = dft_rs.multipole_polarizability(numbers, positions, **arguments)
            after = dft_rs.multipole_polarizability(numbers, positions + shift, **arguments)
            assert before["origin_convention"] == after["origin_convention"] == "center_of_mass"
            np.testing.assert_allclose(np.asarray(after["origin"]) - before["origin"],
                                       shift, rtol=0, atol=1e-14)
            for part in ("real", "imag"):
                key = "multipole_polarizability_" + part
                a, b = np.asarray(before[key]), np.asarray(after[key])
                np.testing.assert_allclose(a, b, rtol=2e-9, atol=1e-8)
                defects[f"response_{rank}_{frequency}_{part}"] = float(np.max(np.abs(a-b)))
    print("Molecular multipole translation invariance", json.dumps(
        dict(reference=reference, charge=charge, shift=shift.tolist(), defects=defects),
        sort_keys=True))
