# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

import numpy as np
import pytest
from ase import Atoms
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from dft_rs import DftRs, periodic_density_response, periodic_dielectric

GEOMETRY = [[0, 0, -0.7], [0, 0, 0.7]]
CELL = np.eye(3)*9.0
SCF = dict(xc="pbe", overlap_cutoff=7.0, grid_level=0, grid_shell=1,
           conv_tol=1e-12, conv_tol_dp=1e-11, conv_tol_grad=1e-10,
           max_iter=250)
RESPONSE = dict(q=(0.043, 0.017, -0.021), omega=0.13, damping=0.04,
                reciprocal_cutoff=0.75)


def test_native_density_and_ase_dielectric_share_the_screened_chi_convention():
    density = periodic_density_response([1, 1], GEOMETRY, CELL, **RESPONSE, **SCF)
    atoms = Atoms("H2", positions=np.asarray(GEOMETRY)*Bohr, cell=CELL*Bohr, pbc=True)
    calc = DftRs(units="atomic", **SCF)
    dielectric = calc.get_periodic_dielectric(atoms, modes=(0,), **RESPONSE)
    chi = complex(density["fourier_density_real"][0], density["fourier_density_imag"][0])
    actual = complex(dielectric["screened_chi_real"][0][0], dielectric["screened_chi_imag"][0][0])
    assert abs(actual-chi) < 1e-11
    wave = np.asarray(dielectric["reciprocal_waves"][0])
    expected = 1 + 4*np.pi*chi/np.dot(wave, wave)
    actual = complex(dielectric["inverse_epsilon_real"][0][0], dielectric["inverse_epsilon_imag"][0][0])
    assert abs(actual-expected) < 1e-10
    assert density["shape"] == [2, 1, 2, 2]
    assert dielectric["shape"] == [1, 1]
    for report in (density, dielectric):
        assert report["converged"]
        assert max(report["residuals"]) <= 1e-10
        assert report["q_units"] == "bohr^-1"
        assert report["frequency_units"] == "hartree"
        assert report["reference"]["particle_number_error"] < 1e-10
        assert "bohr^-3/hartree" in report["units"]
    assert density["occupation_consistency_defect"] < 1e-9


def test_periodic_response_rejects_invalid_requests_before_reference_calculation():
    with pytest.raises(ValueError):
        periodic_dielectric([1,1], GEOMETRY, CELL, modes=())
    with pytest.raises(ValueError):
        periodic_density_response([1,1], GEOMETRY, CELL, q=(float("nan"),0,0))
    with pytest.raises(ValueError):
        periodic_density_response([1,1], GEOMETRY, CELL, imaginary=True, damping=0.1)
