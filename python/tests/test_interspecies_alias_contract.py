# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Equivalent correlation names share the same nonlinear partner-density channel."""
import copy

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from test_fixed_populations import cli_report


@pytest.mark.parametrize("surface", ["native", "ase", "cli"])
def test_equivalent_interspecies_names_share_total_partner_density(tmp_path, surface):
    xyz = [[.13, -.21, -.33], [.13, -.21, 1.07]]
    atoms = Atoms("H2", positions=np.asarray(xyz) * Bohr)
    opts = dict(xc="hf", basis="sto-3g", dispersion="none", quantum_exponents=[1.],
                quantum_lmax=0, species_tolerance=1e-11, conv_tol=1e-13,
                conv_tol_grad=1e-11, conv_tol_dp=1e-11, max_iter=100,
                grid_weight_cutoff=0., cpu_threads=1, memory_mb=512)

    def run(names):
        options = dict(copy.deepcopy(opts), particles=[
            dict(kind="nucleus", atom=i, correlation=name) for i, name in enumerate(names)])
        if surface == "native":
            return dft_rs.multicomponent([1, 1], xyz, **options)
        if surface == "ase":
            return dft_rs.DftRs(units="atomic").get_multicomponent(atoms, **options)
        return cli_report(tmp_path, "multicomponent", atoms, options)

    base = run(["epc17", "epc17"])
    for names in [["epc17", "epc17-1"], ["EPC17-1", "EpC17"],
                  ["epc17-1", "epc17-1"]]:
        other = run(names)
        assert base["converged"] and other["converged"]
        assert len(other["interspecies_correlation"]) == 1
        assert [q["correlation_channel"] for q in other["quantum"]] == [0, 0]
        np.testing.assert_allclose(other["energy"], base["energy"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(other["energy_components"]["interspecies_correlation"],
                                   base["energy_components"]["interspecies_correlation"],
                                   atol=1e-12, rtol=0.)
        np.testing.assert_allclose(other["electronic_density"], base["electronic_density"],
                                   atol=1e-12, rtol=0.)
        for a, b in zip(base["quantum"], other["quantum"]):
            np.testing.assert_allclose(a["density"], b["density"], atol=1e-12, rtol=0.)
    distinct = run(["epc17", "epc17-2"])
    assert distinct["converged"]
    assert len(distinct["interspecies_correlation"]) == 2
    assert [q["correlation_channel"] for q in distinct["quantum"]] == [0, 1]
