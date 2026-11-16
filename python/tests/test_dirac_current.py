# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

"""Public surface comparisons for optional Dirac current interactions."""

import pytest

from test_no_pair_projector import check_dirac_coulomb_scf_public_surfaces


@pytest.mark.parametrize("interaction", ["gaunt", "breit"])
def test_dirac_current_interaction_public_surfaces(tmp_path, interaction):
    check_dirac_coulomb_scf_public_surfaces(tmp_path, interaction)
