# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Regression checks for independent spin-spin reference arithmetic.

These validate the oracle, not the production D provider or SOC contributions.
Production-component comparisons are separate numerical acceptance tests.
"""
from decimal import Decimal, localcontext
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / "tools/oracle/gaussian_spin_pair.py"
SPEC = importlib.util.spec_from_file_location("gaussian_spin_pair_reference", PATH)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
POSITIONS = [[.13, -.21, .17], [2.05, .25, .38],
             [-.65, 1.76, -.44], [-1.19, -1.48, .85]]


@pytest.fixture(scope="module")
def reference():
    return ORACLE.reference(POSITIONS, 70)


def test_stationarity_particle_number_and_unrepaired_tensor(reference):
    assert Decimal(reference["stationarity"]) < Decimal("1e-48")
    assert abs(Decimal(reference["particle_number"]) - 2) < Decimal("1e-48")
    assert Decimal(reference["idempotency"]) < Decimal("1e-48")
    assert Decimal(reference["gap"]) > 0
    assert abs(Decimal(reference["raw_D_trace"])) < Decimal("1e-48")
    tensor = [Decimal(value) for value in reference["spin_spin"]]
    for i in range(3):
        for j in range(3):
            assert abs(tensor[3*i+j] - tensor[3*j+i]) < Decimal("1e-48")


def test_precision_increase_preserves_rounded_tensor(reference):
    finer = ORACLE.reference(POSITIONS, 90, reference["density"])
    assert list(map(float, finer["spin_spin"])) == list(map(float, reference["spin_spin"]))
    assert float(finer["energy"]) == float(reference["energy"])


def test_common_translation_has_no_arbitrary_origin(reference):
    with localcontext() as context:
        context.prec = 90
        shift = [Decimal("2.3"), Decimal("-1.7"), Decimal("0.9")]
        moved = [[Decimal.from_float(value) + delta for value, delta in zip(row, shift)]
                 for row in POSITIONS]
        translated = ORACLE.reference(moved, 70, reference["density"])
        assert abs(Decimal(translated["energy"]) - Decimal(reference["energy"])) < Decimal("1e-40")
        for actual, expected in zip(translated["spin_spin"], reference["spin_spin"]):
            assert abs(Decimal(actual) - Decimal(expected)) < Decimal("1e-40")


@pytest.mark.parametrize("precision", [True, 40, 100, 70.5])
def test_invalid_reference_precision_is_explicit(precision):
    with pytest.raises(ValueError, match="precision"):
        ORACLE.reference(POSITIONS, precision)
