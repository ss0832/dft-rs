# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Source-shell order is independent of reference-engine shell sorting."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

gto = pytest.importorskip("pyscf.gto")
_path = Path(__file__).resolve().parents[2] / "tools/oracle/gen_integrals.py"
_spec = importlib.util.spec_from_file_location("integral_oracle_order", _path)
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


@pytest.mark.parametrize("shells,expected", [
    ([[1, [.9, 1.]], [0, [.4, 1.]], [1, [.2, 1.]], [0, [1.1, 1.]]],
     [4, 2, 3, 0, 7, 5, 6, 1]),
    ([[1, [.9, 1., .2], [.3, -.1, 1.]], [0, [.4, 1.]]],
     [3, 1, 2, 6, 4, 5, 0]),
])
def test_input_shells_and_contraction_columns(shells, expected):
    mol = gto.M(atom="C 0 0 0", basis={"C": shells}, verbose=0)
    np.testing.assert_array_equal(_oracle.dftrs_ao_permutation(mol), expected)


def test_sorted_high_angular_components_keep_their_order():
    shells = [[l, [.6 + l, 1.]] for l in range(7)]
    mol = gto.M(atom="C 0 0 0", basis={"C": shells}, verbose=0)
    expected, offset = [], 0
    for l in range(7):
        components = [0] if l == 0 else [2, 0, 1] if l == 1 else [
            l, *[l + sign * m for m in range(1, l + 1) for sign in (1, -1)]
        ]
        expected.extend(offset + k for k in components)
        offset += 2 * l + 1
    np.testing.assert_array_equal(_oracle.dftrs_ao_permutation(mol), expected)


def test_alias_and_mismatched_source_are_rejected():
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
    with pytest.raises(ValueError, match="explicit spherical"):
        _oracle.dftrs_ao_permutation(mol)
    with pytest.raises(ValueError, match="does not match"):
        _oracle.dftrs_ao_permutation(mol, {"He": [[0, [999., 1.]]]})


@pytest.mark.parametrize("symbol,spin,expected", [
    ("P", 1, [0, 1, 5, 3, 4, 2, 8, 6, 7]),
    ("Fe", 0, [0, 1, 6, 4, 5, 2, 9, 7, 8, 3, 12, 10, 11, 15, 16, 14, 17, 13]),
])
def test_bundled_interleaved_sp_shells(symbol, spin, expected):
    basis = _oracle.parse_g94(str(Path(_oracle.DATA_DIR) / "sto-3g.g94"), {symbol})
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=spin, verbose=0)
    np.testing.assert_array_equal(_oracle.dftrs_ao_permutation(mol), expected)
