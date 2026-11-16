# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Matched molecular DFT+U+V reports through the common driver."""

import json
import os
import subprocess

from ase import Atoms
import pytest

import dft_rs
from dft_rs.units import BOHR_IN_ANGSTROM as BOHR, HARTREE_IN_EV as EH


def test_lih_intersite_hubbard_energy_and_public_units(tmp_path):
    numbers = [3, 1]
    positions = [[0, 0, 0], [0.31, -0.42, 2.98]]
    atoms = Atoms(numbers=numbers, positions=[[BOHR * x for x in r] for r in positions])
    sites = [{"orbitals": [0], "u_eff": 0.025}, {"orbitals": [5], "u_eff": 0.04}]
    v = 0.05
    options = dict(xc="wb97x-d", basis="sto-3g", reference="rks", dispersion="none",
                   grid_level=2, grid_weight_cutoff=0.0, hubbard_sites=sites, cpu_threads=1)
    pairs = [{"sites": [1, 0], "v": v}]
    native = dft_rs.dft_plus_u(numbers, positions, **options, hubbard_pairs=pairs)
    original = dft_rs.dft_plus_u(numbers, positions, **options)
    zero = dft_rs.dft_plus_u(numbers, positions, **options, hubbard_pairs=[{"sites": [0, 1], "v": 0}])
    assert native["converged"] and original["converged"] and zero["converged"]
    assert zero["energy"] == pytest.approx(original["energy"], abs=1e-13, rel=0)
    assert native["hubbard_intersite_energy"] < 0
    reported_pair = native["hubbard_pairs"][0]
    assert reported_pair["sites"] == [0, 1]
    assert reported_pair["interaction"] - reported_pair["double_counting"] == pytest.approx(
        reported_pair["correction"], abs=1e-14, rel=0,
    )

    xyz = tmp_path / "lih.xyz"
    xyz.write_text("2\nLiH\n" + "".join(
        f"{s} {' '.join(f'{x:.17g}' for x in r)}\n" for s, r in zip(atoms.symbols, atoms.positions)
    ), encoding="utf-8")
    command = [os.environ["DFT_RS_CLI"], "dft-plus-u", str(xyz), "--format", "json"]
    for key, value in dict(options, hubbard_pairs=pairs).items():
        command.extend(["--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)])
    process = subprocess.run(command, capture_output=True, check=True, encoding="utf-8")
    assert "[scf] iter" in process.stderr and "SCF residual" in process.stderr
    cli = json.loads(process.stdout)
    atomic = dft_rs.DftRs(units="atomic").get_dft_plus_u(atoms, **options, hubbard_pairs=pairs)
    for other in (cli, atomic):
        assert other["converged"]
        assert other["units"]["hubbard_pairs.correction"] == "hartree"
        for key in ("energy", "hubbard_energy", "hubbard_intersite_energy", "scf_orbital_residual"):
            assert other[key] == pytest.approx(native[key], abs=1e-10, rel=0)
    physical = dft_rs.DftRs().get_dft_plus_u(
        atoms, **options, hubbard_pairs=[dict(pairs[0], v=EH * v)],
    )
    assert physical["units"]["hubbard_pairs.correction"] == "eV"
    assert physical["hubbard_pairs"][0]["v"] == pytest.approx(EH * v, abs=1e-13, rel=0)
    assert physical["hubbard_intersite_energy"] == pytest.approx(
        EH * native["hubbard_intersite_energy"], abs=1e-9, rel=0,
    )
    print(
        f"LiH wb97x-d DFT+U+V E={native['energy']:.15e} Eh "
        f"U+V={native['hubbard_energy']:.15e} Eh "
        f"V={native['hubbard_intersite_energy']:.15e} Eh "
        f"orbital residual={native['scf_orbital_residual']:.12e} Eh"
    )
