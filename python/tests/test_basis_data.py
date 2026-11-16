# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Offline provenance and complete payload hashes for the added BSE data."""
import csv
import hashlib
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest
from ase import Atoms

import dft_rs
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr

ROOT = Path(__file__).resolve().parents[2]


def test_thirty_pinned_bases_keep_their_exact_download_and_catalogue_hashes():
    spec = importlib.util.spec_from_file_location("basis_fetch", ROOT / "tools/fetch_basis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (ROOT / "third_party/bse/catalogue.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 30
    assert {row["name"] for row in rows} == set(module.PINNED_VERSIONS)
    for row in rows:
        data = (ROOT / "src/basis/data" / (row["name"] + ".g94")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        text = data.decode("ascii")
        assert "&version=" + row["version"] in text
        assert str(module.PINNED_VERSIONS[row["name"]]) == row["version"]
        digest = re.search(r"!   Data SHA256: ([0-9a-f]{64})", text).group(1)
        assert hashlib.sha256(text.split("\n\n", 1)[1].encode("ascii")).hexdigest() == digest
        assert "PROVENANCE" in text[:1000]
    notice = (ROOT / "third_party/bse/NOTICE.md").read_text(encoding="utf-8")
    assert "catalogue.csv" in notice
    license_text = (ROOT / "third_party/bse/LICENSE").read_text(encoding="utf-8")
    assert "The Molecular Sciences Software Institute" in license_text


@pytest.mark.parametrize("numbers,positions", [
    ([1, 1], [[.13, -.21, -.71], [.13, -.21, .69]]),
    ([3, 1], [[.11, .07, -.03], [.23, -.13, 3.02]]),
    ([8, 1, 1], [[0., 0., .2217], [0., 1.4309, -.8867], [.13, -1.4309, -.8867]]),
    ([7, 1, 1, 1], [[.03, -.08, .17], [1.72, .12, -.52], [-.71, 1.63, -.64], [-.87, -1.48, -.43]]),
    ([15, 1, 1, 1], [[.03, -.08, .17], [2.47, .14, -.81], [-1.13, 2.29, -.93], [-1.28, -2.17, -.68]]),
], ids=["hydrogen", "lithium-hydride", "water", "ammonia", "phosphine"])
def test_added_orbital_basis_runs_five_molecules_on_public_energy_routes(tmp_path, numbers, positions):
    """Routing/SCF checks, not experimental accuracy or derivative acceptance."""
    from test_fixed_populations import cli_report

    options = dict(xc="hf", basis="pcseg-0", dispersion="none", conv_tol=1e-12,
                   conv_tol_grad=1e-10, conv_tol_dp=1e-10, max_iter=500)
    native = dft_rs.energy(numbers, positions, **options)
    atoms = Atoms(numbers=numbers, positions=np.asarray(positions) * Bohr)
    adapted = DftRs(units="atomic").get_energy_report(atoms, **options)
    cli = cli_report(tmp_path, "energy", atoms, options)
    for result in [native, adapted, cli]:
        assert result["converged"] and np.isfinite(result["energy"])
        assert result["energy"] < 0.
        assert abs(result["energy"] - native["energy"]) < 1e-10
    print("pcseg-0 five-molecule SCF", numbers, native["energy"], flush=True)

