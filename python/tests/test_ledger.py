# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The coverage ledgers must agree with the registry and the test files."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ledger_check", ROOT / "tools/ledger_check.py")
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


@pytest.mark.parametrize("column,value,message", [
    ("rust_entry", "driver::run(missing-provider)", "Rust driver route"),
    ("cli_entry", "dft-rs missing-provider", "CLI route"),
    ("python_entry", "missing_provider", "python_entry"),
    ("ase_entry", "get_missing_provider", "ase_entry"),
    ("hf_restricted", "all_methods_supported", "unreviewed capability status"),
])
def test_capability_audit_rejects_unreachable_routes_and_unqualified_support(column, value, message):
    row = dict(ledger.rows(ROOT / "docs/property_targets.csv")[0])
    row[column] = value
    assert any(message in finding for finding in ledger.check_public_routes(ROOT, [row]))


def test_validated_method_subset_requires_its_scope_and_evidence():
    row = dict(ledger.rows(ROOT / "docs/property_targets.csv")[0])
    row.update(hf_restricted="validated_subset", matrix_evidence="")
    assert any("without scope and evidence" in finding for finding in ledger.check_public_routes(ROOT, [row]))


@pytest.mark.parametrize("column,value,message", [
    ("availability", "validated", "implementation availability"),
    ("validation_state", "implemented", "validation state"),
    ("availability", "", "implementation availability"),
    ("validation_state", "", "validation state"),
])
def test_implementation_and_validation_are_separate_required_fields(column, value, message):
    row = dict(ledger.rows(ROOT / "docs/property_targets.csv")[0])
    row[column] = value
    assert any(message in finding for finding in ledger.check_public_routes(ROOT, [row]))


def test_unimplemented_target_cannot_inherit_an_adjacent_verified_component():
    row = dict(ledger.rows(ROOT / "docs/property_targets.csv")[0])
    row.update(availability="unimplemented", validation_state="verified_within_stated_scope")
    assert any("unimplemented target" in finding for finding in ledger.check_public_routes(ROOT, [row]))


def test_retained_failures_require_a_stated_scope_and_evidence():
    row = dict(ledger.rows(ROOT / "docs/property_targets.csv")[0])
    row.update(validation_state="partial_with_failures", matrix_scope="")
    assert any("requires explicit scope" in finding for finding in ledger.check_public_routes(ROOT, [row]))


def test_machine_statuses_require_actual_glossary_definitions(tmp_path):
    assert ledger.check_status_glossary(ROOT) == []
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/conventions.md").write_text(
        "\n### Status vocabulary\n\nThe word partial is mentioned but not defined.\n")
    assert any("partial" in finding for finding in ledger.check_status_glossary(tmp_path))


def test_ledgers_match_registry_and_test_files():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "tools/ledger_check.py"), "--check"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_inventory_reports_every_target_and_method_cell():
    result = subprocess.run([sys.executable, str(ROOT / "tools/ledger_check.py"), "--release-gaps"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    inventory = json.loads(result.stdout)
    rows = ledger.rows(ROOT / "docs/property_targets.csv")
    assert len(inventory["targets"]) == inventory["physical_targets"] == len(rows)
    assert inventory["registered_features"] == sum(inventory["registry_routes"].values())
    assert inventory["registry"] == ledger.api_reference().registry_specs(ROOT)
    for target, row in zip(inventory["targets"], rows, strict=True):
        assert target["target"] == row["target_id"]
        assert target["method_cells"] == {key: row[key] for key in ledger.method_columns(rows)}
    targets = {target["target"]: target for target in inventory["targets"]}
    assert targets["T160"]["missing_surface_entries"] == ["rust", "cli", "python", "ase"]
    assert targets["T126"]["missing_surface_entries"] == []
    assert targets["T188"]["surface_entries"]["python"] == "polarizability_report"
