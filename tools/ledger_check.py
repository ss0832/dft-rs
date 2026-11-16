# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Consistency checks of the two coverage ledgers against the source tree.

Checks:
- every target id named by a feature of `src/driver/registry.rs` is a row
  of `docs/property_targets.csv`;
- every coverage id named by a target row is an entry of the response
  coverage ledger (section "Response coverage ledger" of `docs/response.md`);
- every test token of the form `pNNN_name`, `tests/...` or `python/...`
  names an existing file, and a `file.py::test_name` token names a test
  that the file defines;
- a row whose tests list names files does not also say
  `acceptance_tests_pending`;
- a coverage row whose status is exactly `pending` has no target row
  that says it is implemented;
- explicit Rust-driver, CLI, Python and ASE entries resolve to the existing
  registry or public definitions;
- method-matrix claims use the reviewed bounded status vocabulary and a
  validated subset names its scope and evidence. This checks the claim's
  structure, not the numerical validity of the evidence.

Exit status 1 and one line per finding when a check fails.
"""
import argparse
import ast
from collections import Counter
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PENDING = "acceptance_tests_pending"
AVAILABILITY = {"implemented", "conditionally_supported", "unimplemented",
                "internal_only_prototype", "not_audited", "not_applicable"}
VALIDATION_STATES = {"verified_within_stated_scope", "partial", "partial_with_failures",
                     "untested", "failed", "not_audited", "not_applicable"}
METHOD_MATRIX_STATUSES = {"backend_only", "implemented_not_yet_validated", "implemented_subset",
                          "implemented_unvalidated", "not_applicable", "not_audited", "pending",
                          "validated_subset"}
METHOD_PREFIXES = ("hf_", "lda_", "gga_", "meta_gga_", "hybrid_", "double_hybrid_", "skala_", "mixed_")


def check_status_glossary(root):
    """Every machine status needs an authoritative definition, not just a mention."""
    text = (root / "docs/conventions.md").read_text(encoding="utf-8")
    heading = "\n### Status vocabulary\n"
    if heading not in text:
        return ["status vocabulary definitions are missing from conventions.md"]
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    defined = set()
    for line in section.splitlines():
        if line.startswith("| `"):
            defined.update(re.findall(r"`([A-Za-z_]+)`", line.split("|", 2)[1]))
    return [f"status token {name} has no glossary definition"
            for name in sorted((AVAILABILITY | VALIDATION_STATES | METHOD_MATRIX_STATUSES) - defined)]


def api_reference():
    specification = importlib.util.spec_from_file_location("ledger_api_reference", Path(__file__).with_name("api_reference.py"))
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)
    return reference


def method_columns(targets):
    return [key for key in targets[0] if key.startswith(METHOD_PREFIXES)] if targets else []


def check_public_routes(root, targets):
    """Audit documented routes and bounded method claims against existing inventories."""
    reference = api_reference()
    registry = reference.registry_specs(root)
    identifiers = {entry["id"] for entry in registry}
    native_tree = ast.parse((root / "python/dft_rs/__init__.py").read_text(encoding="utf-8"))
    native_names = set(reference.exported_names(native_tree, "dft_rs"))
    ase_tree = ast.parse((root / "python/dft_rs/ase_calculator.py").read_text(encoding="utf-8"))
    calculator = next(node for node in ase_tree.body if isinstance(node, ast.ClassDef) and node.name == "DftRs")
    ase_names = {node.name for node in calculator.body if isinstance(node, ast.FunctionDef)}
    # ASE's inherited standard getter is implemented by calculate(properties=['stress']).
    ase_names.add("get_stress")
    matrix_columns = method_columns(targets)
    statuses = METHOD_MATRIX_STATUSES
    findings = []
    for row in targets:
        label = row["target_id"]
        availability = row.get("availability")
        validation_state = row.get("validation_state")
        if availability not in AVAILABILITY:
            findings.append(f"{label}: missing or unreviewed implementation availability {availability!r}")
        if validation_state not in VALIDATION_STATES:
            findings.append(f"{label}: missing or unreviewed validation state {validation_state!r}")
        if availability == "unimplemented" and validation_state == "verified_within_stated_scope":
            findings.append(f"{label}: an unimplemented target cannot claim verified implementation")
        if validation_state in {"verified_within_stated_scope", "partial_with_failures"} and not all(
                row.get(key, "").strip() for key in ("matrix_scope", "matrix_evidence", "validation")):
            findings.append(f"{label}: validation status requires explicit scope, evidence and tests")
        for name in re.findall(r'\bdriver::run\(\s*"?([\w-]+)"?\s*\)', row["rust_entry"]):
            if name not in identifiers:
                findings.append(f"{label}: Rust driver route {name} is not registered")
        for name in re.findall(r'\bdft-rs\s+([\w][\w-]*)', row["cli_entry"]):
            if name not in identifiers:
                findings.append(f"{label}: CLI route {name} is not registered")
        for column, names in (("python_entry", native_names), ("ase_entry", ase_names)):
            for name in test_tokens(row[column]):
                if name != "pending" and re.fullmatch(r"[A-Za-z_]\w*", name) and name not in names:
                    findings.append(f"{label}: {column} names unavailable public entry {name}")
        for column in matrix_columns:
            status = row[column]
            if status not in statuses:
                findings.append(f"{label}: {column} has unreviewed capability status {status!r}")
            if status == "validated_subset" and not all(row[key].strip() for key in ("matrix_scope", "matrix_evidence", "validation")):
                findings.append(f"{label}: {column} claims a validated subset without scope and evidence")
    return findings


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


LEDGER_HEADING = "\n## Response coverage ledger\n"


def coverage_entries(path):
    """Entries of the response coverage ledger: id, status and tests (the
    code spans of the **Tests** line, joined by semicolons)."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if LEDGER_HEADING not in text:
        raise SystemExit(f"{path}: no response coverage ledger section")
    section = text.split(LEDGER_HEADING, 1)[1].split("\n## ", 1)[0]
    entries = []
    for block in section.split("\n### ")[1:]:
        entry_id = block.split()[0]
        status = re.search(r"^- \*\*Status\*\*: (.*)$", block, re.MULTILINE)
        tests = re.search(r"^- \*\*Tests\*\*: (.*)$", block, re.MULTILINE)
        if status is None or tests is None:
            raise SystemExit(f"{path}: ledger entry {entry_id} lacks a Status or Tests line")
        entries.append({
            "id": entry_id,
            "status": status.group(1).strip(),
            "tests": ";".join(re.findall(r"`([^`]+)`", tests.group(1))),
        })
    return entries


def test_tokens(value):
    return [token for token in value.split(";") if token]


def file_token(root, token):
    """Path a test token names, or None for module-path and prose tokens."""
    if re.fullmatch(r"p\d+_\w+", token):
        return root / "tests" / f"{token}.rs"
    if token.startswith(("tests/", "python/")):
        return root / token
    return None


def check(root):
    findings = check_status_glossary(root)
    targets = rows(root / "docs/property_targets.csv")
    findings.extend(check_public_routes(root, targets))
    coverage = coverage_entries(root / "docs/response.md")
    target_ids = {row["target_id"] for row in targets}
    coverage_ids = {row["id"] for row in coverage}

    registry = (root / "src/driver/registry.rs").read_text(encoding="utf-8")
    for target in sorted(set(re.findall(r'"(T\d{3})"', registry)) - target_ids):
        findings.append(f"registry names {target}, which is not a row of property_targets.csv")

    for row in targets:
        coverage_id = row["coverage_id"]
        if coverage_id and coverage_id not in coverage_ids:
            findings.append(f"{row['target_id']}: coverage id {coverage_id} is not an entry of the response coverage ledger")

    for name, table, key, column in (
        ("property_targets.csv", targets, "target_id", "validation"),
        ("response coverage ledger", coverage, "id", "tests"),
    ):
        for row in table:
            tokens = test_tokens(row[column])
            named = []
            for token in tokens:
                file_part, _, function = token.partition("::")
                path = file_token(root, file_part)
                if path is None:
                    continue
                named.append(token)
                if not path.is_file():
                    findings.append(f"{name} {row[key]}: test {token} does not exist")
                elif function and not re.search(
                    rf"^\s*def {re.escape(function)}\(", path.read_text(encoding="utf-8"), re.MULTILINE
                ):
                    findings.append(f"{name} {row[key]}: {file_part} defines no test {function}")
            if PENDING in tokens and named:
                findings.append(f"{name} {row[key]}: says {PENDING} but lists {';'.join(named)}")

    by_coverage = {}
    for row in targets:
        by_coverage.setdefault(row["coverage_id"], []).append(row)
    for row in coverage:
        if row["status"] != "pending":
            continue
        for target in by_coverage.get(row["id"], []):
            if "implemented" in target["implementation_status"]:
                findings.append(
                    f"response coverage ledger {row['id']}: status pending but {target['target_id']} is implemented"
                )
    return findings


def release_gaps(root):
    """Report the entire recorded scope with unresolved connections and evidence."""
    registry = api_reference().registry_specs(root)
    targets = rows(root / "docs/property_targets.csv")
    methods = method_columns(targets)
    surfaces = ("rust", "cli", "python", "ase")
    scope = []
    for target in targets:
        availability = target["availability"]
        matrix = {method: target[method] for method in methods}
        missing = [surface for surface in surfaces
                   if target[surface + "_entry"].strip() in {"", "-", "pending"}]
        scope.append({
            "target": target["target_id"],
            "physical_target": target["physical_target"],
            "feature": target["api_id"],
            "coverage": target["coverage_id"],
            "availability": availability,
            "validation": target["validation_state"],
            "surface_entries": {surface: target[surface + "_entry"] for surface in surfaces},
            "missing_surface_entries": missing if availability != "not_applicable" else [],
            "implementation": target["implementation_status"],
            "scope": target["matrix_scope"],
            "evidence": target["matrix_evidence"],
            "tests": test_tokens(target["validation"]),
            "method_cells": matrix,
        })
    return {
        "registered_features": len(registry),
        "registry_routes": dict(Counter(entry["route"] for entry in registry)),
        "registry": registry,
        "registry_missing_surfaces": [
            {"feature": entry["id"], "missing": [name for name in surfaces if entry[name] == "-"]}
            for entry in registry if any(entry[name] == "-" for name in surfaces)
        ],
        "physical_targets": len(targets),
        "availability_counts": dict(Counter(row["availability"] for row in scope)),
        "validation_counts": dict(Counter(row["validation"] for row in scope)),
        "method_cell_counts": dict(Counter(row["method_cells"][method]
                                           for row in scope for method in methods)),
        "targets": scope,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit with status 1 on any finding")
    parser.add_argument("--release-gaps", action="store_true",
                        help="emit the registered API and full physical-target gap inventory as JSON")
    parser.add_argument("--root", type=Path, default=ROOT, help="package source directory")
    arguments = parser.parse_args()
    findings = check(arguments.root)
    if arguments.release_gaps and not findings:
        print(json.dumps(release_gaps(arguments.root), indent=2))
        return 0
    for finding in findings:
        print(finding)
    if not findings:
        print("ledgers consistent")
    return 1 if findings and (arguments.check or arguments.release_gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
