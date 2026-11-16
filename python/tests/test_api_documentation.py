# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""All explicit Python/ASE API contracts must also be in existing Markdown."""
from pathlib import Path
import importlib.util
import json
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("api_reference", ROOT / "tools/api_reference.py")
api_reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(api_reference)


@pytest.fixture
def api_sources(tmp_path):
    package = tmp_path / "python/dft_rs"
    package.mkdir(parents=True)
    for name in ("__init__", "native", "system", "ase_calculator", "units"):
        shutil.copyfile(ROOT / "python/dft_rs" / f"{name}.py", package / f"{name}.py")
    return tmp_path


@pytest.mark.parametrize("module,before,after,message", [
    ("__init__", "__all__ = [", "removed_exports = [", "no explicit __all__"),
    ("__init__", '    "DftRs",', '    "DftRs", "DftRs",', "repeats exports"),
    ("native", '__all__ += [', '__all__ += ["calculate",', "repeats exports"),
    ("native", '__all__ += [', '__all__ += ["missing_calculation",', "exports undefined"),
    ("system", '__all__ = [', '__all__ = ["missing_calculation",', "exports undefined"),
    ("__init__", '    "DftRs",', '    "missing_calculation",', "exports unresolved"),
    ("__init__", "from .native import Report", "from .system import Report", "imports undocumented or undefined"),
    ("__init__", "from .native import Report", "from .native import Report\nfrom .system import Molecule as Report", "conflicting bindings"),
])
def test_export_audit_detects_broken_contracts(api_sources, module, before, after, message):
    path = api_sources / "python/dft_rs" / f"{module}.py"
    text = path.read_text(encoding="utf-8")
    assert before in text, "the regression mutation no longer matches the source"
    path.write_text(text.replace(before, after, 1), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        api_reference.check_exports(api_sources)


def test_export_audit_accepts_current_sources(api_sources):
    api_reference.check_exports(api_sources)


@pytest.mark.parametrize("key", ["id", "cli", "python", "ase"])
def test_registry_audit_rejects_public_name_collisions(tmp_path, key):
    specs = api_reference.registry_specs(ROOT)
    first, second = dict(specs[0]), dict(specs[1])
    second[key] = first[key]
    path = tmp_path / "src/driver/registry.rs"
    path.parent.mkdir(parents=True)
    rows = []
    for spec in (first, second):
        fields = [spec[name] if name in ("space", "route") else json.dumps(spec[name])
                  for name in ("id", "title", "space", "route", "rust", "cli", "python", "ase")]
        rows.append("// settings-contract: " + spec["settings_contract"] + "\nfeature!("
                    + ", ".join(fields) + ', ["T001"]),')
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match=f"registry repeats {key}"):
        api_reference.registry_specs(tmp_path)


def test_registry_reader_covers_continued_strings_and_every_row():
    source = (ROOT / "src/driver/registry.rs").read_text(encoding="utf-8")
    specs = api_reference.registry_specs(ROOT)
    assert len(specs) == source.count("feature!(")
    assert {"coherent-transport", "pauli-spin-susceptibility"} <= {spec["id"] for spec in specs}
    coherent = next(spec for spec in specs if spec["id"] == "coherent-transport")
    assert "intrinsic Hall" in coherent["title"]


@pytest.mark.parametrize("replacement,message", [
    ("", "lacks a settings-contract"),
    ("// settings-contract: unreviewed\n", "unknown settings-contract"),
    ("// settings-contract: periodic-common\n", "disagrees with space/route"),
])
def test_new_registry_route_requires_an_applicable_settings_contract(tmp_path, replacement, message):
    path = tmp_path / "src/driver/registry.rs"
    path.parent.mkdir(parents=True)
    path.write_text(replacement + 'feature!("new-property", "New property", Molecule, Driver, '
                    '"driver::run", "new-property", "new_property", "get_new_property", []),',
                    encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        api_reference.registry_specs(tmp_path)


def test_every_registered_route_has_an_explicit_settings_applicability_contract():
    specs = api_reference.registry_specs(ROOT)
    assert all(spec["settings_contract"] in api_reference.SETTINGS_CONTRACTS for spec in specs)
    assert next(spec for spec in specs if spec["id"] == "multicomponent")["settings_contract"] == "multicomponent"
    assert next(spec for spec in specs if spec["id"] == "periodic-coulomb")["settings_contract"] == "periodic-operator"


def test_routine_api_check_exits_nonzero_when_a_route_loses_its_settings_contract(tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/driver").mkdir(parents=True)
    for name in ("rust-api.md", "cli.md", "python-api.md"):
        shutil.copy2(ROOT / "docs" / name, tmp_path / "docs" / name)
    script = tmp_path / "tools/api_reference.py"
    shutil.copy2(ROOT / "tools/api_reference.py", script)
    registry = (ROOT / "src/driver/registry.rs").read_text(encoding="utf-8")
    assert "    // settings-contract: molecular-common\n" in registry
    (tmp_path / "src/driver/registry.rs").write_text(
        registry.replace("    // settings-contract: molecular-common\n", "", 1), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), "--check"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "lacks a settings-contract declaration" in result.stderr


@pytest.mark.parametrize("literal,expected", [
    ('"before \\\n    after"', "before after"),
    ('"before \\\r\n \t after"', "before after"),
    (r'"escaped\\backslash"', "escaped\\backslash"),
    (r'"escaped\"quote"', 'escaped"quote'),
])
def test_registry_literal_matches_rust_continuations_and_escapes(literal, expected):
    assert api_reference.registry_literal(literal) == expected


def test_public_api_markdown_matches_signatures_and_descriptions():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, str(root / "tools/api_reference.py"), "--check"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_api_guide_report_wrappers_match_the_registry():
    document = (ROOT / "docs/api-guide.md").read_text(encoding="utf-8")
    chapter = document.split("## Choose a Python entry point\n", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"\| `([\w-]+)` \| `(\w+)` \| `(\w+)` \|", chapter)
    assert {name for name, _, _ in rows} >= {"hessian", "polarizability", "dynamic-polarizability"}
    registry = {entry["id"]: entry for entry in api_reference.registry_specs(ROOT)}
    for feature, native, ase in rows:
        assert (native, ase) == (registry[feature]["python"], registry[feature]["ase"])
