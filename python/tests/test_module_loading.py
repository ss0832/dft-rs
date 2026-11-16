# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Compiler-resolved module ownership, isolated by Cargo target and cfg."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/verify_release.py"
SPEC = importlib.util.spec_from_file_location("module_release", RUNNER)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="compiler module fixtures require Cargo and Clippy")
VALUE = "pub fn value() -> usize { 1 }\n"


def project(root, files, features='default = []\nextra = []\n'):
    (root / "src").mkdir()
    (root / "Cargo.toml").write_text(
        '[package]\nname = "module-policy-fixture"\nversion = "0.1.0"\nedition = "2021"\n'
        '[features]\n' + features, encoding="utf-8")
    (root / "Cargo.lock").write_text(
        'version = 3\n[[package]]\nname = "module-policy-fixture"\nversion = "0.1.0"\n', encoding="utf-8")
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def check(root, *, extra=False, all_targets=True):
    configuration = {"requested": ["extra"] if extra else [], "enabled": []}
    command = release.module_command({"package": "module-policy-fixture"}, configuration, root / "target")
    command[2:2] = ["--offline", "--locked"]
    if not all_targets:
        command[command.index("--all-targets")] = "--lib"
    return subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90,
                          env=release.module_environment(dict(os.environ, CARGO_BUILD_JOBS="1", RAYON_NUM_THREADS="1")))


def assert_duplicate(result):
    assert result.returncode != 0
    assert "duplicate_mod" in result.stderr and "file is loaded as a module multiple times" in result.stderr


@pytest.mark.parametrize("canonical", [False, True])
def test_original_periodic_charge_layout_and_canonical_use(tmp_path, canonical):
    child = ('use super::periodic_charge;\npub fn energy() -> usize { periodic_charge::value() }\n'
             if canonical else '#[path = "periodic_charge.rs"]\npub mod classical;\n')
    project(tmp_path, {"src/lib.rs": "pub mod continuous_species;\n",
                      "src/continuous_species/mod.rs": "pub mod periodic_charge;\npub mod periodic_model;\n",
                      "src/continuous_species/periodic_charge.rs": VALUE,
                      "src/continuous_species/periodic_model.rs": child})
    result = check(tmp_path)
    if canonical:
        assert result.returncode == 0, result.stderr
    else:
        assert_duplicate(result)


@pytest.mark.parametrize("kind", ["lexical-alias", "nested", "cfg-attribute", "macro"])
def test_nested_and_aliased_source_identity_uses_the_compiler(tmp_path, kind):
    source = 'pub mod shared;\n'
    files = {"src/shared.rs": VALUE}
    if kind == "lexical-alias":
        source += '#[path = "nested/../shared.rs"]\npub mod alias;\n'
        files["src/nested/.keep"] = ""
    elif kind == "nested":
        source += 'pub mod branch;\n'
        files["src/branch.rs"] = '#[path = "shared.rs"]\npub mod alias;\n'
    elif kind == "cfg-attribute":
        source += '#[cfg_attr(feature = "extra", path = "shared.rs")]\n#[cfg_attr(not(feature = "extra"), path = "other.rs")]\npub mod alias;\n'
        files["src/other.rs"] = VALUE
    else:
        source += 'macro_rules! load { () => { #[path = "shared.rs"] pub mod alias; }; }\nload!();\n'
    files["src/lib.rs"] = source
    project(tmp_path, files)
    if kind == "cfg-attribute":
        assert check(tmp_path).returncode == 0
    assert_duplicate(check(tmp_path, extra=kind == "cfg-attribute"))


def test_symlink_source_alias_is_resolved_when_supported(tmp_path):
    project(tmp_path, {"src/lib.rs": 'pub mod shared;\n#[path="alias.rs"] pub mod alias;\n', "src/shared.rs": VALUE})
    try:
        (tmp_path / "src/alias.rs").symlink_to("shared.rs")
    except OSError as error:
        pytest.skip(f"filesystem cannot create a symlink: {error.errno}")
    assert_duplicate(check(tmp_path))


def test_test_only_duplicate_is_caught_by_all_targets(tmp_path):
    project(tmp_path, {"src/lib.rs": 'pub mod shared;\n#[cfg(test)]\n#[path="shared.rs"] pub mod test_alias;\n', "src/shared.rs": VALUE})
    assert check(tmp_path, all_targets=False).returncode == 0
    assert_duplicate(check(tmp_path))


def test_reexports_and_separate_integration_targets_are_legitimate(tmp_path):
    project(tmp_path, {"src/lib.rs": 'pub mod shared;\npub use shared as alias;\n', "src/shared.rs": VALUE,
                      "tests/first.rs": '#[path="../src/shared.rs"] pub mod shared;\n',
                      "tests/second.rs": '#[path="../src/shared.rs"] pub mod shared;\n'})
    result = check(tmp_path)
    assert result.returncode == 0, result.stderr


def test_mutually_exclusive_cfg_paths_pass_the_entire_real_cargo_matrix(tmp_path):
    project(tmp_path, {"src/shared.rs": VALUE, "src/lib.rs":
                      '#[cfg(feature="extra")]\n#[path="shared.rs"] pub mod selected;\n'
                      '#[cfg(not(feature="extra"))]\n#[path="shared.rs"] pub mod selected;\n'})
    assert release.main(["exclusive", "--module-only"], root=tmp_path) == 0
    plan = json.loads((tmp_path / "validation/exclusive-module-plan.json").read_text())
    assert {tuple(c["enabled"]) for c in plan["configurations"]} == {(), ("default",), ("extra",), ("default", "extra")}


def test_feature_only_duplicate_fails_the_normal_module_entrypoint(tmp_path):
    project(tmp_path, {"src/shared.rs": VALUE, "src/lib.rs":
                      'pub mod shared;\n#[cfg(feature="extra")]\n#[path="shared.rs"] pub mod alias;\n'})
    assert release.main(["feature", "--module-only"], root=tmp_path) != 0
    logs = list((tmp_path / "validation").glob("feature-clippy-*.log"))
    assert any("duplicate_mod" in path.read_text() for path in logs)


def test_transitive_feature_cfgs_are_checked_as_actual_cargo_compilations(tmp_path):
    project(tmp_path, {"src/shared.rs": VALUE, "src/lib.rs":
                      '#[cfg(feature="base")]\n#[path="shared.rs"] pub mod selected;\n'
                      '#[cfg(not(feature="base"))]\n#[path="shared.rs"] pub mod selected;\n'},
            features='default = []\nbase = []\nderived = ["base"]\n')
    assert release.main(["transitive", "--module-only"], root=tmp_path) == 0
    plan = json.loads((tmp_path / "validation/transitive-module-plan.json").read_text())
    assert len(plan["configurations"]) == 6
    assert all("base" in c["enabled"] for c in plan["configurations"] if "derived" in c["enabled"])


def test_local_allow_cannot_disable_the_explicit_module_gate(tmp_path):
    project(tmp_path, {"src/shared.rs": VALUE, "src/lib.rs":
                      'pub mod shared;\n#[allow(clippy::duplicate_mod)]\n#[path="shared.rs"] pub mod alias;\n'})
    result = check(tmp_path)
    assert result.returncode != 0 and "duplicate_mod" in result.stderr


@pytest.mark.parametrize("variable,value", [
    ("RUSTFLAGS", "--cap-lints=allow"), ("RUSTFLAGS", "--cap-lints allow"),
    ("CARGO_ENCODED_RUSTFLAGS", "--cap-lints=allow"), ("CARGO_ENCODED_RUSTFLAGS", "--cap-lints\x1fallow")])
def test_user_lint_cap_cannot_hide_an_actual_duplicate(tmp_path, monkeypatch, variable, value):
    project(tmp_path, {"src/shared.rs": VALUE, "src/lib.rs":
                      'pub mod shared;\n#[path="shared.rs"] pub mod alias;\n'})
    monkeypatch.setenv(variable, value)
    assert_duplicate(check(tmp_path))


def test_current_repository_declares_the_dedicated_forbid_policy():
    assert 'duplicate_mod = "forbid"' in (ROOT / "Cargo.toml").read_text()
    assert "--forbid=clippy::duplicate_mod" in RUNNER.read_text()
