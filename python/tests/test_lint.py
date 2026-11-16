# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The lint gate checks the whole package and rejects actual Python defects."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools" / "lint.py"


def run_lint(*paths, cwd=None):
    return subprocess.run([sys.executable, str(RUNNER), *map(str, paths)],
                          cwd=cwd, text=True, capture_output=True, check=False)


def test_python_sources_and_tools_are_lint_clean():
    result = run_lint()
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source, rule", [
    ("import math\n", "F401"),
    ("print(missing_value)\n", "F821"),
    ("def broken(:\n", "invalid-syntax"),
])
def test_lint_rejects_defects_outside_the_package(tmp_path, source, rule):
    path = tmp_path / "probe.py"
    path.write_text(source, encoding="utf-8")
    result = run_lint(path.name, cwd=tmp_path)
    assert result.returncode != 0
    assert rule in result.stdout + result.stderr


def test_lint_accepts_valid_code_without_writing_a_cache(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text("import math\nprint(math.pi)\n", encoding="utf-8")
    result = run_lint(path, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".ruff_cache").exists()


@pytest.mark.parametrize("payload,rule", [
    ("".join(map(chr, [0x8a08, 0x7b97, 0x6761, 0x4ef6])), "Japanese-text"),
    ("person" + "@sample.invalid", "personal-contact"),
    ("/" + "home/synthetic/private", "private-path"),
    ("C:" + chr(92) + "Users" + chr(92) + "synthetic", "private-path"),
    ("Sentinel_" + "a" * 32, "conversation-id"),
    ("DESKTOP" + "-SYNTHETIC", "private-host"),
])
def test_hygiene_rejects_private_or_non_english_text_without_repeating_it(tmp_path, payload, rule):
    path = tmp_path / "probe.py"
    path.write_text("# " + payload + "\n", encoding="utf-8")
    result = run_lint(path)
    assert result.returncode == 1
    assert rule in result.stderr
    assert payload not in result.stderr + result.stdout


def test_hygiene_preserves_scientific_symbols_and_matrix_operators(tmp_path):
    path = tmp_path / "probe.py"
    path.write_text('# α and β are scientific symbols.\ndef product(a, b):\n    return a@b.conj().T\n', encoding="utf-8")
    result = run_lint(path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_hygiene_covers_docs_rust_and_tests_without_private_work_evidence(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("portable_policy", ROOT / "tools/portable_build.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    (tmp_path / "Cargo.toml").write_text('[package]\ninclude = [\n"/src/**",\n"/docs/**",\n"/tests/**",\n]\n')
    payload = "".join(map(chr, [0x8a08, 0x7b97]))
    for directory in ("src", "docs", "tests", ".work"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "probe.txt").write_text(payload, encoding="utf-8")
    findings = policy.source_hygiene_findings(tmp_path)
    assert {name for name, _, rule in findings if rule == "Japanese-text"} == {
        "src/probe.txt", "docs/probe.txt", "tests/probe.txt"}
    (tmp_path / "docs/probe.txt").write_text("# Greek α and β; scientific bibliography remains unchanged.\n", encoding="utf-8")
    assert len(policy.source_hygiene_findings(tmp_path)) == 2
    (tmp_path / "src/.work").mkdir()
    (tmp_path / "src/.work/private.txt").write_text("private validation evidence")
    assert ("src/.work", 1, "private-work-in-package") in policy.source_hygiene_findings(tmp_path)
    manifest = tmp_path / "Cargo.toml"
    manifest.write_text(manifest.read_text().replace("\n]\n", '\n"!/**/.work/**",\n]\n'))
    assert not any(rule == "private-work-in-package" for _, _, rule in policy.source_hygiene_findings(tmp_path))
    (tmp_path / "extra.py").write_text("# " + payload, encoding="utf-8")
    assert ("extra.py", 1, "Japanese-text") in policy.source_hygiene_findings(tmp_path)


def test_metadata_and_legal_contact_exceptions_are_narrow(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("portable_policy", ROOT / "tools/portable_build.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    contact = "author" + "@sample.invalid"
    declaration = 'authors = ["Project Author <' + contact + '>"]'
    assert not policy.text_hygiene_findings("Cargo.toml", declaration, {contact})
    assert not policy.text_hygiene_findings("Cargo.toml.orig", 'authors = [\n"' + contact + '"\n]', {contact})
    assert not policy.text_hygiene_findings("PKG-INFO", "Author-email: " + contact, {contact})
    assert policy.text_hygiene_findings("Cargo.toml", "# " + contact, {contact})
    assert policy.text_hygiene_findings("python/dft_rs/module.py", "# " + contact, {contact})
    assert not policy.text_hygiene_findings("third_party/component/NOTICE.md", contact)
    assert policy.text_hygiene_findings("python/NOTICE.md", contact)
    title = "".join(map(chr, [0x8a08, 0x7b97]))
    citation = "Author (2024). " + title + ". https://doi.org/10.0000/example"
    assert not policy.text_hygiene_findings("docs/references.md", citation)
    assert policy.text_hygiene_findings("docs/notes.md", citation)
