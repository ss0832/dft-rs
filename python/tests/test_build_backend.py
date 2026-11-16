# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Dependency metadata keeps its graph when build roots are normalized."""
import base64
from contextlib import nullcontext
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from urllib.parse import quote
import zipfile

import pytest
from tools import build_backend, portable_build


def test_declared_wheel_licenses_cover_every_bundled_notice_and_license():
    root = build_backend.ROOT
    match = re.search(r'(?m)^license-files\s*=\s*(\[[^\n]+\])',
                      (root / "pyproject.toml").read_text(encoding="utf-8"))
    assert match, "the Python package must declare its license-file patterns"
    declared = json.loads(match[1])
    required = {root / "LICENSE", root / "THIRD_PARTY_NOTICES.md"}
    required.update((root / "third_party").rglob("LICENSE"))
    required.update((root / "third_party").rglob("NOTICE.md"))
    selected = {path for pattern in declared for path in root.glob(pattern) if path.is_file()}
    assert required <= selected, "a bundled upstream license or notice would be omitted from the wheel"


def test_build_root_normalization_preserves_third_party_provenance(tmp_path):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    upstream = {"name": "upstream-package", "version": "1.2.3",
                "authors": [{"name": "Upstream Author", "email": "maintainer@example.org"}],
                "copyright": "Copyright 2020 Upstream Authors",
                "licenses": [{"license": {"id": "BSD-3-Clause", "text": {"content": "Original license text"}}}],
                "externalReferences": [{"type": "vcs", "url": "https://example.org/upstream/source"}],
                "purl": "pkg:cargo/upstream-package@1.2.3"}
    original = {"components": [upstream], "build_cache": str(roots[1] / "registry")}
    normalized = json.loads(build_backend.normalize_sbom(json.dumps(original), roots))
    assert normalized["components"] == [upstream]
    assert normalized["build_cache"] == "cargo/registry"


def test_normalization_preserves_components_edges_and_wheel_record(tmp_path):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    reference = "path+" + roots[3].as_uri() + "#dft-rs@0.2.0-alpha.1"
    dependency = "registry+portable-dependency@1"
    sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5",
            "metadata": {"component": {"bom-ref": reference, "name": "dft-rs"}},
            "components": [{"bom-ref": dependency, "name": "portable-dependency",
                            "properties": [{"name": "source", "value": str(roots[1] / "registry")}]}],
            "dependencies": [{"ref": reference, "dependsOn": [dependency]}],
            "encoded": quote(str(roots[0]), safe="")}
    wheel = tmp_path / "test.whl"
    member = "dft_rs_python-0.3.0.dist-info/sboms/dft-rs.cyclonedx.json"
    metadata = b"Metadata-Version: 2.4\nName: dft-rs-python\nVersion: 0.3.0\nAuthor: ss0832\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, json.dumps(sbom))
        archive.writestr("dft_rs_python-0.3.0.dist-info/METADATA", metadata)
        archive.writestr("dft_rs_python-0.3.0.dist-info/RECORD", "")
    docs = {"docs/cli.md": b"# CLI\n", "docs/property_targets.csv": b"feature,scope\n"}
    build_backend.finalize_wheel(wheel, roots, documentation=docs)
    with zipfile.ZipFile(wheel) as archive:
        for name, payload in docs.items():
            assert archive.read("dft_rs/" + name) == payload
        portable = json.loads(archive.read(member))
        assert len(portable["components"]) == len(sbom["components"])
        assert portable["dependencies"][0]["ref"] == portable["metadata"]["component"]["bom-ref"]
        assert portable["dependencies"][0]["dependsOn"] == [portable["components"][0]["bom-ref"]]
        assert "file:./dft-rs" in portable["dependencies"][0]["ref"]
        assert portable["components"][0]["properties"][0]["value"] == "cargo/registry"
        assert archive.read("dft_rs_python-0.3.0.dist-info/METADATA") == metadata
        rows = list(csv.reader(io.StringIO(archive.read("dft_rs_python-0.3.0.dist-info/RECORD").decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, length in rows:
            if name.endswith("/RECORD"):
                assert digest == length == ""
                continue
            data = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            assert digest == "sha256=" + expected and int(length) == len(data)
            assert all(str(root).encode() not in data for root in roots)


def test_native_private_bytes_are_refused_instead_of_patched(tmp_path):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    wheel = tmp_path / "native.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("dft_rs/core.so", b"native-prefix\x00" + str(roots[3]).encode("utf-16-le"))
        archive.writestr("dft_rs_python-0.3.0.dist-info/RECORD", "")
    original = wheel.read_bytes()
    with pytest.raises(ValueError, match="private build path"):
        build_backend.finalize_wheel(wheel, roots)
    assert wheel.read_bytes() == original


@pytest.mark.parametrize("member,payload", [
    ("dft_rs/readme.md", "".join(map(chr, [0x8a08, 0x7b97]))),
    ("dft_rs/notes.txt", "person" + "@sample.invalid"),
    ("dft_rs/notes.txt", "Sentinel_" + "b" * 32),
    ("dft_rs_python-0.3.0.dist-info/licenses/LICENSE", "Legal text\n" + "Sentinel_" + "d" * 32),
    ("dft_rs/.work/probe.txt", "excluded work evidence"),
    ("dft_rs/__pycache__/module.pyc", "cache"),
])
def test_wheel_rejects_hygiene_leaks_and_private_work_members_without_modification(tmp_path, member, payload):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    wheel = tmp_path / "hygiene.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, payload)
        archive.writestr("dft_rs_python-0.3.0.dist-info/RECORD", "")
    original = wheel.read_bytes()
    with pytest.raises(ValueError, match="(hygiene|private work)"):
        build_backend.finalize_wheel(wheel, roots)
    assert wheel.read_bytes() == original


def test_extensionless_wheel_license_keeps_required_attribution_bytes(tmp_path):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    wheel = tmp_path / "legal.whl"
    payload = ("Copyright Public Author <" + "author" + "@sample.invalid>\n" + chr(0x6cd5)).encode()
    member = "dft_rs_python-0.3.0.dist-info/licenses/LICENSE"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(member, payload)
        archive.writestr("dft_rs_python-0.3.0.dist-info/RECORD", "")
    build_backend.finalize_wheel(wheel, roots)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.read(member) == payload


@pytest.mark.parametrize("name,payload", [
    ("docs/probe.md", "".join(map(chr, [0x8a08, 0x7b97]))),
    ("docs/probe.md", "person" + "@sample.invalid"),
    (".work/private.json", "retained private evidence"),
    ("examples/calculation.log", "retained calculation output"),
    ("python/dft_rs/core.so", "compiled cache"),
    ("../escape.txt", "unsafe"),
])
def test_source_archive_rejects_hygiene_and_excluded_members(tmp_path, name, payload):
    archive = tmp_path / "source.tar.gz"
    data = payload.encode("utf-8")
    with tarfile.open(archive, "w:gz") as destination:
        member = tarfile.TarInfo("dft_rs_python-0.3.0/" + name)
        member.size = len(data)
        destination.addfile(member, io.BytesIO(data))
    original = archive.read_bytes()
    with pytest.raises(ValueError, match="(hygiene|private|unsafe)"):
        build_backend.audit_sdist(archive)
    assert archive.read_bytes() == original


def test_source_archive_preserves_scientific_text_and_required_license_attribution(tmp_path):
    archive = tmp_path / "source.tar.gz"
    contents = {"docs/notes.md": "Scientific α and β coefficients.",
                "third_party/component/NOTICE.md": "Copyright Project Author <" + "author" + "@sample.invalid>"}
    with tarfile.open(archive, "w:gz") as destination:
        for name, text in contents.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo("dft_rs_python-0.3.0/" + name)
            member.size = len(data)
            destination.addfile(member, io.BytesIO(data))
    expected = {"docs/notes.md": contents["docs/notes.md"].encode()}
    build_backend.audit_sdist(archive, documentation=expected)
    with pytest.raises(ValueError, match=r"documentation differs: docs/notes\.md"):
        build_backend.audit_sdist(archive, documentation={"docs/notes.md": b"Older notes."})
    with pytest.raises(ValueError, match=r"documentation differs: docs/guide\.md"):
        build_backend.audit_sdist(archive, documentation={**expected, "docs/guide.md": b"# Guide\n"})


def test_sdist_hook_checks_source_before_build_and_actual_archive_afterward(tmp_path, monkeypatch):
    archive = tmp_path / "source.tar.gz"
    data = ("Sentinel_" + "c" * 32).encode()
    with tarfile.open(archive, "w:gz") as destination:
        member = tarfile.TarInfo("dft_rs_python-0.3.0/docs/probe.md")
        member.size = len(data)
        destination.addfile(member, io.BytesIO(data))
    calls = []
    monkeypatch.setattr(build_backend, "audit_source_policy", lambda root: calls.append("source"))

    def build(*args):
        assert calls == ["source"]
        calls.append("backend")
        return archive.name

    monkeypatch.setitem(sys.modules, "maturin", SimpleNamespace(build_sdist=build))
    with pytest.raises(ValueError, match="hygiene"):
        build_backend.build_sdist(str(tmp_path))
    assert calls == ["source", "backend"]


def test_wheel_source_policy_failure_prevents_backend_environment_and_cargo(monkeypatch):
    def reject(root):
        raise SystemExit("source hygiene failed")

    monkeypatch.setattr(build_backend, "audit_source_policy", reject)
    monkeypatch.setattr(build_backend, "audit_cargo_msrv", lambda *args: pytest.fail("source failure reached Cargo"))
    with pytest.raises(SystemExit, match="source hygiene"):
        with build_backend._environment():
            pytest.fail("source failure reached backend")


def test_sbom_keeps_verified_dependency_authors_but_rejects_arbitrary_contact_properties():
    contact = "upstream" + "@sample.invalid"
    metadata = {"packages": [{"name": "portable-dependency", "version": "1", "authors": [contact]}]}
    attributions = portable_build.attribution_contacts(metadata)
    component = {"name": "portable-dependency", "version": "1", "purl": "pkg:cargo/portable-dependency@1",
                 "author": "Public Author <" + contact + ">"}
    sbom = {"bomFormat": "CycloneDX", "components": [component]}
    name = "dft_rs_python-0.3.0.dist-info/sboms/test.cyclonedx.json"
    assert not portable_build.text_hygiene_findings(name, json.dumps(sbom), attributions=attributions)
    assert portable_build.text_hygiene_findings(name, json.dumps(sbom))
    component["properties"] = [{"name": "unapproved-note", "value": contact}]
    assert portable_build.text_hygiene_findings(name, json.dumps(sbom), attributions=attributions)
    component.pop("properties")
    component["purl"] = "pkg:cargo/different-package@1"
    assert portable_build.text_hygiene_findings(name, json.dumps(sbom), attributions=attributions)


def test_sbom_attribution_query_includes_other_targets_without_weakening_msrv(tmp_path, monkeypatch):
    calls = []
    contact = "upstream" + "@sample.invalid"

    def cargo(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"packages": [
            {"name": "portable-dependency", "version": "1", "authors": [contact]}]}), "")

    monkeypatch.setattr(portable_build.subprocess, "run", cargo)
    assert portable_build.audit_cargo_attributions(tmp_path) == {"portable-dependency@1": {contact}}
    assert len(calls) == 1 and "--filter-platform" not in calls[0]
    assert "--offline" in calls[0] and "--locked" in calls[0]


def test_nested_build_environments_keep_fingerprints_and_override_precedence(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    original = ["-Copt-level=1", "--cfg=contract_probe",
                f"--remap-path-prefix={project}=custom-public-root"]
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", "\x1f".join(original))
    first, roots = portable_build.build_environment(project)
    monkeypatch.setenv("CARGO_ENCODED_RUSTFLAGS", first["CARGO_ENCODED_RUSTFLAGS"])
    second, second_roots = portable_build.build_environment(project)
    assert first["CARGO_ENCODED_RUSTFLAGS"] == second["CARGO_ENCODED_RUSTFLAGS"]
    assert roots == second_roots
    flags = second["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
    assert flags[:len(original)] == original
    generated = [f"--remap-path-prefix={root}={public}"
                 for root, public in zip(roots, ("user", "cargo", "rustup", "dft-rs"))]
    assert flags[-len(generated):] == generated
    assert all(flags.count(flag) == 1 for flag in generated)


def test_wheel_author_verification_uses_metadata_after_the_backend_populates_its_cache(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(build_backend, "_environment", lambda: nullcontext([]))

    def build(*args):
        events.append("backend")
        return "probe.whl"

    def authors(*args):
        assert events == ["backend", "identity"]
        events.append("authors")
        return {"portable-dependency@1": set()}

    def audit(path, roots, **kwargs):
        assert events == ["backend", "identity", "authors"]
        assert kwargs["attributions"] == {"portable-dependency@1": set()}
        assert kwargs["documentation"]["docs/property_targets.csv"] == (
            build_backend.ROOT / "docs/property_targets.csv"
        ).read_bytes()
        events.append("audit")

    monkeypatch.setitem(sys.modules, "maturin", SimpleNamespace(build_wheel=build))
    monkeypatch.setattr(build_backend, "audit_wheel_identity", lambda path: events.append("identity"))
    monkeypatch.setattr(build_backend, "audit_cargo_attributions", authors)
    monkeypatch.setattr(build_backend, "finalize_wheel", audit)
    assert build_backend.build_wheel(str(tmp_path)) == "probe.whl"
    assert events == ["backend", "identity", "authors", "audit"]


def distribution_metadata(name="dft-rs-python"):
    return f"Metadata-Version: 2.4\nName: {name}\nVersion: 0.3.0\n\n".encode("ascii")


def identity_wheel(tmp_path, *, filename="dft_rs_python-0.3.0-py3-none-any.whl",
                   directory="dft_rs_python-0.3.0.dist-info", name="dft-rs-python", extra=None):
    wheel = tmp_path / filename
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(directory + "/METADATA", distribution_metadata(name))
        archive.writestr(directory + "/RECORD", "")
        if extra:
            archive.writestr(extra, "")
    return wheel


def test_python_distribution_identity_is_separate_from_crate_and_import_names():
    assert build_backend.distribution_name() == "dft-rs-python"
    project = (build_backend.ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cargo = (build_backend.ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert 'module-name = "dft_rs.dft_rs_core"' in project
    assert 'dft-rs = "dft_rs.native:cli_main"' in project
    assert '[package]\nname = "dft-rs"' in cargo


def test_wheel_identity_accepts_the_canonical_python_distribution(tmp_path):
    build_backend.audit_wheel_identity(identity_wheel(tmp_path))


@pytest.mark.parametrize("options", [
    {"name": "dft-rs"},
    {"filename": "dft_rs-0.3.0-py3-none-any.whl"},
    {"directory": "dft_rs-0.3.0.dist-info"},
    {"extra": "dft_rs-0.3.0.data/scripts/probe"},
    {"extra": "dft_rs-0.3.0.dist-info/WHEEL"},
])
def test_wheel_identity_rejects_stale_or_mixed_distribution_names_without_mutation(tmp_path, options):
    wheel = identity_wheel(tmp_path, **options)
    before = wheel.read_bytes()
    with pytest.raises(ValueError, match="(Name|distribution)"):
        build_backend.audit_wheel_identity(wheel)
    assert wheel.read_bytes() == before


def identity_sdist(tmp_path, *, filename="dft_rs_python-0.3.0.tar.gz", root="dft_rs_python-0.3.0",
                   name="dft-rs-python", project=True, extra=False):
    path = tmp_path / filename
    with tarfile.open(path, "w:gz") as archive:
        contents = {root + "/PKG-INFO": distribution_metadata(name)}
        if project:
            contents[root + "/pyproject.toml"] = b'[project]\nname = "dft-rs-python"\n'
        if extra:
            contents["other-root/file.txt"] = b"separate root"
        for filename, data in contents.items():
            member = tarfile.TarInfo(filename)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return path


def test_sdist_identity_accepts_the_canonical_python_distribution(tmp_path):
    build_backend.audit_sdist_identity(identity_sdist(tmp_path))


@pytest.mark.parametrize("options", [
    {"name": "dft-rs"}, {"filename": "dft_rs-0.3.0.tar.gz"},
    {"root": "dft_rs-0.3.0"}, {"project": False}, {"extra": True},
])
def test_sdist_identity_rejects_stale_or_mixed_names_without_mutation(tmp_path, options):
    archive = identity_sdist(tmp_path, **options)
    before = archive.read_bytes()
    with pytest.raises(ValueError, match="(Name|distribution|canonical root)"):
        build_backend.audit_sdist_identity(archive)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("hook", ["prepare_metadata_for_build_wheel", "prepare_metadata_for_build_editable"])
@pytest.mark.parametrize("name", ["dft-rs-python", "dft-rs"])
def test_metadata_hooks_validate_the_actual_generated_name(tmp_path, monkeypatch, hook, name):
    directory = "dft_rs_python-0.3.0.dist-info"

    def prepare(destination, settings):
        target = Path(destination) / directory
        target.mkdir()
        (target / "METADATA").write_bytes(distribution_metadata(name))
        return directory

    monkeypatch.setitem(sys.modules, "maturin", SimpleNamespace(prepare_metadata_for_build_wheel=prepare))
    if name == "dft-rs-python":
        assert getattr(build_backend, hook)(str(tmp_path)) == directory
    else:
        with pytest.raises(ValueError, match="metadata Name"):
            getattr(build_backend, hook)(str(tmp_path))


def test_real_build_hook_refuses_stale_identity_before_finalization(tmp_path, monkeypatch):
    wheel = identity_wheel(tmp_path, name="dft-rs")
    monkeypatch.setattr(build_backend, "_environment", lambda: nullcontext([]))
    monkeypatch.setitem(sys.modules, "maturin", SimpleNamespace(build_wheel=lambda *args: wheel.name))
    monkeypatch.setattr(build_backend, "finalize_wheel", lambda *args: pytest.fail("stale identity was finalized"))
    with pytest.raises(ValueError, match="metadata Name"):
        build_backend.build_wheel(str(tmp_path))


@pytest.mark.parametrize("encoded", [False, True])
def test_unknown_private_metadata_path_fails_closed(tmp_path, encoded):
    roots = [tmp_path / name for name in ("home", "cargo", "rustup", "project")]
    wheel = tmp_path / "metadata.whl"
    unknown = Path.home().parent / (Path.home().name + "-unmapped") / "build"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("dft_rs_python-0.3.0.dist-info/sboms/extra.json", json.dumps({"path": quote(str(unknown), safe="") if encoded else str(unknown)}))
        archive.writestr("dft_rs_python-0.3.0.dist-info/RECORD", "")
    with pytest.raises(ValueError, match="private build path"):
        build_backend.finalize_wheel(wheel, roots)


def rust_metadata(declared="1.82", direct="1.84.0", transitive="1.85"):
    names = ["root", "direct", "transitive", "development"]
    versions = [declared, direct, transitive, "99.0"]
    packages = [{"id": name, "name": name, "version": "0.3.0", "rust_version": version}
                for name, version in zip(names, versions)]

    def dependency(name, kind=None):
        return {"pkg": name, "dep_kinds": [{"kind": kind, "target": None}]}

    nodes = [{"id": "root", "deps": [dependency("direct"), dependency("development", "dev")]},
             {"id": "direct", "deps": [dependency("transitive")]},
             {"id": "transitive", "deps": []}, {"id": "development", "deps": []}]
    return {"packages": packages, "resolve": {"root": "root", "nodes": nodes}}


def test_rust_floor_guard_detects_direct_and_transitive_incompatibility():
    with pytest.raises(ValueError, match="direct.*1.84.0.*transitive.*1.85"):
        portable_build.validate_rust_requirements(rust_metadata())
    with pytest.raises(ValueError, match="transitive.*1.85"):
        portable_build.validate_rust_requirements(rust_metadata(declared="1.84"))


@pytest.mark.parametrize("declared", ["1.85", "1.85.0", "1.86"])
def test_rust_floor_guard_normalizes_patch_versions_and_excludes_development_edges(declared):
    assert portable_build.validate_rust_requirements(rust_metadata(declared=declared)) == declared


def test_metadata_audit_respects_the_requested_target_and_offline_mode(tmp_path, monkeypatch):
    calls = []

    def cargo(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(rust_metadata(declared="1.85")), "")

    monkeypatch.setattr(portable_build.subprocess, "run", cargo)
    portable_build.audit_cargo_msrv(tmp_path, {"CARGO_BUILD_TARGET": "unused-default"},
                                   ["--target=selected-target", "--offline"])
    assert len(calls) == 1
    assert calls[0][0][-3:] == ["--filter-platform", "selected-target", "--offline"]
    assert calls[0][1]["check"] is True


def test_wheel_build_environment_runs_the_shared_rust_floor_guard(monkeypatch):
    checked = []
    monkeypatch.setattr(build_backend, "audit_cargo_msrv", lambda root, env: checked.append(root))
    with build_backend._environment():
        assert checked == [build_backend.ROOT]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="source dependency audit requires Cargo")
def test_source_rust_floor_covers_the_locked_dependency_graph():
    portable_build.audit_cargo_msrv(build_backend.ROOT, dict(os.environ), ["--offline"])
