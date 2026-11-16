# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""The repository-owned release launcher must run pytest and preserve failures."""
import importlib.util
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

from tools import make_release, stage_release

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tools/verify_release.py"
SPEC = importlib.util.spec_from_file_location("verify_release", RUNNER)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
SINGLE_PLAN = {"package": "probe", "targets": [], "configurations": [{"requested": [], "enabled": []}]}


def artifacts(root):
    package = root / "python/dft_rs"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    target = root / "target-python/release"
    target.mkdir(parents=True)
    library = ("dft_rs.dll" if sys.platform == "win32" else
               "libdft_rs.dylib" if sys.platform == "darwin" else "libdft_rs.so")
    for name in (library, "dft-rs" + (".exe" if os.name == "nt" else ""),
                 "dft_rs_cli" + (".exe" if os.name == "nt" else ""),
                 "fit_substituent" + (".exe" if os.name == "nt" else "")):
        (target / name).write_bytes(b"fixture artifact")


@pytest.mark.parametrize("standalone", [[], ["--rust-cli", "missing-executable"]])
def test_installed_mode_requires_a_standalone_rust_cli(tmp_path, capsys, standalone):
    wheel = tmp_path / "fixture.whl"
    sdist = tmp_path / "fixture.tar.gz"
    wheel.touch()
    sdist.touch()
    with pytest.raises(SystemExit) as error:
        release.main(["probe", "--installed-only", "--wheel", str(wheel), "--sdist",
                      str(sdist), *standalone], root=tmp_path)
    assert error.value.code == 2
    assert "--installed-only requires --rust-cli" in capsys.readouterr().err
    assert not (tmp_path / "validation").exists()


@pytest.mark.parametrize("wheel_status,ase_status,expected", [(7, 0, 7), (0, 9, 9), (0, 0, 0)])
def test_installed_mode_separates_environments_and_preserves_pip_failure(
        tmp_path, monkeypatch, wheel_status, ase_status, expected):
    (tmp_path / "Cargo.toml").write_text('[package]\nversion = "0.3.0"\n', encoding="ascii")
    wheel = tmp_path / "fixture.whl"
    sdist = tmp_path / "fixture.tar.gz"
    wheel.write_bytes(b"wheel fixture")
    sdist.write_bytes(b"sdist fixture")
    rust_cli = tmp_path / "standalone-dft-rs"
    rust_cli.touch()
    monkeypatch.setenv("PYTHONPATH", "unrelated-source-tree")
    monkeypatch.setenv("DFT_RS_CLI", "unrelated-executable")
    installs = {}
    checked = []
    downloaded = []
    ase_checked = []

    def run(command, log, root, env):
        assert "PYTHONPATH" not in env and "DFT_RS_CLI" not in env
        if "install" in command:
            assert command[:5] == [sys.executable, "-I", "-m", "pip", "--python"]
            if command[-1] == "dft-rs-python[ase]==0.3.0":
                assert "--no-deps" not in command
                assert any(cli.parent == Path(command[5]).parent for cli in checked)
                downloaded.append(Path(command[5]))
                return ase_status
            assert "--no-deps" in command
            installs[Path(command[-1])] = Path(command[5])
            return wheel_status if Path(command[-1]) == wheel else 0
        assert "venv" in command and "--without-pip" in command
        return 0

    def check(python, cli, directory, env, version, standalone, expected_hash):
        assert python in installs.values() and cli.parent == python.parent
        assert directory == python.parents[2] and version == "0.3.0"
        assert env["DFT_RS_STRICT_UNITS"] == "1"
        assert standalone == rust_cli
        artifact = next(artifact for artifact, interpreter in installs.items() if interpreter == python)
        assert expected_hash == hashlib.sha256(artifact.read_bytes()).hexdigest()
        checked.append(cli)
        return {"energy": {"energy": -1.0}, "archive": expected_hash}

    def check_ase(python, directory, env, references, expected_hash):
        assert python in downloaded and directory == python.parents[2]
        assert env["DFT_RS_STRICT_UNITS"] == "1"
        assert references == {"energy": {"energy": -1.0}, "archive": expected_hash}
        ase_checked.append(python)

    monkeypatch.setattr(release, "run", run)
    monkeypatch.setattr(release, "check_distribution_sources", lambda *_: "0.3.0")
    monkeypatch.setattr(release, "check_installed_cli", check)
    monkeypatch.setattr(release, "check_installed_ase", check_ase)
    status = release.main(["probe", "--installed-only", "--wheel", str(wheel),
                           "--sdist", str(sdist), "--rust-cli", str(rust_cli)], root=tmp_path)
    assert status == expected
    assert set(installs) == {wheel, sdist} and installs[wheel] != installs[sdist]
    assert len(checked) == len(downloaded) == 2 - bool(wheel_status)
    assert len(ase_checked) == (0 if ase_status else len(checked))
    assert (tmp_path / "validation/probe-sdist-cli.log").read_text(encoding="utf-8") == "EXIT 0\n"
    log = json.loads((tmp_path / "validation/probe-artifacts.json").read_text(encoding="utf-8"))
    assert log["wheel"]["sha256"] != log["sdist"]["sha256"]
    assert log["rust_cli"]["name"] == rust_cli.name


@pytest.mark.parametrize("changed", [None, "source", "missing-doc", "stale-doc"])
def test_distribution_sources_compare_actual_archives(tmp_path, monkeypatch, changed):
    files = {"Cargo.toml": b"frozen manifest", "docs/guide.md": b"# Guide\n"}
    manifest = {
        "version": "0.3.0",
        "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in files.items()],
    }
    monkeypatch.setattr(make_release, "source_manifest", lambda _: manifest)
    sdist = tmp_path / "fixture.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in {**files, "PKG-INFO": b"Name: dft-rs-python\n"}.items():
            if changed == "source" and name == "Cargo.toml":
                data = b"older manifest with the same version"
            member = tarfile.TarInfo("dft_rs_python-0.3.0/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("dft_rs/__init__.py", b"")
        if changed != "missing-doc":
            data = b"# Older guide\n" if changed == "stale-doc" else files["docs/guide.md"]
            archive.writestr("dft_rs/docs/guide.md", data)
    if changed is None:
        assert release.check_distribution_sources(tmp_path, wheel, sdist) == "0.3.0"
    else:
        label = "sdist" if changed == "source" else "wheel documentation"
        with pytest.raises(ValueError, match=label + " differs from frozen source"):
            release.check_distribution_sources(tmp_path, wheel, sdist)


@pytest.mark.parametrize("observed", ["", "1" * 64])
def test_installed_source_archive_must_match_frozen_bytes(tmp_path, monkeypatch, observed):
    def probe(command, **kwargs):
        assert command[:2] == [sys.executable, "-I"] and kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, stdout=observed + "\n")

    monkeypatch.setattr(release.subprocess, "run", probe)
    with pytest.raises(ValueError, match="installed artifact SHA-256 mismatch"):
        release.check_installed_archive(sys.executable, tmp_path, {}, "0" * 64)


@pytest.mark.parametrize("fail", [False, True])
def test_release_python_mode_runs_both_clis_and_propagates_pytest_failure(tmp_path, fail):
    artifacts(tmp_path)
    probe = tmp_path / "test_probe.py"
    probe.write_text(
        "import os\nfrom pathlib import Path\n"
        "def test_staged_environment():\n"
        "    stage = Path(os.environ['PYTHONPATH'])\n"
        "    assert (stage / 'dft_rs/__init__.py').is_file()\n"
        "    assert list((stage / 'dft_rs').glob('dft_rs_core*'))\n"
        "    assert Path(os.environ['DFT_RS_CLI']).is_file()\n"
        "    assert (stage / ('fit_substituent.exe' if os.name == 'nt' else 'fit_substituent')).is_file()\n"
        "    assert os.environ['DFT_RS_STRICT_UNITS'] == '1'\n"
        + ("    assert 'dft_rs_cli' not in Path(os.environ['DFT_RS_CLI']).name\n" if fail else ""),
        encoding="utf-8")
    status = release.main(["probe", "--python-only", "--test-path", str(probe)], root=tmp_path)
    assert status == (1 if fail else 0)
    for cli, expected in (("dft-rs", 0), ("dft_rs_cli", 1 if fail else 0)):
        log = tmp_path / "validation" / f"probe-python-{cli}.log"
        assert log.read_text(encoding="utf-8").endswith(f"EXIT {expected}\n")
    if fail:
        assert "AssertionError" in (tmp_path / "validation/probe-python-dft_rs_cli.log").read_text()


def test_release_missing_artifacts_are_a_failure_before_pytest(tmp_path):
    assert release.main(["missing", "--python-only"], root=tmp_path) == 2
    log = (tmp_path / "validation/missing-python.log").read_text(encoding="utf-8")
    assert "Missing release artifacts" in log and log.endswith("EXIT 2\n")


def test_typed_model_export_uses_the_python_artifact_target_and_retains_records(tmp_path, monkeypatch):
    import json

    producer = tmp_path / "tests/continuous_species_few_center_driver.rs"
    producer.parent.mkdir()
    producer.write_text("// Controlled typed test record producer.\n", encoding="utf-8")
    original = tmp_path / "caller-model.json"
    original.write_text("caller-owned", encoding="utf-8")
    environment = {"DFT_RS_TEST_FEW_CENTER_MODELS": str(original)}
    calls = []

    def export(command, log, root, env):
        calls.append(command)
        Path(env["DFT_RS_TEST_FEW_CENTER_MODELS"]).write_text('{"typed": {"model_version": "probe"}}', encoding="utf-8")
        log.write_text("1 test passed\nEXIT 0\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(release, "run", export)
    target = tmp_path / "target-python"
    status, runtime = release.prepare_python_fixtures(tmp_path, target, tmp_path / "validation", "probe", environment)
    assert status == 0 and environment == {"DFT_RS_TEST_FEW_CENTER_MODELS": str(original)}
    assert original.read_text() == "caller-owned"
    command, = calls
    assert command[command.index("--target-dir") + 1] == str(target)
    assert command[command.index("--features") + 1] == "python"
    assert "--exact" in command and "--offline" in command
    path = Path(runtime["DFT_RS_TEST_FEW_CENTER_MODELS"])
    assert path.is_file() and json.loads(path.read_text())["typed"]["model_version"] == "probe"
    assert (path.parent / "producer.json").is_file()


@pytest.mark.parametrize("failure,output,expected", [(7, None, 7), (0, None, 2), (0, "{}", 2), (0, "broken", 2)])
def test_typed_model_export_fails_closed_and_preserves_the_original_failure(tmp_path, monkeypatch, failure, output, expected):
    producer = tmp_path / "tests/continuous_species_few_center_driver.rs"
    producer.parent.mkdir()
    producer.write_text("// Controlled typed test record producer.\n", encoding="utf-8")

    def export(command, log, root, env):
        log.write_text(f"retained exporter diagnostic\nEXIT {failure}\n", encoding="utf-8")
        if output is not None:
            Path(env["DFT_RS_TEST_FEW_CENTER_MODELS"]).write_text(output, encoding="utf-8")
        return failure

    monkeypatch.setattr(release, "run", export)
    status, runtime = release.prepare_python_fixtures(tmp_path, tmp_path / "target-python", tmp_path / "validation", "probe", {})
    assert status == expected
    assert "retained exporter diagnostic" in (Path(runtime["DFT_RS_TEST_FEW_CENTER_MODELS"]).parent / "export.log").read_text()


@pytest.mark.parametrize("failure", [0, 7])
def test_full_release_exports_typed_models_before_running_python(tmp_path, monkeypatch, failure):
    import json

    artifacts(tmp_path)
    producer = tmp_path / "tests/continuous_species_few_center_driver.rs"
    producer.parent.mkdir()
    producer.write_text("// Controlled typed test record producer.\n", encoding="utf-8")
    monkeypatch.setattr(release, "module_plan", lambda *args: SINGLE_PLAN)
    seen = []

    def execute(command, log, root, env):
        if "--exact" in command:
            Path(env["DFT_RS_TEST_FEW_CENTER_MODELS"]).write_text('{"typed": {}}', encoding="utf-8")
            seen.append("typed-export")
            return failure
        return 0

    def python_suite(root, target, log_dir, name, python, env, tests):
        assert seen == ["typed-export"]
        assert json.loads(Path(env["DFT_RS_TEST_FEW_CENTER_MODELS"]).read_text()) == {"typed": {}}
        assert tests == ["python/tests"]
        seen.append("python")
        return 0

    monkeypatch.setattr(release, "run", execute)
    monkeypatch.setattr(release, "run_python_suite", python_suite)
    assert release.main(["probe"], root=tmp_path) == failure
    assert seen == (["typed-export"] if failure else ["typed-export", "python"])


@pytest.mark.parametrize("status", [1, 7])
def test_release_clippy_failure_stops_before_tests_builds_or_stale_artifacts(tmp_path, monkeypatch, status):
    artifacts(tmp_path)
    calls = []
    diagnostic = "error: file is loaded as a module multiple times: periodic_charge.rs"
    probe = tmp_path / "clippy_probe.py"
    probe.write_text(f"import sys\nprint({diagnostic!r}, file=sys.stderr)\nsys.exit({status})\n",
                     encoding="utf-8")
    real_run = release.run

    def failed_clippy(command, log, root, env):
        calls.append(command)
        assert command[:2] == ["cargo", "clippy"]
        assert "--all-targets" in command and "--no-default-features" in command
        assert "--forbid=clippy::duplicate_mod" in command
        assert "--cap-lints=forbid" in command
        assert command[-2:] == ["-D", "warnings"]
        assert log.name == "failed-clippy-clippy-00.log"
        return real_run([sys.executable, str(probe)], log, root, env)

    monkeypatch.setattr(release, "run", failed_clippy)
    monkeypatch.setattr(release, "module_plan", lambda *args: SINGLE_PLAN)
    monkeypatch.setattr(release, "run_python_suite", lambda *args: pytest.fail("Clippy failure reached pytest"))
    assert release.main(["failed-clippy"], root=tmp_path) == status
    assert len(calls) == 1
    log = (tmp_path / "validation/failed-clippy-clippy-00.log").read_text(encoding="utf-8")
    assert diagnostic in log and log.endswith(f"EXIT {status}\n")


def test_release_build_failure_cannot_validate_stale_artifacts(tmp_path, monkeypatch):
    artifacts(tmp_path)
    calls = []

    def failed_build(command, log, root, env):
        calls.append(command)
        return 7 if command[:2] == ["cargo", "build"] else 0

    monkeypatch.setattr(release, "run", failed_build)
    monkeypatch.setattr(release, "module_plan", lambda *args: SINGLE_PLAN)
    assert release.main(["failed-build"], root=tmp_path) == 7
    assert [command[:2] for command in calls] == [
        ["cargo", "clippy"], ["cargo", "test"], ["cargo", "build"]]


def test_module_plan_failure_propagates_before_any_build_or_test(tmp_path, monkeypatch):
    def fail(*args):
        raise release.ModulePlanError(9, "retained metadata failure")

    monkeypatch.setattr(release, "module_plan", fail)
    monkeypatch.setattr(release, "run", lambda *args: pytest.fail("planning failure reached a build"))
    assert release.main(["bad-plan"], root=tmp_path) == 9
    assert (tmp_path / "validation/bad-plan-module-plan.log").read_text().endswith("EXIT 9\n")


def test_cargo_metadata_exit_status_and_parse_errors_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 7, "", "retained Cargo failure"))
    with pytest.raises(release.ModulePlanError) as failure:
        release.module_plan(tmp_path, {})
    assert failure.value.status == 7 and "retained Cargo failure" in str(failure.value)
    monkeypatch.setattr(release.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, "invalid metadata", ""))
    with pytest.raises(release.ModulePlanError) as failure:
        release.module_plan(tmp_path, {})
    assert failure.value.status == 2


def test_module_feature_plan_is_exhaustive_including_the_default_feature_flag():
    plan = release.module_feature_sets({"default": [], "pyo3": ["dep:pyo3"], "python": ["pyo3"]})
    assert {tuple(c["enabled"]) for c in plan} == {
        (), ("default",), ("pyo3",), ("default", "pyo3"),
        ("pyo3", "python"), ("default", "pyo3", "python")}
    assert len(plan) == 6
    with pytest.raises(ValueError, match="reviewed exhaustive limit"):
        release.module_feature_sets({f"feature_{i}": [] for i in range(9)})


def test_module_environment_preserves_other_flags_and_does_not_change_the_caller():
    original = {"RUSTFLAGS": "-C debuginfo=1 --cap-lints allow --cfg marker",
                "CARGO_ENCODED_RUSTFLAGS": "--cfg\x1fmarker\x1f--cap-lints=warn"}
    checked = release.module_environment(original)
    assert checked["RUSTFLAGS"] == "-C debuginfo=1 --cfg marker"
    assert checked["CARGO_ENCODED_RUSTFLAGS"] == "--cfg\x1fmarker"
    assert "--cap-lints allow" in original["RUSTFLAGS"]
    with pytest.raises(release.ModulePlanError):
        release.module_environment({"RUSTFLAGS": "--cap-lints"})


def test_module_only_matrix_stops_at_the_first_failed_configuration(tmp_path, monkeypatch):
    plan = {"package": "probe", "targets": [], "configurations": [
        {"requested": [], "enabled": []}, {"requested": ["extra"], "enabled": ["extra"]}]}
    monkeypatch.setattr(release, "module_plan", lambda *args: plan)
    monkeypatch.delenv("CARGO_BUILD_JOBS", raising=False)
    calls = []

    def clippy(command, log, root, env):
        assert env["CARGO_BUILD_JOBS"] == "1"
        calls.append(command)
        return 7 if "--features" in command else 0

    monkeypatch.setattr(release, "run", clippy)
    assert release.main(["matrix", "--module-only"], root=tmp_path) == 7
    assert len(calls) == 2 and all(command[:2] == ["cargo", "clippy"] for command in calls)


def test_full_release_cannot_silently_select_a_python_subset(tmp_path):
    with pytest.raises(SystemExit) as error:
        release.main(["subset", "--test-path", "test_probe.py"], root=tmp_path)
    assert error.value.code == 2


def test_release_launcher_help_is_supported_and_packaged():
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--python-only" in result.stdout
    assert '"/tools/verify_release.py"' in (ROOT / "Cargo.toml").read_text()
    assert "validation/run_python_suite.py" not in RUNNER.read_text()


def source_version():
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    return re.search(r'(?ms)^\[package\]\n.*?^version\s*=\s*"([^"]+)"', cargo).group(1)


def test_source_versions_and_dynamic_python_metadata_stay_aligned():
    version = source_version()
    lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    locked = re.search(r'name = "dft-rs"\nversion = "([^"]+)"', lock).group(1)
    assert version == locked
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8").split("[project]\n", 1)[1].split("\n[", 1)[0]
    assert 'name = "dft-rs-python"' in project
    assert 'dynamic = ["version"]' in project
    assert not re.search(r"(?m)^version\s*=", project), "Python must derive its version from Cargo"
    for path in (ROOT / "README.md", ROOT / "docs/rust-api.md"):
        assert all(value == version for value in make_release.documented_dependency_versions(
            path.read_text(encoding="utf-8")))
    assert f"**{version}**" in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("declaration", [
    'dft-rs = "0.3.0"',
    'dft-rs = { version = "0.3.0", path = "../dft-rs" }',
    'dft-rs = { path = "../dft-rs", features = ["python"], version = "0.3.0" }',
])
def test_documented_dependency_update_preserves_local_paths_and_features(declaration):
    before = "```toml\n[dependencies]\n" + declaration + "\n```\n"
    after = make_release.update_documented_dependency_versions(before, "0.3.1")
    assert make_release.documented_dependency_versions(after) == ["0.3.1"]
    assert after == before.replace('"0.3.0"', '"0.3.1"')


def test_documented_checkout_dependency_migration_adds_version_without_registry_fallback():
    before = 'dft-rs = { path = "../dft-rs" }\n'
    with pytest.raises(ValueError, match="explicit version"):
        make_release.documented_dependency_versions(before)
    after = make_release.update_documented_dependency_versions(before, "0.3.0")
    assert after == 'dft-rs = { version = "0.3.0", path = "../dft-rs" }\n'


def test_release_version_update_handles_checkout_documentation(tmp_path, monkeypatch):
    (tmp_path / "Cargo.toml").write_text('[package]\nname = "dft-rs"\nversion = "0.3.0"\n')
    (tmp_path / "Cargo.lock").write_text('[[package]]\nname = "dft-rs"\nversion = "0.3.0"\n')
    (tmp_path / "docs").mkdir()
    declaration = 'dft-rs = { version = "0.3.0", path = "../dft-rs" }\n'
    for name in ("README.md", "docs/rust-api.md"):
        (tmp_path / name).write_text(declaration)
    (tmp_path / "README.md").write_text(
        "Current development milestone: **0.3.0**\n"
        "A registry dependency on version `0.3.0` requires publication.\n" + declaration)
    monkeypatch.setattr(make_release, "root", tmp_path)
    make_release.set_version("0.3.1")
    for name in ("README.md", "docs/rust-api.md"):
        text = (tmp_path / name).read_text()
        assert make_release.documented_dependency_versions(text) == ["0.3.1"]
        assert 'path = "../dft-rs"' in text
    assert 'version = "0.3.1"' in (tmp_path / "Cargo.toml").read_text()
    assert 'version = "0.3.1"' in (tmp_path / "Cargo.lock").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "Current development milestone: **0.3.1**" in readme
    assert "A registry dependency on version `0.3.1` requires publication." in readme


@pytest.mark.parametrize("declaration", [
    'dft-rs = { path = "../dft-rs", version = "0.3.0", version = "0.2.0" }',
    'dft-rs = { path = "../dft-rs", version = 3 }',
])
def test_ambiguous_documented_dependency_versions_fail_closed(declaration):
    with pytest.raises(ValueError):
        make_release.documented_dependency_versions(declaration)
    with pytest.raises(ValueError):
        make_release.update_documented_dependency_versions(declaration, "0.3.1")


def test_installation_docs_use_the_python_distribution_and_local_wheel_route():
    for name in ("README.md", "docs/python-api.md", "docs/cli.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "pip install dft-rs-python" in text
        assert "--no-index --find-links dist dft-rs-python" in text
        assert re.search(r"Once a release is\s+published", text), "index availability must not be implied by local packaging tests"
        assert not re.search(r'pip install ["\']?dft-rs(?:\[|\s|$)', text)


def test_compiled_version_discovery_matches_the_source_being_verified():
    import dft_rs

    expected = source_version()
    actual = dft_rs.native._core.__version__
    assert actual == expected, f"stale compiled extension: {actual}; rebuild this {expected} source before acceptance"
    assert dft_rs.__version__ == actual
    assert dft_rs.feature_registry()["version"] == actual
    cli = os.environ.get("DFT_RS_CLI")
    assert cli, "DFT_RS_CLI is required for compiled version acceptance"
    result = subprocess.run([cli, "--version"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"dft-rs {actual}"


def archive_fixture(tmp_path, monkeypatch, extra=()):
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    names = ["Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md", "LICENSE",
             "THIRD_PARTY_NOTICES.md", "src/lib.rs", "python/dft_rs/__init__.py",
             "docs/guide.md", "third_party/basis.json", "third_party/NOTICE.md"]
    cargo = '[package]\nname = "dft-rs"\nversion = "0.3.0"\nrust-version = "1.85"\ninclude = [\n'
    cargo += "".join(f'    "/{name}",\n' for name in names) + "]\n"
    contents = {name: "Portable public fixture.\n" for name in names}
    contents.update({"Cargo.toml": cargo,
                     "Cargo.lock": '[[package]]\nname = "dft-rs"\nversion = "0.3.0"\n',
                     "pyproject.toml": '[project]\nname = "dft-rs-python"\ndynamic = ["version"]\n',
                     "README.md": "Scientific beta: \u03b2.\n", "src/lib.rs": "// Public fixture.\n",
                     "python/dft_rs/__init__.py": "# Public fixture.\n",
                     "third_party/basis.json": '{"basis": "public synthetic fixture"}\n'})
    for name, text in contents.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (source / ".work").mkdir()
    (source / ".work/cache.dat").write_bytes(b"private scratch fixture")

    def listing(command, **kwargs):
        assert command[1:3] == ["package", "--list"]
        assert "--offline" in command and kwargs["cwd"] == source
        return SimpleNamespace(stdout="\n".join(names + list(extra) + ["Cargo.toml.orig"]), returncode=0)

    monkeypatch.setattr(make_release.subprocess, "run", listing)
    return source, names


def test_source_zip_uses_cargo_inventory_and_mandatory_audit_without_private_runner(tmp_path, monkeypatch):
    source, names = archive_fixture(tmp_path, monkeypatch)
    target = make_release.build_zip(tmp_path / "public.zip", source)
    manifest = make_release.audit_zip(target, source)
    assert manifest["version"] == "0.3.0" and manifest["python_distribution"] == "dft-rs-python"
    with zipfile.ZipFile(target) as archive:
        assert set(archive.namelist()) == {f"dft-rs-0.3.0/{name}" for name in names} | {"dft-rs-0.3.0/SOURCE_MANIFEST.json"}
        assert archive.read("dft-rs-0.3.0/third_party/NOTICE.md") == (source / "third_party/NOTICE.md").read_bytes()
        assert all(".work" not in name for name in archive.namelist())


@pytest.mark.parametrize("name", ["../outside.rs", "/absolute.rs", "Z" + ":/outside.rs",
    ".work/history.json", ".WORK/history.json", "src/__pycache__/code.pyc",
    "third_party/.venv/private.json", "src/target-probe/object.o", "third_party/LOADER.DLL"])
def test_source_zip_rejects_unsafe_private_or_compiled_cargo_members(tmp_path, monkeypatch, name):
    source, _ = archive_fixture(tmp_path, monkeypatch, [name])
    with pytest.raises(ValueError):
        make_release.source_manifest(source)


def test_source_zip_rejects_external_symlinks_before_reading_them(tmp_path, monkeypatch):
    source, _ = archive_fixture(tmp_path, monkeypatch, ["src/external.rs"])
    outside = tmp_path / "outside.rs"
    outside.write_text("External synthetic fixture.", encoding="utf-8")
    try:
        (source / "src/external.rs").symlink_to(outside)
    except OSError:
        pytest.skip("ordinary file symlinks are unavailable on this platform")
    original_read = Path.read_text

    def no_link_read(path, *args, **kwargs):
        assert path != source / "src/external.rs", "source audit followed the link before refusing it"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_link_read)
    with pytest.raises((ValueError, SystemExit), match="symlink"):
        make_release.source_manifest(source)


@pytest.mark.parametrize("field", ["rust-version", "python-name"])
def test_source_zip_requires_consistent_package_identity(tmp_path, monkeypatch, field):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    path = source / ("Cargo.lock" if field == "rust-version" else "pyproject.toml")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("0.3.0", "0.3.1") if field == "rust-version"
                    else text.replace("dft-rs-python", "unrelated-name"), encoding="utf-8")
    with pytest.raises(ValueError):
        make_release.source_manifest(source)


@pytest.mark.parametrize("payload", ["changed payload", chr(0x65E5), "/".join(("", "home", "example-user", "private"))])
def test_source_zip_detects_modified_or_private_payloads_after_packaging(tmp_path, monkeypatch, payload):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    original = make_release.build_zip(tmp_path / "original.zip", source)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(original) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    path = "dft-rs-0.3.0/README.md"
    contents[path] = payload.encode("utf-8")
    if payload != "changed payload":
        key = "dft-rs-0.3.0/SOURCE_MANIFEST.json"
        manifest = json.loads(contents[key])
        row, = [r for r in manifest["files"] if r["path"] == "README.md"]
        row.update(bytes=len(contents[path]), sha256=hashlib.sha256(contents[path]).hexdigest())
        contents[key] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError):
        make_release.audit_zip(damaged, source)


def test_source_zip_and_staging_preserve_existing_user_paths(tmp_path, monkeypatch):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    target = tmp_path / "accepted.zip"
    target.write_bytes(b"accepted artifact")
    with pytest.raises(FileExistsError):
        make_release.build_zip(target, source)
    assert target.read_bytes() == b"accepted artifact"
    existing = tmp_path / "existing-stage"
    existing.mkdir()
    marker = existing / "preserve.txt"
    marker.write_text("Preserve existing content.", encoding="utf-8")
    with pytest.raises(FileExistsError):
        stage_release.stage_release(existing, source)
    assert marker.read_text(encoding="utf-8") == "Preserve existing content."


def test_source_zip_staging_copies_only_the_same_audited_inventory(tmp_path, monkeypatch):
    source, names = archive_fixture(tmp_path, monkeypatch)
    stage = stage_release.stage_release(tmp_path / "fresh-stage", source)
    copied = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert copied == set(names) | {"SOURCE_MANIFEST.json"}
    assert not (stage / ".work").exists()


def test_source_zip_policy_does_not_read_excluded_private_rust_work(tmp_path, monkeypatch):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    private = source / "src/.work/private.rs"
    private.parent.mkdir()
    private.write_text("Synthetic excluded work.\n", encoding="utf-8")
    original_read = Path.read_text

    def no_private_read(path, *args, **kwargs):
        assert path != private, "source audit read excluded private work"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_private_read)
    manifest = make_release.source_manifest(source)
    assert all(".work" not in row["path"] for row in manifest["files"])


def test_source_zip_staging_detects_a_changed_copy_without_replacing_old_stages(tmp_path, monkeypatch):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    original_copy = stage_release.shutil.copy2

    def damaged_copy(origin, destination):
        result = original_copy(origin, destination)
        if Path(destination).name == "README.md":
            Path(destination).write_text("Changed synthetic copy.\n", encoding="utf-8")
        return result

    monkeypatch.setattr(stage_release.shutil, "copy2", damaged_copy)
    with pytest.raises(ValueError, match=r"source changed during staging: README\.md"):
        stage_release.stage_release(tmp_path / "new-incomplete-stage", source)
    assert (tmp_path / "new-incomplete-stage").is_dir()


def test_source_zip_audit_refuses_a_removed_license_even_with_an_updated_manifest(tmp_path, monkeypatch):
    source, _ = archive_fixture(tmp_path, monkeypatch)
    original = make_release.build_zip(tmp_path / "original.zip", source)
    with zipfile.ZipFile(original) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    del contents["dft-rs-0.3.0/LICENSE"]
    key = "dft-rs-0.3.0/SOURCE_MANIFEST.json"
    manifest = json.loads(contents[key])
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "LICENSE"]
    contents[key] = json.dumps(manifest).encode("utf-8")
    damaged = tmp_path / "missing-license.zip"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="licenses"):
        make_release.audit_zip(damaged, source)
