# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Verify a frozen crate using Cargo and the existing pytest suite.

Run with a Python interpreter containing the project's test dependencies:
    python tools/verify_release.py NAME

The default checks Rust module ownership on all targets across every declared
root-feature configuration, runs the full Rust regression, builds Python and
both CLIs, then runs the full Python suite against copies of those artifacts.
The module gate explicitly forbids duplicate_mod, independently of warning
defaults. A failed plan or Clippy check stops before builds/tests.
Logs live in validation/ and end with EXIT <status>. No external runner or
private environment directory is required. --python-only reuses an existing
release build; --test-path selects a diagnostic subset only in that mode.
--installed-only checks explicit wheel and sdist artifacts through the
verifier interpreter's pip, using fresh target environments. It checks the
dependency-free console first, then installs the declared ASE extra.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MOLECULAR_FUNCTIONALS = ("wb97x-d", "wb97x-v")


class ModulePlanError(ValueError):
    """Planning failures must stop validation with the original Cargo status."""

    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def module_feature_sets(features):
    """Enumerate distinct root-feature closures from Cargo's own inventory."""
    names = sorted(features)
    if len(names) > 8:
        raise ValueError("module feature inventory exceeds the reviewed exhaustive limit")
    closures = {}
    for mask in range(1 << len(names)):
        requested = [name for i, name in enumerate(names) if mask & (1 << i)]
        enabled = set(requested)
        pending = list(requested)
        while pending:
            for edge in features[pending.pop()]:
                # Cargo handles dependency-owned features and weak edges. Only
                # named root features are used to deduplicate configurations.
                if edge in features and edge not in enabled:
                    enabled.add(edge)
                    pending.append(edge)
                elif edge not in features and not (edge.startswith("dep:") or "/" in edge):
                    raise ValueError("unrecognized Cargo root-feature edge")
        key = tuple(sorted(enabled))
        previous = closures.get(key)
        if previous is None or len(requested) < len(previous):
            closures[key] = requested
    if len(closures) > 32:
        raise ValueError("module feature combinations exceed the reviewed exhaustive limit")
    return [dict(requested=closures[key], enabled=list(key)) for key in sorted(closures)]


def module_plan(root, env):
    """Cargo defines target and feature boundaries; Clippy resolves Rust modules."""
    command = ["cargo", "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"]
    try:
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    except OSError as error:
        raise ModulePlanError(2, str(error)) from error
    if result.returncode:
        raise ModulePlanError(result.returncode, result.stdout + result.stderr)
    try:
        metadata = json.loads(result.stdout)
        packages = [package for package in metadata["packages"]
                    if Path(package["manifest_path"]).resolve() == (root / "Cargo.toml").resolve()]
        if len(packages) != 1:
            raise ValueError("module gate requires one package at this manifest; virtual workspaces need a package-level run")
        package = packages[0]
        return dict(package=package["name"], targets=[dict(name=t["name"], kind=t["kind"])
                    for t in package["targets"]], configurations=module_feature_sets(package["features"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ModulePlanError(2, "Invalid module-check plan: " + str(error)) from error


def module_command(plan, configuration, target_directory, target=None):
    command = ["cargo", "clippy", "--package", plan["package"], "--all-targets", "--no-deps",
               "--no-default-features", "--target-dir", str(target_directory)]
    if configuration["requested"]:
        command += ["--features", ",".join(configuration["requested"])]
    if target:
        command += ["--target", target]
    return command + ["--", "--cap-lints=forbid", "--forbid=clippy::duplicate_mod", "-D", "warnings"]


def module_environment(env):
    """Do not let inherited lint caps bypass the compiler's module analysis."""
    result = dict(env)
    levels = {"allow", "warn", "force-warn", "deny", "forbid"}
    for variable, separator in (("RUSTFLAGS", " "), ("CARGO_ENCODED_RUSTFLAGS", "\x1f")):
        if variable not in result:
            continue
        flags = result[variable].split() if separator == " " else result[variable].split(separator)
        retained = []
        index = 0
        changed = False
        while index < len(flags):
            flag = flags[index]
            if flag == "--cap-lints":
                index += 1
                if index == len(flags) or flags[index] not in levels:
                    raise ModulePlanError(2, "Malformed inherited lint cap")
                changed = True
            elif flag.startswith("--cap-lints="):
                if flag.split("=", 1)[1] not in levels:
                    raise ModulePlanError(2, "Malformed inherited lint cap")
                changed = True
            else:
                retained.append(flag)
            index += 1
        if changed:
            result[variable] = separator.join(retained)
    return result


def run(command, log, root, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        try:
            status = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT).returncode
        except OSError as error:
            stream.write(f"Could not start validation command: {error}\n")
            status = 2
        stream.write(f"EXIT {status}\n")
    return status


def prepare_python_fixtures(root, target, log_dir, name, env):
    """Export typed records with the same Python-feature library being staged."""
    producer = root / "tests/continuous_species_few_center_driver.rs"
    log_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{name}-typed-models-", dir=log_dir))
    model_file = directory / "few-center-models.json"
    runtime = dict(env, DFT_RS_TEST_FEW_CENTER_MODELS=str(model_file))
    command = ["cargo", "test", "--release", "--locked", "--offline", "--features", "python",
               "--target-dir", str(target), "--test", "continuous_species_few_center_driver", "--",
               "pure_record_roundtrips_and_optional_native_fixture_bundle_no_scf", "--exact",
               "--nocapture", "--test-threads=1"]
    log = directory / "export.log"
    if not producer.is_file():
        log.write_text("Typed model exporter source is missing.\nEXIT 2\n", encoding="utf-8")
        return 2, runtime
    status = run(command, log, root, runtime)
    if not status:
        try:
            models = json.loads(model_file.read_text(encoding="utf-8"))
            if not isinstance(models, dict) or not models:
                raise ValueError("typed model bundle must be a nonempty object")
        except (OSError, ValueError) as error:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"Typed model export was rejected: {error}\nEXIT 2\n")
            status = 2
    (directory / "producer.json").write_text(json.dumps({
        "command": command, "exit": status, "model_file": model_file.name,
        "source": str(producer.relative_to(root)),
        "scope": "Typed test records from the selected Python-feature library; no SCF or chemical acceptance",
    }, indent=2) + "\n", encoding="utf-8")
    return status, runtime


def run_python_suite(root, target, log_dir, name, python, env, tests):
    """Stage the built package once, then run pytest with each supported CLI."""
    library = ("dft_rs.dll" if sys.platform == "win32" else
               "libdft_rs.dylib" if sys.platform == "darwin" else "libdft_rs.so")
    suffix = ".exe" if os.name == "nt" else ""
    extension = "dft_rs_core.pyd" if sys.platform == "win32" else "dft_rs_core.abi3.so"
    binaries = [(target / "release" / library, extension)]
    binaries += [(target / "release" / (cli + suffix), cli + suffix)
                 for cli in ("dft-rs", "dft_rs_cli", "fit_substituent")]
    log_dir.mkdir(parents=True, exist_ok=True)
    missing = [str(source) for source, _ in binaries if not source.is_file()]
    if missing:
        (log_dir / f"{name}-python.log").write_text(
            "Missing release artifacts: " + ", ".join(missing) + "\nEXIT 2\n", encoding="utf-8")
        return 2
    with tempfile.TemporaryDirectory(prefix=f"{name}-python-", dir=log_dir) as directory:
        stage = Path(directory)
        package = stage / "dft_rs"
        package.mkdir()
        for source in (root / "python/dft_rs").glob("*.py"):
            shutil.copyfile(source, package / source.name)
        for index, (source, destination) in enumerate(binaries):
            shutil.copy2(source, (package if index == 0 else stage) / destination)
        statuses = []
        for cli in ("dft-rs", "dft_rs_cli"):
            runtime = dict(env, PYTHONPATH=str(stage), DFT_RS_CLI=str(stage / (cli + suffix)))
            status = run([python, "-W", "error", "-m", "pytest", "-q", *tests],
                         log_dir / f"{name}-python-{cli}.log", root, runtime)
            statuses.append(status)
            print(f"{cli}: EXIT {status}", flush=True)
    return next((status for status in statuses if status), 0)


def compare_installed_report(feature, report, reference, route):
    """Compare the physical result and diagnostics with the installed console."""
    if (any(report[key] != reference[key] for key in ("feature", "version", "scf_iterations"))
            or not report["converged"] or report["units"]["energy"] != reference["units"]["energy"]
            or not math.isclose(report["energy"], reference["energy"], rel_tol=0.0, abs_tol=1e-10)):
        raise ValueError(f"installed {feature} {route} energy, units or SCF iterations disagree")
    if feature != "energy":
        residuals = (report["orbital_residual"], reference["orbital_residual"])
        if (any(not 0 <= residual < 1e-9 for residual in residuals)
                or not math.isclose(*residuals, rel_tol=0.0, abs_tol=1e-10)
                or any(item["units"]["orbital_residual"] != "hartree" for item in (report, reference))):
            raise ValueError(f"installed {feature} {route} SCF residuals disagree")
    if feature == "stress":
        tensor, expected = report["stress"], reference["stress"]
        stress_unit = report["units"]["stress"]
        if (not stress_unit.startswith("hartree/bohr^3")
                or stress_unit != reference["units"]["stress"]
                or any(len(matrix) != 3 or any(len(row) != 3 for row in matrix)
                       for matrix in (tensor, expected))
                or any(not math.isclose(tensor[i][j], expected[i][j], rel_tol=0.0, abs_tol=1e-12)
                       for i in range(3) for j in range(3))):
            raise ValueError(f"installed {feature} {route} stress tensors disagree")


def check_installed_archive(python, directory, env, expected_hash):
    """Match pip's recorded local archive against the frozen input."""
    if expected_hash is None:
        return
    probe = subprocess.run(
        [str(python), "-I", "-c", """
import importlib.metadata as md
import json
data = md.distribution('dft-rs-python').read_text('direct_url.json')
archive = json.loads(data or '{}').get('archive_info', {})
digest = archive.get('hashes', {}).get('sha256')
if digest is None:
    algorithm, _, encoded = archive.get('hash', '').partition('=')
    digest = encoded if algorithm == 'sha256' else ''
print(digest)
"""],
        cwd=directory, env=env, check=True, capture_output=True, text=True,
    )
    observed = probe.stdout.strip()
    if observed != expected_hash:
        raise ValueError(f"installed artifact SHA-256 mismatch (expected {expected_hash}, observed {observed!r})")
    print(f"Installed archive SHA-256 {observed}")


def check_installed_cli(python, cli, directory, env, version, rust_cli, expected_hash=None):
    """Check the installed console script with only its native package."""
    check_installed_archive(python, directory, env, expected_hash)
    def invoke(*args, expected=0, executable=cli):
        try:
            proc = subprocess.run(
                [str(executable), *args], cwd=directory, env=env, capture_output=True,
                text=True, encoding="utf-8",
            )
        except OSError as error:
            raise ValueError(f"could not start installed dft-rs {args[0]}: {error}") from error
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode != expected:
            raise ValueError(f"{args[0]} exited with {proc.returncode}, expected {expected}")
        return proc

    metadata = subprocess.run(
        [str(python), "-I", "-c",
         "import importlib.metadata as m; print(m.version('dft-rs-python'))"],
        cwd=directory, env=env, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if metadata != version or invoke("--version").stdout.strip() != f"dft-rs {metadata}":
        raise ValueError("console and installed distribution versions differ")
    if invoke("--version", executable=rust_cli).stdout.strip() != f"dft-rs {metadata}":
        raise ValueError("Rust CLI and installed distribution versions differ")
    registry = json.loads(invoke("feature-registry", "--format", "json").stdout)
    required = {"energy", "periodic-energy", "stress", "constrained-dft", "car-parrinello-md"}
    if registry["version"] != metadata or not required <= {
        entry["id"] for entry in registry["features"]
    }:
        raise ValueError("installed feature registry is incomplete or has a different version")
    geometry = directory / "hydrogen.xyz"
    geometry.write_text("2\nInstalled CLI smoke test\nH 0 0 0\nH 0 0 0.74\n", encoding="ascii")
    native = subprocess.run(
        [str(python), "-I", "-c", """
import json
from dft_rs import run_feature
from dft_rs.units import BOHR_IN_ANGSTROM

edge = 5 / BOHR_IN_ANGSTROM
positions = [[0, 0, 0], [0, 0, 0.74 / BOHR_IN_ANGSTROM]]
cell = [[edge, 0, 0], [0, edge, 0], [0, 0, edge]]
print(json.dumps({
    "energy": run_feature("energy", [1, 1], positions, xc="pbe", grid_level=0),
    "periodic-energy": run_feature(
        "periodic-energy", [1, 1], positions, cell=cell, xc="pbe", overlap_cutoff=5, grid_level=0, conv_tol_grad=1e-10,
    ),
    "stress": run_feature(
        "stress", [1, 1], positions, cell=cell, xc="pbe", overlap_cutoff=5, grid_level=0, conv_tol_grad=1e-10, forces=False,
    ),
}))
"""],
        cwd=directory, env=env, check=True, capture_output=True, text=True,
        encoding="utf-8",
    )
    native_reports = json.loads(native.stdout)
    print(native.stdout, end="")
    reports = {}
    for feature, flags, tag, unit in [
        ("energy", (), "[scf] iter", "hartree"),
        ("periodic-energy", ("--cell", "5", "--overlap-cutoff", "5", "--conv-tol-grad", "1e-10"), "[pbc-scf] iter", "hartree per cell"),
        ("stress", ("--cell", "5", "--overlap-cutoff", "5", "--conv-tol-grad", "1e-10", "--forces", "false"),
         "[pbc-scf] iter", "hartree per cell"),
    ]:
        args = (feature, str(geometry), "--xc", "pbe", "--grid-level", "0", "--format", "json", *flags)
        result = invoke(*args)
        report = json.loads(result.stdout)
        if (report["feature"] != feature or report["version"] != metadata
                or not report["converged"] or not math.isfinite(report["energy"])
                or report["units"]["energy"] != unit):
            raise ValueError(f"invalid installed {feature} report")
        compare_installed_report(feature, native_reports[feature], report, "native Python")
        rust_result = invoke(*args, executable=rust_cli)
        standalone = json.loads(rust_result.stdout)
        compare_installed_report(feature, standalone, report, "Rust CLI")
        if any(not any(tag in line and "E" in line and "residual" in line
                       for line in log.splitlines()) for log in (result.stderr, rust_result.stderr)):
            raise ValueError(f"missing default SCF energy/residual logging for {feature}")
        reports[feature] = report
    invalid = invoke("energy", str(geometry), "--format", "json",
                     "--unknown-install-smoke-control", expected=1)
    if "unknown_install_smoke_control" not in invalid.stderr:
        raise ValueError("invalid control was not diagnosed on stderr")
    check_installed_molecular_methods(python, directory, env, metadata, invoke)
    return reports


def check_installed_ase(python, directory, env, references, expected_hash=None):
    """Compare installed ASE getters with the already checked console reports."""
    check_installed_archive(python, directory, env, expected_hash)
    probe = subprocess.run(
        [str(python), "-I", "-W", "error", "-c", """
import json
from ase import Atoms
from dft_rs import DftRs

positions = [[0, 0, 0], [0, 0, 0.74]]
molecule = Atoms('H2', positions=positions)
crystal = Atoms('H2', positions=positions, cell=[5, 5, 5], pbc=True)
calculator = DftRs(units='atomic')
molecular = dict(xc='pbe', grid_level=0)
periodic = dict(molecular, overlap_cutoff=5, conv_tol_grad=1e-10)
print(json.dumps({
    'energy': calculator.get_energy_report(molecule, **molecular),
    'periodic-energy': calculator.get_periodic_energy_report(crystal, **periodic),
    'stress': calculator.get_periodic_stress_report(crystal, forces=False, **periodic),
}))
"""],
        cwd=directory, env=env, check=True, capture_output=True, text=True, encoding="utf-8",
    )
    reports = json.loads(probe.stdout)
    print(probe.stdout, end="")
    for feature, reference in references.items():
        compare_installed_report(feature, reports[feature], reference, "ASE")
        print(f"installed {feature} ASE agrees with console and native Python", flush=True)


def check_installed_molecular_methods(python, directory, env, version, invoke):
    """Exercise the dispersion and nonlocal correlation paths after each pip install."""
    numbers = [8, 1, 1]
    positions = [[0.06, -0.09, 0.04], [0.89, 0.32, -0.15], [-0.35, 0.72, 0.35]]
    geometry = directory / "water.xyz"
    geometry.write_text(
        "3\nInstalled molecular method validation\n"
        + "".join(f"{symbol} {x} {y} {z}\n"
                  for symbol, (x, y, z) in zip(("O", "H", "H"), positions)),
        encoding="ascii",
    )
    controls = {"reference": "rks", "grid_level": 0, "vv10_grid_level": 0, "grid_weight_cutoff": 0.0}
    jobs = {
        "constrained-dft": {"density_constraints": [{"atoms": [1], "target": 1.04}]},
        "car-parrinello-md": {
            "md_steps": 2, "md_timestep": 1.0, "fictitious_mass": 200.0,
            "nuclear_velocities": [[0.0, 0.0, 0.0], [0.0015, -0.0003, 0.0007], [-0.0004, 0.0011, -0.0003]],
        },
    }
    for xc in REQUIRED_MOLECULAR_FUNCTIONALS:
        for feature, extra in jobs.items():
            settings = dict(controls, xc=xc, **extra)
            argv = []
            for key, value in settings.items():
                argv.extend(("--" + key.replace("_", "-"), value if isinstance(value, str) else json.dumps(value)))
            proc = invoke(feature, str(geometry), "--format", "json", *argv)
            report = json.loads(proc.stdout)
            native = subprocess.run(
                [str(python), "-I", "-c", """
import json
import sys
from dft_rs import run_feature
from dft_rs.units import BOHR_IN_ANGSTROM

feature, numbers, positions, controls = json.loads(sys.argv[1])
bohr = [[coordinate / BOHR_IN_ANGSTROM for coordinate in row] for row in positions]
print(json.dumps(run_feature(feature, numbers, bohr, **controls)))
""", json.dumps([feature, numbers, positions, settings])],
                cwd=directory, env=env, check=True, capture_output=True, text=True, encoding="utf-8",
            )
            reference = json.loads(native.stdout)
            label = f"{feature}/{xc}"
            valid = all(item["version"] == version and item["feature"] == feature
                        and item["converged"] and item["reference_type"] == "rks"
                        for item in (report, reference))
            logged = any("[scf] iter" in line and "E" in line and "residual" in line
                         for line in proc.stderr.splitlines())
            if not valid or not logged:
                raise ValueError(f"invalid installed {label} report or missing default SCF logging")

            def compare(key, unit, *, tolerance=1e-9, index=None):
                left, right = report, reference
                if index is not None:
                    left, right = (item["frames"][index] for item in (left, right))
                unit_key = key if index is None else f"frames.{key}"
                if (any(item["units"][unit_key] != unit for item in (report, reference))
                        or not math.isclose(left[key], right[key], rel_tol=0.0, abs_tol=tolerance)):
                    raise ValueError(f"installed {label} CLI and native Python disagree on {unit_key}")

            compare("scf_orbital_residual", "hartree")
            if not 0 <= report["scf_orbital_residual"] < 1e-6:
                raise ValueError(f"installed {label} has a large orbital residual")
            if feature == "constrained-dft":
                for key in ("energy", "dispersion_energy", "nonlocal_correlation_energy"):
                    compare(key, "hartree")
                if any(len(item[key]) != 1 for item in (report, reference)
                       for key in ("constraint_multipliers", "constraint_residuals")):
                    raise ValueError(f"installed {label} returned an incomplete constraint report")
                multipliers = [item["constraint_multipliers"][0] for item in (report, reference)]
                resolved = all(item["units"]["constraint_multipliers"] == "hartree"
                               and abs(item["constraint_residuals"][0]) <= 1e-9
                               for item in (report, reference))
                required_term = ("dispersion_energy" if xc == "wb97x-d" else "nonlocal_correlation_energy")
                if (not math.isclose(*multipliers, rel_tol=0.0, abs_tol=1e-8)
                        or not resolved or abs(report[required_term]) <= 1e-8):
                    raise ValueError(f"installed {label} did not resolve the constraint with its required XC term")
            else:
                compare("max_extended_energy_drift", "hartree")
                if any(len(item["frames"]) != 3 for item in (report, reference)):
                    raise ValueError(f"installed {label} returned an incomplete trajectory")
                for index in range(3):
                    compare("extended_energy", "hartree", index=index)
                    compare("electronic_kinetic_energy", "hartree", index=index)
                if not (0 <= report["max_extended_energy_drift"] < 2e-5
                        and report["frames"][-1]["electronic_kinetic_energy"] > 0):
                    raise ValueError(f"installed {label} did not propagate a stable trajectory")
            print(f"installed {label} CLI and native Python agree", flush=True)


def record_installed_check(log, check):
    with log.open("w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
        try:
            result = check()
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError,
                tarfile.TarError, zipfile.BadZipFile) as error:
            print(error)
            if isinstance(error, subprocess.CalledProcessError):
                print(error.stderr or "", end="")
            print("EXIT 1")
            return 1, None
        print("EXIT 0")
        return 0, result


def stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    with path.open("rb") as stream:
        return stream_sha256(stream)


def check_distribution_sources(root, wheel, sdist):
    """Match the source archive and wheel documentation to the frozen inventory."""
    try:
        from tools.make_release import source_manifest, source_member
    except ImportError:
        from make_release import source_manifest, source_member

    manifest = source_manifest(root)
    version = manifest["version"]
    expected = {file["path"]: file["sha256"] for file in manifest["files"]}
    docs = {"dft_rs/" + name: digest for name, digest in expected.items()
            if name.startswith("docs/")}
    if not docs:
        raise ValueError("frozen source inventory contains no public documentation")

    def compare(observed, required, label):
        differences = sorted(name for name in observed.keys() | required.keys()
                             if observed.get(name) != required.get(name))
        if differences:
            details = ", ".join(differences[:10])
            raise ValueError(f"{label} differs from frozen source: {len(differences)} paths ({details})")

    prefix = f"dft_rs_python-{version}/"
    observed = {}
    with tarfile.open(sdist, "r:*") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError("unexpected source archive member: " + member.name)
            name = source_member(member.name[len(prefix):])
            if name in observed:
                raise ValueError("duplicate source archive member: " + name)
            with archive.extractfile(member) as stream:
                observed[name] = stream_sha256(stream)
    # The build backend adds package metadata to the Cargo source inventory.
    if observed.pop("PKG-INFO", None) is None:
        raise ValueError("source archive requires PKG-INFO")
    compare(observed, expected, "sdist")

    observed = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            name = member.filename
            if member.is_dir() or not name.startswith("dft_rs/docs/"):
                continue
            if name in observed:
                raise ValueError("duplicate wheel documentation member: " + name)
            with archive.open(member) as stream:
                observed[name] = stream_sha256(stream)
    compare(observed, docs, "wheel documentation")
    print(f"Matched {len(expected)} sdist files and {len(docs)} wheel documentation files.")
    return version


def run_installed_checks(root, log_dir, name, python, env, wheel, sdist, wheelhouse, rust_cli):
    """Install each artifact into a fresh, separate environment."""
    log_dir.mkdir(parents=True, exist_ok=True)
    status, version = record_installed_check(
        log_dir / f"{name}-distribution-sources.log",
        lambda: check_distribution_sources(root, wheel, sdist),
    )
    if status:
        print("Installed validation stopped at the distribution source check.", flush=True)
        return status

    clean = {key: value for key, value in env.items()
             if key not in {"PYTHONHOME", "PYTHONPATH", "DFT_RS_CLI", "VIRTUAL_ENV"}}
    archives = {"wheel": wheel, "sdist": sdist}
    fingerprints = {kind: {"name": path.name, "sha256": file_sha256(path)}
                    for kind, path in archives.items()}
    fingerprints["rust_cli"] = {"name": rust_cli.name, "sha256": file_sha256(rust_cli)}
    (log_dir / f"{name}-artifacts.json").write_text(
        json.dumps(fingerprints, indent=2) + "\n", encoding="utf-8",
    )
    statuses = []
    for kind, artifact in archives.items():
        expected_hash = fingerprints[kind]["sha256"]
        with tempfile.TemporaryDirectory(prefix=f"{name}-{kind}-", dir=log_dir) as directory:
            directory = Path(directory)
            environment = directory / "venv"
            bin_dir = environment / ("Scripts" if os.name == "nt" else "bin")
            suffix = ".exe" if os.name == "nt" else ""
            interpreter = bin_dir / ("python" + suffix)
            cli = bin_dir / ("dft-rs" + suffix)
            command = [python, "-I", "-m", "pip", "--python", str(interpreter), "install"]
            if wheelhouse:
                command.extend(["--no-index", "--find-links", str(wheelhouse)])
            status = run([python, "-I", "-m", "venv", "--without-pip", str(environment)],
                         log_dir / f"{name}-{kind}-venv.log", root, clean)
            if not status:
                status = run([*command, "--no-deps", str(artifact)],
                             log_dir / f"{name}-{kind}-install.log", directory, clean)
            if not status:
                status, references = record_installed_check(
                    log_dir / f"{name}-{kind}-cli.log",
                    lambda: check_installed_cli(
                        interpreter, cli, directory, clean, version, rust_cli, expected_hash,
                    ),
                )
            if not status:
                status = run([*command, f"dft-rs-python[ase]=={version}"],
                             log_dir / f"{name}-{kind}-ase-install.log", directory, clean)
            if not status:
                status, _ = record_installed_check(
                    log_dir / f"{name}-{kind}-ase.log",
                    lambda: check_installed_ase(interpreter, directory, clean, references, expected_hash),
                )
            statuses.append(status)
        print(f"installed {kind}: EXIT {status}", flush=True)
    return next((status for status in statuses if status), 0)


def main(argv=None, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--python-only", action="store_true")
    mode.add_argument("--module-only", action="store_true", help="Check Rust module loading for every declared feature configuration.")
    mode.add_argument("--list-module-checks", action="store_true", help="Print the Cargo-derived module plan without compiling.")
    mode.add_argument("--installed-only", action="store_true", help="Verify pip-installed console scripts from a wheel and sdist.")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--wheelhouse", type=Path, help="Offline pip build and ASE dependencies.")
    parser.add_argument("--rust-cli", type=Path,
                        help="Standalone Rust CLI built from the same frozen source, required with --installed-only.")
    parser.add_argument("--module-target", help="Rust target triple; valid only for module-only checks.")
    parser.add_argument("--target-dir", type=Path, default=Path("target-python"))
    parser.add_argument("--log-dir", type=Path, default=Path("validation"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--test-path", action="append")
    options = parser.parse_args(argv)
    if not options.name or Path(options.name).name != options.name or options.name in (".", ".."):
        parser.error("name must be a single log-name component")
    if options.test_path and not options.python_only:
        parser.error("--test-path requires --python-only; full release validation always runs the full suite")
    if options.module_target and not (options.module_only or options.list_module_checks):
        parser.error("--module-target requires --module-only or --list-module-checks")
    if options.installed_only:
        if not (options.wheel and options.wheel.is_file() and options.wheel.suffix == ".whl"
                and options.sdist and options.sdist.is_file() and options.sdist.name.endswith(".tar.gz")):
            parser.error("--installed-only requires a wheel and sdist file")
        if options.wheelhouse and not options.wheelhouse.is_dir():
            parser.error("--wheelhouse must be a directory")
        if options.rust_cli is None or not options.rust_cli.is_file():
            parser.error("--installed-only requires --rust-cli pointing to a standalone Rust CLI file")
    elif options.wheel or options.sdist or options.wheelhouse or options.rust_cli:
        parser.error("--wheel, --sdist, --wheelhouse and --rust-cli require --installed-only")
    target = (root / options.target_dir).resolve()
    log_dir = (root / options.log_dir).resolve()
    env = dict(os.environ, DFT_RS_STRICT_UNITS="1")
    env.setdefault("RAYON_NUM_THREADS", "1")
    env.setdefault("CARGO_BUILD_JOBS", "1")
    if options.installed_only:
        return run_installed_checks(
            root, log_dir, options.name, options.python, env, options.wheel.resolve(),
            options.sdist.resolve(), options.wheelhouse.resolve() if options.wheelhouse else None,
            options.rust_cli.resolve(),
        )
    statuses = []
    if not options.python_only:
        try:
            module_env = module_environment(env)
            plan = module_plan(root, module_env)
        except ModulePlanError as error:
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{options.name}-module-plan.log").write_text(
                str(error) + f"\nEXIT {error.status}\n", encoding="utf-8")
            print("Rust module validation could not be planned; inspect its log.", flush=True)
            return error.status
        if options.list_module_checks:
            print(json.dumps(plan, indent=2))
            return 0
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{options.name}-module-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        for index, configuration in enumerate(plan["configurations"]):
            command = module_command(plan, configuration, root / "target-release", options.module_target)
            clippy = run(command, log_dir / f"{options.name}-clippy-{index:02d}.log", root, module_env)
            if clippy:
                print("Release validation stopped because the Rust module/Clippy gate failed.", flush=True)
                return clippy
        if options.module_only:
            return 0
        statuses.append(run(["cargo", "test", "--release", "--target-dir", "target-release", "--no-fail-fast"],
                            log_dir / f"{options.name}-rust.log", root, env))
        build = run(["cargo", "build", "--release", "--features", "python", "--lib", "--bins",
                     "--target-dir", str(target)], log_dir / f"{options.name}-python-build.log", root, env)
        statuses.append(build)
        if build:
            print("Python validation was not run because the Python-feature build failed.", flush=True)
            return next(status for status in statuses if status)
        fixtures, env = prepare_python_fixtures(root, target, log_dir, options.name, env)
        statuses.append(fixtures)
        if fixtures:
            print("Python validation was not run because typed fixture export failed.", flush=True)
            return next(status for status in statuses if status)
    statuses.append(run_python_suite(root, target, log_dir, options.name, options.python,
                                     env, options.test_path or ["python/tests"]))
    print("statuses", statuses)
    return next((status for status in statuses if status), 0)


if __name__ == "__main__":
    raise SystemExit(main())
