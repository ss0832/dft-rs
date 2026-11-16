# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Set the package version and build an audited source release archive.

Usage (from the crate root):
    python3 tools/make_release.py --set-version 0.3.0
    python3 tools/make_release.py --zip            # archive the current version
    python3 tools/make_release.py --set-version 0.3.0 --zip

The archive contains exactly the files that `cargo package --list` selects
through the `include` whitelist of Cargo.toml (sources, tests, Python package,
documentation and bundled data). Build directories, validation logs, Python
environments and external inputs are never included. The archive is written
to `../dist/dft-rs-<version>.zip` after mandatory repository-owned source,
identity, membership, hash and privacy audits. No private validation runner
is required. An existing archive is never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import tempfile
import zipfile

if __package__:
    from .portable_build import (audit_source_policy, author_contacts, is_text_member,
                                 private_member, text_hygiene_findings)
else:
    from portable_build import (audit_source_policy, author_contacts, is_text_member,
                                private_member, text_hygiene_findings)

root = Path(__file__).resolve().parents[1]
semver = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?$")


def write_text(path, text):
    # Preserve portable LF files when this helper is run on Windows too.
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


_DOCUMENTED_DEPENDENCY = re.compile(r'(?m)^(dft-rs\s*=\s*)("[^"\n]*"|\{[^}\n]*\})')
_INLINE_VERSION = re.compile(r'\bversion\s*=\s*"([^"\n]+)"')


def documented_dependency_versions(text):
    """Read explicit versions in the supported one-line Cargo documentation examples."""
    versions = []
    for entry in _DOCUMENTED_DEPENDENCY.finditer(text):
        value = entry[2]
        if value.startswith('"'):
            versions.append(value[1:-1])
        else:
            matches = _INLINE_VERSION.findall(value)
            if len(matches) != 1:
                raise ValueError("documented dft-rs inline dependency must have one explicit version")
            versions.append(matches[0])
    if not versions:
        raise ValueError("missing documented dft-rs dependency")
    return versions


def update_documented_dependency_versions(text, version):
    """Update versions without removing checkout paths or declared feature lists."""
    if not semver.fullmatch(version):
        raise ValueError("documentation dependency version must be a semantic version")

    def replace(entry):
        value = entry[2]
        if value.startswith('"'):
            return entry[1] + '"' + version + '"'
        count = len(_INLINE_VERSION.findall(value))
        if count > 1 or count == 0 and re.search(r'\bversion\s*=', value):
            raise ValueError("ambiguous documented dft-rs dependency version")
        if count:
            value = _INLINE_VERSION.sub(lambda _: 'version = "' + version + '"', value)
        elif re.search(r'\bpath\s*=\s*"', value):
            value = '{ version = "' + version + '",' + value[1:]
        else:
            raise ValueError("a documented inline dependency needs a version or checkout path")
        return entry[1] + value

    updated, count = _DOCUMENTED_DEPENDENCY.subn(replace, text)
    if not count:
        raise ValueError("missing documented dft-rs dependency")
    return updated


def set_version(version):
    if not semver.match(version):
        raise SystemExit(f"{version!r} is not a semantic version")
    toml = root / "Cargo.toml"
    text = toml.read_text(encoding="utf-8")
    text, count = re.subn(r'(?m)^(\[package\]\nname = "dft-rs"\nversion = )"[^"]+"', rf'\1"{version}"', text)
    if count != 1:
        raise SystemExit("could not locate the [package] version in Cargo.toml")
    write_text(toml, text)
    lock = root / "Cargo.lock"
    text = lock.read_text(encoding="utf-8")
    text, count = re.subn(r'(name = "dft-rs"\nversion = )"[^"]+"', rf'\1"{version}"', text)
    if count != 1:
        raise SystemExit("could not locate the dft-rs entry in Cargo.lock")
    write_text(lock, text)
    for doc in ("README.md", "docs/rust-api.md"):
        path = root / doc
        text = path.read_text(encoding="utf-8")
        text = update_documented_dependency_versions(text, version)
        if doc == "README.md":
            text = re.sub(r'(?m)^(Current development milestone: \*\*)[^*\r\n]+(\*\*)',
                          lambda match: match[1] + version + match[2], text)
            text = re.sub(r'(A registry dependency on version `)[^`]+(` requires)',
                          lambda match: match[1] + version + match[2], text)
        write_text(path, text)
    print(f"version set to {version}")


def current_version(source=None):
    text = (Path(source or root) / "Cargo.toml").read_text(encoding="utf-8")
    return re.search(r'(?m)^version\s*=\s*"([^"]+)"', text).group(1)


_COMPILED_SUFFIXES = {".so", ".pyd", ".dll", ".dylib", ".rlib", ".rmeta", ".o",
                      ".a", ".lib", ".exe", ".pdb", ".pyc", ".exp"}
_GENERATED_CARGO_MEMBERS = {"Cargo.toml.orig", ".cargo_vcs_info.json"}
_REQUIRED_SOURCE_MEMBERS = {"Cargo.toml", "Cargo.lock", "pyproject.toml", "README.md",
                            "LICENSE", "THIRD_PARTY_NOTICES.md"}


def source_member(name):
    """Validate a relative source member before accessing any filesystem path."""
    name = str(name).replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or re.match(r"^[A-Za-z]:", name):
        raise ValueError("unsafe source archive path")
    if private_member(name.casefold()) or path.suffix.lower() in _COMPILED_SUFFIXES:
        raise ValueError("private work or compiled artifact in source archive")
    if text_hygiene_findings("SOURCE_ARCHIVE_PATH", name, set()):
        raise ValueError("private or nonportable source archive filename")
    return path.as_posix()


def package_identity(cargo, lock, project):
    """Check actual Rust and Python metadata without importing the extension."""
    package = re.search(r"(?ms)^\[package\]\s*\n(.*?)(?=^\[|\Z)", cargo)
    block = package.group(1) if package else ""
    names = re.findall(r'(?m)^name\s*=\s*"([^"]+)"', block)
    versions = re.findall(r'(?m)^version\s*=\s*"([^"]+)"', block)
    if names != ["dft-rs"] or len(versions) != 1 or not semver.fullmatch(versions[0]):
        raise ValueError("source archive requires the versioned dft-rs Rust package")
    if re.findall(r'(?m)^name = "dft-rs"\r?\nversion = "([^"]+)"', lock) != versions:
        raise ValueError("Rust manifest and lockfile versions differ")
    table = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", project)
    names = re.findall(r'(?m)^name\s*=\s*"([^"]+)"', table.group(1)) if table else []
    if names != ["dft-rs-python"]:
        raise ValueError("Python distribution must be dft-rs-python")
    return versions[0]


def source_manifest(source=None):
    """Use Cargo's existing inventory and shared hygiene policy, not a new list."""
    source = Path(source or root).resolve()
    audit_source_policy(source)
    version = package_identity(*[(source / name).read_text(encoding="utf-8")
                                 for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")])
    listing = subprocess.run(
        [os.environ.get("CARGO", "cargo"), "package", "--list", "--locked", "--offline", "--allow-dirty"],
        cwd=source, capture_output=True, text=True, check=True,
    )
    files = []
    seen = set()
    for name in listing.stdout.splitlines():
        if not name.strip() or name.strip() in _GENERATED_CARGO_MEMBERS:
            continue
        name = source_member(name.strip())
        if name in seen:
            raise ValueError("duplicate Cargo source member")
        seen.add(name)
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError("missing or external/symlinked source member")
        data = path.read_bytes()
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    if not _REQUIRED_SOURCE_MEMBERS.issubset(seen):
        raise ValueError("source inventory omits required package metadata or licenses")
    return dict(schema="dft-rs-source-manifest-v1", version=version, rust_package="dft-rs",
                python_distribution="dft-rs-python", python_import="dft_rs",
                files=sorted(files, key=lambda row: row["path"]))


def audit_zip(path, source=None, expected=None):
    """Audit the actual archive independently, preserving legal attribution."""
    source = Path(source or root).resolve()
    contacts = author_contacts(source)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        manifest_names = [name for name in names if name.endswith("/SOURCE_MANIFEST.json")]
        if len(manifest_names) != 1:
            raise ValueError("ZIP requires one source manifest")
        manifest_text = archive.read(manifest_names[0]).decode("utf-8")
        if text_hygiene_findings("SOURCE_MANIFEST.json", manifest_text, contacts):
            raise ValueError("source manifest privacy policy failed")
        manifest = json.loads(manifest_text)
        if expected is not None and manifest != expected:
            raise ValueError("ZIP manifest differs from the frozen source inventory")
        if (manifest.get("schema") != "dft-rs-source-manifest-v1"
                or manifest.get("rust_package") != "dft-rs"
                or manifest.get("python_distribution") != "dft-rs-python"
                or manifest.get("version") != current_version(source)):
            raise ValueError("ZIP identity differs from the source package")
        top = f"dft-rs-{manifest['version']}"
        rows = manifest["files"]
        paths = [source_member(row["path"]) for row in rows]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate source-manifest member")
        if not _REQUIRED_SOURCE_MEMBERS.issubset(paths):
            raise ValueError("source ZIP omits required package metadata or licenses")
        if set(names) != {f"{top}/{name}" for name in paths} | {f"{top}/SOURCE_MANIFEST.json"}:
            raise ValueError("ZIP contents differ from the declared inventory")
        version = package_identity(*[archive.read(f"{top}/{name}").decode("utf-8")
                                     for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml")])
        if version != manifest["version"]:
            raise ValueError("ZIP payload versions differ from the manifest")
        for info in infos:
            if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1:
                raise ValueError("source ZIP cannot contain symlinks or encrypted members")
        if archive.testzip() is not None:
            raise ValueError("source ZIP CRC check failed")
        for row, name in zip(rows, paths):
            data = archive.read(f"{top}/{name}")
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("source ZIP content hash differs from the manifest")
            if is_text_member(name):
                if text_hygiene_findings(name, data.decode("utf-8"), contacts):
                    raise ValueError("source ZIP text privacy/English policy failed")
    return manifest


def build_zip(destination=None, source=None):
    source = Path(source or root).resolve()
    manifest = source_manifest(source)
    version = manifest["version"]
    top = f"dft-rs-{version}"
    target = Path(destination) if destination else source.parent / "dist" / f"{top}.zip"
    if target.exists():
        raise FileExistsError("preserve the existing source archive; select a new output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".dft-rs-source-", suffix=".zip", dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for row in manifest["files"]:
                archive.write(source / row["path"], f"{top}/{row['path']}")
            archive.writestr(f"{top}/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        audit_zip(temporary, source, manifest)
        # Both operations publish the audited file without replacing a
        # destination created by another writer.
        if os.name == "nt":
            os.rename(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote audited source ZIP ({len(manifest['files'])} source files, {target.stat().st_size} bytes)")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-version")
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--output", type=Path, help="explicit ZIP output; existing files are preserved")
    args = parser.parse_args()
    if args.set_version:
        set_version(args.set_version)
    if args.zip:
        build_zip(args.output)
    if not (args.set_version or args.zip):
        parser.print_help()


if __name__ == "__main__":
    main()
