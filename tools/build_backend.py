# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Portable PEP 517/660 builds with the existing maturin backend.

Compiler paths are remapped before building. Generated CycloneDX metadata
keeps its components, identifiers and dependency edges, with local roots
consistently replaced by relative public roots in every JSON string. Wheel
RECORD hashes are regenerated after normalization. Unknown private paths
fail the build; native bytes are audited, never patched.
"""
import base64
from contextlib import contextmanager
import csv
from email.parser import BytesParser
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
from urllib.parse import quote, unquote
import zipfile

from .portable_build import (audit_cargo_attributions, audit_cargo_msrv, audit_source_policy, author_contacts,
                             build_environment, is_text_member, private_member, text_hygiene_findings)

ROOT = Path(__file__).resolve().parents[1]


def distribution_name():
    """The Python distribution identity comes from the existing project table."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = re.search(r"(?ms)^\[project\]\s*\n(.*?)(?=^\[|\Z)", text)
    names = re.findall(r'(?m)^name\s*=\s*"([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"\s*$', project.group(1)) if project else []
    if len(names) != 1:
        raise ValueError("pyproject.toml must declare one Python project name")
    return names[0]


def _distribution_identity(data):
    metadata = BytesParser().parsebytes(data, headersonly=True)
    if metadata.get_all("Name") != [distribution_name()]:
        raise ValueError("distribution metadata Name differs from pyproject.toml")
    versions = metadata.get_all("Version", [])
    if len(versions) != 1 or not re.fullmatch(r"[A-Za-z0-9.!+]+", versions[0]):
        raise ValueError("distribution metadata must contain one normalized Version")
    name = re.sub(r"[-_.]+", "_", distribution_name()).lower()
    return f"{name}-{versions[0]}"


def audit_metadata_directory(path):
    """PEP 517 metadata must expose the same name as the eventual installation."""
    path = Path(path)
    stem = _distribution_identity((path / "METADATA").read_bytes())
    if path.name != stem + ".dist-info":
        raise ValueError("metadata directory has a noncanonical distribution name or version")


def audit_wheel_identity(path):
    """Reject stale names in the wheel filename and its single metadata tree."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("wheel must contain one distribution METADATA")
        stem = _distribution_identity(archive.read(metadata[0]))
        expected = stem + ".dist-info"
        if metadata[0] != expected + "/METADATA":
            raise ValueError("wheel metadata directory has a noncanonical distribution name or version")
        for name in names:
            top = name.split("/", 1)[0]
            if top.endswith(".dist-info") and top != expected:
                raise ValueError("wheel contains a foreign distribution metadata tree")
            if top.endswith(".data") and top != stem + ".data":
                raise ValueError("wheel data directory has a noncanonical distribution name or version")
        parts = path.name.removesuffix(".whl").split("-")
        if path.suffix != ".whl" or len(parts) not in (5, 6) or "-".join(parts[:2]) != stem:
            raise ValueError("wheel filename has a noncanonical distribution name or version")


def audit_sdist_identity(path):
    """Check actual sdist metadata, filename and single archive root together."""
    path = Path(path)
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        metadata = [name for name in names if name.endswith("/PKG-INFO")]
        if len(metadata) != 1:
            raise ValueError("sdist must contain one distribution PKG-INFO")
        stem = _distribution_identity(archive.extractfile(metadata[0]).read())
        if path.name != stem + ".tar.gz" or metadata[0] != stem + "/PKG-INFO":
            raise ValueError("sdist filename or metadata root has a noncanonical distribution name or version")
        if any(name.split("/", 1)[0] != stem for name in names) or stem + "/pyproject.toml" not in names:
            raise ValueError("sdist must have one canonical root containing pyproject.toml")


@contextmanager
def _environment():
    previous = dict(os.environ)
    audit_source_policy(ROOT)
    environment, roots = build_environment(ROOT)
    audit_cargo_msrv(ROOT, environment)
    os.environ.update(environment)
    try:
        yield roots
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _replacements(roots):
    pairs = {}
    for path, public in zip(roots, ("user", "cargo", "rustup", "dft-rs")):
        path = Path(path)
        for private, replacement in [(str(path), public), (path.as_posix(), public),
                                     (path.as_uri(), "file:./" + public)]:
            pairs[private] = replacement
            pairs[quote(private, safe="")] = quote(replacement, safe="")
            pairs[quote(private, safe="/:")] = quote(replacement, safe="/:")
    return sorted(pairs.items(), key=lambda pair: len(pair[0]), reverse=True)


def normalize_sbom(data, roots):
    """Normalize all identifier occurrences together, preserving graph links."""
    replacements = _replacements(roots)

    def visit(value):
        if isinstance(value, str):
            for private, public in replacements:
                if "\\" in private and private in value:
                    # Normalize Windows separators only in the matched path.
                    path = re.escape(private) + r'(?:\\[^\\\s"<>|]+)*'
                    value = re.sub(path, lambda match: public + match[0][len(private):].replace("\\", "/"), value)
                else:
                    value = value.replace(private, public)
            return value
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {visit(key): visit(item) for key, item in value.items()}
        return value

    return (json.dumps(visit(json.loads(data)), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def documentation_files():
    """Collect the public docs tree with archive-relative paths."""
    from .make_release import source_member

    directory = ROOT / "docs"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("the source distribution requires a docs directory")
    documents = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(ROOT).as_posix()
        if private_member(relative):
            continue
        if path.is_symlink():
            raise ValueError("symlinked documentation cannot be distributed: " + relative)
        if path.is_file():
            documents[source_member(relative)] = path.read_bytes()
    if not documents:
        raise ValueError("the source docs directory contains no distributable files")
    return documents


def finalize_wheel(path, roots, editable=False, attributions=None, documentation=None):
    """Normalize generated metadata, audit every member and atomically rehash."""
    path = Path(path)
    tokens = {spelling.encode(encoding) for private, _ in _replacements(roots)
              for spelling in (private,) for encoding in ("utf-8", "utf-16-le")}
    private_path = re.compile(rb"(?i)(?:\b[a-z]:[\\/]|/mnt/[a-z]/|/(?:Users|home)/[^/\\\s\"']+)")
    with zipfile.ZipFile(path) as source:
        infos = source.infolist()
        names = {info.filename for info in infos}
        if len(names) != len(infos):
            raise ValueError("wheel has duplicate archive members")
        additions = {}
        for relative, data in (documentation or {}).items():
            name = "dft_rs/" + relative
            if name in names:
                raise ValueError("wheel already contains documentation member " + name)
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            infos.append(info)
            additions[name] = data
        contents = {}
        records = []
        for info in infos:
            name = info.filename
            if private_member(name):
                raise ValueError("private work or cache member cannot be distributed")
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError("wheel member escapes its archive root")
            if name.endswith(("/RECORD.jws", "/RECORD.p7s")):
                raise ValueError("cannot normalize an already signed wheel")
            if name.endswith(".dist-info/RECORD"):
                records.append(name)
                continue
            data = additions[name] if name in additions else source.read(name)
            if ".dist-info/sboms/" in name and name.endswith(".json"):
                data = normalize_sbom(data, roots)
            # An editable .pth intentionally points at its local checkout;
            # editable wheels are local installation mechanisms, not releases.
            if not (editable and name.endswith(".pth")):
                text_member = is_text_member(name)
                decoded = data
                if text_member:
                    decoded = data.decode("utf-8")
                    while True:
                        unpacked = unquote(decoded)
                        if unpacked == decoded:
                            break
                        decoded = unpacked
                    decoded = decoded.encode("utf-8")
                if any(token in data for token in tokens) or (text_member and private_path.search(decoded)):
                    raise ValueError("private build path remains in wheel member " + name)
                if text_member and text_hygiene_findings(name, data.decode("utf-8"), author_contacts(ROOT), attributions):
                    raise ValueError("text hygiene failed in wheel member " + name)
            contents[name] = data
    if len(records) != 1:
        raise ValueError("wheel must have exactly one RECORD")
    record = records[0]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, data in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((record, "", ""))
    contents[record] = stream.getvalue().encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".whl.tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w") as destination:
            for info in infos:
                destination.writestr(info, contents[info.filename])
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print("Portable wheel audit passed; dependency graph retained and RECORD hashes updated.")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    import maturin
    with _environment() as roots:
        documents = documentation_files()
        name = maturin.build_wheel(wheel_directory, config_settings, metadata_directory)
        audit_wheel_identity(Path(wheel_directory) / name)
        attributions = audit_cargo_attributions(ROOT, dict(os.environ))
        finalize_wheel(Path(wheel_directory) / name, roots,
                       attributions=attributions, documentation=documents)
    return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    import maturin
    with _environment() as roots:
        name = maturin.build_editable(wheel_directory, config_settings, metadata_directory)
        audit_wheel_identity(Path(wheel_directory) / name)
        attributions = audit_cargo_attributions(ROOT, dict(os.environ))
        finalize_wheel(Path(wheel_directory) / name, roots, editable=True, attributions=attributions)
    return name


def build_sdist(sdist_directory, config_settings=None):
    import maturin
    audit_source_policy(ROOT)
    documents = documentation_files()
    name = maturin.build_sdist(sdist_directory, config_settings)
    audit_sdist(Path(sdist_directory) / name, documentation=documents)
    audit_sdist_identity(Path(sdist_directory) / name)
    return name


def audit_sdist(path, documentation=None):
    """Audit the actual source archive without extracting or rewriting it."""
    with tarfile.open(path, "r:*") as archive:
        contacts = author_contacts(ROOT)
        packaged_docs = {}
        for member in archive.getmembers():
            relative = "/".join(member.name.split("/")[1:])
            if member.name.startswith("/") or ".." in Path(member.name).parts or private_member(relative):
                raise ValueError("private or unsafe source archive member")
            if member.issym() or member.islnk():
                raise ValueError("unreviewed source archive link")
            if member.isfile():
                if Path(member.name).suffix.lower() in (".so", ".pyd", ".dll", ".dylib", ".rlib", ".rmeta", ".pdb", ".o", ".pyc"):
                    raise ValueError("private compiled artifact in source archive")
                data = archive.extractfile(member).read()
                if documentation is not None and relative.startswith("docs/"):
                    if relative in packaged_docs:
                        raise ValueError("duplicate source documentation member " + relative)
                    packaged_docs[relative] = data
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if text_hygiene_findings(relative, text, contacts):
                    raise ValueError("text hygiene failed in source archive member")
        if documentation is not None and packaged_docs != documentation:
            mismatches = sorted(name for name in packaged_docs.keys() | documentation.keys()
                                if packaged_docs.get(name) != documentation.get(name))
            raise ValueError("source archive documentation differs: " + ", ".join(mismatches))


def get_requires_for_build_wheel(config_settings=None):
    import maturin
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    import maturin
    return maturin.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    import maturin
    name = maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    audit_metadata_directory(Path(metadata_directory) / name)
    return name


get_requires_for_build_editable = get_requires_for_build_wheel
prepare_metadata_for_build_editable = prepare_metadata_for_build_wheel
