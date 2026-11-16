# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Run Cargo with private build roots remapped, then audit release artifacts.

Usage from any directory, with a Python 3 interpreter and the Rust toolchain:
    python tools/portable_build.py build --release --locked --lib --bins
    python tools/portable_build.py build --release --locked --features python

Only the standard library is required. Existing compiler flags are preserved.
The crate, Cargo, Rustup and home roots are derived on the current computer;
none is stored in source or printed by this script. Native binaries still
target the selected operating system and architecture, as with ordinary Cargo.
"""
import argparse
import fnmatch
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tokenize
from urllib.parse import unquote


_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]")
_PRIVATE_PATH = re.compile(r"(?i)(?:\b[a-z]:[\\/]|/mnt/[a-z]/|/(?:home|Users)/[^/\\\s\"']+)")
_CONTACT = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b(?![\w.]|\()")
_THREAD_ID = re.compile(r"(?i)\b01[a-f0-9]{6}-[a-f0-9]{4}-7[a-f0-9]{3}-[a-f0-9]{4}-[a-f0-9]{12}\b|\bSentinel_[a-f0-9]+\b")
_HOST = re.compile(r"(?i)\b(?:DESKTOP|LAPTOP)-[A-Z0-9]+\b")
_PRIVATE_DIRECTORIES = {".work", ".git", ".codex", ".agents", ".venv", ".pytest_cache",
                        ".ruff_cache", "__pycache__", "validation", "runs", "d4si"}


def is_text_member(name):
    """Share source/archive text coverage, including extensionless notices."""
    path = PurePosixPath(str(name).replace("\\", "/"))
    return (path.suffix.lower() in (".rs", ".py", ".md", ".csv", ".toml", ".json", ".txt",
            ".yaml", ".yml", ".cfg", ".ini", ".lock", ".sh", ".bat", ".cmd", ".ps1",
            ".rst", ".xyz", ".inp", ".svg", ".pth")
            or path.name in (".gitignore", ".gitattributes", "LICENSE", "COPYING", "NOTICE",
                             "COPYRIGHT", "Makefile", "METADATA", "WHEEL", "RECORD", "PKG-INFO"))


def private_member(name):
    """Private work evidence and build caches are never package members."""
    if PurePosixPath(str(name).replace("\\", "/")).suffix.lower() == ".log":
        return True
    return any(part.casefold() in _PRIVATE_DIRECTORIES or part.casefold() == "target"
               or part.casefold().startswith(("target-", "staging-"))
               for part in PurePosixPath(str(name).replace("\\", "/")).parts)


def author_contacts(root):
    """Only explicitly declared package authors have a contact exception."""
    cargo = root / "Cargo.toml"
    if not cargo.is_file():
        return set()
    text = cargo.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^authors\s*=\s*\[.*?\]", text)
    return set(_CONTACT.findall(match.group())) if match else set()


def attribution_contacts(metadata):
    """Published dependency authors are provenance, not arbitrary contact fields."""
    return {package["name"] + "@" + package["version"]:
            {contact for author in package.get("authors", []) for contact in _CONTACT.findall(author)}
            for package in (metadata or {}).get("packages", [])}


def sbom_contacts_valid(text, contacts, attributions):
    """Allow only declared authors at standard CycloneDX author locations."""
    try:
        data = json.loads(text)
    except ValueError:
        return False
    if not isinstance(data, dict) or data.get("bomFormat") != "CycloneDX":
        return False
    metadata = data.get("metadata", {})
    components = data.get("components", [])
    if not isinstance(metadata, dict) or not isinstance(components, list):
        return False
    contexts = {("metadata", "authors"): set(contacts)}
    for prefix, component in [(("metadata", "component"), metadata.get("component", {}))] + [
            (("components", i), value) for i, value in enumerate(components)]:
        if not isinstance(component, dict):
            continue
        key = str(component.get("name", "")) + "@" + str(component.get("version", ""))
        purl = component.get("purl", "")
        approved = set((attributions or {}).get(key, ()))
        if not isinstance(purl, str) or purl.split("?", 1)[0] != "pkg:cargo/" + key:
            approved = set()
        if prefix == ("metadata", "component"):
            approved &= set(contacts)
        for field in ("author", "authors"):
            contexts[prefix + (field,)] = approved

    def visit(value, path=(), approved=()):
        approved = contexts.get(path, approved)
        if isinstance(value, dict):
            return all(visit(child, path + (key,), approved) for key, child in value.items())
        if isinstance(value, list):
            return all(visit(child, path + (i,), approved) for i, child in enumerate(value))
        if isinstance(value, str):
            return all(contact in approved or contact.rsplit("@", 1)[1].lower() in
                       ("example.com", "example.org", "example.net", "example.invalid")
                       for contact in _CONTACT.findall(unquote(value)))
        return True

    return visit(data)


def text_hygiene_findings(name, text, contacts=(), attributions=None):
    """Return location/rule only; diagnostics must not repeat leaked values."""
    name = Path(name).as_posix()
    legal_names = ("LICENSE", "COPYING", "NOTICE", "NOTICE.md", "THIRD_PARTY_NOTICES.md")
    legal = (name in ("LICENSE", "THIRD_PARTY_NOTICES.md")
             or ((name.startswith("third_party/") or ".dist-info/licenses/" in name)
                 and Path(name).name in legal_names))
    findings = set()
    lines = text.splitlines()
    author_lines = set()
    if name in ("Cargo.toml", "Cargo.toml.orig", "pyproject.toml"):
        for field in re.finditer(r"(?ms)^authors\s*=\s*\[.*?\]", text):
            first = text.count("\n", 0, field.start()) + 1
            author_lines.update(range(first, first + field.group().count("\n") + 1))
    # Matrix multiplication such as a@b.conj() is code, not a contact address.
    contact_lines = {}
    if name.endswith(".py"):
        try:
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type in (tokenize.STRING, tokenize.COMMENT):
                    for offset, value in enumerate(token.string.splitlines()):
                        contact_lines.setdefault(token.start[0] + offset, []).append(value)
        except (tokenize.TokenError, IndentationError):
            # Syntax is diagnosed separately by Ruff; privacy still fails closed.
            contact_lines = {i: [line] for i, line in enumerate(lines, 1)}
    else:
        contact_lines = {i: [line] for i, line in enumerate(lines, 1)}
    if ".dist-info/sboms/" in name and name.endswith(".json"):
        if _CONTACT.search(text) and not sbom_contacts_valid(text, contacts, attributions):
            findings.add((name, 1, "personal-contact"))
        contact_lines = {}
    for number, line in enumerate(lines, 1):
        decoded = line
        while unquote(decoded) != decoded:
            decoded = unquote(decoded)
        for pattern, rule in ((_PRIVATE_PATH, "private-path"), (_THREAD_ID, "conversation-id"),
                              (_HOST, "private-host")):
            if pattern.search(decoded):
                findings.add((name, number, rule))
        # Greek letters, mathematical symbols and accented author names are valid.
        bibliography = (name == "docs/references.md" and "doi.org/" in line
                        and re.search(r"\b(?:19|20)\d{2}\b", line))
        if not legal and not bibliography and _JAPANESE.search(line):
            findings.add((name, number, "Japanese-text"))
        for value in contact_lines.get(number, []):
            if "@" not in value:
                continue
            for contact in _CONTACT.findall(unquote(value)):
                domain = contact.rsplit("@", 1)[1].lower()
                placeholder = domain in ("example.com", "example.org", "example.net", "example.invalid")
                metadata = (number in author_lines
                            or ((name.endswith(".dist-info/METADATA") or name == "PKG-INFO")
                                and line.startswith(("Author:", "Author-email:"))))
                if not (legal or placeholder or (metadata and contact in contacts)):
                    findings.add((name, number, "personal-contact"))
    return sorted(findings)


def source_hygiene_findings(root, paths=None):
    """Inspect distributable source, tests, tools, examples and public documents."""
    root = Path(root).resolve()
    contacts = author_contacts(root)
    private_trees = []
    if paths is None:
        # Cargo's include inventory defines the distribution scope, not a second catalogue.
        cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
        block = re.search(r"(?ms)^include\s*=\s*\[(.*?)^\]", cargo)
        if block is None:
            raise ValueError("Cargo must declare its distributable include inventory")
        patterns = re.findall(r'"([^"\n]+)"', block.group(1))

        def included(name):
            selected = False
            for pattern in patterns:
                if fnmatch.fnmatchcase(name, pattern.lstrip("!/")):
                    selected = not pattern.startswith("!")
            return selected

        paths = []
        for directory, directories, names in os.walk(root):
            for name in directories:
                child = Path(directory) / name
                relative = child.relative_to(root).as_posix()
                if child.is_symlink() and not private_member(name.casefold()):
                    private_trees.append((relative, 1, "unreviewed-source-symlink"))
                if private_member(name) and included(relative + "/__member__"):
                    private_trees.append((relative, 1, "private-work-in-package"))
            directories[:] = [name for name in directories
                              if not private_member(name.casefold())
                              and not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if included(relative) or is_text_member(relative):
                    paths.append(path)
    findings = list(private_trees)
    for path in paths:
        path = Path(path)
        if path.is_symlink():
            findings.append((path.name, 1, "unreviewed-source-symlink"))
        elif path.is_dir():
            findings.extend(source_hygiene_findings(root, (p for p in path.rglob("*") if p.is_file())))
        elif path.is_file() and not private_member(path.relative_to(root) if path.is_relative_to(root) else path.name):
            name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            findings.extend(text_hygiene_findings(name, text, contacts))
    return findings


def build_environment(root):
    env = dict(os.environ)
    home = Path.home().resolve()
    roots = [
        (home, "user"),
        (Path(env.get("CARGO_HOME", home / ".cargo")).resolve(), "cargo"),
        (Path(env.get("RUSTUP_HOME", home / ".rustup")).resolve(), "rustup"),
        (root.resolve(), "dft-rs"),
    ]
    flags = (env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
             if "CARGO_ENCODED_RUSTFLAGS" in env else env.get("RUSTFLAGS", "").split())
    # Rust uses the last matching prefix. Put the most specific roots last.
    generated = [f"--remap-path-prefix={path}={public}" for path, public in roots]
    # PEP 517 and Cargo wrappers may resolve the same environment in sequence.
    # Keep one final copy of our identical maps: nesting must not change the
    # fingerprint and trigger an otherwise unnecessary whole-library rebuild.
    # Different user maps and all unrelated flags remain in their old order;
    # the generated most-specific map retains its previous final precedence.
    flags = [flag for flag in flags if flag not in generated] + generated
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    return env, [path for path, _ in roots]


def audit_source_policy(root):
    # Reject links before any diagnostic-source read can follow them, and
    # retain the same private-tree exclusions during the Rust-only pass.
    root = Path(root).resolve()
    findings = source_hygiene_findings(root)
    if findings:
        raise SystemExit("source hygiene failed:\n" + "\n".join(
            f"{name}:{line}: {rule}" for name, line, rule in findings))
    diagnostic_suppression = re.compile(r"#\s*!?\[\s*(?:allow|expect|cfg_attr)\b[^\]]*\bdead_code\b", re.S)
    for directory in (root / "src", root / "tests"):
        for base, directories, names in os.walk(directory):
            directories[:] = [name for name in directories if not private_member(name.casefold())]
            for name in names:
                path = Path(base) / name
                if path.suffix != ".rs":
                    continue
                if path.is_symlink():
                    raise SystemExit("unreviewed Rust source symlink")
                if diagnostic_suppression.search(path.read_text(encoding="utf-8")):
                    raise SystemExit(f"unused-code diagnostic suppression is forbidden: {path.relative_to(root)}")


def validate_rust_requirements(metadata):
    """Check declared requirements on resolved non-development dependency edges."""
    packages = {package["id"]: package for package in metadata["packages"]}
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]

    def version(text):
        match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
        if match is None:
            raise ValueError("Cargo packages must declare a stable rust-version for compatibility checking")
        return tuple(int(part or 0) for part in match.groups())

    declared = packages[root].get("rust_version")
    floor = version(declared)
    seen = set()
    pending = [root]
    findings = []
    while pending:
        identifier = pending.pop()
        if identifier in seen:
            continue
        seen.add(identifier)
        package = packages[identifier]
        required = package.get("rust_version")
        if required and version(required) > floor:
            findings.append(f"{package['name']} {package['version']} requires Rust {required}")
        for dependency in nodes[identifier]["deps"]:
            if any(kind["kind"] != "dev" for kind in dependency["dep_kinds"]):
                pending.append(dependency["pkg"])
    if findings:
        raise ValueError(f"declared rust-version {declared} is below selected dependency requirements: "
                         + "; ".join(sorted(findings)))
    return declared


def audit_cargo_msrv(root, env=None, arguments=()):
    """Audit the selected platform using Cargo metadata; do not compile or infer an MSRV."""
    env = dict(os.environ) if env is None else env
    target = env.get("CARGO_BUILD_TARGET")
    for index, argument in enumerate(arguments):
        if argument == "--target" and index + 1 < len(arguments):
            target = arguments[index + 1]
        elif argument.startswith("--target="):
            target = argument.split("=", 1)[1]
    if target is None:
        compiler = subprocess.run([env.get("RUSTC", "rustc"), "-vV"], cwd=root, env=env,
                                  capture_output=True, text=True, check=True)
        target = next(line.split(":", 1)[1].strip() for line in compiler.stdout.splitlines()
                      if line.startswith("host:"))
    command = [env.get("CARGO", "cargo"), "metadata", "--locked", "--format-version", "1",
               "--all-features", "--filter-platform", target]
    if "--offline" in arguments:
        command.append("--offline")
    process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
    metadata = json.loads(process.stdout)
    declared = validate_rust_requirements(metadata)
    print(f"Cargo declared Rust-version audit passed: {declared} covers selected dependency requirements.")
    return metadata


def audit_cargo_attributions(root, env=None):
    """Resolve author provenance for all SBOM targets, independently of MSRV.

    The selected-target compatibility gate remains separate. This query does
    not compile or claim support for the other targets in a dependency SBOM.
    """
    env = dict(os.environ) if env is None else env
    command = [env.get("CARGO", "cargo"), "metadata", "--locked", "--offline",
               "--format-version", "1", "--all-features"]
    process = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=True)
    return attribution_contacts(json.loads(process.stdout))


def target_directory(root, arguments, env):
    target = Path(env.get("CARGO_TARGET_DIR", "target"))
    triple = env.get("CARGO_BUILD_TARGET")
    profile = "release" if "--release" in arguments else "debug"
    for i, arg in enumerate(arguments):
        for name in ("--target-dir", "--target", "--profile"):
            value = (arguments[i+1] if arg == name and i+1 < len(arguments)
                     else arg[len(name)+1:] if arg.startswith(name+"=") else None)
            if value is not None:
                if name == "--target-dir": target = Path(value)
                elif name == "--target": triple = value
                else: profile = {"dev":"debug", "test":"debug", "bench":"release"}.get(value,value)
    target = target if target.is_absolute() else root / target
    if triple:
        target = target / Path(triple).stem
    return target / profile


def audit_artifacts(directory, roots):
    tokens = set()
    for root in roots:
        for spelling in (str(root), root.as_posix()):
            tokens.add(spelling.encode("utf-8"))
            tokens.add(spelling.encode("utf-16-le"))
    names = ("dft-rs", "dft_rs_cli", "fit_substituent", "libdft_rs.so", "libdft_rs.dylib",
             "dft-rs.exe", "dft_rs_cli.exe", "fit_substituent.exe", "dft_rs.dll",
             "dft_rs.pdb", "dft_rs_cli.pdb", "fit_substituent.pdb")
    checked = 0
    for name in names:
        path = directory / name
        if not path.is_file():
            continue
        checked += 1
        data = path.read_bytes()
        if any(token in data for token in tokens):
            raise SystemExit(f"private build root remains in {name}; artifact cannot be distributed")
    if checked == 0:
        raise SystemExit("no native artifacts found to audit in the selected target/profile")
    print(f"Portable artifact audit passed: {checked} files, no private build roots.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cargo_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.cargo_arguments or ["build", "--release", "--locked", "--lib", "--bins"]
    if arguments[0] not in ("build", "test", "check"):
        parser.error("the Cargo command must be build, test or check")
    root = Path(__file__).resolve().parents[1]
    audit_source_policy(root)
    env, roots = build_environment(root)
    audit_cargo_msrv(root, env, arguments)
    status = subprocess.run(["cargo", *arguments], cwd=root, env=env).returncode
    if status:
        return status
    if arguments[0] == "build":
        audit_artifacts(target_directory(root, arguments, env), roots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
