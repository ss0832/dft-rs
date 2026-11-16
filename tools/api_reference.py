# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Keep the existing Markdown Python/ASE API reference complete using ASTs.

No compiled extension or optional package is imported. Signatures, defaults,
descriptions and source links are extracted from the public Python source;
missing descriptions are errors. Use --write after an intentional API change,
and --check for the acceptance check. Existing authored documentation is kept.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- python-public-reference:begin -->"
END = "<!-- python-public-reference:end -->"

# These are applicability families, not a second inventory of calculations.
# Each registry row explicitly selects one; source comments are checked before
# API generation. Runtime propagation remains covered by the adapter/config tests.
SETTINGS_CONTRACTS = {
    "molecular-common": ({"Molecule"}, {"Driver"}),
    "molecular-dedicated": ({"Molecule"}, {"Dedicated"}),
    "periodic-common": ({"Periodic"}, {"Driver"}),
    "periodic-dedicated": ({"Periodic"}, {"Dedicated"}),
    "periodic-operator": ({"Periodic"}, {"Dedicated"}),
    "supplied-data": ({"Data", "Molecule"}, {"Driver"}),
    "conditional": ({"Data", "Molecule", "Periodic"}, {"Driver"}),
    "multicomponent": ({"Molecule"}, {"Driver"}),
    "multiconfiguration": ({"Molecule"}, {"Driver"}),
    "frozen-model": ({"Molecule"}, {"Driver"}),
    "cli": ({"Data"}, {"Dedicated"}),
}


def registry_literal(literal):
    """Decode the registry's quoted strings with Rust continuation whitespace."""
    result = []
    index = 0
    while index < len(literal):
        character = literal[index]
        if character == "\\" and index + 1 < len(literal):
            following = literal[index + 1]
            if following in "\r\n":
                index += 2
                while index < len(literal) and literal[index] in " \t\r\n":
                    index += 1
                continue
            result.extend((character, following))
            index += 2
        else:
            result.append(character)
            index += 1
    return ast.literal_eval("".join(result))


def registry_specs(root):
    """Read public names from the existing Rust registry, including continued strings."""
    source = (root / "src/driver/registry.rs").read_text(encoding="utf-8")
    literal = r'"(?:[^"\\]|\\[\s\S])*"'
    pattern = (rf'feature!\(\s*({literal}),\s*({literal}),\s*(\w+),\s*(Driver|Dedicated),'
               rf'\s*({literal}),\s*({literal}),\s*({literal}),\s*({literal})')
    matches = list(re.finditer(pattern, source))
    if not matches or len(matches) != len(re.findall(r'\bfeature!\s*\(', source)):
        raise ValueError("could not read every feature from the Rust registry")
    specs = [dict(zip(("id", "title", "space", "route", "rust", "cli", "python", "ase"),
                      (value if index in (2, 3) else registry_literal(value)
                       for index, value in enumerate(match.groups())))) for match in matches]
    for match, spec in zip(matches, specs):
        declaration = re.search(r"// settings-contract: ([a-z-]+)\s*$", source[:match.start()])
        if declaration is None:
            raise ValueError(f"{spec['id']} lacks a settings-contract declaration in the registry")
        profile = declaration[1]
        if profile not in SETTINGS_CONTRACTS:
            raise ValueError(f"{spec['id']} has unknown settings-contract {profile}")
        spaces, routes = SETTINGS_CONTRACTS[profile]
        if spec["space"] not in spaces or spec["route"] not in routes:
            raise ValueError(f"{spec['id']} settings-contract {profile} disagrees with space/route")
        spec["settings_contract"] = profile
    for key in ("id", "cli", "python", "ase"):
        names = [spec[key] for spec in specs if spec[key] != "-"]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"registry repeats {key} names: {', '.join(duplicates)}")
    return specs


def exported_names(tree, module):
    """Read the explicit export contract, rejecting missing or ambiguous lists."""
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, (list, tuple)) or not all(isinstance(name, str) for name in value):
            raise ValueError(f"{module}.__all__ must contain literal names")
        if isinstance(node, ast.Assign):
            if names is not None:
                raise ValueError(f"{module}.__all__ is assigned more than once")
            names = list(value)
        elif names is None or not isinstance(node.op, ast.Add):
            raise ValueError(f"{module}.__all__ needs an explicit initial list and += additions")
        else:
            names.extend(value)
    if names is None:
        raise ValueError(f"{module} has no explicit __all__")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{module} repeats exports: {', '.join(duplicates)}")
    return names


def check_exports(root):
    """Resolve exported aliases to their actual documented source declarations."""
    modules = {}
    for module in ("native", "system", "ase_calculator", "units", "__init__"):
        modules[module] = ast.parse((root / "python/dft_rs" / f"{module}.py").read_text(encoding="utf-8"))
    declarations_by_module = {
        module: {name for name, _ in declarations(tree) if "." not in name}
        for module, tree in modules.items()
    }
    for module in ("native", "system"):
        declared = declarations_by_module[module] | {
            target.id for node in modules[module].body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        missing = set(exported_names(modules[module], module)) - declared
        if missing:
            raise ValueError(f"{module} exports undefined names: {', '.join(sorted(missing))}")
    bindings = {}
    for node in ast.walk(modules["__init__"]):
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if node.module is None and alias.name in ("native", "units"):
                bindings[name] = (alias.name, None)
            elif node.module in declarations_by_module:
                if alias.name not in declarations_by_module[node.module]:
                    raise ValueError(f"dft_rs imports undocumented or undefined {node.module}.{alias.name}")
                binding = (node.module, alias.name)
                if name in bindings and bindings[name] != binding:
                    raise ValueError(f"dft_rs has conflicting bindings for {name}")
                bindings[name] = binding
    for name in exported_names(modules["__init__"], "dft_rs"):
        if name not in bindings:
            raise ValueError(f"dft_rs exports unresolved name {name}")


def declarations(tree, prefix=""):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_") and node.name != "__init__":
                continue
            yield prefix + node.name, node
            if isinstance(node, ast.ClassDef):
                yield from declarations(node, prefix + node.name + ".")


def markdown():
    check_exports(ROOT)
    lines = [BEGIN, "## Complete Python and ASE declaration reference", "",
             "This section describes every explicit public function, class, method and",
             "property in `native.py`, `system.py`, the ASE adapter and the IGMH",
             "visualization module, including",
             "signatures and defaults. It is maintained by",
             "`python tools/api_reference.py --write`; `--check` rejects stale or",
             "missing entries and missing source descriptions without importing the core.", "",
             "The [Python and ASE calculation table](#table-3-native-python-and-ase),",
             "[Rust table](rust-api.md#table-1-rust) and [CLI table](cli.md#table-2-cli) describe",
             "the matching Rust/CLI calls, units, physical scope and result conventions.",
             "Dynamic calculator methods come from that same registry and have signature",
             "`calculator.<registered_python_name>(**controls)`. Inherited ASE base-class",
             "methods follow the ASE contract. Private implementation helpers are excluded.", "",
             "The signatures below are the explicit source declarations. Native functions",
             "and explicit DftRs calculation/report getters also accept keyword-only",
             "`cpu_threads`, `memory_mb`, `log_file`, `log_mode` and",
             "`log_level` (all default to `None`); `inspect.signature` exposes the combined",
             "call signature. See [persistent logs](#persistent-calculation-logs) and the",
             "[resource contract](rust-api.md#errors-and-threading) for their shared semantics.", "",
             "The top-level exports in `dft_rs.__all__` are aliases of the declarations",
             "below, plus the documented `native` and `units` modules. Their coverage is",
             "checked explicitly; a new exported module or name requires documentation.", ""]
    counts = {}
    documented = {"native", "units"}
    for filename, module in [("native.py", "dft_rs.native"), ("system.py", "dft_rs.system"),
                             ("ase_calculator.py", "dft_rs.ase_calculator"),
                             ("igmh_visualization.py", "dft_rs.igmh_visualization")]:
        tree = ast.parse((ROOT / "python/dft_rs" / filename).read_text(encoding="utf-8"))
        count = 0
        for name, node in declarations(tree):
            if "." not in name:
                documented.add(name)
            description = ast.get_docstring(node)
            if not description:
                raise ValueError(f"missing API description: {module}.{name}")
            qualified = f"{module}.{name}"
            lines.extend([f"### `{qualified}`", ""])
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(base) for base in node.bases)
                signature = "class " + name + ("(" + bases + ")" if bases else "")
                dataclass_fields = [
                    field for field in node.body
                    if isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
                    and not field.target.id.startswith("_")
                ]
                if any(ast.unparse(dec).startswith("dataclass(") for dec in node.decorator_list):
                    arguments = []
                    for field in dataclass_fields:
                        argument = field.target.id + ": " + ast.unparse(field.annotation)
                        if field.value is not None:
                            argument += " = " + ast.unparse(field.value)
                        arguments.append(argument)
                    signature += "\n" + name + "(" + ", ".join(arguments) + ")"
            else:
                signature = name + "(" + ast.unparse(node.args) + ")"
                if node.returns is not None:
                    signature += " -> " + ast.unparse(node.returns)
                decorators = ["@" + ast.unparse(decorator) for decorator in node.decorator_list
                              if ast.unparse(decorator) in ("property", "classmethod", "staticmethod")]
                signature = "\n".join(decorators + [signature])
            lines.extend(["```python", signature, "```", "",
                          re.sub(r":(?:class|func|mod):`([^`]+)`", r"`\1`", description), "",
                          f"[Source](../python/dft_rs/{filename}#L{node.lineno}).", ""])
            count += 1
        counts[filename] = count
    tree = ast.parse((ROOT / "python/dft_rs/__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__"
                                               for target in node.targets):
            missing = sorted(set(ast.literal_eval(node.value)) - documented)
            if missing:
                raise ValueError("undocumented top-level API exports: " + ", ".join(missing))
    lines.extend(["### Unit conversion constants", "",
                  "`dft_rs.units.X_IN_Y` is the value of one X expressed in Y; native",
                  "calculations retain atomic units. The optional ASE adapter uses the same",
                  "package CODATA 2018 factors at its boundary, independently of ASE defaults.",
                  "The Rust CODATA constants are described",
                  "in [the Rust reference](rust-api.md#errors-and-threading).", "",
                  "| Name | Definition |", "|---|---|"])
    tree = ast.parse((ROOT / "python/dft_rs/units.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                lines.append(f"| `dft_rs.units.{target.id}` | `{ast.unparse(node.value)}` |")
    lines.extend(["", "`dft_rs.native.BOHR_PER_ANGSTROM` equals `dft_rs.units.ANGSTROM_IN_BOHR`;",
                  "`dft_rs.native.HARTREE_IN_EV` equals `dft_rs.units.HARTREE_IN_EV`.", "", END])
    return "\n".join(lines), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    options = parser.parse_args()
    tables = []
    for manual in ("rust-api.md", "cli.md", "python-api.md"):
        content = (ROOT / "docs" / manual).read_text(encoding="utf-8")
        sections = content.split("## Table ")[1:]
        if len(sections) != 1:
            raise SystemExit(f"{manual} needs exactly one aligned calculation table")
        table = re.search(r"(?m)^\| ID \|[^\n]*\n(?:\|[^\n]*\n)+", sections[0])
        if table is None:
            raise SystemExit(f"{manual} has no calculation table")
        tables.append(re.findall(r"(?m)^\| `([^`]+)` \|", table.group()))
    if not tables[0] or len(set(tables[0])) != len(tables[0]) or not all(t == tables[0] for t in tables):
        raise SystemExit("the Rust, CLI and Python/ASE calculation tables differ or repeat an ID")
    if tables[0] != [spec["id"] for spec in registry_specs(ROOT)]:
        raise SystemExit("the API calculation tables differ from the Rust registry")
    generated, counts = markdown()
    path = ROOT / "docs/python-api.md"
    text = path.read_text(encoding="utf-8")
    if BEGIN in text:
        start = text.index(BEGIN)
        stop = text.index(END, start) + len(END)
        expected = text[:start] + generated + text[stop:]
    else:
        expected = text.rstrip() + "\n\n" + generated + "\n"
    if text != expected:
        if not options.write:
            raise SystemExit("public API Markdown is stale; run python tools/api_reference.py --write")
        line_ending = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
        path.write_bytes(expected.replace("\n", line_ending).encode("utf-8"))
    print("Public API Markdown matches source:", counts)


if __name__ == "__main__":
    main()
