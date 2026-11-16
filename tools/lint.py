# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Audit product-source hygiene and run Python correctness checks with Ruff.

Install the development tool with ``python -m pip install '.[lint]'``.
Run from any directory: ``python tools/lint.py [--fix] [PATH ...]``.
Without paths, every Python source, test and maintenance tool is checked.
The explicit configuration also applies to supplied paths outside the crate.
No cache is written into the source tree; findings and missing Ruff fail.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

if __package__:
    from .portable_build import source_hygiene_findings
else:
    from portable_build import source_hygiene_findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Apply Ruff's safe fixes.")
    parser.add_argument("paths", nargs="*", help="Files or directories, relative to the caller.")
    args = parser.parse_args(argv)
    if importlib.util.find_spec("ruff") is None:
        parser.exit(2, "Ruff is required: python -m pip install '.[lint]'\n")
    root = Path(__file__).resolve().parents[1]
    paths = [str(Path(p).resolve()) for p in args.paths] if args.paths else [
        str(root / "python"), str(root / "tools")
    ]
    findings = source_hygiene_findings(root, [Path(p) for p in paths] if args.paths else None)
    if findings:
        for name, line, rule in findings:
            print(f"{name}:{line}: {rule}", file=sys.stderr)
        return 1
    command = [sys.executable, "-m", "ruff", "check", "--no-cache", "--config",
               str(root / "pyproject.toml")]
    if args.fix:
        command.append("--fix")
    return subprocess.run([*command, *paths], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
