# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832

"""Freeze an exact copy of the crate for release verification.

Usage (WSL, crate root):
    python3 tools/stage_release.py STAGE_DIR

Copies the crate into STAGE_DIR (which must lie outside the crate) without
build directories, logs, staged Python packages or the validation Python
environment. Run tools/verify_release.py with a Python interpreter containing
the project's test dependencies. The full regression,
the Python suite and `tools/make_release.py --zip` then run against exactly
the files that will be archived, while development continues in the crate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

if __package__:
    from .make_release import source_manifest
else:
    from make_release import source_manifest

ROOT = Path(__file__).resolve().parents[1]


def stage_release(destination, source=ROOT):
    """Copy the audited Cargo inventory into a fresh directory, never delete."""
    source = Path(source).resolve()
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("a source stage cannot replace a symlink")
    stage = destination.resolve()
    if stage == source or source in stage.parents:
        raise ValueError("the stage directory must lie outside the crate")
    if stage.exists():
        raise FileExistsError("preserve the existing stage; select a fresh directory")
    manifest = source_manifest(source)
    stage.mkdir(parents=True)
    for row in manifest["files"]:
        target = stage / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / row["path"], target)
        data = target.read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"source changed during staging: {row['path']}")
    (stage / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    options = parser.parse_args()
    stage_release(options.stage)
    print("Staged the audited source inventory without private work or existing-file deletion.")


if __name__ == "__main__":
    main()
