# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""The pip console entry shares the compiled CLI and its process contract."""
import json
import sys

import dft_rs


def test_console_entry_reads_process_arguments_without_the_program_name(monkeypatch, capfd):
    monkeypatch.setattr(sys, "argv", ["dft-rs", "feature-registry", "--format", "json"])
    assert dft_rs.cli_main() == 0
    report = json.loads(capfd.readouterr().out)
    assert any(feature["id"] == "energy" for feature in report["features"])


def test_explicit_arguments_remain_supported_and_errors_have_nonzero_status(monkeypatch, capfd, tmp_path):
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nconsole contract\nH 0 0 0\nH .74 0 0\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["dft-rs", "unknown-console-command", str(xyz)])
    assert dft_rs.cli_main(["--version"]) == 0
    assert capfd.readouterr().out.startswith("dft-rs ")
    assert dft_rs.cli_main() == 1
    output = capfd.readouterr()
    assert "unknown-console-command" in output.err
