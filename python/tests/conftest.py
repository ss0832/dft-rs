# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Suite-wide setting: the common driver refuses a report in which a numeric
entry has no unit label or a label is malformed, so every test of a
calculation is also a test of its unit labels."""

import os

os.environ.setdefault("DFT_RS_STRICT_UNITS", "1")
