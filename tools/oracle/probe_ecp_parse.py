# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
import os
from gen_ecp import parse_g94_with_ecp
from gen_integrals import DATA_DIR

b, e = parse_g94_with_ecp(os.path.join(DATA_DIR, "lanl2dz.g94"), {"H", "Na", "Cl"})
print("basis shells:", {k: len(v) for k, v in b.items()})
print("ecps:", {k: (v[0], [ent[0] for ent in v[1]]) for k, v in e.items()})
for k, v in b.items():
    if not v:
        print("EMPTY basis for", k)
