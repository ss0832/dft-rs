# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Availability probe for the PCM/SMD oracle stack (task #22 groundwork).
import importlib

import pyscf

print("pyscf", pyscf.__version__)
for m in [
    "pyscf.solvent",
    "pyscf.solvent.pcm",
    "pyscf.solvent.ddcosmo",
    "pyscf.solvent.smd",
    "pyscf.solvent.grad.pcm",
    "pyscf.solvent.hessian.pcm",
]:
    try:
        importlib.import_module(m)
        print("OK  ", m)
    except Exception as e:
        print("MISS", m, type(e).__name__, e)

try:
    from pyscf.solvent import pcm

    print("PCM methods:", [x for x in dir(pcm.PCM) if not x.startswith("_")][:20])
    print("PCM defaults: method attr present:", hasattr(pcm.PCM, "method"))
except Exception as e:
    print("pcm introspection failed:", e)
