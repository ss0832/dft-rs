# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Identify the SMD pairwise surface-tension shape T(R) from the measured
# aqueous curves.
#
#   excess(R) = T(R) * (s_ab A_a(R) + s_ba A_b(R))
#
# The SMx family uses T(R) = exp(d / (R - R0)) for R < R0 and 0 beyond, so
# each pair has three unknowns (R0, d) plus the two linear coefficients.
# For a fixed (R0, d) the coefficients are an exact 2x2 least-squares solve,
# so the fit is a two-dimensional scan with an exact inner solve.
#
# NOTE ON THE INPUT: these curves are only meaningful if the baseline
# `sum_Z sigma_Z A_Z` used the CDS radii (Bondi), not the intrinsic Coulomb
# radii. An earlier pass used the latter and produced fits that looked good
# and were not; regenerate with measure_smd_pairs.py before trusting
# anything here.
#
# GUARDS against the overfitting that bit the first attempt: R0 must not sit
# at a scan edge, the window must keep at least twelve points, and the
# excess must genuinely vanish beyond R0.
import os
from collections import defaultdict

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "smd_pair_curves.csv")
R0_LO, R0_HI = 1.20, 6.00
PIN_D = os.environ.get("PIN_D", "0") == "1"


def load():
    curves = defaultdict(list)
    with open(SRC) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            a, b, r, aa, ab, ex = line.split(",")
            curves[(a, b)].append((float(r), float(aa), float(ab), float(ex)))
    return curves


def fit_pair(rows):
    rows = sorted(rows)
    best = None
    r_max = max(r for r, _, _, _ in rows)
    for r0 in np.arange(R0_LO, min(R0_HI, r_max + 0.4), 0.005):
        sub = [t for t in rows if t[0] < r0 - 1e-9]
        if len(sub) < 12:
            continue
        outside = [abs(ex) for r, _, _, ex in rows if r >= r0]
        if outside and max(outside) > 2.0:
            continue
        # Four of the six pairs land on d = 0.300 independently, so the
        # scan is run BOTH ways: free d, and d pinned to that common value.
        # A pair that only fits with a wildly different d is not evidence of
        # a different exponent, it is evidence the window is wrong.
        d_grid = [0.300] if PIN_D else np.arange(0.02, 2.51, 0.01)
        for d in d_grid:
            t = np.array([np.exp(d / (r - r0)) for r, _, _, _ in sub])
            A = np.array([[t[k] * aa, t[k] * ab] for k, (_, aa, ab, _) in enumerate(sub)])
            y = np.array([ex for _, _, _, ex in sub])
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            rel = np.abs(A @ sol - y).max() / max(np.abs(y).max(), 1e-12)
            if best is None or rel < best[0]:
                best = (rel, r0, d, sol[0], sol[1], len(sub))
    return best


def main():
    curves = load()
    print("# pair,R0,d,s_ab,s_ba,npts,max_rel_resid,verdict")
    for key in sorted(curves):
        best = fit_pair(curves[key])
        if best is None:
            print(f"# {key[0]}-{key[1]}: no usable window")
            continue
        rel, r0, d, sab, sba, n = best
        edge = abs(r0 - R0_LO) < 1e-6 or (
            not PIN_D and (abs(d - 0.02) < 1e-9 or abs(d - 2.50) < 1e-9)
        )
        verdict = "EDGE" if edge else ("ok" if rel < 0.05 else "poor")
        print(f"{key[0]}-{key[1]},{r0:.3f},{d:.3f},{sab:.4f},{sba:.4f},"
              f"{n},{rel:.5f},{verdict}")


if __name__ == "__main__":
    main()
