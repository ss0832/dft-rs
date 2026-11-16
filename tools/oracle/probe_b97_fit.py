# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover the exact B97-family series coefficients used by libxc via linear
# least squares (the energy is linear in the 15 coefficients at fixed γ) —
# a digit-reliable check of the published parameter tables.
import numpy as np
from pyscf.dft import libxc

rng = np.random.default_rng(7)
N = 400
pts = []
for _ in range(N):
    ra, rb = rng.uniform(0.05, 2.0, 2)
    ga = rng.normal(0, 0.5, 3)
    gb = rng.normal(0, 0.5, 3)
    pts.append((ra, rb, ga, gb))

def pw_g(rs, A, a1, b1, b2, b3, b4):
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def pw92_eps(ra, rb):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = pw_g(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = pw_g(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    ac = -pw_g(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

CX = -0.930525736349100025
GX, GSS, GAB = 0.004, 0.2, 0.006

def design_row(ra, rb, ga, gb, n):
    saa, sbb = ga @ ga, gb @ gb
    row = []
    xa2 = saa / ra ** (8 / 3)
    xb2 = sbb / rb ** (8 / 3)
    uxa = GX * xa2 / (1 + GX * xa2)
    uxb = GX * xb2 / (1 + GX * xb2)
    for k in range(n):
        row.append(CX * (ra ** (4 / 3) * uxa**k + rb ** (4 / 3) * uxb**k))
    ussa = GSS * xa2 / (1 + GSS * xa2)
    ussb = GSS * xb2 / (1 + GSS * xb2)
    ea = ra * pw92_eps(ra, 1e-300)
    eb = rb * pw92_eps(rb, 1e-300)
    for k in range(n):
        row.append(ea * ussa**k + eb * ussb**k)
    ecab = (ra + rb) * pw92_eps(ra, rb) - ea - eb
    xavg = 0.5 * (xa2 + xb2)
    uab = GAB * xavg / (1 + GAB * xavg)
    for k in range(n):
        row.append(ecab * uab**k)
    return row

for name, n in [("hyb_gga_xc_b97", 3), ("hcth93", 5), ("hcth120", 5), ("hcth147", 5), ("hcth407", 5)]:
    A = []
    y = []
    for ra, rb, ga, gb in pts:
        rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]]])
        rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]]])
        exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
        A.append(design_row(ra, rb, ga, gb, n))
        y.append(exc)
    coef, res, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)
    resid = np.abs(np.asarray(A) @ coef - y).max()
    cx, css, cab = coef[:n], coef[n : 2 * n], coef[2 * n :]
    print(f"{name}: max residual {resid:.2e}")
    print("  cx :", np.round(cx, 7).tolist())
    print("  css:", np.round(css, 7).tolist())
    print("  cab:", np.round(cab, 7).tolist())
