# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover wB97X-D / wB97X-V series coefficients from libxc by linear least
# squares (SR-LSDA-attenuated exchange series + B97 correlation series).
import numpy as np
from pyscf.dft import libxc

rng = np.random.default_rng(11)
N = 500
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

def srf(a):
    # Short-range LSDA attenuation factor F(a).
    if a > 5.0:
        u = 1 / (2 * a)
        return u * u / 9 - u**4 / 60
    b = (np.sqrt(np.pi) * np.math.erf(1 / (2 * a)) if False else np.sqrt(np.pi) * __import__("math").erf(1 / (2 * a))) \
        - 3 * a + 4 * a**3 + (2 * a - 4 * a**3) * np.exp(-1 / (4 * a * a))
    return 1 - 8 / 3 * a * b

def design_row(ra, rb, ga, gb, n, omega):
    saa, sbb = ga @ ga, gb @ gb
    row = []
    xa2 = saa / ra ** (8 / 3)
    xb2 = sbb / rb ** (8 / 3)
    uxa = GX * xa2 / (1 + GX * xa2)
    uxb = GX * xb2 / (1 + GX * xb2)
    fa = srf(omega / (2 * (6 * np.pi**2 * ra) ** (1 / 3)))
    fb = srf(omega / (2 * (6 * np.pi**2 * rb) ** (1 / 3)))
    for k in range(n):
        row.append(CX * (ra ** (4 / 3) * fa * uxa**k + rb ** (4 / 3) * fb * uxb**k))
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

for name, omega in [("wb97x_d", 0.2), ("wb97x_v", 0.3), ("wb97", 0.4), ("wb97x", 0.3)]:
    try:
        print(name, "rsh:", libxc.rsh_coeff(name))
    except Exception as e:
        print(name, "rsh error", e)
        continue
    n = 5
    A, y = [], []
    for ra, rb, ga, gb in pts:
        rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]]])
        rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]]])
        exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
        A.append(design_row(ra, rb, ga, gb, n, omega))
        y.append(exc)
    coef, *_ = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=None)
    resid = np.abs(np.asarray(A) @ coef - y).max()
    print(f"  max residual {resid:.2e}")
    print("  cx :", np.round(coef[:n], 7).tolist())
    print("  css:", np.round(coef[n : 2 * n], 7).tolist())
    print("  cab:", np.round(coef[2 * n :], 7).tolist())
