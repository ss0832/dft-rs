# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover the B97M-V (semilocal part) sparse 2D series from libxc by LSQ
# (same machinery as wB97M-V but with plain LSDA exchange, no SR factor).
import numpy as np
from pyscf.dft import libxc

rng = np.random.default_rng(31)
N = 800
CF = 0.6 * (6 * np.pi**2) ** (2 / 3)
CX = -0.930525736349100025
GX, GSS, GAB = 0.004, 0.2, 0.006
NW, NU = 9, 9

PW_MOD = [
    (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]

def pw_g(rs, A, a1, b1, b2, b3, b4):
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def pw92_eps(ra, rb):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = pw_g(rs, *PW_MOD[0]); e1 = pw_g(rs, *PW_MOD[1]); ac = -pw_g(rs, *PW_MOD[2])
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

pts = []
for _ in range(N):
    ra, rb = rng.uniform(0.05, 2.0, 2)
    ga, gb = rng.normal(0, 0.6, 3), rng.normal(0, 0.6, 3)
    tf = lambda r: 0.3 * (6 * np.pi**2) ** (2 / 3) * r ** (5 / 3)
    ta = (ga @ ga) / (8 * ra) + rng.uniform(0.1, 2.5) * tf(ra)
    tb = (gb @ gb) / (8 * rb) + rng.uniform(0.1, 2.5) * tf(rb)
    pts.append((ra, rb, ga, gb, ta, tb))

exc_ref = []
for ra, rb, ga, gb, ta, tb in pts:
    rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
    rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
    exc_ref.append(libxc.eval_xc("b97m_v", (rho_a, rho_b), spin=1)[0][0] * (ra + rb))
exc_ref = np.array(exc_ref)

def design_row(ra, rb, ga, gb, ta, tb):
    saa, sbb = ga @ ga, gb @ gb
    tls = lambda r: 0.3 * (6 * np.pi**2) ** (2 / 3) * r ** (5 / 3)
    row = []
    xcols = np.zeros(NW * NU)
    for rho, sig, tau in [(ra, saa, ta), (rb, sbb, tb)]:
        t0 = tls(rho)
        w = (t0 - tau) / (t0 + tau)
        x2 = sig / rho ** (8 / 3)
        u = GX * x2 / (1 + GX * x2)
        e0 = CX * rho ** (4 / 3)
        k = 0
        for i in range(NW):
            for j in range(NU):
                xcols[k] += e0 * w**i * u**j
                k += 1
    row.extend(xcols)
    scols = np.zeros(NW * NU)
    ea = ra * pw92_eps(ra, 1e-300)
    eb = rb * pw92_eps(rb, 1e-300)
    for rho, sig, tau, e_ss in [(ra, saa, ta, ea), (rb, sbb, tb, eb)]:
        t0 = tls(rho)
        w = (t0 - tau) / (t0 + tau)
        x2 = sig / rho ** (8 / 3)
        u = GSS * x2 / (1 + GSS * x2)
        k = 0
        for i in range(NW):
            for j in range(NU):
                scols[k] += e_ss * w**i * u**j
                k += 1
    row.extend(scols)
    eab = (ra + rb) * pw92_eps(ra, rb) - ea - eb
    xa2, xb2 = saa / ra ** (8 / 3), sbb / rb ** (8 / 3)
    xbar2 = 0.5 * (xa2 + xb2)
    u = GAB * xbar2 / (1 + GAB * xbar2)
    t0a, t0b = tls(ra), tls(rb)
    tbar = 0.5 * (t0a / ta + t0b / tb)
    w = (tbar - 1) / (tbar + 1)
    ocols = np.zeros(NW * NU)
    k = 0
    for i in range(NW):
        for j in range(NU):
            ocols[k] = eab * w**i * u**j
            k += 1
    row.extend(ocols)
    return row

A = np.array([design_row(*p) for p in pts])
coef, *_ = np.linalg.lstsq(A, exc_ref, rcond=None)
resid = np.abs(A @ coef - exc_ref).max()
print(f"b97m_v: resid {resid:.2e}")
for ch, off in [("x", 0), ("ss", NW * NU), ("ab", 2 * NW * NU)]:
    print(f" channel {ch}:")
    for i in range(NW):
        for j in range(NU):
            v = coef[off + i * NU + j]
            if abs(v) > 5e-4:
                print(f"   ({i},{j}): {v:.9g}")
