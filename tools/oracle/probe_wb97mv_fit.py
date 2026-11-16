# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover the wB97M-V (semilocal part) functional form + coefficients from
# libxc by linear least squares over 2D (w, u) power series, testing form
# hypotheses (D factor, opposite-spin variable definitions) by residual.
import itertools
import numpy as np
from pyscf.dft import libxc

rng = np.random.default_rng(11)
N = 800
CF = 0.6 * (6 * np.pi**2) ** (2 / 3)
CX = -0.930525736349100025
OMEGA = 0.3
GX, GSS, GAB = 0.004, 0.2, 0.006
NW, NU = 9, 9  # w powers 0..8, u powers 0..8

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
    e0 = pw_g(rs, *PW_MOD[0])
    e1 = pw_g(rs, *PW_MOD[1])
    ac = -pw_g(rs, *PW_MOD[2])
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

def sr_factor(a):
    # F(a) = 1 − (8/3)a[√π erf(1/(2a)) − 3a + 4a³ + (2a − 4a³)exp(−1/(4a²))]
    from scipy.special import erf
    e = np.exp(-1 / (4 * a * a))
    return 1 - (8 / 3) * a * (np.sqrt(np.pi) * erf(1 / (2 * a)) - 3 * a + 4 * a**3
                              + (2 * a - 4 * a**3) * e)

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
    exc_ref.append(libxc.eval_xc("wb97m_v", (rho_a, rho_b), spin=1)[0][0] * (ra + rb))
exc_ref = np.array(exc_ref)

def design_row(ra, rb, ga, gb, ta, tb, use_d, xab_mode, wab_mode):
    saa, sbb = ga @ ga, gb @ gb
    tls = lambda r: 0.3 * (6 * np.pi**2) ** (2 / 3) * r ** (5 / 3)
    row = []
    # Exchange: SR-LSDA × w^i u^j.
    xcols = np.zeros(NW * NU)
    for rho, sig, tau in [(ra, saa, ta), (rb, sbb, tb)]:
        t0 = tls(rho)
        w = (t0 - tau) / (t0 + tau)
        x2 = sig / rho ** (8 / 3)
        u = GX * x2 / (1 + GX * x2)
        kf = (6 * np.pi**2 * rho) ** (1 / 3)
        e0 = CX * rho ** (4 / 3) * sr_factor(OMEGA / (2 * kf))
        k = 0
        for i in range(NW):
            for j in range(NU):
                xcols[k] += e0 * w**i * u**j
                k += 1
    row.extend(xcols)
    # Same-spin correlation.
    scols = np.zeros(NW * NU)
    ea = ra * pw92_eps(ra, 1e-300)
    eb = rb * pw92_eps(rb, 1e-300)
    for rho, sig, tau, e_ss in [(ra, saa, ta, ea), (rb, sbb, tb, eb)]:
        t0 = tls(rho)
        w = (t0 - tau) / (t0 + tau)
        x2 = sig / rho ** (8 / 3)
        u = GSS * x2 / (1 + GSS * x2)
        z = 2 * tau / rho ** (5 / 3) - CF
        d = 1 - x2 / (4 * (z + CF)) if use_d else 1.0
        k = 0
        for i in range(NW):
            for j in range(NU):
                scols[k] += e_ss * d * w**i * u**j
                k += 1
    row.extend(scols)
    # Opposite-spin correlation.
    eab = (ra + rb) * pw92_eps(ra, rb) - ea - eb
    xa2, xb2 = saa / ra ** (8 / 3), sbb / rb ** (8 / 3)
    xbar2 = xa2 + xb2 if xab_mode == "sum" else 0.5 * (xa2 + xb2)
    u = GAB * xbar2 / (1 + GAB * xbar2)
    t0a, t0b = tls(ra), tls(rb)
    if wab_mode == "t_avg":
        tbar = 0.5 * (t0a / ta + t0b / tb)
        w = (tbar - 1) / (tbar + 1)
    elif wab_mode == "w_avg":
        w = 0.5 * ((t0a - ta) / (t0a + ta) + (t0b - tb) / (t0b + tb))
    else:  # tau_sum
        w = (t0a + t0b - ta - tb) / (t0a + t0b + ta + tb)
    ocols = np.zeros(NW * NU)
    k = 0
    for i in range(NW):
        for j in range(NU):
            ocols[k] = eab * w**i * u**j
            k += 1
    row.extend(ocols)
    return row

best = []
for use_d, xab_mode, wab_mode in itertools.product(
    [True, False], ["sum", "avg"], ["t_avg"]
):
    A = np.array([design_row(*p, use_d, xab_mode, wab_mode) for p in pts])
    coef, *_ = np.linalg.lstsq(A, exc_ref, rcond=None)
    resid = np.abs(A @ coef - exc_ref).max()
    best.append((resid, use_d, xab_mode, wab_mode, coef))
    print(f"D={use_d!s:5} xab={xab_mode:3} wab={wab_mode:7}: resid {resid:.2e}")

best.sort(key=lambda t: t[0])
resid, use_d, xab_mode, wab_mode, coef = best[0]
print(f"\nBEST: D={use_d} xab={xab_mode} wab={wab_mode} resid={resid:.2e}")
for ch, off in [("x", 0), ("ss", NW * NU), ("ab", 2 * NW * NU)]:
    print(f" channel {ch}: nonzero (i=w-pow, j=u-pow):")
    for i in range(NW):
        for j in range(NU):
            v = coef[off + i * NU + j]
            if abs(v) > 5e-4:
                print(f"   ({i},{j}): {v:.9g}")
