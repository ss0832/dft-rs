# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Recover M06-family coefficients from libxc by linear least squares —
# validates the functional FORM (PBE base, VS98 h, w-series, D factor, τ
# conventions) and yields digit-reliable parameter tables.
import numpy as np
from pyscf.dft import libxc

import sys

# Usage: probe_revm06_fit.py [libxc x,c pair ...]; default: revM06-L, revM06.
NAMES = sys.argv[1:] or ["mgga_x_revm06_l,mgga_c_revm06_l", "hyb_mgga_x_revm06,mgga_c_revm06"]
rng = np.random.default_rng(23)
N = 900
CF = 0.6 * (6 * np.pi**2) ** (2 / 3)
CX = -0.930525736349100025
ALPHA_X = 0.00186726
ALPHA_SS = 0.00515088
ALPHA_AB = 0.00304966
GC_SS = 0.06
GC_AB = 0.0031

pts = []
for _ in range(N):
    ra, rb = rng.uniform(0.05, 2.0, 2)
    ga = rng.normal(0, 0.6, 3)
    gb = rng.normal(0, 0.6, 3)
    # τ between the von Weizsäcker bound and a few× Thomas-Fermi.
    for rho, g, tag in [(ra, ga, "a"), (rb, gb, "b")]:
        pass
    saa, sbb = ga @ ga, gb @ gb
    tw_a = saa / (8 * ra)
    tw_b = sbb / (8 * rb)
    tf_a = 0.3 * (6 * np.pi**2) ** (2 / 3) * ra ** (5 / 3)
    tf_b = 0.3 * (6 * np.pi**2) ** (2 / 3) * rb ** (5 / 3)
    ta = tw_a + rng.uniform(0.1, 2.5) * tf_a
    tb = tw_b + rng.uniform(0.1, 2.5) * tf_b
    pts.append((ra, rb, ga, gb, ta, tb))

def pw_g(rs, A, a1, b1, b2, b3, b4):
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def pw92_eps(ra, rb):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = pw_g(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = pw_g(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    ac = -pw_g(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

def pbe_x_spin(rho, sig):
    kappa, mu = 0.804, 0.2195149727645171
    r = 2 * rho
    s4 = 4 * sig
    kf2 = (3 * np.pi**2 * r) ** (2 / 3)
    s2 = s4 / (4 * kf2 * r * r)
    fx = 1 + kappa - kappa / (1 + mu * s2 / kappa)
    return 0.5 * (-0.75 * (3 / np.pi) ** (1 / 3)) * r ** (4 / 3) * fx

def h_terms(x2, z, alpha):
    g = 1 + alpha * (x2 + z)
    return [1 / g, x2 / g**2, z / g**2, x2 * x2 / g**3, x2 * z / g**3, z * z / g**3]

def design_row(ra, rb, ga, gb, ta, tb):
    saa, sbb = ga @ ga, gb @ gb
    row = []
    # Exchange: 12 w-series columns + 6 VS98 columns.
    wcols = [0.0] * 12
    hcols = [0.0] * 6
    for rho, sig, tau in [(ra, saa, ta), (rb, sbb, tb)]:
        tlsda = 0.3 * (6 * np.pi**2) ** (2 / 3) * rho ** (5 / 3)
        t = tlsda / tau
        w = (t - 1) / (t + 1)
        epbe = pbe_x_spin(rho, sig)
        for i in range(12):
            wcols[i] += epbe * w**i
        x2 = sig / rho ** (8 / 3)
        z = 2 * tau / rho ** (5 / 3) - CF
        elsda = CX * rho ** (4 / 3)
        for k, h in enumerate(h_terms(x2, z, ALPHA_X)):
            hcols[k] += elsda * h
    row += wcols + hcols
    # Correlation ab: 5 g columns + 6 h columns.
    ea = ra * pw92_eps(ra, 1e-300)
    eb = rb * pw92_eps(rb, 1e-300)
    eab = (ra + rb) * pw92_eps(ra, rb) - ea - eb
    xa2 = saa / ra ** (8 / 3)
    xb2 = sbb / rb ** (8 / 3)
    za = 2 * ta / ra ** (5 / 3) - CF
    zb = 2 * tb / rb ** (5 / 3) - CF
    xbar2 = xa2 + xb2
    zbar = za + zb
    uab = GC_AB * xbar2 / (1 + GC_AB * xbar2)
    row += [eab * uab**i for i in range(5)]
    row += [eab * h for h in h_terms(xbar2, zbar, ALPHA_AB)]
    # Correlation ss: 5 g + 6 h columns (D factor included).
    gss = [0.0] * 5
    hss = [0.0] * 6
    for rho, sig, tau, e_ss, x2, z in [
        (ra, saa, ta, ea, xa2, za),
        (rb, sbb, tb, eb, xb2, zb),
    ]:
        d = 1 - x2 / (4 * (z + CF))
        u = GC_SS * x2 / (1 + GC_SS * x2)
        for i in range(5):
            gss[i] += e_ss * d * u**i
        for k, h in enumerate(h_terms(x2, z, ALPHA_SS)):
            hss[k] += e_ss * d * h
    row += gss + hss
    return row

for name in NAMES:
    A, y = [], []
    for ra, rb, ga, gb, ta, tb in pts:
        rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
        rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
        exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
        A.append(design_row(ra, rb, ga, gb, ta, tb))
        y.append(exc)
    A = np.asarray(A)
    y = np.asarray(y)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = np.abs(A @ coef - y).max()
    print(f"{name}: max residual {resid:.2e}")
    labels = ["ax(w)", "dx(VS98)", "c_ab", "d_ab", "c_ss", "d_ss"]
    sizes = [12, 6, 5, 6, 5, 6]
    off = 0
    for lab, s in zip(labels, sizes):
        print(f"  {lab}: [{', '.join(f'{v:.15g}' for v in coef[off:off+s])}]")
        off += s
