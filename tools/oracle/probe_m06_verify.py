# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Evaluate the M06 model with PUBLISHED coefficient tables against libxc
# over random points, for both PW92 digit conventions (paper vs MOD).
import numpy as np
from pyscf.dft import libxc

rng = np.random.default_rng(7)
N = 300
CF = 0.6 * (6 * np.pi**2) ** (2 / 3)
CX = -0.930525736349100025
ALPHA_X, ALPHA_SS, ALPHA_AB = 0.00186726, 0.00515088, 0.00304966
GC_SS, GC_AB = 0.06, 0.0031

PW_PAPER = [
    (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]
PW_MOD = [
    (0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]

TABLES = {
    "m06_l": dict(
        a=[0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615,
           30.97273, -23.18489, -56.73480, 21.60364, 34.21814, -9.049762],
        dx=[0.6012244, 0.004748822, -0.008635108, -0.000009308062, 0.00004482811, 0.0],
        cab=[0.6042374, 177.6783, -251.3252, 76.35173, -12.55699],
        dab=[0.3957626, -0.5614546, 0.01403963, 0.0009831442, -0.003577176, 0.0],
        css=[0.5349466, 0.5396620, -31.61217, 51.49592, -29.19613],
        dss=[0.4650534, 0.1617589, 0.1833657, 0.0004692100, -0.004990573, 0.0],
    ),
    "m06": dict(
        a=[0.5877943, -0.1371776, 0.2682367, -2.515898, -2.978892, 8.710679,
           16.88195, -4.489724, -32.99983, -14.49050, 20.43747, 12.56504],
        dx=[0.1422057, 0.0007370319, -0.01601373, 0.0, 0.0, 0.0],
        cab=[3.741539, 218.7098, -453.1252, 293.6479, -62.87470],
        dab=[-2.741539, -0.6720113, -0.07932688, 0.001918681, -0.002032902, 0.0],
        css=[0.5094055, -1.491085, 17.23922, -38.59018, 28.45044],
        dss=[0.4905945, -0.1437348, 0.2357824, 0.001871015, -0.003788963, 0.0],
    ),
    "m06_2x": dict(
        a=[0.4600000, -0.2206052, -0.09431788, 2.164494, -2.556466, -14.22133,
           15.55044, 35.98078, -27.22754, -39.24093, 15.22808, 15.22227],
        dx=[0.0] * 6,
        cab=[0.8833596, 33.57972, -70.43548, 49.78271, -18.52891],
        dab=[0.1166404, -0.09120847, -0.06726189, 0.00006720580, 0.0008448011, 0.0],
        css=[0.3097855, -5.528642, 13.47420, -32.13623, 28.46742],
        dss=[0.6902145, 0.09847204, 0.2214797, -0.001968264, -0.006775479, 0.0],
    ),
}

def pw_g(rs, A, a1, b1, b2, b3, b4):
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def pw92_eps(ra, rb, P):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = pw_g(rs, *P[0])
    e1 = pw_g(rs, *P[1])
    ac = -pw_g(rs, *P[2])
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

def pbe_x_spin(rho, sig):
    kappa, mu = 0.804, 0.2195149727645171
    r = 2 * rho
    kf2 = (3 * np.pi**2 * r) ** (2 / 3)
    s2 = 4 * sig / (4 * kf2 * r * r)
    fx = 1 + kappa - kappa / (1 + mu * s2 / kappa)
    return 0.5 * (-0.75 * (3 / np.pi) ** (1 / 3)) * r ** (4 / 3) * fx

def h_of(x2, z, alpha, d):
    g = 1 + alpha * (x2 + z)
    return (d[0] / g + (d[1] * x2 + d[2] * z) / g**2
            + (d[3] * x2 * x2 + d[4] * x2 * z + d[5] * z * z) / g**3)

def model(ra, rb, ga, gb, ta, tb, T, P):
    saa, sbb = ga @ ga, gb @ gb
    e = 0.0
    for rho, sig, tau in [(ra, saa, ta), (rb, sbb, tb)]:
        tls = 0.3 * (6 * np.pi**2) ** (2 / 3) * rho ** (5 / 3)
        w = (tls / tau - 1) / (tls / tau + 1)
        e += pbe_x_spin(rho, sig) * sum(T["a"][i] * w**i for i in range(12))
        x2 = sig / rho ** (8 / 3)
        z = 2 * tau / rho ** (5 / 3) - CF
        e += CX * rho ** (4 / 3) * h_of(x2, z, ALPHA_X, T["dx"])
    ea = ra * pw92_eps(ra, 1e-300, P)
    eb = rb * pw92_eps(rb, 1e-300, P)
    eab = (ra + rb) * pw92_eps(ra, rb, P) - ea - eb
    xa2, xb2 = saa / ra ** (8 / 3), sbb / rb ** (8 / 3)
    za, zb = 2 * ta / ra ** (5 / 3) - CF, 2 * tb / rb ** (5 / 3) - CF
    xbar2, zbar = xa2 + xb2, za + zb
    u = GC_AB * xbar2 / (1 + GC_AB * xbar2)
    e += eab * (sum(T["cab"][i] * u**i for i in range(5)) + h_of(xbar2, zbar, ALPHA_AB, T["dab"]))
    for e_ss, x2, z, tau in [(ea, xa2, za, ta), (eb, xb2, zb, tb)]:
        d = 1 - x2 / (4 * (z + CF))
        u = GC_SS * x2 / (1 + GC_SS * x2)
        e += e_ss * d * (sum(T["css"][i] * u**i for i in range(5)) + h_of(x2, z, ALPHA_SS, T["dss"]))
    return e

pts = []
for _ in range(N):
    ra, rb = rng.uniform(0.05, 2.0, 2)
    ga, gb = rng.normal(0, 0.6, 3), rng.normal(0, 0.6, 3)
    ta = (ga @ ga) / (8 * ra) + rng.uniform(0.1, 2.5) * 0.3 * (6 * np.pi**2) ** (2 / 3) * ra ** (5 / 3)
    tb = (gb @ gb) / (8 * rb) + rng.uniform(0.1, 2.5) * 0.3 * (6 * np.pi**2) ** (2 / 3) * rb ** (5 / 3)
    pts.append((ra, rb, ga, gb, ta, tb))

for name, T in TABLES.items():
    errs = {"paper": [], "mod": []}
    for ra, rb, ga, gb, ta, tb in pts:
        rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
        rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
        exc = libxc.eval_xc(name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
        errs["paper"].append(abs(model(ra, rb, ga, gb, ta, tb, T, PW_PAPER) - exc))
        errs["mod"].append(abs(model(ra, rb, ga, gb, ta, tb, T, PW_MOD) - exc))
    print(f"{name}: paper {max(errs['paper']):.2e}  mod {max(errs['mod']):.2e}")
