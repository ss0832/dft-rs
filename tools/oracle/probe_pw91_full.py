# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Full spin-polarized PW91-C vs libxc at the pinned test points, scanning
# constant-convention variants (nu, beta, fpp0) to find the libxc combo.
import itertools
import numpy as np
from pyscf.dft import libxc

PAPER = [
    (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
    (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671),
]

def pw_g(rs, P):
    A, a1, b1, b2, b3, b4 = P
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def c_rs(rs):
    return 0.001667 + (0.002568 + 0.023266 * rs + 7.389e-6 * rs**2) / (
        1 + 8.723 * rs + 0.472 * rs**2 + 1e4 * 7.389e-6 * rs**3
    )

def mine(ra, rb, sig, nu, beta, fpp0):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = pw_g(rs, PAPER[0]); e1 = pw_g(rs, PAPER[1]); ac = -pw_g(rs, PAPER[2])
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    z4 = z**4
    eps = e0 + ac * fz / fpp0 * (1 - z4) + (e1 - e0) * fz * z4
    g = 0.5 * ((1 + z) ** (2 / 3) + (1 - z) ** (2 / 3))
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    t2 = sig / (4 * g * g * ks**2 * rho**2)
    alpha = 0.09
    cc0 = 0.004235
    cx = -0.001667
    g3 = g**3
    A = (2 * alpha / beta) / (np.exp(-2 * alpha * eps / (g3 * beta**2)) - 1)
    h0 = g3 * (beta**2 / (2 * alpha)) * np.log(
        1 + (2 * alpha / beta) * t2 * (1 + A * t2) / (1 + A * t2 + A**2 * t2**2)
    )
    h1 = nu * (c_rs(rs) - cc0 - 3 * cx / 7) * g3 * t2 * np.exp(
        -100 * g3 * g * (ks**2 / kf**2) * t2
    )
    return rho * (eps + h0 + h1)

CASES = [
    ([0.3, 0.1], [0.1, 0.1, 0.15], [0.08, 0.05, 0.12]),
    ([1.2, 0.9], [0.3, -0.2, 0.4], [0.25, 0.1, -0.3]),
    ([0.02, 0.05], [0.01, 0.0, 0.02], [0.02, 0.01, -0.01]),
]

NU_EXACT = (16 / np.pi) * (3 * np.pi**2) ** (1 / 3)
FPP0_EXACT = 8.0 / (9.0 * (2 ** (4 / 3) - 2))

refs = []
for (r, ga, gb) in CASES:
    ga = np.array(ga); gb = np.array(gb)
    va = np.array([[r[0]], [ga[0]], [ga[1]], [ga[2]]])
    vb = np.array([[r[1]], [gb[0]], [gb[1]], [gb[2]]])
    exc = libxc.eval_xc(",gga_c_pw91", (va, vb), spin=1)[0][0]
    refs.append(exc * (r[0] + r[1]))
print("libxc refs:", ["%.12f" % x for x in refs])

variants = {
    "nu": {"exact": NU_EXACT, "paper7": 15.75592},
    "beta": {"nu*cc0": None, "pbe": 0.06672455060314922},
    "fpp0": {"exact": FPP0_EXACT, "paper": 1.709921},
}
best = None
for nu_k, beta_k, f_k in itertools.product(
    variants["nu"], variants["beta"], variants["fpp0"]
):
    nu = variants["nu"][nu_k]
    beta = nu * 0.004235 if beta_k == "nu*cc0" else variants["beta"][beta_k]
    fpp0 = variants["fpp0"][f_k]
    diffs = []
    for (r, ga, gb), ref in zip(CASES, refs):
        ga = np.array(ga); gb = np.array(gb)
        sig = float(ga @ ga + 2 * ga @ gb + gb @ gb)
        diffs.append(mine(r[0], r[1], sig, nu, beta, fpp0) - ref)
    worst = max(abs(d) for d in diffs)
    print(f"nu={nu_k:7s} beta={beta_k:7s} fpp0={f_k:6s}: worst {worst:.3e}  "
          + " ".join(f"{d:+.2e}" for d in diffs))
    if best is None or worst < best[0]:
        best = (worst, nu_k, beta_k, f_k)
print("BEST:", best)
