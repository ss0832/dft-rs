# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: replicate the dft-rs B97-family formula in Python and
# compare against libxc pointwise (find transcription/base-LDA issues).
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
ga = np.array([0.1, 0.1, 0.15])
gb = np.array([0.08, 0.05, 0.12])
saa, sab, sbb = ga @ ga, ga @ gb, gb @ gb
rho_a = np.array([[ra], [ga[0]], [ga[1]], [ga[2]]])
rho_b = np.array([[rb], [gb[0]], [gb[1]], [gb[2]]])

# --- PW92 epsilon (both digit sets) ---
def pw_g(rs, A, a1, b1, b2, b3, b4):
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

def pw92_eps(ra, rb, mod):
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    if mod:
        A0, A1, Aa = 0.0310907, 0.01554535, 0.0168869
    else:
        A0, A1, Aa = 0.031091, 0.015545, 0.016887
    e0 = pw_g(rs, A0, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = pw_g(rs, A1, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    ac = -pw_g(rs, Aa, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4

CX = -0.930525736349100025

def b97_like(cx, css, cab, gammas=(0.004, 0.2, 0.006), mod=True):
    gx, gss, gab = gammas
    def g(u, c):
        return sum(ck * u**k for k, ck in enumerate(c))
    e = 0.0
    x2 = {}
    for rho, sig, tag in [(ra, saa, "a"), (rb, sbb, "b")]:
        xs2 = sig / rho ** (8 / 3)
        x2[tag] = xs2
        ux = gx * xs2 / (1 + gx * xs2)
        e += CX * rho ** (4 / 3) * g(ux, cx)
        uss = gss * xs2 / (1 + gss * xs2)
        e += rho * pw92_eps(rho, 1e-300, mod) * g(uss, css)
    ecab = (ra + rb) * pw92_eps(ra, rb, mod) \
        - ra * pw92_eps(ra, 1e-300, mod) - rb * pw92_eps(rb, 1e-300, mod)
    xavg = 0.5 * (x2["a"] + x2["b"])
    uab = gab * xavg / (1 + gab * xavg)
    e += ecab * g(uab, cab)
    return e

SETS = {
    "b97": ([0.8094, 0.5073, 0.7481], [0.1737, 2.3487, -2.4868], [0.9454, 0.7471, -4.5961]),
    "hcth93": (
        [1.09320, -0.744056, 5.59920, -6.78549, 4.49357],
        [0.222601, -0.0338622, -0.0125170, -0.802496, 1.55396],
        [0.729974, 3.35287, -11.5430, 8.08564, -4.47857],
    ),
    "hcth147": (
        [1.09025, -0.799194, 5.57212, -5.86760, 3.04544],
        [0.562576, 0.0171436, -1.30636, -0.0341445, 2.32803],
        [0.542352, 7.01464, -28.3822, 35.0329, -20.4284],
    ),
}

for name, (cx, css, cab) in SETS.items():
    libxc_name = name if name != "b97" else "hyb_gga_xc_b97"
    exc = libxc.eval_xc(libxc_name, (rho_a, rho_b), spin=1)[0][0] * (ra + rb)
    ours_mod = b97_like(cx, css, cab, mod=True)
    ours_paper = b97_like(cx, css, cab, mod=False)
    print(f"{name:8s} libxc {exc:.10f}  ours(MOD) {ours_mod:.10f} (d={ours_mod-exc:+.2e})"
          f"  ours(paper) {ours_paper:.10f} (d={ours_paper-exc:+.2e})")
