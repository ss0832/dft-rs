# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Extract libxc gga_c_pw91's implied C(rs) from the sigma->0 limit of the
# gradient term H1 (H0/t^2 -> beta exactly), then fit the Rasolt-Geldart
# rational to identify which digits differ from the P86 convention.
import numpy as np
from scipy.optimize import least_squares
from pyscf.dft import libxc

PAPER0 = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)

def pw_g(rs, P):
    A, a1, b1, b2, b3, b4 = P
    s = np.sqrt(rs)
    den = 2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)
    return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / den)

NU = (16 / np.pi) * (3 * np.pi**2) ** (1 / 3)
BETA = NU * 0.004235

rss, cimpl = [], []
for rho in [8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    eps = pw_g(rs, PAPER0)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    sig = 1e-10 * rho * rho
    r = rho / 2
    gr = np.sqrt(sig) / 2
    va = np.array([[r], [gr], [0.0], [0.0]])
    ref = libxc.eval_xc(",gga_c_pw91", (va, va), spin=1)[0][0]
    t2 = sig / (4 * ks**2 * rho**2)
    dc = ((ref - eps) / t2 - BETA) / NU
    # C(rs) = dc + Cc0 + (3/7) Cx  with Cc0 = 0.004235, Cx = -0.001667
    c = dc + 0.004235 + (3.0 / 7.0) * (-0.001667)
    rss.append(rs)
    cimpl.append(c)
    print(f"rs={rs:.6f}: C_impl={c:.12f}")

rss = np.array(rss)
cimpl = np.array(cimpl)

def model(p, rs):
    c0, n0, n1, n2, d1, d2, d3 = p
    return c0 + (n0 + n1 * rs + n2 * rs * rs) / (
        1 + d1 * rs + d2 * rs * rs + d3 * rs**3
    )

p0 = [0.001667, 0.002568, 0.023266, 7.389e-6, 8.723, 0.472, 7.389e-2]
mine = model(p0, rss)
print("\nmine vs impl:")
for rs, a, b in zip(rss, mine, cimpl):
    print(f"  rs={rs:.4f}: mine {a:.10f} impl {b:.10f} diff {b - a:+.3e}")

res = least_squares(lambda p: model(p, rss) - cimpl, p0, xtol=1e-15, ftol=1e-15)
print("\nfit:", ["%.8g" % v for v in res.x])
print("fit residual:", np.max(np.abs(model(res.x, rss) - cimpl)))
