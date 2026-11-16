# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Settle the conventions of the M08 exchange/correlation and MN12 exchange
# forms numerically: candidate closed forms (written from the published
# definitions) are evaluated with the libxc default parameters at random
# spin-polarized meta-GGA points and compared with libxc's energy density
# (numerical oracle only). Prints the largest relative deviation of every
# candidate.
#
# Run (WSL): <venv>/bin/python probe_m08_form.py

import itertools

import numpy as np
from pyscf.dft import libxc

from xc_params import load, params_of

rng = np.random.default_rng(7)
CF = 0.3 * (6 * np.pi**2) ** (2 / 3)  # tau_LSDA / rho_sigma^(5/3), per spin
CX = -0.75 * (6 / np.pi) ** (1 / 3)  # e_x^LSDA / rho_sigma^(4/3), per spin


def points(n=300):
    out = []
    for _ in range(n):
        ra, rb = rng.uniform(0.05, 2.0, 2)
        ga, gb = rng.normal(0, 0.6, 3), rng.normal(0, 0.6, 3)
        ta = ga @ ga / (8 * ra) + rng.uniform(0.1, 2.5) * CF * ra ** (5 / 3)
        tb = gb @ gb / (8 * rb) + rng.uniform(0.1, 2.5) * CF * rb ** (5 / 3)
        out.append((ra, rb, ga, gb, ta, tb))
    return out


def libxc_e(name, p):
    ra, rb, ga, gb, ta, tb = p
    va = np.array([[ra], [ga[0]], [ga[1]], [ga[2]], [0.0], [ta]])
    vb = np.array([[rb], [gb[0]], [gb[1]], [gb[2]], [0.0], [tb]])
    return libxc.eval_xc(name, (va, vb), spin=1)[0][0] * (ra + rb)


def pbe_like(s2, kappa, mu, rpbe):
    if rpbe:
        return 1 + kappa * (1 - np.exp(-mu * s2 / kappa))
    return 1 + kappa - kappa / (1 + mu * s2 / kappa)


def spin_s2(rho, sig):
    # s for the spin-scaled density 2 rho_sigma.
    r = 2 * rho
    return 4 * sig / (4 * (3 * np.pi**2 * r) ** (2 / 3) * r * r)


def w_spin(rho, tau):
    t = CF * rho ** (5 / 3) / tau
    return (t - 1) / (t + 1)


def m08_x(p, a, b):
    ra, rb, ga, gb, ta, tb = p
    e = 0.0
    for rho, g, tau in [(ra, ga, ta), (rb, gb, tb)]:
        s2 = spin_s2(rho, g @ g)
        w = w_spin(rho, tau)
        fa = sum(c * w**i for i, c in enumerate(a))
        fb = sum(c * w**i for i, c in enumerate(b))
        e += CX * rho ** (4 / 3) * (pbe_like(s2, 0.804, 0.21951, False) * fa
                                    + pbe_like(s2, 0.804, 0.21951, True) * fb)
    return e


def pw92(ra, rb):
    def g(rs, A, a1, b1, b2, b3, b4):
        s = np.sqrt(rs)
        return -2 * A * (1 + a1 * rs) * np.log(1 + 1 / (2 * A * (b1 * s + b2 * rs + b3 * rs * s + b4 * rs * rs)))
    rho = ra + rb
    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)
    z = (ra - rb) / rho
    e0 = g(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = g(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    ac = -g(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    fz = ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    fpp0 = 8.0 / (9.0 * (2 ** (4 / 3) - 2))
    return e0 + ac * fz / fpp0 * (1 - z**4) + (e1 - e0) * fz * z**4, z, rs


def pbe_h(ra, rb, sig, eps, z, rs):
    beta, gamma = 0.06672455060314922, (1 - np.log(2)) / np.pi**2
    rho = ra + rb
    phi = 0.5 * ((1 + z) ** (2 / 3) + (1 - z) ** (2 / 3))
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    ks = np.sqrt(4 * kf / np.pi)
    t2 = sig / (4 * phi**2 * ks**2 * rho**2)
    A = beta / gamma / (np.exp(-eps / (gamma * phi**3)) - 1)
    return gamma * phi**3 * np.log(1 + beta / gamma * t2 * (1 + A * t2) / (1 + A * t2 + A * A * t2 * t2))


def m08_c(p, a, b, total_w):
    ra, rb, ga, gb, ta, tb = p
    g = ga + gb
    eps, z, rs = pw92(ra, rb)
    rho = ra + rb
    if total_w:
        t = 0.3 * (3 * np.pi**2) ** (2 / 3) * rho ** (5 / 3) / (ta + tb)
    else:
        t = CF * (ra ** (5 / 3) + rb ** (5 / 3)) / (ta + tb)
    w = (t - 1) / (t + 1)
    fa = sum(c * w**i for i, c in enumerate(a))
    fb = sum(c * w**i for i, c in enumerate(b))
    return rho * (eps * fa + pbe_h(ra, rb, g @ g, eps, z, rs) * fb)


def mn12_x(p, cc, order):
    ra, rb, ga, gb, ta, tb = p
    e = 0.0
    for rho, g, tau in [(ra, ga, ta), (rb, gb, tb)]:
        x2 = g @ g / rho ** (8 / 3)
        u = 0.004 * x2 / (1 + 0.004 * x2)
        v = 2.5 * rho ** (1 / 3) / (1 + 2.5 * rho ** (1 / 3))
        w = w_spin(rho, tau)
        var = {"u": u, "v": v, "w": w}
        f = 0.0
        for (i, j, k), c in cc.items():
            f += c * var[order[0]] ** i * var[order[1]] ** j * var[order[2]] ** k
        e += CX * rho ** (4 / 3) * f
    return e


def report(label, pts, ref, candidate):
    dev = max(abs(candidate(p) - r) / max(abs(r), 1e-12) for p, r in zip(pts, ref))
    print(f"  {label}: largest relative deviation {dev:.3e}")


def main():
    lib = load()
    pts = points()
    xp = params_of(lib, "HYB_MGGA_X_M08_HX")
    a = [xp[f"_a{i}"] for i in range(12)]
    b = [xp[f"_b{i}"] for i in range(12)]
    ref = [libxc_e("HYB_MGGA_X_M08_HX", p) for p in pts]
    print("M08-HX exchange")
    report("PBE f(w) + RPBE g(w)", pts, ref, lambda p: m08_x(p, a, b))
    cp = params_of(lib, "MGGA_C_M08_HX")
    a = [cp[f"_a{i}"] for i in range(12)]
    b = [cp[f"_b{i}"] for i in range(12)]
    ref = [libxc_e("MGGA_C_M08_HX", p) for p in pts]
    print("M08-HX correlation")
    report("w of the total density", pts, ref, lambda p: m08_c(p, a, b, True))
    report("w of the spin-resolved tau_LSDA", pts, ref, lambda p: m08_c(p, a, b, False))
    xp = params_of(lib, "MGGA_X_MN12_L")
    cc = {tuple(int(d) for d in k[3:]): v for k, v in xp.items() if k.startswith("_CC")}
    ref = [libxc_e("MGGA_X_MN12_L", p) for p in pts]
    print("MN12-L exchange, CC_ijk with variables (first, second, third)")
    for order in itertools.permutations("uvw"):
        report("".join(order), pts, ref, lambda p, o=order: mn12_x(p, cc, o))


if __name__ == "__main__":
    main()
