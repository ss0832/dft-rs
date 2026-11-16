# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
# Throwaway probe: replicate the dft-rs VWN formula in Python and compare
# against libxc at a spin-polarized point.
import numpy as np
from pyscf.dft import libxc

ra, rb = 0.3, 0.1
rho = ra + rb
rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
x = np.sqrt(rs)
z = (ra - rb) / rho

def fit(x, A, x0, b, c):
    X = x * x + b * x + c
    X0 = x0 * x0 + b * x0 + c
    Q = np.sqrt(4 * c - b * b)
    at = np.arctan(Q / (2 * x + b))
    return A * (
        np.log(x * x / X)
        + 2 * b / Q * at
        - b * x0 / X0 * (np.log((x - x0) ** 2 / X) + 2 * (b + 2 * x0) / Q * at)
    )

def fz(z):
    return ((1 + z) ** (4 / 3) + (1 - z) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)

VWN5 = dict(
    P=(0.0310907, -0.10498, 3.72744, 12.9352),
    F=(0.01554535, -0.32500, 7.06042, 18.0578),
    A=(-1 / (6 * np.pi**2), -0.0047584, 1.13107, 13.0045),
)
VWN3 = dict(
    P=(0.0310907, -0.409286, 13.0720, 42.7198),
    F=(0.01554535, -0.743294, 20.1231, 101.578),
    A=(-1 / (6 * np.pi**2), -0.228344, 1.06835, 11.4813),
)

fpp0 = 4.0 / (9.0 * (2.0 ** (1.0 / 3.0) - 1.0))
for name, prm in [("vwn5", VWN5), ("vwn3", VWN3)]:
    eP = fit(x, *prm["P"])
    eF = fit(x, *prm["F"])
    aC = fit(x, *prm["A"])
    z4 = z**4
    ec_full = eP + aC * fz(z) / fpp0 * (1 - z4) + (eF - eP) * fz(z) * z4
    ec_simple = eP + (eF - eP) * fz(z)
    print(f"{name}: full-interp e = {rho * ec_full:.12f}   simple-interp e = {rho * ec_simple:.12f}")

for name in ["lda_c_vwn", "lda_c_vwn_rpa", "lda_c_vwn_3"]:
    exc = libxc.eval_xc(name, np.array([[ra], [rb]]), spin=1)[0]
    print(f"libxc {name:14s} e = {exc[0] * rho:.12f}")
