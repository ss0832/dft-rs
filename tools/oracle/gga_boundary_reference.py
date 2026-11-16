# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Narrow, independently derived physical limits for the approved point10 FXC.

Libxc 7.0.0 generated log(x+sqrt(1+x*x)) loses the asinh value near its
sigma floor. B88 E_sigma_sigma(0+) = 12 beta**2/rho_spin**4. ITYH follows
G(E)=E*R(a), a proportional sqrt(-E): G'=R+a*R'/2 and
G''=a*(3*R'+a*R'')/(4*E). LYP is affine in sigma.
120-digit derivations, native API reproduction and isolated stabilized C
replays are archived in gga-zero-boundary-root-cause-20261031. Application
and explicit fourteen-entry approval are in gga-approved-fourteen-reference-20261031.
Only the two listed diagonal values at the exact approved point can change.
Unexpected reference outputs raise an error rather than being overwritten.
"""

from math import isclose

POINT = [0.31, 0.22, 0.0, 0.0, 0.0, 0.37, 0.21]
EDF2 = {
    "GGA_X_B88,": [(0.01432560818866058, 0.022920973101856916),
                    (0.05647667509049932, 0.09036268014479885)],
    "REFERENCE_EDF2_B88": [(0.009948339019903182, 0.01591734243184508),
                           (0.0392199132572912, 0.06275186121166587)],
    "HYB_GGA_XC_EDF2": [(0.008372402739082279, 0.013395844382531637),
                         (0.033007008358377186, 0.05281121337340346)],
}
SCREENED = {
    0.30: [(0.009595636447841664, 0.015167652254889277),
           (0.035904814349904834, 0.056661036170865646)],
    0.27: [(0.010012326079839843, 0.015847358983318648),
           (0.03768405469221927, 0.059559325443062514)],
}
# Preserve the independently rounded weighted limits, not a second rounding
# of binary64 primitive values.
PARENTS = {
    0.30: [(0.004509949130485582, 0.00712879655979796),
           (0.01687526274445527, 0.026630687000306852)],
    0.27: [(0.003504314127943945, 0.005546575644161527),
           (0.013189419142276743, 0.020845763905071882)],
}


def _apply(result, pairs):
    for axis, (raw, limit) in zip([2, 4], pairs):
        actual = result["kernel"][axis][axis]
        # A future reference that already evaluates the correct limit needs
        # no numerical correction. Reject any third, unexplained value.
        if isclose(actual, limit, rel_tol=0.0, abs_tol=1e-14):
            result["kernel"][axis][axis] = limit
        elif isclose(actual, raw, rel_tol=0.0, abs_tol=1e-16):
            result["kernel"][axis][axis] = limit
        else:
            raise ValueError(f"Unrecognized point10 diagonal FXC: {actual}, raw={raw}, limit={limit}")


def correct_edf2_boundary(code, point, result):
    if list(point) == POINT and code in EDF2:
        _apply(result, EDF2[code])


def correct_wb2_boundary(code, point, omega, result):
    if list(point) != POINT or omega not in SCREENED:
        return
    if code == "GGA_X_ITYH,":
        _apply(result, SCREENED[omega])
    elif ((omega == 0.30 and code == "0.47*GGA_X_ITYH,0.73*GGA_C_LYP")
          or (omega == 0.27 and code == "0.35*GGA_X_ITYH,0.64*GGA_C_LYP")):
        _apply(result, PARENTS[omega])
