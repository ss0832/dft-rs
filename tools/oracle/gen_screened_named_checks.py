# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Independent H2/STO-3G energy oracle for screened and xDH recipes.

The H2 orbitals at 1.4 Bohr have fixed parity. Two-center s-Gaussian
integrals and cylindrical quadrature give their Hartree, screened exchange
and semilocal terms without an SCF solver. The unscaled opposite-spin PT2
term uses a population derivative of the quadrature and the Coulomb
transition integral. This program uses only the Python standard library.
"""

import argparse
import json
from math import cos, erf, exp, expm1, log, log1p, pi, sqrt


DISTANCE = 1.4
CENTERS = (-DISTANCE / 2, DISTANCE / 2)
EXPONENTS = (3.425250914, 0.6239137298, 0.1688554040)
CONTRACTION = (0.1543289673, 0.5353281423, 0.4446345422)
PW92 = (
    (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294),
    (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517),
)
FITS = {
    "wb97x-2-lp": (
        (0.251767, 1.57375, -5.26624, 6.74313),
        (1.15698, -3.31669, 6.27265, -4.51464),
        (0.553261, -1.16626, 6.84409, -8.90640),
    ),
    "wb97x-2-tqz": (
        (0.315503, 1.04772, -2.33506, 3.19909),
        (0.908460, -2.80936, 6.02676, -4.56981),
        (0.518198, -0.585956, 4.27080, -6.48897),
    ),
}
# T. M. Henderson, B. G. Janesko and G. E. Scuseria, JCP 128, 194105 (2008).
HJS_A = (0.0047333, 0.0403304, -0.0574615, 0.0435395, -0.0216251, 0.0063721)
HJS_B = (1.0, 8.52056, -13.9885, 9.28583, -3.27287, 0.843499,
         -0.235543, 0.0847074, -0.0171561, 0.0050552)


def orbital_pairs(left, right, primitive):
    for i, a in enumerate(CENTERS):
        for j, b in enumerate(CENTERS):
            for alpha, ca in zip(EXPONENTS, primitive):
                for beta, cb in zip(EXPONENTS, primitive):
                    p = alpha + beta
                    reduced = alpha * beta / p
                    center = (alpha * a + beta * b) / p
                    weight = left[i] * right[j] * ca * cb * exp(-reduced * (a - b) ** 2)
                    yield p, center, weight, reduced * (a - b) ** 2, reduced


def orbital_point(x, z, orbital, primitive):
    value = radial = axial = 0.0
    for coefficient, center in zip(orbital, CENTERS):
        for alpha, weight in zip(EXPONENTS, primitive):
            contribution = coefficient * weight * exp(-alpha * (x * x + (z - center) ** 2))
            value += contribution
            radial -= 2 * alpha * x * contribution
            axial -= 2 * alpha * (z - center) * contribution
    return value * value, 2 * value * radial, 2 * value * axial


def boys(t):
    if t < 1e-8:
        return 1 - t / 3 + t * t / 10
    return sqrt(pi / t) * erf(sqrt(t)) / 2


def one_electron(pairs):
    total = 0.0
    for p, center, weight, scaled_distance, reduced in pairs:
        kinetic = (pi / p) ** 1.5 * reduced * (3 - 2 * scaled_distance)
        attraction = -2 * pi * sum(boys(p * (center - atom) ** 2) for atom in CENTERS) / p
        total += weight * (kinetic + attraction)
    return total


def repulsion(left, right, omega=None):
    total = 0.0
    for p, x, w, _, _ in left:
        for q, z, v, _, _ in right:
            reduced = p * q / (p + q)
            factor = 1 if omega is None else omega / sqrt(omega * omega + reduced)
            total += (2 * pi ** 2.5 / (p * q * sqrt(p + q)) * w * v * factor
                      * boys(reduced * (factor * (x - z)) ** 2))
    return total


def quadrature(order, boundary):
    for j in range(order):
        root = cos(pi * (j + 0.75) / (order + 0.5))
        for _ in range(40):
            previous, value = 0.0, 1.0
            for degree in range(1, order + 1):
                previous, value = value, ((2 * degree - 1) * root * value
                                          - (degree - 1) * previous) / degree
            slope = order * (root * value - previous) / (root * root - 1)
            correction = value / slope
            if abs(correction) < 6e-16:
                break
            root -= correction
        else:
            raise ArithmeticError("Gauss-Legendre root did not converge")
        yield (root + 1) * boundary / 2, boundary / ((1 - root * root) * slope * slope)


def lda_c(density, polarized, modified=False):
    a, a1, b1, b2, b3, b4 = PW92[int(polarized)]
    if modified:
        a = 0.01554535 if polarized else 0.0310907
    rs = (3 / (4 * pi * density)) ** (1 / 3)
    denominator = 2 * a * (b1 * sqrt(rs) + b2 * rs + b3 * rs ** 1.5 + b4 * rs * rs)
    return -2 * a * (1 + a1 * rs) * log1p(1 / denominator)


def pbe_terms(rho, sigma, kf, local_exchange, solid):
    mu, beta = (10 / 81, 0.046) if solid else (0.2195149727645171, 0.06672455060314922)
    s = sqrt(sigma) / (2 * rho * kf)
    exchange = 2 * local_exchange * (1.804 - 0.804 / (1 + mu * s * s / 0.804))
    local = lda_c(2 * rho, False, modified=True)
    gamma = (1 - log(2)) / (pi * pi)
    q = sigma / (4 * rho * rho * (4 * kf / pi))
    u = (beta / gamma) * q / expm1(-local / gamma)
    correlation = 2 * rho * (local + gamma * log1p((beta / gamma) * q * (1 + u) / (1 + u + u * u)))
    return exchange, correlation


def pbesol_hole(s, nu):
    a, b, c, d = 0.757211, -0.106364, -0.118649, 0.609650
    z = sum(x * s ** (i + 4) for i, x in enumerate(HJS_A)) / sum(x * s ** i for i, x in enumerate(HJS_B))
    eta, lam = a + z, d + z
    f = 1 - s * s / (27 * c * (1 + s * s / 4)) - z / (2 * c)
    g = (-2 / 5 * c * f * lam - 4 / 15 * b * lam ** 2 - 6 / 5 * a * lam ** 3
         - 4 / 5 * sqrt(pi) * lam ** 3.5 - 12 / 5 * lam ** 3.5 * (sqrt(z) - sqrt(eta)))
    w, v, u = sqrt(nu * nu + lam), sqrt(nu * nu + z), sqrt(nu * nu + eta)
    chi = nu / w
    return (a - 4 / 9 * b / lam * (1 - chi) - 4 / 9 * c * f / lam ** 2 * (1 - 3 / 2 * chi + chi ** 3 / 2)
            - 8 / 9 * g / lam ** 3 * (1 - 15 / 8 * chi + 5 / 4 * chi ** 3 - 3 / 8 * chi ** 5)
            - 2 * nu * a / (v + u) + 2 * z * log((nu + v) / (nu + w)) - 2 * eta * log((nu + u) / (nu + w)))


def xc_terms(model, rho, dx, dz):
    if rho < 1e-26:
        return (0.0, 0.0) if model in ("pbe0", "xdh-pbe0") else (0.0, 0.0, 0.0)
    sigma = dx * dx + dz * dz
    kf = (6 * pi * pi * rho) ** (1 / 3)
    exchange = -0.75 * (6 / pi) ** (1 / 3) * rho ** (4 / 3)
    if model == "hsesol":
        x, c = pbe_terms(rho, sigma, kf, exchange, solid=True)
        s = sqrt(sigma) / (2 * rho * kf)
        return x, -exchange / 2 * pbesol_hole(s, 0.11 / kf), c
    if model in ("pbe0", "xdh-pbe0"):
        x, c = pbe_terms(rho, sigma, kf, exchange, solid=False)
        return (0.75 * x, c) if model == "pbe0" else (0.1665 * x, 0.5292 * c)

    a = 0.3 / (2 * kf)
    t = 1 / (2 * a)
    if t < 0.1:
        attenuation = t * t / 9 - t ** 4 / 60 + t ** 6 / 420 - t ** 8 / 3240
    else:
        attenuation = 1 - 8 / 3 * a * (sqrt(pi) * erf(t) - 3 * a + 4 * a ** 3
                                       + (2 * a - 4 * a ** 3) * exp(-t * t))
    same = 2 * rho * lda_c(rho, True)
    other = 2 * rho * lda_c(2 * rho, False) - same
    reduced_sigma = sigma / rho ** (8 / 3)
    values = []
    for coeffs, gamma, base in zip(FITS[model], (0.004, 0.2, 0.006),
                                   (2 * exchange * attenuation, same, other)):
        u = gamma * reduced_sigma / (1 + gamma * reduced_sigma)
        values.append(base * sum(coef * u ** i for i, coef in enumerate(coeffs)))
    return values


def main(order):
    primitive = tuple(c * (2 * a / pi) ** 0.75 for c, a in zip(CONTRACTION, EXPONENTS))
    norm = sum(w * (pi / p) ** 1.5 for p, _, w, _, _ in orbital_pairs((1, 0), (1, 0), primitive))
    primitive = tuple(c / sqrt(norm) for c in primitive)
    overlap = sum(w * (pi / p) ** 1.5 for p, _, w, _, _ in orbital_pairs((1, 0), (0, 1), primitive))
    g = tuple(c / sqrt(2 * (1 + overlap)) for c in (1, 1))
    u = tuple(c / sqrt(2 * (1 - overlap)) for c in (1, -1))
    gg = list(orbital_pairs(g, g, primitive))
    gu = list(orbital_pairs(g, u, primitive))
    uu = list(orbital_pairs(u, u, primitive))
    hg, hu = one_electron(gg), one_electron(uu)
    gggg, uugg, gugu = repulsion(gg, gg), repulsion(uu, gg), repulsion(gu, gu)
    hartree, core = 2 * gggg, 2 * hg + 2 * gggg + 1 / DISTANCE
    nodes = list(quadrature(order, 13.0))
    geometry = [(4 * pi * x * wx * wz, orbital_point(x, z, g, primitive), orbital_point(x, z, u, primitive))
                for x, wx in nodes for z, wz in nodes]

    def integrate(model, step=0.0, virtual=False):
        output = [0.0] * (2 if model in ("pbe0", "xdh-pbe0") else 3)
        for weight, occupied, unoccupied in geometry:
            variables = (a + step * b for a, b in zip(occupied, unoccupied if virtual else occupied))
            for i, value in enumerate(xc_terms(model, *variables)):
                output[i] += weight * value
        return output

    def potential(model, virtual):
        h = 1e-4
        e = [sum(integrate(model, j * h, virtual)) for j in (2, 1, -1, -2)]
        return (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (24 * h)

    reports = {}
    for model, cx, lr, omega, os in (
        ("hsesol", 0.25, -0.25, 0.11, 0),
        ("wb97x-2-lp", 0.678792, 1 - 0.678792, 0.3, 0.477992),
        ("wb97x-2-tqz", 0.636158, 1 - 0.636158, 0.3, 0.447105),
        ("pbe0", 0.25, 0.0, 0.0, 0.5428),
    ):
        exchange = -cx * gggg - lr * repulsion(gg, gg, omega)
        local = integrate(model)
        parent = core + exchange + sum(local)
        scf = hg + 2 * gggg + exchange + potential(model, False)
        virtual = hu + 2 * uugg - cx * gugu - lr * repulsion(gu, gu, omega) + potential(model, True)
        opposite_spin = gugu * gugu / (2 * (scf - virtual)) if os else 0.0
        target = parent
        if model == "pbe0":
            model = "xdh-pbe0"
            local = integrate(model)
            exchange = -0.8335 * gggg
            target = core + exchange + sum(local)
        reports[model] = dict(parent=parent, target=target, energy=target + os * opposite_spin,
                              same_spin_alpha=0.0, same_spin_beta=0.0, opposite_spin=opposite_spin,
                              one_electron=2 * hg, hartree=hartree, exact_exchange=exchange, semilocal=local)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=240)
    main(parser.parse_args().order)
