# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.

"""Independent Decimal oracle for PRL 107, 207402 (2011), Eq. (2).

Uses the published rational expression, quotient rule and density chain
rule, without importing the production evaluator. The extreme dilute
points need more than 200 digits to resolve cancellation in the raw
second derivative. Default precision is 500 decimal digits.

Print CSV to stdout, or pass --output FILE to save a reviewed fixture.
Only the Python standard library is needed.
"""
import argparse
from decimal import Decimal, localcontext
from pathlib import Path


def oracle_csv(precision=500):
    if precision < 400:
        raise ValueError("the dilute second-derivative oracle requires at least 400 decimal digits")
    with localcontext() as context:
        context.prec = precision
        d = Decimal
        pi = d("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")
        am, a0, a1, b1, b2, low = map(d, ["-0.260361", "-0.261762", "0.00375534",
                                          "0.113718", "0.0270912", "0.262005"])
        rows = []
        for ns, ps in [("0.02", "0.7"), ("0.2", "0.03"), ("1.3", "2.1"), ("5", "0.5"),
                       ("1e-200", "1e-100"), ("1e-280", "1e-200"), ("1e-300", "1e-200")]:
            n, p = d(ns), d(ps)
            r = (d(3) / (4 * pi * n)) ** (d(1) / 3)
            a = am / r + a0 + a1 * r - low * b2 * r * r
            da = -am / (r * r) + a1 - 2 * low * b2 * r
            d2a = 2 * am / (r * r * r) - 2 * low * b2
            b = 1 + b1 * r + b2 * r * r
            db = b1 + 2 * b2 * r
            value = a / b
            first = (da * b - a * db) / (b * b)
            second = (d2a / b - 2 * da * db / (b * b) - 2 * a * b2 / (b * b)
                      + 2 * a * db * db / (b * b * b))
            mixed = first * -r / (3 * n)
            own = p * (second * r * r + 4 * first * r) / (9 * n * n)
            values = [p * value, p * mixed, value, own, mixed]
            rows.append(",".join([ns, ps] + [f"{v:.35e}" for v in values]))
    return (f"# Independent Decimal({precision}) evaluation of PRL 107, 207402 (2011), Eq. (2).\n"
            "# Analytic quotient and chain rules; no production evaluator is used.\n"
            "electron,partner,energy,electron_potential,partner_potential,h_ee,h_ep\n"
            + "\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision", type=int, default=500)
    arguments = parser.parse_args()
    text = oracle_csv(arguments.precision)
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
