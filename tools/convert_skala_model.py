# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Convert a Skala checkpoint into the model file read by ``dft_rs::skala``.

Usage::

    python tools/convert_skala_model.py OUTPUT.model --functional skala-1.1
    python tools/convert_skala_model.py OUTPUT.model --checkpoint FILE.fun

The script needs the official ``skala`` Python package with its dependencies
(``torch`` and ``e3nn``); nothing of them is needed to evaluate the converted
file. No checkpoint is part of this repository.

The conversion is literal: perceptron weights, radial temperatures and the
coupling tensors of the symmetric contraction are copied, and the real
spherical harmonics are stored as polynomial coefficients obtained from
``e3nn`` itself, so no convention is derived a second time. Scalars that are
code constants of the official implementation (feature epsilon, distance
epsilon, sigmoid scale, LDA exchange prefactor) can be set on the command
line. Every size of the network is taken from the shapes of the checkpoint.

Status: this script has not been run in the environment in which dft-rs is
developed. Read the printed summary and compare an energy with the official
implementation before relying on a converted file.
"""
import argparse
import re
import struct

import numpy as np

MAGIC = b"DFTRS-SKALA\0"
VERSION = 1


def load_state(arguments):
    """Tensors of the checkpoint as float64 arrays, without a common prefix."""
    import torch

    if arguments.checkpoint:
        module = torch.jit.load(arguments.checkpoint, map_location="cpu")
    else:
        from skala.functional import load_functional

        module = load_functional(arguments.functional)
    state = {
        key: value.detach().cpu().double().numpy()
        for key, value in module.state_dict().items()
    }
    anchor = "input_model.0.weight"
    prefixes = [key[: -len(anchor)] for key in state if key.endswith(anchor)]
    if len(prefixes) != 1:
        raise SystemExit(
            f"cannot locate `{anchor}` among {len(state)} tensors; first names: {sorted(state)[:20]}"
        )
    prefix = prefixes[0]
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def numbered(state, stem):
    """Linear layers ``stem.N.weight`` in the order of N."""
    pattern = re.compile(re.escape(stem) + r"\.(\d+)\.weight$")
    indices = sorted(int(match.group(1)) for match in map(pattern.match, state) if match)
    if not indices:
        raise SystemExit(f"no linear layer `{stem}.N.weight` in the checkpoint")
    return [(state[f"{stem}.{i}.weight"], state[f"{stem}.{i}.bias"]) for i in indices]


def harmonic_polynomials(lmax):
    """Coefficients of the real harmonics in the monomials x^a y^b z^c,
    a + b + c = l, with a descending and then b descending."""
    import torch
    from e3nn import o3

    generator = np.random.default_rng(0)
    blocks = []
    for l in range(lmax + 1):
        exponents = [(a, b, l - a - b) for a in range(l, -1, -1) for b in range(l - a, -1, -1)]
        points = generator.normal(size=(8 * len(exponents) + 8, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        values = o3.spherical_harmonics(
            l, torch.tensor(points, dtype=torch.float64), normalize=False, normalization="norm"
        ).numpy()
        design = np.stack(
            [points[:, 0] ** a * points[:, 1] ** b * points[:, 2] ** c for a, b, c in exponents],
            axis=1,
        )
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = float(np.abs(design @ coefficients - values).max())
        if residual > 1e-10:
            raise SystemExit(f"harmonics of l = {l} are not polynomials of degree {l}: residual {residual:.3e}")
        blocks.append(coefficients.T.copy())
    return blocks


def convert(state, arguments):
    tensors = {}

    def put(name, array):
        tensors[name] = np.ascontiguousarray(np.asarray(array, dtype=np.float64))

    for i, (weight, bias) in enumerate(numbered(state, "input_model")):
        put(f"input.{i}.weight", weight)
        put(f"input.{i}.bias", bias)
    for i, (weight, bias) in enumerate(numbered(state, "output_model")):
        put(f"output.{i}.weight", weight)
        put(f"output.{i}.bias", bias)
    put("radial.temperatures", state["radial_basis.temps"])
    layers = sorted(
        {int(match.group(1)) for match in (re.match(r"non_local_layers\.(\d+)\.", key) for key in state) if match}
    )
    if not layers:
        raise SystemExit("the checkpoint has no non-local layer")
    shells = None
    correlation = None
    for n in layers:
        base = f"non_local_layers.{n}."
        down = state[base + "tp_down._batched_W"]
        up = state[base + "tp_up._batched_W"]
        shells = down.shape[0]
        nsph = shells * shells
        for ours, theirs in (("pre_down", "pre_down_layer.0"), ("post_up", "post_up_layer.0"), ("concat", "concat_layer.0")):
            put(f"nonlocal.{n}.{ours}.weight", state[base + theirs + ".weight"])
            put(f"nonlocal.{n}.{ours}.bias", state[base + theirs + ".bias"])
        for l in range(shells):
            put(f"nonlocal.{n}.down.{l}", down[l])
            put(f"nonlocal.{n}.up.{l}", up[l])
            put(f"nonlocal.{n}.coarse.{l}", state[base + f"coarse_linear.weight_{l}_{l}"])
            contraction = base + f"symmetric_product.contractions.{l}."
            pattern = re.compile(re.escape(contraction) + r"u_matrix_(\d+)$")
            orders = sorted(int(match.group(1)) for match in map(pattern.match, state) if match)
            if not orders:
                raise SystemExit(f"no coupling tensor `{contraction}u_matrix_N`")
            correlation = max(orders)
            for nu in orders:
                coupling = state[contraction + f"u_matrix_{nu}"]
                block = (2 * l + 1) * nsph**nu
                if coupling.size % block:
                    raise SystemExit(f"`{contraction}u_matrix_{nu}` has the unexpected shape {coupling.shape}")
                paths = coupling.size // block
                put(
                    f"nonlocal.{n}.contraction.{l}.coupling.{nu}",
                    coupling.reshape([2 * l + 1] + [nsph] * nu + [paths]),
                )
                # The weights of the highest order are `weights_max`; `weights.j`
                # belongs to the order `correlation - 1 - j`.
                key = contraction + ("weights_max" if nu == correlation else f"weights.{correlation - 1 - nu}")
                put(f"nonlocal.{n}.contraction.{l}.weights.{nu}", state[key])
    for l, block in enumerate(harmonic_polynomials(shells - 1)):
        put(f"harmonics.{l}", block)
    put("feature_epsilon", [arguments.feature_epsilon])
    put("distance_epsilon", [arguments.distance_epsilon])
    put("sigmoid_scale", [arguments.sigmoid_scale])
    put("lda_prefactor", [arguments.lda_prefactor])
    summary = dict(
        features=tensors["input.0.weight"].shape[0],
        input_layers=len(numbered(state, "input_model")),
        output_layers=len(numbered(state, "output_model")),
        non_local_layers=len(layers),
        channels=int(tensors["radial.temperatures"].shape[0]),
        lmax=shells - 1,
        correlation=correlation,
        parameters=int(sum(array.size for name, array in tensors.items() if ".coupling." not in name)),
    )
    return tensors, summary


def write(path, tensors):
    with open(path, "wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<II", VERSION, len(tensors)))
        for name in sorted(tensors):
            array = tensors[name]
            encoded = name.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
            stream.write(struct.pack("<I", array.ndim))
            stream.write(struct.pack(f"<{array.ndim}Q", *array.shape))
            stream.write(array.astype("<f8").tobytes(order="C"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="model file to write")
    parser.add_argument("--functional", default="skala-1.1", help="identifier of an official functional")
    parser.add_argument("--checkpoint", help="path of a TorchScript checkpoint instead of an identifier")
    parser.add_argument("--feature-epsilon", type=float, default=1e-5)
    parser.add_argument("--distance-epsilon", type=float, default=1e-20)
    parser.add_argument("--sigmoid-scale", type=float, default=2.0)
    parser.add_argument("--lda-prefactor", type=float, default=-0.9305257363491001)
    arguments = parser.parse_args()
    tensors, summary = convert(load_state(arguments), arguments)
    write(arguments.output, tensors)
    print(f"wrote {arguments.output}: {len(tensors)} tensors, {summary}")


if __name__ == "__main__":
    main()
