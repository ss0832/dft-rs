# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""CUBE, SVG and browser visualization of affine IGMH reports."""

from __future__ import annotations

from array import array
from base64 import b64encode
from dataclasses import dataclass
from itertools import chain, product
import json
from math import dist, floor, hypot, isclose, isfinite, pi, prod, sqrt
from numbers import Integral
from pathlib import Path
from sys import byteorder
from typing import Sequence

from .units import ANGSTROM_IN_BOHR

__all__ = ["write_igmh_cubes", "write_igmh_slice", "write_igmh_viewer"]

_FIELDS = {
    "total": "igmh_delta_g",
    "inter": "igmh_delta_g_inter",
    "intra": "igmh_delta_g_intra",
}


def _length_scale(unit):
    if unit == "bohr":
        return 1.0
    if unit == "angstrom":
        return ANGSTROM_IN_BOHR
    raise ValueError("IGMH visualization requires bohr or angstrom coordinates")


def _xyz(row, scale):
    point = tuple(float(x) * scale for x in row)
    if len(point) != 3 or not all(map(isfinite, point)):
        raise ValueError("IGMH visualization requires finite Cartesian triples")
    return point


@dataclass
class _Volume:
    shape: tuple[int, int, int]
    strides: tuple[int, int, int]
    points: Sequence
    delta_g: Sequence
    signed_density: Sequence
    length_scale: float

    def point(self, index):
        return _xyz(self.points[index], self.length_scale)

    def step(self, axis):
        if self.shape[axis] < 2:
            raise ValueError(
                "the grid step of a one-point axis must be supplied explicitly"
            )
        return tuple(
            a - b for a, b in zip(self.point(self.strides[axis]), self.point(0))
        )


def _volume(report, field):
    if field not in _FIELDS:
        raise ValueError("IGMH visualization field must be inter, intra or total")
    key = _FIELDS[field]
    try:
        shape = tuple(report["igmh_grid_shape"])
        points = report["igmh_points"]
        delta_g = report[key]
        signed = report["igmh_signed_density"]
        units = report["units"]
        length_unit = units["igmh_points"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "IGMH visualization requires an affine-grid IGMH report"
        ) from error
    if len(shape) != 3 or any(
        isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in shape
    ):
        raise ValueError("IGMH grid shape must contain three positive integers")
    nx, ny, nz = map(int, shape)
    size = prod((nx, ny, nz))
    if any(len(array) != size for array in (points, delta_g, signed)):
        raise ValueError("IGMH field and point counts must match the grid shape")
    if (
        units.get(key) != f"1/{length_unit}^4"
        or units.get("igmh_signed_density") != f"1/{length_unit}^3"
    ):
        raise ValueError(
            "IGMH density and gradient units must match the grid coordinates"
        )
    scale = _length_scale(length_unit)
    if any(not isfinite(x) for array in (delta_g, signed) for x in array):
        raise ValueError("IGMH visualization fields must be finite")
    return _Volume((nx, ny, nz), (ny * nz, nz, 1), points, delta_g, signed, scale)


def _cube_frame(volume, grid):
    if grid is None:
        origin = volume.point(0)
        steps = tuple(volume.step(axis) for axis in range(3))
    else:
        vectors = tuple(_xyz(row, volume.length_scale) for row in grid)
        if len(vectors) != 4:
            raise ValueError("grid must supply the origin and three IGMH grid steps")
        origin, *steps = vectors
    u, v, w = steps
    determinant = sum(
        u[a] * (v[(a + 1) % 3] * w[(a + 2) % 3] - v[(a + 2) % 3] * w[(a + 1) % 3])
        for a in range(3)
    )
    if not isfinite(determinant) or determinant == 0.0:
        raise ValueError("IGMH cube grid steps must span three dimensions")
    components = tuple(zip(origin, zip(*steps)))
    for flat, indices in enumerate(product(*(range(n) for n in volume.shape))):
        predicted = tuple(
            o + sum(i * d for i, d in zip(indices, ds)) for o, ds in components
        )
        if not all(
            isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)
            for a, b in zip(predicted, volume.point(flat))
        ):
            raise ValueError(
                "IGMH sample points do not match the affine cube grid in C order"
            )
    return origin, steps


def _write_cube(path, description, origin, steps, shape, geometry, values, scale):
    def vector(row):
        return " ".join(f"{coordinate:.12e}" for coordinate in row)

    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{description}\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        stream.write(f"{len(geometry)} {vector(origin)}\n")
        for count, step in zip(shape, steps):
            stream.write(f"{count} {vector(step)}\n")
        for z, point in geometry:
            stream.write(f"{z} {float(z):.1f} {vector(point)}\n")
        for start in range(0, len(values), 6):
            stream.write(
                " ".join(f"{value * scale:.12e}" for value in values[start : start + 6])
                + "\n"
            )


def _geometry(numbers, positions, position_unit):
    numbers = tuple(numbers)
    scale = _length_scale(position_unit)
    positions = tuple(_xyz(point, scale) for point in positions)
    if (
        not numbers
        or len(numbers) != len(positions)
        or any(
            isinstance(z, bool) or not isinstance(z, Integral) or not 1 <= z <= 118
            for z in numbers
        )
    ):
        raise ValueError("visualization requires one valid atomic number per position")
    return list(zip(map(int, numbers), positions))


def write_igmh_cubes(
    report,
    prefix,
    numbers,
    positions,
    *,
    field="inter",
    position_unit="bohr",
    grid=None,
):
    """Write matching interaction and signed-density CUBE files. Return their paths.

    Use a report obtained with igmh_grid and igmh_grid_shape. The optional
    grid argument repeats the original origin and step vectors, in the report's
    coordinate units. Supply it when any grid axis has one sample.

    The two filenames append -delta-g-{field}.cube and -signed-density.cube
    to prefix. Draw an isosurface of the first file and colour it with the
    second. The exported positions and fields use atomic units. Supply the
    molecular geometry in Bohr or set position_unit="angstrom", for example
    for positions from an ASE Atoms object. Choose field="inter", "intra" or
    "total" for the corresponding delta-g field.
    """
    volume = _volume(report, field)
    origin, steps = _cube_frame(volume, grid)
    geometry = _geometry(numbers, positions, position_unit)
    prefix = Path(prefix)
    interaction = prefix.with_name(f"{prefix.name}-delta-g-{field}.cube")
    signed = prefix.with_name(f"{prefix.name}-signed-density.cube")
    _write_cube(
        interaction,
        f"IGMH {field} delta-g (electrons/bohr^4)",
        origin,
        steps,
        volume.shape,
        geometry,
        volume.delta_g,
        volume.length_scale**-4,
    )
    _write_cube(
        signed,
        "IGMH sign(lambda_2) rho (electrons/bohr^3)",
        origin,
        steps,
        volume.shape,
        geometry,
        volume.signed_density,
        volume.length_scale**-3,
    )
    return interaction, signed


def _color(value):
    negative, middle, positive = (48, 89, 179), (87, 157, 101), (195, 77, 66)
    end = positive if value >= 0.0 else negative
    weight = min(abs(value), 1.0)
    return "#" + "".join(
        f"{round(a + weight * (b - a)):02x}" for a, b in zip(middle, end)
    )


def write_igmh_slice(
    report, path, *, axis=2, index=None, field="inter", isovalue=0.005, color_limit=0.05
):
    """Write and return a standalone SVG of a two-dimensional IGMH grid slice.

    Axis and index select the fixed grid coordinate, counting from zero.
    The default index selects the middle sample of that axis. Sample cells
    at or above the delta-g isovalue use a blue to green to red signed-density
    scale. Isovalue and color_limit use electrons/Bohr^4 and electrons/Bohr^3,
    respectively, also when the input report uses ASE units. The preview
    uses sample-cell shading on the selected slice.
    """
    volume = _volume(report, field)
    if isinstance(axis, bool) or not isinstance(axis, Integral) or axis not in range(3):
        raise ValueError("IGMH slice axis must be 0, 1 or 2")
    if index is None:
        index = volume.shape[axis] // 2
    if (
        isinstance(index, bool)
        or not isinstance(index, Integral)
        or not 0 <= index < volume.shape[axis]
    ):
        raise ValueError("IGMH slice index is outside its grid axis")
    if (
        not isfinite(isovalue)
        or isovalue < 0.0
        or not isfinite(color_limit)
        or color_limit <= 0.0
    ):
        raise ValueError(
            "IGMH slice requires a nonnegative isovalue and a positive colour limit"
        )
    a, b = (i for i in range(3) if i != axis)
    na, nb = volume.shape[a], volume.shape[b]
    u, v = volume.step(a), volume.step(b)
    along = sqrt(sum(x * x for x in u))
    if along == 0.0:
        raise ValueError("IGMH slice grid steps must span a plane")
    projection = sum(x * y for x, y in zip(u, v)) / along
    perpendicular = sqrt(sum((y - projection * x / along) ** 2 for x, y in zip(u, v)))
    if perpendicular == 0.0:
        raise ValueError("IGMH slice grid steps must span a plane")
    width = na * along + nb * abs(projection)
    magnify = min(620.0 / width, 390.0 / (nb * perpendicular))
    left = min(0.0, nb * projection)
    plot_left, plot_top = 30.0, 62.0
    height = plot_top + nb * perpendicular * magnify + 65.0

    def projected(i, j):
        px = plot_left + (i * along + j * projection - left) * magnify
        py = plot_top + ((nb - j) * perpendicular) * magnify
        return f"{px:.4f},{py:.4f}"

    def polygon(i, j, color):
        points = " ".join(
            projected(i + di, j + dj) for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))
        )
        return f'<polygon points="{points}" fill="{color}"/>\n'

    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="680" height="{height:.2f}" '
            f'viewBox="0 0 680 {height:.2f}" role="img" aria-label="IGMH slice">\n'
        )
        stream.write('<rect width="100%" height="100%" fill="#fff"/>\n')
        stream.write('<g font-family="sans-serif" fill="#20242a">\n')
        stream.write(
            f'<text x="30" y="30" font-size="20">IGMH {field} slice</text>\n'
            f'<text x="30" y="49" font-size="13">Grid axis {axis}, sample {index} '
            f"· Δg ≥ {isovalue:.5g} electrons/Bohr⁴</text></g>\n"
        )
        border = " ".join(
            projected(i, j) for i, j in ((0, 0), (na, 0), (na, nb), (0, nb))
        )
        stream.write(f'<polygon points="{border}" fill="#f4f5f6" stroke="#a4a9af"/>\n')
        delta_scale = volume.length_scale**-4
        density_scale = volume.length_scale**-3 / color_limit
        for i, j in product(range(na), range(nb)):
            flat = (
                index * volume.strides[axis]
                + i * volume.strides[a]
                + j * volume.strides[b]
            )
            if volume.delta_g[flat] * delta_scale >= isovalue:
                stream.write(
                    polygon(i, j, _color(volume.signed_density[flat] * density_scale))
                )
        y = height - 30.0
        stream.write(
            '<defs><linearGradient id="signed"><stop stop-color="#3059b3"/>'
            '<stop offset="50%" stop-color="#579d65"/><stop offset="100%" stop-color="#c34d42"/>'
            "</linearGradient></defs>\n"
        )
        stream.write(
            f'<rect x="30" y="{y - 15:.2f}" width="210" height="10" fill="url(#signed)"/>\n'
        )
        stream.write(
            f'<text x="30" y="{y + 9:.2f}" font-family="sans-serif" font-size="13" fill="#20242a">'
            f"sign(λ₂)ρ from −{color_limit:.5g} to +{color_limit:.5g} electrons/Bohr³</text>\n</svg>\n"
        )
    return path


# The six tetrahedra share a body diagonal. Adjacent grid cells use the
# same triangulation of each shared face, including on a skew affine grid.
_TETRAHEDRA = (
    (0, 4, 6, 7),
    (0, 6, 2, 7),
    (0, 2, 3, 7),
    (0, 3, 1, 7),
    (0, 1, 5, 7),
    (0, 5, 4, 7),
)


def _isosurface(volume, density, isovalue):
    _cube_frame(volume, None)
    delta = volume.delta_g
    level = isovalue * volume.length_scale**4
    vertices, faces, edges = [], [], {}
    color_scale = volume.length_scale**-3

    def crossing(a, b):
        if delta[a] == level:
            b = a
        elif delta[b] == level:
            a = b
        key = (min(a, b), max(a, b))
        if key in edges:
            return edges[key]
        pa, pb = volume.point(a), volume.point(b)
        t = 0.0 if a == b else (level - delta[a]) / (delta[b] - delta[a])
        position = [x + t * (y - x) for x, y in zip(pa, pb)]
        da, db = volume.signed_density[a], volume.signed_density[b]
        # Interpolate signed colour and unsigned density at the same crossing.
        ra, rb = density[a], density[b]
        edges[key] = len(vertices)
        vertices.append(
            [
                *position,
                (da + t * (db - da)) * color_scale,
                (ra + t * (rb - ra)) * color_scale,
            ]
        )
        return edges[key]

    def triangle(a, b, c, outside):
        if a == b or b == c or a == c:
            return
        p, q, r = vertices[a], vertices[b], vertices[c]
        u = [q[i] - p[i] for i in range(3)]
        v = [r[i] - p[i] for i in range(3)]
        n = [
            u[(i + 1) % 3] * v[(i + 2) % 3] - u[(i + 2) % 3] * v[(i + 1) % 3]
            for i in range(3)
        ]
        if any(n):
            toward_outside = sum(d * (x - y) for d, x, y in zip(n, outside, p))
            faces.append([a, b, c] if toward_outside >= 0 else [a, c, b])

    di, dj, dk = volume.strides
    offsets = [x * di + y * dj + z * dk for x, y, z in product(range(2), repeat=3)]
    for i, j, k in product(*(range(n - 1) for n in volume.shape)):
        base = i * di + j * dj + k * dk
        ids = [base + offset for offset in offsets]
        selected = [delta[index] >= level for index in ids]
        if all(selected) or not any(selected):
            continue
        for tetra in _TETRAHEDRA:
            inside = [ids[a] for a in tetra if selected[a]]
            outside = [ids[a] for a in tetra if not selected[a]]
            if not inside or not outside:
                continue
            edge = volume.point(outside[0])
            if len(inside) == 2:
                a, b = (crossing(inside[0], x) for x in outside)
                d, c = (crossing(inside[1], x) for x in outside)
                triangle(a, b, c, edge)
                triangle(a, c, d, edge)
            else:
                sole, others = (
                    (inside, outside) if len(inside) == 1 else (outside, inside)
                )
                a, b, c = (crossing(sole[0], x) for x in others)
                triangle(a, b, c, edge)
    return vertices, faces


# Display links use covalent radii in Angstrom for common nonmetals.
# Callers can supply exact bond indices for any element or bonding model.
_RADII = {
    1: ("H", 0.31),
    5: ("B", 0.84),
    6: ("C", 0.76),
    7: ("N", 0.71),
    8: ("O", 0.66),
    9: ("F", 0.57),
    14: ("Si", 1.11),
    15: ("P", 1.07),
    16: ("S", 1.05),
    17: ("Cl", 1.02),
    35: ("Br", 1.20),
    53: ("I", 1.39),
}


def _viewer_bonds(geometry, fragments, bonds):
    if bonds is not None:
        result = []
        for bond in bonds:
            pair = tuple(bond)
            if (
                len(pair) != 2
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, Integral)
                    or not 0 <= i < len(geometry)
                    for i in pair
                )
                or pair[0] == pair[1]
            ):
                raise ValueError(
                    "IGMH viewer bonds must be pairs of distinct zero-based atom indices"
                )
            result.append(list(map(int, pair)))
        return result
    if len(fragments) != len(geometry):
        raise ValueError("IGMH viewer geometry must match the report's atom count")
    inferred = []
    # Bin common nonmetals by fragment and maximum possible bond length.
    factor = 1.18 * ANGSTROM_IN_BOHR
    width = 2 * factor * max(radius for _, radius in _RADII.values())
    neighbors = tuple(product(range(-1, 2), repeat=3))
    bins = {}
    for i, (a, p) in enumerate(geometry):
        if a not in _RADII:
            continue
        x, y, z = (floor(coordinate / width) for coordinate in p)
        fragment = fragments[i]
        linked = []
        for dx, dy, dz in neighbors:
            for j in bins.get((fragment, x + dx, y + dy, z + dz), ()):
                b, q = geometry[j]
                cutoff = factor * (_RADII[a][1] + _RADII[b][1])
                if 0.1 < dist(p, q) <= cutoff:
                    linked.append(j)
        inferred.extend([j, i] for j in sorted(linked))
        bins.setdefault((fragment, x, y, z), []).append(i)
    return inferred


def _viewer_fields(report, volume):
    try:
        density = report["igmh_density"]
        gradient = report["igmh_density_gradient"]
        labels = report["units"]
        length = labels["igmh_points"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "IGMH viewer requires molecular density and gradient fields"
        ) from error
    if len(density) != len(volume.points) or len(gradient) != len(volume.points):
        raise ValueError("IGMH viewer density and gradient counts must match the grid")
    if (
        labels.get("igmh_density") != f"1/{length}^3"
        or labels.get("igmh_density_gradient") != f"1/{length}^4"
    ):
        raise ValueError("IGMH viewer density and gradient units must match the grid")
    return density, gradient


def _scatter(density, gradient, volume, density_limit):
    rho_scale = volume.length_scale**-3
    grad_scale = volume.length_scale**-4
    coefficient = 2 * (3 * pi**2) ** (1 / 3)
    samples = array("f")
    for rho, drho, signed, delta in zip(
        density, gradient, volume.signed_density, volume.delta_g
    ):
        rho = float(rho) * rho_scale
        if not isfinite(rho) or rho < 0:
            raise ValueError("IGMH viewer density must be finite and nonnegative")
        if 1e-8 <= rho <= density_limit:
            rdg = hypot(*_xyz(drho, grad_scale)) / (coefficient * rho ** (4 / 3))
            if not isfinite(rdg):
                raise ValueError("NCI reduced density gradient must be finite")
            samples.extend((float(signed) * rho_scale, float(delta) * grad_scale, rdg))
    return samples


def _packed(values):
    # The offline JavaScript renderer reads the same portable 32-bit format.
    if values.itemsize != 4:
        raise ValueError("IGMH viewer requires 32-bit packed arrays")
    if byteorder != "little":
        values.byteswap()
    return b64encode(values).decode("ascii")


def write_igmh_viewer(
    report,
    path,
    numbers,
    positions,
    *,
    field="inter",
    position_unit="bohr",
    bonds=None,
    isovalue=0.005,
    color_limit=0.05,
    igmh_limit=None,
    nci_limit=None,
    surface_density_min=None,
    surface_density_max=None,
    surface_density_field="density",
    surface_color_range=None,
    scatter_density_limit=None,
):
    """Write a standalone rotatable IGMH scene and an IGMH/NCI scatter plot.

    Supply a three-dimensional affine-grid report with at least two points
    on each axis. Geometry uses Bohr or position_unit="angstrom". Optional
    bonds are zero-based atom-index pairs. By default, links for common
    nonmetals are inferred within each fragment from their covalent radii.

    The scene contours delta-g for field="inter", "intra" or "total" and
    colours it by signed density. Isovalue and igmh_limit use electrons/Bohr^4.
    Color_limit sets the signed colour scale. Scatter_density_limit sets the
    horizontal scatter window; it defaults to color_limit. Sampled molecular
    densities below 1e-8 electrons/Bohr^3 are outside that window. The NCI
    ordinate shows the full range of the dimensionless reduced density gradient
    by default. Supply nci_limit to start with a custom upper limit, or select
    the low-gradient detail range in the browser.

    Surface_density_min/max set clipping bounds on the selected surface
    density field. Use surface_density_field="signed_density" to select
    signed density. Surface_color_range sets an independent pair of signed
    colour endpoints and defaults to (-color_limit, color_limit). The saved
    file allows decimal adjustments to these controls. Density arguments use
    electrons/Bohr^3. The HTML embeds the complete indexed mesh, scatter
    samples and JavaScript. Viewer arrays use 32-bit floats and indices.
    The source report retains its original precision.
    """
    from ._igmh_browser import PAGE

    if scatter_density_limit is None:
        scatter_density_limit = color_limit
    if not all(
        isfinite(x) and x > 0 for x in (isovalue, color_limit, scatter_density_limit)
    ):
        raise ValueError("IGMH viewer limits and isovalue must be finite and positive")
    if igmh_limit is not None and (not isfinite(igmh_limit) or igmh_limit <= 0):
        raise ValueError("IGMH scatter limit must be finite and positive")
    if nci_limit is not None and (not isfinite(nci_limit) or nci_limit <= 0):
        raise ValueError("NCI scatter limit must be finite and positive")
    if surface_density_field not in ("density", "signed_density"):
        raise ValueError("surface density field must be density or signed_density")
    if any(
        not isfinite(x)
        for x in (surface_density_min, surface_density_max)
        if x is not None
    ):
        raise ValueError("surface density bounds must be finite when supplied")
    if surface_density_field == "density":
        if surface_density_min is None:
            surface_density_min = 0.0
        elif surface_density_min < 0:
            raise ValueError("unsigned surface density cannot have a negative minimum")
    if (
        surface_density_min is not None
        and surface_density_max is not None
        and surface_density_min >= surface_density_max
    ):
        raise ValueError("surface density bounds require minimum < maximum")
    if surface_color_range is None:
        surface_color_range = (-color_limit, color_limit)
    color_min, color_max = surface_color_range
    if not (isfinite(color_min) and isfinite(color_max) and color_min < color_max):
        raise ValueError("surface colour range requires finite minimum < maximum")
    volume = _volume(report, field)
    geometry = _geometry(numbers, positions, position_unit)
    fragments = report.get("fragments", [0] * len(geometry))
    links = _viewer_bonds(geometry, fragments, bonds)
    density, gradient = _viewer_fields(report, volume)
    samples = _scatter(density, gradient, volume, scatter_density_limit)
    vertices, faces = _isosurface(volume, density, isovalue)
    if igmh_limit is None:
        ordered = sorted(samples[1::3])
        quantile = ordered[int(0.97 * (len(ordered) - 1))] if ordered else 0.0
        igmh_limit = max(quantile, 2 * isovalue)
    payload = {
        "field": field,
        "isovalue": isovalue,
        "colorLimit": color_limit,
        "scatterDensityLimit": scatter_density_limit,
        "igmhLimit": igmh_limit,
        "nciDetailLimit": nci_limit if nci_limit is not None else 2.0,
        "nciStartsInDetail": nci_limit is not None,
        "surfaceDensityMin": surface_density_min,
        "surfaceDensityMax": surface_density_max,
        "surfaceDensityField": surface_density_field,
        "surfaceColorRange": [color_min, color_max],
        "bonds": links,
        "atoms": [[z, *p, _RADII.get(z, (str(z), 0))[0]] for z, p in geometry],
        "vertices": _packed(array("f", chain.from_iterable(vertices))),
        "faces": _packed(array("I", chain.from_iterable(faces))),
        "scatter": _packed(samples),
    }
    path = Path(path)
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
    path.write_text(PAGE.replace("__IGMH_DATA__", encoded), encoding="utf-8")
    return path
