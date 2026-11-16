# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
"""Round trips of paired IGMH CUBE files, the slice and the offline viewer."""

from base64 import b64decode
from copy import deepcopy
from itertools import product
import json
from math import pi
import re
from xml.etree import ElementTree

import numpy as np
import pytest
from ase import Atoms
from ase.io.cube import read_cube
from ase.units import Bohr as AseBohr

from dft_rs import DftRs, igmh
from dft_rs.igmh_visualization import (
    write_igmh_cubes,
    write_igmh_slice,
    write_igmh_viewer,
)
from dft_rs.units import BOHR_IN_ANGSTROM as Bohr
from test_igmh import WATER_DIMER


def load_cube(path):
    with path.open(encoding="ascii") as stream:
        return read_cube(stream)


def load_viewer(path):
    source = path.read_text(encoding="utf-8")
    payload = re.search(
        r'<script id="igmh-data" type="application/json">(.*?)</script>', source
    )
    view = json.loads(payload[1])
    for key, dtype, width in (
        ("vertices", "<f4", 5),
        ("faces", "<u4", 3),
        ("scatter", "<f4", 3),
    ):
        view[key] = (
            np.frombuffer(b64decode(view[key], validate=True), dtype=dtype)
            .reshape(-1, width)
            .tolist()
        )
    return view


def test_viewer_interpolates_the_analytic_plane_and_nci_values(tmp_path):
    shape = (3, 4, 2)
    frame = np.array(
        [[0.3, -0.2, 0.8], [0.2, 0.01, -0.01], [0.03, 0.4, 0.02], [-0.02, 0.06, 0.5]]
    )
    reduced = np.array(list(product(*(range(n) for n in shape))))
    points = frame[0] + reduced @ frame[1:]
    signed = 0.012 * reduced[:, 1] - 0.02
    density = 0.023 + 0.009 * reduced[:, 1] + 0.002 * reduced[:, 2]
    gradient = np.tile([0.2, 0.3, -0.4], (len(points), 1))
    report = dict(
        igmh_points=points.tolist(),
        igmh_grid_shape=shape,
        igmh_density=density.tolist(),
        igmh_density_gradient=gradient.tolist(),
        igmh_delta_g_inter=(0.02 * reduced[:, 0]).tolist(),
        igmh_signed_density=signed.tolist(),
        fragments=[0, 0],
        units={
            "igmh_points": "bohr",
            "igmh_density": "1/bohr^3",
            "igmh_density_gradient": "1/bohr^4",
            "igmh_signed_density": "1/bohr^3",
            "igmh_delta_g_inter": "1/bohr^4",
        },
    )
    path = write_igmh_viewer(
        report,
        tmp_path / "plane.html",
        [8, 1],
        [[0, 0, 0], [1.7, 0, 0]],
        isovalue=0.01,
        bonds=[(0, 1)],
    )
    view = load_viewer(path)
    vertices = np.array(view["vertices"])
    lattice_fraction = (vertices[:, :3] - frame[0]) @ np.linalg.inv(frame[1:])
    np.testing.assert_allclose(lattice_fraction[:, 0], 0.5, atol=5e-7, rtol=0.0)
    np.testing.assert_allclose(
        vertices[:, 3], 0.012 * lattice_fraction[:, 1] - 0.02, atol=1e-8, rtol=0.0
    )
    np.testing.assert_allclose(
        vertices[:, 4],
        0.023 + 0.009 * lattice_fraction[:, 1] + 0.002 * lattice_fraction[:, 2],
        atol=1e-8,
        rtol=0.0,
    )
    normal = np.linalg.inv(frame[1:])[:, 0]
    for face in view["faces"]:
        a, b, c = vertices[face, :3]
        assert np.cross(b - a, c - a) @ normal < 0.0
    assert view["faces"] and len({tuple(v[:3]) for v in vertices}) == len(vertices)
    assert view["bonds"] == [[0, 1]]
    expected_rdg = np.linalg.norm(gradient[0]) / (
        2 * (3 * pi**2) ** (1 / 3) * density ** (4 / 3)
    )
    keep = density <= 0.05
    np.testing.assert_allclose(
        np.asarray(view["scatter"])[:, 2], expected_rdg[keep], rtol=6e-8
    )
    assert max(expected_rdg) > 2
    assert view["nciDetailLimit"] == 2
    assert view["nciStartsInDetail"] is False
    nci_detail = load_viewer(
        write_igmh_viewer(
            report,
            tmp_path / "plane-detail.html",
            [8, 1],
            [[0, 0, 0], [1.7, 0, 0]],
            isovalue=0.01,
            color_limit=0.01,
            scatter_density_limit=0.055,
            surface_density_min=0.03,
            surface_density_max=0.04,
            nci_limit=1.5,
        )
    )
    assert nci_detail["nciStartsInDetail"] is True
    assert nci_detail["nciDetailLimit"] == 1.5
    assert nci_detail["surfaceDensityMin"] == 0.03
    assert nci_detail["surfaceDensityMax"] == 0.04
    assert nci_detail["faces"] == view["faces"]
    assert len(nci_detail["scatter"]) == density.size
    source = path.read_text(encoding="utf-8")
    assert "clipDensity" in source and "requestAnimationFrame" in source
    assert not re.search(r"<(?:script|link)[^>]+(?:src|href)=", source)
    assert view["surfaceDensityField"] == "density"
    assert view["surfaceColorRange"] == [-0.05, 0.05]
    signed_view = load_viewer(
        write_igmh_viewer(
            report,
            tmp_path / "signed-window.html",
            [8, 1],
            [[0, 0, 0], [1.7, 0, 0]],
            isovalue=0.01,
            surface_density_field="signed_density",
            surface_density_min=-0.0175,
            surface_density_max=-0.0032,
            surface_color_range=(-0.0157, -0.0003),
        )
    )
    assert signed_view["surfaceDensityField"] == "signed_density"
    assert signed_view["surfaceDensityMin"] == -0.0175
    assert signed_view["surfaceDensityMax"] == -0.0032
    assert signed_view["surfaceColorRange"] == [-0.0157, -0.0003]
    assert signed_view["scatter"] == view["scatter"]

    with pytest.raises(ValueError, match="zero-based atom indices"):
        write_igmh_viewer(
            report,
            tmp_path / "invalid.html",
            [8, 1],
            [[0, 0, 0], [1.7, 0, 0]],
            bonds=[(0, 2)],
        )
    with pytest.raises(ValueError, match="surface density bounds"):
        write_igmh_viewer(
            report,
            tmp_path / "inverted.html",
            [8, 1],
            [[0, 0, 0], [1.7, 0, 0]],
            surface_density_min=0.03,
            surface_density_max=0.02,
        )
    with pytest.raises(ValueError, match="unsigned surface density"):
        write_igmh_viewer(
            report,
            tmp_path / "negative-density.html",
            [8, 1],
            [[0, 0, 0], [1.7, 0, 0]],
            surface_density_min=-0.0125,
        )


def test_water_dimer_visualization_in_native_and_ase_units(tmp_path):
    positions = np.asarray(WATER_DIMER)
    atoms = Atoms("OHHOHH", positions=positions * Bohr)
    shape = [7, 5, 11]
    grid = np.array(
        [
            [-1.0, -0.65, 1.3],
            [0.30, 0.02, 0.01],
            [-0.06, 0.30, 0.01],
            [0.01, -0.01, 0.30],
        ]
    )
    controls = dict(
        xc="wb97x-d",
        basis="sto-3g",
        grid_level=0,
        cpu_threads=1,
        grid_weight_cutoff=0.0,
        fragments=[0, 0, 0, 1, 1, 1],
        igmh_grid=grid,
        igmh_grid_shape=shape,
    )
    native = igmh(atoms.numbers, positions, **controls)
    physical = DftRs().get_igmh(atoms, **dict(controls, igmh_grid=grid * Bohr))
    atomic_cubes = write_igmh_cubes(
        native, tmp_path / "atomic", atoms.numbers, positions
    )
    ase_cubes = write_igmh_cubes(
        physical,
        tmp_path / "ase",
        atoms.numbers,
        atoms.positions,
        position_unit="angstrom",
    )

    max_error = 0.0
    for key, unit_exponent, atomic_file, ase_file in zip(
        ["igmh_delta_g_inter", "igmh_signed_density"], [4, 3], atomic_cubes, ase_cubes
    ):
        a, b = load_cube(atomic_file), load_cube(ase_file)
        expected = np.asarray(native[key]).reshape(shape)
        np.testing.assert_allclose(a["data"], expected, rtol=5e-13, atol=2e-15)
        np.testing.assert_allclose(b["data"], expected, rtol=5e-13, atol=2e-15)
        max_error = max(
            max_error,
            np.max(np.abs(a["data"] - expected)),
            np.max(np.abs(b["data"] - expected)),
        )
        np.testing.assert_allclose(
            np.asarray(physical[key]).reshape(shape) * Bohr**unit_exponent,
            expected,
            rtol=0.0,
            atol=1e-13,
        )
        for cube in (a, b):
            np.testing.assert_allclose(
                cube["origin"] / AseBohr, grid[0], rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                cube["spacing"] / AseBohr, grid[1:], rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(
                cube["atoms"].positions / AseBohr, positions, rtol=0.0, atol=1e-12
            )
            np.testing.assert_array_equal(cube["atoms"].numbers, atoms.numbers)

    svg = write_igmh_slice(
        physical, tmp_path / "inter.svg", axis=1, index=2, isovalue=0.025
    )
    image = ElementTree.parse(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    colored = image.findall("svg:polygon", ns)[1:]  # Skip the plot background.
    expected_count = np.count_nonzero(
        np.asarray(native["igmh_delta_g_inter"]).reshape(shape)[:, 2, :] >= 0.025
    )
    assert expected_count > 0
    assert len(colored) == expected_count
    assert {face.get("fill") for face in colored} != {"#579d65"}
    viewer = write_igmh_viewer(
        physical,
        tmp_path / "viewer.html",
        atoms.numbers,
        atoms.positions,
        position_unit="angstrom",
        isovalue=0.025,
    )
    view = load_viewer(viewer)
    assert len(view["atoms"]) == 6 and view["bonds"] == [[0, 1], [0, 2], [3, 4], [3, 5]]
    np.testing.assert_allclose(
        np.asarray(view["atoms"], dtype=object)[:, 1:4].astype(float),
        positions,
        atol=1e-12,
        rtol=0.0,
    )
    assert view["faces"] and view["scatter"]
    print(
        f"water dimer CUBE shape {shape}, maximum field error {max_error:.3e}, "
        f"SVG cells above isovalue {expected_count}"
    )


def test_affine_report_requires_explicit_step_for_one_point_axis(tmp_path):
    numbers, positions = [1, 1], [[0, 0, -0.7], [0, 0, 0.7]]
    grid = [[-0.1, 0.2, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]]
    report = igmh(
        numbers,
        positions,
        xc="pbe",
        basis="sto-3g",
        grid_level=0,
        cpu_threads=1,
        igmh_grid=grid,
        igmh_grid_shape=[2, 1, 3],
        fragments=[0, 1],
    )
    with pytest.raises(ValueError, match="one-point axis"):
        write_igmh_cubes(report, tmp_path / "h2", numbers, positions)
    cubes = write_igmh_cubes(
        report, tmp_path / "h2", numbers, positions, field="total", grid=grid
    )
    np.testing.assert_allclose(
        load_cube(cubes[0])["data"].ravel(),
        report["igmh_delta_g"],
        rtol=5e-13,
        atol=2e-15,
    )
    assert write_igmh_slice(report, tmp_path / "plane.svg", axis=1).is_file()

    bad = deepcopy(report)
    bad["igmh_points"][3][2] += 0.1
    with pytest.raises(ValueError, match="do not match the affine cube grid"):
        write_igmh_cubes(bad, tmp_path / "bad", numbers, positions, grid=grid)
    assert not list(tmp_path.glob("bad*"))

    del bad["igmh_grid_shape"]
    with pytest.raises(ValueError, match="requires an affine-grid IGMH report"):
        write_igmh_slice(bad, tmp_path / "bad.svg")
