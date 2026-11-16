// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;
use crate::acceptance::{components, stencils};
use crate::periodic::{features::PeriodicBoundary, Lattice};
use crate::system::{Atom, Molecule};
use stencils::Stencil;

fn geometry_case(pbc: [bool; 3], mesh: [usize; 3]) {
    let basis_set = crate::basis::g94::parse(
        "fourier-strain-s-and-p",
        "H 0\nS 1 1.0\n0.91 1.0\nP 1 1.0\n0.67 1.0\n****\n",
    )
    .unwrap();
    let centres = [
        Vec3::new(0.12, -0.35, 0.25),
        Vec3::new(1.67, 0.43, -0.29),
        Vec3::new(-0.41, 1.23, 1.57),
    ];
    let cell = [
        Vec3::new(4.65, 0.47, 0.39),
        Vec3::new(0.61, 5.62, 0.33),
        Vec3::new(-0.24, -0.76, 6.29),
    ];
    let coordinate_count = 3 * centres.len();
    let build = |coordinate, h: f64| {
        let affine = |v: Vec3| {
            let mut v = v.to_array();
            if coordinate >= coordinate_count && coordinate < coordinate_count + 9 {
                let strain = coordinate - coordinate_count;
                v[strain / 3] += h * v[strain % 3];
            }
            Vec3::new(v[0], v[1], v[2])
        };
        let molecule = Molecule::new(
            centres
                .iter()
                .enumerate()
                .map(|(atom, &centre)| {
                    let mut point = affine(centre).to_array();
                    if coordinate / 3 == atom {
                        point[coordinate % 3] += h;
                    }
                    Atom {
                        z: 1,
                        position: Vec3::new(point[0], point[1], point[2]),
                    }
                })
                .collect(),
        );
        let basis = MolBasis::build(&molecule, &basis_set).unwrap();
        let [a, b, c] = cell.map(affine);
        let geometry =
            PeriodicGeometry::new(Lattice::new(a, b, c), PeriodicBoundary { axes: pbc }).unwrap();
        let reciprocal = geometry.reciprocal();
        let wave = reciprocal[0] - reciprocal[1] + reciprocal[2];
        let field = StaticPeriodicField {
            wave,
            vector_potential: [
                Complex::new(0.026, -0.038),
                Complex::new(-0.041, 0.023),
                Complex::new(0.037, 0.018),
            ],
            scalar_potential: Complex::new(0.095, -0.053),
            amplitude: 0.71,
        };
        let points = geometry.k_points(mesh).unwrap();
        (basis, geometry, field, points)
    };
    // Keep the same image set on every displaced cell. The nearest
    // excluded pair is at 3.644 Bohr and the last retained one is at 3.251.
    let cutoff = 3.45;
    let (basis, geometry, field, points) = build(usize::MAX, 0.);
    let density: Vec<_> = points
        .iter()
        .enumerate()
        .map(|(k, _)| {
            let n = basis.nao_sph;
            let mut matrix = BlochMatrix::zeros(n, n);
            for i in 0..n {
                for j in 0..=i {
                    let phase = 0.29 * (i + 1) as f64 + 0.61 * (j + k + 1) as f64;
                    let re = phase.cos() * 0.17 + if i == j { 0.28 } else { 0.0 };
                    let im = if i == j { 0.0 } else { phase.sin() * 0.12 };
                    matrix.re[(i, j)] = re;
                    matrix.re[(j, i)] = re;
                    matrix.im[(i, j)] = im;
                    matrix.im[(j, i)] = -im;
                }
            }
            matrix
        })
        .collect();
    let analytic = field
        .geometry_derivative(&basis, &geometry, &points, &density, cutoff, centres.len())
        .unwrap();
    let sample = |coordinate, h| {
        let (basis, geometry, field, points) = build(coordinate, h);
        let operator = field.operators(&basis, &geometry, &points, cutoff).unwrap();
        density
            .iter()
            .zip(operator.hamiltonian)
            .map(|(p, h)| p.trace_product(&h))
            .sum::<f64>()
            / points.len() as f64
    };
    let defect = (sample(usize::MAX, 0.) - analytic.energy).abs();
    assert!(defect < 1e-13);
    let derivative: Vec<_> = analytic
        .gradient
        .iter()
        .flat_map(|v| v.to_array())
        .chain(analytic.strain.into_iter().flatten())
        .collect();
    let labels: Vec<_> = (0..coordinate_count)
        .map(|i| format!("Fourier field {pbc:?} atom[{},{}]", i / 3, i % 3))
        .chain((0..9).map(|i| format!("Fourier field {pbc:?} affine[{},{}]", i / 3, i % 3)))
        .collect();
    let mut failed = Vec::new();
    stencils::fit_components_staged(
        &labels,
        &[
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
        ],
        |steps, stencil| {
            derivative
                .iter()
                .enumerate()
                .map(|(i, &analytic)| {
                    steps.map(|h| {
                        let estimate = stencil.first(sample(i, h), sample(i, -h), 0.0, 0.0, h);
                        (estimate - analytic).abs()
                    })
                })
                .collect()
        },
        &mut failed,
    );
    components::assert_no_failures("fixed-density Fourier-field geometry", &failed);
    println!("Fourier field geometry {pbc:?} energy={:.17e} reconstruction={defect:.17e} gradient={:?} affine={:?}",
        analytic.energy, analytic.gradient, analytic.strain);
}

#[test]
fn finite_fourier_field_geometry_fits_all_components() {
    for (pbc, mesh) in [
        ([true, false, false], [3, 1, 1]),
        ([true, false, true], [2, 1, 2]),
        ([true; 3], [2, 2, 1]),
    ] {
        geometry_case(pbc, mesh);
    }
}
