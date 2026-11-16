// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Bounded local Taylor values for contracted Gaussian integral blocks. Runtime
//! arena handles retain every scalar operation until the entire block returns;
//! high-angular Cartesian blocks therefore need local coefficient storage.
use super::GeometrySources;
use crate::basis::{n_cart, n_sph, MolBasis};
use crate::dual::{Scalar, Taylor};
use crate::integrals::analytic::GaussianKernel;
use crate::integrals::analytic_basis::{self, AnalyticNucleus, AnalyticShell};
use crate::integrals::{eri_quartet, CoulombKernel};
use crate::math::Vec3;
use crate::response::series::evaluate_series_many_result;
use crate::system::Molecule;
use crate::{DftError, Result};

macro_rules! local_order {
    ($order:expr, $fixed:ident($($argument:expr),*), $fallback:expr) => {
        match $order {
            0 => $fixed::<1>($($argument),*),
            1 => $fixed::<2>($($argument),*),
            2 => $fixed::<3>($($argument),*),
            3 => $fixed::<4>($($argument),*),
            4 => $fixed::<5>($($argument),*),
            5..=7 => $fixed::<8>($($argument),*),
            8..=15 => $fixed::<16>($($argument),*),
            16..=31 => $fixed::<32>($($argument),*),
            _ => $fallback,
        }
    };
}

fn width(order: usize) -> Option<usize> {
    match order {
        0..=4 => Some(order + 1),
        5..=7 => Some(8),
        8..=15 => Some(16),
        16..=31 => Some(32),
        _ => None,
    }
}

fn tuple_layout<const W: usize>() -> (usize, usize) {
    (
        std::mem::size_of::<(Taylor<W>, [usize; 3], [usize; 3])>(),
        std::mem::size_of::<([usize; 3], Taylor<W>)>(),
    )
}

/// Conservative local block/transform, Hermite/Boys and scalar input storage.
/// This excludes the separately counted AO output tensors. The local scalar
/// workspace is independent of the number of arithmetic operations.
pub(super) fn workspace_bytes(
    angular: [u8; 4],
    primitives: [usize; 4],
    order: usize,
) -> Result<Option<usize>> {
    let Some(width) = width(order) else {
        return Ok(None);
    };
    let fail = || DftError::InvalidInput("nuclear integral workspace size overflow".into());
    let product = |values: &[usize]| {
        values
            .iter()
            .try_fold(1usize, |n, &m| n.checked_mul(m))
            .ok_or_else(fail)
    };
    let sum = |values: &[usize]| {
        values
            .iter()
            .try_fold(0usize, |n, &m| n.checked_add(m))
            .ok_or_else(fail)
    };
    let scalar = width * std::mem::size_of::<f64>();
    let (member, entry) = match width {
        1 => tuple_layout::<1>(),
        2 => tuple_layout::<2>(),
        3 => tuple_layout::<3>(),
        4 => tuple_layout::<4>(),
        5 => tuple_layout::<5>(),
        8 => tuple_layout::<8>(),
        16 => tuple_layout::<16>(),
        32 => tuple_layout::<32>(),
        _ => unreachable!("bounded integral scalar width"),
    };
    let vector = std::mem::size_of::<Vec<f64>>();
    let cartesian = product(&angular.map(n_cart))?;
    let spherical = product(&angular.map(n_sph))?;
    let total = angular.iter().map(|&l| l as usize).sum::<usize>();
    let pair = |a: usize, b: usize| (a + 1) * (b + 1) * (a + b + 1);
    let hermite = 3 * (total + 1).pow(3)
        + 3 * (pair(angular[0] as usize, angular[1] as usize)
            + pair(angular[2] as usize, angular[3] as usize))
        + 4 * (total + order + 1);
    let mut pairs = 0usize;
    for [a, b] in [[angular[0], angular[1]], [angular[2], angular[3]]] {
        let count = product(&[n_sph(a), n_sph(b)])?;
        let degree = a as usize + b as usize;
        let tetrahedron = product(&[degree + 1, degree + 2, degree + 3])? / 6;
        // Spherical expansion members reserve their exact nonzero count.
        // Hermite entries grow dynamically: four times the reachable count
        // bounds the allocator's minimum capacity and geometric growth.
        pairs = sum(&[
            pairs,
            product(&[count, n_cart(a), n_cart(b), member])?,
            product(&[count, 4, tetrahedron, entry])?,
            product(&[count, 2, vector])?,
        ])?;
    }
    let dynamic = sum(&[product(&[spherical, 3 * scalar + vector])?, pairs])?;
    let static_block = sum(&[
        product(&[cartesian, 4, 8])?,
        product(&[spherical, scalar + vector])?,
    ])?;
    Ok(Some(sum(&[
        dynamic.max(static_block),
        product(&[hermite, scalar])?,
        (total + 1).pow(3), // temporary touched-entry mask
        product(&[sum(&primitives)?, 2, scalar])?,
        64 * 1024,
    ])?))
}

pub(super) fn pair_workspace_bytes(
    angular: [u8; 2],
    primitives: [usize; 2],
    atoms: usize,
    order: usize,
) -> Result<Option<usize>> {
    let Some(width) = width(order) else {
        return Ok(None);
    };
    let fail = || DftError::InvalidInput("nuclear pair workspace size overflow".into());
    let cartesian = n_cart(angular[0]) * n_cart(angular[1]);
    let spherical = n_sph(angular[0]) * n_sph(angular[1]);
    let total = angular[0] as usize + angular[1] as usize;
    let primitive_slots = primitives[0]
        .checked_add(primitives[1])
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(fail)?;
    let slots =
        20 * spherical + 4 * cartesian + 4 * (total + 3).pow(3) + 4 * (total + order + 1) + 16;
    let slots = atoms
        .checked_mul(10)
        .and_then(|n| n.checked_add(slots))
        .and_then(|n| n.checked_add(primitive_slots))
        .ok_or_else(fail)?;
    let bytes = slots
        .checked_mul(width)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(5 * spherical * std::mem::size_of::<Vec<f64>>()))
        .and_then(|n| n.checked_add(64 * 1024))
        .ok_or_else(fail)?;
    Ok(Some(bytes))
}

fn placed<S: Scalar, const N: usize>(
    mb: &MolBasis,
    molecule: &Molecule,
    coordinates: [[S; 3]; N],
    indices: [usize; N],
) -> Result<Vec<AnalyticShell<S>>> {
    indices
        .into_iter()
        .zip(coordinates)
        .map(|(i, position)| super::displaced_shell(&mb.shells[i], molecule, position))
        .collect()
}

fn evaluate_centres<S: Scalar>(
    mb: &MolBasis,
    molecule: &Molecule,
    coordinates: [[S; 3]; 4],
    indices: [usize; 4],
    omega: Option<f64>,
    order: usize,
) -> Result<Vec<S>> {
    let sh = placed(mb, molecule, coordinates, indices)?;
    let kernel = omega.map_or(GaussianKernel::Coulomb, |omega| GaussianKernel::LongRange {
        omega: S::from_f64(omega),
    });
    Ok(analytic_basis::repulsion(&sh[0], &sh[1], &sh[2], &sh[3], kernel, order)?.values)
}

fn coefficients<const W: usize>(values: Vec<Taylor<W>>, order: usize) -> Result<Vec<Vec<f64>>> {
    let result = values
        .into_iter()
        .map(|value| value.c[..=order].to_vec())
        .collect::<Vec<_>>();
    if result.iter().flatten().any(|value| !value.is_finite()) {
        return Err(DftError::InvalidInput(
            "nuclear integral Taylor expansion is non-finite at the supplied geometry".into(),
        ));
    }
    Ok(result)
}

fn evaluate<S: Scalar>(
    mb: &MolBasis,
    molecule: &Molecule,
    values: &[S],
    indices: [usize; 4],
    omega: Option<f64>,
    order: usize,
) -> Result<Vec<S>> {
    let coordinates =
        indices.map(|i| std::array::from_fn(|axis| values[3 * mb.shells[i].atom + axis]));
    evaluate_centres(mb, molecule, coordinates, indices, omega, order)
}

fn fixed<const W: usize>(
    mb: &MolBasis,
    molecule: &Molecule,
    input: &[Vec<f64>],
    indices: [usize; 4],
    omega: Option<f64>,
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    let coordinates = indices.map(|i| {
        std::array::from_fn(|axis| {
            Taylor::<W>::from_coefficients(&input[3 * mb.shells[i].atom + axis])
        })
    });
    coefficients(
        evaluate_centres(mb, molecule, coordinates, indices, omega, order)?,
        order,
    )
}

fn pair_values<S: Scalar>(
    mb: &MolBasis,
    molecule: &Molecule,
    values: &[S],
    indices: [usize; 2],
    charges: &[f64],
    sources: Option<&GeometrySources<'_>>,
    order: usize,
) -> Result<Vec<S>> {
    let positions =
        indices.map(|i| std::array::from_fn(|axis| values[3 * mb.shells[i].atom + axis]));
    let sh = placed(mb, molecule, positions, indices)?;
    let nuclei = values[..3 * molecule.len()]
        .chunks_exact(3)
        .zip(charges)
        .map(|(p, &charge)| AnalyticNucleus {
            center: [p[0], p[1], p[2]],
            charge: S::from_f64(charge),
        })
        .collect::<Vec<_>>();
    let overlap = analytic_basis::overlap(&sh[0], &sh[1])?;
    let kinetic = analytic_basis::kinetic(&sh[0], &sh[1])?;
    let attraction =
        analytic_basis::attraction(&sh[0], &sh[1], &nuclei, GaussianKernel::Coulomb, order)?;
    let dipoles = (0..3)
        .map(|axis| analytic_basis::dipole(&sh[0], &sh[1], axis, [S::from_f64(0.0); 3]))
        .collect::<Result<Vec<_>>>()?;
    let [a, b] = indices;
    let field_offset = 3 * (molecule.len() + sources.map_or(0, |s| s.additional_centers.len()));
    let (mass, charge, same) = sources.map_or((1.0, -1.0, true), |s| {
        let a = s.particles[mb.shells[a].sph_off];
        let b = s.particles[mb.shells[b].sph_off];
        (a.mass, a.charge, a.block == b.block)
    });
    let mut output = overlap.values;
    for (i, (&t, vn)) in kinetic.values.iter().zip(attraction.values).enumerate() {
        let mut h = if same {
            t.scale(1.0 / mass) - vn.scale(charge)
        } else {
            S::from_f64(0.0)
        };
        for axis in 0..3 {
            if same {
                h = h - values[field_offset + axis] * dipoles[axis].values[i].scale(charge);
            }
        }
        output.push(h);
    }
    for dipole in dipoles {
        output.extend(dipole.values);
    }
    Ok(output)
}

fn pair_fixed<const W: usize>(
    mb: &MolBasis,
    molecule: &Molecule,
    input: &[Vec<f64>],
    indices: [usize; 2],
    charges: &[f64],
    sources: Option<&GeometrySources<'_>>,
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    let values = input
        .iter()
        .map(|v| Taylor::<W>::from_coefficients(v))
        .collect::<Vec<_>>();
    coefficients(
        pair_values(mb, molecule, &values, indices, charges, sources, order)?,
        order,
    )
}

pub(super) fn one_electron(
    mb: &MolBasis,
    molecule: &Molecule,
    input: &[Vec<f64>],
    indices: [usize; 2],
    charges: &[f64],
    sources: Option<&GeometrySources<'_>>,
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    local_order!(
        order,
        pair_fixed(mb, molecule, input, indices, charges, sources, order),
        evaluate_series_many_result(input, |values| {
            pair_values(mb, molecule, values, indices, charges, sources, order)
        })
    )
}

pub(super) fn repulsion(
    mb: &MolBasis,
    molecule: &Molecule,
    input: &[Vec<f64>],
    indices: [usize; 4],
    omega: Option<f64>,
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    if omega.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(DftError::InvalidInput(
            "range separation must be finite and nonnegative".into(),
        ));
    }
    let owners = indices.map(|i| mb.shells[i].atom);
    if owners.iter().all(|&atom| {
        (0..3).all(|axis| input[3 * atom + axis][1..] == input[3 * owners[0] + axis][1..])
    }) {
        // Bare two-electron integrals depend only on relative centres. All
        // four centres may follow the same curved path: keep the complete
        // static block and its exact vanishing translation derivatives.
        let placed = indices.map(|i| {
            let mut shell = mb.shells[i].clone();
            if let Some(atom) = molecule.atoms.get(shell.atom) {
                let origin = atom.position;
                shell.center += Vec3::new(
                    input[3 * shell.atom][0] - origin.x,
                    input[3 * shell.atom + 1][0] - origin.y,
                    input[3 * shell.atom + 2][0] - origin.z,
                );
            }
            shell
        });
        let kernel = omega.map_or(CoulombKernel::Coulomb, |omega| CoulombKernel::Erf { omega });
        return Ok(
            eri_quartet(&placed[0], &placed[1], &placed[2], &placed[3], kernel)
                .data
                .into_iter()
                .map(|value| {
                    let mut coefficients = vec![0.0; order + 1];
                    coefficients[0] = value;
                    coefficients
                })
                .collect(),
        );
    }
    local_order!(order, fixed(mb, molecule, input, indices, omega, order), {
        evaluate_series_many_result(input, |v| evaluate(mb, molecule, v, indices, omega, order))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils, symmetric_cartesian_directions};
    use crate::basis::{BasisSet, ElementBasis, Shell};
    use crate::system::Atom;
    use std::collections::BTreeMap;

    fn fixture(l: u8) -> (Molecule, MolBasis) {
        let positions = [
            [0.13, -0.27, 0.08],
            [1.39, 0.21, -0.34],
            [-0.41, 1.52, 0.31],
            [0.28, -0.33, 1.71],
        ];
        let molecule = Molecule::new(
            positions
                .into_iter()
                .enumerate()
                .map(|(i, p)| Atom {
                    z: i as u8 + 1,
                    position: Vec3::new(p[0], p[1], p[2]),
                })
                .collect(),
        );
        let mut basis = BasisSet::new("local Taylor workspace test");
        for i in 0..4 {
            basis.insert(ElementBasis {
                z: i + 1,
                shells: vec![Shell {
                    l: if i == 3 { l } else { 0 },
                    exps: vec![0.51 + 0.13 * i as f64],
                    coefs: vec![1.0],
                }],
            });
        }
        let placed = MolBasis::build(&molecule, &basis).unwrap();
        (molecule, placed)
    }

    fn curved_input(molecule: &Molecule, order: usize) -> Vec<Vec<f64>> {
        let direction = (0..4)
            .map(|atom| std::array::from_fn(|axis| 0.17 * (3 * atom + axis + 1) as f64))
            .collect::<Vec<_>>();
        let mut input = super::super::inputs(molecule, &direction, order);
        for (coordinate, coefficients) in input.iter_mut().enumerate() {
            for (n, value) in coefficients.iter_mut().enumerate().skip(2) {
                *value = 0.03 * ((coordinate + 2 * n + 1) as f64).sin() / n as f64;
            }
        }
        input
    }

    fn compare(actual: &[Vec<f64>], expected: &[Vec<f64>]) {
        assert_eq!(actual.len(), expected.len());
        for (entry, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.len(), b.len());
            for (order, (&a, &b)) in a.iter().zip(b).enumerate() {
                assert!(
                    (a - b).abs() < 2e-13 * (1.0 + b.abs()),
                    "entry={entry} order={order} actual={a:.16e} expected={b:.16e}"
                );
            }
        }
    }

    fn pair_input(molecule: &Molecule) -> Vec<Vec<f64>> {
        let mut input = curved_input(molecule, 4);
        for axis in 0..3 {
            input.push(vec![
                0.013 * (axis + 1) as f64,
                -0.021 * (axis + 2) as f64,
                0.004,
                -0.002,
                0.001,
            ]);
        }
        input
    }

    fn runtime_pair(
        molecule: &Molecule,
        basis: &MolBasis,
        input: &[Vec<f64>],
        indices: [usize; 2],
        charges: &[f64],
        sources: Option<&GeometrySources<'_>>,
        order: usize,
    ) -> Vec<Vec<f64>> {
        // Keep the former complete-pair runtime arena as the independent
        // arithmetic/assembly oracle, including all shell lifts and fields.
        evaluate_series_many_result(input, |v| {
            let centres = super::super::centers(&v[..3 * molecule.len()]);
            let sh = super::super::shells(basis, molecule, &centres)?;
            let nuclei = centres
                .iter()
                .zip(charges)
                .map(|(&center, &charge)| AnalyticNucleus {
                    center,
                    charge: crate::response::series::SeriesScalar::from_f64(charge),
                })
                .collect::<Vec<_>>();
            let [a, b] = indices;
            let overlap = analytic_basis::overlap(&sh[a], &sh[b])?;
            let kinetic = analytic_basis::kinetic(&sh[a], &sh[b])?;
            let attraction = analytic_basis::attraction(
                &sh[a],
                &sh[b],
                &nuclei,
                GaussianKernel::Coulomb,
                order,
            )?;
            let dipoles = (0..3)
                .map(|axis| {
                    analytic_basis::dipole(
                        &sh[a],
                        &sh[b],
                        axis,
                        [crate::response::series::SeriesScalar::from_f64(0.0); 3],
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let mut out = overlap.values;
            for (i, (&t, vn)) in kinetic.values.iter().zip(attraction.values).enumerate() {
                let (mass, charge, same) = sources.map_or((1., -1., true), |s| {
                    let a = s.particles[basis.shells[a].sph_off];
                    let b = s.particles[basis.shells[b].sph_off];
                    (a.mass, a.charge, a.block == b.block)
                });
                let mut h = if same {
                    t.scale(1. / mass) - vn.scale(charge)
                } else {
                    crate::response::series::SeriesScalar::from_f64(0.)
                };
                for axis in 0..3 {
                    if same {
                        h = h - v[3 * molecule.len() + axis]
                            * dipoles[axis].values[i].scale(charge);
                    }
                }
                out.push(h);
            }
            for dipole in dipoles {
                out.extend(dipole.values);
            }
            Ok(out)
        })
        .unwrap()
    }

    #[test]
    fn all_one_electron_coefficients_match_the_former_runtime_assembly() {
        for l in 0..=crate::basis::aos::MAX_L {
            let (molecule, basis) = fixture(l);
            let input = pair_input(&molecule);
            let charges = [1., 2., 3., 4.];
            for indices in [[0, 3], [3, 0], [3, 3]] {
                let expected = runtime_pair(&molecule, &basis, &input, indices, &charges, None, 4);
                let actual =
                    one_electron(&basis, &molecule, &input, indices, &charges, None, 4).unwrap();
                compare(&actual, &expected);
                let maximum = actual
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f64, f64::max);
                println!("one-electron runtime comparison l={l} indices={indices:?} coefficients={} max_defect={maximum:.12e}",actual.len()*5);
                assert_eq!(maximum, 0.0);
            }
        }
    }

    #[test]
    fn every_nonzero_high_angular_one_electron_path_entry_has_four_step_fits() {
        let mut failures = Vec::new();
        for l in 2..=crate::basis::aos::MAX_L {
            let (molecule, basis) = fixture(l);
            let input = pair_input(&molecule);
            let charges = [1., 2., 3., 4.];
            for indices in [[0, 3], [3, 3]] {
                let exact =
                    one_electron(&basis, &molecule, &input, indices, &charges, None, 4).unwrap();
                let count = n_sph(basis.shells[indices[0]].l) * n_sph(basis.shells[indices[1]].l);
                // Same-centre overlap/kinetic/dipoles have structural zeros;
                // their full coefficients are checked above. The complete
                // anisotropic nuclear/core matrix remains nonzero in C1.
                let entries = if indices[0] == indices[1] {
                    (count..2 * count).collect::<Vec<_>>()
                } else {
                    (0..5 * count).collect::<Vec<_>>()
                };
                let mut cache = BTreeMap::new();
                for order in [1usize, 2, 3, 4] {
                    let names = entries
                        .iter()
                        .map(|i| {
                            format!(
                                "C1 one-electron l={l} indices={indices:?} entry={i} order={order}"
                            )
                        })
                        .collect::<Vec<_>>();
                    stencils::fit_components_staged(
                        &names,
                        &[
                            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Second),
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Second),
                            (
                                [0.005, 0.00375, 0.0025, 0.001875],
                                stencils::Stencil::Second,
                            ),
                            ([0.16, 0.12, 0.08, 0.06], stencils::Stencil::Second),
                            ([0.04, 0.03, 0.02, 0.015], stencils::Stencil::Fourth),
                            ([0.02, 0.015, 0.01, 0.0075], stencils::Stencil::Fourth),
                            ([0.08, 0.06, 0.04, 0.03], stencils::Stencil::Fourth),
                        ],
                        |steps, stencil| {
                            let mut at = |t: f64| {
                                cache
                                    .entry(t.to_bits())
                                    .or_insert_with(|| {
                                        let shifted = input
                                            .iter()
                                            .map(|v| {
                                                vec![
                                                    (0..v.len())
                                                        .map(|n| v[n] * t.powi(n as i32))
                                                        .sum(),
                                                    (1..v.len())
                                                        .map(|n| {
                                                            n as f64 * v[n] * t.powi(n as i32 - 1)
                                                        })
                                                        .sum(),
                                                ]
                                            })
                                            .collect::<Vec<_>>();
                                        runtime_pair(
                                            &molecule, &basis, &shifted, indices, &charges, None, 1,
                                        )
                                    })
                                    .clone()
                            };
                            let mut errors = vec![[0.; 4]; entries.len()];
                            for (step, &h) in steps.iter().enumerate() {
                                let mut near = |h: f64| {
                                    let (p, m) = (at(h), at(-h));
                                    if order == 1 {
                                        return entries
                                            .iter()
                                            .map(|&i| (p[i][0] - m[i][0]) / (2. * h))
                                            .collect::<Vec<_>>();
                                    }
                                    if order == 2 {
                                        return entries
                                            .iter()
                                            .map(|&i| (p[i][1] - m[i][1]) / (2. * h))
                                            .collect::<Vec<_>>();
                                    }
                                    if order == 3 {
                                        let z = at(0.);
                                        return entries
                                            .iter()
                                            .map(|&i| {
                                                (p[i][1] - 2. * z[i][1] + m[i][1]) / h.powi(2)
                                            })
                                            .collect::<Vec<_>>();
                                    }
                                    let (p2, m2) = (at(2. * h), at(-2. * h));
                                    entries
                                        .iter()
                                        .map(|&i| {
                                            (p2[i][1] - 2. * p[i][1] + 2. * m[i][1] - m2[i][1])
                                                / (2. * h.powi(3))
                                        })
                                        .collect::<Vec<_>>()
                                };
                                let first = near(h);
                                let second = if stencil == stencils::Stencil::Fourth {
                                    near(2. * h)
                                } else {
                                    vec![0.; entries.len()]
                                };
                                for (slot, &i) in entries.iter().enumerate() {
                                    let value = if stencil == stencils::Stencil::Fourth {
                                        (4. * first[slot] - second[slot]) / 3.
                                    } else {
                                        first[slot]
                                    };
                                    errors[slot][step] = (value
                                        - exact[i][order] * (1..=order).product::<usize>() as f64)
                                        .abs();
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
            }
        }
        components::assert_no_failures("complete C1 one-electron path matrices", &failures);
    }

    struct CartesianQuarticTerm {
        powers: [i32; 9],
        degree: usize,
        integrals: Vec<f64>,
    }

    /// At the undeformed geometry, expand the derivatives of the first three
    /// primitive Gaussians as Cartesian monomials. No displaced evaluations or
    /// Taylor coefficients enter this independent absolute-accuracy oracle.
    fn cartesian_quartic_terms(
        basis: &MolBasis,
        kernel: CoulombKernel,
    ) -> Vec<CartesianQuarticTerm> {
        use crate::basis::cart_components;
        use crate::integrals::eri::eri_quartet_masked;

        const FACTORIAL: [f64; 5] = [1.0, 1.0, 2.0, 6.0, 24.0];
        let mut terms = Vec::new();
        for degree in [0, 2, 4] {
            for a in 0..=degree {
                for b in 0..=degree - a {
                    let orders = [a, b, degree - a - b];
                    let shells: [_; 3] = std::array::from_fn(|i| {
                        let mut shell = basis.shells[i].clone();
                        assert_eq!(shell.l, 0);
                        assert_eq!(shell.exps.len(), 1);
                        // Keep the contraction coefficient of the original
                        // s Gaussian. The raised indices stay Cartesian.
                        shell.l = orders[i] as u8;
                        shell
                    });
                    let block = eri_quartet_masked(
                        &shells[0],
                        &shells[1],
                        &shells[2],
                        &basis.shells[3],
                        kernel,
                        [false, false, false, true],
                    );
                    let cartesian = shells.each_ref().map(|shell| cart_components(shell.l));
                    for (i, &(ax, ay, az)) in cartesian[0].iter().enumerate() {
                        for (j, &(bx, by, bz)) in cartesian[1].iter().enumerate() {
                            for (k, &(cx, cy, cz)) in cartesian[2].iter().enumerate() {
                                let powers = [ax, ay, az, bx, by, bz, cx, cy, cz];
                                let prefactor = FACTORIAL[degree]
                                    / powers
                                        .iter()
                                        .map(|&n| FACTORIAL[n as usize])
                                        .product::<f64>();
                                let integrals = (0..block.nd)
                                    .map(|ao| prefactor * block.get(i, j, k, ao))
                                    .collect();
                                terms.push(CartesianQuarticTerm {
                                    powers,
                                    degree,
                                    integrals,
                                });
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(terms.len(), 1 + 45 + 495);
        terms
    }

    fn cartesian_quartic_direction(
        terms: &[CartesianQuarticTerm],
        basis: &MolBasis,
        motion: &[[f64; 3]; 4],
        count: usize,
    ) -> Vec<f64> {
        assert_eq!(motion[3], [0.0; 3]);
        let mut displacement = [0.0_f64; 9];
        let mut quadratic = 0.0;
        for (i, vector) in motion[..3].iter().enumerate() {
            let exponent = basis.shells[i].exps[0];
            quadratic += exponent * vector.iter().map(|v| v * v).sum::<f64>();
            for (axis, value) in vector.iter().enumerate() {
                displacement[3 * i + axis] = 2.0 * exponent * value;
            }
        }
        // For the product of shifted s Gaussians, exp(L t - Q t^2),
        // the fourth derivative at zero is L^4 - 12 Q L^2 + 12 Q^2.
        let mut out = vec![0.0; count];
        for term in terms {
            let group = match term.degree {
                0 => 12.0 * quadratic * quadratic,
                2 => -12.0 * quadratic,
                4 => 1.0,
                _ => unreachable!(),
            };
            let multiplier = group
                * displacement
                    .iter()
                    .zip(term.powers)
                    .map(|(value, power)| value.powi(power))
                    .product::<f64>();
            for (value, integral) in out.iter_mut().zip(&term.integrals) {
                *value = multiplier.mul_add(*integral, *value);
            }
        }
        out
    }

    #[test]
    fn complete_quartic_eri_tensors_from_d_to_i_fit_real_quartet_differences() {
        use stencils::Stencil;

        // Uniform translation removes three centre coordinates. Fix the
        // high-angular shell and use a full-rank set in the other nine
        // relative Cartesian coordinates, independently for each AO entry.
        let directions = symmetric_cartesian_directions::directions(9, 4);
        assert_eq!(directions.len(), 495);
        let stages = [
            ([0.12, 0.09, 0.06, 0.04], Stencil::Second),
            ([0.06, 0.045, 0.03, 0.02], Stencil::Second),
            ([0.20, 0.15, 0.10, 0.07], Stencil::Second),
            ([0.12, 0.09, 0.06, 0.04], Stencil::Fourth),
            ([0.22, 0.16, 0.12, 0.08], Stencil::Fourth),
        ];
        let mut failures = Vec::new();
        for l in 2..=crate::basis::aos::MAX_L {
            let (molecule, basis) = fixture(l);
            let count = n_sph(l);
            for omega in [None, Some(0.43)] {
                let kernel =
                    omega.map_or(CoulombKernel::Coulomb, |omega| CoulombKernel::Erf { omega });
                let terms = cartesian_quartic_terms(&basis, kernel);
                let mut max_absolute_defect = 0.0_f64;
                let point = |motion: &[[f64; 3]; 4], h: f64| {
                    let shells: [_; 4] = std::array::from_fn(|i| {
                        let mut shell = basis.shells[i].clone();
                        let [x, y, z] = motion[i];
                        shell.center += Vec3::new(x, y, z) * h;
                        shell
                    });
                    eri_quartet(&shells[0], &shells[1], &shells[2], &shells[3], kernel).data
                };
                let centre = point(&[[0.0; 3]; 4], 0.0);
                assert_eq!(centre.len(), count);
                let translation = [[0.23, -0.31, 0.17]; 4];
                let translated = point(&translation, 0.31);
                let input = super::super::inputs(&molecule, &translation, 4);
                let invariant =
                    repulsion(&basis, &molecule, &input, [0, 1, 2, 3], omega, 4).unwrap();
                assert_eq!(invariant.len(), count);
                assert_eq!(translated.len(), count);
                for ((row, &value), &shifted) in invariant.iter().zip(&centre).zip(&translated) {
                    assert!(row[1..].iter().all(|&coefficient| coefficient == 0.0));
                    assert!((row[0] - value).abs() < 2e-13 * (1.0 + value.abs()));
                    assert!((shifted - value).abs() < 2e-13 * (1.0 + value.abs()));
                }
                for (index, direction) in directions.iter().enumerate() {
                    let mut motion = [[0.0; 3]; 4];
                    for (atom, shift) in direction.chunks_exact(3).enumerate() {
                        motion[atom].copy_from_slice(shift);
                    }
                    let input = super::super::inputs(&molecule, &motion, 4);
                    let coefficients =
                        repulsion(&basis, &molecule, &input, [0, 1, 2, 3], omega, 4).unwrap();
                    assert_eq!(coefficients.len(), count);
                    let oracle = cartesian_quartic_direction(&terms, &basis, &motion, count);
                    for (ao, (path, &reference)) in coefficients.iter().zip(&oracle).enumerate() {
                        let defect = (24.0 * path[4] - reference).abs();
                        max_absolute_defect = max_absolute_defect.max(defect);
                        assert!(
                            defect < 2e-13 * (1.0 + reference.abs()),
                            "C1 ERI l={l} omega={omega:?} AO={ao} direction={index} Cartesian oracle defect={defect:.17e}"
                        );
                    }
                    let names: Vec<_> = (0..count)
                        .map(|ao| {
                            format!("C1 ERI l={l} omega={omega:?} AO={ao} rank-4 direction={index}")
                        })
                        .collect();
                    let mut samples = BTreeMap::new();
                    stencils::fit_components_staged(
                        &names,
                        &stages,
                        |steps, stencil| {
                            let mut at = |h: f64| {
                                samples
                                    .entry(h.to_bits())
                                    .or_insert_with(|| point(&motion, h))
                                    .clone()
                            };
                            let mut near = |h: f64| {
                                let (plus, minus, plus2, minus2) =
                                    (at(h), at(-h), at(2.0 * h), at(-2.0 * h));
                                (0..count)
                                    .map(|ao| {
                                        // Form even differences around the same
                                        // value before the fourth differentiation.
                                        let inner =
                                            (plus[ao] - centre[ao]) + (minus[ao] - centre[ao]);
                                        let outer =
                                            (plus2[ao] - centre[ao]) + (minus2[ao] - centre[ao]);
                                        (outer - 4.0 * inner) / h.powi(4)
                                    })
                                    .collect::<Vec<_>>()
                            };
                            let mut errors = vec![[0.0; 4]; count];
                            for (step, &h) in steps.iter().enumerate() {
                                let first = near(h);
                                let second = (stencil == Stencil::Fourth).then(|| near(2.0 * h));
                                for (ao, error) in errors.iter_mut().enumerate() {
                                    let estimate = second
                                        .as_ref()
                                        .map_or(first[ao], |far| (4.0 * first[ao] - far[ao]) / 3.0);
                                    error[step] = (estimate - 24.0 * coefficients[ao][4]).abs();
                                }
                            }
                            errors
                        },
                        &mut failures,
                    );
                }
                println!("C1 quartic ERI l={l} omega={omega:?} Cartesian oracle max absolute defect={max_absolute_defect:.17e}");
            }
        }
        components::assert_no_failures(
            "all d/f/g/h/i fourth-order shell-quartet tensors",
            &failures,
        );
    }

    #[test]
    fn bounded_quartets_match_runtime_coefficients_through_i() {
        for l in 0..=crate::basis::aos::MAX_L {
            let (molecule, basis) = fixture(l);
            let input = curved_input(&molecule, 4);
            for omega in [None, Some(0.43)] {
                let expected = evaluate_series_many_result(&input, |v| {
                    evaluate(&basis, &molecule, v, [0, 1, 2, 3], omega, 4)
                })
                .unwrap();
                let actual = repulsion(&basis, &molecule, &input, [0, 1, 2, 3], omega, 4).unwrap();
                compare(&actual, &expected);
                let all_coordinates = input
                    .iter()
                    .map(|v| Taylor::<5>::from_coefficients(v))
                    .collect::<Vec<_>>();
                let previous =
                    evaluate(&basis, &molecule, &all_coordinates, [0, 1, 2, 3], omega, 4).unwrap();
                for (row, old) in actual.iter().zip(previous) {
                    for (&new, old) in row.iter().zip(old.c) {
                        assert_eq!(new.to_bits(), old.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn width_dispatch_preserves_curved_coefficients_and_runtime_fallback() {
        let (molecule, basis) = fixture(0);
        for order in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32] {
            let input = curved_input(&molecule, order);
            let expected = evaluate_series_many_result(&input, |v| {
                evaluate(&basis, &molecule, v, [0, 1, 2, 3], Some(0.43), order)
            })
            .unwrap();
            let actual =
                repulsion(&basis, &molecule, &input, [0, 1, 2, 3], Some(0.43), order).unwrap();
            compare(&actual, &expected);
        }
    }

    #[test]
    fn common_curved_translation_keeps_the_full_static_block() {
        let (molecule, basis) = fixture(2);
        let mut input = curved_input(&molecule, 4);
        for axis in 0..3 {
            for atom in 1..4 {
                let path = input[axis][1..].to_vec();
                input[3 * atom + axis][1..].copy_from_slice(&path);
            }
        }
        for indices in [[0, 1, 2, 3], [3, 3, 3, 3]] {
            for omega in [None, Some(0.43)] {
                let actual = repulsion(&basis, &molecule, &input, indices, omega, 4).unwrap();
                let expected = evaluate_series_many_result(&input, |v| {
                    evaluate(&basis, &molecule, v, indices, omega, 4)
                })
                .unwrap();
                compare(&actual, &expected);
                assert!(actual.iter().all(|row| row[1..].iter().all(|&x| x == 0.0)));
                assert!(actual.iter().any(|row| row[0] != 0.0));
            }
        }
    }

    #[test]
    fn workspace_counts_cartesian_and_hermite_storage() {
        let mut previous = 0;
        for l in 0..=crate::basis::aos::MAX_L {
            let bytes = workspace_bytes([l; 4], [1; 4], 4).unwrap().unwrap();
            assert!(bytes > previous);
            assert!(bytes >= 4 * n_cart(l).pow(4) * 8);
            previous = bytes;
        }
        assert!(previous < 128 * 1024 * 1024);
        assert!(workspace_bytes([6; 4], [1; 4], 32).unwrap().is_none());
        let compact = workspace_bytes([0; 4], [1; 4], 4).unwrap().unwrap();
        let contracted = workspace_bytes([0; 4], [1001; 4], 4).unwrap().unwrap();
        assert_eq!(contracted - compact, 4 * 1000 * 2 * 5 * 8);
        assert!(workspace_bytes([6, 6, 0, 0], [1; 4], 4).unwrap().unwrap() > 16 * 1024 * 1024);
        assert!(workspace_bytes([0; 4], [usize::MAX; 4], 4).is_err());
        let pair = pair_workspace_bytes([6, 6], [1, 1], 4, 4).unwrap().unwrap();
        assert!(pair < 2 * 1024 * 1024);
        let more_atoms = pair_workspace_bytes([6, 6], [1, 1], 1004, 4)
            .unwrap()
            .unwrap();
        assert_eq!(more_atoms - pair, 1000 * 10 * 5 * 8);
        assert!(pair_workspace_bytes([6, 6], [1, 1], 4, 32)
            .unwrap()
            .is_none());
        assert!(pair_workspace_bytes([0, 0], [1, 1], usize::MAX, 4).is_err());
    }

    #[test]
    fn static_shortcut_preserves_range_parameter_validation() {
        let (molecule, basis) = fixture(0);
        let input = curved_input(&molecule, 0);
        for omega in [f64::NAN, f64::INFINITY, -0.1] {
            assert!(matches!(
                repulsion(&basis, &molecule, &input, [0, 1, 2, 3], Some(omega), 0),
                Err(DftError::InvalidInput(_))
            ));
        }
    }
}
