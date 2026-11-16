// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic contractions of translated scalar ECP centre derivatives.

#[cfg(test)]
#[path = "ecp_second_tests.rs"]
mod second_tests;

use super::{
    dimensional_ecp_images_near_pair, dimensional_short_range_images, ecp_images_near_pair,
    short_range_images, ImageBlock, ShortRangeImage,
};
use crate::basis::{AtomEcp, MolBasis, MolShell};
use crate::integrals::ecp::{ecp_pair_path, EcpBraDerivative};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::{BlochMatrix, Cell, Lattice, PeriodicGeometry};
use crate::Result;

#[derive(Clone, Copy)]
pub(crate) enum EcpBoundary<'a> {
    Bulk(&'a Lattice),
    Dimensional(&'a PeriodicGeometry),
}

pub(crate) struct EcpGeometry {
    pub gradient: Vec<Vec3>,
    pub strain: [[f64; 3]; 3],
}

fn retained_images(
    basis: &MolBasis,
    boundary: EcpBoundary<'_>,
    cutoff: f64,
    retained: &[ImageBlock],
) -> Result<Vec<ShortRangeImage>> {
    if retained.is_empty() {
        return Ok(Vec::new());
    }
    let mut images = match boundary {
        EcpBoundary::Bulk(lattice) => short_range_images(basis, lattice, cutoff),
        EcpBoundary::Dimensional(geometry) => {
            dimensional_short_range_images(basis, geometry, cutoff)?
        }
    };
    images.retain(|image| {
        retained
            .iter()
            .any(|block| (block.translation - image.translation).norm() < 1e-10)
    });
    Ok(images)
}

/// Share the retained AO and projector image partition between derivative
/// orders. Construct first-derivative workspaces once per AO shell pair.
fn visit_image_pairs(
    basis: &MolBasis,
    boundary: EcpBoundary<'_>,
    cutoff: f64,
    image: ShortRangeImage,
    mut visit: impl FnMut(&MolShell, &MolShell, &[(usize, Vec3)]) -> Result<()>,
) -> Result<()> {
    let lattice = match boundary {
        EcpBoundary::Bulk(lattice) => lattice,
        EcpBoundary::Dimensional(geometry) => geometry.lattice(),
    };
    for (left, right) in image.shell_pairs {
        let a = &basis.shells[left];
        let mut b = basis.shells[right].clone();
        b.center += image.translation;
        let keys = match boundary {
            EcpBoundary::Bulk(lattice) => {
                ecp_images_near_pair(basis, lattice, a.center, b.center, cutoff)
            }
            EcpBoundary::Dimensional(geometry) => {
                dimensional_ecp_images_near_pair(basis, geometry, a.center, b.center, cutoff)?
            }
        };
        if !keys.is_empty() {
            let centers: Vec<_> = keys
                .into_iter()
                .map(|(index, translation)| {
                    (
                        index,
                        basis.ecps[index].center + lattice.translation(translation),
                    )
                })
                .collect();
            visit(a, &b, &centers)?;
        }
    }
    Ok(())
}

/// Translation invariance eliminates the third (ECP) centre derivative.
fn visit_pair_derivatives(
    basis: &MolBasis,
    boundary: EcpBoundary<'_>,
    cutoff: f64,
    image: ShortRangeImage,
    mut visit: impl FnMut(&MolShell, &MolShell, &AtomEcp, Vec3, &[Matrix; 3], &[Matrix; 3]),
) -> Result<()> {
    visit_image_pairs(basis, boundary, cutoff, image, |a, b, centers| {
        let bra = EcpBraDerivative::new(a);
        let ket = EcpBraDerivative::new(b);
        for &(index, center) in centers {
            let ecp = &basis.ecps[index];
            let da = bra.spherical(b, center, &ecp.potential);
            let db = ket.spherical(a, center, &ecp.potential);
            visit(a, b, ecp, center, &da, &db);
        }
        Ok(())
    })
}

/// Contract the two spherical centre derivatives with one real density image.
fn contract_pair_first(
    a: &MolShell,
    b: &MolShell,
    weight: &Matrix,
    da: &[Matrix; 3],
    db: &[Matrix; 3],
) -> [[f64; 3]; 2] {
    let mut first = [[0.0; 3]; 2];
    for axis in 0..3 {
        for i in 0..da[axis].rows {
            for j in 0..da[axis].cols {
                let density = weight[(a.sph_off + i, b.sph_off + j)];
                first[0][axis] += density * da[axis][(i, j)];
                first[1][axis] += density * db[axis][(j, i)];
            }
        }
    }
    first
}

fn relative_velocity(
    shell: &MolShell,
    ecp: &AtomEcp,
    center: Vec3,
    displacement: Option<&[[f64; 3]]>,
    affine: Option<&[[f64; 3]; 3]>,
) -> [f64; 3] {
    std::array::from_fn(|axis| {
        displacement.map_or(0.0, |u| u[shell.atom][axis] - u[ecp.atom][axis])
            + affine.map_or(0.0, |f| {
                (0..3)
                    .map(|j| f[axis][j] * (shell.center - center).get(j))
                    .sum()
            })
    })
}

/// Fixed-density nuclear Hessian of the translated scalar ECP operator.
/// The reference density images already contain the finite-k Fourier sum.
/// Each local bra/ket derivative scatters with +1 on its AO owner and -1
/// on the projector owner. This enforces translation covariance per term,
/// including coincident owners, without projecting the assembled Hessian.
pub(crate) fn contract_ecp_hessian(cell: &Cell, density: &[ImageBlock]) -> Result<Matrix> {
    let basis = cell.basis();
    let dim = 3 * cell.molecule().len();
    let mut hessian = Matrix::zeros(dim, dim);
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    for image in retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())? {
        let Some(weight) = crate::periodic::strain::find_block(density, image.translation) else {
            continue;
        };
        visit_image_pairs(basis, boundary, cutoff, image, |a, b, centers| {
            let cart = crate::periodic::strain::cartesian_density(weight, a, b);
            for &(index, center) in centers {
                let ecp = &basis.ecps[index];
                if a.atom == ecp.atom && b.atom == ecp.atom {
                    continue;
                }
                crate::integrals::ecp::check_ecp_second_order([a, b], &ecp.potential)?;
                let blocks =
                    crate::periodic::hessian::pair_second_blocks(a, b, &cart, &|left, right| {
                        crate::integrals::ecp::ecp_shell_pair(left, right, center, &ecp.potential)
                    });
                let owners = [a.atom, b.atom];
                for left in 0..2 {
                    for right in 0..2 {
                        for (x, row) in blocks[2 * left + right].iter().enumerate() {
                            for (y, &value) in row.iter().enumerate() {
                                for (i, si) in [(owners[left], 1.0), (ecp.atom, -1.0)] {
                                    for (j, sj) in [(owners[right], 1.0), (ecp.atom, -1.0)] {
                                        hessian[(3 * i + x, 3 * j + y)] += si * sj * value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(hessian)
}

/// Affine derivative of each retained ECP image block at fixed reduced k.
pub(crate) fn ecp_strain_blocks(
    cell: &Cell,
) -> Result<Vec<crate::periodic::strain_fock::StrainBlock>> {
    use crate::periodic::strain_fock::StrainBlock;
    let basis = cell.basis();
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())?
        .into_iter()
        .map(|image| {
            let mut block = StrainBlock::zeros(image.translation, basis.nao_sph);
            visit_pair_derivatives(basis, boundary, cutoff, image, |a, b, _, center, da, db| {
                let relative = [a.center - center, b.center - center];
                for x in 0..3 {
                    let ket = db[x].transpose();
                    for y in 0..3 {
                        block.add_sph(3 * x + y, a.sph_off, b.sph_off, &da[x], relative[0].get(y));
                        block.add_sph(3 * x + y, a.sph_off, b.sph_off, &ket, relative[1].get(y));
                    }
                }
            })?;
            Ok(block)
        })
        .collect()
}

/// Fixed-density affine and mixed strain/displacement ECP terms. In the
/// chart F(R+T+u), the mixed derivative also differentiates F times u.
pub(crate) fn ecp_strain_second(
    cell: &Cell,
    density: &[ImageBlock],
) -> Result<crate::periodic::strain::StrainSecond> {
    use crate::periodic::strain::{cartesian_density, find_block, StrainSecond};
    let basis = cell.basis();
    let natoms = cell.molecule().len();
    let dim = 3 * natoms;
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    let mut out = StrainSecond::zeros(natoms);
    for image in retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())? {
        let Some(weight) = find_block(density, image.translation) else {
            continue;
        };
        visit_image_pairs(basis, boundary, cutoff, image, |a, b, centers| {
            let cart = cartesian_density(weight, a, b);
            if cart.as_slice().iter().all(|&value| value == 0.0) {
                return Ok(());
            }
            let bra = EcpBraDerivative::new(a);
            let ket = EcpBraDerivative::new(b);
            for &(index, center) in centers {
                let ecp = &basis.ecps[index];
                let relative = [a.center - center, b.center - center];
                if [a.atom, b.atom] == [ecp.atom; 2]
                    && relative.iter().all(|r| r.to_array() == [0.0; 3])
                {
                    continue;
                }
                crate::integrals::ecp::check_ecp_second_order([a, b], &ecp.potential)?;
                let hessian =
                    crate::periodic::hessian::pair_second_blocks(a, b, &cart, &|left, right| {
                        crate::integrals::ecp::ecp_shell_pair(left, right, center, &ecp.potential)
                    });
                let owners = [a.atom, b.atom];
                let first = if owners == [ecp.atom; 2] {
                    [[0.0; 3]; 2]
                } else {
                    let da = bra.spherical(b, center, &ecp.potential);
                    let db = ket.spherical(a, center, &ecp.potential);
                    contract_pair_first(a, b, weight, &da, &db)
                };
                for x in 0..3 {
                    for y in 0..3 {
                        let strain = 3 * x + y;
                        for (left, r) in relative.iter().enumerate() {
                            for (right, s) in relative.iter().enumerate() {
                                for z in 0..3 {
                                    let value = hessian[2 * left + right][x][z] * r.get(y);
                                    for w in 0..3 {
                                        out.strain[9 * strain + 3 * z + w] += value * s.get(w);
                                    }
                                    if owners[right] != ecp.atom {
                                        out.mixed[dim * strain + 3 * owners[right] + z] += value;
                                        out.mixed[dim * strain + 3 * ecp.atom + z] -= value;
                                    }
                                }
                            }
                            if owners[left] != ecp.atom {
                                out.mixed[dim * strain + 3 * owners[left] + y] += first[left][x];
                                out.mixed[dim * strain + 3 * ecp.atom + y] -= first[left][x];
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(out)
}

/// The quadratic coefficient for F(t)(R+T+u(t)). Its chain rule is
/// one half of D(v1)^2 plus D(v2), with v2 = u2 + F2(R+T) + F1 u1.
pub(crate) fn gamma_ecp_second(
    cell: &Cell,
    path: &crate::periodic::path_series::LatticePath,
) -> Result<Matrix> {
    let basis = cell.basis();
    let displacement = path.displacement.first().map(Vec::as_slice);
    let affine = path.deformation.first();
    let acceleration: Vec<[f64; 3]> = (0..cell.molecule().len())
        .map(|atom| {
            std::array::from_fn(|axis| {
                path.displacement.get(1).map_or(0.0, |u| u[atom][axis])
                    + affine.map_or(0.0, |f| {
                        displacement.map_or(0.0, |u| (0..3).map(|j| f[axis][j] * u[atom][j]).sum())
                    })
            })
        })
        .collect();
    let curved = acceleration.iter().flatten().any(|&v| v != 0.0)
        || path
            .deformation
            .get(1)
            .is_some_and(|f| f.iter().flatten().any(|&v| v != 0.0));
    let mut coefficient = if curved {
        gamma_ecp_derivative(cell, Some(&acceleration), path.deformation.get(1))?
    } else {
        Matrix::zeros(basis.nao_sph, basis.nao_sph)
    };
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    for image in retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())? {
        visit_image_pairs(basis, boundary, cutoff, image, |a, b, centers| {
            for &(index, center) in centers {
                let ecp = &basis.ecps[index];
                let velocity =
                    [a, b].map(|shell| relative_velocity(shell, ecp, center, displacement, affine));
                if velocity.iter().flatten().all(|&v| v == 0.0) {
                    continue;
                }
                let coefficients =
                    ecp_pair_path([a, b], center, &ecp.potential, &[velocity, [[0.0; 3]; 2]])?;
                let block = &coefficients[2];
                for i in 0..block.rows {
                    for j in 0..block.cols {
                        coefficient[(a.sph_off + i, b.sph_off + j)] += block[(i, j)];
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(coefficient)
}

/// Higher coefficients of the scalar Gamma ECP operator. Translation
/// invariance lets us propagate the two AO centres relative to each image
/// of the projector, including products of deformation and displacement.
pub(crate) fn gamma_ecp_path(
    cell: &Cell,
    path: &crate::periodic::path_series::LatticePath,
    order: usize,
) -> Result<Vec<Matrix>> {
    let basis = cell.basis();
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    let mut result = vec![Matrix::zeros(basis.nao_sph, basis.nao_sph); order + 1];
    for image in retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())? {
        visit_image_pairs(basis, boundary, cutoff, image, |a, b, centers| {
            for &(index, center) in centers {
                let ecp = &basis.ecps[index];
                let motion: Vec<_> = (1..=order)
                    .map(|n| {
                        [a, b].map(|shell| {
                            let displacement = |k: usize, axis: usize| {
                                path.displacement
                                    .get(k - 1)
                                    .map_or(0.0, |u| u[shell.atom][axis] - u[ecp.atom][axis])
                            };
                            let relative = shell.center - center;
                            std::array::from_fn(|axis| {
                                displacement(n, axis)
                                    + (1..=n)
                                        .map(|k| {
                                            path.deformation.get(k - 1).map_or(0.0, |f| {
                                                (0..3)
                                                    .map(|j| {
                                                        f[axis][j]
                                                            * if k == n {
                                                                relative.get(j)
                                                            } else {
                                                                displacement(n - k, j)
                                                            }
                                                    })
                                                    .sum::<f64>()
                                            })
                                        })
                                        .sum::<f64>()
                            })
                        })
                    })
                    .collect();
                if motion.iter().flatten().flatten().all(|&v| v == 0.0) {
                    continue;
                }
                let coefficients = ecp_pair_path([a, b], center, &ecp.potential, &motion)?;
                for (target, block) in result.iter_mut().zip(coefficients).skip(1) {
                    for i in 0..block.rows {
                        for j in 0..block.cols {
                            target[(a.sph_off + i, b.sph_off + j)] += block[(i, j)];
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(result)
}

/// First scalar Gamma ECP coefficient for nuclear and affine motion.
pub(crate) fn gamma_ecp_derivative(
    cell: &Cell,
    displacement: Option<&[[f64; 3]]>,
    affine: Option<&[[f64; 3]; 3]>,
) -> Result<Matrix> {
    let basis = cell.basis();
    let boundary = EcpBoundary::Bulk(cell.lattice());
    let cutoff = cell.overlap_cutoff();
    let mut derivative = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    for image in retained_images(basis, boundary, cutoff, cell.ecp_image_blocks())? {
        visit_pair_derivatives(
            basis,
            boundary,
            cutoff,
            image,
            |a, b, ecp, center, da, db| {
                // Relative velocities include the affine motion of every image.
                let va = relative_velocity(a, ecp, center, displacement, affine);
                let vb = relative_velocity(b, ecp, center, displacement, affine);
                for axis in 0..3 {
                    for i in 0..da[axis].rows {
                        for j in 0..da[axis].cols {
                            derivative[(a.sph_off + i, b.sph_off + j)] +=
                                va[axis] * da[axis][(i, j)] + vb[axis] * db[axis][(j, i)];
                        }
                    }
                }
            },
        )?;
    }
    Ok(derivative)
}

/// Hold the reduced k points and the supplied Bloch density fixed. Both
/// the atom and all its images follow the same affine map. Evaluate the
/// ket derivative by interchanging the AO shells and the core derivative
/// by translation invariance of each individual ECP integral.
pub(crate) fn contract_ecp_geometry(
    basis: &MolBasis,
    boundary: EcpBoundary<'_>,
    cutoff: f64,
    atom_count: usize,
    retained: &[ImageBlock],
    points: &[Vec3],
    density: &[BlochMatrix],
) -> Result<EcpGeometry> {
    let mut result = EcpGeometry {
        gradient: vec![Vec3::zero(); atom_count],
        strain: [[0.0; 3]; 3],
    };
    let inv_nk = 1.0 / points.len() as f64;
    let n = basis.nao_sph;
    for image in retained_images(basis, boundary, cutoff, retained)? {
        let t = image.translation;
        let mut weights = Matrix::zeros(n, n);
        for (&k, p) in points.iter().zip(density) {
            let (sin, cos) = k.dot(t).sin_cos();
            for ((weight, re), im) in weights
                .as_mut_slice()
                .iter_mut()
                .zip(p.re.as_slice())
                .zip(p.im.as_slice())
            {
                *weight += inv_nk * (cos * re + sin * im);
            }
        }
        visit_pair_derivatives(
            basis,
            boundary,
            cutoff,
            image,
            |a, b, ecp, center, da, db| {
                let [ga, gb] = contract_pair_first(a, b, &weights, da, db);
                for axis in 0..3 {
                    for component in 0..3 {
                        result.strain[axis][component] += ga[axis]
                            * (a.center - center).get(component)
                            + gb[axis] * (b.center - center).get(component);
                    }
                }
                let ga = Vec3::new(ga[0], ga[1], ga[2]);
                let gb = Vec3::new(gb[0], gb[1], gb[2]);
                result.gradient[a.atom] += ga;
                result.gradient[b.atom] += gb;
                result.gradient[ecp.atom] -= ga + gb;
            },
        )?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::{components, stencils};
    use crate::periodic::features::PeriodicBoundary;
    use crate::system::{Atom, Molecule};
    use stencils::Stencil;

    #[test]
    fn translated_scalar_ecp_gradient_and_strain_fit_all_components() {
        const BASIS: &str = include_str!("../../../tests/data/periodic_scalar_ecp.g94");
        let set = crate::basis::g94::parse("scalar geometry fixture", BASIS).unwrap();
        let positions = [
            Vec3::new(0.13, -0.21, 0.07),
            Vec3::new(2.0, 0.45, -0.31),
            Vec3::new(-0.46, 1.29, 1.62),
        ];
        let vectors = [
            Vec3::new(4.8, 0.37, 0.18),
            Vec3::new(0.24, 5.18, 0.35),
            Vec3::new(-0.14, -0.22, 5.39),
        ];
        let re = Matrix::from_row_major(
            3,
            3,
            vec![0.67, 0.16, -0.09, 0.16, 0.54, 0.12, -0.09, 0.12, 0.28],
        );
        let im = Matrix::from_row_major(
            3,
            3,
            vec![0.0, 0.047, -0.06, -0.047, 0.0, 0.082, 0.06, -0.082, 0.0],
        );
        let mut conjugate = im.clone();
        for value in conjugate.as_mut_slice() {
            *value = -*value;
        }
        let density = [
            BlochMatrix::from_pair((re.clone(), Matrix::zeros(3, 3))),
            BlochMatrix::from_pair((re.clone(), im.clone())),
            BlochMatrix::from_pair((re, conjugate)),
        ];
        let cutoff = 4.17;
        for (axes, bulk) in [
            ([true, false, false], false),
            ([true, false, true], false),
            ([true; 3], false),
            ([true; 3], true),
        ] {
            let build_path = |numbers: [u8; 3], step: [f64; 18]| {
                let mut centers = positions;
                let mut rows = vectors;
                for vector in centers.iter_mut().chain(rows.iter_mut()) {
                    let affine: [f64; 3] = std::array::from_fn(|a| {
                        (0..3).map(|b| step[9 + 3 * a + b] * vector.get(b)).sum()
                    });
                    *vector += Vec3::new(affine[0], affine[1], affine[2]);
                }
                for (center, u) in centers.iter_mut().zip(step[..9].chunks_exact(3)) {
                    *center += Vec3::new(u[0], u[1], u[2]);
                }
                let mol = Molecule::new(
                    numbers
                        .into_iter()
                        .zip(centers)
                        .map(|(z, position)| Atom { z, position })
                        .collect(),
                );
                let mb = MolBasis::build(&mol, &set).unwrap();
                let geometry = PeriodicGeometry::new(
                    Lattice::new(rows[0], rows[1], rows[2]),
                    PeriodicBoundary::new(axes),
                )
                .unwrap();
                let blocks = if bulk {
                    super::super::ecp_image_blocks(&mb, geometry.lattice(), cutoff, cutoff)
                } else {
                    super::super::dimensional_ecp_image_blocks(&mb, &geometry, cutoff, cutoff)
                }
                .unwrap();
                assert!(blocks.iter().any(|block| block.translation.norm() > 1.0));
                (mb, geometry, blocks)
            };
            let build = |component: usize, h: f64| {
                build_path(
                    [1, 11, 1],
                    std::array::from_fn(|i| if component == i { h } else { 0.0 }),
                )
            };
            let (mb, geometry, blocks) = build(18, 0.0);
            let points = geometry.k_points([3, 1, 1]).unwrap();
            let boundary = if bulk {
                EcpBoundary::Bulk(geometry.lattice())
            } else {
                EcpBoundary::Dimensional(&geometry)
            };
            let analytic =
                contract_ecp_geometry(&mb, boundary, cutoff, 3, &blocks, &points, &density)
                    .unwrap();
            let derivatives: Vec<_> = analytic
                .gradient
                .iter()
                .flat_map(|v| v.to_array())
                .chain(analytic.strain.into_iter().flatten())
                .collect();
            if bulk {
                // A second core makes the onsite ECP diagonal responsive.
                // An isolated core gives a constant integral on its own AO.
                let numbers = [11, 11, 1];
                let cell = Cell::new(
                    Molecule::new(
                        numbers
                            .into_iter()
                            .zip(positions)
                            .map(|(z, position)| Atom { z, position })
                            .collect(),
                    ),
                    geometry.lattice().clone(),
                    &set,
                    cutoff,
                )
                .unwrap();
                // Columns of I + v 1^T span the entire input space. Mixing
                // the directions avoids zeros from isolated AO centres.
                let mixing: [f64; 18] = std::array::from_fn(|i| (i as f64 - 7.0) * 0.013);
                assert!((1.0 + mixing.iter().sum::<f64>()).abs() > 0.5);
                let directions: [[f64; 18]; 18] = std::array::from_fn(|column| {
                    std::array::from_fn(|row| mixing[row] + f64::from(row == column))
                });
                let mut matrices = Vec::new();
                for direction in &directions {
                    let displacement = std::array::from_fn::<_, 3, _>(|a| {
                        std::array::from_fn(|b| direction[3 * a + b])
                    });
                    let affine =
                        std::array::from_fn(|a| std::array::from_fn(|b| direction[9 + 3 * a + b]));
                    matrices.push(
                        gamma_ecp_derivative(&cell, Some(&displacement), Some(&affine)).unwrap(),
                    );
                }
                let entries: Vec<_> = (0..18)
                    .flat_map(|component| {
                        (0..3).flat_map(move |a| (0..=a).map(move |b| (component, a, b)))
                    })
                    .collect();
                let labels: Vec<_> = entries
                    .iter()
                    .map(|&(c, a, b)| format!("Gamma ECP AO coefficient {c} [{a},{b}]"))
                    .collect();
                let mut failed = Vec::new();
                stencils::fit_components_staged(
                    &labels,
                    &[
                        ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
                        ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
                        ([0.01, 0.007, 0.005, 0.0035], Stencil::Fourth),
                    ],
                    |steps, stencil| {
                        let references: Vec<_> = (0..18)
                            .map(|component| {
                                steps.map(|h| {
                                    let evaluate = |step| {
                                        let (_, _, blocks) = build_path(
                                            numbers,
                                            directions[component].map(|d| step * d),
                                        );
                                        super::super::bloch(&blocks, Vec3::zero()).0
                                    };
                                    let plus = evaluate(h);
                                    let minus = evaluate(-h);
                                    let (plus2, minus2) = if stencil == Stencil::Fourth {
                                        (evaluate(2.0 * h), evaluate(-2.0 * h))
                                    } else {
                                        (Matrix::zeros(3, 3), Matrix::zeros(3, 3))
                                    };
                                    std::array::from_fn::<_, 9, _>(|i| {
                                        let (a, b) = (i / 3, i % 3);
                                        stencil.first(
                                            plus[(a, b)],
                                            minus[(a, b)],
                                            plus2[(a, b)],
                                            minus2[(a, b)],
                                            h,
                                        )
                                    })
                                })
                            })
                            .collect();
                        entries
                            .iter()
                            .map(|&(c, a, b)| {
                                references[c].map(|m| (m[3 * a + b] - matrices[c][(a, b)]).abs())
                            })
                            .collect()
                    },
                    &mut failed,
                );
                components::assert_no_failures("Gamma ECP AO derivatives", &failed);
                for matrix in &matrices {
                    assert!(matrix.max_abs_diff(&matrix.transpose()) < 1e-13);
                }
                let translated =
                    gamma_ecp_derivative(&cell, Some(&[[0.2, -0.1, 0.3]; 3]), None).unwrap();
                assert!(translated.as_slice().iter().all(|&value| value == 0.0));
            }
            assert!(
                analytic
                    .gradient
                    .into_iter()
                    .fold(Vec3::zero(), |a, b| a + b)
                    .norm()
                    < 1e-13
            );
            let names: Vec<_> = (0..18)
                .map(|component| format!("scalar ECP {axes:?} bulk={bulk} component={component}"))
                .collect();
            let energy = |component, h| {
                let (_, geometry, blocks) = build(component, h);
                geometry
                    .k_points([3, 1, 1])
                    .unwrap()
                    .into_iter()
                    .zip(&density)
                    .map(|(k, p)| {
                        p.trace_product(&BlochMatrix::from_pair(super::super::bloch(&blocks, k)))
                    })
                    .sum::<f64>()
                    / 3.0
            };
            println!("Scalar ECP fixed-density axes={axes:?} bulk={bulk} energy={:.17e} derivatives={derivatives:?}", energy(18, 0.0));
            let mut failures = Vec::new();
            stencils::fit_components_staged(
                &names,
                &[
                    ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
                    ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
                ],
                |steps, stencil| {
                    derivatives
                        .iter()
                        .enumerate()
                        .map(|(component, &exact)| {
                            steps.map(|h| {
                                (stencil.first(
                                    energy(component, h),
                                    energy(component, -h),
                                    0.0,
                                    0.0,
                                    h,
                                ) - exact)
                                    .abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
            components::assert_no_failures("translated scalar ECP operator derivatives", &failures);
        }
    }
}
