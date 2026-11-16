// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Stationary PT2 amplitude response in a moving orthonormal frame.
//! Occupied and virtual Fock blocks may be fully nondiagonal. The same
//! coefficient equations apply at arbitrary order without within-space
//! eigenvector derivatives or degenerate orbital energy divisions.

use super::pt2::Pt2Problem;
use crate::linalg::Matrix;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::{DftError, Result};
use rayon::prelude::*;

mod frequency;
mod generalized;
mod retained;
pub use frequency::{Pt2FrequencyKernel, Pt2FrequencyResponse, Pt2FrequencyVertex};
pub use generalized::{generalized_taylor_response, spinor_transform_path, GeneralizedPt2Taylor};

#[cfg(test)]
mod convergence {
    include!("../../tests/support/convergence.rs");
}

/// Direct analytic rotation between occupied projectors in an orthonormal
/// frame. Degenerate occupied/virtual subspaces need no eigenvector gauge.
pub(crate) fn projector_frames(projectors: &[Matrix], occupied: usize) -> (Vec<Matrix>, f64) {
    use crate::response::nuclear::{matrix_product, metric_transport};
    let order = projectors.len() - 1;
    let nmo = projectors[0].rows;
    let mut q = vec![Matrix::identity(nmo)];
    for p in projectors.iter().skip(1) {
        let mut value = p.clone();
        for i in 0..nmo {
            for j in occupied..nmo {
                value[(i, j)] = -value[(i, j)];
            }
        }
        q.push(value);
    }
    let qt: Vec<_> = q.iter().map(Matrix::transpose).collect();
    let gram = matrix_product(&qt, &q, order);
    let normalizer = metric_transport(&Matrix::identity(nmo), &gram, order);
    let frame = matrix_product(&q, &normalizer, order);
    let transpose: Vec<_> = frame.iter().map(Matrix::transpose).collect();
    let check = matrix_product(&transpose, &frame, order);
    let mut defect = 0.0_f64;
    for n in 0..=order {
        for i in 0..nmo {
            for j in 0..nmo {
                let exact = if n == 0 && i == j { 1.0 } else { 0.0 };
                defect = defect.max((check[n][(i, j)] - exact).abs());
            }
        }
    }
    (frame, defect)
}

/// Complex counterpart of the direct projector rotation. Uses the same
/// Hermitian metric transport as the London stationary reference, without
/// eigenvector derivatives inside degenerate occupied/virtual subspaces.
pub fn complex_projector_frames(
    projectors: &[crate::integrals::london::ComplexMatrix],
    occupied: usize,
) -> Result<(Vec<crate::integrals::london::ComplexMatrix>, f64)> {
    use crate::integrals::london::ComplexMatrix as Cm;
    use crate::response::field_path::{adjoint, series_product, transport};
    let first = projectors
        .first()
        .ok_or_else(|| DftError::InvalidInput("complex projector path cannot be empty".into()))?;
    let n = first.re.rows;
    if occupied > n
        || projectors.iter().any(|p| {
            p.re.rows != n
                || p.re.cols != n
                || p.im.rows != n
                || p.im.cols != n
                || p.re
                    .as_slice()
                    .iter()
                    .chain(p.im.as_slice())
                    .any(|x| !x.is_finite())
        })
    {
        return Err(DftError::InvalidInput(
            "complex projector dimensions or occupations disagree".into(),
        ));
    }
    let order = projectors.len() - 1;
    let mut identity = Cm::zeros(n, n);
    identity.re = Matrix::identity(n);
    let mut defect = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let exact = if i == j && i < occupied { 1.0 } else { 0.0 };
            defect = defect
                .max((first.re[(i, j)] - exact).abs())
                .max(first.im[(i, j)].abs());
        }
    }
    if defect > 1e-8 {
        return Err(DftError::InvalidInput(
            "complex projector base must be the semicanonical occupied projector".into(),
        ));
    }
    let mut q = vec![identity.clone()];
    for p in &projectors[1..] {
        let mut p = p.clone();
        for i in 0..n {
            for j in occupied..n {
                p.re[(i, j)] = -p.re[(i, j)];
                p.im[(i, j)] = -p.im[(i, j)];
            }
        }
        q.push(p);
    }
    let adjoints: Vec<_> = q.iter().map(adjoint).collect();
    let gram = series_product(&adjoints, &q, order);
    let inverse = transport(&identity, &gram, order);
    let frames = series_product(&q, &inverse, order);
    let adjoints: Vec<_> = frames.iter().map(adjoint).collect();
    let metric = series_product(&adjoints, &frames, order);
    for (k, value) in metric.iter().enumerate() {
        for i in 0..n {
            for j in 0..n {
                let exact = if k == 0 && i == j { 1.0 } else { 0.0 };
                defect = defect
                    .max((value.re[(i, j)] - exact).abs())
                    .max(value.im[(i, j)].abs());
            }
        }
    }
    let occupied_frames: Vec<_> = frames
        .iter()
        .map(|u| {
            let mut u = u.clone();
            for i in 0..n {
                for j in occupied..n {
                    u.re[(i, j)] = 0.0;
                    u.im[(i, j)] = 0.0;
                }
            }
            u
        })
        .collect();
    let projected = series_product(&occupied_frames, &adjoints, order);
    for (a, b) in projected.iter().zip(projectors) {
        defect = defect
            .max(a.re.max_abs_diff(&b.re))
            .max(a.im.max_abs_diff(&b.im));
    }
    Ok((frames, defect))
}

/// Moving AO integrals and moving AO-to-MO Taylor coefficients.
/// Reused by molecular and periodic response; O(N^4) tensor storage.
pub(crate) fn transform_path<const W: usize>(
    nao: usize,
    integrals: &[Vec<f64>],
    orbitals: &[Vec<Matrix>; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> Result<Vec<Pt2Problem<f64>>> {
    if W == 0 || orbitals.iter().any(|c| c.len() != W) || integrals.len() != W {
        return Err(DftError::InvalidInput(
            "PT2 scalar width differs from its integral/orbital path".into(),
        ));
    }
    let size = nao
        .checked_pow(4)
        .ok_or_else(|| DftError::InvalidInput("PT2 path tensor size overflow".into()))?;
    if integrals
        .iter()
        .any(|g| g.len() != size || g.iter().any(|v| !v.is_finite()))
        || orbitals.iter().enumerate().any(|(s, path)| {
            path.iter().any(|c| {
                c.rows != nao
                    || c.cols != energies[s].len()
                    || c.as_slice().iter().any(|v| !v.is_finite())
            })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid PT2 AO integral or orbital Taylor coefficients".into(),
        ));
    }
    transform_real_path(
        nao,
        integrals,
        [orbitals; 4],
        energies,
        occupied,
        PairProduct::Taylor,
        None,
    )
}

/// The same four packed one-index transformations with retained orbital
/// and intermediate integral remainders. Sources and physical denominators
/// are unchanged; the low parts enter the PT2 calculation itself.
pub(crate) fn transform_path_retained(
    nao: usize,
    integrals: &[Vec<f64>],
    orbitals: &[crate::response::nuclear::precision::Series; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> Result<Vec<Pt2Problem<f64>>> {
    let high = std::array::from_fn(|s| orbitals[s].iter().map(|c| c.high.clone()).collect());
    transform_real_path(
        nao,
        integrals,
        [&high; 4],
        energies,
        occupied,
        PairProduct::Taylor,
        Some(orbitals),
    )
}

#[derive(Clone, Copy)]
enum PairProduct {
    Taylor,
    LabelledSubsets,
}

impl PairProduct {
    fn partner(self, coefficient: usize, left: usize) -> Option<usize> {
        match self {
            Self::Taylor => Some(coefficient - left),
            Self::LabelledSubsets if left & coefficient == left => Some(coefficient ^ left),
            Self::LabelledSubsets => None,
        }
    }
}

/// Coefficient convolution of four one-index transformations. Each dense
/// contraction uses faer's borrowed row-major SIMD kernel. Independent
/// spin channels and coefficient orders share Rayon's existing pool; dense
/// products inside that parallel work stay sequential to avoid oversubscription.
fn transform_real_path(
    nao: usize,
    integrals: &[Vec<f64>],
    orbital_paths: [&[Vec<Matrix>; 2]; 4],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
    product: PairProduct,
    retained_orbitals: Option<&[crate::response::nuclear::precision::Series; 2]>,
) -> Result<Vec<Pt2Problem<f64>>> {
    let width = integrals.len();
    if occupied.iter().zip(energies).any(|(no, e)| *no > e.len()) {
        return Err(DftError::InvalidInput(
            "PT2 occupied count exceeds the orbital space".into(),
        ));
    }
    let channels: Result<Vec<Vec<Vec<f64>>>> = [(0usize, 0usize), (0, 1), (1, 1)]
        .into_par_iter()
        .map(|(left, right)| {
            let spins = [left, left, right, right];
            let mut shape = [nao; 4];
            let mut tensor = integrals.to_vec();
            let mut tensor_low = retained_orbitals.map(|_| vec![vec![0.0; nao.pow(4)]; width]);
            for axis in 0..4 {
                let spin = spins[axis];
                let (offset, count) = if axis % 2 == 0 {
                    (0, occupied[spin])
                } else {
                    (occupied[spin], energies[spin].len() - occupied[spin])
                };
                let rows = shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != axis)
                    .map(|(_, n)| *n)
                    .product::<usize>();
                let coordinates: Vec<[usize; 4]> = (0..rows)
                    .map(|mut row| {
                        let mut x = [0; 4];
                        for a in (0..4).rev() {
                            if a != axis {
                                x[a] = row % shape[a];
                                row /= shape[a];
                            }
                        }
                        x
                    })
                    .collect();
                let flat = |x: [usize; 4], shape: [usize; 4]| {
                    ((x[0] * shape[1] + x[1]) * shape[2] + x[2]) * shape[3] + x[3]
                };
                let packed: Vec<_> = tensor
                    .par_iter()
                    .map(|plane| {
                        let mut matrix = Matrix::zeros(rows, nao);
                        for (row, x) in coordinates.iter().enumerate() {
                            for mu in 0..nao {
                                let mut source = *x;
                                source[axis] = mu;
                                matrix[(row, mu)] = plane[flat(source, shape)];
                            }
                        }
                        matrix
                    })
                    .collect();
                let coefficients: Vec<_> = orbital_paths[axis][spin]
                    .iter()
                    .map(|c| {
                        let mut matrix = Matrix::zeros(nao, count);
                        for mu in 0..nao {
                            for m in 0..count {
                                matrix[(mu, m)] = c[(mu, offset + m)];
                            }
                        }
                        matrix
                    })
                    .collect();
                let mut next_shape = shape;
                next_shape[axis] = count;
                let size = next_shape
                    .iter()
                    .try_fold(1usize, |n, k| n.checked_mul(*k))
                    .ok_or_else(|| {
                        DftError::InvalidInput("PT2 transformed path size overflow".into())
                    })?;
                let retained = retained_orbitals.map(|orbitals| {
                    use crate::response::nuclear::precision::MatrixParts;
                    let left = packed
                        .iter()
                        .enumerate()
                        .map(|(n, high)| {
                            let mut low = Matrix::zeros(rows, nao);
                            for (row, x) in coordinates.iter().enumerate() {
                                for mu in 0..nao {
                                    let mut source = *x;
                                    source[axis] = mu;
                                    low[(row, mu)] =
                                        tensor_low.as_ref().expect("retained tensor allocated")[n]
                                            [flat(source, shape)];
                                }
                            }
                            MatrixParts {
                                high: high.clone(),
                                low,
                            }
                        })
                        .collect::<Vec<_>>();
                    let right = coefficients
                        .iter()
                        .enumerate()
                        .map(|(n, high)| {
                            let mut low = Matrix::zeros(nao, count);
                            for mu in 0..nao {
                                for m in 0..count {
                                    low[(mu, m)] = orbitals[spin][n].low[(mu, offset + m)];
                                }
                            }
                            MatrixParts {
                                high: high.clone(),
                                low,
                            }
                        })
                        .collect::<Vec<_>>();
                    (left, right)
                });
                let transformed: Vec<_> = (0..width)
                    .into_par_iter()
                    .map(|n| {
                        let mut sum = Matrix::zeros(rows, count);
                        let mut low = None;
                        if let Some((a, b)) = &retained {
                            let result = crate::response::nuclear::precision::coefficient(a, b, n);
                            sum = result.high;
                            low = Some(result.low);
                        } else {
                            for k in 0..=n {
                                let Some(partner) = product.partner(n, k) else {
                                    continue;
                                };
                                let term = packed[k].matmul_seq(&coefficients[partner]);
                                for (v, t) in sum.as_mut_slice().iter_mut().zip(term.as_slice()) {
                                    *v += t;
                                }
                            }
                        }
                        let mut plane = vec![0.0; size];
                        let mut remainder = retained.as_ref().map(|_| vec![0.0; size]);
                        for (row, x) in coordinates.iter().enumerate() {
                            for m in 0..count {
                                let mut target = *x;
                                target[axis] = m;
                                plane[flat(target, next_shape)] = sum[(row, m)];
                                if let Some(remainder) = &mut remainder {
                                    remainder[flat(target, next_shape)] = low
                                        .as_ref()
                                        .expect("retained matrix has its low part")[(row, m)];
                                }
                            }
                        }
                        (plane, remainder)
                    })
                    .collect();
                if retained_orbitals.is_some() {
                    let (high, low) = transformed
                        .into_iter()
                        .map(|(high, low)| {
                            (
                                high,
                                low.expect("retained contraction returns its remainder"),
                            )
                        })
                        .unzip();
                    tensor = high;
                    tensor_low = Some(low);
                } else {
                    tensor = transformed.into_iter().map(|(high, _)| high).collect();
                }
                shape = next_shape;
            }
            Ok(tensor)
        })
        .collect();
    let channels = channels?;
    Ok((0..width)
        .map(|n| Pt2Problem {
            energies: energies.clone(),
            occupied,
            ovov: std::array::from_fn(|s| channels[s][n].clone()),
        })
        .collect())
}

/// Complex London AO-to-MO paths, using the same packed real SIMD
/// transformations as the real nuclear/electric provider. The returned
/// [i,a,j,b] source is (a i|b j), the conjugate of chemist (i a|j b).
/// Occupied coefficients are kets, virtual coefficients are bras; the AO
/// integral is conjugated as well. Exact-zero coefficient sectors are skipped,
/// without any numerical screening threshold.
pub fn complex_transform_path(
    nao: usize,
    integrals: &[Vec<crate::response::complex::Complex>],
    orbitals: &[Vec<crate::integrals::london::ComplexMatrix>; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> Result<Vec<Pt2Problem<crate::response::complex::Complex>>> {
    use crate::response::complex::Complex;
    let width = integrals.len();
    let size = nao
        .checked_pow(4)
        .ok_or_else(|| DftError::InvalidInput("complex PT2 integral size overflow".into()))?;
    if width == 0
        || nao == 0
        || energies.iter().flatten().any(|e| !e.is_finite())
        || integrals
            .iter()
            .any(|g| g.len() != size || g.iter().any(|z| !z.is_finite()))
        || orbitals.iter().enumerate().any(|(spin, c)| {
            c.len() != width
                || c.iter().any(|c| {
                    c.re.rows != nao
                        || c.re.cols != energies[spin].len()
                        || c.im.rows != c.re.rows
                        || c.im.cols != c.re.cols
                        || c.re
                            .as_slice()
                            .iter()
                            .chain(c.im.as_slice())
                            .any(|z| !z.is_finite())
                })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid complex PT2 integral/orbital Taylor path".into(),
        ));
    }
    let real: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| orbitals[s].iter().map(|c| c.re.clone()).collect());
    let imaginary: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| orbitals[s].iter().map(|c| c.im.clone()).collect());
    let tensors: [Vec<Vec<f64>>; 2] = std::array::from_fn(|part| {
        integrals
            .iter()
            .map(|g| {
                g.iter()
                    .map(|z| if part == 0 { z.re } else { z.im })
                    .collect()
            })
            .collect()
    });
    let first_orbital = imaginary
        .iter()
        .map(|c| {
            c.iter()
                .position(|m| m.as_slice().iter().any(|x| *x != 0.0))
                .unwrap_or(width)
        })
        .min()
        .unwrap_or(width);
    let first_integral = tensors[1]
        .iter()
        .position(|g| g.iter().any(|x| *x != 0.0))
        .unwrap_or(width);
    let mut result: Option<Vec<Pt2Problem<Complex>>> = None;
    for mask in 0u8..32 {
        let axes = mask & 15;
        let integral = (mask >> 4) as usize;
        if axes.count_ones() as usize * first_orbital + integral * first_integral >= width {
            continue;
        }
        let transformed = transform_real_path(
            nao,
            &tensors[integral],
            std::array::from_fn(|axis| {
                if axes & (1 << axis) == 0 {
                    &real
                } else {
                    &imaginary
                }
            }),
            energies,
            occupied,
            PairProduct::Taylor,
            None,
        )?;
        let power = mask.count_ones() % 4;
        let sign = if (mask & 0b11010).count_ones() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let phase = match power {
            0 => Complex::from(sign),
            1 => Complex::new(0.0, sign),
            2 => Complex::from(-sign),
            _ => Complex::new(0.0, -sign),
        };
        let out = result.get_or_insert_with(|| {
            transformed
                .iter()
                .map(|p| Pt2Problem {
                    energies: energies
                        .each_ref()
                        .map(|e| e.iter().copied().map(Complex::from).collect()),
                    occupied,
                    ovov: p.ovov.each_ref().map(|g| vec![Complex::ZERO; g.len()]),
                })
                .collect()
        });
        for (out, input) in out.iter_mut().zip(transformed) {
            for spin in 0..3 {
                for (a, b) in out.ovov[spin].iter_mut().zip(input.ovov[spin].iter()) {
                    *a += phase * *b;
                }
            }
        }
    }
    result.ok_or_else(|| DftError::InvalidInput("complex PT2 path has no base coefficient".into()))
}

/// Two labelled Fourier legs in subset order [0, a, b, ab].
/// The left and right analytic orbital frames are supplied independently.
/// Occupied slots use the first frame and virtual slots use the second.
pub(crate) fn transform_frequency_path(
    nao: usize,
    integrals: &[f64],
    occupied_frame: &[Vec<crate::integrals::london::ComplexMatrix>; 2],
    virtual_frame: &[Vec<crate::integrals::london::ComplexMatrix>; 2],
    energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
) -> Result<Vec<[Vec<crate::response::complex::Complex>; 3]>> {
    use crate::response::complex::Complex;
    let size = nao
        .checked_pow(4)
        .ok_or_else(|| DftError::InvalidInput("Fourier PT2 integral size overflow".into()))?;
    let frames = [occupied_frame, virtual_frame];
    if nao == 0
        || integrals.len() != size
        || integrals.iter().any(|v| !v.is_finite())
        || frames.iter().any(|frame| {
            frame.iter().zip(energies).any(|(lanes, e)| {
                lanes.len() != 4
                    || lanes.iter().any(|c| {
                        c.re.rows != nao
                            || c.re.cols != e.len()
                            || c.im.rows != nao
                            || c.im.cols != e.len()
                            || c.re
                                .as_slice()
                                .iter()
                                .chain(c.im.as_slice())
                                .any(|v| !v.is_finite())
                    })
            })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid Fourier PT2 orbital or integral coefficients".into(),
        ));
    }
    let parts: [[[Vec<Matrix>; 2]; 2]; 2] = frames.map(|frame| {
        [
            frame
                .each_ref()
                .map(|c| c.iter().map(|c| c.re.clone()).collect()),
            frame
                .each_ref()
                .map(|c| c.iter().map(|c| c.im.clone()).collect()),
        ]
    });
    let imaginary_degree = parts.each_ref().map(|part| {
        part[1]
            .iter()
            .flatten()
            .enumerate()
            .filter(|(_, lane)| lane.as_slice().iter().any(|&v| v != 0.0))
            .map(|(index, _)| (index % 4).count_ones())
            .min()
            .unwrap_or(3)
    });
    let lanes: Vec<_> = (0..4)
        .map(|lane| {
            if lane == 0 {
                integrals.to_vec()
            } else {
                vec![0.0; size]
            }
        })
        .collect();
    let mut result: Option<Vec<[Vec<Complex>; 3]>> = None;
    for mask in 0u32..16 {
        let degree: u32 = (0..4)
            .filter(|axis| mask & (1 << axis) != 0)
            .map(|axis| imaginary_degree[axis % 2])
            .sum();
        if degree > 2 {
            continue;
        }
        let selected = std::array::from_fn(|axis| &parts[axis % 2][((mask >> axis) & 1) as usize]);
        let transformed = transform_real_path(
            nao,
            &lanes,
            selected,
            energies,
            occupied,
            PairProduct::LabelledSubsets,
            None,
        )?;
        let phase = match mask.count_ones() % 4 {
            0 => Complex::ONE,
            1 => Complex::new(0.0, 1.0),
            2 => Complex::from(-1.0),
            _ => Complex::new(0.0, -1.0),
        };
        let out = result.get_or_insert_with(|| {
            transformed
                .iter()
                .map(|p| p.ovov.each_ref().map(|v| vec![Complex::ZERO; v.len()]))
                .collect()
        });
        for (out, input) in out.iter_mut().zip(transformed) {
            for (out, input) in out.iter_mut().zip(input.ovov) {
                for (element, coefficient) in out.iter_mut().zip(input) {
                    *element += phase * coefficient;
                }
            }
        }
    }
    result.ok_or_else(|| {
        DftError::InvalidInput("Fourier PT2 path has no reference coefficient".into())
    })
}

#[derive(Clone, Debug)]
pub struct Pt2LinearResponse {
    pub same_spin_alpha: f64,
    pub same_spin_beta: f64,
    pub opposite_spin: f64,
    pub correlation: f64,
    pub amplitude_derivatives: [Vec<f64>; 3],
    /// max |Delta t' + Delta' t - g'| on the physical amplitude equation.
    pub residual: f64,
}

/// Factorial-normalized coefficients, including the unperturbed solution.
#[derive(Clone, Debug)]
pub struct Pt2TaylorResponse<S = f64> {
    pub same_spin_alpha: Vec<S>,
    pub same_spin_beta: Vec<S>,
    pub opposite_spin: Vec<S>,
    pub correlation: Vec<S>,
    /// [Taylor order][alpha-alpha,alpha-beta,beta-beta][i,a,j,b].
    pub amplitude_coefficients: Vec<[Vec<S>; 3]>,
    /// Recomputed max absolute coefficient residual for every order.
    pub residuals: Vec<f64>,
    /// Max over spin channels of ||A0 t_n - rhs_n||_2 / ||rhs_n||_2,
    /// with rhs_n = g_n - sum_(k>0) A_k t_(n-k). No norm floor is used.
    /// A zero right-hand side requires an exactly zero residual.
    pub relative_residuals: Vec<f64>,
}

fn index(shape: [usize; 4], i: usize, a: usize, j: usize, b: usize) -> usize {
    ((i * shape[1] + a) * shape[2] + j) * shape[3] + b
}

/// Minimal field algebra shared by real and complex amplitude paths.
trait AmplitudeNumber:
    Copy
    + From<f64>
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<f64, Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
{
    fn conjugate(self) -> Self;
    fn magnitude(self) -> f64;
    fn finite(self) -> bool;
}
impl AmplitudeNumber for f64 {
    fn conjugate(self) -> Self {
        self
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}
impl AmplitudeNumber for crate::response::complex::Complex {
    fn conjugate(self) -> Self {
        self.conj()
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
}
trait FockCoefficient<S: AmplitudeNumber> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn element(&self, row: usize, col: usize) -> S;
    fn finite(&self) -> bool;
}
impl FockCoefficient<f64> for Matrix {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn element(&self, row: usize, col: usize) -> f64 {
        self[(row, col)]
    }
    fn finite(&self) -> bool {
        self.as_slice().iter().all(|v| v.is_finite())
    }
}
impl FockCoefficient<crate::response::complex::Complex>
    for crate::integrals::london::ComplexMatrix
{
    fn rows(&self) -> usize {
        self.re.rows
    }
    fn cols(&self) -> usize {
        self.re.cols
    }
    fn element(&self, row: usize, col: usize) -> crate::response::complex::Complex {
        crate::response::complex::Complex::new(self.re[(row, col)], self.im[(row, col)])
    }
    fn finite(&self) -> bool {
        self.im.rows == self.re.rows
            && self.im.cols == self.re.cols
            && self
                .re
                .as_slice()
                .iter()
                .chain(self.im.as_slice())
                .all(|v| v.is_finite())
    }
}

/// Action of an occupied/virtual Fock coefficient on one pair amplitude.
fn pair_action<S: AmplitudeNumber, F: FockCoefficient<S>>(
    shape: [usize; 4],
    f: [&F; 2],
    t: &[S],
) -> Vec<S> {
    let [nl, vl, nr, vr] = shape;
    let mut result = vec![S::from(0.0); t.len()];
    for i in 0..nl {
        for a in 0..vl {
            for j in 0..nr {
                for b in 0..vr {
                    let p = index(shape, i, a, j, b);
                    for k in 0..nl {
                        result[p] += f[0].element(k, i) * t[index(shape, k, a, j, b)];
                    }
                    for k in 0..nr {
                        result[p] += f[1].element(k, j) * t[index(shape, i, a, k, b)];
                    }
                    for c in 0..vl {
                        result[p] -= f[0].element(nl + a, nl + c) * t[index(shape, i, c, j, b)];
                    }
                    for c in 0..vr {
                        result[p] -= f[1].element(nr + b, nr + c) * t[index(shape, i, a, j, c)];
                    }
                }
            }
        }
    }
    result
}

/// Shared occupied-minus-virtual tensor action. The many-electron
/// one-body connection on a double-substitution coefficient is its
/// occupied trace times that coefficient minus this action.
pub(crate) fn complex_pair_action(
    shape: [usize; 4],
    matrices: [&crate::integrals::london::ComplexMatrix; 2],
    amplitudes: &[crate::response::complex::Complex],
) -> Vec<crate::response::complex::Complex> {
    pair_action(shape, matrices, amplitudes)
}

/// Real counterpart, sharing the stationary amplitude operator.
pub(crate) fn real_pair_action(
    shape: [usize; 4],
    matrices: [&Matrix; 2],
    amplitudes: &[f64],
) -> Vec<f64> {
    pair_action(shape, matrices, amplitudes)
}

/// Solve A_0 t_n = g_n - sum_(k=1..n) A_k t_(n-k).
/// Entries of the two input slices are coefficients of t^1, t^2, ...,
/// not derivatives; their lengths agree. The semicanonical A_0 and g_0
/// come from `problem`. Higher Fock coefficients include the induced field
/// and moving-frame terms. Orbital/density response is supplied by the
/// reference solver, rather than approximated inside this contraction.
pub fn taylor_response(
    problem: &Pt2Problem<f64>,
    ovov_coefficients: &[[Vec<f64>; 3]],
    fock_coefficients: &[[Matrix; 2]],
    coefficients: Pt2Coefficients,
) -> Result<Pt2TaylorResponse> {
    taylor_response_impl(
        &problem.energies,
        problem.occupied,
        &problem.ovov,
        ovov_coefficients,
        fock_coefficients,
        coefficients,
    )
}

/// Complex static amplitude/orbital paths around a real semicanonical parent.
/// Virtual indices transform covariantly and occupied indices contragrediently;
/// energies contract conjugate source amplitudes. No imaginary part is discarded.
pub fn complex_taylor_response(
    problem: &Pt2Problem<f64>,
    ovov_coefficients: &[[Vec<crate::response::complex::Complex>; 3]],
    fock_coefficients: &[[crate::integrals::london::ComplexMatrix; 2]],
    coefficients: Pt2Coefficients,
) -> Result<Pt2TaylorResponse<crate::response::complex::Complex>> {
    let base = problem.ovov.each_ref().map(|g| {
        g.iter()
            .copied()
            .map(crate::response::complex::Complex::from)
            .collect()
    });
    taylor_response_impl(
        &problem.energies,
        problem.occupied,
        &base,
        ovov_coefficients,
        fock_coefficients,
        coefficients,
    )
}

fn taylor_response_impl<S: AmplitudeNumber, F: FockCoefficient<S>>(
    orbital_energies: &[Vec<f64>; 2],
    occupied: [usize; 2],
    base_couplings: &[Vec<S>; 3],
    ovov_coefficients: &[[Vec<S>; 3]],
    fock_coefficients: &[[F; 2]],
    coefficients: Pt2Coefficients,
) -> Result<Pt2TaylorResponse<S>> {
    coefficients.validate()?;
    if orbital_energies
        .iter()
        .zip(occupied)
        .any(|(e, n)| n > e.len() || e.iter().any(|v| !v.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "invalid PT2 semicanonical orbital energies".into(),
        ));
    }
    for (channel, (left, right)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        let size = [
            occupied[left],
            orbital_energies[left].len() - occupied[left],
            occupied[right],
            orbital_energies[right].len() - occupied[right],
        ]
        .into_iter()
        .try_fold(1usize, |a, b| a.checked_mul(b))
        .ok_or_else(|| DftError::InvalidInput("PT2 tensor size overflows".into()))?;
        if base_couplings[channel].len() != size
            || base_couplings[channel].iter().any(|g| !g.finite())
        {
            return Err(DftError::InvalidInput(
                "invalid PT2 occupied-virtual integral block".into(),
            ));
        }
    }
    if ovov_coefficients.len() != fock_coefficients.len()
        || ovov_coefficients.iter().any(|blocks| {
            blocks
                .iter()
                .zip(base_couplings)
                .any(|(a, b)| a.len() != b.len() || a.iter().any(|value| !value.finite()))
        })
        || fock_coefficients.iter().any(|blocks| {
            blocks.iter().enumerate().any(|(s, f)| {
                f.rows() != orbital_energies[s].len() || f.cols() != f.rows() || !f.finite()
            })
        })
    {
        return Err(DftError::InvalidInput(
            "invalid PT2 integral or orbital-Fock Taylor path".into(),
        ));
    }
    let width = ovov_coefficients.len() + 1;
    let mut amplitudes = vec![
        base_couplings
            .each_ref()
            .map(|g| vec![S::from(0.0); g.len()]);
        width
    ];
    let mut energies: [Vec<S>; 3] = std::array::from_fn(|_| vec![S::from(0.0); width]);
    let mut residuals = vec![0.0_f64; width];
    let mut relative_residuals = vec![0.0_f64; width];
    for (channel, (left, right)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
        let nl = occupied[left];
        let nr = occupied[right];
        let vl = orbital_energies[left].len() - nl;
        let vr = orbital_energies[right].len() - nr;
        let shape = [nl, vl, nr, vr];
        let couplings: Vec<Vec<S>> = std::iter::once(base_couplings)
            .chain(ovov_coefficients.iter())
            .map(|blocks| {
                let raw = &blocks[channel];
                let mut g = raw.clone();
                if left == right {
                    for i in 0..nl {
                        for a in 0..vl {
                            for j in 0..nr {
                                for b in 0..vr {
                                    let p = index(shape, i, a, j, b);
                                    g[p] -= raw[index(shape, i, b, j, a)];
                                }
                            }
                        }
                    }
                }
                g
            })
            .collect();
        let mut denominator = vec![0.0; couplings[0].len()];
        for i in 0..nl {
            for a in 0..vl {
                for j in 0..nr {
                    for b in 0..vr {
                        denominator[index(shape, i, a, j, b)] = orbital_energies[left][i]
                            + orbital_energies[right][j]
                            - orbital_energies[left][nl + a]
                            - orbital_energies[right][nr + b];
                    }
                }
            }
        }
        if denominator.iter().any(|d| !d.is_finite() || *d >= 0.0) {
            return Err(DftError::InvalidInput("PT2 needs strictly negative physical pair denominators; no regularization or floor is applied".into()));
        }
        for ((t, g), d) in amplitudes[0][channel]
            .iter_mut()
            .zip(&couplings[0])
            .zip(&denominator)
        {
            *t = *g / *d;
        }
        let mut source_norms = vec![0.0_f64; width];
        source_norms[0] = couplings[0]
            .iter()
            .fold(0.0_f64, |norm, x| norm.hypot(x.magnitude()));
        for n in 1..width {
            let mut source = couplings[n].clone();
            for k in 1..=n {
                let f = &fock_coefficients[k - 1];
                let action = pair_action(shape, [&f[left], &f[right]], &amplitudes[n - k][channel]);
                for (source, value) in source.iter_mut().zip(action) {
                    *source -= value;
                }
            }
            source_norms[n] = source
                .iter()
                .fold(0.0_f64, |norm, x| norm.hypot(x.magnitude()));
            for ((t, source), delta) in amplitudes[n][channel]
                .iter_mut()
                .zip(source)
                .zip(&denominator)
            {
                *t = source / *delta;
            }
        }
        for n in 0..width {
            let mut equation: Vec<_> = denominator
                .iter()
                .zip(&amplitudes[n][channel])
                .zip(&couplings[n])
                .map(|((delta, t), g)| S::from(*delta) * *t - *g)
                .collect();
            for k in 1..=n {
                let f = &fock_coefficients[k - 1];
                let action = pair_action(shape, [&f[left], &f[right]], &amplitudes[n - k][channel]);
                for (value, action) in equation.iter_mut().zip(action) {
                    *value += action;
                }
            }
            if equation.iter().any(|value| !value.finite()) {
                return Err(DftError::InvalidInput(
                    "PT2 Taylor amplitude equation overflows the value range".into(),
                ));
            }
            residuals[n] = residuals[n].max(
                equation
                    .iter()
                    .map(|v| v.magnitude())
                    .fold(0.0_f64, f64::max),
            );
            let norm = equation
                .iter()
                .fold(0.0_f64, |norm, x| norm.hypot(x.magnitude()));
            let relative = if source_norms[n] > 0.0 {
                norm / source_norms[n]
            } else if norm == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            relative_residuals[n] = relative_residuals[n].max(relative);
            let weight = if left == right { 0.25 } else { 1.0 };
            for k in 0..=n {
                energies[channel][n] += S::from(weight)
                    * couplings[k]
                        .iter()
                        .zip(&amplitudes[n - k][channel])
                        .map(|(g, t)| g.conjugate() * *t)
                        .fold(S::from(0.0), |a, b| a + b);
            }
        }
    }
    let correlation: Vec<_> = (0..width)
        .map(|n| {
            S::from(coefficients.same_spin) * (energies[0][n] + energies[2][n])
                + S::from(coefficients.opposite_spin) * energies[1][n]
        })
        .collect();
    if correlation
        .iter()
        .chain(energies.iter().flatten())
        .any(|v| !v.finite())
        || amplitudes
            .iter()
            .flat_map(|a| a.iter().flatten())
            .any(|v| !v.finite())
    {
        return Err(DftError::InvalidInput(
            "PT2 Taylor response overflows the value range".into(),
        ));
    }
    Ok(Pt2TaylorResponse {
        same_spin_alpha: energies[0].clone(),
        same_spin_beta: energies[2].clone(),
        opposite_spin: energies[1].clone(),
        correlation,
        amplitude_coefficients: amplitudes,
        residuals,
        relative_residuals,
    })
}

/// Compatibility wrapper: first-order coefficients are actual derivatives.
pub fn linear_response(
    problem: &Pt2Problem<f64>,
    ovov_direction: &[Vec<f64>; 3],
    fock_direction: &[Matrix; 2],
    coefficients: Pt2Coefficients,
) -> Result<Pt2LinearResponse> {
    let response = taylor_response(
        problem,
        std::slice::from_ref(ovov_direction),
        std::slice::from_ref(fock_direction),
        coefficients,
    )?;
    Ok(Pt2LinearResponse {
        same_spin_alpha: response.same_spin_alpha[1],
        same_spin_beta: response.same_spin_beta[1],
        opposite_spin: response.opposite_spin[1],
        correlation: response.correlation[1],
        amplitude_derivatives: response.amplitude_coefficients[1].clone(),
        residual: response.residuals[1],
    })
}

#[cfg(test)]
mod transform_tests {
    use super::*;
    use crate::dual::{Scalar, Taylor};
    use crate::properties::pt2::{from_ao, OrbitalSpace};

    #[test]
    fn amplitude_relative_residuals_preserve_large_sources_and_zero_channels() {
        use crate::integrals::london::ComplexMatrix;
        use crate::response::complex::Complex;
        let coefficients = Pt2Coefficients {
            same_spin: 0.31,
            opposite_spin: 0.47,
        };
        let fock: Vec<[Matrix; 2]> = (1..=4)
            .map(|n| {
                std::array::from_fn(|s| {
                    let mut m = Matrix::zeros(2, 2);
                    m[(0, 0)] = (n + s + 1) as f64 * 0.17;
                    m[(1, 1)] = -((2 * n + s + 1) as f64) * 0.13;
                    m
                })
            })
            .collect();
        let complex_fock: Vec<_> = fock
            .iter()
            .map(|spin| {
                std::array::from_fn(|s| ComplexMatrix {
                    re: spin[s].clone(),
                    im: Matrix::zeros(2, 2),
                })
            })
            .collect();
        let mut large_absolute = 0.0_f64;
        for scale in [1e-8, 1.0, 1e8] {
            let problem = Pt2Problem {
                energies: [vec![-1.11, 0.73], vec![-0.97, 0.61]],
                occupied: [1, 1],
                ovov: [vec![0.0], vec![0.314159265 * scale], vec![0.0]],
            };
            let source: Vec<[Vec<f64>; 3]> = (1..=4)
                .map(|n| {
                    [
                        vec![0.0],
                        vec![((n as f64) * 0.37).sin() * scale],
                        vec![0.0],
                    ]
                })
                .collect();
            let real = taylor_response(&problem, &source, &fock, coefficients).unwrap();
            let complex_source: Vec<_> = source
                .iter()
                .map(|spin| {
                    std::array::from_fn(|s| {
                        spin[s]
                            .iter()
                            .map(|&v| Complex::new(v, if s == 1 { 0.27182818 * v } else { 0.0 }))
                            .collect()
                    })
                })
                .collect();
            let complex =
                complex_taylor_response(&problem, &complex_source, &complex_fock, coefficients)
                    .unwrap();
            for residuals in [&real.relative_residuals, &complex.relative_residuals] {
                assert_eq!(residuals.len(), 5);
                assert!(residuals.iter().all(|r| r.is_finite() && *r < 3e-15));
            }
            large_absolute = large_absolute.max(real.residuals.iter().copied().fold(0.0, f64::max));
            // One occupied/virtual orbital makes both antisymmetrized SS
            // channels exactly empty at every order, without a norm floor.
            for amplitudes in &complex.amplitude_coefficients {
                assert_eq!(amplitudes[0], vec![Complex::ZERO]);
                assert_eq!(amplitudes[2], vec![Complex::ZERO]);
            }
        }
        assert!(large_absolute > 1e-12);
        let zero = Pt2Problem {
            energies: [vec![-1.11, 0.73], vec![-0.97, 0.61]],
            occupied: [1, 1],
            ovov: [vec![0.0], vec![0.0], vec![0.0]],
        };
        let response = taylor_response(&zero, &[], &[], coefficients).unwrap();
        assert_eq!(response.residuals, vec![0.0]);
        assert_eq!(response.relative_residuals, vec![0.0]);
    }

    #[test]
    fn parallel_faer_coefficients_match_scalar_oracle_and_four_step_derivative() {
        const W: usize = 5;
        let n = 3usize;
        let energies = [vec![-1.0, -0.4, 0.7], vec![-0.8, 0.3, 0.9]];
        let occupied = [2, 1];
        let integrals: Vec<Vec<f64>> = (0..W)
            .map(|k| {
                (0..n.pow(4))
                    .map(|i| {
                        ((i + 1) as f64 * 0.17 + (k + 1) as f64 * 0.31).sin()
                            * 0.3_f64.powi(k as i32)
                    })
                    .collect()
            })
            .collect();
        let orbitals: [Vec<Matrix>; 2] = std::array::from_fn(|s| {
            (0..W)
                .map(|k| {
                    let mut c = Matrix::zeros(n, n);
                    for i in 0..n {
                        for j in 0..n {
                            c[(i, j)] = if k == 0 && i == j {
                                1.0
                            } else {
                                ((i + 2 * j + 3 * s + 1) as f64 * 0.23).cos()
                                    * 0.2_f64.powi(k as i32 + 1)
                            };
                        }
                    }
                    c
                })
                .collect()
        });
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    transform_path::<W>(n, &integrals, &orbitals, &energies, occupied).unwrap()
                })
        };
        let sequential = run(1);
        let parallel = run(4);
        let spaces = std::array::from_fn(|s| OrbitalSpace {
            energies: energies[s]
                .iter()
                .map(|e| Taylor::<W>::from_f64(*e))
                .collect(),
            occupied: occupied[s],
            coefficients: (0..n * n)
                .map(|i| {
                    Taylor::<W>::from_coefficients(
                        &orbitals[s]
                            .iter()
                            .map(|c| c.as_slice()[i])
                            .collect::<Vec<_>>(),
                    )
                })
                .collect(),
        });
        let tensor: Vec<_> = (0..n.pow(4))
            .map(|i| {
                Taylor::<W>::from_coefficients(&integrals.iter().map(|g| g[i]).collect::<Vec<_>>())
            })
            .collect();
        let oracle = from_ao(n, &tensor, &spaces).unwrap();
        for k in 0..W {
            for s in 0..3 {
                for i in 0..oracle.ovov[s].len() {
                    assert!((parallel[k].ovov[s][i] - oracle.ovov[s][i].c[k]).abs() < 1e-13);
                    assert_eq!(parallel[k].ovov[s][i], sequential[k].ovov[s][i]);
                }
            }
        }
        let direct = |t: f64| {
            let spaces = std::array::from_fn(|s| OrbitalSpace {
                energies: energies[s].clone(),
                occupied: occupied[s],
                coefficients: (0..n * n)
                    .map(|i| {
                        orbitals[s]
                            .iter()
                            .rev()
                            .fold(0.0, |sum, c| sum * t + c.as_slice()[i])
                    })
                    .collect(),
            });
            let tensor: Vec<_> = (0..n.pow(4))
                .map(|i| integrals.iter().rev().fold(0.0, |sum, g| sum * t + g[i]))
                .collect();
            from_ao(n, &tensor, &spaces).unwrap().ovov[1]
                .iter()
                .sum::<f64>()
        };
        let exact = parallel[1].ovov[1].iter().sum::<f64>();
        let steps = [0.08, 0.04, 0.02, 0.01];
        let errors: Vec<_> = steps
            .iter()
            .map(|h| ((direct(*h) - direct(-*h)) / (2.0 * h) - exact).abs())
            .collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        println!(
            "parallel faer PT2 transform steps={steps:?} errors={errors:?} p={:.9} R2={:.12}",
            fit.exponent, fit.r_squared
        );
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    }
}
