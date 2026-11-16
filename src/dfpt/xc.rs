// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic finite-q semilocal kernels in independent Bloch AO spaces.
//! The density and potential have opposite AO phase contractions. No
//! same-q Hermitian projection or finite displacement is used.

use crate::dual::DualN;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::xc::{AoDerivativeOrder, BlochAos, PeriodicXcGrid};
use crate::response::complex::Complex;
use crate::skala::SkalaInput;
use crate::xc::{Family, GridVars, ResolvedFunctional, XcOut};
use crate::{DftError, Result};

fn entry(a: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(a.re[(i, j)], a.im[(i, j)])
}

fn validate_pairs(
    ket: &PeriodicXcGrid,
    bra: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
) -> Result<()> {
    if ket.points != bra.points
        || ket.weights != bra.weights
        || ket.is_empty()
        || ket.k_points.is_empty()
        || ket.k_points.len() != bra.k_points.len()
        || ket.aos.len() != ket.k_points.len()
        || bra.aos.len() != bra.k_points.len()
        || density.iter().any(|s| s.len() != ket.k_points.len())
    {
        return Err(DftError::InvalidInput(
            "finite-q XC requires paired Bloch meshes on the same quadrature".into(),
        ));
    }
    for (ik, (left, right)) in bra.aos.iter().zip(&ket.aos).enumerate() {
        for spin in density {
            let d = &spin[ik];
            if d.rows() != left.phi.cols()
                || d.cols() != right.phi.cols()
                || d.re.rows != d.im.rows
                || d.re.cols != d.im.cols
                || d.re
                    .as_slice()
                    .iter()
                    .chain(d.im.as_slice())
                    .any(|v| !v.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "invalid rectangular finite-q AO density".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Pointwise `(rho_a,rho_b,grad_a[3],grad_b[3],tau_a,tau_b)`.
fn spatial_density(
    ket: &PeriodicXcGrid,
    bra: &PeriodicXcGrid,
    density: &[Vec<BlochMatrix>; 2],
    family: Family,
) -> Result<Vec<[Complex; 10]>> {
    validate_pairs(ket, bra, density)?;
    let gga = family != Family::Lda;
    let meta = family == Family::MetaGga;
    if gga && ket.aos.iter().chain(&bra.aos).any(|a| a.grad.is_none()) {
        return Err(DftError::InvalidInput(
            "finite-q GGA kernel requires Bloch AO gradients".into(),
        ));
    }
    let mut result = vec![[Complex::ZERO; 10]; ket.len()];
    let weight = 1.0 / ket.k_points.len() as f64;
    for spin in 0..2 {
        for (ik, (left, right)) in bra.aos.iter().zip(&ket.aos).enumerate() {
            let d = &density[spin][ik];
            for (point, values) in result.iter_mut().enumerate() {
                for mu in 0..d.rows() {
                    for nu in 0..d.cols() {
                        let coefficient = entry(d, mu, nu) * weight;
                        let l = entry(&left.phi, point, mu);
                        let r = entry(&right.phi, point, nu).conj();
                        values[spin] += coefficient * l * r;
                        if gga {
                            let lg = left.grad.as_ref().unwrap();
                            let rg = right.grad.as_ref().unwrap();
                            for axis in 0..3 {
                                let dl = entry(&lg[axis], point, mu);
                                let dr = entry(&rg[axis], point, nu).conj();
                                values[2 + 3 * spin + axis] += coefficient * (dl * r + l * dr);
                                if meta {
                                    values[8 + spin] += coefficient * dl * dr * 0.5;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(result)
}

/// Cached semilocal reference derivatives. Spin channels are coupled by
/// the functional, while arbitrary complex rectangular density sources
/// remain independent at each `(k+q,k)` pair.
pub struct FiniteQXc {
    ket: PeriodicXcGrid,
    bra: PeriodicXcGrid,
    family: Family,
    ground: Vec<[f64; 10]>,
    potentials: Vec<[f64; 7]>,
    kernels: Vec<[[f64; 7]; 7]>,
}

impl FiniteQXc {
    pub fn new(
        functional: &ResolvedFunctional,
        ket: &PeriodicXcGrid,
        bra: &PeriodicXcGrid,
        ground_density: &[Vec<BlochMatrix>; 2],
    ) -> Result<Self> {
        Self::with_reference(functional, ket, ground_density, ket, bra)
    }

    /// Keep the physical reference integration mesh independent of the
    /// bra and ket meshes used by the perturbation.
    pub fn with_reference(
        functional: &ResolvedFunctional,
        reference: &PeriodicXcGrid,
        ground_density: &[Vec<BlochMatrix>; 2],
        ket: &PeriodicXcGrid,
        bra: &PeriodicXcGrid,
    ) -> Result<Self> {
        let order = AoDerivativeOrder::for_scf(functional.family);
        if reference.derivative_order() < order
            || ket.derivative_order() < order
            || bra.derivative_order() < order
        {
            return Err(DftError::InvalidInput(
                "insufficient spatial derivatives for finite-q XC".into(),
            ));
        }
        if reference.points != ket.points
            || reference.weights != ket.weights
            || reference.points != bra.points
            || reference.weights != bra.weights
        {
            return Err(DftError::InvalidInput(
                "reference and response XC quadratures must be identical".into(),
            ));
        }
        validate_pairs(reference, reference, ground_density)?;
        if ground_density
            .iter()
            .flatten()
            .any(|p| p.rows() != p.cols() || p.hermiticity_error() > 1e-9)
        {
            return Err(DftError::InvalidInput(
                "finite-q XC reference density must be Hermitian".into(),
            ));
        }
        let complex_ground =
            spatial_density(reference, reference, ground_density, functional.family)?;
        if complex_ground.iter().flatten().any(|v| v.im.abs() > 1e-9) {
            return Err(DftError::InvalidInput(
                "non-real periodic reference density variables".into(),
            ));
        }
        let mut ground: Vec<_> = complex_ground
            .iter()
            .map(|v| std::array::from_fn(|i| v[i].re))
            .collect();
        for p in &mut ground {
            for spin in 0..2 {
                if p[spin] < 0.0 && p[spin] > -1e-12 {
                    p[spin] = 0.0;
                }
            }
        }
        let rho: [Vec<_>; 2] = std::array::from_fn(|s| ground.iter().map(|p| p[s]).collect());
        let sigma: [Vec<_>; 3] = std::array::from_fn(|s| {
            ground
                .iter()
                .map(|p| {
                    let (a, b) = [(0, 0), (0, 1), (1, 1)][s];
                    (0..3)
                        .map(|axis| p[2 + 3 * a + axis] * p[2 + 3 * b + axis])
                        .sum::<f64>()
                })
                .collect()
        });
        let tau: [Vec<_>; 2] = std::array::from_fn(|s| ground.iter().map(|p| p[8 + s]).collect());
        let vars = GridVars {
            rho: [&rho[0], &rho[1]],
            sigma: (functional.family != Family::Lda).then_some([&sigma[0], &sigma[1], &sigma[2]]),
            tau: (functional.family == Family::MetaGga).then_some([&tau[0], &tau[1]]),
        };
        let mut out = XcOut::default();
        functional.eval(&vars, 2, &mut out)?;
        let derivatives = (0..ket.len())
            .map(|p| out.potential_and_kernel(p, functional.family))
            .collect::<Result<Vec<_>>>()?;
        let (potentials, kernels) = derivatives.into_iter().unzip();
        Ok(Self {
            ket: ket.clone(),
            bra: bra.clone(),
            family: functional.family,
            ground,
            potentials,
            kernels,
        })
    }

    pub fn apply(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        let changes = spatial_density(&self.ket, &self.bra, density, self.family)?;
        let mut output: [Vec<_>; 2] = std::array::from_fn(|_| {
            self.bra
                .aos
                .iter()
                .zip(&self.ket.aos)
                .map(|(l, r)| BlochMatrix::zeros(l.phi.cols(), r.phi.cols()))
                .collect()
        });
        for (point, change) in changes.iter().enumerate() {
            let ground = &self.ground[point];
            let mut delta = [Complex::ZERO; 7];
            delta[0] = change[0];
            delta[1] = change[1];
            for (s, (a, b)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                for axis in 0..3 {
                    delta[2 + s] += change[2 + 3 * a + axis] * ground[2 + 3 * b + axis]
                        + change[2 + 3 * b + axis] * ground[2 + 3 * a + axis];
                }
            }
            delta[5] = change[8];
            delta[6] = change[9];
            let dv: [Complex; 7] = std::array::from_fn(|i| {
                (0..7).fold(Complex::ZERO, |s, j| {
                    s + delta[j] * self.kernels[point][i][j]
                })
            });
            for spin in 0..2 {
                let own = if spin == 0 { 2 } else { 4 };
                let b: [Complex; 3] = std::array::from_fn(|axis| {
                    (dv[own] * ground[2 + 3 * spin + axis]
                        + change[2 + 3 * spin + axis] * self.potentials[point][own])
                        * 2.0
                        + dv[3] * ground[2 + 3 * (1 - spin) + axis]
                        + change[2 + 3 * (1 - spin) + axis] * self.potentials[point][3]
                });
                for (ik, (left, right)) in self.bra.aos.iter().zip(&self.ket.aos).enumerate() {
                    assemble_point(
                        &mut output[spin][ik],
                        left,
                        right,
                        point,
                        self.ket.weights[point],
                        dv[spin],
                        b,
                        dv[5 + spin],
                        self.family,
                    );
                }
            }
        }
        Ok(output)
    }
}

/// Kernel of a learned functional at `q = 0`: the change of its Kohn-Sham
/// matrices with the density. The reverse pass of the network is evaluated
/// once on numbers that carry the real and the imaginary part of the density
/// change as two directional derivatives, which is exact for the non-local
/// layers.
pub struct LearnedXc {
    grid: PeriodicXcGrid,
    points: Vec<[f64; 3]>,
    ground: Vec<[f64; 10]>,
}

impl LearnedXc {
    /// `grid` carries the learned functional; `ground_density` is the
    /// self-consistent density of its k points.
    pub fn new(grid: &PeriodicXcGrid, ground_density: &[Vec<BlochMatrix>; 2]) -> Result<Self> {
        if grid.learned.is_none()
            || grid.derivative_order() < AoDerivativeOrder::for_scf(Family::MetaGga)
        {
            return Err(DftError::InvalidInput(
                "the learned kernel needs a grid with the learned functional and Bloch AO gradients"
                    .into(),
            ));
        }
        validate_pairs(grid, grid, ground_density)?;
        let complex_ground = spatial_density(grid, grid, ground_density, Family::MetaGga)?;
        if complex_ground.iter().flatten().any(|v| v.im.abs() > 1e-9) {
            return Err(DftError::InvalidInput(
                "non-real periodic reference density variables".into(),
            ));
        }
        let mut ground: Vec<[f64; 10]> = complex_ground
            .iter()
            .map(|v| std::array::from_fn(|i| v[i].re))
            .collect();
        for p in &mut ground {
            for spin in 0..2 {
                if p[spin] < 0.0 && p[spin] > -1e-12 {
                    p[spin] = 0.0;
                }
            }
        }
        Ok(Self {
            grid: grid.clone(),
            points: grid.points.iter().map(|p| p.to_array()).collect(),
            ground,
        })
    }

    pub fn apply(&self, density: &[Vec<BlochMatrix>; 2]) -> Result<[Vec<BlochMatrix>; 2]> {
        type Pair = DualN<f64, 2>;
        let learned = self.grid.learned.as_ref().expect("checked at construction");
        let changes = spatial_density(&self.grid, &self.grid, density, Family::MetaGga)?;
        let join = |value: f64, change: Complex| Pair {
            re: value,
            im: [change.re, change.im],
        };
        let rho: [Vec<Pair>; 2] = std::array::from_fn(|s| {
            self.ground
                .iter()
                .zip(&changes)
                .map(|(g, c)| join(g[s], c[s]))
                .collect()
        });
        let gradient: [Vec<[Pair; 3]>; 2] = std::array::from_fn(|s| {
            self.ground
                .iter()
                .zip(&changes)
                .map(|(g, c)| {
                    std::array::from_fn(|axis| join(g[2 + 3 * s + axis], c[2 + 3 * s + axis]))
                })
                .collect()
        });
        let tau: [Vec<Pair>; 2] = std::array::from_fn(|s| {
            self.ground
                .iter()
                .zip(&changes)
                .map(|(g, c)| join(g[8 + s], c[8 + s]))
                .collect()
        });
        let weights: Vec<Pair> = self
            .grid
            .weights
            .iter()
            .map(|w| Pair::constant(*w))
            .collect();
        let out = learned.model.evaluate(
            &SkalaInput {
                atom_of: &self.grid.atom_of,
                points: &self.points,
                centers: &learned.centers,
                weights: &weights,
                atomic_weights: &self.grid.raw_weights,
                rho: [&rho[0], &rho[1]],
                gradient: [&gradient[0], &gradient[1]],
                tau: [&tau[0], &tau[1]],
            },
            true,
        )?;
        let change = |v: Pair| Complex::new(v.im[0], v.im[1]);
        let mut output: [Vec<BlochMatrix>; 2] = std::array::from_fn(|_| {
            self.grid
                .aos
                .iter()
                .map(|a| BlochMatrix::zeros(a.phi.cols(), a.phi.cols()))
                .collect()
        });
        for point in 0..self.grid.len() {
            for (spin, matrices) in output.iter_mut().enumerate() {
                let b = out.d_gradient[spin][point];
                for (matrix, aos) in matrices.iter_mut().zip(&self.grid.aos) {
                    // The derivatives of the energy already hold the weights.
                    assemble_point(
                        matrix,
                        aos,
                        aos,
                        point,
                        1.0,
                        change(out.d_rho[spin][point]),
                        [change(b[0]), change(b[1]), change(b[2])],
                        change(out.d_tau[spin][point]),
                        Family::MetaGga,
                    );
                }
            }
        }
        Ok(output)
    }
}

fn assemble_point(
    output: &mut BlochMatrix,
    left: &BlochAos,
    right: &BlochAos,
    point: usize,
    weight: f64,
    local: Complex,
    b: [Complex; 3],
    kinetic: Complex,
    family: Family,
) {
    for mu in 0..output.rows() {
        for nu in 0..output.cols() {
            let l = entry(&left.phi, point, mu).conj();
            let r = entry(&right.phi, point, nu);
            let mut term = l * r * local;
            if family != Family::Lda {
                let lg = left.grad.as_ref().unwrap();
                let rg = right.grad.as_ref().unwrap();
                for axis in 0..3 {
                    let dl = entry(&lg[axis], point, mu).conj();
                    let dr = entry(&rg[axis], point, nu);
                    term += (dl * r + l * dr) * b[axis];
                    if family == Family::MetaGga {
                        term += dl * dr * kinetic * 0.5;
                    }
                }
            }
            output.re[(mu, nu)] += weight * term.re;
            output.im[(mu, nu)] += weight * term.im;
        }
    }
}
