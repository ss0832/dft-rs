// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Contracted spherical AO nuclear/Hartree operators in a periodic subspace.
//! Gaussian charge, nuclear and electronic terms use one physical Green gauge.

use super::bloch::BlochMatrix;
use super::electrostatics::{
    ChargeCompensation, DimensionalCoulomb, ElectrostaticComponents, ElectrostaticOptions,
    ElectrostaticSource,
};
use super::ewald::EwaldParams;
use super::gaussian_coulomb::{
    contract_integral_tensor, prepare_integral_tensor, CartesianGaussian, GaussianNucleus,
};
use super::integrals::{bloch, dimensional_image_blocks, Operator};
use super::quasiperiodic::{contract_quartet, prepare_quartet, PeriodicTransfer};
use super::PeriodicGeometry;
use crate::basis::{
    aos::{cart_to_sph, sph_block, MAX_L},
    cart_components, n_sph, MolBasis, MolShell,
};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::system::Molecule;
use crate::{DftError, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct PeriodicCoulombOptions {
    pub k_mesh: [usize; 3],
    /// Actual shell-pair distance cutoff, independent of electrostatic cutoffs.
    pub basis_cutoff: f64,
    /// Empty selects point nuclei; otherwise one physical width per atom.
    pub nuclear_widths: Vec<f64>,
    pub electrostatics: ElectrostaticOptions,
}
impl Default for PeriodicCoulombOptions {
    fn default() -> Self {
        Self {
            k_mesh: [1; 3],
            basis_cutoff: 12.0,
            nuclear_widths: Vec::new(),
            electrostatics: ElectrostaticOptions::default(),
        }
    }
}
#[derive(Clone)]
struct PairImage {
    index: [i64; 3],
    translation: Vec3,
    left: usize,
    right: usize,
}
struct PairResult {
    parts: [BlochMatrix; 3],
    derivatives: Vec<[BlochMatrix; 3]>,
}
impl PairResult {
    fn new(rows: usize, cols: usize, centres: usize) -> Self {
        Self {
            parts: std::array::from_fn(|_| BlochMatrix::zeros(rows, cols)),
            derivatives: (0..centres)
                .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(rows, cols)))
                .collect(),
        }
    }
}
struct ImageResult {
    translation: Vec3,
    parts: [BlochMatrix; 3],
    derivatives: Vec<[BlochMatrix; 3]>,
}
struct Operators {
    parts: Vec<[BlochMatrix; 3]>,
    derivatives: Vec<Vec<[BlochMatrix; 3]>>,
    imaginary_image_defect: f64,
    images: Vec<ImageResult>,
}
impl Operators {
    fn at(&self, k_points: &[Vec3]) -> Vec<BlochMatrix> {
        let n = self.parts[0][0].rows();
        k_points
            .iter()
            .map(|&k| {
                let mut out = BlochMatrix::zeros(n, n);
                for image in &self.images {
                    let phase = k.dot(image.translation);
                    let phase = Complex::new(phase.cos(), phase.sin());
                    for part in &image.parts {
                        for i in 0..n {
                            for j in 0..n {
                                let z = phase * Complex::new(part.re[(i, j)], part.im[(i, j)]);
                                out.re[(i, j)] += z.re;
                                out.im[(i, j)] += z.im;
                            }
                        }
                    }
                }
                out
            })
            .collect()
    }
}
fn max_component(matrix: &BlochMatrix) -> f64 {
    matrix
        .re
        .as_slice()
        .iter()
        .chain(matrix.im.as_slice())
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
}
fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.as_slice()
        .iter()
        .zip(b.re.as_slice())
        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}
fn total(parts: &[BlochMatrix; 3]) -> BlochMatrix {
    let mut result = parts[0].clone();
    for part in &parts[1..] {
        result.add_scaled(part, 1.0);
    }
    result
}
fn transformed_cart_density(
    left: &MolShell,
    right: &MolShell,
    density: &BlochMatrix,
) -> BlochMatrix {
    let sub = |matrix: &Matrix| {
        let mut out = Matrix::zeros(n_sph(left.l), n_sph(right.l));
        for i in 0..out.rows {
            for j in 0..out.cols {
                out[(i, j)] = matrix[(left.sph_off + i, right.sph_off + j)];
            }
        }
        out
    };
    let a = cart_to_sph(left.l);
    let b = cart_to_sph(right.l).transpose();
    BlochMatrix {
        re: a.matmul_seq(&sub(&density.re)).matmul_seq(&b),
        im: a.matmul_seq(&sub(&density.im)).matmul_seq(&b),
    }
}
fn spherical(left: &MolShell, right: &MolShell, cart: PairResult) -> PairResult {
    let transform = |m: BlochMatrix| BlochMatrix {
        re: sph_block(left.l, right.l, &m.re),
        im: sph_block(left.l, right.l, &m.im),
    };
    PairResult {
        parts: cart.parts.map(&transform),
        derivatives: cart
            .derivatives
            .into_iter()
            .map(|axes| axes.map(&transform))
            .collect(),
    }
}
fn add_cart(
    result: &mut PairResult,
    row: usize,
    col: usize,
    values: [f64; 3],
    gradients: &[Vec3],
    owners: &[usize],
    weight_re: f64,
    weight_im: f64,
) {
    for c in 0..3 {
        result.parts[c].re[(row, col)] += weight_re * values[c];
        result.parts[c].im[(row, col)] += weight_im * values[c];
    }
    for (gradient, &owner) in gradients.iter().zip(owners) {
        for (axis, value) in [gradient.x, gradient.y, gradient.z].into_iter().enumerate() {
            result.derivatives[owner][axis].re[(row, col)] += weight_re * value;
            result.derivatives[owner][axis].im[(row, col)] += weight_im * value;
        }
    }
}
fn nuclear_pair(
    kernel: &DimensionalCoulomb,
    a: &MolShell,
    b: &MolShell,
    sources: &[(GaussianNucleus, usize)],
    centres: usize,
) -> Result<PairResult> {
    let ca = cart_components(a.l);
    let cb = cart_components(b.l);
    let mut result = PairResult::new(ca.len(), cb.len(), centres);
    let angular = |v: (i32, i32, i32)| [v.0 as usize, v.1 as usize, v.2 as usize];
    for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
            let mut gs = vec![
                CartesianGaussian {
                    center: a.center,
                    exponent: ea,
                    angular: [0; 3],
                },
                CartesianGaussian {
                    center: b.center,
                    exponent: eb,
                    angular: [0; 3],
                },
            ];
            for (source, owner) in sources {
                let tensor =
                    prepare_integral_tensor(kernel, &gs, Some(source), (a.l + b.l) as usize + 1)?;
                for (i, &ia) in ca.iter().enumerate() {
                    for (j, &ib) in cb.iter().enumerate() {
                        gs[0].angular = angular(ia);
                        gs[1].angular = angular(ib);
                        let (values, gradients) = contract_integral_tensor(&tensor, &gs, true)?;
                        add_cart(
                            &mut result,
                            i,
                            j,
                            values,
                            &gradients,
                            &[a.atom, b.atom, *owner],
                            -source.charge * wa * wb,
                            0.0,
                        );
                    }
                }
            }
        }
    }
    Ok(spherical(a, b, result))
}
fn hartree_block(
    kernel: &DimensionalCoulomb,
    a: &MolShell,
    b: &MolShell,
    c: &MolShell,
    d: &MolShell,
) -> Result<HartreeBlock> {
    let (ca, cb) = (cart_components(a.l), cart_components(b.l));
    let (cc, cd) = (cart_components(c.l), cart_components(d.l));
    let angular = |v: (i32, i32, i32)| [v.0 as usize, v.1 as usize, v.2 as usize];
    let size = ca.len() * cb.len() * cc.len() * cd.len();
    let mut block = HartreeBlock {
        values: vec![[0.0; 3]; size],
        gradients: vec![[Vec3::zero(); 4]; size],
    };
    for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
        for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
            for (&ec, &wc) in c.exps.iter().zip(&c.coefs) {
                for (&ed, &wd) in d.exps.iter().zip(&d.coefs) {
                    let mut gs = vec![
                        CartesianGaussian {
                            center: a.center,
                            exponent: ea,
                            angular: [0; 3],
                        },
                        CartesianGaussian {
                            center: b.center,
                            exponent: eb,
                            angular: [0; 3],
                        },
                        CartesianGaussian {
                            center: c.center,
                            exponent: ec,
                            angular: [0; 3],
                        },
                        CartesianGaussian {
                            center: d.center,
                            exponent: ed,
                            angular: [0; 3],
                        },
                    ];
                    let tensor = prepare_integral_tensor(
                        kernel,
                        &gs,
                        None,
                        (a.l + b.l + c.l + d.l) as usize + 1,
                    )?;
                    let weight = wa * wb * wc * wd;
                    let mut index = 0;
                    for &ia in &ca {
                        for &ib in &cb {
                            for &ic in &cc {
                                for &id in &cd {
                                    gs[0].angular = angular(ia);
                                    gs[1].angular = angular(ib);
                                    gs[2].angular = angular(ic);
                                    gs[3].angular = angular(id);
                                    let (values, gradients) =
                                        contract_integral_tensor(&tensor, &gs, false)?;
                                    for (sum, value) in block.values[index].iter_mut().zip(values) {
                                        *sum += weight * value;
                                    }
                                    for (sum, gradient) in
                                        block.gradients[index].iter_mut().zip(&gradients)
                                    {
                                        *sum += *gradient * weight;
                                    }
                                    index += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(block)
}
/// The Hartree operator block of one shell pair from cached integrals.
fn hartree_pair_cached(
    a: &MolShell,
    b: &MolShell,
    blocks: &[HartreeBlock],
    density_pairs: &[(MolShell, MolShell, BlochMatrix)],
    centres: usize,
) -> PairResult {
    let (ca, cb) = (cart_components(a.l).len(), cart_components(b.l).len());
    let mut result = PairResult::new(ca, cb, centres);
    for ((c, d, density), block) in density_pairs.iter().zip(blocks) {
        if max_component(density) == 0.0 {
            continue;
        }
        let (cc, cd) = (cart_components(c.l).len(), cart_components(d.l).len());
        for i in 0..ca {
            for j in 0..cb {
                for k in 0..cc {
                    for l in 0..cd {
                        let (re, im) = (density.re[(k, l)], density.im[(k, l)]);
                        if re == 0.0 && im == 0.0 {
                            continue;
                        }
                        let index = ((i * cb + j) * cc + k) * cd + l;
                        add_cart(
                            &mut result,
                            i,
                            j,
                            block.values[index],
                            &block.gradients[index],
                            &[a.atom, b.atom, c.atom, d.atom],
                            re,
                            im,
                        );
                    }
                }
            }
        }
    }
    spherical(a, b, result)
}
fn hartree_pair(
    kernel: &DimensionalCoulomb,
    a: &MolShell,
    b: &MolShell,
    density_pairs: &[(MolShell, MolShell, BlochMatrix)],
    centres: usize,
) -> Result<PairResult> {
    let ca = cart_components(a.l);
    let cb = cart_components(b.l);
    let angular = |v: (i32, i32, i32)| [v.0 as usize, v.1 as usize, v.2 as usize];
    let mut result = PairResult::new(ca.len(), cb.len(), centres);
    for (c, d, density) in density_pairs {
        let cc = cart_components(c.l);
        let cd = cart_components(d.l);
        if max_component(density) == 0.0 {
            continue;
        }
        for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
            for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                for (&ec, &wc) in c.exps.iter().zip(&c.coefs) {
                    for (&ed, &wd) in d.exps.iter().zip(&d.coefs) {
                        let mut gs = vec![
                            CartesianGaussian {
                                center: a.center,
                                exponent: ea,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: b.center,
                                exponent: eb,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: c.center,
                                exponent: ec,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: d.center,
                                exponent: ed,
                                angular: [0; 3],
                            },
                        ];
                        let tensor = prepare_integral_tensor(
                            kernel,
                            &gs,
                            None,
                            (a.l + b.l + c.l + d.l) as usize + 1,
                        )?;
                        let weight = wa * wb * wc * wd;
                        for (i, &ia) in ca.iter().enumerate() {
                            for (j, &ib) in cb.iter().enumerate() {
                                gs[0].angular = angular(ia);
                                gs[1].angular = angular(ib);
                                for (k, &ic) in cc.iter().enumerate() {
                                    for (l, &id) in cd.iter().enumerate() {
                                        let re = density.re[(k, l)];
                                        let im = density.im[(k, l)];
                                        if re == 0.0 && im == 0.0 {
                                            continue;
                                        }
                                        gs[2].angular = angular(ic);
                                        gs[3].angular = angular(id);
                                        let (values, gradients) =
                                            contract_integral_tensor(&tensor, &gs, false)?;
                                        add_cart(
                                            &mut result,
                                            i,
                                            j,
                                            values,
                                            &gradients,
                                            &[a.atom, b.atom, c.atom, d.atom],
                                            weight * re,
                                            weight * im,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(spherical(a, b, result))
}
fn assemble(
    mb: &MolBasis,
    pairs: &[PairImage],
    results: Vec<PairResult>,
    centres: usize,
    k_points: &[Vec3],
) -> Operators {
    let mut images: BTreeMap<[i64; 3], ImageResult> = BTreeMap::new();
    for (pair, result) in pairs.iter().zip(results) {
        let image = images.entry(pair.index).or_insert_with(|| ImageResult {
            translation: pair.translation,
            parts: std::array::from_fn(|_| BlochMatrix::zeros(mb.nao_sph, mb.nao_sph)),
            derivatives: (0..centres)
                .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(mb.nao_sph, mb.nao_sph)))
                .collect(),
        });
        let (a, b) = (&mb.shells[pair.left], &mb.shells[pair.right]);
        for i in 0..n_sph(a.l) {
            for j in 0..n_sph(b.l) {
                for component in 0..3 {
                    image.parts[component].re[(a.sph_off + i, b.sph_off + j)] +=
                        result.parts[component].re[(i, j)];
                    image.parts[component].im[(a.sph_off + i, b.sph_off + j)] +=
                        result.parts[component].im[(i, j)];
                }
                for owner in 0..centres {
                    for axis in 0..3 {
                        image.derivatives[owner][axis].re[(a.sph_off + i, b.sph_off + j)] +=
                            result.derivatives[owner][axis].re[(i, j)];
                        image.derivatives[owner][axis].im[(a.sph_off + i, b.sph_off + j)] +=
                            result.derivatives[owner][axis].im[(i, j)];
                    }
                }
            }
        }
    }
    let mut imaginary_image_defect = 0.0f64;
    for image in images.values() {
        let sum = total(&image.parts);
        imaginary_image_defect = imaginary_image_defect.max(
            sum.im
                .as_slice()
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0, f64::max),
        );
    }
    let transform = |select: &dyn Fn(&ImageResult) -> &BlochMatrix, k: Vec3| {
        let mut result = BlochMatrix::zeros(mb.nao_sph, mb.nao_sph);
        for image in images.values() {
            let phase = k.dot(image.translation);
            let (co, si) = (phase.cos(), phase.sin());
            let m = select(image);
            for i in 0..mb.nao_sph {
                for j in 0..mb.nao_sph {
                    result.re[(i, j)] += co * m.re[(i, j)] - si * m.im[(i, j)];
                    result.im[(i, j)] += co * m.im[(i, j)] + si * m.re[(i, j)];
                }
            }
        }
        result
    };
    Operators {
        parts: k_points
            .iter()
            .map(|&k| std::array::from_fn(|c| transform(&|m| &m.parts[c], k)))
            .collect(),
        derivatives: k_points
            .iter()
            .map(|&k| {
                (0..centres)
                    .map(|owner| {
                        std::array::from_fn(|axis| transform(&|m| &m.derivatives[owner][axis], k))
                    })
                    .collect()
            })
            .collect(),
        imaginary_image_defect,
        images: images.into_values().collect(),
    }
}

/// Reusable reference-independent Coulomb build. It accepts signed q=0 density
/// directions as well as physical k-point densities, without requiring time reversal.
pub struct DimensionalCoulombBackend {
    geometry: PeriodicGeometry,
    basis: MolBasis,
    basis_cutoff: f64,
    pairs: Vec<PairImage>,
    kernels: [DimensionalCoulomb; 2],
    nuclear: [Operators; 2],
    source_results: [(ElectrostaticComponents, Vec<Vec3>, Vec<f64>); 2],
    overlap: Vec<BlochMatrix>,
    k_points: Vec<Vec3>,
    centres: usize,
    expected_electrons: f64,
    tolerance: f64,
    quadrature_order: usize,
    requested_charge: i32,
    compensation: ChargeCompensation,
    sources: Vec<(GaussianNucleus, usize)>,
    /// Contracted Hartree integrals per kernel, `[pair][density pair]`, built
    /// on first use; `None` when they exceed [`HARTREE_CACHE_MB`].
    hartree_cache: [std::sync::OnceLock<Option<Vec<Vec<HartreeBlock>>>>; 2],
}
/// Largest memory of the cached Hartree integrals of one kernel; larger
/// bases evaluate the integrals again at every density.
const HARTREE_CACHE_MB: usize = 1024;
/// Primitive-contracted Cartesian four-centre integrals of one shell pair
/// with one density shell pair: the three Coulomb components and the
/// gradients on the four centres, `[i][j][k][l]` row-major. They do not
/// depend on the density, so the self-consistent iterations contract them
/// instead of repeating the quadrature.
struct HartreeBlock {
    values: Vec<[f64; 3]>,
    gradients: Vec<[Vec3; 4]>,
}
pub struct NuclearSpinOrbitReport {
    pub operators: [Vec<BlochMatrix>; 3],
    pub quadrature_defect: f64,
    pub hermiticity_defect: f64,
}
pub struct PeriodicCoulombReport {
    pub k_points: Vec<Vec3>,
    pub nuclear: Vec<BlochMatrix>,
    pub nuclear_components: Vec<[BlochMatrix; 3]>,
    pub nuclear_derivatives: Vec<Vec<[BlochMatrix; 3]>>,
    pub hartree: Option<Vec<BlochMatrix>>,
    pub hartree_components: Option<Vec<[BlochMatrix; 3]>>,
    pub hartree_derivatives: Option<Vec<Vec<[BlochMatrix; 3]>>>,
    pub source_energy_components: ElectrostaticComponents,
    pub source_energy: f64,
    pub source_gradients: Vec<Vec3>,
    pub electron_nuclear_components: Option<[f64; 3]>,
    pub hartree_energy_components: Option<[f64; 3]>,
    pub total_coulomb_energy: Option<f64>,
    /// Fixed Cartesian k and fixed supplied AO density coefficients.
    pub energy_gradients: Option<Vec<Vec3>>,
    pub electron_count: Option<f64>,
    pub expected_electrons: f64,
    pub charge_residual: Option<f64>,
    pub formal_charge: i32,
    pub compensating_charge: f64,
    pub compensation: ChargeCompensation,
    pub quadrature_operator_defect: f64,
    pub quadrature_derivative_defect: f64,
    pub quadrature_source_defect: f64,
    pub hermiticity_defect: f64,
    pub translation_defect: f64,
    pub real_space_density_imaginary_max: f64,
    pub hartree_potential_imaginary_defect: f64,
    pub params: EwaldParams,
    pub quadrature_order: usize,
    pub converged: bool,
    pub centre_count: usize,
    pub model: &'static str,
    pub units: &'static str,
    pub warnings: Vec<String>,
}
impl DimensionalCoulombBackend {
    pub fn new(
        mol: &Molecule,
        mb: &MolBasis,
        geometry: &PeriodicGeometry,
        options: &PeriodicCoulombOptions,
    ) -> Result<Self> {
        let invalid = |message: &str| DftError::InvalidInput(message.into());
        let o = &options.electrostatics;
        if mol.is_empty()
            || mb.nao_sph == 0
            || geometry.dimension() == 0
            || !options.basis_cutoff.is_finite()
            || options.basis_cutoff <= 0.0
            || !o.tolerance.is_finite()
            || o.tolerance <= 0.0
            || !(16..=1024).contains(&o.quadrature_order)
            || [o.alpha, o.real_cutoff, o.reciprocal_cutoff]
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "dimensional Coulomb operators require atoms, AOs, 1D/2D/3D geometry and \
                positive finite controls",
            ));
        }
        let mut occupied = vec![false; mb.nao_sph];
        if mol
            .atoms
            .iter()
            .any(|a| !(1..=118).contains(&a.z) || !a.position.norm().is_finite())
            || mb
                .ecps
                .iter()
                .any(|e| e.atom >= mol.len() || e.potential.n_core > mol.atoms[e.atom].z as u32)
        {
            return Err(invalid(
                "Coulomb atoms and effective nuclear charges must be valid and finite",
            ));
        }
        for shell in &mb.shells {
            if shell.l > MAX_L
                || shell.atom >= mol.len()
                || !shell.center.norm().is_finite()
                || (shell.center - mol.atoms[shell.atom].position).norm() > 1e-10
                || shell.exps.is_empty()
                || shell.exps.len() != shell.coefs.len()
                || shell.exps.iter().any(|v| !v.is_finite() || *v <= 0.0)
                || shell.coefs.iter().any(|v| !v.is_finite())
                || shell
                    .sph_off
                    .checked_add(n_sph(shell.l))
                    .is_none_or(|end| end > mb.nao_sph)
            {
                return Err(invalid(
                    "Coulomb shells must have valid exponents, coefficients, angular order, \
                    AO offsets and matching atom centres",
                ));
            }
            for ao in shell.sph_off..shell.sph_off + n_sph(shell.l) {
                if occupied[ao] {
                    return Err(invalid("Coulomb basis spherical AO ranges overlap"));
                }
                occupied[ao] = true;
            }
        }
        if occupied.contains(&false) {
            return Err(invalid("Coulomb basis has unassigned spherical AOs"));
        }
        let widths = if options.nuclear_widths.is_empty() {
            vec![0.0; mol.len()]
        } else {
            options.nuclear_widths.clone()
        };
        if widths.len() != mol.len() || widths.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(invalid(
                "provide one finite non-negative nuclear width per atom",
            ));
        }
        let charges = mb.effective_charges(mol);
        let expected_electrons = charges.iter().sum::<f64>() - mol.charge as f64;
        if expected_electrons < 0.0 {
            return Err(invalid(
                "formal charge exceeds the effective nuclear charge",
            ));
        }
        let mut sources: Vec<_> = mol
            .atoms
            .iter()
            .zip(charges)
            .zip(widths)
            .enumerate()
            .map(|(i, ((a, charge), width))| {
                (
                    GaussianNucleus {
                        center: a.position,
                        charge,
                        width,
                    },
                    i,
                )
            })
            .collect();
        match o.compensation {
            ChargeCompensation::RequireNeutral if mol.charge != 0 => {
                return Err(invalid(
                    "nonzero formal charge requires an explicit Coulomb compensation model",
                ))
            }
            ChargeCompensation::UniformBackground if geometry.dimension() != 3 => {
                return Err(invalid("uniform background requires 3D periodicity"))
            }
            ChargeCompensation::GaussianCountercharge { center, width } => {
                if !center.norm().is_finite() || !width.is_finite() || width <= 0.0 {
                    return Err(invalid(
                        "Gaussian compensation requires finite centre and positive width",
                    ));
                }
                sources.push((
                    GaussianNucleus {
                        center,
                        charge: -(mol.charge as f64),
                        width,
                    },
                    mol.len(),
                ));
            }
            _ => {}
        }
        let centres = mol.len()
            + usize::from(matches!(
                o.compensation,
                ChargeCompensation::GaussianCountercharge { .. }
            ));
        let source_charges: Vec<_> = sources
            .iter()
            .map(|(s, _)| ElectrostaticSource {
                position: s.center,
                charge: s.charge,
                width: s.width,
            })
            .collect();
        let mut pairs = Vec::new();
        for (i, a) in mb.shells.iter().enumerate() {
            for (j, b) in mb.shells.iter().enumerate() {
                for index in geometry
                    .nearby_translation_indices(b.center - a.center, options.basis_cutoff)?
                {
                    pairs.push(PairImage {
                        index,
                        translation: geometry.lattice().translation(index),
                        left: i,
                        right: j,
                    });
                }
            }
        }
        let alpha = o
            .alpha
            .unwrap_or(PI.sqrt() / geometry.measure().powf(1.0 / geometry.dimension() as f64));
        let minimum = mb
            .shells
            .iter()
            .flat_map(|s| s.exps.iter())
            .copied()
            .fold(f64::INFINITY, f64::min);
        let widest = sources.iter().map(|(s, _)| s.width).fold(0.0, f64::max);
        let width = (minimum.recip() + 2.0 * widest * widest).sqrt();
        let order = 4 * mb.shells.iter().map(|s| s.l as usize).max().unwrap_or(0) + 1;
        let margin =
            (-o.tolerance.min(1e-12).ln() + 2.0 * order as f64 * ((order + 2) as f64).ln()).sqrt();
        let params = EwaldParams {
            alpha,
            rcut: o.real_cutoff.unwrap_or(margin * width.max(alpha.recip())),
            gcut: o.reciprocal_cutoff.unwrap_or(2.0 * alpha * margin),
        };
        let kernels = [
            DimensionalCoulomb::new(geometry.clone(), params, o.quadrature_order)?,
            DimensionalCoulomb::new(geometry.clone(), params, 2 * o.quadrature_order)?,
        ];
        let k_points = geometry.k_points(options.k_mesh)?;
        let evaluate = |kernel: &DimensionalCoulomb| -> Result<Operators> {
            let values = pairs
                .par_iter()
                .map(|pair| {
                    let a = &mb.shells[pair.left];
                    let mut b = mb.shells[pair.right].clone();
                    b.center += pair.translation;
                    nuclear_pair(kernel, a, &b, &sources, centres)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(assemble(mb, &pairs, values, centres, &k_points))
        };
        let nuclear = [evaluate(&kernels[0])?, evaluate(&kernels[1])?];
        let source_results = [
            kernels[0].evaluate(&source_charges)?,
            kernels[1].evaluate(&source_charges)?,
        ];
        let overlap_images =
            dimensional_image_blocks(mb, geometry, options.basis_cutoff, Operator::Overlap)?;
        let overlap = k_points
            .iter()
            .map(|&k| BlochMatrix::from_pair(bloch(&overlap_images, k)))
            .collect();
        Ok(Self {
            geometry: geometry.clone(),
            basis: mb.clone(),
            basis_cutoff: options.basis_cutoff,
            pairs,
            kernels,
            nuclear,
            source_results,
            overlap,
            k_points,
            centres,
            expected_electrons,
            tolerance: o.tolerance,
            quadrature_order: 2 * o.quadrature_order,
            requested_charge: mol.charge,
            compensation: o.compensation.clone(),
            sources,
            hartree_cache: [std::sync::OnceLock::new(), std::sync::OnceLock::new()],
        })
    }

    /// Shells of one pair image, the right shell translated.
    fn pair_shells(&self, pair: &PairImage) -> (MolShell, MolShell) {
        let a = self.basis.shells[pair.left].clone();
        let mut b = self.basis.shells[pair.right].clone();
        b.center += pair.translation;
        (a, b)
    }

    /// Cached contracted Hartree integrals of kernel `index`, or `None` when
    /// they exceed [`HARTREE_CACHE_MB`].
    fn hartree_cache(&self, index: usize) -> Result<Option<&Vec<Vec<HartreeBlock>>>> {
        if let Some(cache) = self.hartree_cache[index].get() {
            return Ok(cache.as_ref());
        }
        let carts = |pair: &PairImage| {
            cart_components(self.basis.shells[pair.left].l).len()
                * cart_components(self.basis.shells[pair.right].l).len()
        };
        let total: usize = self.pairs.iter().map(carts).sum();
        let bytes = (total as f64).powi(2) * 15.0 * 8.0;
        let built = if bytes > (HARTREE_CACHE_MB * 1024 * 1024) as f64 {
            None
        } else {
            let kernel = &self.kernels[index];
            Some(
                self.pairs
                    .par_iter()
                    .map(|pair| {
                        let (a, b) = self.pair_shells(pair);
                        self.pairs
                            .iter()
                            .map(|other| {
                                let (c, d) = self.pair_shells(other);
                                hartree_block(kernel, &a, &b, &c, &d)
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        let _ = self.hartree_cache[index].set(built);
        Ok(self.hartree_cache[index].get().and_then(Option::as_ref))
    }

    /// Analytic one-electron Pauli nuclear SOC in the same dimensional Green
    /// gauge as the scalar nuclear attraction. Integration by parts gives
    /// <a|grad(V) cross p|b> = i <grad(a)|V cross grad(b)>.
    /// The image matrices are imaginary before Bloch transformation; their
    /// spinor assembly retains all three Pauli directions.
    pub fn nuclear_spin_orbit(&self) -> Result<[Vec<BlochMatrix>; 3]> {
        self.nuclear_spin_orbit_at_order(1)
    }
    pub fn nuclear_spin_orbit_report(&self) -> Result<NuclearSpinOrbitReport> {
        let coarse = self.nuclear_spin_orbit_at_order(0)?;
        let fine = self.nuclear_spin_orbit_at_order(1)?;
        let quadrature_defect = coarse
            .iter()
            .flatten()
            .zip(fine.iter().flatten())
            .map(|(a, b)| difference(a, b))
            .fold(0.0_f64, f64::max);
        let hermiticity_defect = fine
            .iter()
            .flatten()
            .map(BlochMatrix::hermiticity_error)
            .fold(0.0_f64, f64::max);
        Ok(NuclearSpinOrbitReport {
            operators: fine,
            quadrature_defect,
            hermiticity_defect,
        })
    }
    /// Nuclear Pauli grad(V) cross A contact for A=M r at unit amplitude.
    /// The common origin phase is already removed from the AO gauge frame.
    pub fn nuclear_uniform_contact_report(
        &self,
        matrix: [[f64; 3]; 3],
    ) -> Result<NuclearSpinOrbitReport> {
        if matrix.iter().flatten().any(|v| !v.is_finite()) || !self.basis.ecps.is_empty() {
            return Err(DftError::InvalidInput(
                "uniform nuclear magnetic contact requires a finite linear gauge and \
                all-electron nuclear sources"
                    .into(),
            ));
        }
        let coarse = self.nuclear_uniform_contact_at_order(matrix, 0)?;
        let fine = self.nuclear_uniform_contact_at_order(matrix, 1)?;
        let quadrature_defect = coarse
            .iter()
            .flatten()
            .zip(fine.iter().flatten())
            .map(|(a, b)| difference(a, b))
            .fold(0.0_f64, f64::max);
        let hermiticity_defect = fine
            .iter()
            .flatten()
            .map(BlochMatrix::hermiticity_error)
            .fold(0.0_f64, f64::max);
        Ok(NuclearSpinOrbitReport {
            operators: fine,
            quadrature_defect,
            hermiticity_defect,
        })
    }
    fn nuclear_uniform_contact_at_order(
        &self,
        matrix: [[f64; 3]; 3],
        refinement: usize,
    ) -> Result<[Vec<BlochMatrix>; 3]> {
        let c = 137.035_999_084_f64;
        let results = self
            .pairs
            .par_iter()
            .map(|pair| {
                let a = &self.basis.shells[pair.left];
                let mut b = self.basis.shells[pair.right].clone();
                b.center += pair.translation;
                let ca = cart_components(a.l);
                let cb = cart_components(b.l);
                let mut cart =
                    std::array::from_fn::<_, 3, _>(|_| Matrix::zeros(ca.len(), cb.len()));
                for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                    for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                        let mut gs = vec![
                            CartesianGaussian {
                                center: a.center,
                                exponent: ea,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: b.center,
                                exponent: eb,
                                angular: [0; 3],
                            },
                        ];
                        for (source, _) in &self.sources {
                            let tensor = prepare_integral_tensor(
                                &self.kernels[refinement],
                                &gs,
                                Some(source),
                                (a.l + b.l) as usize + 2,
                            )?;
                            for (i, ia) in ca.iter().enumerate() {
                                for (j, ib) in cb.iter().enumerate() {
                                    gs[0].angular = [ia.0 as usize, ia.1 as usize, ia.2 as usize];
                                    gs[1].angular = [ib.0 as usize, ib.1 as usize, ib.2 as usize];
                                    for axis in 0..3 {
                                        let x = (axis + 1) % 3;
                                        let y = (axis + 2) % 3;
                                        let xy = super::gaussian_coulomb::linear_gradient_integral(
                                            &tensor, &gs, x, matrix[y],
                                        )?;
                                        let yx = super::gaussian_coulomb::linear_gradient_integral(
                                            &tensor, &gs, y, matrix[x],
                                        )?;
                                        cart[axis][(i, j)] +=
                                            -source.charge * wa * wb * (xy - yx) / (4.0 * c * c);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(cart.map(|m| sph_block(a.l, b.l, &m)))
            })
            .collect::<Result<Vec<_>>>()?;
        let n = self.basis.nao_sph;
        let mut out: [Vec<BlochMatrix>; 3] =
            std::array::from_fn(|_| vec![BlochMatrix::zeros(n, n); self.k_points.len()]);
        for (pair, parts) in self.pairs.iter().zip(&results) {
            let a = &self.basis.shells[pair.left];
            let b = &self.basis.shells[pair.right];
            for (ik, k) in self.k_points.iter().enumerate() {
                let (s, c) = k.dot(pair.translation).sin_cos();
                for axis in 0..3 {
                    for i in 0..n_sph(a.l) {
                        for j in 0..n_sph(b.l) {
                            let value = parts[axis][(i, j)];
                            let row = a.sph_off + i;
                            let col = b.sph_off + j;
                            out[axis][ik].re[(row, col)] += value * c;
                            out[axis][ik].im[(row, col)] += value * s;
                        }
                    }
                }
            }
        }
        if out.iter().flatten().any(|h| {
            h.hermiticity_error() > self.tolerance
                || h.re
                    .as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|v| !v.is_finite())
        }) {
            return Err(DftError::InvalidInput(
                "uniform nuclear magnetic contact is non-finite or violates Hermiticity \
                before projection"
                    .into(),
            ));
        }
        Ok(out)
    }
    fn nuclear_spin_orbit_at_order(&self, refinement: usize) -> Result<[Vec<BlochMatrix>; 3]> {
        if !self.basis.ecps.is_empty() {
            return Err(DftError::Unsupported {
                feature: "Pauli nuclear SOC with ECP".into(),
                missing: "the defining angular-momentum resolved spin-orbit core operator; \
                    effective point charge cannot replace it"
                    .into(),
            });
        }
        let c = 137.035_999_084_f64;
        let results = self
            .pairs
            .par_iter()
            .map(|pair| {
                let a = &self.basis.shells[pair.left];
                let mut b = self.basis.shells[pair.right].clone();
                b.center += pair.translation;
                let ca = cart_components(a.l);
                let cb = cart_components(b.l);
                let mut cart =
                    std::array::from_fn::<_, 3, _>(|_| Matrix::zeros(ca.len(), cb.len()));
                for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                    for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                        let mut gs = vec![
                            CartesianGaussian {
                                center: a.center,
                                exponent: ea,
                                angular: [0; 3],
                            },
                            CartesianGaussian {
                                center: b.center,
                                exponent: eb,
                                angular: [0; 3],
                            },
                        ];
                        for (source, _) in &self.sources {
                            let tensor = prepare_integral_tensor(
                                &self.kernels[refinement],
                                &gs,
                                Some(source),
                                (a.l + b.l) as usize + 2,
                            )?;
                            for (i, ia) in ca.iter().enumerate() {
                                for (j, ib) in cb.iter().enumerate() {
                                    gs[0].angular = [ia.0 as usize, ia.1 as usize, ia.2 as usize];
                                    gs[1].angular = [ib.0 as usize, ib.1 as usize, ib.2 as usize];
                                    for axis in 0..3 {
                                        let x = (axis + 1) % 3;
                                        let y = (axis + 2) % 3;
                                        let xy = super::gaussian_coulomb::mixed_basis_derivative(
                                            &tensor, &gs, x, y,
                                        )?;
                                        let yx = super::gaussian_coulomb::mixed_basis_derivative(
                                            &tensor, &gs, y, x,
                                        )?;
                                        cart[axis][(i, j)] +=
                                            -source.charge * wa * wb * (xy - yx) / (4.0 * c * c);
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(cart.map(|m| sph_block(a.l, b.l, &m)))
            })
            .collect::<Result<Vec<_>>>()?;
        let n = self.basis.nao_sph;
        let mut out: [Vec<BlochMatrix>; 3] =
            std::array::from_fn(|_| vec![BlochMatrix::zeros(n, n); self.k_points.len()]);
        for (pair, parts) in self.pairs.iter().zip(&results) {
            let a = &self.basis.shells[pair.left];
            let b = &self.basis.shells[pair.right];
            for (ik, k) in self.k_points.iter().enumerate() {
                let phase = k.dot(pair.translation);
                for axis in 0..3 {
                    for i in 0..n_sph(a.l) {
                        for j in 0..n_sph(b.l) {
                            let value = parts[axis][(i, j)];
                            let row = a.sph_off + i;
                            let col = b.sph_off + j;
                            out[axis][ik].re[(row, col)] -= value * phase.sin();
                            out[axis][ik].im[(row, col)] += value * phase.cos();
                        }
                    }
                }
            }
        }
        if out.iter().flatten().any(|h| {
            h.hermiticity_error() > self.tolerance
                || h.re
                    .as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|v| !v.is_finite())
        }) {
            return Err(DftError::InvalidInput(
                "analytic nuclear SOC is non-finite or violates Hermiticity before projection"
                    .into(),
            ));
        }
        Ok(out)
    }
    fn density_pairs(
        &self,
        density: &[BlochMatrix],
    ) -> Result<(Vec<(MolShell, MolShell, BlochMatrix)>, f64)> {
        self.paired_density(density, true)
    }
    fn paired_density(
        &self,
        density: &[BlochMatrix],
        hermitian: bool,
    ) -> Result<(Vec<(MolShell, MolShell, BlochMatrix)>, f64)> {
        let n = self.basis.nao_sph;
        if density.len() != self.k_points.len()
            || density.iter().any(|d| {
                d.re.rows != n
                    || d.re.cols != n
                    || d.im.rows != n
                    || d.im.cols != n
                    || d.re
                        .as_slice()
                        .iter()
                        .chain(d.im.as_slice())
                        .any(|v| !v.is_finite())
                    || (hermitian && d.hermiticity_error() > self.tolerance)
            })
        {
            return Err(DftError::InvalidInput(
                "provide one finite AO density per k point; uniform reference densities \
                    must be Hermitian"
                    .into(),
            ));
        }
        let mut imaginary_max = 0.0f64;
        let values = self
            .pairs
            .iter()
            .map(|pair| {
                let a = self.basis.shells[pair.left].clone();
                let mut b = self.basis.shells[pair.right].clone();
                b.center += pair.translation;
                let mut matrix = BlochMatrix::zeros(n, n);
                for (&k, d) in self.k_points.iter().zip(density) {
                    let phase = k.dot(pair.translation);
                    let (co, si) = (phase.cos(), phase.sin());
                    let scale = 1.0 / density.len() as f64;
                    for i in 0..n {
                        for j in 0..n {
                            matrix.re[(i, j)] += scale * (co * d.re[(i, j)] + si * d.im[(i, j)]);
                            matrix.im[(i, j)] += scale * (co * d.im[(i, j)] - si * d.re[(i, j)]);
                        }
                    }
                }
                imaginary_max = imaginary_max.max(
                    matrix
                        .im
                        .as_slice()
                        .iter()
                        .copied()
                        .map(f64::abs)
                        .fold(0.0, f64::max),
                );
                let cart = transformed_cart_density(&a, &b, &matrix);
                (a, b, cart)
            })
            .collect();
        Ok((values, imaginary_max))
    }
    /// Interpolate the stationary reference from its original integration
    /// mesh. Target points do not replace the source density quadrature.
    pub(crate) fn reference_operators_at(
        &self,
        density: &[BlochMatrix],
        points: &[Vec3],
    ) -> Result<(Vec<BlochMatrix>, Vec<BlochMatrix>)> {
        let (pairs, _) = self.density_pairs(density)?;
        let values = self
            .pairs
            .par_iter()
            .map(|pair| {
                let a = &self.basis.shells[pair.left];
                let mut b = self.basis.shells[pair.right].clone();
                b.center += pair.translation;
                hartree_pair(&self.kernels[1], a, &b, &pairs, self.centres)
            })
            .collect::<Result<Vec<_>>>()?;
        let operators = assemble(
            &self.basis,
            &self.pairs,
            values,
            self.centres,
            &self.k_points,
        );
        Ok((self.nuclear[1].at(points), operators.at(points)))
    }
    /// Complex (k+q,k) Hartree action. Neither the density nor its potential
    /// is Hermitianized within the same nonuniform Bloch channel.
    pub(crate) fn finite_q_hartree(
        &self,
        transfer: &PeriodicTransfer,
        density: &[BlochMatrix],
        coarse: bool,
    ) -> Result<Vec<BlochMatrix>> {
        if !transfer.matches(&self.geometry) {
            return Err(DftError::InvalidInput(
                "Hartree transfer geometry mismatch".into(),
            ));
        }
        let (sources, _) = self.paired_density(density, false)?;
        let kernel = &self.kernels[usize::from(!coarse)];
        let values = self
            .pairs
            .par_iter()
            .map(|pair| {
                let a = &self.basis.shells[pair.left];
                let mut b = self.basis.shells[pair.right].clone();
                b.center += pair.translation;
                let ca = cart_components(a.l);
                let cb = cart_components(b.l);
                let mut result = PairResult::new(ca.len(), cb.len(), 0);
                let angular = |v: (i32, i32, i32)| [v.0 as usize, v.1 as usize, v.2 as usize];
                for (c, d, p) in &sources {
                    if max_component(p) == 0.0 {
                        continue;
                    }
                    let cc = cart_components(c.l);
                    let cd = cart_components(d.l);
                    for (&ea, &wa) in a.exps.iter().zip(&a.coefs) {
                        for (&eb, &wb) in b.exps.iter().zip(&b.coefs) {
                            for (&ec, &wc) in c.exps.iter().zip(&c.coefs) {
                                for (&ed, &wd) in d.exps.iter().zip(&d.coefs) {
                                    let mut gs = vec![
                                        CartesianGaussian {
                                            center: a.center,
                                            exponent: ea,
                                            angular: [0; 3],
                                        },
                                        CartesianGaussian {
                                            center: b.center,
                                            exponent: eb,
                                            angular: [0; 3],
                                        },
                                        CartesianGaussian {
                                            center: c.center,
                                            exponent: ec,
                                            angular: [0; 3],
                                        },
                                        CartesianGaussian {
                                            center: d.center,
                                            exponent: ed,
                                            angular: [0; 3],
                                        },
                                    ];
                                    let tensor = prepare_quartet(
                                        kernel,
                                        transfer,
                                        &gs,
                                        (a.l + b.l + c.l + d.l) as usize,
                                    )?;
                                    let weight = wa * wb * wc * wd;
                                    for (i, &ia) in ca.iter().enumerate() {
                                        for (j, &ib) in cb.iter().enumerate() {
                                            gs[0].angular = angular(ia);
                                            gs[1].angular = angular(ib);
                                            for (k, &ic) in cc.iter().enumerate() {
                                                for (l, &id) in cd.iter().enumerate() {
                                                    let coefficient =
                                                        Complex::new(p.re[(k, l)], p.im[(k, l)]);
                                                    if coefficient == Complex::ZERO {
                                                        continue;
                                                    }
                                                    gs[2].angular = angular(ic);
                                                    gs[3].angular = angular(id);
                                                    let z = contract_quartet(&tensor, &gs)?
                                                        * coefficient
                                                        * weight;
                                                    result.parts[0].re[(i, j)] += z.re;
                                                    result.parts[0].im[(i, j)] += z.im;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(spherical(a, &b, result))
            })
            .collect::<Result<Vec<_>>>()?;
        let operators = assemble(&self.basis, &self.pairs, values, 0, &self.k_points);
        let out = operators.parts.iter().map(total).collect::<Vec<_>>();
        if out.iter().any(|m| {
            m.re.as_slice()
                .iter()
                .chain(m.im.as_slice())
                .any(|v| !v.is_finite())
        }) {
            return Err(DftError::LinearAlgebra(
                "finite-q Hartree action exceeds finite range".into(),
            ));
        }
        Ok(out)
    }
    /// Apply the exact linear Hartree map to the supplied uniformly weighted
    /// k-point density using the same fine quadrature as SCF.
    pub(crate) fn hartree_action(&self, density: &[BlochMatrix]) -> Result<Vec<BlochMatrix>> {
        let (pairs, _) = self.density_pairs(density)?;
        let operators = self.contract_hartree(1, &pairs)?;
        Ok(operators.parts.iter().map(total).collect())
    }

    fn contract_hartree(
        &self,
        index: usize,
        density_pairs: &[(MolShell, MolShell, BlochMatrix)],
    ) -> Result<Operators> {
        let kernel = &self.kernels[index];
        let cache = self.hartree_cache(index)?;
        let values = self
            .pairs
            .par_iter()
            .enumerate()
            .map(|(p, pair)| {
                let (a, b) = self.pair_shells(pair);
                match cache {
                    Some(blocks) => Ok(hartree_pair_cached(
                        &a,
                        &b,
                        &blocks[p],
                        density_pairs,
                        self.centres,
                    )),
                    None => hartree_pair(kernel, &a, &b, density_pairs, self.centres),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(assemble(
            &self.basis,
            &self.pairs,
            values,
            self.centres,
            &self.k_points,
        ))
    }

    /// Apply the exact linear Hartree map to the supplied uniformly weighted
    /// k-point density. No occupation, XC or self-consistency is inferred.
    pub fn evaluate(&self, density: Option<&[BlochMatrix]>) -> Result<PeriodicCoulombReport> {
        let (hartree, imaginary_density) = if let Some(density) = density {
            let (density_pairs, imaginary) = self.density_pairs(density)?;
            let coarse = self.contract_hartree(0, &density_pairs)?;
            let fine = self.contract_hartree(1, &density_pairs)?;
            (Some([coarse, fine]), imaginary)
        } else {
            (None, 0.0)
        };
        let mut quadrature_operator_defect = 0.0f64;
        let mut quadrature_derivative_defect = 0.0f64;
        let mut hermiticity_defect = 0.0f64;
        let mut translation_defect = 0.0f64;
        for pair in std::iter::once(&self.nuclear).chain(hartree.as_ref()) {
            if pair.iter().any(|op| {
                op.parts
                    .iter()
                    .flatten()
                    .chain(op.derivatives.iter().flatten().flatten())
                    .any(|m| {
                        m.re.as_slice()
                            .iter()
                            .chain(m.im.as_slice())
                            .any(|v| !v.is_finite())
                    })
            }) {
                return Err(DftError::InvalidInput(
                    "contracted Coulomb operator exceeds finite numerical range".into(),
                ));
            }
            for (old, new) in pair[0].parts.iter().zip(&pair[1].parts) {
                for component in 0..3 {
                    quadrature_operator_defect = quadrature_operator_defect
                        .max(difference(&old[component], &new[component]));
                }
                hermiticity_defect = hermiticity_defect.max(total(new).hermiticity_error());
            }
            for (old, new) in pair[0].derivatives.iter().zip(&pair[1].derivatives) {
                for axis in 0..3 {
                    let mut sum = BlochMatrix::zeros(self.basis.nao_sph, self.basis.nao_sph);
                    for (old, new) in old.iter().zip(new) {
                        quadrature_derivative_defect =
                            quadrature_derivative_defect.max(difference(&old[axis], &new[axis]));
                        hermiticity_defect = hermiticity_defect.max(new[axis].hermiticity_error());
                        sum.add_scaled(&new[axis], 1.0);
                    }
                    translation_defect = translation_defect.max(max_component(&sum));
                }
            }
        }
        let (source_components, source_forces, _) = &self.source_results[1];
        let quadrature_source_defect = (source_components.total()
            - self.source_results[0].0.total())
        .abs()
        .max(
            source_forces
                .iter()
                .zip(&self.source_results[0].1)
                .map(|(a, b)| (*a - *b).norm())
                .fold(0.0, f64::max),
        );
        let source_gradients: Vec<_> = source_forces.iter().map(|f| *f * (-1.0)).collect();
        if !source_components.total().is_finite()
            || source_gradients.iter().any(|g| !g.norm().is_finite())
        {
            return Err(DftError::InvalidInput(
                "Coulomb source component exceeds finite numerical range".into(),
            ));
        }
        let trace_parts = |parts: &[[BlochMatrix; 3]], density: &[BlochMatrix], scale: f64| {
            std::array::from_fn(|c| {
                parts
                    .iter()
                    .zip(density)
                    .map(|(m, d)| d.trace_product(&m[c]))
                    .sum::<f64>()
                    * scale
                    / density.len() as f64
            })
        };
        let electron_nuclear_components =
            density.map(|d| trace_parts(&self.nuclear[1].parts, d, 1.0));
        let hartree_energy_components = density
            .zip(hartree.as_ref())
            .map(|(d, j)| trace_parts(&j[1].parts, d, 0.5));
        let total_coulomb_energy = electron_nuclear_components
            .zip(hartree_energy_components)
            .map(|(v, j)| {
                source_components.total() + v.into_iter().sum::<f64>() + j.into_iter().sum::<f64>()
            });
        let energy_gradients = density.zip(hartree.as_ref()).map(|(density, j)| {
            let mut result = source_gradients.clone();
            for (k, d) in density.iter().enumerate() {
                for (centre, gradient) in result.iter_mut().enumerate() {
                    let values = std::array::from_fn::<_, 3, _>(|axis| {
                        (d.trace_product(&self.nuclear[1].derivatives[k][centre][axis])
                            + 0.5 * d.trace_product(&j[1].derivatives[k][centre][axis]))
                            / density.len() as f64
                    });
                    *gradient += Vec3::new(values[0], values[1], values[2]);
                }
            }
            result
        });
        let electron_count = density.map(|d| {
            d.iter()
                .zip(&self.overlap)
                .map(|(d, s)| d.trace_product(s))
                .sum::<f64>()
                / d.len() as f64
        });
        let charge_residual = electron_count.map(|n| n - self.expected_electrons);
        let potential_imaginary = hartree
            .as_ref()
            .map_or(0.0, |j| j[1].imaginary_image_defect);
        let converged = [
            quadrature_operator_defect,
            quadrature_derivative_defect,
            quadrature_source_defect,
            hermiticity_defect,
            translation_defect,
            potential_imaginary,
        ]
        .iter()
        .all(|d| d.is_finite() && *d <= self.tolerance)
            && charge_residual.is_none_or(|d| d.abs() <= self.tolerance);
        let mut warnings = vec![
            "fixed supplied Bloch density operators are not a self-consistent SCF \
            result; independently converge all three lattice cutoffs"
                .into(),
            "energy gradients hold the supplied AO density fixed; SCF forces \
                additionally require stationarity, metric/Pulay and charge constraints"
                .into(),
        ];
        warnings.push(
            "the uncoupled nuclear/countercharge energy is a gauge component; \
            physical Coulomb energy combines it with electronic terms and a \
            satisfied particle-number constraint"
                .into(),
        );
        if charge_residual.is_some_and(|d| d.abs() > self.tolerance) {
            warnings.push(format!(
                "density electron count differs from the requested formal charge {} by {:e}",
                self.requested_charge,
                charge_residual.unwrap()
            ));
        }
        if !converged {
            warnings.push(
                "one or more raw Coulomb, translation, Hermiticity or charge defects \
                exceed tolerance"
                    .into(),
            );
        }
        Ok(PeriodicCoulombReport {
            k_points: self.k_points.clone(),
            nuclear: self.nuclear[1].parts.iter().map(total).collect(),
            nuclear_components: self.nuclear[1].parts.clone(),
            nuclear_derivatives: self.nuclear[1].derivatives.clone(),
            hartree: hartree
                .as_ref()
                .map(|j| j[1].parts.iter().map(total).collect()),
            hartree_components: hartree.as_ref().map(|j| j[1].parts.clone()),
            hartree_derivatives: hartree.as_ref().map(|j| j[1].derivatives.clone()),
            source_energy_components: source_components.clone(),
            source_energy: source_components.total(),
            source_gradients,
            electron_nuclear_components,
            hartree_energy_components,
            total_coulomb_energy,
            energy_gradients,
            electron_count,
            expected_electrons: self.expected_electrons,
            charge_residual,
            formal_charge: self.requested_charge,
            compensating_charge: if matches!(self.compensation, ChargeCompensation::RequireNeutral)
            {
                0.0
            } else {
                -(self.requested_charge as f64)
            },
            compensation: self.compensation.clone(),
            quadrature_operator_defect,
            quadrature_derivative_defect,
            quadrature_source_defect,
            hermiticity_defect,
            translation_defect,
            real_space_density_imaginary_max: imaginary_density,
            hartree_potential_imaginary_defect: potential_imaginary,
            params: self.kernels[1].params(),
            quadrature_order: self.quadrature_order,
            converged,
            centre_count: self.centres,
            model: "contracted spherical AO dimensional nuclear/Hartree Green operators; \
                common compensated-charge gauge; fixed supplied Bloch density",
            units: "operators/energies Hartree; centre derivatives Hartree/Bohr; geometry \
                Bohr; density electrons",
            warnings,
        })
    }
    pub fn geometry(&self) -> &PeriodicGeometry {
        &self.geometry
    }
    pub fn k_points(&self) -> &[Vec3] {
        &self.k_points
    }
}
/// Common driver; SCF may retain the backend and reuse its nuclear/overlap cache.
pub fn periodic_coulomb_operators(
    mol: &Molecule,
    mb: &MolBasis,
    geometry: &PeriodicGeometry,
    density: Option<&[BlochMatrix]>,
    options: &PeriodicCoulombOptions,
) -> Result<PeriodicCoulombReport> {
    DimensionalCoulombBackend::new(mol, mb, geometry, options)?.evaluate(density)
}

#[cfg(test)]
mod finite_transfer_tests {
    use super::*;
    use crate::basis::{BasisSet, ElementBasis, Shell};
    use crate::periodic::{features::PeriodicBoundary, Lattice};
    use crate::system::Atom;

    #[test]
    fn neighboring_image_complex_hartree_matches_supercell_fold_on_all_masks() {
        let mut set = BasisSet::new("primitive-folding-oracle");
        set.insert(ElementBasis {
            z: 1,
            shells: vec![Shell {
                l: 0,
                exps: vec![0.7],
                coefs: vec![1.0],
            }],
        });
        for mask in 1..8 {
            let boundary = PeriodicBoundary::new(std::array::from_fn(|a| mask & (1 << a) != 0));
            let a = boundary.axes.iter().position(|&a| a).unwrap();
            let geometry = PeriodicGeometry::new(Lattice::cubic(5.5), boundary).unwrap();
            let molecule = Molecule::new(vec![Atom {
                z: 1,
                position: Vec3::new(0.2, -0.1, 0.3),
            }])
            .with_multiplicity(2);
            let basis = MolBasis::build(&molecule, &set).unwrap();
            let mut options = PeriodicCoulombOptions {
                basis_cutoff: 6.0,
                ..Default::default()
            };
            options.k_mesh[a] = 3;
            options.electrostatics.alpha = Some(0.65);
            options.electrostatics.real_cutoff = Some(18.0);
            options.electrostatics.reciprocal_cutoff = Some(3.7);
            let primitive =
                DimensionalCoulombBackend::new(&molecule, &basis, &geometry, &options).unwrap();
            let mut f = [0.0; 3];
            f[a] = 1.0 / 3.0;
            let transfer = PeriodicTransfer::new(&geometry, f).unwrap();
            let density: Vec<_> = primitive
                .k_points
                .iter()
                .enumerate()
                .map(|(ik, _)| {
                    let mut d = BlochMatrix::zeros(1, 1);
                    d.re[(0, 0)] = 0.17 + 0.09 * ik as f64;
                    d.im[(0, 0)] = -0.11 + 0.07 * ik as f64;
                    d
                })
                .collect();
            let actual = primitive
                .finite_q_hartree(&transfer, &density, false)
                .unwrap();
            let translation = geometry.lattice().a[a];
            let mut super_lattice = geometry.lattice().clone();
            super_lattice.a[a] = translation * 3.0;
            let super_geometry = PeriodicGeometry::new(super_lattice, boundary).unwrap();
            let super_molecule = Molecule::new(
                (0..3)
                    .map(|s| Atom {
                        z: 1,
                        position: molecule.atoms[0].position + translation * s as f64,
                    })
                    .collect(),
            )
            .with_multiplicity(2);
            let super_basis = MolBasis::build(&super_molecule, &set).unwrap();
            options.k_mesh = [1; 3];
            let super_backend = DimensionalCoulombBackend::new(
                &super_molecule,
                &super_basis,
                &super_geometry,
                &options,
            )
            .unwrap();
            let q = transfer.reduced_cartesian();
            let mut super_density = BlochMatrix::zeros(3, 3);
            for s in 0..3 {
                for t in 0..3 {
                    let z = primitive.k_points.iter().zip(&density).fold(
                        Complex::ZERO,
                        |sum, (&k, d)| {
                            let angle =
                                (k + q).dot(translation) * s as f64 - k.dot(translation) * t as f64;
                            sum + Complex::new(angle.cos(), angle.sin())
                                * Complex::new(d.re[(0, 0)], d.im[(0, 0)])
                                / 3.0
                        },
                    );
                    super_density.re[(s, t)] = z.re;
                    super_density.im[(s, t)] = z.im;
                }
            }
            let uniform = PeriodicTransfer::new(&super_geometry, [0.0; 3]).unwrap();
            let super_potential = super_backend
                .finite_q_hartree(&uniform, &[super_density], false)
                .unwrap();
            for (ik, &k) in primitive.k_points.iter().enumerate() {
                let mut expected = Complex::ZERO;
                for s in 0..3 {
                    for t in 0..3 {
                        let angle =
                            -(k + q).dot(translation) * s as f64 + k.dot(translation) * t as f64;
                        expected += Complex::new(angle.cos(), angle.sin())
                            * Complex::new(
                                super_potential[0].re[(s, t)],
                                super_potential[0].im[(s, t)],
                            )
                            / 3.0;
                    }
                }
                let value = Complex::new(actual[ik].re[(0, 0)], actual[ik].im[(0, 0)]);
                let defect = (value - expected).abs();
                println!(
                    "dimensional neighboring-image Hartree fold mask={mask} k={ik} \
                    defect={defect:.12e}"
                );
                assert!(
                    defect < 1e-10,
                    "mask={mask} k={ik} primitive={value:?} super={expected:?}"
                );
            }
        }
    }
}

mod affine;
mod point_lattice;
pub use affine::PeriodicCoulombStrain;
