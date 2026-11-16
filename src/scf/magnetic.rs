// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Self-consistent field of a molecule in a finite uniform magnetic field
//! with London orbitals.
//!
//! Every integral over London orbitals is an entire function of the field,
//! so the Taylor coefficients of [`crate::integrals::london`] are summed at
//! the requested field until the last retained term is negligible. The
//! series expand `exp(i k . r)` about the coordinate origin, so the molecule
//! is moved to its geometric centre first; this multiplies every London
//! orbital by a constant phase and changes no observable. The overlap and
//! the one-electron matrix are the sums of their series; the field-free
//! two-electron integrals act through the integral-direct engines, and their
//! field coefficients are contracted quartet by quartet in one pass. The
//! semilocal functional is evaluated on the grid from the complex orbitals
//! themselves, with the gauge-invariant kinetic energy density
//! `sum_i |(-i grad + A) psi_i|^2 / 2` on the legacy mechanical route.
//! The explicit-core route uses the current-invariant kinetic density and
//! includes complete learned and VV10 energy/potential providers. The result
//! records the chosen auxiliary model; moving-nuclear forces require their
//! corresponding derivatives. The orbitals solve the complex Hermitian
//! generalized eigenproblem `F C = S C e`.

use super::diis::Diis;
use super::fock::JkEngine;
use super::{ScfOptions, ScfResult};
use crate::basis::{n_sph, MolBasis, MolShell};
use crate::dfpt::orbitals::generalized_orbitals;
use crate::dual::Scalar;
use crate::error::{DftError, Result};
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::integrals::london::{
    contract_summed_repulsion, contracted_repulsion_series, one_electron_gradient,
    one_electron_series, repulsion_gradient, summed_repulsion_series, ComplexMatrix, FieldPath,
};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::shifted;
use crate::periodic::strain_jets::cos_sin;
use crate::periodic::xc_strain::shell_values;
use crate::response::nuclear::becke_weight;
use crate::system::Molecule;
use crate::xc::{Family, GridVars, MagneticKineticDensity, ResolvedFunctional, XcOut};
use rayon::prelude::*;

type Cm = ComplexMatrix;
type Cx = (f64, f64);

/// Highest order of the field series of the integrals.
const MAX_ORDER: usize = 24;

fn mul(a: Cx, b: Cx) -> Cx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn conj(a: Cx) -> Cx {
    (a.0, -a.1)
}
fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += weight * b;
    }
}
fn cadd(target: &mut Cm, source: &Cm, weight: f64) {
    add(&mut target.re, &source.re, weight);
    add(&mut target.im, &source.im, weight);
}
fn cmul(a: &Cm, b: &Cm) -> Cm {
    a.matmul_seq(b)
}
/// `Re Tr[a b]`.
fn trace(a: &Cm, b: &Cm) -> f64 {
    a.real_trace_product(b)
}
fn largest(a: &Cm) -> f64 {
    a.re.as_slice()
        .iter()
        .chain(a.im.as_slice())
        .fold(0.0_f64, |m, x| m.max(x.abs()))
}
/// `(b x v) / 2`.
fn half_cross(b: [f64; 3], v: [f64; 3]) -> [f64; 3] {
    [
        0.5 * (b[1] * v[2] - b[2] * v[1]),
        0.5 * (b[2] * v[0] - b[0] * v[2]),
        0.5 * (b[0] * v[1] - b[1] * v[0]),
    ]
}

/// Controls of [`magnetic_field_scf`].
#[derive(Clone, Debug)]
pub struct MagneticFieldOptions {
    /// Uniform magnetic field in atomic units (1 a.u. = 2.35052e5 T).
    pub field: [f64; 3],
    /// Add the spin Zeeman term `g_e B . S / 2` with the spins quantized
    /// along the field.
    pub spin_zeeman: bool,
    /// Highest order of the field series of the integrals; `None` chooses it
    /// from the field, the geometry and `series_tolerance`.
    pub series_order: Option<usize>,
    /// Bound on the last retained coefficient of the one-electron series.
    pub series_tolerance: f64,
}

/// Self-consistent state in the field.
#[derive(Clone, Debug)]
pub struct MagneticFieldResult {
    /// Magnetic extension of any meta-GGA kinetic-density dependence.
    pub kinetic_density_model: MagneticKineticDensity,
    /// Total energy in the field, Hartree.
    pub energy: f64,
    /// Applied electric field in the original laboratory coordinates.
    pub electric_field: [f64; 3],
    /// Spin Zeeman part of it.
    pub zeeman_energy: f64,
    pub e_xc: f64,
    /// Variational two-endpoint nonlocal correlation energy, Hartree.
    pub e_nlc: f64,
    /// Explicit auxiliary providers used by this stationary energy.
    pub nonlocal_parameters: Option<crate::nlc::Vv10Params>,
    pub learned_functional: bool,
    /// Complex Hermitian spin densities `C C^H` over the London orbitals of
    /// the centred molecule.
    pub density: [ComplexMatrix; 2],
    pub fock: [ComplexMatrix; 2],
    pub overlap: ComplexMatrix,
    pub core: ComplexMatrix,
    pub orbital_energies: [Vec<f64>; 2],
    pub coefficients: [ComplexMatrix; 2],
    pub n_alpha: usize,
    pub n_beta: usize,
    pub unrestricted: bool,
    pub converged: bool,
    pub iterations: usize,
    /// Largest element of `F D S - S D F`.
    pub residual: f64,
    /// Order and maximum of the first two omitted one-electron coefficients.
    pub series_order: usize,
    pub series_tail: f64,
    /// Translation that moved the molecule to its geometric centre, Bohr.
    pub shift: [f64; 3],
}

fn least(shell: &MolShell) -> f64 {
    shell.exps.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Order at which the series of a product of two phases has decayed below
/// `tolerance`: the phase of a pair of functions is at most `|k| |r|` with
/// `k = b x (R_m - R_n) / 2` and `r` up to four widths beyond the atoms.
fn series_order(mol: &Molecule, mb: &MolBasis, field: [f64; 3], tolerance: f64) -> usize {
    let reach = mol
        .atoms
        .iter()
        .map(|a| a.position.norm())
        .fold(0.0_f64, f64::max);
    let mut x = 0.0_f64;
    for a in &mb.shells {
        for b in &mb.shells {
            let k = half_cross(field, (a.center - b.center).to_array());
            let size = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
            let width = 4.0 / least(a).min(least(b)).sqrt();
            x = x.max(size * (reach + width));
        }
    }
    // A two-electron integral carries two such phases.
    let x = 2.0 * x;
    let mut term = 1.0;
    let mut order = 0;
    while order < MAX_ORDER {
        order += 1;
        term *= x / order as f64;
        // The diamagnetic term is of second order in the field.
        if term < tolerance && order >= 2 {
            break;
        }
    }
    order
}

type XcPart = (f64, [(Vec<f64>, Vec<f64>); 2]);

/// Estimate the London phase degree after the same centering as the SCF.
pub(crate) fn recommended_series_order(
    mol: &Molecule,
    mb: &MolBasis,
    field: [f64; 3],
    tolerance: f64,
) -> usize {
    let centre = mol
        .atoms
        .iter()
        .fold(Vec3::zero(), |sum, atom| sum + atom.position)
        / mol.len().max(1) as f64;
    let mut molecule = mol.clone();
    for atom in &mut molecule.atoms {
        atom.position -= centre;
    }
    series_order(
        &molecule,
        &shifted(mb, Vec3::zero() - centre),
        field,
        tolerance,
    )
}

/// Exchange-correlation energy and matrices of complex spin densities over
/// London orbitals on the molecular grid.
fn xc_build(
    mb: &MolBasis,
    points: &[Vec3],
    weights: &[f64],
    functional: &ResolvedFunctional,
    field: [f64; 3],
    density: &[Cm; 2],
    kinetic_model: MagneticKineticDensity,
) -> Result<(f64, [Cm; 2])> {
    let nao = mb.nao_sph;
    let gga = functional.family != Family::Lda;
    let meta = functional.family == Family::MetaGga;
    let mut centres = vec![[0.0; 3]; nao];
    for shell in &mb.shells {
        for k in 0..n_sph(shell.l) {
            centres[shell.sph_off + k] = shell.center.to_array();
        }
    }
    // Gradient of the phase of every function.
    let slopes: Vec<[f64; 3]> = centres.iter().map(|c| half_cross(field, *c)).collect();
    const BATCH: usize = 128;
    let starts: Vec<usize> = (0..points.len()).step_by(BATCH).collect();
    let zero = || -> XcPart {
        (
            0.0,
            std::array::from_fn(|_| (vec![0.0; nao * nao], vec![0.0; nao * nao])),
        )
    };
    let total = starts
        .par_iter()
        .try_fold(zero, |mut acc, &start| -> Result<XcPart> {
            let end = (start + BATCH).min(points.len());
            let batch = eval_aos_full(mb, &points[start..end], true, false);
            let gradients = batch.grad.as_ref().expect("gradients were requested");
            let npts = end - start;
            // Phased functions, their gradients and (-i grad + A) on them.
            let mut w = vec![vec![(0.0, 0.0); nao]; npts];
            let mut gw = vec![vec![[(0.0, 0.0); 3]; nao]; npts];
            let mut pw = vec![vec![[(0.0, 0.0); 3]; nao]; npts];
            let mut rho = [vec![0.0; npts], vec![0.0; npts]];
            let mut grad = [vec![[0.0; 3]; npts], vec![[0.0; 3]; npts]];
            let mut tau = [vec![0.0; npts], vec![0.0; npts]];
            let mut current=[vec![[0.0;3];npts],vec![[0.0;3];npts]];
            for i in 0..npts {
                let r = points[start + i].to_array();
                for mu in 0..nao {
                    let theta = slopes[mu][0] * r[0] + slopes[mu][1] * r[1] + slopes[mu][2] * r[2];
                    let e: Cx = (theta.cos(), -theta.sin());
                    let phi = batch.phi[(i, mu)];
                    w[i][mu] = (phi * e.0, phi * e.1);
                    let gauge = half_cross(field, std::array::from_fn(|a| r[a] - centres[mu][a]));
                    for a in 0..3 {
                        let g = gradients[a][(i, mu)];
                        gw[i][mu][a] = mul((g, -slopes[mu][a] * phi), e);
                        pw[i][mu][a] = mul((gauge[a] * phi, -g), e);
                    }
                }
                for s in 0..2 {
                    for nu in 0..nao {
                        let mut y: Cx = (0.0, 0.0);
                        let mut z = [(0.0, 0.0); 3];
                        for mu in 0..nao {
                            let d = (density[s].re[(mu, nu)], density[s].im[(mu, nu)]);
                            if d.0 == 0.0 && d.1 == 0.0 {
                                continue;
                            }
                            let v = mul(d, w[i][mu]);
                            y = (y.0 + v.0, y.1 + v.1);
                            if meta {
                                for a in 0..3 {
                                    let v = mul(d, pw[i][mu][a]);
                                    z[a] = (z[a].0 + v.0, z[a].1 + v.1);
                                }
                            }
                        }
                        rho[s][i] += mul(y, conj(w[i][nu])).0;
                        if gga {
                            for a in 0..3 {
                                grad[s][i][a] += 2.0 * mul(y, conj(gw[i][nu][a])).0;
                            }
                        }
                        if meta {
                            for a in 0..3 {
                                tau[s][i] += 0.5 * mul(z[a], conj(pw[i][nu][a])).0;
                                current[s][i][a]+=mul(z[a],conj(w[i][nu])).0;
                            }
                        }
                    }
                    rho[s][i] = rho[s][i].max(0.0);
                    tau[s][i] = tau[s][i].max(0.0);
                    if meta && kinetic_model==MagneticKineticDensity::CurrentInvariant {
                        if rho[s][i]>0.0 {
                            tau[s][i]-=0.5*current[s][i].iter().map(|&j|j*(j/rho[s][i])).sum::<f64>();
                        } else if current[s][i].iter().any(|&j|j!=0.0) {
                            return Err(DftError::InvalidInput("a current-invariant finite-field kinetic density needs a positive density or an inactive spin channel".into()));
                        }
                    }
                }
            }
            let sigma: [Vec<f64>; 3] = [(0, 0), (0, 1), (1, 1)].map(|(s, t)| {
                (0..npts)
                    .map(|i| (0..3).map(|a| grad[s][i][a] * grad[t][i][a]).sum())
                    .collect()
            });
            let vars = GridVars {
                rho: [&rho[0], &rho[1]],
                sigma: gga.then_some([&sigma[0], &sigma[1], &sigma[2]]),
                tau: meta.then_some([&tau[0], &tau[1]]),
            };
            let mut out = XcOut::default();
            functional.eval(&vars, 1, &mut out)?;
            for i in 0..npts {
                let weight = weights[start + i];
                acc.0 += weight * out.e[i];
                for s in 0..2 {
                    let mut v_rho = out.vrho[s][i];
                    let b: [f64; 3] = if gga {
                        std::array::from_fn(|a| {
                            2.0 * out.vsigma[2 * s][i] * grad[s][i][a]
                                + out.vsigma[1][i] * grad[1 - s][i][a]
                        })
                    } else {
                        [0.0; 3]
                    };
                    let v_tau = if meta { out.vtau[s][i] } else { 0.0 };
                    let velocity=if meta && kinetic_model==MagneticKineticDensity::CurrentInvariant && rho[s][i]>0.0 {
                        current[s][i].map(|j|j/rho[s][i])
                    }else {[0.0;3]};
                    v_rho+=0.5*v_tau*velocity.iter().map(|x|x*x).sum::<f64>();
                    for nu in 0..nao {
                        let mut right = (v_rho * w[i][nu].0, v_rho * w[i][nu].1);
                        for a in 0..3 {
                            right.0 += b[a] * gw[i][nu][a].0;
                            right.1 += b[a] * gw[i][nu][a].1;
                            right.0-=0.5*v_tau*velocity[a]*pw[i][nu][a].0;
                            right.1-=0.5*v_tau*velocity[a]*pw[i][nu][a].1;
                        }
                        for mu in 0..nao {
                            let mut value = mul(conj(w[i][mu]), right);
                            for a in 0..3 {
                                if gga {
                                    let v = mul(
                                        conj(gw[i][mu][a]),
                                        (b[a] * w[i][nu].0, b[a] * w[i][nu].1),
                                    );
                                    value = (value.0 + v.0, value.1 + v.1);
                                }
                                if meta {
                                    let v = mul(
                                        conj(pw[i][mu][a]),
                                        (
                                            0.5 * v_tau * (pw[i][nu][a].0-velocity[a]*w[i][nu].0),
                                            0.5 * v_tau * (pw[i][nu][a].1-velocity[a]*w[i][nu].1),
                                        ),
                                    );
                                    value = (value.0 + v.0, value.1 + v.1);
                                }
                            }
                            acc.1[s].0[mu * nao + nu] += weight * value.0;
                            acc.1[s].1[mu * nao + nu] += weight * value.1;
                        }
                    }
                }
            }
            Ok(acc)
        })
        .try_reduce(zero, |mut a, b| {
            a.0 += b.0;
            for s in 0..2 {
                for (t, v) in a.1[s].0.iter_mut().zip(&b.1[s].0) {
                    *t += *v;
                }
                for (t, v) in a.1[s].1.iter_mut().zip(&b.1[s].1) {
                    *t += *v;
                }
            }
            Ok(a)
        })?;
    let (energy, flat) = total;
    let [first, second] = flat;
    let matrix = |part: (Vec<f64>, Vec<f64>)| Cm {
        re: Matrix::from_row_major(nao, nao, part.0),
        im: Matrix::from_row_major(nao, nao, part.1),
    };
    Ok((energy, [matrix(first), matrix(second)]))
}

/// `C_occ C_occ^H`.
fn occupied_density(c: &BlochMatrix, n_occ: usize) -> Cm {
    let n = c.rows();
    let mut out = Cm::zeros(n, n);
    for m in 0..n {
        for k in 0..n {
            let mut value: Cx = (0.0, 0.0);
            for i in 0..n_occ {
                let v = mul((c.re[(m, i)], c.im[(m, i)]), (c.re[(k, i)], -c.im[(k, i)]));
                value = (value.0 + v.0, value.1 + v.1);
            }
            out.re[(m, k)] = value.0;
            out.im[(m, k)] = value.1;
        }
    }
    out
}

/// Self-consistent field of `mol` in the uniform magnetic field of
/// `options`, started from the zero-field state `scf`.
#[allow(clippy::too_many_arguments)]
pub fn magnetic_field_scf(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    options: &MagneticFieldOptions,
) -> Result<MagneticFieldResult> {
    magnetic_field_scf_impl(
        mol,
        mb,
        functional,
        grid,
        Some(scf),
        opts,
        options,
        None,
        MagneticKineticDensity::Mechanical,
    )
}

/// Start directly from the core Hamiltonian in the actual London overlap
/// metric. A finite-field state does not require convergence of a different
/// zero-field Hamiltonian. The final physical convergence tests are identical
/// to those of [`magnetic_field_scf`].
#[allow(clippy::too_many_arguments)]
pub fn magnetic_field_scf_from_core(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    opts: &ScfOptions,
    options: &MagneticFieldOptions,
) -> Result<MagneticFieldResult> {
    magnetic_field_scf_impl(
        mol,
        mb,
        functional,
        grid,
        None,
        opts,
        options,
        None,
        MagneticKineticDensity::Mechanical,
    )
}

/// Converge the same finite-field problem with additional Hermitian spin
/// operators in the London AO basis of the geometrically centred molecule.
/// The supplied matrices already include the finite perturbation strength.
/// This explicit convention avoids mixing laboratory AO phases with the
/// centred basis used by the finite-field integral engine.
/// Meta-GGA terms use the current-invariant kinetic density, consistently
/// with the complex post-SCF field response provider.
#[allow(clippy::too_many_arguments)]
pub fn magnetic_field_scf_with_core(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    options: &MagneticFieldOptions,
    core: &[ComplexMatrix; 2],
) -> Result<MagneticFieldResult> {
    magnetic_field_scf_impl(
        mol,
        mb,
        functional,
        grid,
        Some(scf),
        opts,
        options,
        Some(core),
        MagneticKineticDensity::CurrentInvariant,
    )
}

#[allow(clippy::too_many_arguments)]
fn magnetic_field_scf_impl(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    initial: Option<&ScfResult>,
    opts: &ScfOptions,
    options: &MagneticFieldOptions,
    extra_core: Option<&[Cm; 2]>,
    kinetic_model: MagneticKineticDensity,
) -> Result<MagneticFieldResult> {
    if let Some(extra) = extra_core {
        for h in extra {
            if h.re.rows != mb.nao_sph
                || h.re.cols != mb.nao_sph
                || h.im.rows != mb.nao_sph
                || h.im.cols != mb.nao_sph
                || h.re
                    .as_slice()
                    .iter()
                    .chain(h.im.as_slice())
                    .any(|x| !x.is_finite())
            {
                return Err(DftError::InvalidInput(
                    "finite-field extra core dimensions or values are invalid".into(),
                ));
            }
            for i in 0..mb.nao_sph {
                for j in 0..mb.nao_sph {
                    if (h.re[(i, j)] - h.re[(j, i)]).abs() > 1e-12
                        || (h.im[(i, j)] + h.im[(j, i)]).abs() > 1e-12
                    {
                        return Err(DftError::InvalidInput(
                            "finite-field extra core must be Hermitian".into(),
                        ));
                    }
                }
            }
        }
    }
    if opts.occupation.is_some() || opts.fixed_occupations.is_some() {
        return Err(DftError::Unsupported {
            feature: "complex magnetic determinant tracking".into(),
            missing: "the complex AO-metric occupation selector in the London spin blocks".into(),
        });
    }
    if !mb.ecps.is_empty()
        || opts.pcm.is_some()
        || opts.density_fit.is_some()
        || opts.extra_hcore.is_some()
        || functional.required_nonlocal_kind().is_some_and(|kind| {
            kind != crate::xc::RequiredNonlocalTerm::Vv10 || opts.vv10.is_none()
        })
        || !matches!(opts.kernel, CoulombKernel::Coulomb)
        || opts.electronic_temperature > 0.0
        || initial.is_some_and(|s| s.electronic_temperature > 0.0)
    {
        return Err(DftError::Unsupported {
            feature: "finite magnetic field SCF with effective core potentials, solvent, \
                      nonlocal correlation, density fitting, \
                      additional one-electron terms or smeared occupations"
                .into(),
            missing: "those terms over London orbitals at a finite field".into(),
        });
    }
    let field = options.field;
    let electric = opts.external_field.unwrap_or(Vec3::zero());
    if field.iter().any(|v| !v.is_finite())
        || !electric.norm().is_finite()
        || !(options.series_tolerance.is_finite() && options.series_tolerance > 0.0)
        || options.series_order.is_some_and(|order| order > MAX_ORDER)
    {
        return Err(DftError::InvalidInput(
            "the magnetic field has three finite components, the series tolerance is positive and the series order is at most 24"
                .into(),
        ));
    }
    let has_xc = !functional.parts.is_empty();
    if (has_xc || opts.skala.is_some()) && grid.is_none() {
        return Err(DftError::InvalidInput(
            "the finite-field Kohn-Sham build needs the molecular grid".into(),
        ));
    }
    let nao = mb.nao_sph;
    let natoms = mol.len().max(1) as f64;
    // Geometric centre at the origin.
    let mut centre = Vec3::zero();
    for atom in &mol.atoms {
        centre += atom.position;
    }
    let shift = Vec3::zero() - centre / natoms;
    let mut molecule = mol.clone();
    for atom in &mut molecule.atoms {
        atom.position += shift;
    }
    let basis = shifted(mb, shift);
    let points: Vec<Vec3> = grid
        .map(|g| g.points.iter().map(|p| *p + shift).collect())
        .unwrap_or_default();
    let weights: &[f64] = grid.map_or(&[], |g| &g.weights);
    let learned = opts.skala.as_ref().map(|model| {
        let mut shifted_grid = grid.expect("the learned magnetic grid was checked").clone();
        shifted_grid.points = points.clone();
        crate::response::field_path::LearnedFiniteField::new(
            model,
            &molecule,
            &basis,
            &shifted_grid,
            field,
            kinetic_model,
        )
    });

    let nonlocal = opts
        .vv10
        .map(|parameters| {
            crate::response::field_path::NonlocalFiniteField::new(
                &molecule, &basis, parameters, opts, field,
            )
        })
        .transpose()?;
    let order = match options.series_order {
        Some(order) => order.max(2),
        None => series_order(&molecule, &basis, field, options.series_tolerance),
    };
    let path = FieldPath {
        electric: electric.to_array(),
        magnetic: field,
        origin: [0.0; 3],
        london: true,
        order,
    };
    // A polynomial can terminate at the last retained coefficient: axial
    // London phases vanish but the physical B^2 diamagnetic term remains.
    // Its size is not a truncation error. Probe both omitted parities.
    let guard_path = FieldPath {
        order: order + 2,
        ..path
    };
    let one = one_electron_series(&molecule, &basis, &guard_path)?;
    let mut overlap = Cm::zeros(nao, nao);
    let mut core = Cm::zeros(nao, nao);
    for n in 0..=order {
        cadd(&mut overlap, &one.overlap[n], 1.0);
        cadd(&mut core, &one.core[n], 1.0);
    }
    let series_tail = (order + 1..=order + 2)
        .map(|n| largest(&one.overlap[n]).max(largest(&one.core[n])))
        .fold(0.0_f64, f64::max);
    if series_tail > 1e3 * options.series_tolerance {
        return Err(DftError::Unsupported {
            feature: format!(
                "finite magnetic field SCF: the field series of the integrals has not converged \
                 at order {order} (omitted coefficient {series_tail:.3e})"
            ),
            missing: "a higher series order (at most 24) or a weaker field".into(),
        });
    }
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let want_k = cx != 0.0;
    let jk = JkEngine::from_options(&molecule, &basis, opts)?;
    let jk_lr = if clr != 0.0 {
        Some(JkEngine::new(
            &basis,
            opts.eri_mode,
            CoulombKernel::Erf {
                omega: functional.hybrid.omega,
            },
        )?)
    } else {
        None
    };
    // The field series of the repulsion integrals is the same in every
    // iteration and every order carries the same density, so the SCF needs
    // only the sum of its coefficients. Store it when the tensors fit the
    // in-core budget; otherwise the quartet-direct series is regenerated per
    // iteration.
    let kernels = 1 + usize::from(jk_lr.is_some());
    let summed_mib = (2 * 8 * kernels) as f64 * (nao as f64).powi(4) / (1u64 << 20) as f64;
    let summed = if summed_mib <= opts.incore_mb as f64 {
        let full = summed_repulsion_series(&basis, &path, None, order)?;
        let long_range = if jk_lr.is_some() {
            Some(summed_repulsion_series(
                &basis,
                &path,
                Some(functional.hybrid.omega),
                order,
            )?)
        } else {
            None
        };
        crate::logger::log(&format!(
            "[magnetic-scf] London repulsion series of order {order} stored ({summed_mib:.1} MiB)"
        ));
        Some((full, long_range))
    } else {
        None
    };
    // Recentring changes a charged system's scalar-potential energy. Restore
    // the original laboratory origin after adding the nuclear electric term.
    let e_nuc = molecule.nuclear_repulsion()
        - molecule
            .atoms
            .iter()
            .map(|a| a.z as f64 * electric.dot(a.position))
            .sum::<f64>()
        + mol.charge as f64 * electric.dot(shift);
    let strength = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
    // g_e B S_z / 2: +g_e B / 4 per alpha electron, -g_e B / 4 per beta one.
    let zeeman = if options.spin_zeeman {
        0.25 * crate::constants::ELECTRON_G_FACTOR * strength
    } else {
        0.0
    };
    let zero_k = || [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    // Kohn-Sham matrices, total energy, exchange-correlation and Zeeman
    // energies of a pair of spin densities.
    let build = |d: &[Cm; 2]| -> Result<([Cm; 2], f64, f64, f64, f64)> {
        let mut total = d[0].clone();
        cadd(&mut total, &d[1], 1.0);
        // Field-free integrals: the real symmetric part of the density gives
        // J and K, its imaginary antisymmetric part an exchange matrix only.
        let (j_re, k_re) = jk.build(&basis, &total.re, [&d[0].re, &d[1].re], want_k)?;
        let k_im = if want_k {
            jk.build_k_only(&basis, [&d[0].im, &d[1].im])?
        } else {
            zero_k()
        };
        let [k_re0, k_re1] = k_re;
        let [k_im0, k_im1] = k_im;
        let mut j = Cm {
            re: j_re,
            im: Matrix::zeros(nao, nao),
        };
        let mut k = [
            Cm {
                re: k_re0,
                im: k_im0,
            },
            Cm {
                re: k_re1,
                im: k_im1,
            },
        ];
        let mut k_lr = [Cm::zeros(nao, nao), Cm::zeros(nao, nao)];
        if let Some(engine) = &jk_lr {
            let [re0, re1] = engine.build_k_only(&basis, [&d[0].re, &d[1].re])?;
            let [im0, im1] = engine.build_k_only(&basis, [&d[0].im, &d[1].im])?;
            k_lr = [Cm { re: re0, im: im0 }, Cm { re: re1, im: im1 }];
        }
        // Field coefficients of the integrals: with the same density at every
        // order, the order-`order` coefficient of the contraction is the sum
        // of all of them.
        let ((dj, dk), dk_lr) = if let Some((full, long_range)) = &summed {
            let densities = [&d[0], &d[1]];
            (
                contract_summed_repulsion(full, nao, densities, true, want_k),
                long_range
                    .as_ref()
                    .map(|g| contract_summed_repulsion(g, nao, densities, false, true).1),
            )
        } else {
            let repeated: [Vec<Cm>; 2] = std::array::from_fn(|s| vec![d[s].clone(); order]);
            let copies = [&repeated[0][..], &repeated[1][..]];
            let full =
                contracted_repulsion_series(&basis, &path, None, copies, order, true, want_k)?;
            let long_range = if jk_lr.is_some() {
                Some(
                    contracted_repulsion_series(
                        &basis,
                        &path,
                        Some(functional.hybrid.omega),
                        copies,
                        order,
                        false,
                        true,
                    )?
                    .1,
                )
            } else {
                None
            };
            (full, long_range)
        };
        cadd(&mut j, &dj, 1.0);
        for s in 0..2 {
            cadd(&mut k[s], &dk[s], 1.0);
        }
        if let Some(dk) = dk_lr {
            for s in 0..2 {
                cadd(&mut k_lr[s], &dk[s], 1.0);
            }
        }
        let mut energy = e_nuc + trace(&total, &core) + 0.5 * trace(&total, &j);
        let mut fock = [core.clone(), core.clone()];
        for s in 0..2 {
            cadd(&mut fock[s], &j, 1.0);
            cadd(&mut fock[s], &k[s], -cx);
            cadd(&mut fock[s], &k_lr[s], -clr);
            energy -= 0.5 * cx * trace(&d[s], &k[s]);
            energy -= 0.5 * clr * trace(&d[s], &k_lr[s]);
        }
        if let Some(extra) = extra_core {
            for s in 0..2 {
                cadd(&mut fock[s], &extra[s], 1.0);
                energy += trace(&d[s], &extra[s]);
            }
        }
        let mut exc = 0.0;
        if has_xc {
            let (value, v) = xc_build(
                &basis,
                &points,
                weights,
                functional,
                field,
                d,
                kinetic_model,
            )?;
            exc = value;
            energy += value;
            for s in 0..2 {
                cadd(&mut fock[s], &v[s], 1.0);
            }
        }
        if let Some(provider) = &learned {
            let (value, potential) = provider.evaluate(d)?;
            exc += value;
            energy += value;
            for s in 0..2 {
                cadd(&mut fock[s], &potential[s], 1.0);
            }
        }
        let mut enlc = 0.0;
        if let Some(provider) = &nonlocal {
            let (value, potential) = provider.evaluate(d)?;
            enlc = value;
            energy += value;
            for s in 0..2 {
                cadd(&mut fock[s], &potential[s], 1.0);
            }
        }
        cadd(&mut fock[0], &overlap, zeeman);
        cadd(&mut fock[1], &overlap, -zeeman);
        let spin = zeeman * (trace(&d[0], &overlap) - trace(&d[1], &overlap));
        energy += spin;
        Ok((fock, energy, exc, spin, enlc))
    };
    let (n_alpha, n_beta) = mol.n_alpha_beta()?;
    let unrestricted = match opts.reference {
        super::Reference::Auto => n_alpha != n_beta,
        super::Reference::Uhf => true,
        super::Reference::Rhf if n_alpha == n_beta => false,
        super::Reference::Rhf => {
            return Err(DftError::InvalidInput(
                "restricted finite-field SCF requires equal spin populations".into(),
            ))
        }
    };
    let n_occ = [n_alpha, n_beta];
    if !unrestricted
        && extra_core.is_some_and(|h| {
            h[0].re
                .as_slice()
                .iter()
                .zip(h[1].re.as_slice())
                .chain(h[0].im.as_slice().iter().zip(h[1].im.as_slice()))
                .any(|(a, b)| (a - b).abs() > 1e-12)
        })
    {
        return Err(DftError::InvalidInput(
            "spin-dependent extra core requires an unrestricted finite-field reference".into(),
        ));
    }
    let overlap_bloch = BlochMatrix::from_pair((overlap.re.clone(), overlap.im.clone()));
    let mut density: [Cm; 2] = if let Some(scf) = initial {
        if !scf.converged
            || [scf.n_alpha, scf.n_beta] != n_occ
            || scf.unrestricted != unrestricted
            || scf.density.iter().any(|p| {
                p.rows != nao || p.cols != nao || p.as_slice().iter().any(|x| !x.is_finite())
            })
        {
            return Err(DftError::InvalidInput(
                "finite-field seed must be a converged reference with matching spin populations and AO dimensions".into()));
        }
        std::array::from_fn(|s| Cm {
            re: scf.density[s].clone(),
            im: Matrix::zeros(nao, nao),
        })
    } else {
        let orbitals = generalized_orbitals(
            &BlochMatrix::from_pair((core.re.clone(), core.im.clone())),
            &overlap_bloch,
            opts.orth_threshold,
        )?;
        if n_alpha.max(n_beta) > orbitals.energies.len() {
            return Err(DftError::InvalidInput(
                "the basis holds fewer independent functions than occupied orbitals".into(),
            ));
        }
        std::array::from_fn(|s| occupied_density(&orbitals.coefficients, n_occ[s]))
    };
    let mut diis = Diis::new(8);
    let mut last_energy = f64::INFINITY;
    let mut residual = f64::INFINITY;
    for iteration in 1..=opts.max_iter {
        let (fock, energy, exc, spin, enlc) = build(&density)?;
        let mut errors = Vec::with_capacity(4 * nao * nao);
        residual = 0.0;
        for s in 0..2 {
            let fds = cmul(&cmul(&fock[s], &density[s]), &overlap);
            let sdf = cmul(&cmul(&overlap, &density[s]), &fock[s]);
            for (x, y) in fds
                .re
                .as_slice()
                .iter()
                .zip(sdf.re.as_slice())
                .chain(fds.im.as_slice().iter().zip(sdf.im.as_slice()))
            {
                let e = x - y;
                residual = residual.max(e.abs());
                errors.push(e);
            }
        }
        let change = (energy - last_energy).abs();
        crate::logger::log(&format!(
            "[magnetic-scf] iter {iteration:>3} E={energy:.12} dE={change:.3e} residual={residual:.3e}"
        ));
        let converged = residual < opts.conv_tol_grad && change < opts.e_tol;
        last_energy = energy;
        let used: [Cm; 2] = if converged {
            fock.clone()
        } else {
            diis.push_with_errors(
                &[
                    fock[0].re.clone(),
                    fock[0].im.clone(),
                    fock[1].re.clone(),
                    fock[1].im.clone(),
                ],
                errors,
            );
            match diis.extrapolate() {
                Some(list) if list.len() == 4 => {
                    let mut parts = list.into_iter();
                    let mut next = || parts.next().expect("four matrices");
                    [
                        Cm {
                            re: next(),
                            im: next(),
                        },
                        Cm {
                            re: next(),
                            im: next(),
                        },
                    ]
                }
                _ => fock.clone(),
            }
        };
        let mut energies: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut coefficients = [Cm::zeros(nao, 0), Cm::zeros(nao, 0)];
        for s in 0..2 {
            if s == 1 && !unrestricted {
                // The Zeeman term shifts the levels of a closed shell without
                // changing its orbitals.
                density[1] = density[0].clone();
                energies[1] = energies[0].iter().map(|e| e - 2.0 * zeeman).collect();
                coefficients[1] = coefficients[0].clone();
                continue;
            }
            let orbitals = generalized_orbitals(
                &BlochMatrix::from_pair((used[s].re.clone(), used[s].im.clone())),
                &overlap_bloch,
                opts.orth_threshold,
            )?;
            if n_occ[s] > orbitals.energies.len() {
                return Err(DftError::InvalidInput(
                    "the basis holds fewer independent functions than occupied orbitals".into(),
                ));
            }
            density[s] = occupied_density(&orbitals.coefficients, n_occ[s]);
            energies[s] = orbitals.energies;
            coefficients[s] = Cm {
                re: orbitals.coefficients.re,
                im: orbitals.coefficients.im,
            };
        }
        if converged {
            return Ok(MagneticFieldResult {
                kinetic_density_model: kinetic_model,
                energy,
                electric_field: electric.to_array(),
                zeeman_energy: spin,
                e_xc: exc,
                e_nlc: enlc,
                nonlocal_parameters: opts.vv10,
                learned_functional: opts.skala.is_some(),
                density,
                fock,
                overlap,
                core,
                orbital_energies: energies,
                coefficients,
                n_alpha,
                n_beta,
                unrestricted,
                converged: true,
                iterations: iteration,
                residual,
                series_order: order,
                series_tail,
                shift: shift.to_array(),
            });
        }
    }
    Err(DftError::ScfNotConverged {
        iterations: opts.max_iter,
        error: residual,
    })
}

type Dual3 = crate::dual::DualN<f64, 3>;
type Cd = (Dual3, Dual3);

fn dmul(a: Cd, b: Cd) -> Cd {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `(b x v) / 2` of a dual vector.
fn half_cross_dual(b: [f64; 3], v: [Dual3; 3]) -> [Dual3; 3] {
    [
        (v[2].scale(b[1]) - v[1].scale(b[2])).scale(0.5),
        (v[0].scale(b[2]) - v[2].scale(b[0])).scale(0.5),
        (v[1].scale(b[0]) - v[0].scale(b[1])).scale(0.5),
    ]
}

/// Derivative of the exchange-correlation energy of fixed complex spin
/// densities with respect to the position of every atom. The quadrature
/// points and the Becke weights follow the atoms and the London phases
/// follow the centres; the functional enters through its first derivatives.
fn xc_gradient(
    molecule: &Molecule,
    basis: &MolBasis,
    grid: &MolecularGrid,
    points: &[Vec3],
    functional: &ResolvedFunctional,
    field: [f64; 3],
    density: &[Cm; 2],
) -> Result<Vec<[f64; 3]>> {
    let natoms = molecule.len();
    let nao = basis.nao_sph;
    let gga = functional.family != Family::Lda;
    let meta = functional.family == Family::MetaGga;
    let aij = MolecularGrid::becke_aij(molecule);
    let zero = Dual3::constant(0.0);
    let mut out = Vec::with_capacity(natoms);
    for moving in 0..natoms {
        let seed = |atom: usize, p: [f64; 3]| -> [Dual3; 3] {
            std::array::from_fn(|k| {
                if atom == moving {
                    Dual3::variable(p[k], k)
                } else {
                    Dual3::constant(p[k])
                }
            })
        };
        let atoms: Vec<[Dual3; 3]> = molecule
            .atoms
            .iter()
            .enumerate()
            .map(|(atom, a)| seed(atom, a.position.to_array()))
            .collect();
        let parts: Result<Vec<[f64; 3]>> = (0..points.len())
            .into_par_iter()
            .map(|i| -> Result<[f64; 3]> {
                let owner = grid.atom_of[i];
                let r = seed(owner, points[i].to_array());
                let weight = becke_weight(&atoms, &aij, owner, r, 1)?.scale(grid.raw_weights[i]);
                // Phased functions, their gradients and (-i grad + A) on them.
                let mut w: Vec<Cd> = vec![(zero, zero); nao];
                let mut gw: Vec<[Cd; 3]> = vec![[(zero, zero); 3]; nao];
                let mut pw: Vec<[Cd; 3]> = vec![[(zero, zero); 3]; nao];
                for shell in &basis.shells {
                    let offset = shell.center - molecule.atoms[shell.atom].position;
                    let centre: [Dual3; 3] = std::array::from_fn(|k| {
                        atoms[shell.atom][k] + Dual3::constant(offset.get(k))
                    });
                    let d: [Dual3; 3] = std::array::from_fn(|k| r[k] - centre[k]);
                    let (values, slopes) = shell_values(shell, d);
                    let slope = half_cross_dual(field, centre);
                    let theta = slope[0] * r[0] + slope[1] * r[1] + slope[2] * r[2];
                    let (cs, sn) = cos_sin(theta);
                    let e: Cd = (cs, -sn);
                    let gauge = half_cross_dual(field, d);
                    for (m, (phi, g)) in values.into_iter().zip(slopes).enumerate() {
                        let mu = shell.sph_off + m;
                        w[mu] = (phi * e.0, phi * e.1);
                        for a in 0..3 {
                            gw[mu][a] = dmul((g[a], -(slope[a] * phi)), e);
                            pw[mu][a] = dmul((gauge[a] * phi, -g[a]), e);
                        }
                    }
                }
                let mut rho = [zero; 2];
                let mut grad = [[zero; 3]; 2];
                let mut tau = [zero; 2];
                for s in 0..2 {
                    for nu in 0..nao {
                        let mut y: Cd = (zero, zero);
                        let mut z: [Cd; 3] = [(zero, zero); 3];
                        for mu in 0..nao {
                            let (dr, di) = (density[s].re[(mu, nu)], density[s].im[(mu, nu)]);
                            if dr == 0.0 && di == 0.0 {
                                continue;
                            }
                            let f = w[mu];
                            y = (
                                y.0 + f.0.scale(dr) - f.1.scale(di),
                                y.1 + f.1.scale(dr) + f.0.scale(di),
                            );
                            if meta {
                                for a in 0..3 {
                                    let f = pw[mu][a];
                                    z[a] = (
                                        z[a].0 + f.0.scale(dr) - f.1.scale(di),
                                        z[a].1 + f.1.scale(dr) + f.0.scale(di),
                                    );
                                }
                            }
                        }
                        rho[s] = rho[s] + y.0 * w[nu].0 + y.1 * w[nu].1;
                        if gga {
                            for a in 0..3 {
                                grad[s][a] =
                                    grad[s][a] + (y.0 * gw[nu][a].0 + y.1 * gw[nu][a].1).scale(2.0);
                            }
                        }
                        if meta {
                            for a in 0..3 {
                                tau[s] = tau[s]
                                    + (z[a].0 * pw[nu][a].0 + z[a].1 * pw[nu][a].1).scale(0.5);
                            }
                        }
                    }
                }
                let contraction = |s: usize, t: usize| -> Dual3 {
                    grad[s][0] * grad[t][0] + grad[s][1] * grad[t][1] + grad[s][2] * grad[t][2]
                };
                let sigma = [contraction(0, 0), contraction(0, 1), contraction(1, 1)];
                // The functional and its first derivatives at the values.
                let rho_v = [vec![rho[0].re.max(0.0)], vec![rho[1].re.max(0.0)]];
                let sigma_v = [vec![sigma[0].re], vec![sigma[1].re], vec![sigma[2].re]];
                let tau_v = [vec![tau[0].re.max(0.0)], vec![tau[1].re.max(0.0)]];
                let vars = GridVars {
                    rho: [&rho_v[0], &rho_v[1]],
                    sigma: gga.then_some([&sigma_v[0], &sigma_v[1], &sigma_v[2]]),
                    tau: meta.then_some([&tau_v[0], &tau_v[1]]),
                };
                let mut xc = XcOut::default();
                functional.eval(&vars, 1, &mut xc)?;
                let mut value = [0.0; 3];
                for (a, target) in value.iter_mut().enumerate() {
                    let mut slope = 0.0;
                    for s in 0..2 {
                        slope += xc.vrho[s][0] * rho[s].im[a];
                        if meta {
                            slope += xc.vtau[s][0] * tau[s].im[a];
                        }
                    }
                    if gga {
                        for (p, term) in sigma.iter().enumerate() {
                            slope += xc.vsigma[p][0] * term.im[a];
                        }
                    }
                    *target = weight.im[a] * xc.e[0] + weight.re * slope;
                }
                Ok(value)
            })
            .collect();
        let mut total = [0.0; 3];
        for part in parts? {
            for a in 0..3 {
                total[a] += part[a];
            }
        }
        out.push(total);
    }
    Ok(out)
}

/// Gradient of the energy of the converged state `state` in the field with
/// respect to the positions of the atoms, Hartree/Bohr: the derivative of
/// the summed integral series and of the quadrature at fixed densities,
/// with the energy-weighted densities `D F D` on the overlap.
pub fn magnetic_field_gradient(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    options: &MagneticFieldOptions,
    state: &MagneticFieldResult,
) -> Result<Vec<[f64; 3]>> {
    if state.nonlocal_parameters.is_some() || state.learned_functional {
        return Err(DftError::Unsupported {
            feature: "finite-field nuclear gradient with an auxiliary global functional".into(),
            missing: "moving nuclear/grid/phase derivatives of the same nonlocal or learned field provider; an energy-only auxiliary term must not be omitted from forces".into(),
        });
    }
    let nao = mb.nao_sph;
    let natoms = mol.len();
    let shift = Vec3::new(state.shift[0], state.shift[1], state.shift[2]);
    let mut molecule = mol.clone();
    for atom in &mut molecule.atoms {
        atom.position += shift;
    }
    let basis = shifted(mb, shift);
    let field = options.field;
    let path = FieldPath {
        electric: state.electric_field,
        magnetic: field,
        origin: [0.0; 3],
        london: true,
        order: state.series_order,
    };
    let strength = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
    let zeeman = if options.spin_zeeman {
        0.25 * crate::constants::ELECTRON_G_FACTOR * strength
    } else {
        0.0
    };
    let mut total = state.density[0].clone();
    cadd(&mut total, &state.density[1], 1.0);
    // Minus the energy-weighted densities D F D = C e C^H, and the Zeeman
    // term on the overlap.
    let mut overlap_density = Cm::zeros(nao, nao);
    for s in 0..2 {
        let weighted = cmul(&cmul(&state.density[s], &state.fock[s]), &state.density[s]);
        cadd(&mut overlap_density, &weighted, -1.0);
    }
    cadd(&mut overlap_density, &state.density[0], zeeman);
    cadd(&mut overlap_density, &state.density[1], -zeeman);
    let mut gradient = one_electron_gradient(&molecule, &basis, &path, &total, &overlap_density)?;
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let spins = [&state.density[0], &state.density[1]];
    let (coulomb, exchange) =
        repulsion_gradient(&basis, &path, None, spins, natoms, true, cx != 0.0)?;
    for atom in 0..natoms {
        for a in 0..3 {
            gradient[atom][a] += coulomb[atom][a] - cx * exchange[atom][a];
        }
    }
    if clr != 0.0 {
        let (_, long_range) = repulsion_gradient(
            &basis,
            &path,
            Some(functional.hybrid.omega),
            spins,
            natoms,
            false,
            true,
        )?;
        for atom in 0..natoms {
            for a in 0..3 {
                gradient[atom][a] -= clr * long_range[atom][a];
            }
        }
    }
    // Repulsion of the nuclei.
    let charges = basis.effective_charges(&molecule);
    for a in 0..natoms {
        for k in 0..3 {
            gradient[a][k] -= charges[a] * state.electric_field[k];
        }
        for b in 0..natoms {
            if a == b {
                continue;
            }
            let d = molecule.atoms[a].position - molecule.atoms[b].position;
            let r = d.norm();
            let factor = -charges[a] * charges[b] / (r * r * r);
            for k in 0..3 {
                gradient[a][k] += factor * d.get(k);
            }
        }
    }
    if !functional.parts.is_empty() {
        let grid = grid.ok_or_else(|| {
            DftError::InvalidInput("the finite-field gradient needs the molecular grid".into())
        })?;
        let points: Vec<Vec3> = grid.points.iter().map(|p| *p + shift).collect();
        let part = xc_gradient(
            &molecule,
            &basis,
            grid,
            &points,
            functional,
            field,
            &state.density,
        )?;
        for atom in 0..natoms {
            for a in 0..3 {
                gradient[atom][a] += part[atom][a];
            }
        }
    }
    Ok(gradient)
}
