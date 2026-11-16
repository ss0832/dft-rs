// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Magnetic-field response and NMR shielding tensors.
//!
//! # The perturbation is imaginary
//!
//! Every first-order magnetic quantity here is written as `i` times a REAL
//! matrix, and the real matrices are ANTISYMMETRIC:
//!
//! ```text
//! dS/dB_a = i Shat_a,  dh/dB_a = i Hhat_a,  dP/dB_a = i D_a,  U^{B_a} = i Uhat_a
//! ```
//!
//! Two consequences drive the whole solver:
//!
//! - the first-order density change `sum_mn D_mn chi_m chi_n` VANISHES for
//!   antisymmetric `D`, and so do its gradient and kinetic-energy density.
//!   The Coulomb response `J[D]` is therefore identically zero, and so is
//!   every density/gradient semilocal XC kernel contribution. A meta-GGA is
//!   the exception: gauge-covariant `tau_tilde = tau-j_p^2/(2 rho)` leaves a
//!   current-current Hessian, so even a pure meta-GGA remains coupled;
//! - the exchange coupling becomes the `A - B` combination rather than
//!   `A + B`, and it does so BY ITSELF. Projecting the ordinary AO build
//!   `K[X]_mn = sum_ls X_ls (ml|ns)` gives `(ab|ij) + (aj|ib)` when `X` is
//!   the symmetric `C_v U C_o^T + transpose` and `(ab|ij) - (aj|ib)` when
//!   `X` is the antisymmetric one. So the response term keeps its usual
//!   form `-(cx/2) K[D]`; flipping that sign by hand — the tempting move,
//!   since "imaginary perturbations use A - B" — double-counts the change
//!   and shows up as a residual gauge-origin dependence of a few ppm,
//!   small enough to look like rounding and large enough to be wrong.
//!
//! Antisymmetric spin densities retain the magnetic `A - B` coupling.
//! The response uses the common matrix-free Krylov driver with a fresh
//! relative residual; it supports restricted and unrestricted references.
//!
//! # Units
//!
//! In Hartree atomic units the vector potential enters as
//! `(1/2)(p + alpha A)^2`, so a mixed `B`-`m` second derivative carries
//! `alpha^2`. Everything internal is the `alpha = 1` quantity; the reported
//! shielding is `alpha^2 x 1e6` times it, i.e. ppm.
//!
//! # Scope
//!
//! Collinear orbital exact exchange (global or range-separated), LDA, GGA, and
//! current-density-corrected meta-GGA without ECPs.

use crate::basis::MolBasis;
use crate::constants::FINE_STRUCTURE;
use crate::error::{DftError, Result};
use crate::grid::MolecularGrid;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{EriMode, JkEngine, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;

use super::giao_eri::{giao_eri_fock_derivative, giao_eri_fock_derivative_weighted};
use super::giao_xc::{giao_density_xc_fock_derivative, MetaCurrentKernel, WeightedDensityPartials};
use super::nmr::{
    angular_momentum_operator, giao_core_hamiltonian_derivative, giao_overlap_derivative_about,
    paramagnetic_operator, shielding_mixed_operator,
};

mod reference;
pub(crate) use reference::spinor_orbital_field_sources;
pub use reference::{common_orbital_nmr_response, spinor_nmr_response, ReferenceShielding};

/// How the magnetic vector potential's gauge origin is handled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gauge {
    /// London (gauge-including) atomic orbitals: each basis function
    /// carries its own field-dependent phase, and the result does not
    /// depend on the origin.
    #[default]
    Giao,
    /// One common origin for every basis function. Origin-DEPENDENT, and
    /// kept because in a one-centre system with the origin on the atom it
    /// is identical to the GIAO answer by construction — which pins the two
    /// paths against each other.
    CommonOrigin,
}

impl Gauge {
    pub fn from_name(name: &str) -> Result<Self> {
        match name
            .trim()
            .to_ascii_lowercase()
            .replace(['-', '_'], "")
            .as_str()
        {
            "giao" | "london" => Ok(Self::Giao),
            "common" | "commonorigin" => Ok(Self::CommonOrigin),
            _ => Err(DftError::InvalidInput(
                "magnetic gauge must be giao or common_origin".into(),
            )),
        }
    }
}

/// The magnetic-field density response.
pub struct MagneticResponse {
    /// `dP/dB_a = i D[a]`; real, antisymmetric.
    pub d: [Matrix; 3],
    /// `dS/dB_a = i S[a]`; real, antisymmetric (zero for a common origin).
    pub s: [Matrix; 3],
    /// Fresh relative residual actually reached, per component, maximizing spins.
    pub residual: [f64; 3],
}

/// Solve the magnetic-field CPHF and return the first-order density.
///
/// The equations, with `Uhat` the real rotation amplitudes and
/// `Fhat = Hhat + Ghat - (cx/2) K[D]`:
///
/// ```text
/// (eps_a - eps_i) Uhat_ai = eps_i Shat^MO_ai - Fhat^MO_ai
/// Uhat_ij = -(1/2) Shat^MO_ij      (occupied-occupied, from C^dag S C = 1)
/// D = 2 (X - X^T) - 2 C_o Shat^oo C_o^T,   X = C_v Uhat C_o^T
/// ```
pub fn magnetic_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
) -> Result<MagneticResponse> {
    magnetic_response_with_grid(mol, mb, functional, None, scf, opts, gauge, origin)
}

/// Magnetic response with the molecular XC grid used by the converged SCF.
/// The grid is required for GIAO semilocal functionals and for meta-GGA in
/// either gauge (its current-current response is a grid contraction).
#[allow(clippy::too_many_arguments)]
pub fn magnetic_response_with_grid(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
) -> Result<MagneticResponse> {
    let report = magnetic_density_response(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        gauge,
        origin,
        &crate::response::ResponseSolverConfig::default(),
    )?;
    let value = report.value;
    let mut d = value.density_hat[0].clone();
    for axis in 0..3 {
        add_scaled(&mut d[axis], &value.density_hat[1][axis], 1.0);
    }
    let residual =
        std::array::from_fn(|axis| report.residuals[axis].max(report.residuals[3 + axis]));
    Ok(MagneticResponse {
        d,
        s: value.overlap_hat,
        residual,
    })
}

/// Spin-resolved orbital density derivatives: `dP_s/dB_a = i D_s,a`.
/// Fields use the reduced atomic unit entering `(p+A)^2/2`, with
/// `A = B x (r-origin)/2`. Spin Zeeman and hyperfine operators are separate.
pub struct SpinMagneticResponse {
    pub density_hat: [[Matrix; 3]; 2],
    pub overlap_hat: [Matrix; 3],
    /// Explicit Fock derivative at fixed reference density, with `i` removed.
    pub explicit_fock_hat: [[Matrix; 3]; 2],
    /// Occupied-virtual property gradient `Fhat_ai - e_i Shat_ai` of the
    /// response equations per spin and field axis, index `a * n_occ + i`,
    /// with `i` removed; `Fhat` holds the exchange of the occupied-occupied
    /// overlap response.
    pub gradient_hat: [[Vec<f64>; 3]; 2],
    /// Occupied-virtual rotation amplitudes `Uhat_ai` per spin and field
    /// axis, index `a * n_occ + i` (the occupied-occupied block is
    /// `-Shat_ij / 2`).
    pub amplitudes_hat: [[Vec<f64>; 3]; 2],
    pub projector_defect: f64,
    pub particle_number_defect: f64,
    pub antisymmetry_defect: f64,
    pub gauge: Gauge,
    pub origin: Vec3,
    pub solver_methods: Vec<&'static str>,
    pub occupied: [usize; 2],
}

fn add_scaled(target: &mut Matrix, source: &Matrix, scale: f64) {
    for (a, &b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += scale * b;
    }
}

/// Explicit London magnetic sources shared by collinear and constrained
/// common-orbital response. The current-density kernel is returned alongside
/// the basis derivative so that both contributions use the same convention.
fn collinear_magnetic_sources(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
) -> Result<(
    [Matrix; 3],
    [[Matrix; 3]; 2],
    Option<MetaCurrentKernel>,
    bool,
)> {
    check_supported(mb, functional, grid, scf, opts, gauge)?;
    let nao = mb.nao_sph;
    let zeros = || std::array::from_fn(|_| Matrix::zeros(nao, nao));
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let omega = functional.hybrid.omega;
    let learned = opts
        .skala
        .as_ref()
        .map(|model| {
            WeightedDensityPartials::learned(
                model,
                mol,
                mb,
                grid.expect("checked grid"),
                [&scf.density[0], &scf.density[1]],
            )
        })
        .transpose()?;
    let mut meta_current = if functional.family == crate::xc::Family::MetaGga {
        MetaCurrentKernel::build(
            functional,
            mb,
            grid.expect("checked grid"),
            [&scf.density[0], &scf.density[1]],
        )?
    } else {
        None
    };
    if let Some(fields) = &learned {
        let extra = MetaCurrentKernel::from_partials(fields);
        if let Some(current) = &mut meta_current {
            current.add(&extra);
        } else {
            meta_current = Some(extra);
        }
    }
    let (s_hat, h_hat) = match gauge {
        Gauge::Giao => (
            giao_overlap_derivative_about(mb, origin),
            giao_core_hamiltonian_derivative(mol, mb, origin)?,
        ),
        Gauge::CommonOrigin => {
            let mut h = angular_momentum_operator(mb, origin);
            for a in &mut h {
                for v in a.as_mut_slice() {
                    *v *= -0.5;
                }
            }
            (zeros(), h)
        }
    };
    let mut explicit: [[Matrix; 3]; 2] = [h_hat.clone(), h_hat];
    if gauge == Gauge::Giao {
        let total = scf.p_total();
        if !scf.unrestricted {
            let mut common = giao_eri_fock_derivative(mb, &total, cx, origin, opts.kernel);
            if clr != 0.0 {
                let lr = giao_eri_fock_derivative_weighted(
                    mb,
                    &total,
                    0.0,
                    clr,
                    origin,
                    crate::integrals::CoulombKernel::Erf { omega },
                );
                for a in 0..3 {
                    add_scaled(&mut common[a], &lr[a], 1.0);
                }
            }
            for spin in 0..2 {
                for a in 0..3 {
                    add_scaled(&mut explicit[spin][a], &common[a], 1.0);
                }
            }
        } else {
            let j = giao_eri_fock_derivative_weighted(mb, &total, 1.0, 0.0, origin, opts.kernel);
            for spin in 0..2 {
                let k = giao_eri_fock_derivative_weighted(
                    mb,
                    &scf.density[spin],
                    0.0,
                    2.0 * cx,
                    origin,
                    opts.kernel,
                );
                let lr = if clr != 0.0 {
                    Some(giao_eri_fock_derivative_weighted(
                        mb,
                        &scf.density[spin],
                        0.0,
                        2.0 * clr,
                        origin,
                        crate::integrals::CoulombKernel::Erf { omega },
                    ))
                } else {
                    None
                };
                for a in 0..3 {
                    add_scaled(&mut explicit[spin][a], &j[a], 1.0);
                    add_scaled(&mut explicit[spin][a], &k[a], 1.0);
                    if let Some(lr) = &lr {
                        add_scaled(&mut explicit[spin][a], &lr[a], 1.0);
                    }
                }
            }
        }
        if !functional.parts.is_empty() || learned.is_some() {
            let xc = giao_density_xc_fock_derivative(
                functional,
                mb,
                grid.expect("checked grid"),
                [&scf.density[0], &scf.density[1]],
                origin,
                learned.as_ref(),
            )?;
            for spin in 0..2 {
                for a in 0..3 {
                    add_scaled(&mut explicit[spin][a], &xc[spin][a], 1.0);
                    if let Some(current) = &meta_current {
                        let v = current.basis_fock(
                            mb,
                            grid.expect("checked grid"),
                            &scf.density[spin],
                            origin,
                            a,
                            spin,
                        )?;
                        add_scaled(&mut explicit[spin][a], &v, 1.0);
                    }
                }
            }
        }
    }
    Ok((s_hat, explicit, meta_current, learned.is_some()))
}

/// Matrix-free collinear orbital magnetic CPKS for restricted or unrestricted
/// references. Physical energy gaps remain unchanged. Hermitian response is
/// solved by MINRES, allowing an invertible indefinite Hessian; the response
/// of an unstable stationary reference is a formal derivative, not a minimum.
#[allow(clippy::too_many_arguments)]
pub fn magnetic_density_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
    config: &crate::response::ResponseSolverConfig,
) -> Result<crate::response::ResponseReport<SpinMagneticResponse>> {
    use crate::response::complex::Complex;
    use crate::response::krylov::{self, LinearOperator, OperatorStructure};
    config.validate()?;
    if !scf.converged
        || [origin.x, origin.y, origin.z]
            .iter()
            .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "magnetic response requires a converged reference and finite gauge origin".into(),
        ));
    }
    check_supported(mb, functional, grid, scf, opts, gauge)?;
    let reference = crate::response::diagnostics::reference_diagnostics(scf)?;
    if mb.nao_sph != scf.overlap.rows {
        return Err(DftError::InvalidInput(
            "magnetic basis and reference AO dimensions disagree".into(),
        ));
    }
    let cx = functional.hybrid.cx;
    let clr = functional.hybrid.clr;
    let omega = functional.hybrid.omega;
    let nao = mb.nao_sph;
    let zeros = || std::array::from_fn(|_| Matrix::zeros(nao, nao));
    let (s_hat, explicit, meta_current, learned) =
        collinear_magnetic_sources(mol, mb, functional, grid, scf, opts, gauge, origin)?;
    // Magnetic response uses K only. Avoid constructing RI-J state and do
    // not allocate a full-range tensor at all for LC-style (cx=0) RSHs.
    // When both kernels are requested in-core, count their aggregate n^4
    // storage against the one user limit; if only one tensor fits, keep the
    // full-range path in-core and make the LR path direct.
    // Resolve the same memory-aware integral cache policy as SCF before
    // choosing the aggregate full/long-range storage below. Setup options
    // may still say Direct even when the converged reference used a cache.
    let (std_mode, lr_mode) = match opts.effective_eri_mode(nao) {
        EriMode::InCore { limit_mb } if cx != 0.0 || clr != 0.0 => {
            let n = mb.nao_sph;
            let bytes = n
                .checked_mul(n)
                .and_then(|v| v.checked_mul(n))
                .and_then(|v| v.checked_mul(n))
                .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
                .ok_or(DftError::ResourceLimit {
                    operation: "magnetic-response in-core ERI tensor",
                    required_mb: usize::MAX,
                    limit_mb,
                })?;
            let mib = 1024 * 1024;
            let one_mb = bytes.checked_add(mib - 1).ok_or(DftError::ResourceLimit {
                operation: "magnetic-response in-core ERI tensor",
                required_mb: usize::MAX,
                limit_mb,
            })? / mib;
            if one_mb > limit_mb {
                return Err(DftError::ResourceLimit {
                    operation: "magnetic-response in-core ERI tensor",
                    required_mb: one_mb,
                    limit_mb,
                });
            }
            let two_fit = one_mb
                .checked_mul(2)
                .is_some_and(|required| required <= limit_mb);
            let lr_mode = if cx == 0.0 || two_fit {
                EriMode::InCore { limit_mb: one_mb }
            } else {
                EriMode::Direct {
                    screen_threshold: 1e-12,
                }
            };
            (EriMode::InCore { limit_mb: one_mb }, lr_mode)
        }
        mode => (mode, mode),
    };
    // The physical kernel must be linear even for a nearly forbidden source.
    // Reuse the reference-fixed integral mask: screening with the trial
    // response density changes the operator when Krylov vectors are rescaled.
    let reference_total = scf.p_total();
    let reference_spin = [&scf.density[0], &scf.density[1]];
    let jk = if cx != 0.0 {
        Some(
            JkEngine::new(mb, std_mode, opts.kernel)?
                .with_reference_screening(&reference_total, reference_spin),
        )
    } else {
        None
    };
    let jk_lr = if clr != 0.0 {
        Some(
            JkEngine::new(mb, lr_mode, crate::integrals::CoulombKernel::Erf { omega })?
                .with_reference_screening(&reference_total, reference_spin),
        )
    } else {
        None
    };
    let k_std = |d: &Matrix| -> Result<Matrix> {
        let Some(engine) = &jk else {
            return Ok(Matrix::zeros(nao, nao));
        };
        Ok(engine
            .build_k_only(mb, [d, d])?
            .into_iter()
            .next()
            .expect("K block"))
    };
    let k_lr = |d: &Matrix| -> Result<Matrix> {
        let Some(engine) = &jk_lr else {
            return Ok(Matrix::zeros(nao, nao));
        };
        Ok(engine
            .build_k_only(mb, [d, d])?
            .into_iter()
            .next()
            .expect("long-range K block"))
    };

    let occupied = [scf.n_alpha, scf.n_beta];
    let mut density_hat = [zeros(), zeros()];
    let mut gradient_hat: [[Vec<f64>; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut amplitudes_hat: [[Vec<f64>; 3]; 2] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut residuals = Vec::with_capacity(6);
    let mut iterations = Vec::with_capacity(6);
    let mut methods = Vec::with_capacity(6);
    // K acts within each spin. Density/gradient kernels vanish for real
    // current-free collinear references; the spin-resolved current kernel remains.
    for spin in 0..2 {
        if spin == 1 && !scf.unrestricted {
            density_hat[1] = density_hat[0].clone();
            gradient_hat[1] = gradient_hat[0].clone();
            amplitudes_hat[1] = amplitudes_hat[0].clone();
            for a in 0..3 {
                residuals.push(residuals[a]);
                iterations.push(iterations[a]);
                methods.push(methods[a]);
            }
            continue;
        }
        let c = &scf.mo_coeff[spin];
        let no = occupied[spin];
        let nv = c.cols - no;
        let eps = &scf.mo_energy[spin];
        let len = no * nv;
        let current_fock = |d: &Matrix| -> Result<Matrix> {
            let mut v = k_std(d)?;
            for x in v.as_mut_slice() {
                *x *= -cx;
            }
            if clr != 0.0 {
                add_scaled(&mut v, &k_lr(d)?, -clr);
            }
            if let Some(current) = &meta_current {
                add_scaled(
                    &mut v,
                    &current.response_fock(mb, grid.expect("checked grid"), d, spin)?,
                    1.0,
                );
            }
            Ok(v)
        };
        let apply_real = |u: &[f64]| -> Result<Vec<f64>> {
            let d = spin_virtual_density(c, u, no);
            let f = current_fock(&d)?;
            let mo = c.transpose_matmul_seq(&f.matmul_seq(c));
            Ok((0..len)
                .map(|k| {
                    let a = k / no;
                    let i = k % no;
                    (eps[no + a] - eps[i]) * u[k] + mo[(no + a, i)]
                })
                .collect())
        };
        struct MagneticOperator<'a, F> {
            apply_real: &'a F,
            len: usize,
        }
        impl<F: Fn(&[f64]) -> Result<Vec<f64>>> LinearOperator for MagneticOperator<'_, F> {
            fn dimension(&self) -> usize {
                self.len
            }
            fn structure(&self) -> OperatorStructure {
                OperatorStructure::Hermitian
            }
            fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
                krylov::apply_real_linear(x, self.apply_real)
            }
        }
        let op = MagneticOperator {
            apply_real: &apply_real,
            len,
        };
        let mut prepared = Vec::with_capacity(3);
        for axis in 0..3 {
            let sm = c.transpose_matmul_seq(&s_hat[axis].matmul_seq(c));
            let mut d_occ = Matrix::zeros(nao, nao);
            for m in 0..nao {
                for n in 0..nao {
                    d_occ[(m, n)] = -(0..no)
                        .flat_map(|j| (0..no).map(move |i| (j, i)))
                        .map(|(j, i)| c[(m, j)] * sm[(j, i)] * c[(n, i)])
                        .sum::<f64>();
                }
            }
            let mut f0 = explicit[spin][axis].clone();
            add_scaled(&mut f0, &current_fock(&d_occ)?, 1.0);
            let fm = c.transpose_matmul_seq(&f0.matmul_seq(c));
            let rhs = (0..len)
                .map(|k| {
                    let a = k / no;
                    let i = k % no;
                    Complex::from(eps[i] * sm[(no + a, i)] - fm[(no + a, i)])
                })
                .collect::<Vec<_>>();
            prepared.push((d_occ, rhs));
        }
        for (axis, (d_occ, rhs)) in prepared.into_iter().enumerate() {
            gradient_hat[spin][axis] = rhs.iter().map(|z| -z.re).collect();
            let solution = krylov::solve(&op, &rhs, None, config)?;
            let u = solution.amplitudes.iter().map(|z| z.re).collect::<Vec<_>>();
            let mut d = spin_virtual_density(c, &u, no);
            add_scaled(&mut d, &d_occ, 1.0);
            density_hat[spin][axis] = d;
            amplitudes_hat[spin][axis] = u;
            residuals.push(solution.relative_residual);
            iterations.push(solution.iterations);
            methods.push(solution.method);
        }
    }
    let mut projector_defect = 0.0_f64;
    let mut particle_number_defect = 0.0_f64;
    let mut antisymmetry_defect = 0.0_f64;
    for spin in 0..2 {
        for a in 0..3 {
            let d = &density_hat[spin][a];
            let p = &scf.density[spin];
            let mut defect = d.matmul_seq(&scf.overlap).matmul_seq(p);
            add_scaled(&mut defect, &p.matmul_seq(&scf.overlap).matmul_seq(d), 1.0);
            add_scaled(&mut defect, &p.matmul_seq(&s_hat[a]).matmul_seq(p), 1.0);
            add_scaled(&mut defect, d, -1.0);
            projector_defect = projector_defect.max(
                defect
                    .as_slice()
                    .iter()
                    .fold(0.0_f64, |m, v| m.max(v.abs())),
            );
            particle_number_defect = particle_number_defect
                .max((trace_product(d, &scf.overlap) + trace_product(p, &s_hat[a])).abs());
            for m in 0..nao {
                for n in 0..nao {
                    antisymmetry_defect = antisymmetry_defect
                        .max((d[(m, n)] + d[(n, m)]).abs())
                        .max((s_hat[a][(m, n)] + s_hat[a][(n, m)]).abs())
                        .max((explicit[spin][a][(m, n)] + explicit[spin][a][(n, m)]).abs());
                }
            }
        }
    }
    let mut warnings=vec!["Fixed spin occupations and orbital minimal coupling; spin Zeeman, hyperfine and Curie contributions are separate.".into(),
        "No stability assumption is made by MINRES; use stability analysis to classify the stationary reference.".into()];
    if learned {
        warnings.push("The learned kinetic-density input uses tau-j_p^2/(2 rho); the response is adiabatic, orbital and collinear.".into());
    }
    if gauge == Gauge::CommonOrigin {
        warnings.push(
            "A common-origin finite basis has gauge-origin dependent orbital response.".into(),
        );
    }
    Ok(crate::response::ResponseReport {
        value:SpinMagneticResponse {density_hat,overlap_hat:s_hat,explicit_fock_hat:explicit,gradient_hat,amplitudes_hat,
            projector_defect,particle_number_defect,antisymmetry_defect,gauge,origin,
            solver_methods:methods,occupied},
        shape:vec![2,3,nao,nao], functional:functional.description.clone(), reference,
        units:"inverse reduced atomic magnetic field",
        model:"Zero-field collinear orbital magnetic CPKS; gauge-covariant kinetic-density functional",
        axis_convention:"spin, external B (x,y,z), spherical AO row, spherical AO column; derivative = i times density_hat",
        converged:residuals.iter().all(|&r|r<=config.tolerance),residuals,iterations,warnings,
    })
}

fn spin_virtual_density(c: &Matrix, u: &[f64], no: usize) -> Matrix {
    let mut x = Matrix::zeros(c.rows, c.rows);
    for a in no..c.cols {
        for i in 0..no {
            let v = u[(a - no) * no + i];
            for m in 0..c.rows {
                for n in 0..c.rows {
                    x[(m, n)] += v * c[(m, a)] * c[(n, i)];
                }
            }
        }
    }
    let mut d = Matrix::zeros(c.rows, c.rows);
    for m in 0..c.rows {
        for n in 0..c.rows {
            d[(m, n)] = x[(m, n)] - x[(n, m)];
        }
    }
    d
}

/// `Tr[A B]` for square matrices — not the Frobenius dot, and the two
/// differ by a sign whenever both are antisymmetric, which is exactly the
/// case here.
fn trace_product(a: &Matrix, b: &Matrix) -> f64 {
    let mut s = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            s += a[(i, j)] * b[(j, i)];
        }
    }
    s
}

/// One nucleus's shielding tensor.
#[derive(Clone, Copy, Debug)]
pub struct Shielding {
    /// `sigma_ij = d^2 E / dm_i dB_j` in ppm.
    pub tensor: [[f64; 3]; 3],
    /// Non-response ("diamagnetic") half, ppm.
    pub diamagnetic: [[f64; 3]; 3],
    /// Response ("paramagnetic") half, ppm.
    pub paramagnetic: [[f64; 3]; 3],
}

impl Shielding {
    /// Isotropic shielding (ppm).
    pub fn isotropic(&self) -> f64 {
        (self.tensor[0][0] + self.tensor[1][1] + self.tensor[2][2]) / 3.0
    }

    /// Isotropic diamagnetic and paramagnetic parts (ppm).
    pub fn isotropic_parts(&self) -> (f64, f64) {
        let tr = |t: &[[f64; 3]; 3]| (t[0][0] + t[1][1] + t[2][2]) / 3.0;
        (tr(&self.diamagnetic), tr(&self.paramagnetic))
    }
}

/// Shielding tensors of every nucleus, in ppm.
#[allow(clippy::too_many_arguments)]
pub fn shielding_tensors(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
) -> Result<Vec<Shielding>> {
    shielding_tensors_with_grid(mol, mb, functional, None, scf, opts, gauge, origin)
}

/// Shielding tensors using the same molecular grid as a semilocal SCF.
#[allow(clippy::too_many_arguments)]
pub fn shielding_tensors_with_grid(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
) -> Result<Vec<Shielding>> {
    Ok(molecular_nmr_response(
        mol,
        mb,
        functional,
        grid,
        scf,
        opts,
        gauge,
        origin,
        &crate::response::ResponseSolverConfig::default(),
    )?
    .value)
}

/// Orbital NMR shielding in ppm, including unrestricted spin densities.
/// The report retains magnetic and reference residuals. Paramagnetic shifts
/// from spin Zeeman/hyperfine/temperature are separate observables.
#[allow(clippy::too_many_arguments)]
pub fn molecular_nmr_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
    config: &crate::response::ResponseSolverConfig,
) -> Result<crate::response::ResponseReport<Vec<Shielding>>> {
    let response =
        magnetic_density_response(mol, mb, functional, grid, scf, opts, gauge, origin, config)?;
    let mut total = response.value.density_hat[0].clone();
    for a in 0..3 {
        add_scaled(&mut total[a], &response.value.density_hat[1][a], 1.0);
    }
    let p_tot = scf.p_total();
    let giao = gauge == Gauge::Giao;
    let scale = FINE_STRUCTURE * FINE_STRUCTURE * 1e6;
    let mut out = Vec::with_capacity(mol.len());
    for atom in &mol.atoms {
        let mixed = shielding_mixed_operator(mb, atom.position, origin, giao);
        let pso = paramagnetic_operator(mb, atom.position);
        let mut dia = [[0.0; 3]; 3];
        let mut para = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                // sigma_ij = d^2E/dm_i dB_j, so the B index is the column.
                dia[b][a] = scale * trace_product(&p_tot, &mixed[a][b]);
                para[b][a] = scale * trace_product(&total[a], &pso[b]);
            }
        }
        let mut tensor = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                tensor[i][j] = dia[i][j] + para[i][j];
            }
        }
        out.push(Shielding {
            tensor,
            diamagnetic: dia,
            paramagnetic: para,
        });
    }
    Ok(crate::response::ResponseReport {
        value: out,
        shape: vec![mol.len(), 3, 3],
        functional: response.functional,
        reference: response.reference,
        units: "ppm",
        model: match gauge {
            Gauge::Giao => "Zero-field orbital NMR shielding; London CPKS; fixed collinear spin occupations",
            Gauge::CommonOrigin => "Zero-field orbital NMR shielding; common-origin CPKS; fixed collinear spin occupations",
        },
        axis_convention: "nucleus, nuclear magnetic moment (x,y,z), external B (x,y,z)",
        converged: response.converged,
        residuals: response.residuals,
        iterations: response.iterations,
        warnings: response.warnings,
    })
}

/// Contributions of orthonormal occupied orbitals to the NMR shielding.
#[derive(Clone, Debug)]
pub struct OrbitalShielding {
    /// The shielding of every nucleus.
    pub totals: Vec<Shielding>,
    /// `[spin][nucleus][orbital]` diamagnetic and paramagnetic tensors, ppm,
    /// with the index convention of [`Shielding`].
    pub diamagnetic: [Vec<Vec<[[f64; 3]; 3]>>; 2],
    pub paramagnetic: [Vec<Vec<[[f64; 3]; 3]>>; 2],
    /// Largest `|sum over spins and orbitals - total|` (ppm).
    pub decomposition_residual: f64,
    /// Largest `|T^T T - I|` of the occupied rotations.
    pub rotation_residual: f64,
}

/// Natural chemical shielding analysis on any orthonormal occupied set:
/// `orbitals[s]` (AO coefficients) span the occupied space of spin `s`. The
/// rotation `T = C_o^T S L` carries the first-order canonical orbitals
/// `C_v Uhat - C_o Shat^oo / 2` to the set, and orbital `k` contributes
/// `Tr[l_k l_k^T h_mixed]` to the diamagnetic and
/// `Tr[(u_k l_k^T - l_k u_k^T) h_pso]` to the paramagnetic shielding.
#[allow(clippy::too_many_arguments)]
pub fn orbital_shielding(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
    origin: Vec3,
    config: &crate::response::ResponseSolverConfig,
    orbitals: [&Matrix; 2],
) -> Result<OrbitalShielding> {
    let response =
        magnetic_density_response(mol, mb, functional, grid, scf, opts, gauge, origin, config)?;
    let totals =
        molecular_nmr_response(mol, mb, functional, grid, scf, opts, gauge, origin, config)?.value;
    let r = &response.value;
    let nao = mb.nao_sph;
    let giao = gauge == Gauge::Giao;
    let scale = FINE_STRUCTURE * FINE_STRUCTURE * 1e6;
    let operators: Vec<([[Matrix; 3]; 3], [Matrix; 3])> = mol
        .atoms
        .iter()
        .map(|atom| {
            (
                shielding_mixed_operator(mb, atom.position, origin, giao),
                paramagnetic_operator(mb, atom.position),
            )
        })
        .collect();
    let mut diamagnetic: [Vec<Vec<[[f64; 3]; 3]>>; 2] = [Vec::new(), Vec::new()];
    let mut paramagnetic: [Vec<Vec<[[f64; 3]; 3]>>; 2] = [Vec::new(), Vec::new()];
    let mut rotation_residual = 0.0_f64;
    for spin in 0..2 {
        let c = &scf.mo_coeff[spin];
        let no = r.occupied[spin];
        let nv = c.cols - no;
        let l = orbitals[spin];
        if l.rows != nao || l.cols != no {
            return Err(DftError::InvalidInput(format!(
                "the shielding decomposition needs {no} occupied orbitals of spin {spin}, not {}",
                l.cols
            )));
        }
        let mut co = Matrix::zeros(nao, no);
        let mut cv = Matrix::zeros(nao, nv);
        for m in 0..nao {
            for i in 0..no {
                co[(m, i)] = c[(m, i)];
            }
            for a in 0..nv {
                cv[(m, a)] = c[(m, no + a)];
            }
        }
        let t = co.transpose_matmul(&scf.overlap.matmul(l));
        let defect = t.transpose_matmul(&t).max_abs_diff(&Matrix::identity(no));
        rotation_residual = rotation_residual.max(defect);
        if defect > 1e-6 {
            return Err(DftError::InvalidInput(format!(
                "the orbitals of the shielding decomposition do not span the occupied space \
                 of spin {spin} (|T^T T - I| = {defect:e})"
            )));
        }
        // First-order orbitals of the set per axis, AO x orbital.
        let first: [Matrix; 3] = std::array::from_fn(|axis| {
            let u = &r.amplitudes_hat[spin][axis];
            let mut amplitudes = Matrix::zeros(nv, no);
            for a in 0..nv {
                for i in 0..no {
                    amplitudes[(a, i)] = u[a * no + i];
                }
            }
            let s_oo = co.transpose_matmul(&r.overlap_hat[axis].matmul(&co));
            let mut canonical = cv.matmul(&amplitudes);
            let occupied = co.matmul(&s_oo);
            add_scaled(&mut canonical, &occupied, -0.5);
            canonical.matmul(&t)
        });
        let mut dia_spin = Vec::with_capacity(mol.len());
        let mut para_spin = Vec::with_capacity(mol.len());
        for (mixed, pso) in &operators {
            let mut dia_atom = Vec::with_capacity(no);
            let mut para_atom = Vec::with_capacity(no);
            for k in 0..no {
                let mut lk = Matrix::zeros(nao, 1);
                for m in 0..nao {
                    lk[(m, 0)] = l[(m, k)];
                }
                let density = lk.matmul_transpose_seq(&lk);
                let mut dia = [[0.0; 3]; 3];
                let mut para = [[0.0; 3]; 3];
                for a in 0..3 {
                    let mut uk = Matrix::zeros(nao, 1);
                    for m in 0..nao {
                        uk[(m, 0)] = first[a][(m, k)];
                    }
                    let mut d = uk.matmul_transpose_seq(&lk);
                    let back = lk.matmul_transpose_seq(&uk);
                    add_scaled(&mut d, &back, -1.0);
                    for b in 0..3 {
                        dia[b][a] = scale * trace_product(&density, &mixed[a][b]);
                        para[b][a] = scale * trace_product(&d, &pso[b]);
                    }
                }
                dia_atom.push(dia);
                para_atom.push(para);
            }
            dia_spin.push(dia_atom);
            para_spin.push(para_atom);
        }
        diamagnetic[spin] = dia_spin;
        paramagnetic[spin] = para_spin;
    }
    let mut decomposition_residual = 0.0_f64;
    for (n, total) in totals.iter().enumerate() {
        for i in 0..3 {
            for j in 0..3 {
                let dia: f64 = (0..2)
                    .map(|s| diamagnetic[s][n].iter().map(|t| t[i][j]).sum::<f64>())
                    .sum();
                let para: f64 = (0..2)
                    .map(|s| paramagnetic[s][n].iter().map(|t| t[i][j]).sum::<f64>())
                    .sum();
                decomposition_residual = decomposition_residual
                    .max((dia - total.diamagnetic[i][j]).abs())
                    .max((para - total.paramagnetic[i][j]).abs());
            }
        }
    }
    Ok(OrbitalShielding {
        totals,
        diamagnetic,
        paramagnetic,
        decomposition_residual,
        rotation_residual,
    })
}

fn check_supported(
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    _scf: &ScfResult,
    opts: &ScfOptions,
    gauge: Gauge,
) -> Result<()> {
    if !mb.ecps.is_empty() {
        return Err(DftError::Unsupported {
            feature: "magnetic properties with ECP basis sets".into(),
            missing:
                "the magnetic and mixed field-nuclear derivatives of the semilocal core potential"
                    .into(),
        });
    }
    if opts.pcm.is_some() || opts.extra_hcore.is_some() || opts.density_fit.is_some() {
        return Err(DftError::Unsupported {
            feature: "magnetic response of an externally modified or fitted reference".into(),
            missing: "consistent London-phase derivatives of the embedding, reaction field or fitted Coulomb operator".into(),
        });
    }
    // A uniform electric field couples through the B-independent dipole
    // operator, so in the common-origin gauge — where the basis carries no
    // London phases — the existing kinetic, Coulomb and semilocal derivative
    // machinery is complete for a field-polarized reference. In the GIAO
    // gauge the field operator acquires London phases whose B-derivatives
    // are not implemented, so that combination stays refused.
    if opts.external_field.is_some() && gauge == Gauge::Giao {
        return Err(DftError::Unsupported {
            feature: "GIAO magnetic response in a finite electric field".into(),
            missing: "the London-phase derivative of the electric-field operator".into(),
        });
    }
    let needs_grid = (gauge == Gauge::Giao && !functional.parts.is_empty())
        || functional.family == crate::xc::Family::MetaGga
        || opts.skala.is_some();
    if needs_grid && grid.is_none() {
        return Err(DftError::InvalidInput(
            "semilocal GIAO and meta-GGA current response require the molecular grid used by the SCF; \
             call magnetic_response_with_grid/shielding_tensors_with_grid"
                .into(),
        ));
    }
    if let Some(term) = functional.required_nonlocal_term() {
        return Err(DftError::Unsupported {
            feature: format!("GIAO magnetic properties with nonlocal {term}"),
            missing: "the London-phase derivative of the nonlocal potential".into(),
        });
    }
    if opts.vv10.is_some() {
        return Err(DftError::Unsupported {
            feature: "GIAO magnetic properties with explicit SC-VV10".into(),
            missing: "the London-phase derivative of the self-consistent nonlocal potential".into(),
        });
    }
    Ok(())
}
