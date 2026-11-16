// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Analytic field response by order-by-order density-projector equations.
//! In the orthonormal ground-state MO basis, D^2 = D fixes the occupied and
//! virtual diagonal blocks at each order; [F,D] = 0 determines the remaining
//! occupied/virtual block with the same orbital Hessian as linear CPKS.
//! Coefficients are factorial-normalized, including nonlinear XC sources.

use super::{ResponseReport, ResponseSolver, ResponseSolverConfig};
use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos_full, MolecularGrid};
use crate::integrals::dipole_matrices;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{Family, ResolvedFunctional, XcTaylorInput};
use crate::{DftError, Result};

#[derive(Clone, Debug)]
pub struct ElectricFieldTaylor {
    /// E(t) = E_reference + t direction, in atomic units.
    pub direction: [f64; 3],
    /// mu(t) = sum_n dipole_coefficients[n] t^n, in atomic units.
    pub dipole_coefficients: Vec<[f64; 3]>,
    /// Per-spin AO density coefficients, starting at the reference density.
    pub density_coefficients: Vec<[Matrix; 2]>,
    /// Fock coefficients in the fixed orthonormal reference MO frame.
    pub mo_fock_coefficients: Vec<[Matrix; 2]>,
    /// Idempotency defects in the orthonormal reference MO space at each order.
    pub projector_defects: Vec<f64>,
    /// Fixed-spectrum constraints, when prescribed populations are selected.
    /// The raw projector defects then measure non-idempotency, not convergence.
    pub fixed_population: Option<super::ensemble::FixedPopulationDiagnostics>,
    /// Canonical Fermi law defects and chemical-potential coefficients.
    pub thermal_population: Option<super::ensemble::FermiPathDiagnostics>,
    /// Full MO commutator coefficient defects, in Hartree.
    pub stationarity_defects: Vec<f64>,
    pub particle_number_derivatives: Vec<[f64; 2]>,
}

#[derive(Clone, Debug)]
pub struct MixedElectricFieldResponse {
    pub directions: Vec<[f64; 3]>,
    /// The mixed derivative itself (no factorial normalization), a dipole vector.
    pub dipole_derivative: [f64; 3],
    pub accumulation_roundoff: [f64; 3],
    pub directional_evaluations: usize,
    pub fixed_population_defect: Option<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn mixed_electric_field_response(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    directions: &[[f64; 3]],
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<MixedElectricFieldResponse>> {
    config.validate()?;
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut warnings = Vec::new();
    let mut converged = true;
    let mut fixed_population_defect: Option<f64> = None;
    let vectors: Vec<_> = directions.iter().map(|v| v.to_vec()).collect();
    let mixed = super::mixed::polarized_mixed(3, &vectors, |d, n| {
        let report = electric_field_taylor(
            mol,
            mb,
            functional,
            grid,
            scf,
            opts,
            [d[0], d[1], d[2]],
            n,
            config,
        )?;
        converged &= report.converged;
        if let Some(d) = &report.value.fixed_population {
            fixed_population_defect = Some(
                fixed_population_defect
                    .unwrap_or(0.0)
                    .max(d.maximum_defect()),
            );
        }
        residuals.extend(report.residuals);
        iterations.extend(report.iterations);
        warnings.extend(report.warnings);
        Ok(report.value.dipole_coefficients[n].to_vec())
    })?;
    for i in 0..3 {
        if mixed.accumulation_roundoff[i] > config.tolerance * mixed.value[i].abs().max(1.0) {
            warnings.push(format!(
                "mixed derivative component {i} has polarization roundoff estimate {:.3e}",
                mixed.accumulation_roundoff[i]
            ));
        }
    }
    Ok(ResponseReport {
        value: MixedElectricFieldResponse {
            directions: directions.to_vec(),
            dipole_derivative: std::array::from_fn(|i| mixed.value[i]),
            accumulation_roundoff: std::array::from_fn(|i| mixed.accumulation_roundoff[i]),
            directional_evaluations: mixed.directional_evaluations,
            fixed_population_defect,
        },
        shape: vec![3],
        functional: functional.description.clone(),
        reference: if scf.fixed_occupation_residual.is_some() || scf.electronic_temperature > 0.0 {
            super::diagnostics::density_diagnostics(scf)?
        } else {
            super::diagnostics::reference_diagnostics(scf)?
        },
        units: "atomic units; unnormalized mixed derivative",
        model: if scf.electronic_temperature > 0.0 {
            "static canonical Fermi-Dirac Mermin response; analytic polarization identity"
        } else if scf.fixed_occupation_residual.is_some() {
            "static prescribed-population KS/HF isospectral response; analytic polarization identity"
        } else {
            "static integer-occupation KS/HF projector response; analytic polarization identity"
        },
        axis_convention:
            "Cartesian dipole vector; one directional field derivative per input direction",
        converged,
        residuals,
        iterations,
        warnings,
    })
}

fn add(target: &mut Matrix, source: &Matrix, weight: f64) {
    for (a, b) in target.as_mut_slice().iter_mut().zip(source.as_slice()) {
        *a += weight * b;
    }
}
fn to_ao(c: &Matrix, d: &Matrix) -> Matrix {
    c.matmul_seq(d).matmul_seq(&c.transpose())
}
fn to_mo(c: &Matrix, f: &Matrix) -> Matrix {
    c.transpose().matmul_seq(f).matmul_seq(c)
}

/// Integer-occupation collinear molecular response for semilocal,
/// hybrid/range-separated, self-consistent VV10 and linear PCM models.
#[allow(clippy::too_many_arguments)]
pub fn electric_field_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: [f64; 3],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<ElectricFieldTaylor>> {
    config.validate()?;
    if order == usize::MAX || direction.iter().any(|x| !x.is_finite()) {
        return Err(DftError::InvalidInput(
            "invalid electric-field Taylor request".into(),
        ));
    }
    if !scf.converged {
        return Err(DftError::InvalidInput(
            "field response requires a converged SCF reference".into(),
        ));
    }
    if functional.required_nonlocal_kind() == Some(crate::xc::RequiredNonlocalTerm::Vv10)
        && opts.vv10.is_none()
    {
        return Err(DftError::InvalidInput(
            "the functional requires self-consistent VV10 options for field response".into(),
        ));
    }
    if scf.fixed_occupation_residual.is_some() {
        return fixed_population_field_taylor(
            mol, mb, functional, grid, scf, opts, direction, order, config,
        );
    }
    if scf.electronic_temperature > 0.0 {
        let path = super::nuclear::MolecularDirection {
            nuclear: vec![[0.0; 3]; mol.len()],
            electric: direction,
        };
        let report =
            super::nuclear::molecular_taylor(mol, mb, functional, grid, scf, opts, &path, order, config)?;
        let value = report.value;
        return Ok(ResponseReport {
            value: ElectricFieldTaylor {
                direction,
                dipole_coefficients: value.dipole_coefficients,
                density_coefficients: value.density_coefficients,
                mo_fock_coefficients: value.orthonormal_fock_coefficients,
                projector_defects: value.projector_defects,
                fixed_population: None,
                thermal_population: value.thermal_population,
                stationarity_defects: value.stationarity_defects,
                particle_number_derivatives: value.particle_number_coefficients,
            },
            shape: vec![order + 1, 3],
            units: "atomic units; coefficient n = directional dipole derivative n / n!",
            axis_convention: "[Taylor order][dipole Cartesian component]; fixed-temperature electric response",
            functional: report.functional,
            reference: report.reference,
            model: report.model,
            converged: report.converged,
            residuals: report.residuals,
            iterations: report.iterations,
            warnings: report.warnings,
        });
    }
    let solver = ResponseSolver::new(mol, mb, functional, grid, scf, opts)?;
    let dips = dipole_matrices(mb, Vec3::zero());
    let sources = std::array::from_fn::<_, 3, _>(|a| (&dips[a], direction[a]));
    let path = one_body_taylor(&solver, &sources, order, config)?;
    let mu0 = crate::properties::dipole_moment(mol, mb, scf);
    let mut dipole_coefficients = vec![[mu0.x, mu0.y, mu0.z]];
    dipole_coefficients.extend(path.density_coefficients.iter().skip(1).map(|density| {
        std::array::from_fn(|a| {
            -density[0].frobenius_dot(&dips[a]) - density[1].frobenius_dot(&dips[a])
        })
    }));
    Ok(ResponseReport {
        value: ElectricFieldTaylor {
            direction,
            dipole_coefficients,
            density_coefficients: path.density_coefficients,
            mo_fock_coefficients: path.mo_fock_coefficients,
            projector_defects: path.projector_defects,
            fixed_population: None,
            thermal_population: None,
            stationarity_defects: path.stationarity_defects,
            particle_number_derivatives: path.particle_number_derivatives,
        },
        shape: vec![order + 1, 3],
        functional: functional.description.clone(),
        reference: super::diagnostics::reference_diagnostics(scf)?,
        units: "atomic units; coefficient n = derivative n / n!",
        model: "static integer-occupation KS/HF density-projector response; adiabatic semilocal XC",
        axis_convention: "[Taylor order][dipole Cartesian component]; E(t)=E_reference+t direction",
        converged: true,
        residuals: path.residuals,
        iterations: path.iterations,
        warnings: Vec::new(),
    })
}

pub(crate) struct OneBodyTaylor {
    pub density_coefficients: Vec<[Matrix; 2]>,
    pub mo_fock_coefficients: Vec<[Matrix; 2]>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub particle_number_derivatives: Vec<[f64; 2]>,
    pub residuals: Vec<f64>,
    pub iterations: Vec<usize>,
}

/// Fixed-geometry integer-occupation path with a linear, spin-independent
/// one-electron AO perturbation given as a weighted sum.
pub(crate) fn one_body_taylor(
    solver: &ResponseSolver<'_>,
    sources: &[(&Matrix, f64)],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<OneBodyTaylor> {
    let scf = solver.scf;
    let mut densities = vec![scf.density.clone()];
    let mut ds: Vec<[Matrix; 2]> = vec![std::array::from_fn(|s| {
        let mut d = Matrix::zeros(scf.mo_coeff[s].cols, scf.mo_coeff[s].cols);
        for i in 0..solver.dims.n_occ[s] {
            d[(i, i)] = 1.0;
        }
        d
    })];
    let mut fs: Vec<[Matrix; 2]> = vec![std::array::from_fn(|s| {
        let mut f = Matrix::zeros(scf.mo_coeff[s].cols, scf.mo_coeff[s].cols);
        for (i, &e) in scf.mo_energy[s].iter().enumerate() {
            f[(i, i)] = e;
        }
        f
    })];
    let mut residuals = Vec::new();
    let mut iterations = Vec::new();
    let mut defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    let mut particle_number_derivatives = Vec::new();
    for n in 1..=order {
        // Idempotency fixes D_n,oo = -R_oo and D_n,vv = R_vv.
        let mut dn: [Matrix; 2] =
            std::array::from_fn(|s| Matrix::zeros(ds[0][s].rows, ds[0][s].cols));
        let mut products = dn.clone();
        for s in 0..2 {
            for m in 1..n {
                add(&mut products[s], &ds[m][s].matmul_seq(&ds[n - m][s]), 1.0);
            }
            let no = solver.dims.n_occ[s];
            for i in 0..dn[s].rows {
                for j in 0..dn[s].cols {
                    if i < no && j < no {
                        dn[s][(i, j)] = -products[s][(i, j)];
                    }
                    if i >= no && j >= no {
                        dn[s][(i, j)] = products[s][(i, j)];
                    }
                }
            }
        }
        let diag_ao: [Matrix; 2] = std::array::from_fn(|s| to_ao(&scf.mo_coeff[s], &dn[s]));
        densities.push(diag_ao.clone());
        let fn_ao = one_body_fock_coefficient(solver, &densities, sources, n)?;
        let mut fn_mo: [Matrix; 2] = std::array::from_fn(|s| to_mo(&scf.mo_coeff[s], &fn_ao[s]));
        let mut commutators: [Matrix; 2] =
            std::array::from_fn(|s| Matrix::zeros(dn[s].rows, dn[s].cols));
        for s in 0..2 {
            for m in 1..n {
                add(
                    &mut commutators[s],
                    &fs[m][s].matmul_seq(&ds[n - m][s]),
                    1.0,
                );
                add(
                    &mut commutators[s],
                    &ds[n - m][s].matmul_seq(&fs[m][s]),
                    -1.0,
                );
            }
        }
        let mut source = Vec::new();
        for s in 0..solver.spins() {
            let no = solver.dims.n_occ[s];
            for a in 0..solver.dims.n_vir[s] {
                for i in 0..no {
                    source.push(fn_mo[s][(no + a, i)] + commutators[s][(no + a, i)]);
                }
            }
        }
        let report = solver.solve_ov_report(&source, config)?;
        let u: Vec<_> = report.amplitudes.iter().map(|u| u.re).collect();
        let off_ao = solver.delta_density(&u);
        let df_ao = solver.fock_response(&off_ao)?;
        let mut offset = 0;
        for s in 0..solver.spins() {
            let no = solver.dims.n_occ[s];
            for a in 0..solver.dims.n_vir[s] {
                for i in 0..no {
                    dn[s][(no + a, i)] = u[offset];
                    dn[s][(i, no + a)] = u[offset];
                    offset += 1;
                }
            }
        }
        if solver.spins() == 1 {
            dn[1] = dn[0].clone();
        }
        for s in 0..2 {
            add(&mut fn_mo[s], &to_mo(&scf.mo_coeff[s], &df_ao[s]), 1.0);
            add(&mut densities[n][s], &off_ao[s], 1.0);
        }
        let mut defect: f64 = 0.0;
        for s in 0..2 {
            let mut r = products[s].clone();
            add(&mut r, &ds[0][s].matmul_seq(&dn[s]), 1.0);
            add(&mut r, &dn[s].matmul_seq(&ds[0][s]), 1.0);
            add(&mut r, &dn[s], -1.0);
            defect = defect.max(r.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs())));
        }
        ds.push(dn);
        fs.push(fn_mo);
        let mut stationary_defect = 0.0_f64;
        for s in 0..2 {
            let mut comm = commutators[s].clone();
            for (f, d, sign) in [
                (&fs[0][s], &ds[n][s], 1.0),
                (&fs[n][s], &ds[0][s], 1.0),
                (&ds[n][s], &fs[0][s], -1.0),
                (&ds[0][s], &fs[n][s], -1.0),
            ] {
                add(&mut comm, &f.matmul_seq(d), sign);
            }
            stationary_defect =
                stationary_defect.max(comm.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs())));
        }
        stationarity_defects.push(stationary_defect);
        particle_number_derivatives.push(std::array::from_fn(|s| {
            densities[n][s].frobenius_dot(&scf.overlap)
        }));
        defects.push(defect);
        residuals.push(report.relative_residual);
        iterations.push(report.iterations);
    }
    Ok(OneBodyTaylor {
        density_coefficients: densities,
        mo_fock_coefficients: fs,
        projector_defects: defects,
        stationarity_defects,
        particle_number_derivatives,
        residuals,
        iterations,
    })
}

pub(super) fn field_fock_coefficient(
    solver: &ResponseSolver<'_>,
    densities: &[[Matrix; 2]],
    dipoles: &[Matrix; 3],
    direction: [f64; 3],
    n: usize,
) -> Result<[Matrix; 2]> {
    let sources = std::array::from_fn::<_, 3, _>(|a| (&dipoles[a], direction[a]));
    one_body_fock_coefficient(solver, densities, &sources, n)
}

/// One complete coefficient provider for both integer and fixed spectra.
/// Linear terms keep their common kernel; nonlinear XC replaces exactly its
/// first-order contribution rather than being added twice.
fn one_body_fock_coefficient(
    solver: &ResponseSolver<'_>,
    densities: &[[Matrix; 2]],
    sources: &[(&Matrix, f64)],
    n: usize,
) -> Result<[Matrix; 2]> {
    if n == 0 {
        return Ok(solver.scf.fock.clone());
    }
    let dp = &densities[n];
    let mut f = solver.fock_response(dp)?;
    if n > 1 {
        if let Some((params, grid, p0)) = &solver.nlc {
            let mut total = dp[0].clone();
            add(&mut total, &dp[1], 1.0);
            let linear = crate::nlc::vv10_fxc_contraction(solver.mb, grid, p0, &total, *params)?;
            let totals: Vec<_> = densities
                .iter()
                .map(|p| {
                    let mut t = p[0].clone();
                    add(&mut t, &p[1], 1.0);
                    t
                })
                .collect();
            let nonlinear = super::nonlocal::vv10_taylor(solver.mb, grid, &totals, *params)?;
            for s in 0..2 {
                add(&mut f[s], &linear, -1.0);
                add(&mut f[s], &nonlinear.fock_coefficients[n], 1.0);
            }
        }
        if !solver.functional.parts.is_empty() {
            let linear = solver.fxc_contraction_semilocal(dp)?;
            let nonlinear = semilocal_fock_coefficient(
                solver.mb,
                solver.grid.expect("response grid checked"),
                solver.functional,
                densities,
                n,
            )?;
            for s in 0..2 {
                add(&mut f[s], &linear[s], -1.0);
                add(&mut f[s], &nonlinear[s], 1.0);
            }
        }
        if let Some((model, molecule)) = &solver.learned {
            let grid = solver.grid.expect("response grid checked");
            let linear = crate::skala::skala_fxc_contraction(
                model,
                molecule,
                solver.mb,
                grid,
                [&solver.scf.density[0], &solver.scf.density[1]],
                [&dp[0], &dp[1]],
            )?;
            let nonlinear = super::nuclear::learned_fock_coefficient(
                model, molecule, solver.mb, grid, densities, n,
            )?;
            for s in 0..2 {
                add(&mut f[s], &linear[s], -1.0);
                add(&mut f[s], &nonlinear[s], 1.0);
            }
        }
    }
    if n == 1 {
        for spin in &mut f {
            for &(operator, weight) in sources {
                add(spin, operator, weight);
            }
        }
    }
    Ok(f)
}

#[allow(clippy::too_many_arguments)]
fn fixed_population_field_taylor(
    mol: &Molecule,
    mb: &MolBasis,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    scf: &ScfResult,
    opts: &ScfOptions,
    direction: [f64; 3],
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<ResponseReport<ElectricFieldTaylor>> {
    let solver = super::ensemble::EnsembleSolver::new(mol, mb, functional, grid, scf, opts)?;
    let dipoles = dipole_matrices(mb, Vec3::zero());
    let mut b: [Vec<Matrix>; 2] =
        std::array::from_fn(|s| vec![Matrix::zeros(mb.nao_sph, scf.mo_coeff[s].cols); order + 1]);
    for s in 0..2 {
        b[s][0] = scf.mo_coeff[s].clone();
    }
    let mut overlap = vec![Matrix::zeros(mb.nao_sph, mb.nao_sph); order + 1];
    overlap[0] = scf.overlap.clone();
    let path =
        super::ensemble::fixed_population_path(&solver, &b, &overlap, order, config, |p, n| {
            field_fock_coefficient(&solver.kernel, p, &dipoles, direction, n)
        })?;
    let mut dipole_coefficients = Vec::with_capacity(order + 1);
    let charges = mb.effective_charges(mol);
    let nuclear = mol
        .atoms
        .iter()
        .zip(charges)
        .fold(Vec3::zero(), |a, (atom, z)| a + atom.position * z);
    for (n, p) in path.density.iter().enumerate() {
        dipole_coefficients.push(std::array::from_fn(|axis| {
            (if n == 0 {
                nuclear.to_array()[axis]
            } else {
                0.0
            }) - p
                .iter()
                .map(|d| d.frobenius_dot(&dipoles[axis]))
                .sum::<f64>()
        }));
    }
    let reference = super::diagnostics::density_diagnostics(scf)?;
    let d = &path.diagnostics;
    let converged = d.reference_population_defect <= config.acceptance()
        && d.reference_density_defect <= config.acceptance()
        && reference.orthonormality_defect <= config.acceptance()
        && reference.eigen_residual <= config.acceptance()
        && d.population_defects
            .iter()
            .chain(&d.orthogonality_defects)
            .chain(&d.relative_stationarity_defects)
            .all(|v| v.is_finite() && *v <= config.acceptance())
        && path
            .particle_numbers
            .iter()
            .flatten()
            .all(|v| v.is_finite() && v.abs() <= config.acceptance());
    let warnings = if converged {
        Vec::new()
    } else {
        vec!["fixed-population reference or Taylor constraint diagnostics exceed the requested tolerance; inspect the raw reference, population, metric and stationarity measurements".into()]
    };
    Ok(ResponseReport {
        value:ElectricFieldTaylor {direction,dipole_coefficients,density_coefficients:path.density,
            mo_fock_coefficients:path.mo_fock,projector_defects:path.projector_defects,
            fixed_population:Some(path.diagnostics),thermal_population:None,stationarity_defects:path.stationarity_defects,
            particle_number_derivatives:path.particle_numbers},
        shape:vec![order+1,3],functional:functional.description.clone(),reference,
        units:"atomic units; coefficient n = derivative n / n!",
        model:"static prescribed-population KS/HF; analytic isospectral orbital rotations; adiabatic XC and connected nonlocal/linear-continuum kernels",
        axis_convention:"[Taylor order][dipole Cartesian component]; E(t)=E_reference+t direction; every spin-orbital population is held fixed",
        converged,residuals:path.residuals,iterations:path.iterations,warnings,
    })
}

/// Exact Taylor coefficient of the semilocal Fock matrix along an AO
/// density path. Gradients enter sigma by convolution; tau remains linear
/// in the density; the chain rule is retained in the AO gradient weights.
pub fn semilocal_fock_coefficient(
    mb: &MolBasis,
    grid: &MolecularGrid,
    functional: &ResolvedFunctional,
    densities: &[[Matrix; 2]],
    order: usize,
) -> Result<[Matrix; 2]> {
    let nao = mb.nao_sph;
    if densities.len() <= order
        || densities
            .iter()
            .take(order + 1)
            .flatten()
            .any(|d| d.rows != nao || d.cols != nao || d.as_slice().iter().any(|x| !x.is_finite()))
    {
        return Err(DftError::InvalidInput(
            "invalid semilocal density Taylor path".into(),
        ));
    }
    let gga = functional.family != Family::Lda;
    let mgga = functional.family == Family::MetaGga;
    let width = order + 1;
    let mut fock = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for start in (0..grid.len()).step_by(128) {
        let end = (start + 128).min(grid.len());
        let batch = eval_aos_full(mb, &grid.points[start..end], gga, false);
        let grads = batch.grad.as_ref();
        for point in 0..end - start {
            let mut variables: [Vec<f64>; 7] = std::array::from_fn(|_| vec![0.0; width]);
            let mut density_grads: [[Vec<f64>; 3]; 2] =
                std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; width]));
            for n in 0..width {
                for s in 0..2 {
                    for mu in 0..nao {
                        for nu in 0..nao {
                            let d = densities[n][s][(mu, nu)];
                            variables[s][n] += d * batch.phi[(point, mu)] * batch.phi[(point, nu)];
                            if let Some(g) = grads {
                                for a in 0..3 {
                                    density_grads[s][a][n] +=
                                        2.0 * d * g[a][(point, mu)] * batch.phi[(point, nu)];
                                    if mgga {
                                        variables[5 + s][n] +=
                                            0.5 * d * g[a][(point, mu)] * g[a][(point, nu)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for s in 0..2 {
                variables[s][0] = variables[s][0].max(0.0);
                if mgga {
                    variables[5 + s][0] = variables[5 + s][0].max(0.0);
                }
            }
            if gga {
                for (p, (s, t)) in [(0, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    for n in 0..width {
                        for k in 0..=n {
                            for a in 0..3 {
                                variables[2 + p][n] +=
                                    density_grads[s][a][k] * density_grads[t][a][n - k];
                            }
                        }
                    }
                }
            }
            let out = functional.semilocal_taylor_potentials(&XcTaylorInput {
                variables,
                parameters: None,
            })?;
            let w = grid.weights[start + point];
            for s in 0..2 {
                let gradient_weight: [f64; 3] = std::array::from_fn(|a| {
                    (0..=order)
                        .map(|k| {
                            2.0 * out.potentials[if s == 0 { 2 } else { 4 }][k]
                                * density_grads[s][a][order - k]
                                + out.potentials[3][k] * density_grads[1 - s][a][order - k]
                        })
                        .sum()
                });
                for mu in 0..nao {
                    for nu in 0..nao {
                        let mut v = out.potentials[s][order]
                            * batch.phi[(point, mu)]
                            * batch.phi[(point, nu)];
                        if let Some(g) = grads {
                            for a in 0..3 {
                                v += gradient_weight[a]
                                    * (g[a][(point, mu)] * batch.phi[(point, nu)]
                                        + batch.phi[(point, mu)] * g[a][(point, nu)]);
                                if mgga {
                                    v += 0.5
                                        * out.potentials[5 + s][order]
                                        * g[a][(point, mu)]
                                        * g[a][(point, nu)];
                                }
                            }
                        }
                        fock[s][(mu, nu)] += w * v;
                    }
                }
            }
        }
    }
    Ok(fock)
}
