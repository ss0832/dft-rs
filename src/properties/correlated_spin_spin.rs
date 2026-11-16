// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Complete post-SCF closed-shell FC/SD/PSO/DSO energy derivatives.
//! Spin rotation invariance of a spin-free closed shell makes its three
//! electronic spin axes equivalent. Each is evaluated in the independent
//! alpha/beta longitudinal response space; open shells need an additional
//! transverse spinor provider and are not identified with that limit.
use crate::basis::MolBasis;
use crate::constants::{ELECTRON_G_FACTOR, FINE_STRUCTURE};
use crate::grid::{eval::eval_aos, GridOptions, MolecularGrid};
use crate::integrals::london::{ComplexMatrix, FieldPath};
use crate::linalg::Matrix;
use crate::response::correlated_field::{correlated_field_taylor_for_model, ENERGY_CHANNELS};
use crate::response::field_path::FieldEnergyModel;
use crate::response::spin_spin::{
    diamagnetic_spin_orbit_kernel, nuclear_magnetic_sites, SpinSpinCouplings,
};
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::{double_hybrid::Pt2Coefficients, ResolvedFunctional};
use crate::{DftError, Result};

/// Every channel includes the same four physical coupling contributions.
/// Unscaled SS/OS tensors allow independent correlation-coefficient checks.
pub struct CorrelatedSpinSpin {
    pub response: ResponseReport<SpinSpinCouplings>,
    pub parent: SpinSpinCouplings,
    pub target: SpinSpinCouplings,
    pub correlation: SpinSpinCouplings,
    pub same_spin_alpha: SpinSpinCouplings,
    pub opposite_spin: SpinSpinCouplings,
    pub same_spin_beta: SpinSpinCouplings,
    pub parent_nonlocal: SpinSpinCouplings,
    pub target_nonlocal: SpinSpinCouplings,
    /// Raw PT2 coefficient-equation residuals, Hartree in the scaled source coordinates.
    pub amplitude_residuals: Vec<f64>,
    /// Raw dimensionless orthonormal-frame and projector residuals.
    pub path_residuals: Vec<f64>,
    /// Raw stationarity, reference and imaginary-energy defects, Hartree.
    pub stationarity_defects: Vec<f64>,
}

/// Full energy Hessian with respect to nuclear magnetic moments in a fixed
/// real AO basis, including the relaxed target and PT2 expectation of DSO.
/// Nuclear-source coordinates are divided by alpha^2 for conditioning;
/// the exact alpha^4 factor is restored on every reported coupling.
#[allow(clippy::too_many_arguments)]
pub fn correlated_spin_spin_couplings(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<&ResolvedFunctional>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    nuclei: &[usize],
    contact_only: bool,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedSpinSpin> {
    correlated_spin_spin_couplings_for_model(
        molecule,
        basis,
        parent,
        target.map(FieldEnergyModel::new),
        grid,
        reference,
        options,
        coefficients,
        nuclei,
        contact_only,
        memory_mb,
        config,
    )
}

/// Explicitly composed target model on the same parent and response equations.
#[allow(clippy::too_many_arguments)]
pub fn correlated_spin_spin_couplings_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    nuclei: &[usize],
    contact_only: bool,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedSpinSpin> {
    config.validate()?;
    if reference.unrestricted || reference.n_alpha != reference.n_beta {
        return Err(DftError::Unsupported {
            feature: "post-SCF open-shell nuclear spin-spin coupling".into(),
            missing: "the transverse spinor pair response in addition to the longitudinal channels"
                .into(),
        });
    }
    let sites = nuclear_magnetic_sites(molecule, basis, nuclei, contact_only)?;
    let nao = basis.nao_sph;
    let alpha2 = FINE_STRUCTURE.powi(2);
    let alpha4 = alpha2 * alpha2;
    let blank = || {
        [
            ComplexMatrix::zeros(nao, nao),
            ComplexMatrix::zeros(nao, nao),
        ]
    };
    let mut response_residuals = Vec::new();
    let mut amplitude_residuals = Vec::new();
    let mut path_residuals = Vec::new();
    let mut stationarity_defects = Vec::new();
    let mut evaluate =
        |core: Vec<[ComplexMatrix; 2]>, order: usize| -> Result<[f64; ENERGY_CHANNELS]> {
            let value = correlated_field_taylor_for_model(
                molecule,
                basis,
                parent,
                target.clone(),
                grid,
                reference,
                options,
                coefficients,
                &FieldPath {
                    electric: [0.0; 3],
                    magnetic: [0.0; 3],
                    origin: [0.0; 3],
                    london: false,
                    order,
                },
                Some(&core),
                memory_mb,
                config,
            )?;
            response_residuals.extend(&value.parent.parent.residuals);
            amplitude_residuals.extend(&value.amplitude_residuals);
            path_residuals.push(value.parent.orbital_frame_defect);
            path_residuals.extend(&value.parent.parent.projector_defects);
            stationarity_defects.extend(&value.parent.parent.stationarity_defects);
            stationarity_defects.push(value.parent.parent.reference_defect);
            stationarity_defects.push(value.imaginary_energy_defect);
            Ok(value.energy_channels(order))
        };
    let spin_source = |operator: &Matrix, scale: f64| {
        let mut h = blank();
        for s in 0..2 {
            for (v, a) in h[s].re.as_mut_slice().iter_mut().zip(operator.as_slice()) {
                *v = if s == 0 {
                    scale * a * 0.5
                } else {
                    -scale * a * 0.5
                };
            }
        }
        h
    };
    let orbital_source = |operator: &Matrix| {
        let mut h = blank();
        for spin in &mut h {
            for (v, a) in spin.im.as_mut_slice().iter_mut().zip(operator.as_slice()) {
                *v = -*a;
            }
        }
        h
    };
    let mut mixed =
        |left: &[ComplexMatrix; 2], right: &[ComplexMatrix; 2]| -> Result<[f64; ENERGY_CHANNELS]> {
            let mut values = Vec::new();
            for sign in [1.0, -1.0] {
                let mut source = left.clone();
                for s in 0..2 {
                    for (v, a) in source[s]
                        .re
                        .as_mut_slice()
                        .iter_mut()
                        .zip(right[s].re.as_slice())
                    {
                        *v += sign * a;
                    }
                    for (v, a) in source[s]
                        .im
                        .as_mut_slice()
                        .iter_mut()
                        .zip(right[s].im.as_slice())
                    {
                        *v += sign * a;
                    }
                }
                values.push(evaluate(vec![blank(), source, blank()], 2)?);
            }
            Ok(std::array::from_fn(|c| {
                0.5 * alpha4 * (values[0][c] - values[1][c])
            }))
        };
    let built;
    let quadrature = match grid {
        Some(g) => g,
        None => {
            built = MolecularGrid::build(molecule, &GridOptions::default())?;
            &built
        }
    };
    let empty_couplings = || SpinSpinCouplings {
        pairs: Vec::new(),
        diamagnetic: Vec::new(),
        paramagnetic: Vec::new(),
        fermi_contact: Vec::new(),
        spin_dipole: Vec::new(),
        total: Vec::new(),
        residual: 0.0,
    };
    let mut all: [SpinSpinCouplings; ENERGY_CHANNELS] = std::array::from_fn(|_| empty_couplings());
    // Keep response and expectation paths in separate phases so both share
    // the same evaluator and its diagnostics without interior mutability.
    for m in 0..sites.len() {
        for n in m + 1..sites.len() {
            let contact_scale = 0.5 * ELECTRON_G_FACTOR * 8.0 * std::f64::consts::PI / 3.0;
            let fc = mixed(
                &spin_source(&sites[m].contact, contact_scale),
                &spin_source(&sites[n].contact, contact_scale),
            )?;
            let mut para = [[[0.0; 3]; 3]; ENERGY_CHANNELS];
            let mut dipolar = para;
            let fermi: [[[f64; 3]; 3]; ENERGY_CHANNELS] = std::array::from_fn(|c| {
                std::array::from_fn(|a| std::array::from_fn(|b| if a == b { fc[c] } else { 0.0 }))
            });
            if !contact_only {
                for a in 0..3 {
                    for b in 0..3 {
                        let pso = mixed(
                            &orbital_source(&sites[m].orbital[a]),
                            &orbital_source(&sites[n].orbital[b]),
                        )?;
                        for c in 0..ENERGY_CHANNELS {
                            para[c][a][b] = pso[c];
                            dipolar[c][a][b] = -fermi[c][a][b];
                        }
                        for k in 0..3 {
                            let sd = mixed(
                                &spin_source(&sites[m].triplet[3 * k + a], 1.0 / alpha2),
                                &spin_source(&sites[n].triplet[3 * k + b], 1.0 / alpha2),
                            )?;
                            for c in 0..ENERGY_CHANNELS {
                                dipolar[c][a][b] += sd[c];
                            }
                        }
                    }
                }
            }
            for c in 0..ENERGY_CHANNELS {
                all[c].pairs.push((sites[m].atom, sites[n].atom));
                all[c].diamagnetic.push([[0.0; 3]; 3]);
                all[c].paramagnetic.push(para[c]);
                all[c].fermi_contact.push(fermi[c]);
                all[c].spin_dipole.push(dipolar[c]);
                all[c].total.push([[0.0; 3]; 3]);
            }
        }
    }
    if !contact_only {
        for pair in 0..all[0].pairs.len() {
            let (m, n) = all[0].pairs[pair];
            let mut operators: Vec<Matrix> = (0..9).map(|_| Matrix::zeros(nao, nao)).collect();
            for (start, points) in quadrature.points.chunks(256).enumerate() {
                let batch = eval_aos(basis, points, false);
                for (i, point) in points.iter().enumerate() {
                    let Some(kernel) = diamagnetic_spin_orbit_kernel(
                        point.to_array(),
                        molecule.atoms[m].position.to_array(),
                        molecule.atoms[n].position.to_array(),
                    ) else {
                        continue;
                    };
                    let weight = quadrature.weights[start * 256 + i];
                    for a in 0..3 {
                        for b in 0..3 {
                            for mu in 0..nao {
                                for nu in 0..nao {
                                    operators[3 * a + b][(mu, nu)] += weight
                                        * kernel[a][b]
                                        * batch.phi[(i, mu)]
                                        * batch.phi[(i, nu)];
                                }
                            }
                        }
                    }
                }
            }
            for a in 0..3 {
                for b in 0..3 {
                    let h = ComplexMatrix {
                        re: operators[3 * a + b].clone(),
                        im: Matrix::zeros(nao, nao),
                    };
                    let value = evaluate(vec![blank(), [h.clone(), h]], 1)?;
                    for c in 0..ENERGY_CHANNELS {
                        all[c].diamagnetic[pair][a][b] = alpha4 * value[c];
                    }
                }
            }
        }
    }
    let residual = response_residuals
        .iter()
        .chain(&amplitude_residuals)
        .chain(&path_residuals)
        .chain(&stationarity_defects)
        .copied()
        .fold(0.0_f64, f64::max);
    for channel in &mut all {
        channel.residual = residual;
        for p in 0..channel.pairs.len() {
            for a in 0..3 {
                for b in 0..3 {
                    channel.total[p][a][b] = channel.diamagnetic[p][a][b]
                        + channel.paramagnetic[p][a][b]
                        + channel.fermi_contact[p][a][b]
                        + channel.spin_dipole[p][a][b];
                }
            }
        }
    }
    let [total, parent_value, target_value, correlation, same_spin_alpha, opposite_spin, same_spin_beta, parent_nonlocal, target_nonlocal] =
        all;
    let shape = vec![total.pairs.len(), 3, 3];
    Ok(CorrelatedSpinSpin {
        response: ResponseReport {
            value: total, shape, units: "atomic units", axis_convention: "requested nucleus pairs, first nuclear moment (x,y,z), second nuclear moment (x,y,z)",
            functional: target.as_ref().map_or(parent,FieldEnergyModel::functional).description.clone(), reference: crate::response::diagnostics::reference_diagnostics(reference)?,
            model: "Post-SCF closed-shell FC/SD/PSO/DSO nuclear-moment energy Hessian; full target, orbital, noncanonical SS/OS amplitude and relaxed DSO response; current-invariant kinetic density; spin-free closed-shell isotropic spin response",
            converged: response_residuals.iter().chain(&amplitude_residuals).chain(&path_residuals).chain(&stationarity_defects).all(|r| r.is_finite() && r.abs() <= config.acceptance()),
            residuals: response_residuals, iterations: Vec::new(), warnings: Vec::new(),
        },
        parent: parent_value, target: target_value, correlation, same_spin_alpha, opposite_spin, same_spin_beta,parent_nonlocal,target_nonlocal, amplitude_residuals, path_residuals, stationarity_defects,
    })
}
