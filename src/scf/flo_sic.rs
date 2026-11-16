// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Fixed-descriptor Fermi–Löwdin SIC on a frozen real UKS determinant.

#[path = "flo_sic/complex.rs"]
mod complex;
#[path = "flo_sic/orbitals.rs"]
mod orbitals;
#[path = "flo_sic/stationary.rs"]
mod stationary;

use super::{MolecularDensityModel, SelfEnergy};
use crate::basis::MolBasis;
use crate::grid::{eval::eval_aos, MolecularGrid};
use crate::integrals::CoulombKernel;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::scf::{run_scf, JkEngine, Reference, ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::ResolvedFunctional;
use crate::{DftError, Result};

pub use complex::{run_complex_descriptors, ComplexDescriptor, ComplexFloRun};
pub use orbitals::{fermi_lowdin_orbitals, FloOrbitals};

/// Fixed-descriptor evaluation on a converged parent UKS determinant.
pub struct FixedDescriptorSic {
    /// Frozen parent energy plus the orbital self-interaction correction, Hartree.
    pub energy: f64,
    pub parent_energy: f64,
    pub correction: f64,
    /// Unnegated self-interaction removed from each occupied Fermi–Löwdin orbital.
    pub terms: [Vec<SelfEnergy>; 2],
    pub orbitals: [FloOrbitals; 2],
    /// Largest absolute change of either AO spin-density element after localization.
    pub density_residual: f64,
    /// Largest absolute occupied–virtual parent Fock matrix element, Hartree.
    pub parent_orbital_residual: f64,
    pub parent_iterations: usize,
}

/// Initialize the integer UKS reference and SIC contractions while retaining the
/// same orbital-representation, full- and long-range exchange and XC providers.
pub(super) fn with_uks_sic_reference<T>(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    method: &str,
    evaluate: impl FnOnce(&MolecularDensityModel<'_>, &ScfResult) -> Result<T>,
) -> Result<T> {
    if options.occupation.is_some()
        || options.fixed_occupations.is_some()
        || options.electronic_temperature != 0.0
        || options.electron_offset != [0.0; 2]
        || options.asymptotic_correction.is_some()
        || options.skala.is_some()
        || options.pcm.is_some()
        || options.density_fit.is_some()
    {
        return Err(DftError::Unsupported {
            feature: format!("{method} with the requested density model or occupations"),
            missing:
                "an integer gas-phase UKS determinant with exact Coulomb and exchange contractions"
                    .into(),
        });
    }
    let mut options = options.clone();
    options.reference = Reference::Uhf;
    options.eri_mode = options.effective_eri_mode(basis.nao_sph);
    let seed = run_scf(mol, basis, &options, functional, grid)?;
    if !seed.converged || !seed.has_integer_occupations() {
        return Err(DftError::InvalidInput(format!(
            "{method} requires a converged parent with leading-column integer occupations"
        )));
    }
    let jk = JkEngine::from_options(mol, basis, &options)?;
    let jk_lr = (functional.hybrid.clr != 0.0)
        .then(|| {
            JkEngine::new(
                basis,
                options.eri_mode,
                CoulombKernel::Erf {
                    omega: functional.hybrid.omega,
                },
            )
        })
        .transpose()?;
    let nlc_grid = options
        .vv10
        .map(|_| options.nonlocal_grid(mol))
        .transpose()?;
    let model = MolecularDensityModel {
        mol,
        mb: basis,
        opts: &options,
        functional,
        grid,
        h_core: &seed.h_core,
        extension: None,
        jk: &jk,
        jk_lr: jk_lr.as_ref(),
        pcm_solver: None,
        nlc_grid: nlc_grid.as_ref(),
        restricted: false,
    };
    evaluate(&model, &seed)
}

/// Evaluate FLO-SIC with caller-specified spin descriptors in Bohr. The
/// occupied subspace and the descriptors are held fixed for this evaluation.
/// The returned physical residual refers to the parent UKS equations.
pub fn run_fixed_descriptors(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    descriptors: [&[Vec3]; 2],
) -> Result<FixedDescriptorSic> {
    Ok(run_descriptors(
        mol,
        basis,
        options,
        functional,
        grid,
        descriptors,
        DescriptorMode::Fixed,
    )?
    .state)
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorMode {
    Fixed,
    Gradient,
    Optimize {
        gradient_tolerance: f64,
        max_iterations: usize,
    },
    SelfConsistent(StationaryFloOptions),
}

pub use stationary::{FloStationarity, StationaryFloOptions};
pub type DescriptorGradient = [Vec<[f64; 3]>; 2];

pub struct FloDescriptorRun {
    pub state: FixedDescriptorSic,
    pub descriptors: [Vec<Vec3>; 2],
    /// Hartree per Bohr, at fixed parent occupied subspaces.
    pub descriptor_gradient: Option<DescriptorGradient>,
    pub optimization: Option<crate::pseudo_sub::fit::OptimizerResult>,
    pub stationarity: Option<FloStationarity>,
}

struct DescriptorDerivatives {
    descriptor: DescriptorGradient,
    occupied: [Matrix; 2],
}

#[allow(clippy::too_many_arguments)]
pub fn run_descriptors(
    mol: &Molecule,
    basis: &MolBasis,
    options: &ScfOptions,
    functional: &ResolvedFunctional,
    grid: Option<&MolecularGrid>,
    descriptors: [&[Vec3]; 2],
    mode: DescriptorMode,
) -> Result<FloDescriptorRun> {
    if descriptors
        .iter()
        .flat_map(|x| x.iter().flat_map(|p| p.to_array()))
        .any(|v| !v.is_finite())
    {
        return Err(DftError::InvalidInput(
            "Fermi-orbital descriptor coordinates must be finite".into(),
        ));
    }
    with_uks_sic_reference(
        mol,
        basis,
        options,
        functional,
        grid,
        "FLO-SIC",
        |model, seed| {
            let counts = [seed.n_alpha, seed.n_beta];
            for (spin, &count) in counts.iter().enumerate() {
                if descriptors[spin].len() != count {
                    return Err(DftError::InvalidInput(format!(
                        "FLO-SIC spin channel {spin} requires {count} Fermi-orbital descriptors"
                    )));
                }
            }
            if let DescriptorMode::SelfConsistent(config) = mode {
                return stationary::optimize(model, seed, descriptors, config);
            }
            let occupied = occupied_blocks(&seed.mo_coeff, counts);
            let parent_residual = parent_orbital_residual(seed);
            let (descriptors, optimization) = if let DescriptorMode::Optimize {
                gradient_tolerance,
                max_iterations,
            } = mode
            {
                let (relaxed, result) = optimize_coordinates(
                    model,
                    seed,
                    &occupied,
                    descriptors,
                    gradient_tolerance,
                    max_iterations,
                )?;
                (relaxed, Some(result))
            } else {
                (descriptors.map(<[Vec3]>::to_vec), None)
            };
            let with_gradient = !matches!(mode, DescriptorMode::Fixed);
            let (state, derivatives) = evaluate_descriptors(
                model,
                seed,
                &occupied,
                [&descriptors[0], &descriptors[1]],
                with_gradient,
                parent_residual,
            )?;
            crate::logger::log(&format!(
                "[flo-sic] E={:.12} correction={:.6e} parent_residual={:.3e} density_defect={:.3e}",
                state.energy,
                state.correction,
                state.parent_orbital_residual,
                state.density_residual,
            ));
            Ok(FloDescriptorRun {
                state,
                descriptors,
                descriptor_gradient: derivatives.map(|values| values.descriptor),
                optimization,
                stationarity: None,
            })
        },
    )
}

fn optimize_coordinates(
    model: &MolecularDensityModel<'_>,
    seed: &ScfResult,
    occupied: &[Matrix; 2],
    descriptors: [&[Vec3]; 2],
    tolerance: f64,
    max_iterations: usize,
) -> Result<([Vec<Vec3>; 2], crate::pseudo_sub::fit::OptimizerResult)> {
    use crate::pseudo_sub::fit::projected_lbfgs;
    let coordinates = |x: &[f64]| {
        let (a, b) = x.split_at(3 * occupied[0].cols);
        [a, b].map(|part| {
            part.chunks_exact(3)
                .map(|p| Vec3::new(p[0], p[1], p[2]))
                .collect::<Vec<_>>()
        })
    };
    let initial: Vec<_> = descriptors
        .iter()
        .flat_map(|x| x.iter().flat_map(|p| p.to_array()))
        .collect();
    let result = projected_lbfgs(
        &initial,
        8,
        max_iterations,
        tolerance,
        1e-10,
        |x| {
            let candidate = coordinates(x);
            let (state, derivative) = evaluate_descriptors(
                model,
                seed,
                occupied,
                [&candidate[0], &candidate[1]],
                true,
                0.0,
            )?;
            // During the descriptor update the parent term is constant.
            let gradient = derivative.expect("requested FLO derivatives").descriptor;
            Ok((
                state.correction,
                gradient.into_iter().flatten().flatten().collect(),
            ))
        },
        |_| Ok(0),
    )?;
    Ok((coordinates(&result.parameters), result))
}

fn occupied_blocks(coefficients: &[Matrix; 2], counts: [usize; 2]) -> [Matrix; 2] {
    std::array::from_fn(|spin| {
        let parent = &coefficients[spin];
        let count = counts[spin];
        Matrix::from_row_major(
            parent.rows,
            count,
            (0..parent.rows)
                .flat_map(|row| (0..count).map(move |col| parent[(row, col)]))
                .collect(),
        )
    })
}

fn parent_orbital_residual(seed: &ScfResult) -> f64 {
    let mut residual = 0.0_f64;
    for (spin, count) in [seed.n_alpha, seed.n_beta].into_iter().enumerate() {
        let parent = &seed.mo_coeff[spin];
        let fock = parent.transpose_matmul_seq(&seed.fock[spin].matmul_seq(parent));
        for occupied in 0..count {
            for virtual_index in count..parent.cols {
                residual = residual.max(fock[(virtual_index, occupied)].abs());
            }
        }
    }
    residual
}

fn evaluate_descriptors(
    model: &MolecularDensityModel<'_>,
    seed: &ScfResult,
    occupied: &[Matrix; 2],
    descriptors: [&[Vec3]; 2],
    with_gradient: bool,
    parent_orbital_residual: f64,
) -> Result<(FixedDescriptorSic, Option<DescriptorDerivatives>)> {
    let basis = model.mb;
    let batches = descriptors.map(|points| eval_aos(basis, points, with_gradient));
    let make_frame =
        |spin: usize| orbitals::fermi_frame(&occupied[spin], &seed.overlap, &batches[spin].phi);
    let frames = [make_frame(0)?, make_frame(1)?];
    let zero = Matrix::zeros(basis.nao_sph, basis.nao_sph);
    let mut terms = occupied.each_ref().map(|c| Vec::with_capacity(c.cols));
    let mut gradients = [Vec::new(), Vec::new()];
    let mut occupied_gradients = occupied.each_ref().map(|c| Matrix::zeros(c.rows, c.cols));
    let mut density_residual = 0.0_f64;
    for (spin, frame) in frames.iter().enumerate() {
        let c = &frame.orbitals.coefficients;
        let localized_density = c.leading_columns_gram(c.cols, 1.0);
        let occupied_density = occupied[spin].leading_columns_gram(c.cols, 1.0);
        density_residual = density_residual.max(localized_density.max_abs_diff(&occupied_density));
        let mut coefficient_gradient = with_gradient.then(|| Matrix::zeros(c.rows, c.cols));
        for orbital in 0..c.cols {
            let column = Matrix::from_row_major(
                c.rows,
                1,
                (0..c.rows).map(|row| c[(row, orbital)]).collect(),
            );
            let (energy, potential) =
                model.single_orbital(&column.leading_columns_gram(1, 1.0), &zero)?;
            terms[spin].push(energy);
            if let Some(g) = &mut coefficient_gradient {
                // d[-E_self(phi phi^T)] / d phi = -2 V_self phi.
                let derivative = potential.matmul_seq(&column);
                for row in 0..c.rows {
                    g[(row, orbital)] = -2.0 * derivative[(row, 0)];
                }
            }
        }
        if let Some(g) = &coefficient_gradient {
            let ao_gradient = batches[spin]
                .grad
                .as_ref()
                .expect("requested descriptor AO gradients");
            let value_gradient = frame.value_gradient(&occupied[spin], g);
            gradients[spin] =
                frame.descriptor_gradient(&occupied[spin], ao_gradient, &value_gradient)?;
            occupied_gradients[spin] =
                frame.occupied_gradient(&batches[spin].phi, g, &value_gradient);
        }
    }
    let correction = -terms.iter().flatten().map(SelfEnergy::total).sum::<f64>();
    let energy = seed.energy + correction;
    if !energy.is_finite() {
        return Err(DftError::InvalidInput(
            "FLO-SIC returned a non-finite energy".into(),
        ));
    }
    Ok((
        FixedDescriptorSic {
            energy,
            parent_energy: seed.energy,
            correction,
            terms,
            orbitals: frames.map(|f| f.orbitals),
            density_residual,
            parent_orbital_residual,
            parent_iterations: seed.iterations,
        },
        with_gradient.then_some(DescriptorDerivatives {
            descriptor: gradients,
            occupied: occupied_gradients,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::method::{prepare_molecular, MolecularMethod};

    #[test]
    fn one_electron_hydrogen_molecular_ion_cancels_self_interaction() {
        let setup = prepare_molecular(
            &[1, 1],
            &[[0.13, -0.21, 0.19], [1.72, 0.37, -0.16]],
            &MolecularMethod {
                xc: Some("pbe".into()),
                charge: 1,
                multiplicity: 2,
                unrestricted: true,
                grid_level: 1,
                grid_weight_cutoff: 0.0,
                ..MolecularMethod::default()
            },
        )
        .unwrap();
        let desc = [Vec3::new(0.73, 0.04, 0.13)];
        let result = run_fixed_descriptors(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            [&desc, &[]],
        )
        .unwrap();
        let seed = run_scf(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        let direct = seed.h_core.frobenius_dot(&seed.density[0]) + seed.e_nuc;
        println!(
            "H2+ frozen FLO-SIC E={:.15e} direct={direct:.15e} parent={:.15e} correction={:.15e} density={:.3e} residual={:.3e}",
            result.energy, result.parent_energy, result.correction,
            result.density_residual, result.parent_orbital_residual
        );
        assert!((direct - result.energy).abs() < 1e-11);
        assert!(result.terms[1].is_empty());
        assert!(result.density_residual < 1e-12);
    }

    #[test]
    fn distorted_hydrogen_dimers_match_an_independent_two_orbital_energy() {
        let positions = [
            [0.13, -0.27, 0.19],
            [1.77, 0.36, -0.08],
            [2.44, 2.35, 1.20],
            [3.97, 2.88, 1.67],
        ];
        let setup = prepare_molecular(
            &[1; 4],
            &positions,
            &MolecularMethod {
                xc: Some("pbe".into()),
                unrestricted: true,
                grid_level: 0,
                grid_weight_cutoff: 0.0,
                ..MolecularMethod::default()
            },
        )
        .unwrap();
        let desc = [
            [Vec3::new(0.28, -0.15, 0.31), Vec3::new(2.62, 2.42, 1.19)],
            [Vec3::new(0.46, -0.13, 0.21), Vec3::new(3.35, 2.46, 1.44)],
        ];
        let output = run_fixed_descriptors(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
            [&desc[0], &desc[1]],
        )
        .unwrap();
        let parent = run_scf(
            &setup.molecule,
            &setup.basis,
            &setup.options,
            &setup.functional,
            setup.grid.as_ref(),
        )
        .unwrap();
        let grid = setup.grid.as_ref().unwrap();
        let n = setup.basis.nao_sph;
        let zero = Matrix::zeros(n, n);
        let jk = JkEngine::from_options(&setup.molecule, &setup.basis, &setup.options).unwrap();
        let mut expected = parent.energy;
        for (spin, descriptors) in desc.iter().enumerate() {
            let ao = eval_aos(&setup.basis, descriptors, false).phi;
            let fermi = |i| {
                let value = Matrix::from_row_major(n, 1, (0..n).map(|row| ao[(i, row)]).collect());
                let mut f = parent.density[spin].matmul_seq(&value);
                let norm = value.frobenius_dot(&f).sqrt();
                for v in f.as_mut_slice() {
                    *v /= norm;
                }
                f
            };
            let f = [fermi(0), fermi(1)];
            let s = f[0].frobenius_dot(&parent.overlap.matmul_seq(&f[1]));
            let sum = (1.0 + s).sqrt().recip();
            let difference = (1.0 - s).sqrt().recip();
            for i in 0..2 {
                let orbit = Matrix::from_row_major(
                    n,
                    1,
                    (0..n)
                        .map(|a| {
                            0.5 * ((sum + difference) * f[i][(a, 0)]
                                + (sum - difference) * f[1 - i][(a, 0)])
                        })
                        .collect(),
                );
                let density = orbit.leading_columns_gram(1, 1.0);
                let (j, _) = jk
                    .build(&setup.basis, &density, [&density, &zero], false)
                    .unwrap();
                let u = 0.5 * density.frobenius_dot(&j);
                let xc = crate::ks::xc_contribution(
                    &setup.functional,
                    &setup.basis,
                    grid,
                    [&density, &zero],
                )
                .unwrap()
                .exc;
                expected -= u + xc;
                assert!((output.terms[spin][i].hartree - u).abs() < 2e-12);
                assert!((output.terms[spin][i].semilocal - xc).abs() < 2e-12);
            }
        }
        println!(
            "C1 H4 frozen FLO-SIC E={:.15e} oracle={expected:.15e} correction={:.15e} density={:.3e} residual={:.3e}",
            output.energy, output.correction, output.density_residual, output.parent_orbital_residual
        );
        assert!((output.energy - expected).abs() < 2e-12);
        assert!(output.density_residual < 1e-12);
        assert!(output.parent_orbital_residual < 1e-7);
    }
}
