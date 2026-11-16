// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Molecular collinear PT2 quasi-energy response on the adiabatic parent path.

use super::target_difference;
use crate::driver::double_hybrid::Prepared;
use crate::driver::Value;
use crate::integrals::london::ComplexMatrix;
use crate::linalg::Matrix;
use crate::properties::pt2_response::{
    transform_frequency_path, Pt2FrequencyKernel, Pt2FrequencyVertex,
};
use crate::response::complex::Complex;
use crate::response::dynamic::OperatorPolarizability;
use crate::response::dynamic_tensor::{linear, solve_one_body_path, CMatrix, OneBodyFrequencyPath};
use crate::response::{ResponseSolver, ResponseSolverConfig};
use crate::{DftError, Result};

#[cfg(test)]
#[path = "frequency/floquet_tests.rs"]
mod floquet_tests;
#[cfg(test)]
mod tests;

pub(in crate::driver) const MODEL: &str = "Collinear double-hybrid perturbative quasi-energy susceptibility on an adiabatic parent. The distinct target follows the relaxed parent density and each full-Coulomb spin pair has left/right shifted amplitude equations in the covariant orbital frame. The frequency response includes the perturbative residues and does not resum parent poles.";

type Tensor = Vec<Vec<Complex>>;

pub(in crate::driver) struct CorrelatedFrequencyPolarizability {
    pub response: OperatorPolarizability,
    parent: Tensor,
    target: Tensor,
    pt2: Tensor,
    channels: [Tensor; 3],
    diagnostics: Vec<Value>,
    reference_residual: f64,
}

/// Subset coefficient of a matrix product. The two Fourier legs are
/// labelled, so mask 3 is the actual mixed derivative with both orderings.
fn product(left: &[CMatrix; 4], right: &[CMatrix; 4], mask: usize) -> CMatrix {
    let mut result = CMatrix::zero(left[0].re.rows);
    for sub in 0..=mask {
        if sub & mask == sub {
            result.add(&left[sub].product(&right[mask ^ sub]), Complex::ONE);
        }
    }
    result
}

fn transpose(a: &CMatrix) -> CMatrix {
    CMatrix {
        re: a.re.transpose(),
        im: a.im.transpose(),
    }
}

/// Analytic trace of two complex Fourier coefficients. The coefficients
/// are paired at opposite frequencies, without conjugating at the same z.
fn trace_product(a: &CMatrix, b: &CMatrix) -> Complex {
    let b = transpose(b);
    Complex::new(
        a.re.frobenius_dot(&b.re) - a.im.frobenius_dot(&b.im),
        a.re.frobenius_dot(&b.im) + a.im.frobenius_dot(&b.re),
    )
}

fn to_operator(matrix: CMatrix) -> ComplexMatrix {
    ComplexMatrix {
        re: matrix.re,
        im: matrix.im,
    }
}

struct MovingFrame {
    occupied_kets: Vec<ComplexMatrix>,
    occupied_duals: Vec<ComplexMatrix>,
    generator: [ComplexMatrix; 4],
    inverse_defects: [f64; 4],
    projector_defects: [f64; 4],
}

fn moving_frame(
    path: &OneBodyFrequencyPath,
    reference: &Matrix,
    spin: usize,
    occupied: usize,
    z: Complex,
) -> MovingFrame {
    let n = reference.cols;
    let identity = Matrix::identity(n);
    let mut sign = identity.clone();
    for i in occupied..n {
        sign[(i, i)] = -1.0;
    }
    let density: [CMatrix; 4] = std::array::from_fn(|h| path.density_mo[h][spin].clone());
    let mut forward: [CMatrix; 4] = std::array::from_fn(|h| {
        if h == 0 {
            CMatrix::real(identity.clone())
        } else {
            density[h].transform(&identity, &sign)
        }
    });
    let mut inverse: [CMatrix; 4] = std::array::from_fn(|h| {
        if h == 0 {
            CMatrix::real(identity.clone())
        } else {
            density[h].transform(&sign, &identity)
        }
    });
    // For X=P-P0 and S=2P0-I, Q=I+XS obeys Q_dual Q=I-X².
    // Its direct polar rotation and inverse share the mixed X²/2 term.
    let x1x2 = density[1].product(&density[2]);
    let x2x1 = density[2].product(&density[1]);
    for frame in [&mut forward, &mut inverse] {
        frame[3].add(&x1x2, Complex::from(0.5));
        frame[3].add(&x2x1, Complex::from(0.5));
    }
    let fock = std::array::from_fn(|h| path.fock_mo[h][spin].clone());
    let moving: [CMatrix; 4] = std::array::from_fn(|h| product(&fock, &forward, h));
    let derivatives = [Complex::ZERO, z, -z, Complex::ZERO];
    let du = std::array::from_fn(|h| {
        let mut coefficient = CMatrix::zero(n);
        coefficient.add(&forward[h], derivatives[h]);
        coefficient
    });
    let generator = std::array::from_fn(|h| {
        let mut coefficient = product(&inverse, &moving, h);
        coefficient.add(&product(&inverse, &du, h), Complex::from(-1.0));
        to_operator(coefficient)
    });
    let projected = forward.clone().map(|mut c| {
        for i in 0..n {
            for a in occupied..n {
                c.re[(i, a)] = 0.0;
                c.im[(i, a)] = 0.0;
            }
        }
        c
    });
    let mut inverse_defects = [0.0; 4];
    let mut projector_defects = [0.0; 4];
    for (h, exact) in density.iter().enumerate() {
        let mut overlap = product(&inverse, &forward, h);
        if h == 0 {
            overlap.add(&CMatrix::real(identity.clone()), Complex::from(-1.0));
        }
        inverse_defects[h] = overlap.max_abs();
        let mut image = product(&projected, &inverse, h);
        image.add(exact, Complex::from(-1.0));
        projector_defects[h] = image.max_abs();
    }
    let occupied_kets = forward
        .iter()
        .map(|c| to_operator(c.transform(reference, &identity)))
        .collect();
    let occupied_duals = inverse
        .iter()
        .map(|c| to_operator(transpose(c).transform(reference, &identity)))
        .collect();
    MovingFrame {
        occupied_kets,
        occupied_duals,
        generator,
        inverse_defects,
        projector_defects,
    }
}

pub(in crate::driver) fn solve(
    prepared: &Prepared,
    operators: &[Matrix],
    z: Complex,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedFrequencyPolarizability> {
    let setup = &prepared.setup;
    let scf = &prepared.reference.scf;
    if setup.options.skala.is_some()
        || setup.options.vv10.is_some()
        || prepared.target_vv10.is_some()
    {
        return Err(DftError::Unsupported {
            feature: "frequency-dependent double-hybrid response with a learned or nonlocal parent/target".into(),
            missing: "a verified nonlinear parent path and nonlocal target quasi-energy action".into(),
        });
    }
    config.validate()?;
    let reference_residual = crate::driver::molecular::reference_orbital_residual(scf);
    if !reference_residual.is_finite() || reference_residual > config.acceptance() {
        return Err(DftError::InvalidInput(format!(
            "PT2 frequency-response parent orbital residual is {reference_residual:.6e} Hartree. The response requires a finite residual of at most {:.6e} Hartree. Tighten conv_tol_grad.",
            config.acceptance()
        )));
    }
    // Four Fourier subset lanes, including the unperturbed integral tensor.
    prepared.check_path_memory(3)?;
    let solver = ResponseSolver::new(
        &setup.molecule,
        &setup.basis,
        &setup.functional,
        setup.grid.as_ref(),
        scf,
        &setup.options,
    )?;
    let target_solver = prepared
        .target
        .as_ref()
        .map(|target| {
            ResponseSolver::kernel_only(
                &setup.molecule,
                &setup.basis,
                target,
                setup.grid.as_ref(),
                scf,
                &setup.options,
            )
        })
        .transpose()?;
    let delta = target_difference(setup, &prepared.reference, prepared.target.as_ref(), None)?;
    let count = operators.len();
    let zero = || vec![vec![Complex::ZERO; count]; count];
    let pair_kernel = Pt2FrequencyKernel::new(&prepared.problem, prepared.coefficients)?;
    let mut result = CorrelatedFrequencyPolarizability {
        response: OperatorPolarizability {
            tensor: zero(),
            residuals: Vec::new(),
            iterations: Vec::new(),
        },
        parent: zero(),
        target: zero(),
        pt2: zero(),
        channels: std::array::from_fn(|_| zero()),
        diagnostics: Vec::with_capacity(count * count),
        reference_residual,
    };
    let occupied = prepared.problem.occupied;
    let nao = setup.basis.nao_sph;
    for (a, observable) in operators.iter().enumerate() {
        for (b, source) in operators.iter().enumerate() {
            // Mask 1 is the causal input, mask 2 the dual observation.
            let (path, frames) =
                crate::response::krylov::with_physical_acceptance(config, |inner| {
                    let path =
                        solve_one_body_path(&solver, &[source, observable], &[z, -z], inner)?;
                    let frames = std::array::from_fn::<_, 2, _>(|s| {
                        moving_frame(&path, &scf.mo_coeff[s], s, occupied[s], z)
                    });
                    let fresh =
                        path.defects
                            .iter()
                            .flat_map(|d| [d.projector, d.equation, d.number])
                            .chain(frames.iter().flat_map(|f| {
                                f.inverse_defects.into_iter().chain(f.projector_defects)
                            }))
                            .fold(0.0_f64, f64::max);
                    Ok(((path, frames), fresh))
                })?;
            let parent = -path.density_ao[1]
                .iter()
                .map(|p| p.contract(observable))
                .fold(Complex::ZERO, |sum, value| sum + value);
            let target = if let Some(target_solver) = &target_solver {
                let target_fock = linear(target_solver, &path.density_ao[1])?;
                let parent_fock = linear(&solver, &path.density_ao[1])?;
                -(0..2)
                    .map(|s| {
                        path.density_ao[3][s].contract(&delta[s])
                            + trace_product(&path.density_ao[2][s], &target_fock[s])
                            - trace_product(&path.density_ao[2][s], &parent_fock[s])
                    })
                    .fold(Complex::ZERO, |sum, value| sum + value)
            } else {
                Complex::ZERO
            };
            let inverse_defects = std::array::from_fn::<_, 4, _>(|h| {
                frames[0].inverse_defects[h].max(frames[1].inverse_defects[h])
            });
            let projector_defects = std::array::from_fn::<_, 4, _>(|h| {
                frames[0].projector_defects[h].max(frames[1].projector_defects[h])
            });
            let [kets, duals, generator] = {
                let [alpha, beta] = frames;
                [
                    [alpha.occupied_kets, beta.occupied_kets],
                    [alpha.occupied_duals, beta.occupied_duals],
                    [Vec::from(alpha.generator), Vec::from(beta.generator)],
                ]
            };
            let g_right = transform_frequency_path(
                nao,
                &prepared.tensor,
                &kets,
                &duals,
                &prepared.problem.energies,
                occupied,
            )?;
            let g_left = transform_frequency_path(
                nao,
                &prepared.tensor,
                &duals,
                &kets,
                &prepared.problem.energies,
                occupied,
            )?;
            let vertices: Vec<_> = g_left
                .into_iter()
                .zip(g_right)
                .enumerate()
                .skip(1)
                .map(|(h, (coupling_left, coupling_right))| Pt2FrequencyVertex {
                    coupling_left,
                    coupling_right,
                    orbital_generator: generator.each_ref().map(|f| f[h].clone()),
                })
                .collect();
            let pair = pair_kernel.quadratic(z, [&vertices[0], &vertices[1]], &vertices[2])?;
            let pair_defect = [
                pair.left_absolute_residuals,
                pair.right_absolute_residuals,
                pair.left_relative_residuals,
                pair.right_relative_residuals,
            ]
            .into_iter()
            .flatten()
            .flatten()
            .fold(0.0_f64, f64::max);
            if pair_defect > config.acceptance() {
                return Err(DftError::ResponseNotConverged {
                    iterations: 1,
                    residual: pair_defect,
                });
            }
            if pair
                .minimum_driven_denominator
                .is_some_and(|gap| gap < config.min_denominator)
            {
                return Err(DftError::InvalidInput(
                    "driven PT2 pair denominator is below response_min_denominator".into(),
                ));
            }
            result.response.tensor[a][b] = parent + target + pair.correlation;
            result.parent[a][b] = parent;
            result.target[a][b] = target;
            result.pt2[a][b] = pair.correlation;
            for (tensor, value) in result.channels.iter_mut().zip(pair.channels) {
                tensor[a][b] = value;
            }
            result.response.residuals.extend(path.residuals);
            result.response.iterations.extend(path.iterations);
            result.diagnostics.push(
                Value::object()
                    .with("component", [a, b])
                    .with(
                        "parent_projector_defects",
                        path.defects[1..]
                            .iter()
                            .map(|d| d.projector)
                            .collect::<Vec<_>>(),
                    )
                    .with(
                        "parent_equation_defects",
                        path.defects[1..]
                            .iter()
                            .map(|d| d.equation)
                            .collect::<Vec<_>>(),
                    )
                    .with(
                        "parent_number_defects",
                        path.defects[1..]
                            .iter()
                            .map(|d| d.number)
                            .collect::<Vec<_>>(),
                    )
                    .with("frame_inverse_defects", inverse_defects)
                    .with("frame_projector_defects", projector_defects)
                    .with("left_pair_residuals", pair.left_absolute_residuals)
                    .with("right_pair_residuals", pair.right_absolute_residuals)
                    .with("left_pair_relative_residuals", pair.left_relative_residuals)
                    .with(
                        "right_pair_relative_residuals",
                        pair.right_relative_residuals,
                    )
                    .with(
                        "minimum_driven_denominator",
                        pair.minimum_driven_denominator,
                    ),
            );
        }
    }
    Ok(result)
}

impl CorrelatedFrequencyPolarizability {
    pub(in crate::driver) fn append_channels(
        &self,
        report: &mut Value,
        property: &str,
        scale: f64,
        rank: usize,
        susceptibility_unit: &str,
    ) {
        let mut labels = report.get("units").cloned().unwrap_or_else(Value::object);
        for (name, tensor) in [
            (format!("parent_{property}"), &self.parent),
            (format!("target_{property}_correction"), &self.target),
            (format!("pt2_{property}_correction"), &self.pt2),
            (format!("pt2_same_spin_alpha_{property}"), &self.channels[0]),
            (format!("pt2_opposite_spin_{property}"), &self.channels[1]),
            (format!("pt2_same_spin_beta_{property}"), &self.channels[2]),
        ] {
            let part = |imag| {
                tensor
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|z| scale * if imag { z.im } else { z.re })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            for (suffix, imag) in [("real", false), ("imag", true)] {
                let key = format!("{name}_{suffix}");
                report.set(&key, part(imag));
                labels.set(&key, susceptibility_unit);
            }
        }
        report.set("frequency_pair_diagnostics", &self.diagnostics);
        report.set("response_reference_residual", self.reference_residual);
        labels.set("response_reference_residual", "hartree");
        report.set("frequency_parent_lanes", ["+z", "-z", "mixed zero"]);
        report.set(
            "frequency_frame_lanes",
            ["reference", "+z", "-z", "mixed zero"],
        );
        let first = format!("e*bohr^{rank}/hartree");
        let second = format!("e^2*bohr^{}/hartree^2", 2 * rank);
        let density = format!("[{first}, {first}, {second}]");
        let frame = format!("[dimensionless, {first}, {first}, {second}]");
        let first_energy = format!("e*bohr^{rank}");
        let second_energy = format!("e^2*bohr^{}/hartree", 2 * rank);
        let equation = format!("[{first_energy}, {first_energy}, {second_energy}]");
        for (name, unit) in [
            ("component", "integer"),
            ("parent_projector_defects", density.as_str()),
            ("parent_equation_defects", equation.as_str()),
            ("parent_number_defects", density.as_str()),
            ("frame_inverse_defects", frame.as_str()),
            ("frame_projector_defects", frame.as_str()),
            ("left_pair_residuals", first_energy.as_str()),
            ("right_pair_residuals", first_energy.as_str()),
            ("left_pair_relative_residuals", "dimensionless"),
            ("right_pair_relative_residuals", "dimensionless"),
            ("minimum_driven_denominator", "hartree"),
        ] {
            labels.set(&format!("frequency_pair_diagnostics.{name}"), unit);
        }
        report.set("units", labels);
        report.set("model", MODEL);
    }
}
