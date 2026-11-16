// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Orbital NMR as the mixed derivative of the complete post-SCF energy.
//! Complex amplitude and orbital response use the shared field provider.
use crate::basis::MolBasis;
use crate::constants::FINE_STRUCTURE;
use crate::grid::MolecularGrid;
use crate::integrals::london::{ComplexMatrix, FieldPath};
use crate::math::Vec3;
use crate::properties::magnetic::{Gauge, Shielding};
use crate::properties::nmr::{paramagnetic_operator, shielding_mixed_operator};
use crate::response::correlated_field::{correlated_field_taylor_for_model, ENERGY_CHANNELS};
use crate::response::field_path::FieldEnergyModel;
use crate::response::{ResponseReport, ResponseSolverConfig};
use crate::scf::{ScfOptions, ScfResult};
use crate::system::Molecule;
use crate::xc::double_hybrid::Pt2Coefficients;
use crate::xc::ResolvedFunctional;
use crate::Result;

/// All tensors have axes `[nucleus][nuclear moment][external field]`, ppm.
pub struct CorrelatedNmr {
    pub response: ResponseReport<Vec<Shielding>>,
    pub parent_tensors: Vec<[[f64; 3]; 3]>,
    pub target_tensors: Vec<[[f64; 3]; 3]>,
    pub correlation_tensors: Vec<[[f64; 3]; 3]>,
    pub same_spin_alpha: Vec<[[f64; 3]; 3]>,
    pub opposite_spin: Vec<[[f64; 3]; 3]>,
    pub same_spin_beta: Vec<[[f64; 3]; 3]>,
    pub parent_nonlocal_tensors: Vec<[[f64; 3]; 3]>,
    pub target_nonlocal_tensors: Vec<[[f64; 3]; 3]>,
    pub amplitude_residuals: Vec<f64>,
    pub frame_defects: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub reference_defects: Vec<f64>,
    pub imaginary_energy_defects: Vec<f64>,
}

/// Factorial-normalized diagonal paths are polarized to recover every
/// independent B/m mixed derivative. The B/m contact operator is differentiated
/// with the complete target/PT2 density, including parent orbital relaxation.
/// No finite field, parent-only substitution or denominator floor is used.
#[allow(clippy::too_many_arguments)]
pub fn correlated_molecular_nmr_response(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<&ResolvedFunctional>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    gauge: Gauge,
    origin: Vec3,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedNmr> {
    correlated_molecular_nmr_response_for_model(
        molecule,
        basis,
        parent,
        target.map(FieldEnergyModel::new),
        grid,
        reference,
        options,
        coefficients,
        gauge,
        origin,
        memory_mb,
        config,
    )
}

/// Explicitly composed target model on the same parent and response equations.
#[allow(clippy::too_many_arguments)]
pub fn correlated_molecular_nmr_response_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    gauge: Gauge,
    origin: Vec3,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedNmr> {
    let scale = FINE_STRUCTURE.powi(2) * 1e6;
    let london = gauge == Gauge::Giao;
    // The London field integral engine uses a canonical zero-origin gauge.
    // The PSO contact must use the same AO phases, rather than introducing
    // a different origin in just one derivative of the Hamiltonian.
    let operator_origin = if london { Vec3::zero() } else { origin };
    let nao = basis.nao_sph;
    let empty = || {
        [
            ComplexMatrix::zeros(nao, nao),
            ComplexMatrix::zeros(nao, nao),
        ]
    };
    let mut residuals = Vec::new();
    let mut amplitude_residuals = Vec::new();
    let mut frame_defects = Vec::new();
    let mut projector_defects = Vec::new();
    let mut stationarity_defects = Vec::new();
    let mut reference_defects = Vec::new();
    let mut imaginary_energy_defects = Vec::new();
    let mut evaluate =
        |field, core: &[[ComplexMatrix; 2]], order| -> Result<[f64; ENERGY_CHANNELS]> {
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
                    magnetic: field,
                    origin: operator_origin.to_array(),
                    london,
                    order,
                },
                Some(core),
                memory_mb,
                config,
            )?;
            residuals.extend(&value.parent.parent.residuals);
            amplitude_residuals.extend(&value.amplitude_residuals);
            frame_defects.push(value.parent.orbital_frame_defect);
            projector_defects.extend(&value.parent.parent.projector_defects);
            stationarity_defects.extend(&value.parent.parent.stationarity_defects);
            reference_defects.push(value.parent.parent.reference_defect);
            imaginary_energy_defects.push(value.imaginary_energy_defect);
            Ok(value.energy_channels(order))
        };
    let mut tensors: [Vec<[[f64; 3]; 3]>; ENERGY_CHANNELS] = std::array::from_fn(|_| Vec::new());
    let mut shieldings = Vec::new();
    for atom in &molecule.atoms {
        let pso = paramagnetic_operator(basis, atom.position);
        let contact = shielding_mixed_operator(basis, atom.position, operator_origin, london);
        let mut atom_tensors = [[[0.0; 3]; 3]; ENERGY_CHANNELS];
        let mut diamagnetic = [[0.0; 3]; 3];
        for a in 0..3 {
            let mut field = [0.0; 3];
            field[a] = 1.0;
            for b in 0..3 {
                let mut paths = Vec::new();
                for sign in [1.0, -1.0] {
                    let mut core = vec![empty(), empty(), empty()];
                    for spin in 0..2 {
                        for i in 0..nao {
                            for j in 0..nao {
                                // The real PSO matrix has its factor -i removed.
                                core[1][spin].im[(i, j)] = -sign * pso[b][(i, j)];
                                core[2][spin].re[(i, j)] = sign * contact[a][b][(i, j)];
                            }
                        }
                    }
                    paths.push(evaluate(field, &core, 2)?);
                }
                for c in 0..ENERGY_CHANNELS {
                    atom_tensors[c][b][a] = scale * (paths[0][c] - paths[1][c]) * 0.5;
                }
                let mut explicit = vec![empty(), empty()];
                for spin in 0..2 {
                    explicit[1][spin].re = contact[a][b].clone();
                }
                diamagnetic[b][a] = scale * evaluate([0.0; 3], &explicit, 1)?[0];
            }
        }
        let mut paramagnetic = atom_tensors[0];
        for b in 0..3 {
            for a in 0..3 {
                paramagnetic[b][a] -= diamagnetic[b][a];
            }
        }
        shieldings.push(Shielding {
            tensor: atom_tensors[0],
            diamagnetic,
            paramagnetic,
        });
        for (all, atom) in tensors.iter_mut().zip(atom_tensors) {
            all.push(atom);
        }
    }
    let converged = residuals
        .iter()
        .chain(&amplitude_residuals)
        .chain(&frame_defects)
        .chain(&projector_defects)
        .chain(&stationarity_defects)
        .chain(&reference_defects)
        .chain(&imaginary_energy_defects)
        .all(|x| x.is_finite() && x.abs() <= config.acceptance());
    let [_, parent_tensors, target_tensors, correlation_tensors, same_spin_alpha, opposite_spin, same_spin_beta, parent_nonlocal_tensors, target_nonlocal_tensors] =
        tensors;
    Ok(CorrelatedNmr {
        response: ResponseReport {
            value: shieldings, shape: vec![molecule.len(), 3, 3],
            functional: target.as_ref().map_or(parent,FieldEnergyModel::functional).description.clone(),
            reference: crate::response::diagnostics::reference_diagnostics(reference)?,
            units: "ppm", model: "Post-SCF orbital NMR energy Hessian; complete complex parent-orbital, noncanonical PT2 amplitude and B/m contact response; current-invariant meta-GGA kinetic density tau - |j|^2/(2 rho); integer collinear populations",
            axis_convention: "nucleus, nuclear magnetic moment (x,y,z), external B (x,y,z)",
            converged, residuals, iterations: Vec::new(),
            warnings: vec!["Orbital shieldings exclude spin-Zeeman/Curie shifts and spin-spin multiplets.".into()],
        },
        parent_tensors, target_tensors, correlation_tensors, same_spin_alpha, opposite_spin,
        same_spin_beta,parent_nonlocal_tensors,target_nonlocal_tensors, amplitude_residuals, frame_defects, projector_defects,
        stationarity_defects, reference_defects, imaginary_energy_defects,
    })
}

/// Orbital magnetizability channels, xi_ab = -d2E/dB_a dB_b, in e^2 bohr^2/m_e.
/// Spin Zeeman and current-dependent correlation are outside this collinear model.
#[derive(Default)]
pub struct CorrelatedMagnetizability {
    pub total: [[f64; 3]; 3],
    pub parent: [[f64; 3]; 3],
    pub target: [[f64; 3]; 3],
    pub correlation: [[f64; 3]; 3],
    /// Unscaled pair-channel responses, before the requested SS/OS weights.
    pub same_spin_alpha: [[f64; 3]; 3],
    pub opposite_spin: [[f64; 3]; 3],
    pub same_spin_beta: [[f64; 3]; 3],
    pub parent_nonlocal: [[f64; 3]; 3],
    pub target_nonlocal: [[f64; 3]; 3],
    pub response_residuals: Vec<f64>,
    pub amplitude_residuals: Vec<f64>,
    pub amplitude_relative_residuals: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub reference_defects: Vec<f64>,
    pub orbital_frame_defects: Vec<f64>,
    pub imaginary_energy_defects: Vec<f64>,
}

/// Differentiate the same stationary-parent target and noncanonical pair
/// equations used for orbital NMR. Polarization recovers mixed Cartesian
/// magnetic derivatives without finite fields or a parent-only approximation.
#[allow(clippy::too_many_arguments)]
pub fn correlated_magnetizability_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    gauge: Gauge,
    origin: Vec3,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedMagnetizability> {
    let mut evaluator = FieldContractions::new();
    let mut tensors = [[[0.0; 3]; 3]; ENERGY_CHANNELS];
    for a in 0..3 {
        for b in a..3 {
            let mixed = evaluator.mixed(
                molecule,
                basis,
                parent,
                target.clone(),
                grid,
                reference,
                options,
                coefficients,
                gauge,
                origin,
                memory_mb,
                config,
                &[field_axis(3 + a), field_axis(3 + b)],
            )?;
            for (tensor, derivative) in tensors.iter_mut().zip(mixed.value) {
                tensor[a][b] = -derivative;
                tensor[b][a] = -derivative;
            }
        }
    }
    evaluator.diagnostics.set_tensors(tensors);
    Ok(evaluator.diagnostics)
}

/// Full static eta_abcd = -d4E/(dE_a dE_b dB_c dB_d), including the
/// target-density and complex PT2 orbital/amplitude response. Channels are
/// total, parent, target, weighted PT2, SS-alpha, OS, SS-beta, parent
/// nonlocal and target nonlocal, in that order. The electric and magnetic
/// index pairs each obey their exact static permutation symmetry.
pub struct CorrelatedHypermagnetizability {
    pub polarizability: [[[f64; 3]; 3]; ENERGY_CHANNELS],
    pub tensor: [[[[[f64; 3]; 3]; 3]; 3]; ENERGY_CHANNELS],
    pub magnetic: CorrelatedMagnetizability,
    pub accumulation_roundoff: [f64; ENERGY_CHANNELS],
    pub directional_evaluations: usize,
}

/// Assemble all 36 independent E/E/B/B entries from analytic directional
/// coefficients. The same cached field paths supply alpha and xi, so the
/// Cotton-Mouton orientational term uses the complete correlated energy.
#[allow(clippy::too_many_arguments)]
pub fn correlated_hypermagnetizability_for_model(
    molecule: &Molecule,
    basis: &MolBasis,
    parent: &ResolvedFunctional,
    target: Option<FieldEnergyModel<'_>>,
    grid: Option<&MolecularGrid>,
    reference: &ScfResult,
    options: &ScfOptions,
    coefficients: Pt2Coefficients,
    gauge: Gauge,
    origin: Vec3,
    memory_mb: usize,
    config: &ResponseSolverConfig,
) -> Result<CorrelatedHypermagnetizability> {
    let mut evaluator = FieldContractions::new();
    let mut alpha = [[[0.0; 3]; 3]; ENERGY_CHANNELS];
    let mut xi = alpha;
    let mut tensor = [[[[[0.0; 3]; 3]; 3]; 3]; ENERGY_CHANNELS];
    let mut roundoff = [0.0_f64; ENERGY_CHANNELS];
    let mut mixed = |axes: &[usize]| {
        evaluator.mixed(
            molecule,
            basis,
            parent,
            target.clone(),
            grid,
            reference,
            options,
            coefficients,
            gauge,
            origin,
            memory_mb,
            config,
            &axes.iter().map(|&a| field_axis(a)).collect::<Vec<_>>(),
        )
    };
    for a in 0..3 {
        for b in a..3 {
            for (offset, tensors) in [(0, &mut alpha), (3, &mut xi)] {
                let value = mixed(&[offset + a, offset + b])?;
                for (channel, derivative) in tensors.iter_mut().zip(value.value) {
                    channel[a][b] = -derivative;
                    channel[b][a] = -derivative;
                }
            }
            for c in 0..3 {
                for d in c..3 {
                    let value = mixed(&[a, b, 3 + c, 3 + d])?;
                    for channel in 0..ENERGY_CHANNELS {
                        let derivative = -value.value[channel];
                        for (i, j) in [(a, b), (b, a)] {
                            tensor[channel][i][j][c][d] = derivative;
                            tensor[channel][i][j][d][c] = derivative;
                        }
                        roundoff[channel] =
                            roundoff[channel].max(value.accumulation_roundoff[channel]);
                    }
                }
            }
        }
    }
    evaluator.diagnostics.set_tensors(xi);
    Ok(CorrelatedHypermagnetizability {
        polarizability: alpha,
        tensor,
        magnetic: evaluator.diagnostics,
        accumulation_roundoff: roundoff,
        directional_evaluations: evaluator.cache.evaluations,
    })
}

fn field_axis(axis: usize) -> Vec<f64> {
    let mut vector = vec![0.0; 6];
    vector[axis] = 1.0;
    vector
}

impl CorrelatedMagnetizability {
    fn set_tensors(&mut self, tensors: [[[f64; 3]; 3]; ENERGY_CHANNELS]) {
        [
            self.total,
            self.parent,
            self.target,
            self.correlation,
            self.same_spin_alpha,
            self.opposite_spin,
            self.same_spin_beta,
            self.parent_nonlocal,
            self.target_nonlocal,
        ] = tensors;
    }

    fn record(&mut self, value: &crate::response::correlated_field::CorrelatedFieldTaylor) {
        self.response_residuals
            .extend(&value.parent.parent.residuals);
        self.amplitude_residuals.extend(&value.amplitude_residuals);
        self.amplitude_relative_residuals
            .extend(&value.amplitude_relative_residuals);
        self.projector_defects
            .extend(&value.parent.parent.projector_defects);
        self.stationarity_defects
            .extend(&value.parent.parent.stationarity_defects);
        self.reference_defects
            .push(value.parent.parent.reference_defect);
        self.orbital_frame_defects
            .push(value.parent.orbital_frame_defect);
        self.imaginary_energy_defects
            .push(value.imaginary_energy_defect);
    }
}

/// Share field contractions and raw diagnostics between xi and eta. The
/// cache retains energy coefficients only; large orbital and integral paths
/// are released after each direction instead of accumulating across a tensor.
struct FieldContractions {
    cache: crate::response::mixed::DirectionalCoefficients,
    diagnostics: CorrelatedMagnetizability,
}

impl FieldContractions {
    fn new() -> Self {
        Self {
            cache: Default::default(),
            diagnostics: Default::default(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mixed(
        &mut self,
        molecule: &Molecule,
        basis: &MolBasis,
        parent: &ResolvedFunctional,
        target: Option<FieldEnergyModel<'_>>,
        grid: Option<&MolecularGrid>,
        reference: &ScfResult,
        options: &ScfOptions,
        coefficients: Pt2Coefficients,
        gauge: Gauge,
        origin: Vec3,
        memory_mb: usize,
        config: &ResponseSolverConfig,
        vectors: &[Vec<f64>],
    ) -> Result<crate::response::mixed::MixedDerivative> {
        crate::response::mixed::polarized_mixed(6, vectors, |direction, order| {
            self.cache
                .get(direction, order, ENERGY_CHANNELS, |direction, order| {
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
                            electric: [direction[0], direction[1], direction[2]],
                            magnetic: [direction[3], direction[4], direction[5]],
                            origin: origin.to_array(),
                            london: gauge == Gauge::Giao,
                            order,
                        },
                        None,
                        memory_mb,
                        config,
                    )?;
                    self.diagnostics.record(&value);
                    Ok(value.energy_channels(order).to_vec())
                })
        })
    }
}
