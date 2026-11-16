// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Field-free stationary complex-spinor nuclear gradients.
//! This assembly differentiates the actual dimensional energy model, not a
//! collinear substitute. Lattice strain, fields, SOC and nonlocal XC are separate.
use super::ExchangeNuclearDerivative;
use crate::basis::MolBasis;
use crate::dfpt::orbitals::{generalized_orbitals, matmul};
use crate::integrals::CoulombKernel;
use crate::linalg::Orthogonalizer;
use crate::math::Vec3;
use crate::periodic::bloch::BlochMatrix;
use crate::periodic::integrals::{
    bloch_derivative, contract_ecp_geometry, dimensional_image_derivative_blocks, EcpBoundary,
    EcpGeometry, ImageBlock, Operator,
};
use crate::periodic::occupations::solve_occupations;
use crate::periodic::spinor_scf::{
    pauli_components, SpinorDftBackend, SpinorEnergyBackend, SpinorScfConfig, SpinorScfReport,
};
use crate::periodic::spinor_xc::{SpinorXcGridContext, SpinorXcGridTangent};
use crate::periodic::PeriodicGeometry;
use crate::response::complex::Complex;
use crate::{DftError, Result};

pub(crate) struct FieldFreeFrame<'a> {
    pub molecule: &'a crate::system::Molecule,
    pub basis: &'a MolBasis,
    pub geometry: &'a PeriodicGeometry,
    pub functional: &'a crate::xc::ResolvedFunctional,
    pub options: &'a crate::periodic::scf_boundary::DimensionalScfOptions,
    pub coulomb: &'a crate::periodic::coulomb_boundary::DimensionalCoulombBackend,
    pub grid: Option<&'a crate::periodic::xc::PeriodicXcGrid>,
    pub ecp_images: &'a [ImageBlock],
}

impl FieldFreeFrame<'_> {
    pub(crate) fn ecp_geometry(
        &self,
        points: &[Vec3],
        charge: &[BlochMatrix],
    ) -> Result<EcpGeometry> {
        contract_ecp_geometry(
            self.basis,
            EcpBoundary::Dimensional(self.geometry),
            self.options.basis_cutoff,
            self.molecule.len(),
            self.ecp_images,
            points,
            charge,
        )
    }
}
pub struct DimensionalNuclearGradient {
    pub total: Vec<Vec3>,
    pub kinetic: Vec<Vec3>,
    pub electrostatic: Vec<Vec3>,
    pub ecp: Vec<Vec3>,
    pub full_exchange: Vec<Vec3>,
    pub long_range_exchange: Vec<Vec3>,
    pub semilocal_xc: Vec<Vec3>,
    pub nonlocal_correlation: Option<Vec<Vec3>>,
    pub pulay: Vec<Vec3>,
    pub external_field: Option<Vec<Vec3>>,
    pub external_field_energy_defect: Option<f64>,
    /// Fixed environmental charge centres, after the physical atoms.
    pub environment_gradient: Vec<Vec3>,
    pub fresh_orbital_residual: f64,
    pub fresh_occupation_consistency: f64,
    pub fresh_number_error: f64,
    pub reference_energy_defect: f64,
    pub reference_fock_defect: f64,
    pub xc_reference_energy_defect: f64,
}
pub type SpinorNuclearGradient = DimensionalNuclearGradient;
fn at(m: &BlochMatrix, i: usize, j: usize) -> Complex {
    Complex::new(m.re[(i, j)], m.im[(i, j)])
}
fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.max_abs_diff(&b.re).max(a.im.max_abs_diff(&b.im))
}
pub(super) fn density_images(
    geometry: &PeriodicGeometry,
    points: &[Vec3],
    density: &[BlochMatrix],
    cutoff: f64,
) -> Result<(Vec<ImageBlock>, Vec<ImageBlock>)> {
    let n = density[0].rows();
    let mut real = Vec::new();
    let mut imaginary = Vec::new();
    for translation in geometry.translations_within(2. * cutoff)? {
        let mut re = crate::linalg::Matrix::zeros(n, n);
        let mut im = re.clone();
        for (&k, p) in points.iter().zip(density) {
            let (s, c) = k.dot(translation).sin_cos();
            for i in 0..n {
                for j in 0..n {
                    re[(i, j)] += (c * p.re[(i, j)] + s * p.im[(i, j)]) / points.len() as f64;
                    im[(i, j)] += (c * p.im[(i, j)] - s * p.re[(i, j)]) / points.len() as f64;
                }
            }
        }
        real.push(ImageBlock {
            translation,
            matrix: re,
        });
        imaginary.push(ImageBlock {
            translation,
            matrix: im,
        });
    }
    Ok((real, imaginary))
}

pub(super) struct Vv10Geometry {
    pub(super) gradient: Vec<Vec3>,
    pub(super) strain: Option<[[f64; 3]; 3]>,
    pub(super) energy_defect: f64,
}

impl FieldFreeFrame<'_> {
    pub(super) fn vv10_geometry(
        &self,
        k_points: &[Vec3],
        charge: &[BlochMatrix],
        expected: Option<f64>,
        strain: bool,
    ) -> Result<Option<Vv10Geometry>> {
        let Some(expected) = expected else {
            return Ok(None);
        };
        let grid = self.grid.expect("periodic VV10 has a quadrature grid");
        // VV10 depends on the total, scalar Pauli density only. The
        // local-energy-free contraction supplies all moving-grid terms.
        let n = self.basis.nao_sph;
        let density = [
            charge.to_vec(),
            (0..charge.len())
                .map(|_| BlochMatrix::zeros(n, n))
                .collect(),
        ];
        let (gradient, strain, energy) = crate::periodic::xc_gradient::collinear_xc_geometry(
            crate::periodic::xc_gradient::XcGeometry::Dimensional {
                molecule: self.molecule,
                basis: self.basis,
                geometry: self.geometry,
                grid_cutoff: self.options.grid_cutoff,
            },
            grid,
            &crate::xc::ResolvedFunctional::hartree_fock(),
            k_points,
            &density,
            self.options
                .scf
                .xc_ao_cutoff
                .unwrap_or(self.options.basis_cutoff),
            strain,
        )?;
        let energy_defect = (energy - expected).abs();
        if !energy_defect.is_finite() || energy_defect > self.options.operator_tolerance {
            return Err(DftError::InvalidInput(
                "spinor VV10 geometry grid does not reproduce the stationary nonlocal energy"
                    .into(),
            ));
        }
        Ok(Some(Vv10Geometry {
            gradient,
            strain,
            energy_defect,
        }))
    }
}

pub(crate) fn one_body_gradients(
    frame: &FieldFreeFrame<'_>,
    points: &[Vec3],
    charge: &[BlochMatrix],
    weighted: &[BlochMatrix],
) -> Result<[Vec<Vec3>; 2]> {
    let natoms = frame.molecule.len();
    let n = frame.basis.nao_sph;
    let nk = points.len();
    let mut kinetic = vec![Vec3::zero(); natoms];
    let mut pulay = kinetic.clone();
    for (operator, target, density, factor) in [
        (Operator::Kinetic, &mut kinetic, charge, 1.),
        (Operator::Overlap, &mut pulay, weighted, -1.),
    ] {
        let images = dimensional_image_derivative_blocks(
            frame.basis,
            frame.geometry,
            frame.options.basis_cutoff,
            operator,
        )?;
        for (k, &point) in points.iter().enumerate() {
            let d = bloch_derivative(&images, point, natoms, n);
            for atom in 0..natoms {
                for axis in 0..3 {
                    let m = BlochMatrix::from_pair(d[atom][axis].clone());
                    let value = factor * density[k].trace_product(&m) / nk as f64;
                    target[atom] += match axis {
                        0 => Vec3::new(value, 0., 0.),
                        1 => Vec3::new(0., value, 0.),
                        _ => Vec3::new(0., 0., value),
                    };
                }
            }
        }
    }
    Ok([kinetic, pulay])
}

pub(crate) fn exchange_geometry_terms(natoms: usize) -> [ExchangeNuclearDerivative; 2] {
    std::array::from_fn(|_| ExchangeNuclearDerivative {
        gradient: vec![Vec3::zero(); natoms],
        strain_derivative: [[0.0; 3]; 3],
    })
}

pub(crate) fn add_exchange_geometry(
    frame: &FieldFreeFrame<'_>,
    points: &[Vec3],
    density: &[BlochMatrix],
    energy_factor: f64,
    terms: &mut [ExchangeNuclearDerivative; 2],
) -> Result<()> {
    let hybrid = frame.functional.hybrid;
    if hybrid.cx == 0.0 && hybrid.clr == 0.0 {
        return Ok(());
    }
    let cutoff = frame
        .options
        .scf
        .exchange_cutoff
        .unwrap_or(frame.options.basis_cutoff);
    let (real, imaginary) = density_images(frame.geometry, points, density, cutoff)?;
    for (target, (coefficient, kernel)) in terms.iter_mut().zip([
        (hybrid.cx, CoulombKernel::Coulomb),
        (
            hybrid.clr,
            CoulombKernel::Erf {
                omega: hybrid.omega,
            },
        ),
    ]) {
        if coefficient == 0.0 {
            continue;
        }
        let value = super::complex_exchange_nuclear_derivative(
            frame.basis,
            frame.geometry,
            frame.molecule.len(),
            super::ComplexImageDensity {
                real: &real,
                imaginary: &imaginary,
            },
            cutoff,
            kernel,
            energy_factor * coefficient,
        )?;
        for (g, delta) in target.gradient.iter_mut().zip(value.gradient) {
            *g += delta;
        }
        for (row, change) in target
            .strain_derivative
            .iter_mut()
            .zip(value.strain_derivative)
        {
            for (target, change) in row.iter_mut().zip(change) {
                *target += change;
            }
        }
    }
    Ok(())
}

pub(crate) fn evaluate(
    frame: FieldFreeFrame<'_>,
    backend: &SpinorDftBackend<'_>,
    reference: &SpinorScfReport,
    config: &SpinorScfConfig,
) -> Result<SpinorNuclearGradient> {
    let bad = |message: &str| DftError::InvalidInput(message.into());
    let accepted = accepted_reference(&frame, backend, reference, config)?;
    let natoms = frame.molecule.len();
    let weighted = accepted.weighted;
    let (orbital, consistency, number_error, energy_defect, fock_defect) = (
        accepted.orbital,
        accepted.consistency,
        accepted.number_error,
        accepted.energy_defect,
        accepted.fock_defect,
    );
    let pc = reference
        .density
        .iter()
        .map(pauli_components)
        .collect::<Result<Vec<_>>>()?;
    let charge: Vec<_> = pc.iter().map(|p| p[0].clone()).collect();
    let coulomb = frame.coulomb.evaluate(Some(&charge))?;
    if !coulomb.converged {
        return Err(bad(
            "spinor gradient electrostatic derivative provider is unconverged",
        ));
    }
    let all_coulomb = coulomb
        .energy_gradients
        .ok_or_else(|| bad("missing physical electrostatic nuclear derivative"))?;
    if all_coulomb.len() < natoms {
        return Err(bad("electrostatic centre map omits physical atoms"));
    }
    let electrostatic = all_coulomb[..natoms].to_vec();
    let environment_gradient = all_coulomb[natoms..].to_vec();
    let points = frame.geometry.k_points(frame.options.scf.k_mesh)?;
    let [kinetic, pulay] = one_body_gradients(&frame, &points, &charge, &weighted)?;
    let ecp = frame.ecp_geometry(&points, &charge)?.gradient;
    let mut exchange = exchange_geometry_terms(natoms);
    if frame.functional.hybrid.cx != 0.0 || frame.functional.hybrid.clr != 0.0 {
        for component in 0..4 {
            let density: Vec<_> = pc.iter().map(|p| p[component].clone()).collect();
            add_exchange_geometry(&frame, &points, &density, -0.25, &mut exchange)?;
        }
    }
    let (semilocal_xc, semilocal_defect) = xc_gradient(
        &frame,
        &pc,
        accepted.xc_energy,
        accepted.angular_order,
        config.energy_tolerance,
    )?;
    let vv10 = frame.vv10_geometry(&points, &charge, accepted.nonlocal_energy, false)?;
    let mut total = vec![Vec3::zero(); natoms];
    for term in [
        &kinetic,
        &electrostatic,
        &ecp,
        &exchange[0].gradient,
        &exchange[1].gradient,
        &semilocal_xc,
        &pulay,
    ] {
        for (g, v) in total.iter_mut().zip(term) {
            *g += *v;
        }
    }
    if let Some(vv10) = &vv10 {
        for (g, v) in total.iter_mut().zip(&vv10.gradient) {
            *g += *v;
        }
    }
    let xc_reference_energy_defect =
        semilocal_defect.max(vv10.as_ref().map_or(0.0, |term| term.energy_defect));
    let [full_exchange, long_range_exchange] = exchange.map(|term| term.gradient);
    Ok(SpinorNuclearGradient {
        total,
        kinetic,
        electrostatic,
        ecp,
        full_exchange,
        long_range_exchange,
        semilocal_xc,
        nonlocal_correlation: vv10.map(|term| term.gradient),
        pulay,
        external_field: None,
        external_field_energy_defect: None,
        environment_gradient,
        fresh_orbital_residual: orbital,
        fresh_occupation_consistency: consistency,
        fresh_number_error: number_error,
        reference_energy_defect: energy_defect,
        reference_fock_defect: fock_defect,
        xc_reference_energy_defect,
    })
}

fn xc_gradient(
    frame: &FieldFreeFrame<'_>,
    density: &[[BlochMatrix; 4]],
    expected: f64,
    order: usize,
    tolerance: f64,
) -> Result<(Vec<Vec3>, f64)> {
    let natoms = frame.molecule.len();
    let n = frame.basis.nao_sph;
    let Some(grid) = frame.grid else {
        return Ok((vec![Vec3::zero(); natoms], 0.));
    };
    let ng = grid.points.len();
    let nk = grid.k_points.len();
    let bytes = crate::resources::checked_bytes(
        &[ng, n, nk, 16, 18],
        "spinor XC nuclear geometry buffers",
    )?;
    let _reservation = crate::resources::reserve(bytes, "spinor XC nuclear geometry buffers")?;
    let aos = crate::periodic::xc::build_dimensional_bloch_aos(
        frame.basis,
        frame.geometry,
        &grid.k_points,
        &grid.points,
        frame
            .options
            .scf
            .xc_ao_cutoff
            .unwrap_or(frame.options.basis_cutoff),
        crate::periodic::xc::AoDerivativeOrder::Hessian,
    )?;
    let momentum: Vec<[BlochMatrix; 3]> = aos
        .iter()
        .map(|a| {
            std::array::from_fn(|d| {
                let g = &a.grad.as_ref().expect("geometry AO gradients")[d];
                let mut p = g.clone();
                for ((r, i), (gr, gi)) in
                    p.re.as_mut_slice()
                        .iter_mut()
                        .zip(p.im.as_mut_slice())
                        .zip(g.re.as_slice().iter().zip(g.im.as_slice()))
                {
                    *r = *gi;
                    *i = -*gr;
                }
                p
            })
        })
        .collect();
    let orbitals: Vec<_> = aos.iter().map(|a| &a.phi).collect();
    let momenta: Vec<_> = momentum.iter().collect();
    let context = SpinorXcGridContext::new(
        frame.functional,
        &grid.weights,
        &orbitals,
        &momenta,
        density,
        order,
    )?;
    let defect = (context.energy() - expected).abs();
    if defect > tolerance {
        return Err(DftError::InvalidInput(
            "spinor XC geometry grid does not reproduce the actual SCF functional energy".into(),
        ));
    }
    let owners = frame.basis.sph_ao_atoms();
    let mut weight_response = Vec::with_capacity(ng);
    for g in 0..ng {
        let (_, mut response) = crate::periodic::grid::dimensional_weight_gradient(
            frame.molecule,
            frame.geometry,
            grid.points[g],
            grid.atom_of[g],
            frame.options.grid_cutoff,
        )?;
        let sum = response.iter().fold(Vec3::zero(), |s, v| s + *v);
        response[grid.atom_of[g]] -= sum;
        for r in &mut response {
            *r = *r * grid.raw_weights[g];
        }
        weight_response.push(response);
    }
    let mut result = vec![Vec3::zero(); natoms];
    for atom in 0..natoms {
        for axis in 0..3 {
            let mut phi: Vec<_> = (0..nk).map(|_| BlochMatrix::zeros(ng, n)).collect();
            let mut dp: Vec<[BlochMatrix; 3]> = (0..nk)
                .map(|_| std::array::from_fn(|_| BlochMatrix::zeros(ng, n)))
                .collect();
            for k in 0..nk {
                for g in 0..ng {
                    for a in 0..n {
                        let factor = if grid.atom_of[g] == atom { 1. } else { 0. }
                            - if owners[a] == atom { 1. } else { 0. };
                        let grad = &aos[k].grad.as_ref().expect("gradient")[axis];
                        phi[k].re[(g, a)] = factor * grad.re[(g, a)];
                        phi[k].im[(g, a)] = factor * grad.im[(g, a)];
                        for d in 0..3 {
                            let h = &aos[k].hess.as_ref().expect("Hessian")
                                [crate::grid::eval::second_index(axis, d)];
                            dp[k][d].re[(g, a)] = factor * h.im[(g, a)];
                            dp[k][d].im[(g, a)] = -factor * h.re[(g, a)];
                        }
                    }
                }
            }
            let weights: Vec<_> = weight_response.iter().map(|v| v[atom].get(axis)).collect();
            let p: Vec<_> = phi.iter().collect();
            let m: Vec<_> = dp.iter().collect();
            let derivative = context.differential(SpinorXcGridTangent {
                weights: &weights,
                orbitals: &p,
                momentum: &m,
            })?;
            result[atom] += match axis {
                0 => Vec3::new(derivative, 0., 0.),
                1 => Vec3::new(0., derivative, 0.),
                _ => Vec3::new(0., 0., derivative),
            };
        }
    }
    Ok((result, defect))
}

pub(super) struct AcceptedReference {
    pub(super) weighted: Vec<BlochMatrix>,
    pub(super) orbital: f64,
    pub(super) consistency: f64,
    pub(super) number_error: f64,
    pub(super) energy_defect: f64,
    pub(super) fock_defect: f64,
    pub(super) xc_energy: f64,
    pub(super) nonlocal_energy: Option<f64>,
    pub(super) angular_order: usize,
}
pub(super) fn accepted_reference(
    frame: &FieldFreeFrame<'_>,
    backend: &SpinorDftBackend<'_>,
    reference: &SpinorScfReport,
    config: &SpinorScfConfig,
) -> Result<AcceptedReference> {
    let bad = |message: &str| DftError::InvalidInput(message.into());
    if [
        config.energy_tolerance,
        config.density_tolerance,
        config.orbital_tolerance,
        config.overlap_cutoff,
    ]
    .iter()
    .any(|v| !v.is_finite() || *v <= 0.)
    {
        return Err(bad(
            "spinor gradient requires positive finite reference tolerances",
        ));
    }
    if !reference.converged
        || !reference.fields.operator_converged
        || reference
            .fields
            .xc_diagnostics
            .as_ref()
            .is_some_and(|x| !x.converged)
    {
        let error = reference
            .orbital_residual
            .max(reference.occupation_consistency)
            .max(reference.particle_number_error)
            .max(reference.fields.coulomb_operator_defect)
            .max(reference.fields.xc_grid_number_defect)
            .max(reference.fields.xc_diagnostics.as_ref().map_or(0., |d| {
                d.angular_operator_defect.max(d.angular_energy_defect)
            }));
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error,
        });
    }
    if config.occupation.temperature_kelvin() != 0. {
        return Err(DftError::Unsupported {
            feature: "thermal spinor nuclear gradient".into(),
            missing: "validated free-energy metric/occupation derivative assembly".into(),
        });
    }
    // Geometry-only corrections are evaluated separately by the caller.
    if frame.options.scf.skala.is_some() {
        return Err(DftError::Unsupported {
            feature: "spinor gradient with additional nonlocal/learned providers".into(),
            missing: "geometry derivatives of every additional actual energy provider".into(),
        });
    }
    let fields = backend.evaluate(&reference.density)?;
    if !fields.operator_converged || fields.xc_diagnostics.as_ref().is_some_and(|x| !x.converged) {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: fields
                .coulomb_operator_defect
                .max(fields.xc_grid_number_defect)
                .max(fields.xc_diagnostics.as_ref().map_or(0., |d| {
                    d.angular_operator_defect.max(d.angular_energy_defect)
                })),
        });
    }
    if fields.fock.len() != reference.fields.fock.len() || !reference.fields.energy.is_finite() {
        return Err(bad(
            "spinor gradient reference has inconsistent physical fields",
        ));
    }
    let energy_defect = (fields.energy - reference.fields.energy).abs();
    let fock_defect = fields
        .fock
        .iter()
        .zip(&reference.fields.fock)
        .map(|(a, b)| difference(a, b))
        .fold(0., f64::max);
    if energy_defect > config.energy_tolerance || fock_defect > config.orbital_tolerance {
        return Err(bad(
            "spinor gradient reference does not belong to this immutable physical frame",
        ));
    }
    let bands = fields
        .fock
        .iter()
        .zip(backend.overlap())
        .map(|(f, s)| generalized_orbitals(f, s, config.overlap_cutoff))
        .collect::<Result<Vec<_>>>()?;
    if bands
        .iter()
        .zip(backend.overlap())
        .any(|(b, s)| b.coefficients.cols() != s.rows())
    {
        return Err(DftError::Unsupported {
            feature: "spinor nuclear gradient with a moving discarded overlap subspace".into(),
            missing: "retained/discarded metric projector geometry response".into(),
        });
    }
    let energies: Vec<_> = bands.iter().map(|b| b.energies.clone()).collect();
    let occupations = solve_occupations(&energies, backend.electrons(), config.occupation)?;
    let nk = bands.len();
    let n = frame.basis.nao_sph;
    let mut weighted = Vec::new();
    let mut consistency = 0.0_f64;
    let mut orbital = 0.0_f64;
    for (k, b) in bands.iter().enumerate() {
        let mut p = BlochMatrix::zeros(2 * n, 2 * n);
        let mut w = p.clone();
        for i in 0..2 * n {
            for j in 0..2 * n {
                let mut d = Complex::ZERO;
                let mut e = Complex::ZERO;
                for a in 0..b.energies.len() {
                    let c = at(&b.coefficients, i, a)
                        * at(&b.coefficients, j, a).conj()
                        * occupations.occupations[k][a];
                    d += c;
                    e += c * b.energies[a];
                }
                p.re[(i, j)] = d.re;
                p.im[(i, j)] = d.im;
                w.re[(i, j)] = e.re;
                w.im[(i, j)] = e.im;
            }
        }
        consistency = consistency.max(difference(&p, &reference.density[k]));
        let s = &backend.overlap()[k];
        let f = &fields.fock[k];
        let p = &reference.density[k];
        let mut comm = matmul(&matmul(f, p)?, s)?;
        comm.add_scaled(&matmul(&matmul(s, p)?, f)?, -1.);
        let orth = Orthogonalizer::canonical(&s.real_embedding(), config.overlap_cutoff)?;
        let raw = orth
            .x
            .transpose_matmul(&comm.real_embedding().matmul(&orth.x));
        orbital = orbital.max(raw.as_slice().iter().map(|v| v.abs()).fold(0., f64::max));
        weighted.push(pauli_components(&w)?[0].clone());
    }
    let number = reference
        .density
        .iter()
        .zip(backend.overlap())
        .map(|(p, s)| p.trace_product(s))
        .sum::<f64>()
        / nk as f64;
    let number_error = (number - backend.electrons()).abs();
    if consistency > config.density_tolerance
        || number_error > config.density_tolerance
        || orbital > config.orbital_tolerance
    {
        return Err(DftError::ScfNotConverged {
            iterations: reference.iterations,
            error: consistency.max(number_error).max(orbital),
        });
    }
    let nonlocal_energy = fields.periodic_vv10_energy(frame.options.scf.vv10.is_some());
    Ok(AcceptedReference {
        weighted,
        orbital,
        consistency,
        number_error,
        energy_defect,
        fock_defect,
        xc_energy: fields.energy_components[7] - nonlocal_energy.unwrap_or(0.0),
        nonlocal_energy,
        angular_order: fields
            .xc_diagnostics
            .as_ref()
            .map_or(4, |d| d.angular_order),
    })
}
