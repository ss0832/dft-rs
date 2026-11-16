// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
//! Fixed-AO, integer-population, analytic mixed static response.
//!
//! All derivatives below are unnormalized derivatives, not Taylor coefficients.
//! The semilocal provider alone uses factorial-normalized Taylor coefficients.

use super::*;
use crate::response::nonlinear::semilocal_fock_coefficient;

/// Clamped-geometry third energy derivative, with every Cartesian entry computed.
#[derive(Clone, Debug)]
pub struct PolarizabilityParameterResponse {
    /// `[parameter][dipole Cartesian component][field Cartesian component]`.
    pub alpha_q: Vec<[[f64; 3]; 3]>,
    pub parameter_names: Vec<String>,
    pub parameter_units: Vec<String>,
    pub axis_order: &'static str,
    pub units: &'static str,
    /// First solves: all parameters, then x/y/z field. Mixed solves: q-major xyz.
    pub solver_relative_residuals: Vec<f64>,
    pub projector_defects: Vec<f64>,
    pub stationarity_defects: Vec<f64>,
    pub relative_stationarity_defects: Vec<f64>,
    pub particle_number_derivatives: Vec<[f64; 2]>,
    pub reciprocity_defect: f64,
    pub converged: bool,
}

fn accumulate(a: &mut Matrix, b: &Matrix, scale: f64) {
    for (x, y) in a.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *x += scale * y;
    }
}

fn commutator(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.matmul_seq(b);
    accumulate(&mut out, &b.matmul_seq(a), -1.0);
    out
}

fn max_abs(a: &Matrix) -> f64 {
    a.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn ao(c: &Matrix, d: &Matrix) -> Matrix {
    c.matmul_seq(d).matmul_seq(&c.transpose())
}

fn mo(c: &Matrix, f: &Matrix) -> Matrix {
    c.congruence_accurate(f)
}

// Density coefficients transform contravariantly: C^T P_AO C is NOT P_MO
// in a nonorthogonal AO basis. Construct OV coefficients directly instead.
fn orbital_density(
    coefficients: &[Matrix; 2],
    occupied: [usize; 2],
    spins: usize,
    u: &[f64],
) -> [Matrix; 2] {
    let mut d = std::array::from_fn(|s| Matrix::zeros(coefficients[s].cols, coefficients[s].cols));
    let mut offset = 0;
    for s in 0..spins {
        for a in occupied[s]..coefficients[s].cols {
            for i in 0..occupied[s] {
                d[s][(a, i)] = u[offset];
                d[s][(i, a)] = u[offset];
                offset += 1;
            }
        }
    }
    if spins == 1 {
        d[1] = d[0].clone();
    }
    d
}

impl ContinuousSpeciesModel {
    /// Analytic alpha_q for this affine charge/local-amplitude chart.
    ///
    /// AO, quadrature, geometry, widths and integer spin populations stay fixed.
    /// No field-dependent internal frozen-core modes are included. Classical
    /// affine permanent dipoles and field-independent steric terms have exactly
    /// zero second field derivative, hence contribute no alpha_q term.
    pub fn polarizability_parameter_response(
        &self,
        state: &SpeciesState,
        config: &ResponseSolverConfig,
    ) -> Result<PolarizabilityParameterResponse> {
        self.check_state(state)?;
        config.validate()?;
        if !state.scf.converged {
            return invalid("mixed parameter response requires a converged stationary reference");
        }
        if self.static_core.is_some() {
            return unsupported(
                "generated-core parameter response requires an explicit operator chart",
            );
        }
        if state.scf.fixed_occupation_residual.is_some() || state.scf.electronic_temperature != 0.0
        {
            return unsupported(
                "mixed parameter response requires integer zero-temperature populations",
            );
        }
        // Model construction restricts the functional to the verified LDA/PBE
        // choices. Its existing option validation excludes other auxiliary XC.
        let mut options = self.options.clone();
        options.external_field = Some(state.field);
        let solver = ResponseSolver::new(
            &self.molecule,
            &self.basis,
            &self.functional,
            Some(&self.grid),
            &state.scf,
            &options,
        )?;
        let (occupied, virtuals) = solver.ov_dims();
        let spins = if state.scf.unrestricted { 2 } else { 1 };
        let dimension = (0..spins).map(|s| occupied[s] * virtuals[s]).sum();
        let solve = |rhs: &[f64]| -> Result<(Vec<f64>, f64)> {
            let report =
                solver.solve_ov_report_with_structure(rhs, config, OperatorStructure::Hermitian)?;
            if report
                .amplitudes
                .iter()
                .any(|x| !x.re.is_finite() || x.im.abs() > config.acceptance())
            {
                return invalid("mixed static response has non-real or non-finite amplitudes");
            }
            Ok((
                report.amplitudes.iter().map(|x| x.re).collect(),
                report.relative_residual,
            ))
        };
        let nq = self.parameter_operators.len();
        let mut residuals = Vec::new();
        let mut first_p = Vec::new();
        let mut first_d = Vec::new();
        let mut first_f = Vec::new();
        for h in self.parameter_operators.iter().chain(self.dipoles.iter()) {
            let mut rhs = vec![0.0; dimension];
            solver.project_ov([h, h], &mut rhs);
            let (u, residual) = solve(&rhs)?;
            residuals.push(residual);
            let p = solver.delta_density(&u);
            let mut f = solver.fock_response(&p)?;
            for fs in &mut f {
                accumulate(fs, h, 1.0);
            }
            first_d.push(orbital_density(&state.scf.mo_coeff, occupied, spins, &u));
            first_f.push(std::array::from_fn::<_, 2, _>(|s| {
                mo(&state.scf.mo_coeff[s], &f[s])
            }));
            first_p.push(p);
        }
        let d0: [Matrix; 2] = std::array::from_fn(|s| {
            let mut d = Matrix::zeros(state.scf.mo_coeff[s].cols, state.scf.mo_coeff[s].cols);
            for i in 0..occupied[s] {
                d[(i, i)] = 1.0;
            }
            d
        });
        let f0: [Matrix; 2] =
            std::array::from_fn(|s| mo(&state.scf.mo_coeff[s], &state.scf.fock[s]));
        let zero =
            || std::array::from_fn(|_| Matrix::zeros(self.basis.nao_sph, self.basis.nao_sph));
        let c2 = |p: &[Matrix; 2]| {
            semilocal_fock_coefficient(
                &self.basis,
                &self.grid,
                &self.functional,
                &[state.scf.density.clone(), p.clone(), zero()],
                2,
            )
        };
        let first_c2: Vec<_> = first_p.iter().map(c2).collect::<Result<_>>()?;
        let mut alpha_q = vec![[[0.0; 3]; 3]; nq];
        let mut projector_defects = Vec::new();
        let mut stationarity_defects = Vec::new();
        let mut relative_stationarity_defects = Vec::new();
        let mut particle_number_derivatives = Vec::new();
        for a in 0..nq {
            for j in 0..3 {
                let b = nq + j;
                let mut sum_p = first_p[a].clone();
                for s in 0..2 {
                    accumulate(&mut sum_p[s], &first_p[b][s], 1.0);
                }
                // C2(A+B)-C2(A)-C2(B) = F_xc''[A,B], without a factor two.
                let mut nonlinear = c2(&sum_p)?;
                for s in 0..2 {
                    accumulate(&mut nonlinear[s], &first_c2[a][s], -1.0);
                    accumulate(&mut nonlinear[s], &first_c2[b][s], -1.0);
                }
                let mut products = d0.clone();
                let mut diagonal = d0.clone();
                for s in 0..2 {
                    products[s] = first_d[a][s].matmul_seq(&first_d[b][s]);
                    accumulate(
                        &mut products[s],
                        &first_d[b][s].matmul_seq(&first_d[a][s]),
                        1.0,
                    );
                    diagonal[s].as_mut_slice().fill(0.0);
                    for i in 0..diagonal[s].rows {
                        for k in 0..diagonal[s].cols {
                            if i < occupied[s] && k < occupied[s] {
                                diagonal[s][(i, k)] = -products[s][(i, k)];
                            }
                            if i >= occupied[s] && k >= occupied[s] {
                                diagonal[s][(i, k)] = products[s][(i, k)];
                            }
                        }
                    }
                }
                let mut p: [Matrix; 2] =
                    std::array::from_fn(|s| ao(&state.scf.mo_coeff[s], &diagonal[s]));
                let mut f = solver.fock_response(&p)?;
                for s in 0..2 {
                    accumulate(&mut f[s], &nonlinear[s], 1.0);
                }
                let cross: [Matrix; 2] = std::array::from_fn(|s| {
                    let mut c = commutator(&first_f[a][s], &first_d[b][s]);
                    accumulate(&mut c, &commutator(&first_f[b][s], &first_d[a][s]), 1.0);
                    c
                });
                let mut rhs = Vec::with_capacity(dimension);
                for s in 0..spins {
                    let fm = mo(&state.scf.mo_coeff[s], &f[s]);
                    let mut source = commutator(&f0[s], &diagonal[s]);
                    accumulate(&mut source, &cross[s], 1.0);
                    for v in 0..virtuals[s] {
                        for o in 0..occupied[s] {
                            rhs.push(source[(occupied[s] + v, o)] + fm[(occupied[s] + v, o)]);
                        }
                    }
                }
                let (u, residual) = solve(&rhs)?;
                residuals.push(residual);
                let off = solver.delta_density(&u);
                let off_mo = orbital_density(&state.scf.mo_coeff, occupied, spins, &u);
                for s in 0..2 {
                    accumulate(&mut p[s], &off[s], 1.0);
                }
                let mut fab = solver.fock_response(&p)?;
                for s in 0..2 {
                    accumulate(&mut fab[s], &nonlinear[s], 1.0);
                }
                let mut pd = 0.0_f64;
                let mut sd = 0.0_f64;
                let mut scale = 1.0_f64;
                for s in 0..2 {
                    let mut dab = diagonal[s].clone();
                    accumulate(&mut dab, &off_mo[s], 1.0);
                    let mut defect = d0[s].matmul_seq(&dab);
                    accumulate(&mut defect, &dab.matmul_seq(&d0[s]), 1.0);
                    accumulate(&mut defect, &dab, -1.0);
                    accumulate(&mut defect, &products[s], 1.0);
                    pd = pd.max(max_abs(&defect));
                    let mut stationary = commutator(&f0[s], &dab);
                    let fm = mo(&state.scf.mo_coeff[s], &fab[s]);
                    accumulate(&mut stationary, &commutator(&fm, &d0[s]), 1.0);
                    accumulate(&mut stationary, &cross[s], 1.0);
                    sd = sd.max(max_abs(&stationary));
                    scale = scale
                        .max(max_abs(&f0[s]) * max_abs(&dab) + max_abs(&fm) + max_abs(&cross[s]));
                }
                projector_defects.push(pd);
                stationarity_defects.push(sd);
                relative_stationarity_defects.push(sd / scale);
                particle_number_derivatives
                    .push(std::array::from_fn(|s| p[s].frobenius_dot(&self.overlap)));
                for i in 0..3 {
                    alpha_q[a][i][j] = -p
                        .iter()
                        .map(|ps| ps.frobenius_dot(&self.dipoles[i]))
                        .sum::<f64>();
                }
            }
        }
        let reciprocity_defect = alpha_q
            .iter()
            .flat_map(|a| (0..3).flat_map(move |i| (0..3).map(move |j| (a[i][j] - a[j][i]).abs())))
            .fold(0.0_f64, f64::max);
        if alpha_q.iter().flatten().flatten().any(|x| !x.is_finite()) {
            return invalid("non-finite mixed parameter response");
        }
        let converged = residuals
            .iter()
            .chain(&projector_defects)
            .chain(&relative_stationarity_defects)
            .chain(particle_number_derivatives.iter().flatten())
            .all(|x| x.is_finite() && x.abs() <= config.acceptance());
        Ok(PolarizabilityParameterResponse {
            alpha_q,
            parameter_names: self
                .chart
                .parameters
                .iter()
                .map(|p| p.name.clone())
                .collect(),
            parameter_units: self
                .chart
                .parameters
                .iter()
                .map(|p| p.unit.clone())
                .collect(),
            axis_order: "parameter,dipole Cartesian,field Cartesian",
            units: "bohr^3 / parameter unit",
            solver_relative_residuals: residuals,
            projector_defects,
            stationarity_defects,
            relative_stationarity_defects,
            particle_number_derivatives,
            reciprocity_defect,
            converged,
        })
    }
}
