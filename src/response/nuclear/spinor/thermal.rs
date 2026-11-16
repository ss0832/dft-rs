// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Canonical finite-temperature response in a transported spinor frame.

use super::{cm, Cm, ComplexPathBackend, SpinorGeometry, SpinorNuclearEnergyTaylor};
use crate::linalg::Matrix;
use crate::periodic::{spinor_scf::SpinorScfReport, BlochMatrix};
use crate::response::complex::Complex;
use crate::response::field_path::{adjoint, cadd, series_product, transport};
use crate::response::krylov::{self, LinearOperator, OperatorStructure, ResponseSolverConfig};
use crate::response::thermal_matrix::fermi_matrix_taylor;
use crate::{DftError, Result};

struct CanonicalKernel<'a> {
    geometry: &'a SpinorGeometry<'a>,
    coefficients: &'a Cm,
    weights: Matrix,
    trace_weight: f64,
}

fn trace(p: &Cm) -> f64 {
    (0..p.re.rows).map(|i| p.re[(i, i)]).sum()
}

impl CanonicalKernel<'_> {
    fn induced(&self, density: &Cm) -> Result<Cm> {
        let c = self.coefficients;
        let ao = c.matmul_seq(density).matmul_seq(&adjoint(c));
        let empty = Cm::zeros(ao.re.rows, ao.re.cols);
        let [potential, _] = self.geometry.linear_kernel(&[ao, empty])?;
        let mut mo = adjoint(c).matmul_seq(&potential).matmul_seq(c);
        for ((re, im), w) in mo
            .re
            .as_mut_slice()
            .iter_mut()
            .zip(mo.im.as_mut_slice())
            .zip(self.weights.as_slice())
        {
            *re *= *w;
            *im *= *w;
        }
        Ok(mo)
    }

    fn project_number(&self, mut density: Cm) -> (Cm, f64) {
        let chemical = trace(&density) / self.trace_weight;
        for i in 0..density.re.rows {
            density.re[(i, i)] -= chemical * self.weights[(i, i)];
        }
        (density, chemical)
    }
}

fn pack(p: &Cm) -> Vec<f64> {
    let n = p.re.rows;
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        out.push(p.re[(i, i)]);
        for j in 0..i {
            out.extend([p.re[(i, j)], p.im[(i, j)]]);
        }
    }
    out
}

fn unpack(n: usize, mut values: impl Iterator<Item = f64>) -> Cm {
    let mut result = Cm::zeros(n, n);
    for i in 0..n {
        result.re[(i, i)] = values.next().expect("packed diagonal");
        for j in 0..i {
            let re = values.next().expect("packed real part");
            let im = values.next().expect("packed imaginary part");
            result.re[(i, j)] = re;
            result.re[(j, i)] = re;
            result.im[(i, j)] = im;
            result.im[(j, i)] = -im;
        }
    }
    result
}

impl LinearOperator for CanonicalKernel<'_> {
    fn dimension(&self) -> usize {
        self.weights.rows * self.weights.cols
    }
    fn structure(&self) -> OperatorStructure {
        OperatorStructure::General
    }
    fn apply(&self, x: &[Complex]) -> Result<Vec<Complex>> {
        krylov::apply_real_linear(x, |v| {
            let n = self.weights.rows;
            let mut density = unpack(n, v.iter().copied());
            let (induced, _) = self.project_number(self.induced(&density)?);
            cadd(&mut density, &induced, -1.0);
            Ok(pack(&density))
        })
    }
}

fn matrix_fermi(
    fock: &[Cm],
    chemical: &[f64],
    kt: f64,
    config: &ResponseSolverConfig,
) -> Result<Vec<BlochMatrix>> {
    let n = fock[0].re.rows;
    let shifted = fock
        .iter()
        .zip(chemical)
        .map(|(f, mu)| {
            // The physical Fock is Hermitian. Average both computed triangles
            // before the matrix expansion so roundoff cannot break Hermiticity.
            let mut x = f.clone();
            cadd(&mut x, &adjoint(f), 1.0);
            for value in x.re.as_mut_slice().iter_mut().chain(x.im.as_mut_slice()) {
                *value *= 0.5;
            }
            for i in 0..n {
                x.re[(i, i)] -= mu;
            }
            BlochMatrix::from_pair((x.re, x.im))
        })
        .collect::<Vec<_>>();
    let metric = BlochMatrix::from_pair((Matrix::identity(n), Matrix::zeros(n, n)));
    Ok(fermi_matrix_taylor(&shifted, &[metric], &[kt], fock.len() - 1, 0.5, config)?.density)
}

pub(super) fn solve(
    backend: &SpinorGeometry<'_>,
    reference: &SpinorScfReport,
    kt: f64,
    order: usize,
    config: &ResponseSolverConfig,
) -> Result<SpinorNuclearEnergyTaylor> {
    let mu = reference.chemical_potential.ok_or_else(|| {
        DftError::InvalidInput("thermal spinor response requires a finite Fermi level".into())
    })?;
    let energy = &reference.bands[0].energies;
    let f = &reference.occupations[0];
    let n = f.len();
    let mut weights = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let gap = energy[i] - energy[j];
            weights[(i, j)] = if gap.abs() > 1e-8 * kt {
                (f[i] - f[j]) / gap
            } else {
                let occupation = 0.5 * (f[i] + f[j]);
                -occupation * (1.0 - occupation) / kt
            };
        }
    }
    let trace_weight: f64 = (0..n).map(|i| weights[(i, i)]).sum();
    if !trace_weight.is_finite() || trace_weight == 0.0 {
        return Err(DftError::Unsupported {
            feature: "thermal spinor response with unresolved Fermi population weights".into(),
            missing: "a resolvable derivative of the total electron count with respect to its chemical potential".into(),
        });
    }
    let c = cm(&reference.bands[0].coefficients);
    let kernel = CanonicalKernel {
        geometry: backend,
        coefficients: &c,
        weights,
        trace_weight,
    };
    let frame = transport(&c, backend.overlap(), order);
    let adjoints = frame.iter().map(adjoint).collect::<Vec<_>>();
    let reference_density = cm(&reference.density[0]);
    let ao_density = |density: &[Cm], width: usize| {
        let mut ao = series_product(&series_product(&frame, density, width), &adjoints, width);
        // Bind the analytic grid kernel to the same AO reference density used
        // by SCF. Reconstructing it from eigenvectors would introduce rounding
        // differences and bypass the reference-bound descriptor cache.
        ao[0] = reference_density.clone();
        let dim = frame[0].re.rows;
        [ao, vec![Cm::zeros(dim, dim); width + 1]]
    };
    let fock_series = |density: &[Cm], width: usize| -> Result<Vec<Cm>> {
        let p = ao_density(density, width);
        let mut potential = Vec::with_capacity(width + 1);
        for i in 0..=width {
            let [f, _] = backend.fock_coefficient(&p, i)?;
            potential.push(f);
        }
        Ok(series_product(
            &series_product(&adjoints, &potential, width),
            &frame,
            width,
        ))
    };
    let mut density = vec![Cm::zeros(n, n); order + 1];
    for (i, &occupation) in f.iter().enumerate() {
        density[0].re[(i, i)] = occupation;
    }
    let mut chemical = vec![0.0; order + 1];
    chemical[0] = mu;
    let mut residuals = Vec::with_capacity(order);
    let reference_fock = fock_series(&density, 0)?;
    let actual = adjoint(&c)
        .matmul_seq(&cm(&reference.fields.fock[0]))
        .matmul_seq(&c);
    let reference_defect = actual
        .re
        .max_abs_diff(&reference_fock[0].re)
        .max(actual.im.max_abs_diff(&reference_fock[0].im));
    // The stationary fixed-number Hessian needs only first-order density.
    // Its absent second-order density contraction cancels between E and -TS.
    let solved_order = if order <= 2 { order - 1 } else { order };
    for k in 1..=solved_order {
        let potential = fock_series(&density, k)?;
        let fermi = matrix_fermi(&potential, &chemical[..=k], kt, config)?;
        let bare = cm(&fermi[k]);
        let (source, _) = kernel.project_number(bare.clone());
        let rhs: Vec<_> = pack(&source)
            .into_iter()
            .map(|v| Complex::new(v, 0.0))
            .collect();
        let result = krylov::solve(&kernel, &rhs, None, config)?;
        density[k] = unpack(n, result.amplitudes.iter().map(|v| v.re));
        let mut unprojected = bare;
        cadd(&mut unprojected, &kernel.induced(&density[k])?, 1.0);
        chemical[k] = trace(&unprojected) / trace_weight;
        residuals.push(result.relative_residual);
    }
    let potential = fock_series(&density, solved_order)?;
    let predicted = matrix_fermi(&potential, &chemical[..=solved_order], kt, config)?;
    let stationarity: Vec<_> = predicted
        .iter()
        .zip(&density)
        .map(|(p, d)| p.re.max_abs_diff(&d.re).max(p.im.max_abs_diff(&d.im)))
        .collect();
    let particle_defects = density[..=solved_order]
        .iter()
        .skip(1)
        .map(|d| trace(d).abs())
        .collect::<Vec<_>>();
    let max_defect = stationarity
        .iter()
        .chain(&particle_defects)
        .copied()
        .fold(0.0, f64::max);
    if max_defect > config.acceptance() {
        return Err(DftError::ResponseNotConverged {
            iterations: solved_order,
            residual: max_defect,
        });
    }
    let mut a = backend.energy(&ao_density(&density, order), order)?;
    a[0] -= reference.entropy_term;
    // Integrate d(-T S) = -Tr[(F-mu) dD] in the fixed orthonormal frame.
    for k in 1..=order {
        for j in 0..k.min(potential.len()) {
            let w = (k - j) as f64 / k as f64;
            a[k] -= w
                * (potential[j].real_trace_product(&density[k - j])
                    - chemical[j] * trace(&density[k - j]));
        }
    }
    Ok(SpinorNuclearEnergyTaylor {
        reference_energy_defect: a[0] - reference.free_energy,
        energy: a,
        solved_order,
        residuals,
        reference_defect,
        projector_defects: Vec::new(),
        stationarity_defects: stationarity,
        thermal_number_defects: particle_defects,
        thermal_chemical_potentials: chemical[..=solved_order].to_vec(),
        angular_refinement_history: backend.angular_history.borrow().clone(),
        angular_cache_bound_history: backend
            .cache
            .as_ref()
            .map_or_else(Vec::new, |c| c.angular_history.clone()),
    })
}
