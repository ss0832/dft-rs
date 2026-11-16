// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Thermal fixed points require an occupation residual even when F and P commute.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::periodic::occupations::{OccupationScheme, BOLTZMANN_HARTREE_PER_K};
use dft_rs::periodic::spinor_scf::*;
use dft_rs::periodic::BlochMatrix;
use dft_rs::Result;

struct CommutingThermal {
    overlap: Vec<BlochMatrix>,
    initial: Vec<BlochMatrix>,
    field: f64,
}
impl CommutingThermal {
    fn new(field: f64) -> Self {
        let mut s = BlochMatrix::zeros(2, 2);
        s.re[(0, 0)] = 1.0;
        s.re[(1, 1)] = 1.0;
        let mut f = BlochMatrix::zeros(2, 2);
        f.re[(0, 0)] = field;
        f.re[(1, 1)] = -field;
        Self {
            overlap: vec![s],
            initial: vec![f],
            field,
        }
    }
}
impl SpinorEnergyBackend for CommutingThermal {
    fn overlap(&self) -> &[BlochMatrix] {
        &self.overlap
    }
    fn initial_fock(&self) -> &[BlochMatrix] {
        &self.initial
    }
    fn electrons(&self) -> f64 {
        1.0
    }
    fn evaluate(&self, density: &[BlochMatrix]) -> Result<SpinorFields> {
        let m = density[0].re[(0, 0)] - density[0].re[(1, 1)];
        let mut f = self.initial[0].clone();
        f.re[(0, 0)] += 1.2 * m;
        f.re[(1, 1)] -= 1.2 * m;
        let energy = 0.6 * m * m + self.field * m;
        let mut components = [0.0; 10];
        components[9] = energy;
        Ok(SpinorFields {
            fock: vec![f],
            energy,
            energy_components: components,
            xc_component_energies: Vec::new(),
            nuclear_soc_energy: 0.0,
            operator_converged: true,
            coulomb_operator_defect: 0.0,
            xc_grid_number_defect: 0.0,
            xc_diagnostics: None,
        })
    }
}
fn oracle(field: f64) -> (f64, f64) {
    let mut lower = -1.0;
    let mut upper = 1.0;
    for _ in 0..100 {
        let m = 0.5 * (lower + upper);
        let residual = m + ((1.2 * m + field) / 0.8).tanh();
        if residual > 0.0 {
            upper = m;
        } else {
            lower = m;
        }
    }
    let m = 0.5 * (lower + upper);
    let occupations = [0.5 * (1.0 + m), 0.5 * (1.0 - m)];
    let entropy = -0.4
        * occupations
            .iter()
            .map(|f| f * f.ln() + (1.0 - f) * (1.0 - f).ln())
            .sum::<f64>();
    (m, 0.6 * m * m + field * m - entropy)
}

#[test]
fn commuting_thermal_scf_matches_closed_oracle_and_four_step_field_fit() {
    let config = SpinorScfConfig {
        max_iter: 40,
        energy_tolerance: 1e-14,
        density_tolerance: 1e-13,
        orbital_tolerance: 1e-12,
        occupation: OccupationScheme::FermiDirac {
            temperature_kelvin: 0.4 / BOLTZMANN_HARTREE_PER_K,
        },
        ..SpinorScfConfig::default()
    };
    let calculate = |field: f64| {
        let r = run_spinor_scf(&CommutingThermal::new(field), &config).unwrap();
        assert!(r.converged && r.iterations < 40);
        assert!(r.occupation_consistency < 1e-13);
        assert!(r.particle_number_error < 1e-13);
        r
    };
    let field = 0.23;
    let reference = calculate(field);
    let (magnetization, energy) = oracle(field);
    assert!((reference.free_energy - energy).abs() < 1e-12);
    assert!(
        (reference.density[0].re[(0, 0)] - reference.density[0].re[(1, 1)] - magnetization).abs()
            < 1e-12
    );
    let steps = [0.04, 0.02, 0.01, 0.005];
    let mut errors = Vec::new();
    let mut estimates = Vec::new();
    for h in steps {
        let estimate =
            (calculate(field + h).free_energy - calculate(field - h).free_energy) / (2.0 * h);
        errors.push((estimate - magnetization).abs());
        estimates.push(estimate);
    }
    let fit = convergence::error_power_fit(&steps, &errors);
    let rich = ((4.0 * estimates[3] - estimates[2]) / 3.0 - magnetization).abs();
    println!(
        "commuting thermal spinor: iterations={} steps={steps:?} \
        errors={errors:?} p={:.9} R2={:.10} Richardson={rich:.3e} \
        orbital={:.3e} occupation={:.3e} number={:.3e}",
        reference.iterations,
        fit.exponent,
        fit.r_squared,
        reference.orbital_residual,
        reference.occupation_consistency,
        reference.particle_number_error
    );
    assert!((fit.exponent - 2.0).abs() < 0.02 && fit.r_squared > 0.99999);
    assert!(rich < 1e-8);
}
