// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Ideal-gas rigid-rotor/harmonic-oscillator thermochemistry.
//!
//! Energies are Hartree per molecule and entropies are Hartree/K per
//! molecule, matching the rest of the public Rust API. Translational and
//! rotational partition functions use SI internally so pressure and moments
//! of inertia carry unambiguous units.

use crate::constants::{atomic_mass_au, AMU_TO_AU_MASS, HARTREE_TO_CM1, HARTREE_TO_KJ_MOL};
use crate::error::{DftError, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::system::Molecule;

const BOLTZMANN_J_K: f64 = 1.380_649e-23;
const PLANCK_J_S: f64 = 6.626_070_15e-34;
const AMU_KG: f64 = 1.660_539_066_60e-27;
const BOHR_M: f64 = 5.291_772_109_03e-11;
const GAS_CONSTANT_J_MOL_K: f64 = 8.314_462_618_153_24;

/// `k_B` in Hartree/K (equivalently `R` in Hartree mol⁻¹ K⁻¹).
pub const BOLTZMANN_HARTREE_K: f64 = GAS_CONSTANT_J_MOL_K / (1000.0 * HARTREE_TO_KJ_MOL);

#[derive(Clone, Copy, Debug)]
/// Ideal-gas RRHO standard-state controls.
pub struct ThermochemistryOptions {
    /// Temperature in kelvin.
    pub temperature_kelvin: f64,
    /// Ideal-gas standard pressure in pascal.
    pub pressure_pascal: f64,
    /// External rotational symmetry number.
    pub symmetry_number: usize,
    /// Electronic-state degeneracy, normally the spin multiplicity.
    pub electronic_degeneracy: usize,
}

impl Default for ThermochemistryOptions {
    fn default() -> Self {
        Self {
            temperature_kelvin: 298.15,
            pressure_pascal: 101_325.0,
            symmetry_number: 1,
            electronic_degeneracy: 1,
        }
    }
}

#[derive(Clone, Debug)]
/// Ideal-gas rigid-rotor/harmonic-oscillator corrections and totals.
///
/// Every energy is in Hartree per molecule. Entropies are Hartree/K per
/// molecule; temperature and pressure retain their input units.
pub struct Thermochemistry {
    /// Temperature in kelvin.
    pub temperature_kelvin: f64,
    /// Ideal-gas standard pressure in pascal.
    pub pressure_pascal: f64,
    /// External rotational symmetry number used in the rotational entropy.
    pub symmetry_number: usize,
    /// Electronic-state degeneracy used in the electronic entropy.
    pub electronic_degeneracy: usize,
    /// True for a rotor with one vanishing principal moment. Atoms are false.
    pub is_linear: bool,
    /// Harmonic zero-point vibrational energy.
    pub zero_point_energy: f64,
    /// Thermal vibrational energy above the ZPE.
    pub vibrational_thermal_energy: f64,
    /// Translational contribution to the internal energy.
    pub translational_energy: f64,
    /// Rotational contribution to the internal energy.
    pub rotational_energy: f64,
    /// `U(T) - E_electronic`, including ZPE.
    pub internal_energy_correction: f64,
    /// `H(T) - E_electronic`, including ZPE and the ideal-gas `pV` term.
    pub enthalpy_correction: f64,
    /// Translational entropy.
    pub entropy_translational: f64,
    /// Rotational entropy.
    pub entropy_rotational: f64,
    /// Harmonic vibrational entropy.
    pub entropy_vibrational: f64,
    /// Electronic entropy from the configured degeneracy.
    pub entropy_electronic: f64,
    /// Sum of translational, rotational, vibrational, and electronic entropy.
    pub entropy: f64,
    /// `G(T) - E_electronic = H_correction - T S`.
    pub gibbs_correction: f64,
    /// `E_electronic + internal_energy_correction`.
    pub internal_energy: f64,
    /// `E_electronic + enthalpy_correction`.
    pub enthalpy: f64,
    /// `E_electronic + gibbs_correction`.
    pub gibbs_free_energy: f64,
}

#[derive(Clone, Copy, Debug)]
enum Rotor {
    Atom,
    Linear(f64),
    Nonlinear([f64; 3]),
}

fn rotor(mol: &Molecule) -> Result<Rotor> {
    if mol.is_empty() {
        return Err(DftError::InvalidInput(
            "thermochemistry requires at least one atom".into(),
        ));
    }
    let mut total_mass_amu = 0.0;
    let mut com = Vec3::zero();
    for (index, atom) in mol.atoms.iter().enumerate() {
        let mass_au = atomic_mass_au(atom.z);
        if !mass_au.is_finite() || mass_au <= 0.0 {
            return Err(DftError::InvalidInput(format!(
                "thermochemistry: centre {index} has no positive tabulated mass"
            )));
        }
        let mass_amu = mass_au / AMU_TO_AU_MASS;
        total_mass_amu += mass_amu;
        com += atom.position * mass_amu;
    }
    com = com / total_mass_amu;

    let mut inertia = Matrix::zeros(3, 3);
    for atom in &mol.atoms {
        let mass = atomic_mass_au(atom.z) / AMU_TO_AU_MASS * AMU_KG;
        let r = (atom.position - com) * BOHR_M;
        inertia[(0, 0)] += mass * (r.y * r.y + r.z * r.z);
        inertia[(1, 1)] += mass * (r.x * r.x + r.z * r.z);
        inertia[(2, 2)] += mass * (r.x * r.x + r.y * r.y);
        inertia[(0, 1)] -= mass * r.x * r.y;
        inertia[(1, 0)] = inertia[(0, 1)];
        inertia[(0, 2)] -= mass * r.x * r.z;
        inertia[(2, 0)] = inertia[(0, 2)];
        inertia[(1, 2)] -= mass * r.y * r.z;
        inertia[(2, 1)] = inertia[(1, 2)];
    }
    let (mut moments, _) = symmetric_eigen(&inertia)?;
    moments.sort_by(f64::total_cmp);
    for value in &mut moments {
        *value = value.max(0.0);
    }
    let largest = moments[2];
    if largest <= 1e-50 {
        return Ok(Rotor::Atom);
    }
    if moments[0] / largest < 1e-8 {
        // The two perpendicular moments are equal for an exact linear rotor;
        // their geometric mean avoids choosing an arbitrary transverse axis.
        return Ok(Rotor::Linear((moments[1] * moments[2]).sqrt()));
    }
    Ok(Rotor::Nonlinear([moments[0], moments[1], moments[2]]))
}

/// RRHO thermochemistry from Eckart-projected harmonic frequencies.
///
/// Imaginary or zero frequencies are refused: a saddle needs an explicitly
/// chosen transition-state convention rather than silently treating its
/// unstable coordinate as a harmonic oscillator.
pub fn thermochemistry(
    mol: &Molecule,
    frequencies_cm: &[f64],
    electronic_energy: f64,
    options: ThermochemistryOptions,
) -> Result<Thermochemistry> {
    let t = options.temperature_kelvin;
    let p = options.pressure_pascal;
    if !(t.is_finite() && t > 0.0) {
        return Err(DftError::InvalidInput(
            "thermochemistry temperature must be finite and positive".into(),
        ));
    }
    if !(p.is_finite() && p > 0.0) {
        return Err(DftError::InvalidInput(
            "thermochemistry pressure must be finite and positive".into(),
        ));
    }
    if options.symmetry_number == 0 {
        return Err(DftError::InvalidInput(
            "rotational symmetry number must be at least one".into(),
        ));
    }
    if options.electronic_degeneracy == 0 {
        return Err(DftError::InvalidInput(
            "electronic degeneracy must be at least one".into(),
        ));
    }
    if !electronic_energy.is_finite() {
        return Err(DftError::InvalidInput(
            "electronic energy must be finite".into(),
        ));
    }

    let mut total_mass_kg = 0.0;
    for (index, atom) in mol.atoms.iter().enumerate() {
        let mass_au = atomic_mass_au(atom.z);
        if !mass_au.is_finite() || mass_au <= 0.0 {
            return Err(DftError::InvalidInput(format!(
                "thermochemistry: centre {index} has no positive tabulated mass"
            )));
        }
        total_mass_kg += mass_au / AMU_TO_AU_MASS * AMU_KG;
    }

    let rt = BOLTZMANN_HARTREE_K * t;
    let q_trans = (2.0 * std::f64::consts::PI * total_mass_kg * BOLTZMANN_J_K * t
        / (PLANCK_J_S * PLANCK_J_S))
        .powf(1.5)
        * (BOLTZMANN_J_K * t / p);
    let entropy_translational = BOLTZMANN_HARTREE_K * (q_trans.ln() + 2.5);
    let translational_energy = 1.5 * rt;

    let rotor = rotor(mol)?;
    let (is_linear, rotational_energy, entropy_rotational) = match rotor {
        Rotor::Atom => (false, 0.0, 0.0),
        Rotor::Linear(moment) => {
            let q = 8.0 * std::f64::consts::PI.powi(2) * moment * BOLTZMANN_J_K * t
                / (options.symmetry_number as f64 * PLANCK_J_S.powi(2));
            (true, rt, BOLTZMANN_HARTREE_K * (q.ln() + 1.0))
        }
        Rotor::Nonlinear(moments) => {
            let q = std::f64::consts::PI.sqrt() / options.symmetry_number as f64
                * (8.0 * std::f64::consts::PI.powi(2) * BOLTZMANN_J_K * t / PLANCK_J_S.powi(2))
                    .powf(1.5)
                * (moments[0] * moments[1] * moments[2]).sqrt();
            (false, 1.5 * rt, BOLTZMANN_HARTREE_K * (q.ln() + 1.5))
        }
    };

    let mut zero_point_energy = 0.0;
    let mut vibrational_thermal_energy = 0.0;
    let mut entropy_vibrational = 0.0;
    for (mode, &frequency) in frequencies_cm.iter().enumerate() {
        if !(frequency.is_finite() && frequency > 0.0) {
            return Err(DftError::InvalidInput(format!(
                "thermochemistry requires positive real frequencies; mode {mode} is {frequency} cm^-1"
            )));
        }
        let quantum = frequency / HARTREE_TO_CM1;
        let x = quantum / rt;
        zero_point_energy += 0.5 * quantum;
        vibrational_thermal_energy += quantum / x.exp_m1();
        entropy_vibrational += BOLTZMANN_HARTREE_K * (x / x.exp_m1() - (-(-x).exp()).ln_1p());
    }
    let entropy_electronic = BOLTZMANN_HARTREE_K * (options.electronic_degeneracy as f64).ln();
    let entropy =
        entropy_translational + entropy_rotational + entropy_vibrational + entropy_electronic;
    let internal_energy_correction =
        zero_point_energy + vibrational_thermal_energy + translational_energy + rotational_energy;
    // Ideal-gas H = U + pV = U + RT per molecule.
    let enthalpy_correction = internal_energy_correction + rt;
    let gibbs_correction = enthalpy_correction - t * entropy;

    Ok(Thermochemistry {
        temperature_kelvin: t,
        pressure_pascal: p,
        symmetry_number: options.symmetry_number,
        electronic_degeneracy: options.electronic_degeneracy,
        is_linear,
        zero_point_energy,
        vibrational_thermal_energy,
        translational_energy,
        rotational_energy,
        internal_energy_correction,
        enthalpy_correction,
        entropy_translational,
        entropy_rotational,
        entropy_vibrational,
        entropy_electronic,
        entropy,
        gibbs_correction,
        internal_energy: electronic_energy + internal_energy_correction,
        enthalpy: electronic_energy + enthalpy_correction,
        gibbs_free_energy: electronic_energy + gibbs_correction,
    })
}

/// Equilibrium ideal-gas RRHO thermochemistry on a Mermin electronic surface.
/// `mermin_energy` is A_e in Hartree and `electronic_entropy` is S_e/k_B.
/// Positive electronic and nuclear temperatures must agree; otherwise a
/// single equilibrium Gibbs potential is not defined. At zero electronic
/// temperature this is the ground-electronic-state RRHO approximation.
/// `electronic_degeneracy` counts additional states absent from the supplied
/// electronic ensemble; use one when that ensemble already includes them.
pub fn thermochemistry_from_mermin(
    mol: &Molecule,
    frequencies_cm: &[f64],
    mermin_energy: f64,
    electronic_temperature_kelvin: f64,
    electronic_entropy: f64,
    options: ThermochemistryOptions,
) -> Result<Thermochemistry> {
    let te = electronic_temperature_kelvin;
    if !te.is_finite()
        || te < 0.0
        || !electronic_entropy.is_finite()
        || electronic_entropy < 0.0
        || (te == 0.0 && electronic_entropy != 0.0)
    {
        return Err(DftError::InvalidInput("Mermin thermochemistry requires a nonnegative finite electronic temperature/entropy, with zero entropy at zero electronic temperature".into()));
    }
    if te > 0.0 && (te - options.temperature_kelvin).abs() > 1e-10 * te {
        return Err(DftError::InvalidInput("equilibrium Gibbs thermochemistry requires equal electronic and nuclear temperatures; set temperature=electronic_temperature".into()));
    }
    // Use the same k_B as the electronic occupations for the energy identity.
    let entropy_hartree_per_k =
        electronic_entropy * crate::periodic::occupations::BOLTZMANN_HARTREE_PER_K;
    let electronic_internal = mermin_energy + te * entropy_hartree_per_k;
    let mut result = thermochemistry(mol, frequencies_cm, electronic_internal, options)?;
    result.entropy_electronic += entropy_hartree_per_k;
    result.entropy += entropy_hartree_per_k;
    result.gibbs_correction -= options.temperature_kelvin * entropy_hartree_per_k;
    result.gibbs_free_energy = result.enthalpy - options.temperature_kelvin * result.entropy;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::Atom;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a:.16e} vs {b:.16e}");
    }

    #[test]
    fn harmonic_mode_matches_closed_form() {
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        let opts = ThermochemistryOptions::default();
        let got = thermochemistry(&mol, &[1000.0], -0.5, opts).unwrap();
        let quantum = 1000.0 / HARTREE_TO_CM1;
        let x = quantum / (BOLTZMANN_HARTREE_K * opts.temperature_kelvin);
        approx(got.zero_point_energy, 0.5 * quantum, 1e-16);
        approx(got.vibrational_thermal_energy, quantum / x.exp_m1(), 1e-16);
    }

    #[test]
    fn pressure_changes_entropy_and_gibbs_by_the_ideal_gas_identity() {
        let mol = Molecule::new(vec![Atom {
            z: 4,
            position: Vec3::zero(),
        }]);
        let low = ThermochemistryOptions::default();
        let high = ThermochemistryOptions {
            pressure_pascal: 10.0 * low.pressure_pascal,
            ..low
        };
        let a = thermochemistry(&mol, &[], 0.0, low).unwrap();
        let b = thermochemistry(&mol, &[], 0.0, high).unwrap();
        let want_ds = -BOLTZMANN_HARTREE_K * 10.0_f64.ln();
        approx(b.entropy - a.entropy, want_ds, 1e-16);
        approx(
            b.gibbs_free_energy - a.gibbs_free_energy,
            -low.temperature_kelvin * want_ds,
            1e-15,
        );
    }

    #[test]
    fn linear_and_nonlinear_rotors_have_the_right_equipartition() {
        let h2 = Molecule::new(vec![
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, -0.7),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 0.0, 0.7),
            },
        ]);
        let water = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::zero(),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.43, 1.11),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.43, 1.11),
            },
        ]);
        let opts = ThermochemistryOptions::default();
        let linear = thermochemistry(&h2, &[4400.0], 0.0, opts).unwrap();
        let nonlinear = thermochemistry(&water, &[1600.0, 3700.0, 3800.0], 0.0, opts).unwrap();
        assert!(linear.is_linear);
        assert!(!nonlinear.is_linear);
        approx(
            linear.rotational_energy,
            BOLTZMANN_HARTREE_K * opts.temperature_kelvin,
            1e-16,
        );
        approx(
            nonlinear.rotational_energy,
            1.5 * BOLTZMANN_HARTREE_K * opts.temperature_kelvin,
            1e-16,
        );
    }

    #[test]
    fn imaginary_modes_are_not_silently_thermalized() {
        let mol = Molecule::new(vec![Atom {
            z: 1,
            position: Vec3::zero(),
        }]);
        let err =
            thermochemistry(&mol, &[-100.0], 0.0, ThermochemistryOptions::default()).unwrap_err();
        assert!(matches!(err, DftError::InvalidInput(_)));
    }
}
