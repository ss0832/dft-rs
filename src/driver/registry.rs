// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! The feature registry: one row per public calculation.
//!
//! Each row names the Rust entry point, the `dft-rs` CLI command, the native
//! Python function and the ASE calculator method of one physical calculation,
//! together with the physical-target IDs it serves in
//! `docs/property_targets.csv`. The Rust, CLI and Python surface manuals list
//! the same IDs in three aligned tables; a test keeps them identical, and the
//! Python test suite checks that every named function and method exists.
//!
//! `Route::Driver` rows are executed by [`crate::driver::run`] on every
//! surface. `Route::Dedicated` rows predate the common driver and keep their
//! own bindings with method-specific accepted controls.
//!
//! Each row declares a settings applicability family, checked by the existing
//! API reference audit. See `docs/conventions.md` for the family contract.
//! A declaration is an audit obligation, not evidence of numerical acceptance
//! or universal support for every reference and response solver option.

use super::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Executed through `driver::run` by the CLI, Python and ASE.
    Driver,
    /// Bound individually on each surface.
    Dedicated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Molecule,
    Periodic,
    Both,
    /// Post-processing of supplied physical data; no electronic structure.
    Data,
}

#[derive(Clone, Copy, Debug)]
pub struct FeatureSpec {
    pub id: &'static str,
    pub title: &'static str,
    pub space: Space,
    pub route: Route,
    pub rust: &'static str,
    pub cli: &'static str,
    pub python: &'static str,
    pub ase: &'static str,
    /// Physical target IDs (`Tnnn`) served by this calculation.
    pub targets: &'static [&'static str],
}

macro_rules! feature {
    ($id:literal, $title:literal, $space:ident, $route:ident, $rust:literal, $cli:literal,
     $python:literal, $ase:literal, [$($t:literal),*]) => {
        FeatureSpec {
            id: $id,
            title: $title,
            space: Space::$space,
            route: Route::$route,
            rust: $rust,
            cli: $cli,
            python: $python,
            ase: $ase,
            targets: &[$($t),*],
        }
    };
}

static FEATURES: &[FeatureSpec] = &[
    // settings-contract: frozen-model
    feature!("continuous-species", "Continuous nuclear-charge models and explicit saved-fragment, primitive and frozen scalar actions", Molecule, Driver,
        "driver::run(\"continuous-species\", ...); driver::continuous_species::ContinuousSpeciesModel", "continuous-species", "continuous_species",
        "get_continuous_species", ["T171"]),
    // settings-contract: multicomponent
    feature!("multicomponent", "Experimental simultaneous electronic/quantum-particle SCF, constraints and analytic coupled geometry", Molecule, Driver,
        "driver::run(\"multicomponent\", ...); multicomponent::{solve,constraints::solve,derivatives::cartesian}", "multicomponent", "multicomponent",
        "get_multicomponent", ["T166", "T181", "T182"]),
    // settings-contract: multiconfiguration
    feature!("casscf", "Complete-active-space SCF, state-specific or state-averaged", Molecule, Driver,
        "driver::run(\"casscf\", ...); mcscf::casscf::CasProblem::{evaluate,optimize}", "casscf", "casscf",
        "get_casscf", ["T162"]),
    // settings-contract: multiconfiguration
    feature!("mcpdft", "Multiconfiguration pair-density functional theory with translated on-top functionals and multistate forms", Molecule, Driver,
        "driver::run(\"mcpdft\", ...); mcscf::pdft::{on_top_energy,state_energy,multistate}", "mcpdft", "mcpdft",
        "get_mcpdft", ["T162"]),
    // settings-contract: supplied-data
    feature!("superconducting-state", "Canonical static singlet gap, number and phase response on supplied normal bands", Data, Driver,
        "driver::run(\"superconducting-state\", ...); response::superconducting::solve_gap_taylor", "superconducting-state", "superconducting_state",
        "get_superconducting_state", ["T156", "T158"]),
    // settings-contract: conditional
    feature!("no-pair-projector", "Electronic-sector projection and response of a supplied or generated finite-nucleus balanced Dirac operator", Data, Driver,
        "driver::run(\"no-pair-projector\", ...); properties::no_pair::no_pair_projector_taylor; integrals::dirac::restricted_kinetic_balance", "no-pair-projector", "no_pair_projector",
        "get_no_pair_projector", ["T155"]),
    // settings-contract: conditional
    feature!("dirac-coulomb-scf", "Four-component finite-nucleus no-pair Dirac–Coulomb Hartree–Fock", Molecule, Driver,
        "driver::run(\"dirac-coulomb-scf\", ...); properties::no_pair::stationary::DiracCoulombBackend", "dirac-coulomb-scf", "dirac_coulomb_scf",
        "get_dirac_coulomb_scf", ["T180", "T183"]),
    // settings-contract: conditional
    feature!("two-component-relativistic", "Finite-nucleus X2C, DKH2 and matrix ZORA with picture-changed molecular SCF and electric response", Molecule, Driver,
        "driver::run(\"two-component-relativistic\", ...); properties::two_component::MolecularTwoComponent", "two-component-relativistic", "two_component_relativistic",
        "get_two_component_relativistic", ["T189"]),
    // settings-contract: conditional
    feature!("dirac-dks", "Four-component finite-nucleus no-pair Dirac–Coulomb KS with molecular electric response", Molecule, Driver,
        "driver::run(\"dirac-dks\", ...)", "dirac-dks", "dirac_dks",
        "get_dirac_dks", ["T193"]),
    // settings-contract: conditional
    feature!("scdft-pairing-kernel", "Static singlet phonon/Coulomb pairing K and optional generated phonon mass Z", Data, Driver,
        "driver::run(\"scdft-pairing-kernel\", ...); properties::pairing::Lm2005", "scdft-pairing-kernel", "scdft_pairing_kernel",
        "get_scdft_pairing_kernel", ["T154"]),
    // settings-contract: molecular-common
    feature!("dynamic-polarizability", "Molecular complex dipole susceptibility", Molecule, Driver,
        "driver::run(\"dynamic-polarizability\", ...); response::dynamic_polarizability; response::nuclear::{open_shell_dynamic_polarizability, spinor_electric_response}", "dynamic-polarizability", "dynamic_polarizability",
        "get_dynamic_polarizability", ["T020"]),
    // settings-contract: molecular-common
    feature!("molecular-medium-optics", "Molecular dielectric and anisotropic slab optics", Molecule, Driver,
        "driver::run(\"molecular-medium-optics\", ...); response::medium_optics", "molecular-medium-optics",
        "molecular_medium_optics", "get_molecular_medium_optics", ["T172", "T173", "T174", "T175", "T176"]),
    // settings-contract: molecular-common
    feature!("field-taylor", "Directional dipole response through any order", Molecule, Driver,
        "driver::run(\"field-taylor\", ...); response::electric_field_taylor", "field-taylor", "electric_field_taylor",
        "get_electric_field_taylor", ["T163"]),
    // settings-contract: molecular-common
    feature!("field-mixed", "Static mixed dipole derivative", Molecule, Driver,
        "driver::run(\"field-mixed\", ...); response::mixed_electric_field_response", "field-mixed", "mixed_electric_field_response",
        "get_mixed_electric_field_response", ["T163"]),
    // settings-contract: molecular-common
    feature!("beta", "Complete static first hyperpolarizability", Molecule, Driver,
        "driver::run(\"beta\", ...); response::static_beta", "beta", "beta", "get_beta", ["T034","T163"]),
    // settings-contract: molecular-common
    feature!("gamma", "Complete static second hyperpolarizability", Molecule, Driver,
        "driver::run(\"gamma\", ...); response::static_gamma", "gamma", "gamma", "get_gamma", ["T036","T163"]),
    // settings-contract: molecular-dedicated
    feature!("dynamic-beta", "Complete complex quadratic dipole response", Molecule, Dedicated,
        "response::dynamic_beta", "dynamic-beta", "dynamic_beta", "get_dynamic_beta", ["T035"]),
    // settings-contract: molecular-dedicated
    feature!("dynamic-gamma", "Complete complex cubic dipole response", Molecule, Dedicated,
        "response::dynamic_gamma", "dynamic-gamma", "dynamic_gamma", "get_dynamic_gamma", ["T036"]),
    // settings-contract: molecular-dedicated
    feature!("electric-process", "Nonlinear optical frequency presets", Molecule, Dedicated,
        "response::dynamic_electric_process", "dynamic-beta --electric-process",
        "dynamic_electric_process", "get_dynamic_electric_process", []),
    // settings-contract: molecular-dedicated
    feature!("stability", "Molecular orbital stability", Molecule, Dedicated,
        "response::stability_from_state", "stability", "stability", "get_stability", ["T029"]),
    // settings-contract: molecular-dedicated
    feature!("stab-following", "Orbital-instability descent", Molecule, Dedicated,
        "response::stabilize", "stab-following", "stabilize", "get_stab_following", ["T029"]),
    // settings-contract: molecular-dedicated
    feature!("orbital-energy-gradients", "Relaxed orbital eigenvalue gradients", Molecule, Dedicated,
        "response::orbital_energy_gradients", "orbital-energy-gradients", "orbital_energy_gradients",
        "get_orbital_energy_gradients", ["T103"]),
    // settings-contract: molecular-dedicated
    feature!("frontier-gap-gradient", "Frontier orbital gap gradients", Molecule, Dedicated,
        "response::frontier_gap_gradient", "frontier-gap-gradient", "frontier_gap_gradient",
        "get_frontier_gap_gradient", ["T104"]),
    // settings-contract: molecular-dedicated
    feature!("nuclear-taylor", "Analytic nuclear energy Taylor coefficients", Molecule, Dedicated,
        "response::nuclear::nuclear_taylor", "nuclear-taylor", "nuclear_taylor",
        "get_nuclear_taylor", []),
    // settings-contract: molecular-dedicated
    feature!("nuclear-mixed", "Mixed nuclear energy derivatives", Molecule, Dedicated,
        "response::nuclear::mixed_nuclear_response", "nuclear-mixed", "mixed_nuclear_response",
        "get_mixed_nuclear_response", []),
    // settings-contract: molecular-dedicated
    feature!("molecular-taylor", "Nuclear and electric energy/dipole Taylor response", Molecule, Dedicated,
        "response::nuclear::molecular_taylor", "molecular-taylor", "molecular_taylor",
        "get_molecular_taylor", []),
    // settings-contract: molecular-common
    feature!("force-constants", "Analytic cubic and quartic force constants", Molecule, Driver,
        "driver::run(\"force-constants\", ...); response::force_constants::analytic_force_constants", "force-constants",
        "analytic_force_constants", "get_analytic_force_constants",
        ["T113", "T114", "T115", "T116", "T117", "T118"]),
    // settings-contract: molecular-dedicated
    feature!("anharmonic", "Anharmonic bands and intensities", Molecule, Dedicated,
        "response::anharmonic::molecular::molecular_anharmonic", "anharmonic", "anharmonic",
        "get_anharmonic", ["T037", "T119"]),
    // settings-contract: molecular-dedicated
    feature!("molecular-mixed", "Mixed nuclear/electric derivative", Molecule, Dedicated,
        "response::nuclear::mixed_molecular_response", "molecular-mixed", "mixed_molecular_response",
        "get_mixed_molecular_response", []),
    // settings-contract: molecular-common
    feature!("static-spectroscopy", "Directional analytic APT, Raman and hyper-Raman", Molecule, Driver,
        "driver::run(\"static-spectroscopy\", ...); response::nuclear::static_spectroscopic_derivative", "static-spectroscopy",
        "static_spectroscopic_derivative", "get_static_spectroscopic_derivative", ["T018", "T112"]),
    // settings-contract: periodic-dedicated
    feature!("periodic-response", "Self-consistent finite-q density response", Periodic, Dedicated,
        "dfpt::density::PeriodicDensitySolver::solve", "periodic-response", "periodic_density_response",
        "get_periodic_density_response", []),
    // settings-contract: periodic-dedicated
    feature!("field-response", "Screened periodic electric/magnetic field response", Periodic, Dedicated,
        "dfpt::field::periodic_field_response", "field-response", "periodic_field_response",
        "get_periodic_field_response", []),
    // settings-contract: periodic-dedicated
    feature!("dielectric", "Inverse dielectric matrix", Periodic, Dedicated,
        "dfpt::density::PeriodicDensitySolver::dielectric", "dielectric", "periodic_dielectric",
        "get_periodic_dielectric", []),
    // settings-contract: periodic-dedicated
    feature!("thermal-response", "Mermin thermodynamic response", Periodic, Dedicated,
        "dfpt::density::PeriodicDensitySolver::solve_thermal", "thermal-response",
        "periodic_thermal_response", "get_periodic_thermal_response", []),
    // settings-contract: periodic-operator
    feature!("periodic-electrostatics", "Dimensional point/Gaussian electrostatics", Periodic, Dedicated,
        "periodic::electrostatics::periodic_electrostatics", "periodic-electrostatics",
        "periodic_electrostatics", "get_periodic_electrostatics", ["T012"]),
    // settings-contract: periodic-operator
    feature!("periodic-basis", "Dimensional Gaussian Bloch basis", Periodic, Dedicated,
        "periodic::periodic_basis", "periodic-basis", "periodic_basis", "get_periodic_basis", ["T012"]),
    // settings-contract: periodic-operator
    feature!("periodic-gaussian-integral", "Dimensional Gaussian Coulomb integrals", Periodic, Dedicated,
        "periodic::gaussian_coulomb::periodic_gaussian_integral", "periodic-gaussian-integral",
        "periodic_gaussian_integral", "get_periodic_gaussian_integral", ["T012"]),
    // settings-contract: periodic-operator
    feature!("periodic-coulomb", "Dimensional contracted Coulomb operators", Periodic, Dedicated,
        "periodic::coulomb_boundary::periodic_coulomb_operators", "periodic-coulomb",
        "periodic_coulomb_operators", "get_periodic_coulomb_operators", ["T012"]),
    // settings-contract: periodic-operator
    feature!("periodic-xc", "Dimensional semilocal XC operators", Periodic, Dedicated,
        "periodic::xc_boundary::periodic_xc_operators", "periodic-xc", "periodic_xc_operators",
        "get_periodic_xc_operators", ["T012"]),
    // settings-contract: molecular-dedicated
    feature!("magnetic-response", "Orbital magnetic spin density response", Molecule, Dedicated,
        "properties::magnetic::magnetic_density_response", "magnetic-response", "magnetic_response",
        "get_magnetic_response", []),
    // settings-contract: molecular-dedicated
    feature!("nmr", "Orbital NMR shielding", Molecule, Dedicated,
        "properties::magnetic::molecular_nmr_response", "nmr", "nmr", "get_nmr", ["T044"]),
    // settings-contract: periodic-operator
    feature!("periodic-field", "Dimensional electromagnetic Fourier vertices", Periodic, Dedicated,
        "periodic::field::periodic_field_operators", "periodic-field", "periodic_field_operators",
        "get_periodic_field_operators", []),
    // settings-contract: periodic-dedicated
    feature!("dimensional-scf", "Self-consistent 1D/2D/3D DFT", Periodic, Dedicated,
        "periodic::scf_boundary::run_dimensional_scf", "dimensional-scf", "dimensional_scf",
        "get_dimensional_scf", ["T010"]),
    // settings-contract: periodic-dedicated
    feature!("dimensional-field", "Screened dimensional field response", Periodic, Dedicated,
        "dfpt::field_dimensional::dimensional_field_response", "dimensional-scf --field-transfer",
        "dimensional_field_response", "get_dimensional_field_response", []),
    // settings-contract: molecular-common
    feature!("energy", "Molecular total energy and components", Molecule, Driver,
        "driver::run(\"energy\", ...)", "energy --format json", "energy", "get_energy_report",
        ["T004", "T124", "T127", "T129", "T150", "T151", "T152", "T159", "T163"]),
    // settings-contract: molecular-common
    feature!("mermin-free-energy", "Electronic Mermin free energy of a Fermi-Dirac molecular reference", Molecule, Driver,
        "driver::run(\"mermin-free-energy\", ...)", "mermin-free-energy", "mermin_free_energy", "get_mermin_free_energy",
        ["T005"]),
    // settings-contract: molecular-common
    feature!("gibbs-free-energy", "Ideal-gas RRHO Gibbs free energy from the analytic molecular Hessian", Molecule, Driver,
        "driver::run(\"gibbs-free-energy\", ...); properties::thermochemistry::thermochemistry", "gibbs-free-energy", "gibbs_free_energy", "get_gibbs_free_energy",
        ["T016"]),
    // settings-contract: molecular-common
    feature!("free-energy", "Compatibility alias for molecular electronic Mermin free energy; use mermin-free-energy", Molecule, Driver,
        "driver::run(\"free-energy\", ...)", "free-energy", "free_energy", "get_free_energy_report",
        ["T005"]),
    // settings-contract: molecular-common
    feature!("gradient", "Analytic molecular nuclear gradient", Molecule, Driver,
        "driver::run(\"gradient\", ...)", "gradient --format json", "gradient", "get_gradient_report",
        ["T006", "T159", "T163"]),
    // settings-contract: molecular-common
    feature!("streamed-gradient", "Analytic gradient with integral-direct builds and no stored integral tensor", Molecule, Driver,
        "driver::run(\"streamed-gradient\", ...); gradient::assemble_gradient", "streamed-gradient", "streamed_gradient",
        "get_streamed_gradient", ["T139"]),
    // settings-contract: molecular-common
    feature!("charges", "Mulliken and Loewdin charges and spin populations", Molecule, Driver,
        "driver::run(\"charges\", ...)", "charges --format json", "charges", "get_charges_report",
        ["T007"]),
    // settings-contract: molecular-common
    feature!("natural-bond", "Natural bond orbital analysis: NAOs, natural populations, natural Lewis structure, NLMOs and donor-acceptor energies", Molecule, Driver,
        "driver::run(\"natural-bond\", ...); properties::natural_bond::natural_bond_analysis", "natural-bond --format json", "natural_bond",
        "get_natural_bond", ["T169"]),
    // settings-contract: molecular-common
    feature!("igmh", "Independent gradient model based on Hirshfeld partition of the molecular density", Molecule, Driver,
        "driver::run(\"igmh\", ...); properties::igmh::analyze", "igmh --format json", "igmh",
        "get_igmh", ["T194"]),
    // settings-contract: periodic-common
    feature!("periodic-natural-bond", "Natural bond orbital analysis of a periodic reference on the Born-von Karman supercell of its k mesh", Periodic, Driver,
        "driver::run(\"periodic-natural-bond\", ...); properties::natural_bond::periodic::periodic_natural_bond", "periodic-natural-bond --format json", "periodic_natural_bond",
        "get_periodic_natural_bond", ["T169"]),
    // settings-contract: molecular-common
    feature!("natural-shielding", "Natural chemical shielding analysis: NMR shielding over natural localized molecular orbitals", Molecule, Driver,
        "driver::run(\"natural-shielding\", ...); properties::magnetic::orbital_shielding", "natural-shielding --format json", "natural_shielding",
        "get_natural_shielding", ["T169"]),
    // settings-contract: molecular-common
    feature!("natural-coupling", "Natural J-coupling analysis: indirect spin-spin couplings over natural localized molecular orbitals", Molecule, Driver,
        "driver::run(\"natural-coupling\", ...); response::spin_spin::spin_spin_couplings_by_orbital", "natural-coupling --format json", "natural_coupling",
        "get_natural_coupling", ["T169"]),
    // settings-contract: molecular-common
    feature!("natural-energy-decomposition", "Natural energy decomposition of fragment interactions from natural bond orbitals", Molecule, Driver,
        "driver::run(\"natural-energy-decomposition\", ...)", "natural-energy-decomposition --format json", "natural_energy_decomposition",
        "get_natural_energy_decomposition", ["T169"]),
    // settings-contract: molecular-common
    feature!("sapt-analysis", "Intermolecular perturbation analysis with asymptotic correction, full-response dispersion, charge transfer and atom-pair partition", Molecule, Driver,
        "driver::run(\"sapt-analysis\", ...); sapt::sapt_interaction_with", "sapt-analysis --format json", "sapt_analysis",
        "get_sapt_analysis", ["T170"]),
    // settings-contract: molecular-common
    feature!("dipole", "Molecular electric dipole moment", Molecule, Driver,
        "driver::run(\"dipole\", ...)", "dipole", "dipole", "get_dipole_report", ["T008"]),
    // settings-contract: molecular-common
    feature!("orbitals", "Canonical orbitals of the converged reference", Molecule, Driver,
        "driver::run(\"orbitals\", ...)", "orbitals", "orbitals", "get_orbitals_report", ["T009"]),
    // settings-contract: molecular-common
    feature!("molden", "Molden wavefunction export", Molecule, Driver,
        "driver::run(\"molden\", ...); properties::molden::molden_text", "molden", "write_molden",
        "write_molden", ["T107"]),
    // settings-contract: molecular-common
    feature!("orbitals-save", "JSON orbital export with occupation/density verification", Molecule, Driver,
        "driver::run(\"orbitals-save\", ...)", "orbitals-save", "save_orbitals", "write_orbitals",
        ["T108"]),
    // settings-contract: molecular-common
    feature!("molecular-multipole", "Molecular multipoles to rank 8 in three conventions", Molecule, Driver,
        "driver::run(\"molecular-multipole\", ...)", "molecular-multipole", "molecular_multipole",
        "get_molecular_multipole", ["T101"]),
    // settings-contract: molecular-common
    feature!("quadrupole-polarizability", "Electric quadrupole response to a symmetric traceless field gradient", Molecule, Driver,
        "driver::run(\"quadrupole-polarizability\", ...)", "quadrupole-polarizability", "quadrupole_polarizability",
        "get_quadrupole_polarizability", ["T179"]),
    // settings-contract: molecular-common
    feature!("multipole-polarizability", "Electric response to Racah solid-harmonic multipole fields", Molecule, Driver,
        "driver::run(\"multipole-polarizability\", ...)", "multipole-polarizability", "multipole_polarizability",
        "get_multipole_polarizability", ["T184"]),
    // settings-contract: molecular-common
    feature!("conceptual-dft", "Conceptual DFT descriptors (charge channel)", Molecule, Driver,
        "driver::run(\"conceptual-dft\", ...); response::ensemble::EnsembleSolver", "conceptual-dft",
        "conceptual_dft", "get_conceptual_dft", ["T084"]),
    // settings-contract: molecular-common
    feature!("spin-conceptual-dft", "Spin-resolved conceptual DFT descriptors", Molecule, Driver,
        "driver::run(\"spin-conceptual-dft\", ...)", "spin-conceptual-dft", "spin_conceptual_dft",
        "get_spin_conceptual_dft", ["T085"]),
    // settings-contract: molecular-common
    feature!("hessian", "Analytic Cartesian Hessian with its verdict", Molecule, Driver,
        "driver::run(\"hessian\", ...)", "hessian --format json", "hessian_report", "get_hessian_report",
        ["T013", "T015", "T128", "T163"]),
    // settings-contract: molecular-common
    feature!("frequencies", "Harmonic frequencies, IR intensities and RRHO thermochemistry", Molecule, Driver,
        "driver::run(\"frequencies\", ...)", "frequencies --format json", "frequencies_report",
        "get_frequencies_report", ["T014", "T016"]),
    // settings-contract: molecular-common
    feature!("ir-raman", "Analytic IR intensities and Raman activities", Molecule, Driver,
        "driver::run(\"ir-raman\", ...)", "ir-raman", "vibrational_spectra", "get_vibrational_spectra",
        ["T017"]),
    // settings-contract: molecular-common
    feature!("polarizability", "Static dipole polarizability (analytic CPKS)", Molecule, Driver,
        "driver::run(\"polarizability\", ...)", "polarizability --format json", "polarizability_report",
        "get_polarizability_report", ["T019", "T163"]),
    // settings-contract: molecular-common
    feature!("multipole-gradient", "Analytic nuclear derivatives of multipole moments", Molecule, Driver,
        "driver::run(\"multipole-gradient\", ...)", "multipole-gradient", "multipole_gradient",
        "get_multipole_gradient", ["T102"]),
    // settings-contract: molecular-common
    feature!("response-save", "First-order response export: U^F, dP/dF and dP/dR", Molecule, Driver,
        "driver::run(\"response-save\", ...)", "response-save", "response_derivatives",
        "get_response_derivatives", ["T038"]),
    // settings-contract: molecular-common
    feature!("third-derivative-2n1", "Cubic force constants from first-order responses by the 2n+1 rule", Molecule, Driver,
        "driver::run(\"third-derivative-2n1\", ...); response::nuclear::cubic_force_constants_reduced", "third-derivative-2n1",
        "third_derivative_2n1", "get_third_derivative_2n1", ["T138"]),
    // settings-contract: molecular-common
    feature!("cubic-fc-along", "Cubic force constants contracted with a displacement pattern", Molecule, Driver,
        "driver::run(\"cubic-fc-along\", ...); response::nuclear::cubic_force_constants_along", "cubic-fc-along",
        "cubic_fc_along", "get_cubic_fc_along", ["T086"]),
    // settings-contract: molecular-common
    feature!("quartic-fc-along", "Quartic force constants contracted with two displacement patterns", Molecule, Driver,
        "driver::run(\"quartic-fc-along\", ...); response::nuclear::quartic_force_constants_along", "quartic-fc-along",
        "quartic_fc_along", "get_quartic_fc_along", ["T087"]),
    // settings-contract: molecular-common
    feature!("excitations", "Vertical TDA/RPA excitation energies, oscillator and rotatory strengths", Molecule, Driver,
        "driver::run(\"excitations\", ...); response::tddft::TdaSolver", "excitations", "excitations",
        "get_excitations", ["T021", "T030"]),
    // settings-contract: molecular-common
    feature!("ecd-spectrum", "Gaussian-broadened absorption and electronic circular-dichroism spectra", Molecule, Driver,
        "driver::run(\"ecd-spectrum\", ...)", "ecd-spectrum", "ecd_spectrum", "get_ecd_spectrum",
        ["T001"]),
    // settings-contract: molecular-common
    feature!("td-state-gradient", "Analytic nuclear gradient of a singlet TDA excited state", Molecule, Driver,
        "driver::run(\"td-state-gradient\", ...); response::tddft_grad::tda_gradient_with_grid", "td-state-gradient",
        "td_state_gradient", "get_td_state_gradient", ["T022"]),
    // settings-contract: molecular-common
    feature!("hyperfine", "Contact and spin-dipolar hyperfine tensors with principal axes and raw residuals", Molecule, Driver,
        "driver::run(\"hyperfine\", ...)", "hyperfine", "hyperfine", "get_hyperfine", ["T039"]),
    // settings-contract: molecular-common
    feature!("nmr-shielding", "Nuclear magnetic shielding tensors from the coupled-perturbed magnetic response", Molecule, Driver,
        "driver::run(\"nmr-shielding\", ...); properties::magnetic::molecular_nmr_response", "nmr-shielding", "nmr_shielding",
        "get_nmr_shielding", ["T044"]),
    // settings-contract: molecular-common
    feature!("nmr-spectrum", "First-order or exact isotropic spin-Hamiltonian NMR spectrum", Molecule, Driver,
        "driver::run(\"nmr-spectrum\", ...)", "nmr-spectrum", "nmr_spectrum", "get_nmr_spectrum", ["T003"]),
    // settings-contract: molecular-common
    feature!("spin-spin-couplings", "Indirect nuclear spin-spin coupling tensors and constants", Molecule, Driver,
        "driver::run(\"spin-spin-couplings\", ...); response::spin_spin::spin_spin_couplings", "spin-spin-couplings",
        "spin_spin_couplings", "get_spin_spin_couplings", ["T043"]),
    // settings-contract: molecular-common
    feature!("fermi-contact-couplings", "Fermi-contact part of the indirect nuclear spin-spin couplings", Molecule, Driver,
        "driver::run(\"fermi-contact-couplings\", ...); response::spin_spin::spin_spin_couplings", "fermi-contact-couplings",
        "fermi_contact_couplings", "get_fermi_contact_couplings", ["T042"]),
    // settings-contract: supplied-data
    feature!("exafs", "Single and plane-wave double scattering EXAFS from tabulated scattering functions", Molecule, Driver,
        "driver::run(\"exafs\", ...)", "exafs", "exafs", "get_exafs", ["T122"]),
    // settings-contract: molecular-common
    feature!("xanes", "Core-level absorption from core-valence-separated Tamm-Dancoff response", Molecule, Driver,
        "driver::run(\"xanes\", ...); response::tddft::TdaSolver::solve_core", "xanes", "xanes", "get_xanes", ["T123"]),
    // settings-contract: molecular-common
    feature!("nexafs", "Polarization-resolved near-edge absorption with CVS or per-hole relaxation", Molecule, Driver,
        "driver::run(\"nexafs\", ...); response::tddft::TdaSolver::solve_core_with_residuals", "nexafs", "nexafs", "get_nexafs", ["T187"]),
    // settings-contract: molecular-common
    feature!("constrained-dft", "Variational molecular charge and spin density constraints", Molecule, Driver,
        "driver::run(\"constrained-dft\", ...); scf::constrained::run_collinear", "constrained-dft",
        "constrained_dft", "get_constrained_dft", ["T185"]),
    // settings-contract: molecular-common
    feature!("dft-plus-u", "Variational molecular Dudarev and Liechtenstein projected Hubbard corrections", Molecule, Driver,
        "driver::run(\"dft-plus-u\", ...); scf::hubbard::run_collinear", "dft-plus-u",
        "dft_plus_u", "get_dft_plus_u", ["T190"]),
    // settings-contract: molecular-common
    feature!("pz-sic", "Molecular Perdew-Zunger orbital self-interaction correction", Molecule, Driver,
        "driver::run(\"pz-sic\", ...); scf::sic::run_pz", "pz-sic",
        "pz_sic", "get_pz_sic", ["T191"]),
    // settings-contract: molecular-common
    feature!("flo-sic-fixed", "Molecular Fermi-Lowdin SIC with fixed descriptors on frozen UKS orbitals", Molecule, Driver,
        "driver::run(\"flo-sic-fixed\", ...); scf::sic::flo_sic::run_fixed_descriptors", "flo-sic-fixed",
        "flo_sic_fixed", "get_flo_sic_fixed", ["T192"]),
    // settings-contract: molecular-common
    feature!("car-parrinello-md", "Microcanonical molecular extended-Lagrangian orbital and nuclear dynamics", Molecule, Driver,
        "driver::run(\"car-parrinello-md\", ...)", "car-parrinello-md", "car_parrinello_md", "get_car_parrinello_md", ["T186"]),
    // settings-contract: molecular-common
    feature!("magnetizability", "Magnetizability tensor with a common gauge origin", Molecule, Driver,
        "driver::run(\"magnetizability\", ...); properties::magnetic::magnetic_density_response; response::nuclear::spinor_one_body_response", "magnetizability",
        "magnetizability", "get_magnetizability", ["T081"]),
    // settings-contract: molecular-common
    feature!("magnetic-quadrupole", "Orbital and Pauli magnetic quadrupole in a source-free field gradient", Molecule, Driver,
        "driver::run(\"magnetic-quadrupole\", ...)", "magnetic-quadrupole", "magnetic_quadrupole", "get_magnetic_quadrupole", ["T178"]),
    // settings-contract: molecular-common
    feature!("franck-condon", "Franck-Condon progression of an excited state in the vertical-gradient model", Molecule, Driver,
        "driver::run(\"franck-condon\", ...)", "franck-condon", "franck_condon", "get_franck_condon", ["T121"]),
    // settings-contract: molecular-common
    feature!("excited-optimization", "Geometry relaxation on the surface of a singlet TDA excited state", Molecule, Driver,
        "driver::run(\"excited-optimization\", ...)", "excited-optimization", "excited_optimization",
        "get_excited_optimization", ["T120"]),
    // settings-contract: molecular-common
    feature!("optical-activity", "Optical rotation from the electric dipole - magnetic dipole polarizability", Molecule, Driver,
        "driver::run(\"optical-activity\", ...); response::dynamic::dynamic_optical_rotation", "optical-activity",
        "optical_activity", "get_optical_activity", ["T074"]),
    // settings-contract: molecular-common
    feature!("broken-symmetry-exchange", "Heisenberg exchange coupling from a broken-symmetry determinant", Molecule, Driver,
        "driver::run(\"broken-symmetry-exchange\", ...)", "broken-symmetry-exchange", "broken_symmetry_exchange",
        "get_broken_symmetry_exchange", ["T071"]),
    // settings-contract: molecular-common
    feature!("vcd", "Vibrational circular dichroism from atomic polar and axial tensors", Molecule, Driver,
        "driver::run(\"vcd\", ...)", "vcd", "vcd", "get_vcd", ["T002"]),
    // settings-contract: molecular-common
    feature!("spin-susceptibility", "Static spin susceptibility of a molecular Fermi-Dirac ensemble", Molecule, Driver,
        "driver::run(\"spin-susceptibility\", ...); response::ensemble::EnsembleSolver", "spin-susceptibility",
        "spin_susceptibility", "get_spin_susceptibility", ["T040"]),
    // settings-contract: molecular-common
    feature!("g-tensor", "Electronic g tensor; bounded HF/UHF Pauli nuclear derivatives", Molecule, Driver,
        "driver::run(\"g-tensor\", ...); properties::magnetic::magnetic_density_response; properties::g_nuclear::PauliGNuclearContext", "g-tensor", "g_tensor",
        "get_g_tensor", ["T069"]),
    // settings-contract: molecular-common
    feature!("spin-spin-zfs", "Direct electron spin-spin D tensor and HF nuclear derivatives; SOC excluded", Molecule, Driver,
        "driver::run(\"spin-spin-zfs\", ...); properties::zero_field_splitting::spin_spin_nuclear_derivatives",
        "spin-spin-zfs", "spin_spin_zfs", "get_spin_spin_zfs", []),
    // settings-contract: molecular-common
    feature!("roks", "Restricted open-shell Kohn-Sham energy and analytic gradient", Molecule, Driver,
        "driver::run(\"roks\", ...); scf::roks::run_roks", "roks", "roks", "get_roks", ["T024"]),
    // settings-contract: molecular-common
    feature!("roks-hessian", "Analytic Hessian of the restricted open-shell energy", Molecule, Driver,
        "driver::run(\"roks-hessian\", ...); response::nuclear::open_shell_hessian", "roks-hessian", "roks_hessian", "get_roks_hessian", ["T025"]),
    // settings-contract: molecular-common
    feature!("roks-polarizability", "Dipole moment and static polarizability of a restricted open-shell reference", Molecule, Driver,
        "driver::run(\"roks-polarizability\", ...); response::nuclear::open_shell_dynamic_polarizability", "roks-polarizability", "roks_polarizability", "get_roks_polarizability", ["T024"]),
    // settings-contract: molecular-common
    feature!("dc", "Divide-and-conquer Kohn-Sham energy and gradient of subsystems with a common Fermi level", Molecule, Driver,
        "driver::run(\"dc\", ...); scf::divide_conquer::run_divide_conquer", "dc", "divide_conquer",
        "get_divide_conquer", ["T109"]),
    // settings-contract: molecular-common
    feature!("mrsf", "Mixed-reference spin-flip TDA singlets and triplets on a ROKS triplet", Molecule, Driver,
        "driver::run(\"mrsf\", ...); response::mrsf::mrsf_tda", "mrsf", "mrsf", "get_mrsf", ["T026"]),
    // settings-contract: molecular-common
    feature!("mrsf-state", "Analytic gradient of one mixed-reference spin-flip state", Molecule, Driver,
        "driver::run(\"mrsf-state\", ...); response::nuclear::mixed_reference_surface", "mrsf-state", "mrsf_state", "get_mrsf_state", ["T027"]),
    // settings-contract: molecular-common
    feature!("mrsf-hessian", "Analytic Hessian of one mixed-reference spin-flip state", Molecule, Driver,
        "driver::run(\"mrsf-hessian\", ...); response::nuclear::mixed_reference_surface", "mrsf-hessian", "mrsf_hessian", "get_mrsf_hessian", ["T028"]),
    // settings-contract: molecular-common
    feature!("spin-orbit", "One-shot spin-orbit coupling: spinor energies of a collinear reference", Molecule, Driver,
        "driver::run(\"spin-orbit\", ...)", "spin-orbit", "spin_orbit", "get_spin_orbit", ["T068"]),
    // settings-contract: molecular-common
    feature!("spinor-polarizability", "Dynamic polarizability of the self-consistent spin-orbit spinor state", Molecule, Driver,
        "driver::run(\"spinor-polarizability\", ...); response::spinor::{spin_orbit_spinor_reference, spinor_dynamic_polarizability}", "spinor-polarizability",
        "spinor_polarizability", "get_spinor_polarizability", ["T073"]),
    // settings-contract: molecular-common
    feature!("magnetic-anisotropy", "Magnetic anisotropy energy from the band energies of several spin axes", Molecule, Driver,
        "driver::run(\"magnetic-anisotropy\", ...)", "magnetic-anisotropy", "magnetic_anisotropy",
        "get_magnetic_anisotropy", ["T078"]),
    // settings-contract: molecular-common
    feature!("magnon", "Transverse spin excitations from spin-flip TDA of an unrestricted reference", Molecule, Driver,
        "driver::run(\"magnon\", ...); response::spin_flip::spin_flip_tda", "magnon", "magnon", "get_magnon", ["T079"]),
    // settings-contract: conditional
    feature!("mcd", "Closed-shell TDA A/B, real/complex thermal manifolds, or molecular active-space CI/SOC spectra", Molecule, Driver,
        "driver::run(\"mcd\", ...)", "mcd", "mcd", "get_mcd", ["T033"]),
    // settings-contract: molecular-common
    feature!("td-hessian", "Analytic Hessian and harmonic frequencies of a TDA or RPA excited state", Molecule, Driver,
        "driver::run(\"td-hessian\", ...); response::nuclear::excited_state_hessian", "td-hessian", "td_hessian",
        "get_td_hessian", ["T023"]),
    // settings-contract: molecular-common
    feature!("td-ecd-giao", "Length-gauge rotatory strengths with London orbitals", Molecule, Driver,
        "driver::run(\"td-ecd-giao\", ...); response::london_ecd::london_rotatory_strengths", "td-ecd-giao", "td_ecd_giao",
        "get_td_ecd_giao", ["T031"]),
    // settings-contract: molecular-common
    feature!("roa", "Raman optical activity from analytic normal-mode derivatives of the optical tensors", Molecule, Driver,
        "driver::run(\"roa\", ...); response::nuclear::optical_tensor_path", "roa", "roa", "get_roa", ["T075"]),
    // settings-contract: molecular-common
    feature!("hypermagnetizability", "Static hypermagnetizability with a common gauge origin", Molecule, Driver,
        "driver::run(\"hypermagnetizability\", ...); response::magnetic_taylor::hypermagnetizability",
        "hypermagnetizability", "hypermagnetizability", "get_hypermagnetizability", ["T080"]),
    // settings-contract: molecular-common
    feature!("cotton-mouton", "Cotton-Mouton constant from the hypermagnetizability and the orientational term", Molecule, Driver,
        "driver::run(\"cotton-mouton\", ...); response::magnetic_taylor::hypermagnetizability", "cotton-mouton",
        "cotton_mouton", "get_cotton_mouton", ["T032"]),
    // settings-contract: molecular-common
    feature!("magneto-optical", "Faraday rotation tensor: magnetic-field derivative of the frequency-dependent polarizability", Molecule, Driver,
        "driver::run(\"magneto-optical\", ...); response::field_path::faraday_tensor", "magneto-optical",
        "magneto_optical", "get_magneto_optical", ["T072"]),
    // settings-contract: molecular-common
    feature!("bfield", "Self-consistent field in a finite uniform magnetic field with London orbitals", Molecule, Driver,
        "driver::run(\"bfield\", ...); scf::magnetic::magnetic_field_scf", "bfield",
        "magnetic_field_scf", "get_magnetic_field_scf", ["T082"]),
    // settings-contract: molecular-common
    feature!("bfield-gradient", "Analytic gradient of the energy in a finite uniform magnetic field with London orbitals", Molecule, Driver,
        "driver::run(\"bfield-gradient\", ...); scf::magnetic::magnetic_field_gradient", "bfield-gradient",
        "magnetic_field_gradient", "get_magnetic_field_gradient", ["T083"]),
    // settings-contract: molecular-common
    feature!("magnetoelectric", "Linear magnetoelectric response of an open-shell reference to first order in spin-orbit coupling", Molecule, Driver,
        "driver::run(\"magnetoelectric\", ...); response::field_path::spin_orbit_magnetoelectric", "magnetoelectric",
        "magnetoelectric", "get_magnetoelectric", ["T070"]),
    // settings-contract: periodic-common
    feature!("periodic-energy", "Finite-k periodic energy per cell with analytic forces", Periodic, Driver,
        "driver::run(\"periodic-energy\", ...); periodic::scf::run_periodic_scf", "periodic-energy",
        "periodic_energy", "get_periodic_energy_report", ["T010"]),
    // settings-contract: periodic-common
    feature!("stress", "Analytic stress tensor of the finite-k periodic energy, with forces", Periodic, Driver,
        "driver::run(\"stress\", ...); periodic::stress::analytic_stress", "stress", "stress", "get_stress",
        ["T011"]),
    // settings-contract: periodic-common
    feature!("opt-fixed-cell", "Fixed-cell relaxation of the atoms with analytic finite-k forces", Periodic, Driver,
        "driver::run(\"opt-fixed-cell\", ...); periodic::optimizer::optimize_fixed_cell", "opt-fixed-cell",
        "opt_fixed_cell", "get_opt_fixed_cell", ["T045"]),
    // settings-contract: periodic-common
    feature!("cell-opt", "Variable-cell relaxation with analytic forces and stress at a hydrostatic pressure", Periodic, Driver,
        "driver::run(\"cell-opt\", ...); periodic::optimizer::optimize_cell", "cell-opt", "cell_opt",
        "get_cell_opt", ["T046"]),
    // settings-contract: periodic-common
    feature!("phonons", "Gamma-point phonons from analytic coupled-perturbed force constants", Periodic, Driver,
        "driver::run(\"phonons\", ...); periodic::phonon::gamma_force_constants", "phonons", "phonons",
        "get_phonons", ["T053"]),
    // settings-contract: periodic-common
    feature!("phonon-bands", "Phonon dispersion and DOS from analytic supercell force constants", Periodic, Driver,
        "driver::run(\"phonon-bands\", ...); periodic::phonon_bands::ForceConstantModel", "phonon-bands",
        "phonon_bands", "get_phonon_bands", ["T054", "T157"]),
    // settings-contract: periodic-common
    feature!("phonon-thermo", "Harmonic lattice thermodynamics from analytic supercell force constants", Periodic, Driver,
        "driver::run(\"phonon-thermo\", ...); periodic::phonon_bands::harmonic_thermo", "phonon-thermo",
        "phonon_thermo", "get_phonon_thermo", ["T089"]),
    // settings-contract: periodic-common
    feature!("electron-phonon", "Electron-phonon coupling, Eliashberg function and McMillan/Allen-Dynes Tc", Periodic, Driver,
        "driver::run(\"electron-phonon\", ...); periodic::electron_phonon::electron_phonon", "electron-phonon",
        "electron_phonon", "get_electron_phonon", ["T092"]),
    // settings-contract: periodic-common
    feature!("quasi-harmonic", "Quasi-harmonic volume, bulk modulus, thermal expansion and heat capacities", Periodic, Driver,
        "driver::run(\"quasi-harmonic\", ...); periodic::quasi_harmonic::quasi_harmonic", "quasi-harmonic",
        "quasi_harmonic", "get_quasi_harmonic", ["T088", "T095"]),
    // settings-contract: periodic-common
    feature!("elastic", "Clamped- and relaxed-ion elastic constants from analytic strain and internal-strain responses", Periodic, Driver,
        "driver::run(\"elastic\", ...); periodic::elastic::elastic_constants", "elastic", "elastic",
        "get_elastic", ["T047"]),
    // settings-contract: periodic-common
    feature!("elastic-fd", "Validation route: elastic constants from central differences of the analytic stress", Periodic, Driver,
        "driver::run(\"elastic-fd\", ...)", "elastic-fd", "elastic_fd", "get_elastic_fd", ["T048"]),
    // settings-contract: conditional
    feature!("elastic-moduli", "Voigt, Reuss and Hill moduli, wave speeds and Debye temperature from the analytic elastic tensor", Periodic, Driver,
        "driver::run(\"elastic-moduli\", ...); periodic::elastic::elastic_moduli", "elastic-moduli", "elastic_moduli",
        "get_elastic_moduli", ["T090"]),
    // settings-contract: conditional
    feature!("christoffel", "Christoffel phase velocities and polarizations from the analytic elastic tensor", Periodic, Driver,
        "driver::run(\"christoffel\", ...); periodic::elastic::christoffel", "christoffel", "christoffel",
        "get_christoffel", ["T096"]),
    // settings-contract: periodic-common
    feature!("material-card", "Continuum material card from the analytic elastic tensor (JSON or CSV)", Periodic, Driver,
        "driver::run(\"material-card\", ...); geomech::{compliance, engineering_constants, rotate_stiffness}",
        "material-card", "material_card", "get_material_card", ["T142"]),
    // settings-contract: periodic-common
    feature!("equation-of-state", "Third-order Birch-Murnaghan equation of state with analytic pressures", Periodic, Driver,
        "driver::run(\"equation-of-state\", ...); periodic::eos::fit_birch_murnaghan", "equation-of-state",
        "equation_of_state", "get_equation_of_state", ["T097"]),
    // settings-contract: periodic-common
    feature!("dielectric-tensor", "Electronic dielectric tensor by periodic DFPT with the covariant k derivative", Periodic, Driver,
        "driver::run(\"dielectric-tensor\", ...); periodic::dielectric::dielectric_tensors", "dielectric-tensor",
        "dielectric_tensor", "get_dielectric_tensor", ["T057"]),
    // settings-contract: periodic-common
    feature!("born-charges", "Born effective charges and the electronic dielectric tensor by periodic DFPT", Periodic, Driver,
        "driver::run(\"born-charges\", ...); periodic::dielectric::dielectric_tensors", "born-charges",
        "born_charges", "get_born_charges", ["T055", "T056"]),
    // settings-contract: periodic-common
    feature!("piezoelectric-tensor", "Proper clamped- and relaxed-ion piezoelectric tensors", Periodic, Driver,
        "driver::run(\"piezoelectric-tensor\", ...); periodic::dielectric::piezoelectric_tensor", "piezoelectric-tensor",
        "piezoelectric_tensor", "get_piezoelectric_tensor", ["T052"]),
    // settings-contract: periodic-common
    feature!("berry-polarization", "Berry-phase polarization of an insulating reference", Periodic, Driver,
        "driver::run(\"berry-polarization\", ...); periodic::berry_phase::berry_polarization", "berry-polarization",
        "berry_polarization", "get_berry_polarization", ["T061"]),
    // settings-contract: periodic-common
    feature!("crystal-ir", "Crystal infrared intensities, mode effective charges and LO-TO splitting", Periodic, Driver,
        "driver::run(\"crystal-ir\", ...); periodic::dielectric::polar_modes", "crystal-ir", "crystal_ir",
        "get_crystal_ir", ["T091"]),
    // settings-contract: periodic-common
    feature!("static-dielectric", "Static (relaxed-ion) dielectric tensor from Born charges and Gamma modes", Periodic, Driver,
        "driver::run(\"static-dielectric\", ...); periodic::dielectric::polar_modes", "static-dielectric",
        "static_dielectric", "get_static_dielectric", ["T099"]),
    // settings-contract: periodic-common
    feature!("electronic-structure", "Band energies, occupations, band edges, Mulliken populations and optional complex Bloch orbitals", Periodic, Driver,
        "driver::run(\"electronic-structure\", ...)", "electronic-structure", "electronic_structure",
        "get_electronic_structure", ["T009", "T100"]),
    // settings-contract: periodic-common
    feature!("effective-mass", "Band velocity and effective-mass tensor from analytic k derivatives", Periodic, Driver,
        "driver::run(\"effective-mass\", ...); periodic::band_transport::BandModel::derivatives", "effective-mass",
        "effective_mass", "get_effective_mass", ["T065"]),
    // settings-contract: periodic-common
    feature!("boltzmann-transport", "Scalar-relaxation-time Boltzmann transport: conductivity, Seebeck, electronic thermal conductivity", Periodic, Driver,
        "driver::run(\"boltzmann-transport\", ...); periodic::band_transport::transport_integrals", "boltzmann-transport",
        "boltzmann_transport", "get_boltzmann_transport", ["T064"]),
    // settings-contract: periodic-common
    feature!("hall-transport", "Ordinary weak-field Hall conductivity and Hall coefficient from analytic isolated-band derivatives", Periodic, Driver,
        "driver::run(\"hall-transport\", ...); periodic::band_transport::{hall_integrals,hall_coefficient}", "hall-transport",
        "hall_transport", "get_hall_transport", ["T143"]),
    // settings-contract: periodic-common
    feature!("coherent-transport", "Covariant finite-frequency intra/interband conductivity and intrinsic \
        Hall, including degenerate multiplets", Periodic, Driver,
        "driver::run(\"coherent-transport\", ...); periodic::coherent_transport::conductivity", "coherent-transport",
        "coherent_transport", "get_coherent_transport", ["T144"]),
    // settings-contract: periodic-common
    feature!("quantum-hall", "Non-Abelian occupied-slice Chern invariant and integer quantum Hall conductance", Periodic, Driver,
        "driver::run(\"quantum-hall\", ...); periodic::coherent_transport::quantum_hall_from_links", "quantum-hall",
        "quantum_hall", "get_quantum_hall", ["T066"]),
    // settings-contract: periodic-common
    feature!("wannier-flow", "Hybrid Wannier charge centres and their flow", Periodic, Driver,
        "driver::run(\"wannier-flow\", ...); periodic::coherent_transport::wannier_flow", "wannier-flow",
        "wannier_flow", "get_wannier_flow", ["T062"]),
    // settings-contract: periodic-common
    feature!("periodic-cubic-fc-along", "Cubic force constants of a cell contracted with a Gamma-point displacement pattern", Periodic, Driver,
        "driver::run(\"periodic-cubic-fc-along\", ...); response::nuclear::lattice_force_constants_along", "periodic-cubic-fc-along",
        "periodic_cubic_fc_along", "get_periodic_cubic_fc_along", ["T086"]),
    // settings-contract: periodic-common
    feature!("periodic-quartic-fc-along", "Quartic force constants of a cell contracted with two Gamma-point displacement patterns", Periodic, Driver,
        "driver::run(\"periodic-quartic-fc-along\", ...); response::nuclear::lattice_force_constants_along", "periodic-quartic-fc-along",
        "periodic_quartic_fc_along", "get_periodic_quartic_fc_along", ["T087"]),
    // settings-contract: periodic-common
    feature!("periodic-cubic-fc", "Cubic force constants of a cell for Gamma-point displacements", Periodic, Driver,
        "driver::run(\"periodic-cubic-fc\", ...); response::nuclear::lattice_force_constant_tensor", "periodic-cubic-fc",
        "periodic_cubic_fc", "get_periodic_cubic_fc", ["T067"]),
    // settings-contract: periodic-common
    feature!("periodic-quartic-fc", "Quartic force constants of a cell for Gamma-point displacements", Periodic, Driver,
        "driver::run(\"periodic-quartic-fc\", ...); response::nuclear::lattice_force_constant_tensor", "periodic-quartic-fc",
        "periodic_quartic_fc", "get_periodic_quartic_fc", ["T098"]),
    // settings-contract: periodic-common
    feature!("toec", "Third-order Lagrangian elastic constants at clamped and relaxed ions", Periodic, Driver,
        "driver::run(\"toec\", ...); response::nuclear::lattice_elastic_constants", "toec",
        "toec", "get_toec", ["T049", "T050"]),
    // settings-contract: periodic-common
    feature!("foec", "Fourth-order Lagrangian elastic constants at clamped and relaxed ions", Periodic, Driver,
        "driver::run(\"foec\", ...); response::nuclear::lattice_elastic_constants", "foec",
        "foec", "get_foec", ["T051"]),
    // settings-contract: periodic-common
    feature!("mode-gruneisen", "First/second log-volume Gamma phonon response from analytic energy derivatives", Periodic, Driver,
        "driver::run(\"mode-gruneisen\", ...); response::nuclear::{lattice_mode_gruneisen, lattice_volume_gruneisen}", "mode-gruneisen",
        "mode_gruneisen", "get_mode_gruneisen", ["T088"]),
    // settings-contract: periodic-common
    feature!("periodic-beta", "Second-order electric susceptibility of a periodic cell", Periodic, Driver,
        "driver::run(\"periodic-beta\", ...); response::nuclear::lattice_field_response", "periodic-beta",
        "periodic_beta", "get_periodic_beta", ["T058"]),
    // settings-contract: periodic-common
    feature!("periodic-gamma", "Static clamped-ion third-order electric susceptibility of a periodic cell", Periodic, Driver,
        "driver::run(\"periodic-gamma\", ...); response::nuclear::lattice_electric_tensor", "periodic-gamma",
        "periodic_gamma", "get_periodic_gamma", ["T153"]),
    // settings-contract: periodic-common
    feature!("periodic-raman", "Displacement derivative of the electric susceptibility of a periodic cell", Periodic, Driver,
        "driver::run(\"periodic-raman\", ...); response::nuclear::lattice_field_response", "periodic-raman",
        "periodic_raman", "get_periodic_raman", ["T059"]),
    // settings-contract: periodic-common
    feature!("electrostriction", "Strain derivative of the electric susceptibility of a periodic cell", Periodic, Driver,
        "driver::run(\"electrostriction\", ...); response::nuclear::lattice_field_response", "electrostriction",
        "electrostriction", "get_electrostriction", ["T093"]),
    // settings-contract: periodic-common
    feature!("piezo-higher", "Piezoelectric tensor of a periodic cell and its strain derivative", Periodic, Driver,
        "driver::run(\"piezo-higher\", ...); response::nuclear::lattice_field_response", "piezo-higher",
        "piezo_higher", "get_piezo_higher", ["T094"]),
    // settings-contract: periodic-common
    feature!("periodic-efield", "Self-consistent field of a periodic cell in a finite uniform electric field", Periodic, Driver,
        "driver::run(\"periodic-efield\", ...); response::nuclear::lattice_finite_field", "periodic-efield",
        "periodic_efield", "get_periodic_efield", ["T124"]),
    // settings-contract: periodic-common
    feature!("dielectric-q", "Density response and inverse dielectric function at wave vectors commensurate with the k mesh", Periodic, Driver,
        "driver::run(\"dielectric-q\", ...); response::nuclear::lattice_density_response", "dielectric-q",
        "dielectric_q", "get_dielectric_q", ["T060"]),
    // settings-contract: periodic-common
    feature!("orbital-susceptibility-q", "Orbital magnetic susceptibility of a periodic cell at wave vectors commensurate with the k mesh", Periodic, Driver,
        "driver::run(\"orbital-susceptibility-q\", ...); response::nuclear::lattice_orbital_susceptibility", "orbital-susceptibility-q",
        "orbital_susceptibility_q", "get_orbital_susceptibility_q", ["T148"]),
    // settings-contract: periodic-common
    feature!("periodic-dc", "Divide-and-conquer Kohn-Sham energy and forces of a periodic cell at Gamma", Periodic, Driver,
        "driver::run(\"periodic-dc\", ...); response::nuclear::lattice_divide_conquer", "periodic-dc",
        "periodic_dc", "get_periodic_dc", ["T110"]),
    // settings-contract: periodic-common
    feature!("dc-stress", "Stress of the divide-and-conquer free energy of a periodic cell", Periodic, Driver,
        "driver::run(\"dc-stress\", ...); response::nuclear::lattice_divide_conquer", "dc-stress",
        "dc_stress", "get_dc_stress", ["T111"]),
    // settings-contract: periodic-common
    feature!("periodic-nmr-shielding", "Nuclear magnetic shielding in a periodic cell at wave vectors commensurate with the k mesh", Periodic, Driver,
        "driver::run(\"periodic-nmr-shielding\", ...); response::nuclear::lattice_nmr_shielding", "periodic-nmr-shielding",
        "periodic_nmr_shielding", "get_periodic_nmr_shielding", ["T149"]),
    // settings-contract: periodic-common
    feature!("berry-dipole", "Berry curvature dipole of the Fermi surface", Periodic, Driver,
        "driver::run(\"berry-dipole\", ...); periodic::coherent_transport::berry_curvature_dipole", "berry-dipole",
        "berry_dipole", "get_berry_dipole", ["T063"]),

    // settings-contract: periodic-common
    feature!("lab-frame-spin-spiral", "Energy of a commensurate flat spin spiral by two-component SCF in its supercell", Periodic, Driver,
        "driver::run(\"lab-frame-spin-spiral\", ...); periodic::spinor_scf::{BiasedSpinorBackend, run_spinor_scf}", "lab-frame-spin-spiral",
        "lab_frame_spin_spiral", "get_lab_frame_spin_spiral", ["T076"]),

    // settings-contract: periodic-common
    feature!("dmi", "Chirality-odd spin-orbit energy of a commensurate spin spiral", Periodic, Driver,
        "driver::run(\"dmi\", ...); periodic::spinor_scf::{BiasedSpinorBackend, run_spinor_scf}", "dmi",
        "dmi", "get_dmi", ["T077"]),

    // settings-contract: periodic-common
    feature!("bands", "Band structure along a reduced k path", Periodic, Driver,
        "driver::run(\"bands\", ...); periodic::scf::reference_fock_at", "bands", "bands", "get_bands",
        ["T105"]),
    // settings-contract: periodic-common
    feature!("periodic-spin-response", "Finite-q/frequency longitudinal spin-Zeeman response of a collinear periodic reference", Periodic, Driver,
        "driver::run(\"periodic-spin-response\", ...); dfpt::spin::periodic_longitudinal_spin_response", "periodic-spin-response",
        "periodic_spin_response", "get_periodic_spin_response", ["T041"]),
    // settings-contract: periodic-common
    feature!("periodic-field-scf", "Complex self-consistent orbital reference in a real static periodic magnetic/electric field", Periodic, Driver,
        "driver::run(\"periodic-field-scf\", ...); periodic::scf_boundary::run_dimensional_field_scf", "periodic-field-scf",
        "periodic_field_scf", "get_periodic_field_scf", ["T145"]),
    // settings-contract: periodic-common
    feature!("periodic-spinor-scf", "General complex spinor SCF with prescribed Pauli SOC and Zeeman operators", Periodic, Driver,
        "driver::run(\"periodic-spinor-scf\", ...); periodic::spinor_scf::run_spinor_scf", "periodic-spinor-scf",
        "periodic_spinor_scf", "get_periodic_spinor_scf", ["T146"]),
    // settings-contract: periodic-common
    feature!("periodic-magnetic-scf", "Uniform-flux London spinor SCF and magnetic Bloch integration", Periodic, Driver,
        "driver::run(\"periodic-magnetic-scf\", ...); periodic::magnetic_cell::{MagneticDftBackend,MagneticBlochBackend}", "periodic-magnetic-scf",
        "periodic_magnetic_scf", "get_periodic_magnetic_scf", ["T147"]),
    // settings-contract: periodic-common
    feature!("dos", "Gaussian-broadened total and Mulliken-projected density of states", Periodic, Driver,
        "driver::run(\"dos\", ...)", "dos", "dos", "get_dos", ["T106"]),
    // settings-contract: periodic-common
    feature!("pauli-spin-susceptibility", "Bare Pauli spin susceptibility, Fermi-level density of states and \
        electronic heat capacity of the rigid bands", Periodic, Driver,
        "driver::run(\"pauli-spin-susceptibility\", ...); periodic::fermi_thermodynamics::fermi_thermodynamics",
        "pauli-spin-susceptibility", "pauli_spin_susceptibility", "get_pauli_spin_susceptibility", ["T041"]),
    // settings-contract: cli
    feature!("cli", "The dft-rs command line as a callable entry point", Data, Dedicated,
        "cli::run_arguments", "dft-rs", "cli_main", "-", ["T137"]),
    // settings-contract: supplied-data
    feature!("piezoelectric-strain", "Proper d=eS strain coefficients on supplied matching fixed-field tensors", Data, Driver,
        "driver::run(\"piezoelectric-strain\", ...); geomech::piezoelectric::strain_coefficients", "piezoelectric-strain", "piezoelectric_strain",
        "get_piezoelectric_strain", ["T052"]),
    // settings-contract: supplied-data
    feature!("aggregate-moduli", "Voigt, Reuss, Hill, Hashin-Shtrikman and self-consistent moduli of a mineral aggregate", Data, Driver,
        "driver::run(\"aggregate-moduli\", ...); geomech::aggregate_moduli", "aggregate-moduli", "aggregate_moduli",
        "get_aggregate_moduli", ["T140"]),
    // settings-contract: supplied-data
    feature!("rock-physics", "Dry frame, Gassmann fluid substitution and Biot poroelastic parameters", Data, Driver,
        "driver::run(\"rock-physics\", ...); geomech::fluid_substitution", "rock-physics", "rock_physics",
        "get_rock_physics", ["T141"]),
    // settings-contract: supplied-data
    feature!("feature-registry", "The registry of public calculations as data", Data, Driver,
        "driver::registry::registry_value()", "feature-registry", "feature_registry",
        "get_feature_registry", ["T136"]),
];

/// All registered calculations, in the aligned surface-manual order.
pub fn features() -> &'static [FeatureSpec] {
    FEATURES
}

pub fn find(id: &str) -> Option<&'static FeatureSpec> {
    FEATURES.iter().find(|f| f.id == id)
}

/// Resolve a common-driver request without constructing a physical reference.
/// Dedicated entry points remain public, but require their own adapters.
pub(crate) fn driver_feature(id: &str) -> crate::Result<&'static FeatureSpec> {
    if id.trim().is_empty() {
        return Err(crate::DftError::InvalidInput(
            "a calculation ID is required".into(),
        ));
    }
    let spec = find(id).ok_or_else(|| crate::DftError::Unsupported {
        feature: format!("calculation `{id}`"),
        missing:
            "a registered implementation; feature-registry lists the available calculation IDs"
                .into(),
    })?;
    if spec.route != Route::Driver {
        return Err(crate::DftError::Unsupported {
            feature: format!("common-driver calculation `{id}`"),
            missing: format!("a common-driver adapter; this calculation has dedicated bindings on every surface, call {} directly", spec.rust),
        });
    }
    Ok(spec)
}

/// The registry as data (physical target `feature-registry`).
pub fn registry_value() -> Value {
    Value::Array(
        FEATURES
            .iter()
            .map(|f| {
                Value::object()
                    .with("id", f.id)
                    .with("title", f.title)
                    .with(
                        "space",
                        match f.space {
                            Space::Molecule => "molecule",
                            Space::Periodic => "periodic",
                            Space::Both => "molecule_and_periodic",
                            Space::Data => "data",
                        },
                    )
                    .with(
                        "route",
                        match f.route {
                            Route::Driver => "driver",
                            Route::Dedicated => "dedicated",
                        },
                    )
                    .with("rust", f.rust)
                    .with("cli", f.cli)
                    .with("python", f.python)
                    .with("ase", f.ase)
                    .with("targets", f.targets.to_vec())
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_route_capabilities_are_resolved_from_the_existing_registry() {
        for spec in features() {
            match spec.route {
                Route::Driver => assert_eq!(driver_feature(spec.id).unwrap().id, spec.id),
                Route::Dedicated => {
                    let error = driver_feature(spec.id).unwrap_err();
                    assert!(matches!(error, crate::DftError::Unsupported { .. }));
                    assert!(error.to_string().contains(spec.rust));
                    assert_eq!(find(spec.id).unwrap().route, Route::Dedicated);
                }
            }
        }
        assert!(matches!(
            driver_feature("unavailable-calculation"),
            Err(crate::DftError::Unsupported { .. })
        ));
        for empty in ["", " ", "\t"] {
            assert!(matches!(
                driver_feature(empty),
                Err(crate::DftError::InvalidInput(_))
            ));
        }
    }

    fn api_table_ids(text: &str) -> Vec<Vec<String>> {
        text.split("## Table ")
            .skip(1)
            .map(|chunk| {
                chunk
                    .lines()
                    .skip_while(|l| !l.starts_with("| ID |"))
                    .skip(2)
                    .take_while(|l| l.starts_with('|'))
                    .filter_map(|l| {
                        let cell = l.split('|').nth(1)?.trim();
                        Some(cell.trim_matches('`').to_string())
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn api_reference_tables_list_exactly_the_registry() {
        let manuals = [
            ("Rust", include_str!("../../docs/rust-api.md")),
            ("CLI", include_str!("../../docs/cli.md")),
            (
                "native Python and ASE",
                include_str!("../../docs/python-api.md"),
            ),
        ];
        let tables: Vec<_> = manuals
            .iter()
            .map(|(surface, text)| {
                let mut tables = api_table_ids(text);
                assert_eq!(
                    tables.len(),
                    1,
                    "{surface} manual needs one calculation table"
                );
                tables.remove(0)
            })
            .collect();
        let ids: Vec<String> = FEATURES.iter().map(|f| f.id.to_string()).collect();
        for (i, table) in tables.iter().enumerate() {
            assert_eq!(
                table,
                &ids,
                "surface manual table {} differs from the registry",
                i + 1
            );
        }
    }

    #[test]
    fn registry_ids_and_targets_are_well_formed() {
        let targets = include_str!("../../docs/property_targets.csv");
        for (i, f) in FEATURES.iter().enumerate() {
            assert!(
                FEATURES[..i].iter().all(|g| g.id != f.id),
                "duplicate id {}",
                f.id
            );
            assert!(f
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'));
            // "-": an argv/exit-status entry point has no Atoms-based method.
            assert!(
                f.ase.starts_with("get_")
                    || f.ase.starts_with("write_")
                    || (f.ase == "-" && f.space == Space::Data),
                "{}",
                f.ase
            );
            for t in f.targets {
                assert!(
                    targets.lines().any(|l| l.starts_with(&format!("{t},"))),
                    "{} names unknown target {t}",
                    f.id
                );
            }
        }
    }
}
