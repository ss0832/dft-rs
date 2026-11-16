// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Units of the driver interface: one label per control and per numeric
//! report entry.
//!
//! A label begins with a unit expression: factors joined by `*` and `/`,
//! each a name with an optional power (`hartree/bohr^2`, `e*bohr`, `1/bohr`,
//! `cm^-1`). A `/` divides by the one factor that follows it. A bracketed
//! list (`[dimensionless, dimensionless, 1/bohr]`) gives the units along the
//! last axis of an array. A power may name an integer entry of the same
//! object (`e*bohr^rank`, `e*bohr^(rank-1)`, `bohr^(order+1)`). Explanatory text may follow
//! after a space, comma, semicolon or parenthesis.
//!
//! The words `hartree` and `bohr` occur only inside the leading expression.
//! A surface that works in other units (the ASE calculator, in eV and
//! Angstrom) therefore converts a value from its label alone, and a remark
//! can never be mistaken for a unit. Quantities whose atomic unit is not a
//! product of powers of energy and length (magnetic fields, for example)
//! are labelled `atomic units ...` and stay as they are on every surface,
//! like the entries in spectroscopic and SI units and the entries whose name
//! ends with its unit (`scattering_hartree`).
//!
//! Reports keep their labels in the `units` object, keyed by the name of an
//! entry or by its dotted path (`rows.temperature_k`, array indices
//! omitted); the label of an object or an array covers everything below it.
//! [`complete`] labels the real-valued entries a calculation did not label
//! itself from [`REPORT_UNITS`]. Integers are counts and indices and carry
//! no label.

use super::value::Value;

/// Label given to a real-valued report entry that no table knows.
pub const UNSPECIFIED: &str = "unspecified";

/// Unit of every control of the common driver. `text`, `flag` and `integer`
/// mark controls that are not physical quantities.
pub const CONTROL_UNITS: &[(&str, &str)] = &[
    ("density_constraints", "text native-unit JSON objects"),
    ("hubbard_sites", "text native-unit JSON objects"),
    ("hubbard_pairs", "text JSON objects with two zero-based sites and v"),
    ("hubbard_pairs.v", "hartree"),
    ("fod_alpha", "bohr"),
    ("fod_beta", "bohr"),
    ("fod_alpha_imaginary", "bohr"),
    ("fod_beta_imaginary", "bohr"),
    ("fod_gradient", "flag"),
    ("fod_optimize", "flag"),
    ("fod_gradient_tolerance", "hartree/bohr"),
    ("fod_max_iterations", "integer"),
    ("flo_self_consistent", "flag"),
    ("flo_orbital_tolerance", "hartree"),
    ("optimize_geometry", "flag"),
    ("constraint_electric_response", "flag"),
    ("initial_constraint_multipliers", "hartree"),
    ("constraint_max_step", "hartree"),
    ("constraint_max_iterations", "integer"),
    ("constraint_tolerance", "dimensionless"),
    ("nexafs_model", "text"),
    ("absorption_edge", "text"),
    ("absorber_atoms", "integer"),
    ("core_orbitals_alpha", "integer"),
    ("core_orbitals_beta", "integer"),
    ("nexafs_broadening", "hartree"),
    ("nexafs_energy_window", "hartree"),
    ("line_shape", "text"),
    ("minimum_core_overlap", "dimensionless"),
    ("core_response_tolerance", "hartree"),
    ("polarization", "dimensionless"),
    ("magnetic_field_gradient", "atomic units of magnetic field gradient"),
    ("md_steps", "integer"),
    ("md_timestep", "atomic units of time"),
    ("md_constraint_tolerance", "dimensionless"),
    ("nuclear_velocities", "atomic units of velocity"),
    ("nuclear_masses", "atomic units of mass"),
    ("fictitious_mass", "atomic units of mass"),
    ("piezoelectric_stress", "e/bohr^2"),
    ("piezoelectric_compliance", "bohr^3/hartree"),
    ("ion_response", "text"),
    ("cpu_threads", "integer"),
    ("memory_mb", "MiB"),
    ("log_file", "text"),
    ("log_mode", "text"),
    ("log_level", "text"),
    ("fixed_electron_count", "integer"),
    ("fixed_spin_electron_counts", "integer"),
    ("physical_nuclear_charges", "e"),
    ("charge_directions", "e"),
    ("charge_coordinates", "dimensionless"),
    ("charge_bounds", "dimensionless"),
    ("charge_scales", "dimensionless"),
    ("charge_parameter_names", "text"),
    ("charge_origin", "bohr"),
    ("conserve_total_charge", "flag"),
    ("charge_nuclear_gradient", "flag"),
    ("charge_dynamic_response", "flag"),
    ("charge_probe_points", "bohr"),
    ("fragment_internal_energy", "hartree"),
    ("fragment_mode", "text"),
    ("fragment_endpoint", "text"),
    ("fragment_state", "text"),
    ("fragment_weights", "dimensionless"),
    ("fragment_weight_tolerance", "hartree"),
    ("fragment_weight_max_iter", "dimensionless"),
    ("fragment_weight_strategy", "text"),
    ("fragment_spec", "text native-unit JSON object"),
    ("fragment_model_json", "text native-unit JSON object"),
    ("fragment_basis_template", "text"),
    ("fragment_projection_rule", "text"),
    ("fragment_esp_points", "bohr"),
    ("fragment_esp_weight", "dimensionless"),
    ("local_potential_atoms", "integer"),
    ("local_potential_exponents", "bohr^-2"),
    ("local_potential_reference_amplitudes", "hartree"),
    ("local_potential_directions", "hartree"),
    ("steric_reference_strengths", "hartree"),
    ("steric_directions", "hartree"),
    ("steric_environment_exponents", "bohr^-2"),
    ("steric_environment_hardness", "bohr^-3"),
    ("steric_pair_mask", "integer"),
    ("local_frame_anchors", "integer"),
    ("local_frame_offsets", "bohr"),
    ("reference", "text"),
    ("medium_number_density", "1/bohr^3"),
    ("medium_local_field", "text"),
    ("medium_orientation", "dimensionless"),
    ("medium_magnetic_field", "atomic units of magnetic field"),
    ("slab_thickness", "bohr"),
    ("incident_refractive_index", "dimensionless"),
    ("exit_refractive_index", "dimensionless"),
    ("incidence_angle", "dimensionless, radians"),
    ("gks_operator_real", "hartree"),
    ("gks_operator_imaginary", "hartree"),
    ("spinor_xc_tolerance", "hartree"),
    ("operator_tolerance", "atomic units"),
    ("epr_projection", "text"),
    ("epr_projection_tolerance", "dimensionless"),
    ("quantum_atoms", "integer"),
    ("quantum_masses", "atomic units of mass"),
    ("quantum_exponents", "1/bohr^2"),
    ("quantum_lmax", "integer"),
    ("positron_positions", "bohr"),
    ("interspecies_correlation", "text"),
    ("positron_correlation", "text"),
    ("annihilation_enhancement", "text"),
    ("annihilation_gradient_alpha", "dimensionless"),
    ("particles", "text"),
    ("particles.kind", "text"),
    ("particles.label", "text"),
    ("particles.atom", "integer"),
    ("particles.position", "bohr"),
    ("particles.charge", "e"),
    ("particles.mass", "atomic units of mass"),
    ("particles.exponents", "1/bohr^2"),
    ("particles.lmax", "integer"),
    ("particles.correlation", "text"),
    ("species_stability", "text"),
    ("species_stability_roots", "integer"),
    ("species_tolerance", "dimensionless (atomic-unit orbital and density residuals)"),
    ("species_max_iterations", "integer"),
    ("species_mixing", "dimensionless"),
    ("species_electronic_model", "dimensionless"),
    ("quantum_position_constraint", "flag"),
    ("quantum_position_targets", "bohr"),
    ("species_derivative_order", "integer"),
    ("species_direction", "dimensionless"),
    ("species_polarizability", "flag"),
    ("species_quadrupole_polarizability", "flag"),
    ("species_field_gradient", "hartree/e/bohr^2"),
    ("electric_field_gradient", "hartree/e/bohr^2"),
    ("quantum_mass_response", "flag"),
    ("occupation_mode", "text"),
    ("occupied_alpha", "integer"),
    ("occupied_beta", "integer"),
    ("occupations_alpha", "dimensionless"),
    ("occupations_beta", "dimensionless"),
    ("pairing_model", "text"),
    ("band_weights", "dimensionless"),
    ("band_weight_direction", "dimensionless"),
    ("mass_renormalization", "dimensionless"),
    ("mass_renormalization_direction", "dimensionless"),
    ("electron_number", "dimensionless"),
    ("electron_number_direction", "dimensionless"),
    ("initial_gap_real", "hartree"),
    ("initial_gap_imaginary", "hartree"),
    ("pairing_kernel", "hartree"),
    ("pairing_kernel_direction", "hartree"),
    ("temperature_direction", "K"),
    ("scdft_temperature_bracket", "K"),
    ("scdft_phase_selection", "flag"),
    ("phonon_energy_direction", "hartree"),
    ("pair_couplings_squared_direction", "hartree^2"),
    ("screened_pair_coulomb_direction", "hartree"),
    ("dirac_integrals", "text"),
    ("nuclear_rms_radii", "bohr"),
    ("nuclear_magnetic_rms_radii", "bohr"),
    ("nuclear_magnetic_moments", "atomic units of magnetic dipole moment"),
    ("nuclear_radius_direction", "bohr"),
    ("nuclear_displacements", "bohr"),
    ("dirac_interaction", "dimensionless"),
    ("speed_of_light", "atomic units of velocity"),
    ("speed_of_light_direction", "atomic units of velocity"),
    ("magnetic_directions", "atomic units of magnetic field"),
    ("dirac_hamiltonian_real", "hartree"),
    ("dirac_hamiltonian_imaginary", "hartree"),
    ("dirac_hamiltonian_direction_real", "hartree"),
    ("dirac_hamiltonian_direction_imaginary", "hartree"),
    ("dirac_overlap_real", "dimensionless"),
    ("dirac_overlap_imaginary", "dimensionless"),
    ("dirac_overlap_direction_real", "dimensionless"),
    ("dirac_overlap_direction_imaginary", "dimensionless"),
    ("dirac_operator_real", "dimensionless"),
    ("dirac_operator_imaginary", "dimensionless"),
    ("relativistic_method", "text"),
    ("relativistic_spin", "text"),
    ("electric_origin", "bohr"),
    ("relativistic_xc_angular_tolerance", "hartree"),
    ("nambu_normal_real", "hartree"),
    ("nambu_normal_imaginary", "hartree"),
    ("nambu_normal_direction_real", "hartree"),
    ("nambu_normal_direction_imaginary", "hartree"),
    ("nambu_pairing_real", "hartree"),
    ("nambu_pairing_imaginary", "hartree"),
    ("nambu_pairing_direction_real", "hartree"),
    ("nambu_pairing_direction_imaginary", "hartree"),
    ("nambu_overlap_real", "dimensionless"),
    ("nambu_overlap_imaginary", "dimensionless"),
    ("nambu_overlap_direction_real", "dimensionless"),
    ("nambu_overlap_direction_imaginary", "dimensionless"),
    ("nambu_probe_normal_real", "dimensionless"),
    ("nambu_probe_normal_imaginary", "dimensionless"),
    ("nambu_probe_pairing_real", "dimensionless"),
    ("nambu_probe_pairing_imaginary", "dimensionless"),
    ("nambu_probe_interaction", "hartree"),
    ("spectrum_separator", "hartree"),
    ("metric_eigenvalue_cutoff", "dimensionless"),
    ("wavevector_direction", "1/bohr"),
    ("lattice_flexoelectric", "flag"),
    ("born_charge_rows", "e"),
    ("scdft_pairing", "flag"),
    ("scdft_screening", "text"),
    ("phonon_phase", "text"),
    ("band_energies", "hartree"),
    ("band_energy_direction", "hartree"),
    ("chemical_potential", "hartree"),
    ("chemical_potential_direction", "hartree"),
    ("phonon_energies", "hartree"),
    ("pair_couplings_squared", "hartree^2"),
    ("screened_pair_coulomb", "hartree"),
    ("a2f_points", "integer"),
    ("abc_cutoff", "bohr"),
    ("adiis_until_error", "hartree"),
    ("amplitudes", "flag"),
    ("anion_multiplicity", "integer"),
    ("band", "integer"),
    ("basis", "text"),
    ("berry_mesh", "integer"),
    ("broadening_ev", "eV"),
    ("c", "text"),
    ("card_format", "text"),
    ("card_path", "text"),
    ("carrier_concentrations_cm3", "cm^-3"),
    ("cation_multiplicity", "integer"),
    ("cell", "bohr"),
    ("charge", "integer"),
    ("charge_compensation", "text"),
    ("cn_cutoff", "bohr"),
    ("coefficients", "flag or text"),
    ("conv_tol", "hartree"),
    ("conv_tol_dp", "dimensionless"),
    ("conv_tol_grad", "hartree"),
    ("coordination_number", "dimensionless"),
    ("countercharge_center", "bohr"),
    ("countercharge_width", "bohr"),
    ("critical_porosity", "dimensionless"),
    ("current_points", "bohr"),
    ("damping", "dimensionless"),
    ("degeneracy_tolerance", "hartree"),
    ("delta_window", "dimensionless"),
    ("density", "flag"),
    ("density_fit", "text"),
    ("diis_size", "integer"),
    ("direction", "dimensionless"),
    ("directions", "dimensionless"),
    ("disp", "text"),
    ("disp_cutoff", "bohr"),
    ("dispersion", "text"),
    ("dispersion_energy_scale", "dimensionless"),
    ("dispersion_free_c6", "hartree*bohr^6"),
    ("dispersion_free_polarizabilities", "bohr^3"),
    ("dispersion_free_radii", "bohr"),
    ("dispersion_radius_scale", "dimensionless"),
    ("mbd_frequency_scale", "hartree"),
    ("mbd_quadrature_nodes", "integer"),
    ("vdw_df_angular_degree", "integer"),
    ("vdw_df_momentum_cutoff", "dimensionless"),
    ("vdw_df_radial_order", "integer"),
    ("vdw_df_table_points", "integer"),
    ("dos_mesh", "integer"),
    ("dos_points", "integer"),
    ("dos_sigma", "cm^-1"),
    ("dry_bulk_gpa", "GPa"),
    ("dry_shear_gpa", "GPa"),
    ("dry_tensor_gpa", "GPa"),
    ("ecp", "text"),
    ("effective_pressure_gpa", "GPa"),
    ("electron_offset", "electrons"),
    ("electron_smearing", "hartree"),
    ("electronic_temperature", "K"),
    ("electronic_degeneracy", "integer"),
    ("gibbs", "flag"),
    ("electrostatic_order", "integer"),
    ("eliashberg_cutoff_cm1", "cm^-1"),
    ("eliashberg_t_min_k", "K"),
    ("energy_window", "hartree"),
    ("energy_window_ev", "eV"),
    ("eps", "dimensionless"),
    ("ewald_alpha", "1/bohr"),
    ("ewald_real_cutoff", "bohr"),
    ("ewald_reciprocal_cutoff", "1/bohr"),
    (
        "exchange_coefficients",
        "[dimensionless, dimensionless, 1/bohr] full fraction, long-range fraction, range-separation parameter",
    ),
    ("exchange_cutoff", "bohr"),
    ("exchange_enhanced", "flag"),
    ("fermi_k_mesh", "integer"),
    ("field", "hartree/e/bohr"),
    ("electric_directions", "hartree/e/bohr"),
    ("field_direction", "hartree/e/bohr per dimensionless path parameter"),
    ("field_amplitude", "dimensionless"),
    ("field_scalar", "hartree"),
    ("field_vector_imag", "atomic units, reduced vector potential"),
    ("field_vector_real", "atomic units, reduced vector potential"),
    ("field_wave_indices", "dimensionless"),
    ("fluid_bulk_gpa", "GPa"),
    ("fluid_conductivity_w_mk", "W/m/K"),
    ("fluid_density_kg_m3", "kg/m^3"),
    ("fluid_viscosity_pa_s", "Pa*s"),
    ("forces", "flag"),
    ("frame", "text"),
    ("gap_temperatures_k", "K"),
    ("gradient", "flag"),
    ("grid_angular_scheme", "text"),
    ("grid_cutoff", "bohr"),
    ("grid_level", "integer"),
    ("grid_preset", "text"),
    ("grid_weight_cutoff", "bohr^3"),
    ("grid_tolerance", "electrons"),
    ("grid_shell", "integer"),
    ("gtol", "hartree/bohr"),
    ("guess", "text"),
    ("guess_mix", "dimensionless"),
    ("hall_onsager_tolerance", "dimensionless"),
    ("hartree_compact_exponent", "1/bohr^2"),
    ("history", "integer"),
    ("incore_mb", "MiB"),
    ("ions", "text"),
    ("k_mesh", "integer"),
    ("k_point", "reduced coordinates"),
    ("labels", "text"),
    ("level_shift", "hartree"),
    ("magnetic_exchange_auxiliary_width", "1/bohr"),
    ("magnetic_field", "atomic units of magnetic field"),
    ("magnetic_grid_order", "integer"),
    ("magnetic_model", "text"),
    ("magnetic_open_order", "integer"),
    ("magnetic_operator_tolerance", "atomic units"),
    ("magnetic_origin", "bohr"),
    ("magnetic_reciprocal_cutoff", "1/bohr"),
    ("max_iter", "integer"),
    ("max_rank", "integer"),
    ("multipole_rank", "integer"),
    ("max_steps", "integer"),
    ("mineral_bulk_gpa", "GPa"),
    ("mineral_conductivity_w_mk", "W/m/K"),
    ("mineral_density_kg_m3", "kg/m^3"),
    ("mineral_shear_gpa", "GPa"),
    ("mineral_tensor_gpa", "GPa"),
    ("mode_cutoff_cm1", "cm^-1"),
    ("mu_star", "dimensionless"),
    ("mult", "integer"),
    ("multiplicity", "integer"),
    ("no_adiis", "flag"),
    ("no_soscf", "flag"),
    ("scf_optimizer", "enum"),
    ("nstates", "integer"),
    ("nuclear", "flag"),
    ("nuclear_g", "dimensionless"),
    ("gauge", "dimensionless"),
    ("gauge_origin", "bohr"),
    ("nmr_reference_elements", "dimensionless"),
    ("nmr_reference_shieldings", "ppm"),
    ("nmr_broadening", "ppm"),
    ("nmr_points", "dimensionless"),
    ("nmr_margin", "ppm"),
    ("nmr_couplings", "flag"),
    ("nmr_spin_model", "dimensionless"),
    ("spin_response_constraint", "dimensionless"),
    ("nmr_spin_max_dimension", "dimensionless"),
    ("nmr_spectrometer_mhz", "dimensionless"),
    ("nuclear_spins", "dimensionless"),
    ("nuclei", "dimensionless"),
    ("exafs_absorber", "dimensionless"),
    ("exafs_s02", "dimensionless"),
    ("exafs_mfp", "bohr"),
    ("exafs_rmax", "bohr"),
    ("exafs_scattering", "[dimensionless, 1/bohr, bohr, dimensionless]"),
    ("exafs_k_range", "1/bohr"),
    ("exafs_k_weight", "dimensionless"),
    ("exafs_sigma2", "bohr^2"),
    ("exafs_r_points", "dimensionless"),
    (
        "exafs_scattering_angles",
        "[dimensionless, dimensionless, 1/bohr, bohr, dimensionless]",
    ),
    ("exafs_central_phase", "[1/bohr, dimensionless]"),
    ("exafs_ms_sigma2", "bohr^2"),
    ("xanes_element", "dimensionless"),
    ("core_orbitals", "dimensionless"),
    ("xanes_broadening", "hartree"),
    ("core_hole", "text"),
    ("fc_max_quanta", "dimensionless"),
    ("fc_max_total", "dimensionless"),
    ("fc_threshold", "dimensionless"),
    ("fc_model", "dimensionless"),
    ("fc_max_hot", "dimensionless"),
    ("fc_broadening", "hartree"),
    ("frequencies", "hartree"),
    ("frequency_damping", "hartree"),
    ("frequency", "hartree"),
    ("response_omega", "hartree"),
    ("response_damping", "hartree"),
    ("site_a", "dimensionless"),
    ("site_b", "dimensionless"),
    ("unpaired", "dimensionless"),
    ("sites", "dimensionless"),
    ("probe", "dimensionless"),
    ("magneto_optical_source", "dimensionless"),
    ("second_probe", "dimensionless"),
    ("gruneisen_shear", "flag"),
    ("gruneisen_order", "integer"),
    ("gruneisen_force_tolerance", "hartree/bohr"),
    ("relaxed_ion", "flag"),
    ("nbo_spin", "text"),
    ("nbo_lewis_threshold", "dimensionless (fraction of the maximum occupancy)"),
    ("nbo_three_center", "flag"),
    ("nbo_perturbation_threshold", "hartree"),
    ("nbo_density", "text"),
    ("nbo_cas_state", "integer"),
    ("nbo_resonance", "flag"),
    ("nbo_resonance_structures", "integer"),
    ("nbo_spin_density", "flag"),
    ("nbo_spin_density_points", "bohr"),
    ("nbo_natural_orbitals", "flag"),
    ("nbo_corresponding_orbitals", "flag"),
    ("igmh_points", "bohr"),
    ("igmh_grid", "bohr"),
    ("igmh_grid_shape", "integer"),
    ("igmh_gradient", "text"),
    ("igmh_atom_details", "flag"),
    ("fragments", "integer"),
    ("fragment_charges", "integer"),
    ("fragment_multiplicities", "integer"),
    ("sapt_variant", "text"),
    ("sapt_regularization_eta", "bohr^-2"),
    ("sapt_relaxed_induction30", "flag"),
    ("sapt_correlation_terms", "flag"),
    ("asymptotic_shift_a", "hartree"),
    ("asymptotic_shift_b", "hartree"),
    ("rpa_dispersion", "flag"),
    ("casimir_polder_points", "integer"),
    ("charge_transfer", "flag"),
    ("sapt_partition", "flag"),
    ("use_symmetry", "flag"),
    ("bfield", "atomic units of magnetic flux density"),
    ("mcd_model", "text"),
    ("mcd_direction", "dimensionless light propagation direction"),
    ("mcd_frozen_core", "integer spatial orbital count"),
    ("mcd_active_orbitals", "integer spatial orbital count"),
    ("mcd_orientation", "fixed or isotropic"),
    ("mcd_orientation_order", "integer angular quadrature order"),
    ("mcd_magnetic_basis", "common-origin or london-linear-frozen-soc"),
    ("spectrum_width", "hartree HWHM"),
    ("mcd_thermal_hamiltonian", "hartree"),
    ("mcd_thermal_magnetic_x", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_magnetic_y", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_magnetic_z", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_helicity_x", "dimensionless normalized optical helicity"),
    ("mcd_thermal_helicity_y", "dimensionless normalized optical helicity"),
    ("mcd_thermal_helicity_z", "dimensionless normalized optical helicity"),
    ("mcd_thermal_hamiltonian_imag", "hartree"),
    ("mcd_thermal_magnetic_x_imag", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_magnetic_y_imag", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_magnetic_z_imag", "atomic units of magnetic dipole moment"),
    ("mcd_thermal_helicity_x_imag", "dimensionless normalized optical helicity"),
    ("mcd_thermal_helicity_y_imag", "dimensionless normalized optical helicity"),
    ("mcd_thermal_helicity_z_imag", "dimensionless normalized optical helicity"),
    ("spin_zeeman", "flag"),
    ("bfield_series_order", "integer"),
    ("bfield_initial_guess", "auto, core or zero-field"),
    ("double_hybrid_model", "named double-hybrid recipe or custom"),
    (
        "double_hybrid_target_xc",
        "semilocal or hybrid functional evaluated on custom parent orbitals",
    ),
    ("pt2_same_spin", "dimensionless"),
    ("pt2_opposite_spin", "dimensionless"),
    ("pt2_memory_mb", "MiB"),
    ("response_order", "integer"),
    ("coordinates", "integer"),
    ("force_coordinates", "integer"),
    ("patterns", "bohr"),
    ("nuclear_directions", "bohr"),
    ("response_tol", "dimensionless"),
    ("response_max_iter", "integer"),
    ("field_directions", "hartree/e/bohr per dimensionless path parameter"),
    ("reference_field", "hartree/e/bohr"),
    ("tensor_assembly", "text"),
    ("electric_assembly", "text"),
    ("nuclear_order", "integer"),
    ("derivative_nuclei", "integer"),
    ("nuclear_direction", "dimensionless"),
    ("q_points", "reduced coordinates of the reciprocal lattice"),
    ("berry_dipole_doping", "electrons per cell"),
    ("spiral_q", "reduced coordinates of the reciprocal lattice"),
    ("spiral_axis", "dimensionless"),
    ("spiral_seed_field", "atomic units of magnetic field"),
    ("spiral_max_supercell", "integer"),
    ("dmi_self_consistent", "flag"),
    ("site_sizes", "dimensionless"),
    ("soc_charges", "e"),
    ("spin_axes", "dimensionless"),
    ("spin_axis", "dimensionless"),
    ("spin_orbit_self_consistent", "flag"),
    ("mrsf_coupling", "dimensionless"),
    ("mrsf_state", "text"),
    ("spin_flip_kernel", "dimensionless"),
    ("max_states", "dimensionless"),
    ("dc_fragments", "dimensionless"),
    ("dc_buffer", "bohr"),
    ("dc_smearing", "hartree"),
    ("nuclear_soc_scale", "dimensionless"),
    ("nuclear_widths", "bohr"),
    ("omega", "1/bohr"),
    ("origin", "bohr (or the name of an origin)"),
    ("overlap_cutoff", "bohr"),
    ("pade_points", "integer"),
    ("path", "reduced coordinates of path vertices, or a file name"),
    ("permeability_m2", "m^2"),
    (
        "phases",
        "[dimensionless, GPa, GPa] rows: volume fraction, bulk modulus, shear modulus",
    ),
    ("phonon_smearing_cm1", "cm^-1"),
    ("points", "integer"),
    ("population", "text"),
    ("pore_aspect_ratio", "dimensionless"),
    ("porosity", "dimensionless"),
    ("pressure", "Pa"),
    ("pressure_gpa", "GPa"),
    ("projected", "flag"),
    ("q_mesh", "integer"),
    ("relax_ions", "flag"),
    ("relaxation_time_fs", "fs"),
    ("response_accept_residual", "dimensionless"),
    ("response_divergence_factor", "dimensionless"),
    ("response_max_iterations", "integer"),
    ("response_min_denominator", "hartree"),
    ("response_preconditioner_floor", "hartree"),
    ("response_restart", "integer"),
    ("response_reciprocal_cutoff", "1/bohr"),
    ("response_q", "1/bohr"),
    ("spin_probe_field", "atomic units of magnetic field"),
    ("response_tolerance", "dimensionless"),
    ("root", "integer"),
    ("rotation", "dimensionless"),
    ("route", "text"),
    ("scales", "dimensionless"),
    ("scattering_hartree", "hartree"),
    ("scattering_table", "[hartree, fs] columns: absolute band energy, lifetime"),
    ("sigma", "hartree"),
    ("skala_model", "text"),
    ("soc_scale", "dimensionless"),
    ("solvent", "text"),
    ("spin", "integer"),
    ("spin_g_factor", "dimensionless"),
    ("spin_field_imag", "atomic units of magnetic field"),
    ("spin_field_real", "atomic units of magnetic field"),
    ("spinor_amplitude", "dimensionless"),
    ("spinor_operator_imag", "hartree"),
    ("spinor_operator_real", "hartree"),
    ("stabilize", "flag"),
    ("step_max", "bohr"),
    ("steps", "dimensionless"),
    ("stress", "flag"),
    ("stress_tol", "hartree/bohr^3"),
    ("supercell", "integer"),
    ("symmetry_number", "integer"),
    ("vibrational_projection", "text"),
    ("td_max_iterations", "integer"),
    ("td_method", "text"),
    ("td_nto", "flag"),
    ("td_difference_density", "flag"),
    ("td_spin", "text"),
    ("td_tolerance", "hartree"),
    ("td_gradient_route", "text"),
    ("temperature", "K"),
    ("raman_frequency", "hartree"),
    ("temperatures", "K"),
    ("temperatures_k", "K"),
    ("three_body", "flag"),
    ("topology_fixed_k", "reduced coordinates"),
    ("topology_mesh", "integer"),
    ("topology_plane_axes", "integer"),
    ("wannier_loop_divisions", "integer"),
    ("wannier_flow_points", "integer"),
    ("transport_frequencies_hartree", "hartree"),
    ("transport_k_mesh", "integer"),
    ("transport_mu_shift", "hartree"),
    ("transport_mu_shifts", "hartree"),
    ("transport_temperature_k", "K"),
    ("transport_temperatures_k", "K"),
    ("uniform_magnetic_field", "atomic units of magnetic field, reduced"),
    ("unrestricted", "flag"),
    ("vv10_grid_level", "integer"),
    ("vv10", "flag"),
    ("vv10_b", "dimensionless"),
    ("vv10_c", "dimensionless"),
    ("vv10_cutoff", "bohr"),
    ("magnetic_direction", "atomic units of magnetic field"),
    ("magnetic_phase_order", "integer"),
    ("cas", "integer"),
    ("derivative_order", "integer"),
    ("cas_roots", "integer"),
    ("state_weights", "dimensionless"),
    ("active_orbitals", "integer"),
    ("casci", "flag"),
    ("casscf_max_iter", "integer"),
    ("casscf_gradient_tol", "hartree"),
    ("casscf_energy_tol", "hartree"),
    ("ci_tol", "hartree"),
    ("ci_max_iter", "integer"),
    ("on_top_functional", "text"),
    ("pdft_lambda", "dimensionless"),
    ("pdft_method", "text"),
    ("x", "text"),
    ("xc", "text"),
    ("xc_parameters", "per-parameter units from the selected functional"),
    ("xc_terms", "dimensionless"),
    ("weight", "dimensionless"),
    ("xc_ao_cutoff", "bohr"),
];

/// Labels of report entries that a calculation does not label itself, by
/// entry name.
pub const REPORT_UNITS: &[(&str, &str)] = &[
    ("igmh_points", "bohr"),
    ("igmh_grid_shape", "integer"),
    ("igmh_gradient", "text"),
    ("igmh_proatoms", "text"),
    ("igmh_vacuum_points", "integer"),
    ("igmh_density", "1/bohr^3"),
    ("igmh_signed_density", "1/bohr^3"),
    ("igmh_atom_densities", "1/bohr^3"),
    ("igmh_density_residual", "1/bohr^3"),
    ("igmh_density_gradient", "1/bohr^4"),
    ("igmh_gradient_residual", "1/bohr^4"),
    ("igmh_atom_vectors", "1/bohr^4"),
    ("igmh_delta_g", "1/bohr^4"),
    ("igmh_delta_g_inter", "1/bohr^4"),
    ("igmh_delta_g_intra", "1/bohr^4"),
    ("igmh_density_hessian", "1/bohr^5"),
    ("dispersion_effective_c6", "hartree*bohr^6"),
    ("dispersion_effective_polarizabilities", "bohr^3"),
    ("dispersion_effective_radii", "bohr"),
    ("dispersion_energy", "hartree"),
    ("dispersion_energy_scale", "dimensionless"),
    ("dispersion_radius_scale", "dimensionless"),
    ("dispersion_volume_ratios", "dimensionless"),
    ("mbd_coupled_frequencies", "hartree"),
    ("mbd_frequency_scale", "hartree"),
    ("mbd_frequencies", "hartree"),
    ("mbd_minimum_screening_eigenvalue", "dimensionless"),
    ("mbd_minimum_squared_frequency", "hartree^2"),
    ("mbd_quadrature_nodes", "integer"),
    ("mbd_screening_residual", "dimensionless"),
    ("mbd_unscreened_c6", "hartree*bohr^6"),
    ("mbd_unscreened_polarizabilities", "bohr^3"),
    ("mbd_unscreened_radii", "bohr"),
    ("vdw_df_angular_degree", "integer"),
    ("vdw_df_exchange_energy", "hartree"),
    ("vdw_df_local_correlation_energy", "hartree"),
    ("vdw_df_momentum_cutoff", "dimensionless"),
    ("vdw_df_nonlocal_correlation_energy", "hartree"),
    ("vdw_df_parent_energy", "hartree"),
    ("vdw_df_parent_xc_energy", "hartree"),
    ("vdw_df_radial_order", "integer"),
    ("vdw_df_table_points", "integer"),
    ("vdw_df_xc_energy", "hartree"),
    ("scf_orbital_residual", "hartree"),
    ("occupied_virtual_gaps", "hartree"),
    ("orbital_response_residuals", "hartree/bohr"),
    (
        "particle_number_diagnostics",
        "dimensionless, electron numbers and normalized discrepancies",
    ),
    ("energy_difference", "hartree"),
    ("overlap_weights", "dimensionless"),
    ("boundary_gaps", "dimensionless"),
    ("density_residual", "dimensionless"),
    ("fixed_occupation_residual", "dimensionless"),
    ("fixed_occupation_gap", "hartree"),
    ("vv10_b", "dimensionless"),
    ("vv10_c", "dimensionless"),
    ("vv10_cutoff", "bohr"),
    ("a2f_frequencies_cm1", "cm^-1"),
    ("abc_cutoff", "bohr"),
    ("amplitudes", "dimensionless"),
    ("angular_frequency_hartree", "hartree"),
    ("anion_energy", "hartree"),
    ("anisotropy", "e^2*bohr^2/hartree"),
    ("b0_prime", "dimensionless"),
    ("band_energies", "hartree"),
    ("band_energy", "hartree"),
    ("biot_coefficient", "dimensionless"),
    ("cation_energy", "hartree"),
    ("cell", "bohr"),
    ("charge_hardness", "hartree"),
    ("charge_spin_coupling", "hartree"),
    ("charge_spin_coupling_transpose", "hartree"),
    ("charge_sum_residual", "e"),
    ("chemical_potential", "hartree"),
    ("cn_cutoff", "bohr"),
    ("conduction_minimum", "hartree"),
    ("delta_iw0_hartree", "hartree"),
    ("delta_real_hartree", "hartree"),
    ("delta_window", "dimensionless"),
    ("density", "dimensionless"),
    ("density_change", "dimensionless"),
    ("density_imag", "dimensionless"),
    ("density_real", "dimensionless"),
    ("depolarization_ratios", "dimensionless"),
    ("dipole_norm", "e*bohr"),
    ("dipole_norm_debye", "debye"),
    ("direct_gap", "hartree"),
    ("direction", "dimensionless"),
    ("disp_cutoff", "bohr"),
    ("doping_electrons_per_cell", "electrons"),
    ("dos_at_fermi_level", "states/hartree"),
    ("dos_at_fermi_level_per_ev", "states/eV"),
    ("dos_frequencies_cm1", "cm^-1"),
    ("dry_shear_gpa", "GPa"),
    ("dual_descriptor", "electrons"),
    ("eigenvalue", "hartree"),
    ("electron_count_residual", "electrons"),
    ("electronegativity", "hartree"),
    ("electronic_temperature_kelvin", "K"),
    ("electrons", "electrons"),
    ("electrophilicity", "hartree"),
    ("energy", "hartree"),
    ("energy_after", "hartree"),
    ("energy_before", "hartree"),
    ("energy_change", "hartree"),
    ("energy_cm1", "cm^-1"),
    ("energy_components", "hartree"),
    ("energy_ev", "eV"),
    ("enthalpy", "hartree"),
    ("enthalpy_correction", "hartree"),
    ("entropy_term", "hartree"),
    ("external_pressure_gpa", "GPa"),
    (
        "field_vector_imag",
        "atomic units, reduced vector potential",
    ),
    (
        "field_vector_real",
        "atomic units, reduced vector potential",
    ),
    ("field_wave_indices", "dimensionless"),
    ("final_fock_change", "hartree"),
    ("fock_imag", "hartree"),
    ("fock_real", "hartree"),
    ("forces", "hartree/bohr"),
    ("free_energy", "hartree"),
    ("mermin_free_energy", "hartree"),
    ("helmholtz_free_energy", "hartree"),
    ("reference_mermin_free_energy", "hartree"),
    ("delta_mermin_free_energy", "hartree"),
    ("fukui_minus", "electrons"),
    ("fukui_plus", "electrons"),
    ("fukui_zero", "electrons"),
    ("gap", "hartree"),
    ("gibbs_correction", "hartree"),
    ("gibbs_free_energy", "hartree"),
    ("ensemble_response_residuals", "dimensionless"),
    ("ensemble_number_defects", "electrons/bohr"),
    ("grad_max", "hartree/bohr"),
    ("gradient", "hartree/bohr"),
    ("grid_number_defect", "electrons"),
    ("gruneisen", "dimensionless"),
    ("hardness", "hartree"),
    ("hardness_matrix_symmetry_defect", "hartree"),
    ("hill_bulk_gpa", "GPa"),
    ("hill_shear_gpa", "GPa"),
    ("initial_energy", "hartree"),
    ("integrated", "states"),
    ("internal_energy", "hartree"),
    ("internal_energy_correction", "hartree"),
    ("isotropic", "e^2*bohr^2/hartree"),
    ("isotropic_angstrom3", "angstrom^3"),
    ("k_path_reduced", "reduced coordinates"),
    ("k_point_reduced", "reduced coordinates"),
    ("k_points_reduced", "reduced coordinates"),
    ("kt", "hartree"),
    ("lambda", "dimensionless"),
    ("lo_directions", "dimensionless"),
    ("local_electrophilicity", "electrons*hartree"),
    ("local_softness", "electrons/hartree"),
    ("local_softness_minus", "electrons/hartree"),
    ("mesh_frequencies_cm1", "cm^-1"),
    ("metric_defect", "dimensionless"),
    ("mode_lambda", "dimensionless"),
    ("mu_star", "dimensionless"),
    ("nuclear_soc_scale", "dimensionless"),
    ("objective", "hartree"),
    ("occupations", "electrons"),
    ("oscillator_strength", "dimensionless"),
    ("oscillator_strength_sum", "dimensionless"),
    ("overlap", "dimensionless"),
    ("path_vertices", "reduced coordinates"),
    ("poisson_drained", "dimensionless"),
    ("poisson_ratio", "dimensionless"),
    ("poisson_undrained", "dimensionless"),
    ("positions", "bohr"),
    ("pressure_pascal", "Pa"),
    ("principal_axes", "dimensionless"),
    ("q", "reduced coordinates"),
    ("q_path_reduced", "reduced coordinates"),
    ("reference_level", "hartree"),
    ("relative_residual", "dimensionless"),
    ("requested_concentration_cm3", "cm^-3"),
    ("response_number_defect", "electrons"),
    ("response_relative_residual", "dimensionless"),
    ("reuss_bulk_gpa", "GPa"),
    ("reuss_shear_gpa", "GPa"),
    ("rotation", "dimensionless"),
    ("rotatory_strength", "atomic units"),
    ("rotatory_strength_cgs", "1e-40 erg cm^3"),
    ("rotatory_strength_velocity", "atomic units"),
    ("rotatory_strength_velocity_cgs", "1e-40 erg cm^3"),
    ("s_squared", "hbar^2"),
    ("saturated_shear_gpa", "GPa"),
    ("scales", "dimensionless"),
    ("sigma", "hartree"),
    ("skempton_b", "dimensionless"),
    ("soc_scale", "dimensionless"),
    ("softness", "1/hartree"),
    ("spin_chemical_potential", "hartree"),
    ("spin_chemical_potential_minus", "hartree"),
    ("spin_fukui_minus", "electrons"),
    ("spin_fukui_ns", "electrons"),
    ("spin_fukui_sn", "electrons"),
    ("spin_fukui_ss", "electrons"),
    ("spin_number", "dimensionless"),
    ("spin_population_difference", "electrons"),
    ("static_b0_gpa", "GPa"),
    ("static_b0_prime", "dimensionless"),
    ("static_v0", "bohr^3"),
    ("steps", "dimensionless"),
    ("stress_max", "hartree/bohr^3"),
    ("supercell_force_constants", "hartree/bohr^2"),
    ("temperature_k", "K"),
    ("temperature_kelvin", "K"),
    ("tensor_eigenvalues_gpa", "GPa"),
    ("theta", "radians"),
    ("transition_angular_momentum", "hbar"),
    ("transition_dipole", "e*bohr"),
    ("transition_momentum", "hbar/bohr"),
    ("universal_anisotropy", "dimensionless"),
    ("valence_maximum", "hartree"),
    ("voigt_bulk_gpa", "GPa"),
    ("voigt_shear_gpa", "GPa"),
    ("volume", "bohr^3"),
    ("wavelength_nm", "nm"),
    ("weight", "dimensionless"),
    ("z_iw0", "dimensionless"),
    ("zero_point_energy", "hartree"),
    ("acoustic_residual", "hartree/bohr^2"),
    ("angular_energy_defect", "hartree"),
    ("angular_operator_defect", "hartree"),
    ("branch_effective_masses", "m_e"),
    ("branch_velocities", "hartree*bohr/hbar"),
    ("component_energy_defect", "hartree"),
    ("fock_image_residual", "hartree"),
    ("force_constant_asymmetry", "hartree/bohr^2"),
    ("hermiticity_defect", "atomic units"),
    ("imaginary_residual", "atomic units"),
    ("integer_residual", "dimensionless"),
    ("inverse_mass_eigenvalues", "1/m_e"),
    ("k_points", "1/bohr"),
    ("magnetic_operator_tolerance", "atomic units"),
    ("max_plaquette_phase", "radians"),
    ("min_link_log_abs_determinant", "dimensionless"),
    ("nuclear_soc_hermitian_defect", "hartree"),
    ("nuclear_soc_quadrature_defect", "hartree"),
    ("occupation_consistency", "electrons"),
    ("orbital_density_residual", "dimensionless"),
    ("orbital_fock_residual", "hartree"),
    ("plaquette_phases", "radians"),
    ("projection_sum_residual", "states/hartree"),
    ("raman_permutation_defect", "atomic units"),
    ("sampled_min_gap", "hartree"),
    ("spin_weight", "dimensionless"),
    ("spinful_density_residual", "dimensionless"),
    ("spinful_fock_residual", "hartree"),
    ("strain_derivative_terms", "hartree"),
    ("stress_gpa", "GPa"),
    ("string_spread", "radians"),
    ("translation_residual", "hartree/bohr"),
    ("young_gpa", "GPa"),
    ("orbital_rotations", "atomic units"),
    ("density_derivative", "atomic units"),
    ("mulliken_spin", "electrons"),
];

/// Unit label of one control, or `None` for a name the driver does not read.
pub fn control_unit(name: &str) -> Option<&'static str> {
    CONTROL_UNITS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, unit)| *unit)
}

/// [`CONTROL_UNITS`] as a report object.
pub fn control_units_value() -> Value {
    let mut units = Value::object();
    for (name, unit) in CONTROL_UNITS {
        units.set(name, *unit);
    }
    units
}

fn report_unit(name: &str) -> Option<&'static str> {
    REPORT_UNITS
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, unit)| *unit)
}

/// True when the environment asks the driver to refuse a report with
/// incomplete or malformed unit labels (`DFT_RS_STRICT_UNITS`).
pub fn strict() -> bool {
    std::env::var_os("DFT_RS_STRICT_UNITS").is_some()
}

fn word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn word_end(text: &str, from: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = from;
    while end < bytes.len() && word_byte(bytes[end]) {
        end += 1;
    }
    end
}

/// One unit expression: its Hartree and Bohr powers, whether one of them
/// names a report entry instead of a number, and its length in bytes.
struct Expression {
    powers: [i32; 2],
    symbolic: bool,
    length: usize,
}

fn expression(text: &str) -> Option<Expression> {
    let bytes = text.as_bytes();
    let mut out = Expression {
        powers: [0, 0],
        symbolic: false,
        length: 0,
    };
    let mut position = 0;
    let mut sign = 1;
    loop {
        let end = word_end(text, position);
        if end == position {
            return None;
        }
        let name = text[position..end].to_ascii_lowercase();
        position = end;
        // `None` is a power read from another entry of the same object.
        let mut power = Some(1_i32);
        if bytes.get(position) == Some(&b'^') {
            position += 1;
            if bytes.get(position) == Some(&b'(') {
                let close = position + text[position..].find(')')?;
                let (entry, offset) = text[position + 1..close].split_once(['-', '+'])?;
                let named = !entry.is_empty() && entry.bytes().all(word_byte);
                let counted = !offset.is_empty() && offset.bytes().all(|b| b.is_ascii_digit());
                if !named || !counted {
                    return None;
                }
                power = None;
                position = close + 1;
            } else {
                let mut digits = position;
                if bytes.get(digits) == Some(&b'-') {
                    digits += 1;
                }
                let mut end = digits;
                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                if end > digits {
                    power = Some(text[position..end].parse().ok()?);
                } else {
                    if digits != position {
                        return None;
                    }
                    end = word_end(text, position);
                    if end == position {
                        return None;
                    }
                    power = None;
                }
                position = end;
            }
        }
        let slot = match name.as_str() {
            "hartree" => Some(0),
            "bohr" => Some(1),
            _ => None,
        };
        if let Some(slot) = slot {
            match power {
                Some(power) => out.powers[slot] += sign * power,
                None => out.symbolic = true,
            }
        }
        match bytes.get(position) {
            Some(b'*') => sign = 1,
            Some(b'/') => sign = -1,
            _ => {
                out.length = position;
                return Some(out);
            }
        }
        position += 1;
    }
}

/// The leading unit of a label: Hartree and Bohr powers (one pair, or one
/// per element of a bracketed list) and the remarks that follow.
struct Leading<'a> {
    powers: Vec<[i32; 2]>,
    symbolic: bool,
    remarks: Vec<&'a str>,
}

fn leading(label: &str) -> Option<Leading<'_>> {
    let text = label.trim();
    if let Some(inner) = text.strip_prefix('[') {
        let close = inner.find(']')?;
        let mut out = Leading {
            powers: Vec::new(),
            symbolic: false,
            remarks: vec![&inner[close + 1..]],
        };
        for item in inner[..close].split(',') {
            let item = item.trim();
            let parsed = expression(item)?;
            let rest = &item[parsed.length..];
            if !(rest.is_empty() || rest.starts_with([' ', ';', '('])) {
                return None;
            }
            out.powers.push(parsed.powers);
            out.symbolic |= parsed.symbolic;
            out.remarks.push(rest);
        }
        Some(out)
    } else {
        let parsed = expression(text)?;
        let rest = &text[parsed.length..];
        if !(rest.is_empty() || rest.starts_with([' ', ',', ';', '('])) {
            return None;
        }
        Some(Leading {
            powers: vec![parsed.powers],
            symbolic: parsed.symbolic,
            remarks: vec![rest],
        })
    }
}

/// `[hartree, bohr]` powers of the leading unit expression of a label: one
/// pair, or one per element of a bracketed list. `None` when the label does
/// not begin with a unit expression or a power names a report entry.
pub fn leading_powers(label: &str) -> Option<Vec<[i32; 2]>> {
    leading(label)
        .filter(|unit| !unit.symbolic)
        .map(|unit| unit.powers)
}

/// True when `text` names Hartree or Bohr as a word of its own.
fn names_atomic_unit(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    ["hartree", "bohr"].iter().any(|word| {
        lower.match_indices(word).any(|(at, _)| {
            let before = at > 0 && word_byte(bytes[at - 1]);
            let after = bytes.get(at + word.len()).is_some_and(|b| word_byte(*b));
            !before && !after
        })
    })
}

/// Checks that a label is not empty and names Hartree and Bohr only inside
/// its leading unit expression.
pub fn check_label(label: &str) -> std::result::Result<(), String> {
    if label.trim().is_empty() {
        return Err("the label is empty".into());
    }
    if !names_atomic_unit(label) {
        return Ok(());
    }
    let Some(unit) = leading(label) else {
        return Err(format!(
            "`{label}` names Hartree or Bohr but does not begin with a unit expression"
        ));
    };
    if unit.remarks.iter().any(|remark| names_atomic_unit(remark)) {
        return Err(format!(
            "`{label}` names Hartree or Bohr outside its leading unit expression"
        ));
    }
    Ok(())
}

fn holds_objects(value: &Value) -> bool {
    match value {
        Value::Object(_) => true,
        Value::Array(items) => items.iter().any(holds_objects),
        _ => false,
    }
}

fn holds_reals(value: &Value) -> bool {
    match value {
        Value::Float(_) => true,
        Value::Array(items) => items.iter().any(holds_reals),
        Value::Object(entries) => entries.iter().any(|(_, item)| holds_reals(item)),
        _ => false,
    }
}

struct Labels {
    known: Vec<(String, Value)>,
    problems: Vec<String>,
}

impl Labels {
    fn has(&self, name: &str) -> bool {
        self.known.iter().any(|(key, _)| key == name)
    }

    fn visit(&mut self, path: &str, key: &str, value: &Value) {
        if self.has(path) || self.has(key) || !holds_reals(value) {
            return;
        }
        // The table can label an object or an array as a whole.
        if let Some(unit) = report_unit(key) {
            self.known
                .push((key.to_string(), Value::Str(unit.to_string())));
            return;
        }
        match value {
            Value::Object(entries) => {
                for (child, item) in entries {
                    self.visit(&format!("{path}.{child}"), child, item);
                }
            }
            Value::Array(items) if holds_objects(value) => {
                for item in items {
                    self.visit(path, key, item);
                }
            }
            _ => {
                self.known
                    .push((path.to_string(), Value::Str(UNSPECIFIED.to_string())));
                self.problems.push(format!("`{path}` has no unit label"));
            }
        }
    }
}

/// Completes the `units` object of a report: every real-valued entry that
/// neither it nor an object or array above it labels receives the label of
/// its name in [`REPORT_UNITS`], or [`UNSPECIFIED`]. Returns what is wrong
/// with the labels: entries without one and labels that fail
/// [`check_label`].
pub fn complete(report: &mut Value) -> Vec<String> {
    let Value::Object(entries) = &*report else {
        return Vec::new();
    };
    let known = match entries.iter().find(|(key, _)| key == "units") {
        Some((_, Value::Object(units))) => units.clone(),
        _ => Vec::new(),
    };
    let mut labels = Labels {
        known,
        problems: Vec::new(),
    };
    for (key, value) in entries {
        if key != "units" {
            labels.visit(key, key, value);
        }
    }
    for (key, label) in &labels.known {
        if let Value::Str(text) = label {
            if let Err(problem) = check_label(text) {
                labels
                    .problems
                    .push(format!("unit label of `{key}`: {problem}"));
            }
        }
    }
    report.set("units", Value::Object(labels.known));
    labels.problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_begin_with_a_unit_expression() {
        assert_eq!(
            leading_powers("hartree/bohr^2 per cell"),
            Some(vec![[1, -2]])
        );
        assert_eq!(leading_powers("e*bohr"), Some(vec![[0, 1]]));
        assert_eq!(leading_powers("1/bohr, Cartesian"), Some(vec![[0, -1]]));
        assert_eq!(leading_powers("hartree/e/bohr"), Some(vec![[1, -1]]));
        assert_eq!(
            leading_powers("hartree*bohr/hbar (velocity)"),
            Some(vec![[1, 1]])
        );
        assert_eq!(
            leading_powers("electrons*hartree/hbar/bohr^2"),
            Some(vec![[1, -2]])
        );
        assert_eq!(
            leading_powers("cm^-1 (negative = imaginary)"),
            Some(vec![[0, 0]])
        );
        assert_eq!(
            leading_powers("Hartree per magnetic cell"),
            Some(vec![[1, 0]])
        );
        assert_eq!(
            leading_powers("[dimensionless, hartree, 1/bohr] columns: order, defect, wave"),
            Some(vec![[0, 0], [1, 0], [0, -1]])
        );
        // A power that names an entry is valid but has no fixed value.
        assert_eq!(leading_powers("e*bohr^rank"), None);
        assert!(check_label("e*bohr^rank").is_ok());
        assert!(check_label("e^(order+1)*bohr^(order+1)/hartree^order").is_ok());
        assert!(check_label("e*bohr^(rank-1); rows monomials, columns 3*atom+axis").is_ok());
        assert_eq!(leading_powers("J/(mol K) per mol of primitive cells"), None);
    }

    #[test]
    fn hartree_and_bohr_are_named_only_in_the_leading_expression() {
        for label in [
            "hartree",
            "hartree/K per primitive cell",
            "bohr^3 per primitive cell at each temperature",
            "K",
            "dimensionless, N_alpha minus N_beta",
            "J/(mol K) per mol of primitive cells",
            "atomic units of magnetic field",
            "objects with delta_iw0_hartree and z_iw0",
            "[dimensionless, dimensionless, 1/bohr] fractions and range separation",
        ] {
            assert!(check_label(label).is_ok(), "{label}");
        }
        for label in [
            "",
            "hartree bohr (atomic units of velocity)",
            "states per hartree per cell",
            "hartree/(bohr^2 electron mass)",
            "inverse Bohr",
            "bohr per (sqrt(amu)*bohr) of normal coordinate",
            "electrons per Bohr^3",
            "hartree per molecule; entropy hartree/K",
            "[dimensionless full, inverse Bohr omega]",
            "e*bohr^",
        ] {
            assert!(check_label(label).is_err(), "{label}");
        }
    }

    #[test]
    fn tables_are_well_formed() {
        for table in [CONTROL_UNITS, REPORT_UNITS] {
            for (index, (name, unit)) in table.iter().enumerate() {
                assert!(check_label(unit).is_ok(), "{name}: {unit}");
                assert!(
                    table[..index].iter().all(|(other, _)| other != name),
                    "{name} is listed twice"
                );
            }
        }
        for name in crate::driver::method::MOLECULAR_METHOD_CONTROLS {
            assert!(control_unit(name).is_some(), "{name}");
        }
        assert_eq!(control_unit("overlap_cutoff"), Some("bohr"));
        assert_eq!(control_unit("no_such_control"), None);
    }

    /// Every control name that a driver source reads has a unit.
    #[test]
    fn every_control_read_by_the_driver_has_a_unit() {
        const SOURCES: &[&str] = &[
            include_str!("electric.rs"),
            include_str!("multicomponent.rs"),
            include_str!("band_properties.rs"),
            include_str!("conceptual.rs"),
            include_str!("divide_conquer.rs"),
            include_str!("exafs.rs"),
            include_str!("exchange.rs"),
            include_str!("excited.rs"),
            include_str!("geomech.rs"),
            include_str!("epr.rs"),
            include_str!("gtensor.rs"),
            include_str!("hyperfine.rs"),
            include_str!("spin_spin_zfs.rs"),
            include_str!("magnetizability.rs"),
            include_str!("magnon.rs"),
            include_str!("mcd.rs"),
            include_str!("method.rs"),
            include_str!("molecular.rs"),
            include_str!("mrsf.rs"),
            include_str!("nmr.rs"),
            include_str!("no_pair.rs"),
            include_str!("two_component.rs"),
            include_str!("pairing.rs"),
            include_str!("superconducting.rs"),
            include_str!("optical.rs"),
            include_str!("spin_response.rs"),
            include_str!("spinorbit.rs"),
            include_str!("periodic.rs"),
            include_str!("periodic_roks.rs"),
            include_str!("roks.rs"),
            include_str!("vcd.rs"),
            include_str!("vibrational.rs"),
        ];
        const READERS: &[&str] = &[
            ".take(",
            ".f64(",
            ".opt_f64(",
            ".opt_i64(",
            ".opt_usize(",
            ".usize(",
            ".bool(",
            ".opt_str(",
            ".str(",
            ".opt_f64_list(",
            ".opt_vec3(",
            ".opt_usize_list(",
            ".opt_rows(",
            "required(controls,",
            "tensor(controls,",
        ];
        let mut missing: Vec<&str> = Vec::new();
        let mut seen = 0;
        for source in SOURCES {
            let body = source.split("#[cfg(test)]").next().unwrap();
            for reader in READERS {
                for (at, _) in body.match_indices(reader) {
                    let rest = body[at + reader.len()..].trim_start();
                    let Some(rest) = rest.strip_prefix('"') else {
                        continue;
                    };
                    let Some(end) = rest.find('"') else {
                        continue;
                    };
                    let name = &rest[..end];
                    let plain = !name.is_empty()
                        && name
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
                    if !plain {
                        continue;
                    }
                    seen += 1;
                    if control_unit(name).is_none() && !missing.contains(&name) {
                        missing.push(name);
                    }
                }
            }
        }
        assert!(seen >= 150, "only {seen} control reads were found");
        assert!(missing.is_empty(), "controls without a unit: {missing:?}");
    }

    #[test]
    fn reports_are_completed_from_the_table_and_problems_are_named() {
        let mut report = Value::object()
            .with("energy", -1.5)
            .with("iterations", 7_usize)
            .with("mesh", vec![2_usize, 2, 2])
            .with(
                "rows",
                vec![
                    Value::object()
                        .with("temperature_k", 300.0)
                        .with("chemical_potential", 0.25)
                        .with("band", 2_usize),
                    Value::object()
                        .with("temperature_k", 600.0)
                        .with("chemical_potential", 0.5)
                        .with("band", 3_usize),
                ],
            )
            .with(
                "thermo",
                Value::object().with("mystery", 1.25).with("label", "text"),
            )
            .with("covered", Value::object().with("mystery", vec![1.0, 2.0]))
            .with(
                "units",
                Value::object()
                    .with("rows.temperature_k", "K")
                    .with("covered", "hartree per cell"),
            );
        let problems = complete(&mut report);
        let units = report.get("units").unwrap();
        let label = |name: &str| units.get(name).and_then(Value::as_str);
        assert_eq!(label("energy"), Some("hartree"));
        assert_eq!(label("chemical_potential"), Some("hartree"));
        assert_eq!(label("rows.temperature_k"), Some("K"));
        assert_eq!(label("thermo.mystery"), Some(UNSPECIFIED));
        assert_eq!(label("iterations"), None);
        assert_eq!(label("mesh"), None);
        assert_eq!(label("mystery"), None);
        assert_eq!(
            problems,
            vec!["`thermo.mystery` has no unit label".to_string()]
        );

        let mut malformed = Value::object().with("velocity", 1.0).with(
            "units",
            Value::object().with("velocity", "hartree bohr (velocity)"),
        );
        let problems = complete(&mut malformed);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("velocity"), "{problems:?}");
    }

    #[test]
    fn selected_determinant_diagnostics_use_the_common_unit_vocabulary() {
        let mut report = Value::object().with(
            "occupation_tracking",
            Value::object()
                .with("initial_energy", -1.0)
                .with("energy_difference", 0.2)
                .with("overlap_weights", vec![vec![0.91], vec![0.93]])
                .with("boundary_gaps", vec![Some(0.82), None])
                .with("density_residual", 2e-13),
        );
        assert!(complete(&mut report).is_empty());
        let labels = report.get("units").unwrap();
        assert_eq!(
            labels.get("energy_difference").and_then(Value::as_str),
            Some("hartree")
        );
        assert_eq!(
            labels.get("density_residual").and_then(Value::as_str),
            Some("dimensionless")
        );
        assert_eq!(
            labels.get("boundary_gaps").and_then(Value::as_str),
            Some("dimensionless")
        );
    }

    #[test]
    fn fixed_population_diagnostics_use_the_common_unit_vocabulary() {
        let mut report = Value::object()
            .with("occupations", vec![vec![0.8, 0.2], vec![0.7, 0.3]])
            .with("fixed_occupation_residual", 3e-13)
            .with("fixed_occupation_gap", 0.2);
        assert!(complete(&mut report).is_empty());
        let labels = report.get("units").unwrap();
        assert_eq!(
            labels
                .get("fixed_occupation_residual")
                .and_then(Value::as_str),
            Some("dimensionless")
        );
        assert_eq!(
            labels.get("fixed_occupation_gap").and_then(Value::as_str),
            Some("hartree")
        );
    }
}
