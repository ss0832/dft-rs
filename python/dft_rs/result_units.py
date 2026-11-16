# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Unit labels of the results of the dedicated functions.

The functions that go through the common driver return reports whose
``units`` entry is a dictionary of unit labels. The dedicated functions
(``calculate``, ``nmr``, ``tddft``, ``periodic_scf``, ...) return the
dictionaries of their own bindings, some of them with a one-line ``units``
summary. :func:`attach_unit_labels` gives every real-valued entry of such a
dictionary a label of the same form and collects them in
``result["unit_labels"]``, keyed by the dotted path of the entry (list
indices omitted).

A label begins with a unit expression (``hartree/bohr^2``, ``e*bohr``,
``1/bohr``, ``cm^-1``); the label of an object or a list covers everything
below it. A power may name an integer entry of the same object, with an
offset: ``hartree/bohr^order``, ``e^(order+1)*bohr^(order+1)/hartree^order``.
Quantities whose atomic unit is not a product of powers of energy, length
and charge are labelled ``atomic units``. An entry no table knows is labelled
``unspecified``.
"""

UNSPECIFIED = "unspecified"
ATOMIC = "atomic units"
FIELD = "hartree/e/bohr"
POLARIZABILITY = "e^2*bohr^2/hartree"


def _tensor(parent, root):
    """Derivative tensors: the order is an entry of the same object."""
    if "electric_order" in parent:
        return "e^electric_order*bohr^electric_order/hartree^(electric_order-1)"
    if "translation_residual" in parent or "permutation_defect" in parent:
        # Force constants: per Bohr along Cartesian coordinates, or along
        # supplied displacement patterns that carry the length themselves.
        return "hartree" if parent.get("patterns") else "hartree/bohr^order"
    if "order" in parent:
        return "e^(order+1)*bohr^(order+1)/hartree^order"
    return UNSPECIFIED


def _dynamic_tensor(parent, root):
    order = len(parent.get("input_frequencies") or [])
    return f"e^{order + 1}*bohr^{order + 1}/hartree^{order}" if order else UNSPECIFIED


def _direction(parent, root):
    nuclear = "energy_coefficients" in parent or "field_direction" in parent
    return "bohr" if nuclear else FIELD


def _directions(parent, root):
    return "bohr" if "energy_derivative" in parent else FIELD


def _patterns(parent, root):
    return ATOMIC if "harmonic_frequencies_cm" in parent else "bohr"


def _alpha(parent, root):
    return POLARIZABILITY if "isotropic" in parent else "1/bohr"


def _measure(parent, root):
    return f"bohr^{int(root.get('dimension', 3))}"


def _measure_derivative(parent, root):
    return f"bohr^{int(root.get('dimension', 3)) - 1}"


def _labels(label, names):
    return {name: label for name in names.split()}


# Labels by entry name. A callable takes the object that holds the entry and
# the whole result.
LABELS = {}
LABELS.update(_labels(
    "hartree",
    "energy e_nuc e_elec e_xc e_nlc e_dispersion e_pcm e_smd_cds zero_point_energy "
    "vibrational_thermal_energy translational_energy rotational_energy "
    "internal_energy_correction enthalpy_correction gibbs_correction internal_energy "
    "enthalpy gibbs_free_energy mermin_free_energy helmholtz_free_energy "
    "reference_mermin_free_energy delta_mermin_free_energy initial_energy energy_before energy_after "
    "ground_state_energy e_monomer_a e_monomer_b e_elst e_exch e_ind_a e_ind_b "
    "e_exch_ind e_disp_unc e_disp_sos_uncoupled e_disp_coupled e_exch_disp "
    "e_int_supermolecular delta interaction_total interaction_total_uncoupled_calibration "
    "nuclear_repulsion charged_cell_correction nuclear_energy background_shift free_energy "
    "entropy_term chemical_potential optimization_objective mo_energy orbital_energies "
    "homo lumo gap energy_spread energies degeneracy_tolerance zero_tolerance resolution "
    "eigenvalues lowest eigenvalue energy_curvature input_frequencies output_frequency "
    "omega damping frequency frequency_real frequency_imag temperature_hartree "
    "delta_temperature_hartree delta_chemical_potentials delta_internal_energy "
    "delta_free_energy source_energy electronic_energy semilocal_energy kinetic_energy "
    "electron_nuclear_energy ecp_energy external_field_energy hartree_energy "
    "full_exchange_energy long_range_exchange_energy final_fock_change orbital_residual "
    "semilocal_component_energies component_energies reference_energy "
    "reference_free_energy total_coulomb_energy source_energy_components "
    "electron_nuclear_components hartree_energy_components value components "
    "harmonic_zero_point_energy energy_coefficients energy_components energy_derivative "
    "reference_fock fock h_core kinetic nuclear core_hamiltonian hartree kinetic_real "
    "kinetic_imag potential_real potential_imag scalar_potential_real "
    "scalar_potential_imag stationarity_defect stationarity_defects",
))
LABELS.update(_labels(
    "hartree/K",
    "entropy entropy_translational entropy_rotational entropy_vibrational entropy_electronic",
))
LABELS.update(_labels(
    "bohr",
    "positions optimized_positions origin nuclear_direction nuclear_directions "
    "image_cutoff basis_cutoff grid_cutoff real_cutoff countercharge_center "
    "countercharge_width points",
))
LABELS.update(_labels(
    "1/bohr",
    "k_points target_k_points second_target_k_points reciprocal_vectors reciprocal_waves "
    "q wave reciprocal_cutoff overlap_derivatives_real overlap_derivatives_imag",
))
LABELS.update(_labels(
    "hartree/bohr",
    "gradient gradients grad_max forces electronic_forces nuclear_forces source_gradients "
    "energy_gradients center_gradients compensation_force derivative_matrices "
    "kinetic_derivatives_real kinetic_derivatives_imag",
))
LABELS.update(_labels("e*bohr", "dipole dipole_coefficients dipole_derivative transition_dipole"))
LABELS.update(_labels(
    "e", "mulliken atomic_charges net_charge compensating_charge supplied_charge",
))
LABELS.update(_labels("hartree/e", "site_potentials compensation_potential"))
LABELS.update(_labels(
    POLARIZABILITY,
    "alpha_real alpha_imag isotropic isotropic_real isotropic_imag reciprocity_defect",
))
LABELS.update(_labels(FIELD, "reference_field field_direction electric_directions"))
LABELS.update(_labels(
    "cm^-1", "frequencies harmonic_frequencies_cm frequency_cm harmonic_frequency_cm",
))
LABELS.update(_labels("km/mol", "ir_intensities ir_intensity double_harmonic_ir_intensity"))
LABELS.update(_labels(
    "angstrom^4/amu", "raman_activities raman_activity double_harmonic_raman_activity",
))
LABELS.update(_labels(
    "electrons",
    "occupations atomic_populations spin_populations grid_electron_count electron_count "
    "expected_electrons reference_electrons delta_numbers particle_number_error "
    "particle_number_defect charge_residual grid_number_defect number_density_real "
    "number_density_imag charge_density_real charge_density_imag",
))
LABELS.update(_labels(
    "dimensionless",
    "nao_coefficients density_nao wiberg_bond_indices mo_coeff density overlap eigenvector "
    "amplitude amplitudes relative_residual residuals eigen_residual orthonormality_defect "
    "projector_defect projector_defects curvature_factor dominant_weight polyad_vectors "
    "exterior_mixing_norm minimum_overlap_eigenvalue overlap_real overlap_imag "
    "inverse_epsilon_real inverse_epsilon_imag transfer_fractional "
    "reduced_transfer_fractional density_coefficients overlap_coefficients "
    "particle_number_coefficients particle_number_derivatives oscillator_strength "
    "metric_defects",
))
LABELS.update(_labels(
    ATOMIC,
    "rotatory_strength_length rotatory_strength_velocity cubic quartic dipole_derivatives "
    "polarizability_derivatives rotation maximum_force double_harmonic_dipole "
    "mechanical_dipole electrical_dipole mixed_dipole transition_polarizability "
    "polyad_energies effective_hamiltonian minimum_external_denominator density_hat "
    "overlap_hat explicit_fock_hat reference_density electric_field vector_potential_real "
    "vector_potential_imag electric_field_real electric_field_imag "
    "paramagnetic_number_current_real paramagnetic_number_current_imag "
    "diamagnetic_number_current_real diamagnetic_number_current_imag number_current_real "
    "number_current_imag charge_current_real charge_current_imag fourier_density_real "
    "fourier_density_imag current_vertex_real current_vertex_imag reduced_transfer "
    "orthonormal_fock_coefficients operator_tolerance grid_tolerance",
))
LABELS.update({
    "s_squared": "hbar^2",
    "hessian": "hartree/bohr^2",
    "nuclear_stress": "hartree/bohr^3",
    "volume": "bohr^3",
    "temperature": "K",
    "electronic_temperature": "K",
    "delta_temperature_kelvin": "K",
    "pressure": "Pa",
    "delta_entropy": "k_B",
    "theta": "radians",
    "energy_ev": "eV",
    "shieldings": "ppm",
    "transition_angular_momentum": "hbar",
    "transition_momentum": "hbar/bohr",
    "magnetic_field": "atomic units of magnetic field",
    "magnetic_field_real": "atomic units of magnetic field",
    "magnetic_field_imag": "atomic units of magnetic field",
    "screened_chi_real": "1/bohr^3/hartree",
    "screened_chi_imag": "1/bohr^3/hartree",
    "exact_exchange_coefficients": "[dimensionless, dimensionless, 1/bohr] full fraction, long-range fraction, range-separation parameter",
    "beta_vector": "e^3*bohr^3/hartree^2",
    "beta_vector_norm": "e^3*bohr^3/hartree^2",
    "gamma_isotropic": "e^4*bohr^4/hartree^3",
    "tensor": _tensor,
    "frobenius_norm": _tensor,
    "tensor_real": _dynamic_tensor,
    "tensor_imag": _dynamic_tensor,
    "direction": _direction,
    "directions": _directions,
    "patterns": _patterns,
    "alpha": _alpha,
    "measure": _measure,
    "weights": _measure,
    "raw_weights": _measure,
    "moving_weight_derivatives": _measure_derivative,
})

LABELS.update(_labels(
    "hartree",
    "fock_real fock_imag h_core_real h_core_imag hartree_real hartree_imag "
    "hartree_components_real hartree_components_imag nuclear_real nuclear_imag "
    "nuclear_components_real nuclear_components_imag energy_weighted_density_real "
    "energy_weighted_density_imag",
))
LABELS.update(_labels(
    "hartree/bohr",
    "hartree_derivatives_real hartree_derivatives_imag nuclear_derivatives_real "
    "nuclear_derivatives_imag",
))
LABELS.update(_labels("dimensionless", "density_real density_imag"))
LABELS.update(_labels("electrons", "occupation_consistency density_electron_count"))
LABELS.update(_labels(
    ATOMIC,
    "real_space_density_imaginary_max contact_vertex_real contact_vertex_imag "
    "density_vertex_real density_vertex_imag first_order_real first_order_imag "
    "second_order_real second_order_imag derivatives",
))

# Diagnostics without a stated unit keep their atomic value on every surface.
DIAGNOSTIC_SUFFIXES = (
    "_defect", "_defects", "_residual", "_residuals", "_error", "_errors", "_roundoff",
)


def _holds_reals(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_holds_reals(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_reals(item) for item in value)
    return False


def _holds_objects(value):
    if isinstance(value, dict):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_objects(item) for item in value)
    return False


def _is_report(value):
    """A nested result with its own one-line ``units`` summary."""
    return isinstance(value, dict) and isinstance(value.get("units"), str)


def unit_labels(result):
    """Labels of the real-valued entries of ``result`` by dotted path."""
    labels = {}

    def visit(key, path, value, parent):
        if not _holds_reals(value):
            return
        if _is_report(value):
            # Labelled on its own by attach_unit_labels.
            return
        label = LABELS.get(key)
        if callable(label):
            label = label(parent, result)
        if label is None and key.endswith(DIAGNOSTIC_SUFFIXES):
            label = ATOMIC
        if label is not None:
            labels[path] = label
        elif isinstance(value, dict):
            for child, item in value.items():
                visit(child, f"{path}.{child}", item, value)
        elif _holds_objects(value):
            for item in value:
                visit(key, path, item, parent)
        else:
            labels[path] = UNSPECIFIED

    for key, value in result.items():
        if key not in ("units", "unit_labels"):
            visit(key, key, value, result)
    return labels


def attach_unit_labels(result):
    """Add ``unit_labels`` to a result of a dedicated function, and to every
    nested result that carries its own ``units`` summary. A result without
    real-valued entries is left as it is."""

    def nested(value):
        if isinstance(value, dict):
            if _is_report(value):
                attach_unit_labels(value)
            else:
                for item in value.values():
                    nested(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                nested(item)

    for key, value in list(result.items()):
        if key not in ("units", "unit_labels"):
            nested(value)
    labels = unit_labels(result)
    if labels:
        result["unit_labels"] = labels
    return result
