// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::{Controls, System, Value};
use crate::constants::FINE_STRUCTURE;
use crate::response::complex::Complex;
use crate::response::medium_optics::{
    dielectric_slab, magnetic_kerr, molecular_dielectric, LocalField, SlabBoundary,
};
use crate::response::Frequency;
use crate::{DftError, Result};
use std::array::from_fn;

const LOCAL_CONTROLS: &[&str] = &[
    "medium_number_density",
    "medium_local_field",
    "medium_orientation",
    "medium_magnetic_field",
    "slab_thickness",
    "incident_refractive_index",
    "exit_refractive_index",
    "incidence_angle",
];

fn orientation(controls: &Controls) -> Result<Option<[[f64; 3]; 3]>> {
    let Some(rows) = controls.opt_rows("medium_orientation")? else {
        return Ok(None);
    };
    let error = || {
        DftError::InvalidInput(
            "medium_orientation must be a proper orthogonal 3 by 3 matrix from molecular to slab axes"
                .into(),
        )
    };
    if rows.len() != 3 || rows.iter().any(|row| row.len() != 3) {
        return Err(error());
    }
    let r: [[f64; 3]; 3] = from_fn(|i| from_fn(|j| rows[i][j]));
    for i in 0..3 {
        for j in 0..3 {
            let dot = (0..3).map(|k| r[i][k] * r[j][k]).sum::<f64>();
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > 1e-12 || !dot.is_finite() {
                return Err(error());
            }
        }
    }
    let determinant = (0..3)
        .map(|i| {
            r[0][i]
                * (r[1][(i + 1) % 3] * r[2][(i + 2) % 3] - r[2][(i + 1) % 3] * r[1][(i + 2) % 3])
        })
        .sum::<f64>();
    if determinant <= 0.0 {
        return Err(error());
    }
    Ok(Some(r))
}

fn orient(alpha: [[Complex; 3]; 3], rotation: Option<[[f64; 3]; 3]>) -> [[Complex; 3]; 3] {
    if let Some(r) = rotation {
        from_fn(|i| {
            from_fn(|j| {
                let mut sum = Complex::ZERO;
                for (a, row) in alpha.iter().enumerate() {
                    for (b, value) in row.iter().enumerate() {
                        sum += *value * (r[i][a] * r[j][b]);
                    }
                }
                sum
            })
        })
    } else {
        let trace = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
        from_fn(|i| from_fn(|j| if i == j { trace } else { Complex::ZERO }))
    }
}

fn matrix<const N: usize>(report: &mut Value, name: &str, value: [[Complex; N]; N]) {
    report.set(&format!("{name}_real"), value.map(|row| row.map(|z| z.re)));
    report.set(&format!("{name}_imag"), value.map(|row| row.map(|z| z.im)));
}

fn magnetic_shift(
    derivative: [[[Complex; 3]; 3]; 3],
    bias: [f64; 3],
    rotation: Option<[[f64; 3]; 3]>,
) -> [[Complex; 3]; 3] {
    if let Some(r) = rotation {
        let molecular: [f64; 3] = from_fn(|i| (0..3).map(|j| r[j][i] * bias[j]).sum());
        let shifted = derivative.map(|row| {
            row.map(|entry| (0..3).fold(Complex::ZERO, |sum, k| sum + entry[k] * molecular[k]))
        });
        orient(shifted, rotation)
    } else {
        // Haar average of three entries of a proper rotation is
        // epsilon_abc epsilon_ijk / 6.
        let invariant = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            .into_iter()
            .fold(Complex::ZERO, |sum, (i, j, k)| {
                sum + derivative[i][j][k] - derivative[j][i][k]
            })
            / 6.0;
        let [x, y, z] = bias.map(|b| invariant * b);
        let zero = Complex::ZERO;
        [[zero, z, -y], [-z, zero, x], [y, -x, zero]]
    }
}

pub(super) fn run(system: &System, controls: &Controls) -> Result<Value> {
    let Frequency::Retarded { omega, .. } = super::polarizability::frequency(controls)? else {
        return Err(DftError::InvalidInput(
            "molecular medium optics needs a positive real photon energy".into(),
        ));
    };
    if omega <= 0.0 {
        return Err(DftError::InvalidInput(
            "molecular medium optics needs a positive real photon energy".into(),
        ));
    }
    let number_density = controls.opt_f64("medium_number_density")?.ok_or_else(|| {
        DftError::InvalidInput("molecular medium optics needs medium_number_density".into())
    })?;
    let thickness = controls.opt_f64("slab_thickness")?.ok_or_else(|| {
        DftError::InvalidInput("molecular medium optics needs slab_thickness".into())
    })?;
    let field_name = controls.str("medium_local_field", "lorentz")?;
    let field = match field_name.as_str() {
        "dilute" => LocalField::Dilute,
        "lorentz" => LocalField::Lorentz,
        _ => {
            return Err(DftError::InvalidInput(
                "medium_local_field must be dilute or lorentz".into(),
            ))
        }
    };
    let rotation = orientation(controls)?;
    let magnetic = controls
        .opt_vec3("medium_magnetic_field")?
        .map(|b| b.to_array());
    let k0 = FINE_STRUCTURE * omega;
    let boundary = SlabBoundary {
        incident_index: controls.f64("incident_refractive_index", 1.0)?,
        exit_index: controls.f64("exit_refractive_index", 1.0)?,
        incidence_angle: controls.f64("incidence_angle", 0.0)?,
        optical_thickness: k0 * thickness,
    };
    if number_density < 0.0 || thickness < 0.0 {
        return Err(DftError::InvalidInput(
            "medium density and thickness must be nonnegative".into(),
        ));
    }
    boundary.validate()?;
    let mut delegated = controls.to_value();
    if let Value::Object(entries) = &mut delegated {
        entries.retain(|(key, _)| !LOCAL_CONTROLS.contains(&key.as_str()));
    }
    let mut report = if magnetic.is_some() {
        if controls
            .peek_str("magneto_optical_source")?
            .is_some_and(|source| source != "field")
        {
            return Err(DftError::InvalidInput(
                "medium_magnetic_field selects the orbital Faraday field derivative".into(),
            ));
        }
        delegated.set("magneto_optical_source", "field");
        super::molecular::run(
            "magneto-optical",
            system,
            &Controls::from_value(&delegated)?,
        )?
    } else {
        super::polarizability::run(system, &Controls::from_value(&delegated)?)?
    };
    let read = |key, dimensions: &[usize]| {
        let (shape, values) = report
            .get(key)
            .and_then(Value::numeric_array)
            .ok_or_else(|| {
                DftError::LinearAlgebra(format!("molecular response did not return {key}"))
            })?;
        if shape != dimensions {
            return Err(DftError::LinearAlgebra(format!(
                "molecular response returned non-Cartesian {key}"
            )));
        }
        Ok(values)
    };
    let keys = if magnetic.is_some() {
        ("polarizability_real", "polarizability_imag")
    } else {
        ("alpha_real", "alpha_imag")
    };
    let real = read(keys.0, &[3, 3])?;
    let imag = read(keys.1, &[3, 3])?;
    let alpha = from_fn(|i| from_fn(|j| Complex::new(real[3 * i + j], imag[3 * i + j])));
    let base_alpha = orient(alpha, rotation);
    let (laboratory_alpha, reversed) = if let Some(bias) = magnetic {
        let real = read("faraday_tensor_real", &[3, 3, 3])?;
        let imag = read("faraday_tensor_imag", &[3, 3, 3])?;
        let derivative = from_fn(|i| {
            from_fn(|j| {
                from_fn(|k| {
                    let index = 9 * i + 3 * j + k;
                    Complex::new(real[index], imag[index])
                })
            })
        });
        let shift = magnetic_shift(derivative, bias, rotation);
        let biased = |sign| from_fn(|i| from_fn(|j| base_alpha[i][j] + shift[i][j] * sign));
        let negative_medium = molecular_dielectric(biased(-1.0), number_density, field)?;
        let reversed = dielectric_slab(negative_medium.permittivity, boundary)?;
        (
            biased(1.0),
            Some((reversed, negative_medium.local_field_residual)),
        )
    } else {
        (base_alpha, None)
    };
    let medium = molecular_dielectric(laboratory_alpha, number_density, field)?;
    let indices = medium.normal_indices()?;
    let slab = dielectric_slab(medium.permittivity, boundary)?;
    let molecular_model = report
        .get("model")
        .cloned()
        .expect("molecular response model");
    report.set("molecular_response_model", molecular_model);
    report.set("model", if magnetic.is_some() {
        "Homogeneous molecular dielectric slab with an electric dipole susceptibility linear in the static magnetic bias"
    } else {
        "Homogeneous molecular electric-dipole dielectric slab with lossless isotropic exterior media and unit relative permeability"
    });
    report.set(
        "medium_orientation_model",
        if rotation.is_some() {
            "aligned"
        } else {
            "isotropic"
        },
    );
    if let Some(rotation) = rotation {
        report.set("medium_orientation", rotation);
    }
    report.set("medium_local_field", field_name);
    report.set("medium_number_density", number_density);
    report.set("slab_thickness", thickness);
    report.set("incidence_angle", boundary.incidence_angle);
    report.set("incident_refractive_index", boundary.incident_index);
    report.set("exit_refractive_index", boundary.exit_index);
    report.set("normal_refractive_index_real", indices.map(|z| z.re));
    report.set("normal_refractive_index_imag", indices.map(|z| z.im));
    report.set(
        "normal_absorption_coefficient",
        indices.map(|z| 2.0 * k0 * z.im),
    );
    report.set("reflectance", slab.reflectance);
    report.set("transmittance", slab.transmittance);
    report.set("absorptance", slab.absorptance);
    report.set("flux_balance_residual", slab.flux_balance_residual);
    report.set("slab_boundary_residual", slab.boundary_residual);
    report.set("local_field_residual", medium.local_field_residual);
    report.set("slab_axis_convention", "Slab normal +z with incidence in the xz plane. Jones rows are outgoing (s,p) and columns are incident (s,p). Positive s and p amplitudes have positive tangential E_y and E_x in both directions. Fields use exp(-i omega t).");
    matrix(&mut report, "medium_polarizability", laboratory_alpha);
    matrix(&mut report, "relative_permittivity", medium.permittivity);
    matrix(&mut report, "reflection_jones", slab.reflection);
    matrix(&mut report, "transmission_jones", slab.transmission);
    let mut units = report
        .get("units")
        .cloned()
        .expect("molecular response units");
    for (key, label) in [
        ("medium_polarizability_real", "e^2*bohr^2/hartree"),
        ("medium_polarizability_imag", "e^2*bohr^2/hartree"),
        ("medium_number_density", "1/bohr^3"),
        ("slab_thickness", "bohr"),
        ("incidence_angle", "dimensionless, radians"),
        ("normal_absorption_coefficient", "1/bohr"),
    ] {
        units.set(key, label);
    }
    for key in [
        "medium_orientation",
        "incident_refractive_index",
        "exit_refractive_index",
        "normal_refractive_index_real",
        "normal_refractive_index_imag",
        "relative_permittivity_real",
        "relative_permittivity_imag",
        "reflection_jones_real",
        "reflection_jones_imag",
        "transmission_jones_real",
        "transmission_jones_imag",
        "reflectance",
        "transmittance",
        "absorptance",
        "flux_balance_residual",
        "slab_boundary_residual",
        "local_field_residual",
    ] {
        units.set(key, "dimensionless");
    }
    if let (Some(bias), Some((reverse, reverse_local_residual))) = (magnetic, reversed) {
        let kerr = magnetic_kerr(slab.reflection, reverse.reflection);
        report.set("medium_magnetic_field", bias);
        report.set("magnetic_kerr_rotation", kerr.rotation);
        report.set("magnetic_kerr_ellipticity", kerr.ellipticity);
        report.set("reversed_field_reflectance", reverse.reflectance);
        report.set(
            "reversed_field_slab_boundary_residual",
            reverse.boundary_residual,
        );
        report.set(
            "reversed_field_flux_balance_residual",
            reverse.flux_balance_residual,
        );
        report.set(
            "reversed_field_local_field_residual",
            reverse_local_residual,
        );
        matrix(
            &mut report,
            "reversed_field_reflection_jones",
            reverse.reflection,
        );
        units.set("medium_magnetic_field", "atomic units of magnetic field");
        units.set("magnetic_kerr_rotation", "dimensionless, radians");
        units.set("magnetic_kerr_ellipticity", "dimensionless, radians");
        for key in [
            "reversed_field_reflectance",
            "reversed_field_slab_boundary_residual",
            "reversed_field_flux_balance_residual",
            "reversed_field_local_field_residual",
            "reversed_field_reflection_jones_real",
            "reversed_field_reflection_jones_imag",
        ] {
            units.set(key, "dimensionless");
        }
    }
    report.set("units", units);
    Ok(report)
}
