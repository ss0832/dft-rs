// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

use super::*;

fn diagonal(index_squared: [Complex; 3]) -> Square<3> {
    from_fn(|i| {
        from_fn(|j| {
            if i == j {
                index_squared[i]
            } else {
                Complex::ZERO
            }
        })
    })
}

fn close(actual: Complex, expected: Complex, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn lorentz_cavity_matches_independent_isotropic_closed_form() {
    let alpha = Complex::new(9.1, 0.83);
    let density = 0.0021;
    let x = alpha * (4.0 * PI / 3.0 * density);
    let expected = (Complex::ONE + x * 2.0) / (Complex::ONE - x);
    let tensor = diagonal([alpha; 3]);
    let result = molecular_dielectric(tensor, density, LocalField::Lorentz).unwrap();
    assert!(result.local_field_residual < 1e-15);
    assert!(max_norm(&add(&result.permittivity, &diagonal([expected; 3]), -1.0)) < 1e-15);
    for index in result.normal_indices().unwrap() {
        close(index * index, expected, 1e-15);
        assert!(index.re > 0.0 && index.im > 0.0);
    }
    let dilute = molecular_dielectric(tensor, density, LocalField::Dilute).unwrap();
    close(dilute.permittivity[0][0], Complex::ONE + x * 3.0, 1e-15);
    let vacuum = molecular_dielectric(tensor, 0.0, LocalField::Lorentz).unwrap();
    assert_eq!(vacuum.permittivity, identity());
    println!(
        "Lorentz closed form epsilon={expected:?} indices={:?} residual={:.6e}",
        result.normal_indices().unwrap(),
        result.local_field_residual
    );
}

#[test]
fn isotropic_slab_matches_oblique_fresnel_and_airy_formulas() {
    let boundary = SlabBoundary {
        incident_index: 1.17,
        exit_index: 1.06,
        incidence_angle: 0.51,
        optical_thickness: 2.9,
    };
    let n = [boundary.incident_index, 1.43, boundary.exit_index];
    let p = n[0] * boundary.incidence_angle.sin();
    let q = n.map(|index| (index * index - p * p).sqrt());
    let phase = q[1] * boundary.optical_thickness;
    let z = Complex::new(phase.cos(), phase.sin());
    let result = dielectric_slab(diagonal([Complex::from(n[1].powi(2)); 3]), boundary).unwrap();
    for j in 0..2 {
        let y = if j == 0 {
            q
        } else {
            from_fn(|i| n[i] * n[i] / q[i])
        };
        let r01 = (y[0] - y[1]) / (y[0] + y[1]);
        let r12 = (y[1] - y[2]) / (y[1] + y[2]);
        let denominator = Complex::ONE + z * z * (r01 * r12);
        let r = (Complex::from(r01) + z * z * r12) / denominator;
        let t = z * ((1.0 + r01) * (1.0 + r12)) / denominator;
        let transmittance = y[2] / y[0] * t.abs().powi(2);
        close(result.reflection[j][j], r, 1e-13);
        close(result.reflection[1 - j][j], Complex::ZERO, 1e-13);
        assert!((result.transmittance[j] - transmittance).abs() < 1e-13);
        assert!(result.absorptance[j].abs() < 1e-13);
        assert!(result.flux_balance_residual[j] < 1e-14);
    }
    println!(
        "Oblique isotropic slab R={:?} T={:?} A={:?} flux residual={:?}",
        result.reflectance, result.transmittance, result.absorptance, result.flux_balance_residual
    );
}

#[test]
fn anisotropic_slab_mixes_polarizations_and_eliminates_longitudinal_field() {
    let boundary = SlabBoundary {
        incident_index: 1.0,
        exit_index: 1.19,
        incidence_angle: 0.0,
        optical_thickness: 1.7,
    };
    let n = [1.31_f64, 1.67];
    let mut epsilon = diagonal([n[0].powi(2).into(), n[1].powi(2).into(), 2.1.into()]);
    let axes = dielectric_slab(epsilon, boundary).unwrap();
    let angle = 0.36_f64;
    let (sin, cos) = angle.sin_cos();
    let rotation = [[cos, -sin], [sin, cos]];
    let rotated = |i: usize, j: usize| {
        (0..2)
            .map(|k| rotation[i][k] * rotation[j][k] * n[k].powi(2))
            .sum::<f64>()
    };
    let cross = [0.18, -0.24];
    for i in 0..2 {
        epsilon[i][2] = cross[i].into();
        epsilon[2][i] = cross[i].into();
        for j in 0..2 {
            epsilon[i][j] = (rotated(i, j) + cross[i] * cross[j] / 2.1).into();
        }
    }
    let result = dielectric_slab(epsilon, boundary).unwrap();
    for (measured, reference) in [
        (result.reflection, axes.reflection),
        (result.transmission, axes.transmission),
    ] {
        for i in 0..2 {
            for j in 0..2 {
                // Jones order is y (s), x (p), while epsilon is x, y, z.
                let expected = (0..2).fold(Complex::ZERO, |sum, k| {
                    sum + reference[1 - k][1 - k] * (rotation[1 - i][k] * rotation[1 - j][k])
                });
                close(measured[i][j], expected, 1e-13);
            }
        }
    }
    assert!(result.reflection[1][0].abs() > 1e-3);
    assert!(result.absorptance.iter().all(|a| a.abs() < 1e-13));
    let indices = MolecularDielectric {
        permittivity: epsilon,
        local_field_residual: 0.0,
    }
    .normal_indices()
    .unwrap();
    close(indices[0], n[1].into(), 1e-14);
    close(indices[1], n[0].into(), 1e-14);
}

#[test]
fn absorbing_and_evanescent_limits_remain_bounded() {
    let n = Complex::new(1.32, 0.22);
    let boundary = SlabBoundary {
        incident_index: 1.0,
        exit_index: 1.0,
        incidence_angle: 0.0,
        optical_thickness: 20_000.0,
    };
    let result = dielectric_slab(diagonal([n * n; 3]), boundary).unwrap();
    let fresnel = (Complex::ONE - n) / (Complex::ONE + n);
    for j in 0..2 {
        close(result.reflection[j][j], fresnel, 1e-12);
        assert!(result.transmittance[j] < 1e-30);
        assert!((result.absorptance[j] - (1.0 - fresnel.abs().powi(2))).abs() < 1e-12);
    }
    let zero_thickness = SlabBoundary {
        incident_index: 1.5,
        incidence_angle: 0.85,
        optical_thickness: 0.0,
        ..boundary
    };
    let reflection = dielectric_slab(identity(), zero_thickness).unwrap();
    for j in 0..2 {
        assert!((reflection.reflectance[j] - 1.0).abs() < 1e-14);
        assert_eq!(reflection.transmittance[j], 0.0);
        assert!(reflection.absorptance[j].abs() < 1e-14);
    }
}

#[test]
fn gyrotropic_half_space_kerr_matches_independent_circular_fresnel_channels() {
    let mean = Complex::new(2.25, 0.19);
    let gyrotropy = 0.082;
    let n_plus = Complex::from(gyrotropy) + mean;
    let n_minus = mean - Complex::from(gyrotropy);
    // Independent closed-form square roots on the passive branch.
    let fresnel = |epsilon: Complex| {
        let real = ((epsilon.abs() + epsilon.re) / 2.0).sqrt();
        let n = Complex::new(real, epsilon.im / (2.0 * real));
        (Complex::ONE - n) / (Complex::ONE + n)
    };
    let r_plus = fresnel(n_plus);
    let r_minus = fresnel(n_minus);
    let co = (r_minus + r_plus) * 0.5;
    let cross = Complex::new(0.0, 0.5) * (r_minus - r_plus);
    let ratio = cross / co;
    let expected_rotation = 0.5 * (2.0 * ratio.re).atan2(1.0 - ratio.abs().powi(2));
    let expected_ellipticity = 0.5 * (2.0 * ratio.im / (1.0 + ratio.abs().powi(2))).asin();
    let boundary = SlabBoundary {
        incident_index: 1.0,
        exit_index: 1.0,
        incidence_angle: 0.0,
        optical_thickness: 800.0,
    };
    let scattering = |sign| {
        let mut epsilon = diagonal([mean; 3]);
        epsilon[0][1] = Complex::new(0.0, sign * gyrotropy);
        epsilon[1][0] = -epsilon[0][1];
        dielectric_slab(epsilon, boundary).unwrap()
    };
    let positive = scattering(1.0);
    let negative = scattering(-1.0);
    close(positive.reflection[1][1], co, 1e-12);
    close(positive.reflection[0][1], cross, 1e-12);
    let angles = magnetic_kerr(positive.reflection, negative.reflection);
    assert!((angles.rotation[1].unwrap() - expected_rotation).abs() < 1e-12);
    assert!((angles.ellipticity[1].unwrap() - expected_ellipticity).abs() < 1e-12);
    assert!((angles.rotation[0].unwrap() + expected_rotation).abs() < 1e-12);
    assert!((angles.ellipticity[0].unwrap() + expected_ellipticity).abs() < 1e-12);
    assert_eq!(
        magnetic_kerr([[Complex::ZERO; 2]; 2], [[Complex::ZERO; 2]; 2]).rotation,
        [None; 2]
    );
    println!(
        "Circular Fresnel Kerr rotation={:?} ellipticity={:?} flux residual={:?}",
        angles.rotation, angles.ellipticity, positive.flux_balance_residual
    );
}
