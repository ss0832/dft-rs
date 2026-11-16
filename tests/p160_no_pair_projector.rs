// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Independent exact Dirac projector on a moving non-orthogonal basis.
#[path = "support/convergence.rs"]
mod convergence;
use dft_rs::dual::{Scalar, Taylor};
use dft_rs::linalg::Matrix;
use dft_rs::periodic::BlochMatrix;
use dft_rs::properties::no_pair::no_pair_projector_taylor;

type Jet = Taylor<5>;

fn data(t: Jet, c: f64) -> (Vec<BlochMatrix>, Vec<BlochMatrix>, Vec<BlochMatrix>) {
    let j = Jet::from_f64;
    let p = [
        j(0.4) + t.scale(0.2),
        j(-0.3) + t.scale(0.1),
        j(0.2) - t.scale(0.15),
    ];
    let energy = (p.iter().map(|x| *x * *x).fold(j(c * c), |a, b| a + b) * j(c * c)).sqrt();
    let p2 = p.iter().map(|x| *x * *x).fold(j(0.0), |a, b| a + b);
    let positive = p2.scale(c * c) / (energy + j(c * c));
    let mut hr = [[j(0.0); 4]; 4];
    let mut hi = hr;
    let mut pr = hr;
    let mut pi = hr;
    for i in 0..2 {
        hr[i + 2][i + 2] = j(-2.0 * c * c);
        pr[i][i] = (energy + j(c * c)) / energy.scale(2.0);
        pr[i + 2][i + 2] = positive / energy.scale(2.0);
    }
    for (a, b, x, y) in [
        (0, 2, p[2], j(0.0)),
        (0, 3, p[0], -p[1]),
        (1, 2, p[0], p[1]),
        (1, 3, -p[2], j(0.0)),
    ] {
        hr[a][b] = x.scale(c);
        hr[b][a] = x.scale(c);
        hi[a][b] = y.scale(c);
        hi[b][a] = -y.scale(c);
        pr[a][b] = x.scale(c) / energy.scale(2.0);
        pr[b][a] = pr[a][b];
        pi[a][b] = y.scale(c) / energy.scale(2.0);
        pi[b][a] = -pi[a][b];
    }
    let scale = [
        j(1.1) + t.scale(0.04),
        j(0.9) - t.scale(0.03),
        j(1.2) + t.scale(0.02),
        j(0.95) - t.scale(0.01),
    ];
    let mut h = vec![BlochMatrix::zeros(4, 4); 5];
    let mut s = h.clone();
    let mut oracle = h.clone();
    for a in 0..4 {
        for b in 0..4 {
            let real = hr[a][b] * scale[a] * scale[b];
            let imaginary = hi[a][b] * scale[a] * scale[b];
            let r = pr[a][b] / (scale[a] * scale[b]);
            let i = pi[a][b] / (scale[a] * scale[b]);
            for n in 0..5 {
                h[n].re[(a, b)] = real.c[n];
                h[n].im[(a, b)] = imaginary.c[n];
                oracle[n].re[(a, b)] = r.c[n];
                oracle[n].im[(a, b)] = i.c[n];
            }
        }
    }
    for a in 0..4 {
        let value = scale[a] * scale[a];
        for n in 0..5 {
            s[n].re[(a, a)] = value.c[n];
        }
    }
    (h, s, oracle)
}

fn difference(a: &BlochMatrix, b: &BlochMatrix) -> f64 {
    a.re.as_slice()
        .iter()
        .zip(b.re.as_slice())
        .chain(a.im.as_slice().iter().zip(b.im.as_slice()))
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn four_component_projector_metric_and_negative_sector_response_have_four_step_fits() {
    let (h, s, oracle) = data(Jet::from_coefficients(&[0.0, 1.0]), 5.0);
    let response = no_pair_projector_taylor(&h, &s, 4, -25.0, 1e-12).unwrap();
    assert_eq!(response.electronic_dimension, 2);
    assert!(response.spectral_gap > 50.0);
    for n in 0..5 {
        assert!(difference(&response.projector[n], &oracle[n]) < 1e-11);
    }
    println!("Dirac projector raw projector={:?} metric={:?} commutator={:?} relative_commutator={:?} number={:?} embedding={:?}",
        response.projector_defects,response.metric_defects,response.commutator_defects,response.relative_commutator_defects,
        response.number_defects,response.complex_embedding_defects);
    for array in [
        &response.projector_defects,
        &response.metric_defects,
        &response.number_defects,
        &response.relative_commutator_defects,
        &response.complex_embedding_defects,
    ] {
        assert!(array.iter().all(|x| *x < 1e-10));
    }
    let steps = [0.08, 0.04, 0.02, 0.01];
    for n in 1..=4 {
        let mut target = response.projector[n].clone();
        for x in target
            .re
            .as_mut_slice()
            .iter_mut()
            .chain(target.im.as_mut_slice())
        {
            *x *= n as f64;
        }
        let estimates: Vec<_> = steps
            .iter()
            .map(|&step| {
                let plus = data(Jet::from_coefficients(&[step, 1.0]), 5.0).2;
                let minus = data(Jet::from_coefficients(&[-step, 1.0]), 5.0).2;
                let mut result = plus[n - 1].clone();
                for (x, y) in result
                    .re
                    .as_mut_slice()
                    .iter_mut()
                    .zip(minus[n - 1].re.as_slice())
                {
                    *x = (*x - y) / (2.0 * step);
                }
                for (x, y) in result
                    .im
                    .as_mut_slice()
                    .iter_mut()
                    .zip(minus[n - 1].im.as_slice())
                {
                    *x = (*x - y) / (2.0 * step);
                }
                result
            })
            .collect();
        let errors: Vec<_> = estimates.iter().map(|a| difference(a, &target)).collect();
        let fit = convergence::error_power_fit(&steps, &errors);
        let mut extrapolated = estimates[3].clone();
        for (x, y) in extrapolated
            .re
            .as_mut_slice()
            .iter_mut()
            .zip(estimates[2].re.as_slice())
        {
            *x = (4.0 * *x - y) / 3.0;
        }
        for (x, y) in extrapolated
            .im
            .as_mut_slice()
            .iter_mut()
            .zip(estimates[2].im.as_slice())
        {
            *x = (4.0 * *x - y) / 3.0;
        }
        let rich = difference(&extrapolated, &target);
        println!("Dirac projector order={n} steps={steps:?} errors={errors:?} p={:.9} R2={:.12} Richardson={rich:.12e}",fit.exponent,fit.r_squared);
        assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
        assert!(rich < 1e-6);
    }
    // The projected covariant operator agrees with the independently known
    // positive Dirac energy, including its moving metric response.
    let (energies, _) = dft_rs::linalg::Orthogonalizer::canonical(&s[0].real_embedding(), 1e-12)
        .unwrap()
        .solve_roothaan(&h[0].real_embedding())
        .unwrap();
    let positive = *energies.last().unwrap();
    let expected = response.projector[0].re.clone();
    let expected = s[0].re.matmul_seq(&expected).matmul_seq(&s[0].re);
    for (a, b) in response.projected_hamiltonian[0]
        .re
        .as_slice()
        .iter()
        .zip(expected.as_slice())
    {
        assert!((a - positive * b).abs() < 1e-10);
    }
    // Normal nonrelativistic limit: P tends to the large-component projector.
    let mut errors = Vec::new();
    let inverse_c = [0.04, 0.02, 0.01, 0.005];
    for step in inverse_c {
        let c = 1.0 / step;
        let (h, s, _) = data(Jet::from_f64(0.0), c);
        let result = no_pair_projector_taylor(&h[..1], &s[..1], 0, -c * c, 1e-12).unwrap();
        let ll = result.projector[0].re[(0, 0)] * 1.1_f64.powi(2);
        errors.push((ll - 1.0).abs());
    }
    let fit = convergence::error_power_fit(&inverse_c, &errors);
    println!(
        "Dirac nonrelativistic limit steps={inverse_c:?} errors={errors:?} p={:.9} R2={:.12}",
        fit.exponent, fit.r_squared
    );
    assert!((fit.exponent - 2.0).abs() < 0.05 && fit.r_squared > 0.999);
    assert!(no_pair_projector_taylor(
        &[BlochMatrix {
            re: Matrix::identity(4),
            im: Matrix::zeros(4, 4)
        }],
        &s[..1],
        0,
        1e4,
        1e-12
    )
    .is_err());
}
