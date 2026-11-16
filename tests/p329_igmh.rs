// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Independent scalar-density stencils for all spatial IGMH derivatives.

#[path = "support/c1_geometry.rs"]
mod c1_geometry;
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::grid::{eval::eval_aos, GridOptions, MolecularGrid};
use dft_rs::properties::igmh::{analyze, GradientConvention};
use dft_rs::scf::sad::sad_density;
use dft_rs::{resolve, run_scf, Atom, FunctionalSpec, Matrix, Molecule, ScfOptions, Vec3};
use stencils::Stencil;

const CENTRES: [[f64; 3]; 6] = [
    [0.01, -0.07, -0.1],
    [0.16, 0.08, 1.70],
    [-1.63, 0.53, -0.49],
    [0.27, 0.18, 5.08],
    [1.63, -0.42, 6.10],
    [-1.04, -0.82, 5.78],
];
const Z: [u8; 6] = [8, 1, 1, 8, 1, 1];
const PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

fn vector([x, y, z]: [f64; 3]) -> Vec3 {
    Vec3::new(x, y, z)
}

#[test]
fn single_hydrogen_has_the_exact_gaussian_fields_and_zero_delta_g() {
    let centre = Vec3::new(-0.19, 0.27, -0.31);
    let point = Vec3::new(0.16, 0.59, 0.11);
    let molecule = Molecule::new(vec![Atom {
        z: 1,
        position: centre,
    }]);
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let density = sad_density(&molecule, &basis).unwrap();
    let shell = &basis.shells[0];
    let r = point - centre;
    let r2 = r.norm2();
    let mut expected = [0.0; 10];
    // Expand the squared contraction into primitive Gaussian pairs first.
    // The closed-form density derivatives then depend on the exponent sum.
    for (&a, &ca) in shell.exps.iter().zip(&shell.coefs) {
        for (&b, &cb) in shell.exps.iter().zip(&shell.coefs) {
            let exponent = a + b;
            let term = density[(0, 0)] * ca * cb * (-exponent * r2).exp();
            expected[0] += term;
            for u in 0..3 {
                expected[1 + u] += -2.0 * exponent * r.get(u) * term;
            }
            for (i, (u, v)) in PAIRS.into_iter().enumerate() {
                let diagonal = if u == v { 2.0 * exponent } else { 0.0 };
                expected[4 + i] += (4.0 * exponent.powi(2) * r.get(u) * r.get(v) - diagonal) * term;
            }
        }
    }
    let mut maximum = 0.0_f64;
    for convention in [GradientConvention::Published, GradientConvention::Hirshfeld] {
        let result = analyze(
            &molecule,
            &basis,
            &density,
            &[point],
            None,
            convention,
            true,
        )
        .unwrap();
        let sample = &result.points[0];
        let sampled = [sample.density]
            .into_iter()
            .chain(sample.density_gradient.to_array())
            .chain(sample.density_hessian);
        for (value, expected) in sampled.zip(expected) {
            maximum = maximum.max((value - expected).abs());
        }
        assert_eq!(
            [sample.delta_g, sample.delta_g_inter, sample.delta_g_intra],
            [0.0; 3]
        );
        assert_eq!(sample.signed_density, -sample.density);
        assert_eq!(sample.atom_densities, [sample.density]);
        assert_eq!(sample.atom_vectors, [sample.density_gradient]);
    }
    assert!(
        maximum < 5e-16,
        "closed-form Gaussian field defect {maximum:.3e}"
    );
    println!("single hydrogen maximum closed-form Gaussian field defect {maximum:.3e}");
}

fn scalar_oracle(basis: &MolBasis, p: &Matrix, sad: &Matrix, point: Vec3) -> (f64, [f64; 6]) {
    let values = eval_aos(basis, &[point], false).phi;
    let owners = basis.sph_ao_atoms();
    let mut density = 0.0;
    let mut free = [0.0; 6];
    for (mu, &atom) in owners.iter().enumerate() {
        for (nu, &other) in owners.iter().enumerate() {
            let product = values[(0, mu)] * values[(0, nu)];
            density += product * p[(mu, nu)];
            if atom == other {
                free[atom] += product * sad[(mu, nu)];
            }
        }
    }
    let total = free.iter().sum::<f64>();
    (density, free.map(|x| x / total))
}

#[test]
fn water_dimer_igmh_matches_all_cartesian_density_derivatives() {
    c1_geometry::assert_c1(&Z, &CENTRES);
    let molecule = Molecule::new(
        Z.into_iter()
            .zip(CENTRES)
            .map(|(z, r)| Atom {
                z,
                position: vector(r),
            })
            .collect(),
    );
    let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
    let grid = MolecularGrid::build(
        &molecule,
        &GridOptions {
            level: 1,
            weight_cutoff: 0.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let xc = resolve(&FunctionalSpec::Combined("pbe".into())).unwrap();
    let scf = run_scf(&molecule, &basis, &ScfOptions::default(), &xc, Some(&grid)).unwrap();
    let mut p = scf.density[0].clone();
    for (a, b) in p.as_mut_slice().iter_mut().zip(scf.density[1].as_slice()) {
        *a += b;
    }
    let neutral = sad_density(&molecule, &basis).unwrap();
    let point = Vec3::new(0.31, -0.19, 3.13);
    let groups = [11, 11, 11, 27, 27, 27];
    let evaluate =
        |groups, mode| analyze(&molecule, &basis, &p, &[point], groups, mode, true).unwrap();
    let strict = evaluate(Some(&groups), GradientConvention::Hirshfeld);
    let published = evaluate(Some(&groups), GradientConvention::Published);
    let a = &strict.points[0];
    let b = &published.points[0];
    let (rho, w) = scalar_oracle(&basis, &p, &neutral, point);
    assert!((a.density - rho).abs() < 1e-14);
    assert!(strict.density_residual < 1e-14 && published.density_residual < 1e-14);
    assert!(strict.gradient_residual < 1e-14 && published.gradient_residual < 1e-14);
    for (atom, weight) in w.iter().enumerate() {
        assert!((a.atom_densities[atom] - weight * rho).abs() < 1e-14);
    }

    let mut names = Vec::new();
    let mut analytic = Vec::new();
    for (axis, value) in a.density_gradient.to_array().into_iter().enumerate() {
        names.push(format!("density gradient {axis}"));
        analytic.push(value);
    }
    for (atom, (v, u)) in a.atom_vectors.iter().zip(&b.atom_vectors).enumerate() {
        for (mode, vector) in [("hirshfeld", v), ("published", u)] {
            for (axis, value) in vector.to_array().into_iter().enumerate() {
                names.push(format!("{mode} atom {atom} axis {axis}"));
                analytic.push(value);
            }
        }
    }
    for (component, (u, v)) in PAIRS.into_iter().enumerate() {
        names.push(format!("density Hessian {u},{v}"));
        analytic.push(a.density_hessian[component]);
    }
    let disp = [
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    let eval = |d: Vec3| scalar_oracle(&basis, &p, &neutral, point + d);
    let oracle = |h: f64, method: Stencil| {
        let neighbours = disp.map(|d| {
            [
                eval(d * h),
                eval(-d * h),
                eval(d * (2. * h)),
                eval(-d * (2. * h)),
            ]
        });
        let gradient = neighbours.map(|d| method.first(d[0].0, d[1].0, d[2].0, d[3].0, h));
        let mut values = gradient.to_vec();
        for (atom, weight) in w.iter().enumerate() {
            let atomic_gradient = neighbours.map(|d| {
                let [p, m, p2, m2] = d.map(|(r, w)| r * w[atom]);
                method.first(p, m, p2, m2, h)
            });
            values.extend(atomic_gradient);
            // The published visualization vector is independently recovered
            // from the gradients of the zero-order molecular and stockholder densities.
            values.extend(std::array::from_fn::<_, 3, _>(|u| {
                2.0 * weight * gradient[u] - atomic_gradient[u]
            }));
        }
        for (u, v) in PAIRS {
            let central = |h| {
                if u == v {
                    (eval(disp[u] * h).0 - 2.0 * rho + eval(-disp[u] * h).0) / (h * h)
                } else {
                    (eval((disp[u] + disp[v]) * h).0
                        - eval((disp[u] - disp[v]) * h).0
                        - eval((-disp[u] + disp[v]) * h).0
                        + eval((-disp[u] - disp[v]) * h).0)
                        / (4.0 * h * h)
                }
            };
            values.push(match method {
                Stencil::Second => central(h),
                Stencil::Fourth => (4.0 * central(h / 2.0) - central(h)) / 3.0,
            });
        }
        values
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &names,
        &[
            ([0.036, 0.027, 0.019, 0.013], Stencil::Second),
            ([0.011, 0.008, 0.006, 0.004], Stencil::Second),
            ([0.075, 0.056, 0.038, 0.025], Stencil::Fourth),
        ],
        |steps, stencil| {
            let values = steps.map(|h| oracle(h, stencil));
            analytic
                .iter()
                .enumerate()
                .map(|(k, &value)| std::array::from_fn(|i| (values[i][k] - value).abs()))
                .collect()
        },
        &mut failures,
    );
    components::assert_no_failures("IGMH spatial derivatives", &failures);
    let absolute = oracle(0.0015, Stencil::Fourth)
        .iter()
        .zip(&analytic)
        .map(|(fd, v)| (fd - v).abs())
        .fold(0.0, f64::max);
    assert!(
        absolute < 2e-9,
        "fourth-order density derivative defect {absolute}"
    );
    let single_fragment = evaluate(None, GradientConvention::Published);
    let singletons = evaluate(Some(&[0, 2, 4, 6, 8, 10]), GradientConvention::Published);
    assert_eq!(single_fragment.points[0].delta_g_inter, 0.0);
    assert_eq!(singletons.points[0].delta_g_intra, 0.0);
    assert!((b.delta_g - b.delta_g_intra - b.delta_g_inter).abs() < 1e-14);
    assert!(b.delta_g_inter > 1e-4 && b.signed_density < 0.0);
    println!("water dimer PBE E={:.15e} rho={rho:.15e} published fields [{:.15e}, {:.15e}, {:.15e}] signed density={:.15e}; absolute defect={absolute:.3e}; partition defects [{:.3e}, {:.3e}]",
        scf.energy, b.delta_g, b.delta_g_inter, b.delta_g_intra, b.signed_density,
        published.density_residual, published.gradient_residual);
}
