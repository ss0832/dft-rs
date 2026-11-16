// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Original ωB97X-D3 term, response and dispersion acceptance.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/wb97x_d3_oracle.rs"]
mod oracle;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::dispersion::{d3_zero_params, Dispersion};
use dft_rs::driver::method::{prepare_molecular, run_molecular_reference, MolecularMethod};
use dft_rs::driver::{run, Controls, System, Value};
use dft_rs::dual::{DualN, Scalar};
use dft_rs::grid::eval::eval_aos;
use dft_rs::integrals::{eri_incore, kinetic_matrix, nuclear_matrix, CoulombKernel};
use dft_rs::periodic::dispersion::{
    dimensional_dispersion_force_constant_tensor, PeriodicDispersion,
};
use dft_rs::xc::{
    resolve, Family, FunctionalSpec, GridVars, RequiredNonlocalTerm, XcFunctional, XcOut,
    XcTaylorInput,
};
use dft_rs::Vec3;
use stencils::Stencil;

const XC: &str = "wb97x-d3";
const LOCAL: [f64; 5] = [0.31, 0.22, 0.041, 0.017, 0.029];
const STAGES: [([f64; 4], Stencil); 4] = [
    ([0.08, 0.04, 0.02, 0.01], Stencil::Second),
    ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
    ([0.16, 0.08, 0.04, 0.02], Stencil::Second),
    ([0.10, 0.075, 0.05, 0.035], Stencil::Fourth),
];

fn point_vars(v: &[[f64; 1]; 5]) -> GridVars<'_> {
    GridVars {
        rho: [&v[0], &v[1]],
        sigma: Some([&v[2], &v[3], &v[4]]),
        tau: None,
    }
}

fn evaluate(f: &dyn XcFunctional, point: [f64; 5], order: usize) -> XcOut {
    let mut out = XcOut::default();
    out.prepare(1, order, Family::Gga);
    f.eval_add(&point_vars(&point.map(|x| [x])), order, 1.0, &mut out)
        .unwrap();
    out
}

fn component(f: &dyn XcFunctional, which: usize) -> Box<dyn XcFunctional> {
    let mut parameters = f.params();
    for k in 0..3 {
        if k != which {
            parameters[4 + 5 * k..9 + 5 * k].fill(0.0);
        }
    }
    f.with_params(&parameters).unwrap()
}

#[test]
fn functional_terms_damping_and_physical_limits() {
    let f = resolve(&FunctionalSpec::Combined(XC.into())).unwrap();
    assert_eq!(
        f.required_nonlocal_kind(),
        Some(RequiredNonlocalTerm::D3Zero)
    );
    assert_eq!(f.required_pair_dispersion(), Some(Dispersion::D3Zero));
    assert_eq!(f.hybrid.cx, 0.195728);
    assert_eq!(f.hybrid.clr + f.hybrid.cx, 1.0);
    assert_eq!(f.hybrid.omega, 0.25);
    let p = d3_zero_params("WB97X_D3").unwrap();
    assert_eq!(
        [p.s6, p.s8, p.rs6, p.rs8, p.alp6, p.alp8],
        [1.0, 1.0, 1.281, 1.094, 14.0, 16.0]
    );
    let part = &*f.parts[0].1;
    for (point, want) in [
        (
            LOCAL,
            [
                -0.20217596299780535,
                -0.010701426212140007,
                -0.017980662850889333,
            ],
        ),
        (
            [0.61, 0.11, 0.078, -0.028, 0.017],
            [
                -0.3516897470072206,
                -0.01871089096887487,
                -0.01649368446013476,
            ],
        ),
    ] {
        let oracle = oracle::terms(point, &oracle::PARAMETERS);
        let [a, b, aa, ab, bb] = point;
        let rotated = [b, a, bb, ab, aa];
        let mut sum = 0.0;
        for k in 0..3 {
            let component = component(part, k);
            let found = evaluate(&*component, point, 0).e[0];
            assert!((found - want[k]).abs() < 2e-14);
            assert!((oracle[k] - want[k]).abs() < 2e-14);
            assert!((found - evaluate(&*component, rotated, 0).e[0]).abs() < 2e-14);
            println!(
                "{XC} {point:?} term {k}: density {found:.16e} defect {:.3e}",
                (found - want[k]).abs()
            );
            sum += found;
        }
        assert!((evaluate(part, point, 0).e[0] - sum).abs() < 2e-14);
    }
    let only_up = [0.35, 0.0, 0.014, 0.0, 0.0];
    assert_eq!(evaluate(&*component(part, 2), only_up, 0).e[0], 0.0);
    let mut unscreened = part.params();
    unscreened[3] = 0.0;
    let mut far_screened = part.params();
    far_screened[3] = 2e4;
    let x = component(part, 0);
    for params in [unscreened, far_screened] {
        let mut parameters = x.params();
        parameters[3] = params[3];
        let energy = evaluate(&*x.with_params(&parameters).unwrap(), LOCAL, 0).e[0];
        let want = oracle::terms(LOCAL, &params.try_into().unwrap())[0];
        assert!((energy - want).abs() < 2e-14);
        if parameters[3] > 0.0 {
            assert!(energy.abs() < 1e-9);
        }
    }
}

fn independent_indices(n: usize, order: usize) -> Vec<Vec<usize>> {
    (0..n.pow(order as u32))
        .filter_map(|mut flat| {
            let mut indices = vec![0; order];
            for index in indices.iter_mut().rev() {
                *index = flat % n;
                flat /= n;
            }
            indices
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
                .then_some(indices)
        })
        .collect()
}

fn local_third(out: &XcOut, a: usize, b: usize, c: usize) -> f64 {
    match (a, b, c) {
        (0..=1, 0..=1, 0..=1) => out.v3rho3[a + b + c][0],
        (0..=1, 0..=1, 2..=4) => out.v3rho2sigma[3 * (a + b) + c - 2][0],
        (0..=1, 2..=4, 2..=4) => {
            let pair = (2..b).map(|i| 5 - i).sum::<usize>() + c - b;
            out.v3rhosigma2[6 * a + pair][0]
        }
        _ => {
            let index = independent_indices(3, 3)
                .iter()
                .position(|x| x == &[a - 2, b - 2, c - 2])
                .unwrap();
            out.v3sigma3[index][0]
        }
    }
}

fn local_fourth(f: &dyn XcFunctional, a: usize, b: usize, c: usize, d: usize) -> f64 {
    let input = XcTaylorInput {
        variables: std::array::from_fn(|i| {
            vec![LOCAL.get(i).copied().unwrap_or_default(), f64::from(i == a)]
        }),
        parameters: None,
    };
    let seed = |axis| std::array::from_fn(|i| vec![f64::from(i == axis), 0.0]);
    let mixed = std::array::from_fn(|_| vec![0.0; 2]);
    f.taylor_bidirectional_potentials(&input, &seed(b), &seed(c), &mixed)
        .unwrap()
        .mixed
        .potentials[d][1]
}

type D1 = DualN<f64, 1>;
type D2 = DualN<D1, 1>;
type D3 = DualN<D2, 1>;

fn seed<S: Scalar, const N: usize>(p: [S; N], axis: usize) -> [DualN<S, 1>; N] {
    std::array::from_fn(|i| {
        if i == axis {
            DualN::variable(p[i], 0)
        } else {
            DualN::constant(p[i])
        }
    })
}

fn oracle_local<S: Scalar>(point: [S; 5], term: usize) -> S {
    let terms = oracle::terms(point, &oracle::PARAMETERS.map(S::from_f64));
    terms
        .get(term)
        .copied()
        .unwrap_or_else(|| terms[0] + terms[1] + terms[2])
}

fn local_lower(point: [f64; 5], term: usize, indices: &[usize]) -> f64 {
    match *indices {
        [] => oracle_local(point, term),
        [a] => oracle_local::<D1>(seed(point, a), term).im[0],
        [a, b] => oracle_local::<D2>(seed(seed(point, a), b), term).im[0].im[0],
        [a, b, c] => oracle_local::<D3>(seed(seed(seed(point, a), b), c), term).im[0].im[0].im[0],
        _ => unreachable!(),
    }
}

fn central(value: impl Fn(f64) -> f64, h: f64, stencil: Stencil) -> f64 {
    let (plus2, minus2) = if stencil == Stencil::Fourth {
        (value(2.0 * h), value(-2.0 * h))
    } else {
        (0.0, 0.0)
    };
    stencil.first(value(h), value(-h), plus2, minus2, h)
}

#[test]
fn every_local_component_through_fourth_order_and_all_parameter_tangents() {
    let f = resolve(&FunctionalSpec::Combined(XC.into())).unwrap();
    let full = &*f.parts[0].1;
    let mut failures = Vec::new();
    let steps = STAGES;
    let scales = [0.15, 0.09, 0.1, 0.04, 0.08];
    for term in 0..4 {
        let part = (term < 3).then(|| component(full, term));
        let f = part.as_deref().unwrap_or(full);
        let out = evaluate(f, LOCAL, 3);
        let (first, second) = out.potential_and_kernel(0, Family::Gga).unwrap();
        let mut entries = Vec::new();
        let mut zeros = 0;
        for order in 1..=4 {
            for indices in independent_indices(5, order) {
                let value = match *indices.as_slice() {
                    [a] => first[a],
                    [a, b] => second[a][b],
                    [a, b, c] => local_third(&out, a, b, c),
                    [a, b, c, d] => local_fourth(f, a, b, c, d),
                    _ => unreachable!(),
                };
                let structural_zero = indices.contains(&3)
                    || (term < 2
                        && ![[0, 2], [1, 4]]
                            .iter()
                            .any(|spin| indices.iter().all(|i| spin.contains(i))));
                if structural_zero {
                    assert_eq!(value, 0.0, "term {term} {indices:?}");
                    zeros += 1;
                } else {
                    entries.push((indices, value));
                }
            }
        }
        let labels: Vec<_> = entries
            .iter()
            .map(|(i, _)| format!("{XC} term {term} D{i:?}"))
            .collect();
        println!(
            "{XC} term {term}: {} fitted local derivatives and {zeros} structural zeros",
            labels.len()
        );
        stencils::fit_components_staged(
            &labels,
            &steps,
            |sizes, stencil| {
                entries
                    .iter()
                    .map(|(indices, analytic)| {
                        let (&axis, lower) = indices.split_last().unwrap();
                        sizes.map(|h| {
                            let fd = central(
                                |x| {
                                    let mut shifted = LOCAL;
                                    shifted[axis] += x;
                                    local_lower(shifted, term, lower)
                                },
                                h * scales[axis],
                                stencil,
                            );
                            (fd - analytic).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    let mut parameters = Vec::new();
    full.param_gradient(&point_vars(&LOCAL.map(|x| [x])), &mut parameters)
        .unwrap();
    let labels: Vec<_> = full.param_names()[..4]
        .iter()
        .map(|name| format!("{XC} d/d{name}"))
        .collect();
    let energy = |index, step| {
        let mut changed = oracle::PARAMETERS;
        changed[index] += step;
        oracle::terms(LOCAL, &changed).into_iter().sum::<f64>()
    };
    stencils::fit_components_staged(
        &labels,
        &steps,
        |sizes, stencil| {
            (0..4)
                .map(|index| {
                    sizes.map(|step| {
                        let h = step * oracle::PARAMETERS[index];
                        let fd = central(|x| energy(index, x), h, stencil);
                        (fd - parameters[index][0]).abs()
                    })
                })
                .collect()
        },
        &mut failures,
    );
    for (index, reported) in parameters.iter().enumerate().skip(4) {
        // The fitted polynomial coefficients are strictly linear. Exact
        // finite changes give their derivatives without a truncation term.
        let want = energy(index, 0.5) - energy(index, -0.5);
        assert!((reported[0] - want).abs() < 5e-16);
    }
    components::assert_no_failures("ωB97X-D3 local derivatives", &failures);
}

const NUMBERS: [u8; 4] = [7, 1, 1, 1];
const AMMONIA: [[f64; 3]; 4] = [
    [0.13, -0.21, 0.09],
    [1.61, 0.77, -0.29],
    [-1.34, 0.61, 0.33],
    [0.31, -1.74, 0.58],
];

#[test]
fn neutral_and_cation_ammonia_energies_reconstruct_from_independent_terms() {
    let dispersion = oracle::DispersionOracle::new(
        NUMBERS,
        &AMMONIA,
        [[0.0; 3]; 3],
        [false; 3],
        99.0,
        99.0,
        0.0,
    );
    for charge in [0, 1] {
        let method = MolecularMethod {
            xc: Some(XC.into()),
            grid_level: 1,
            grid_weight_cutoff: 0.0,
            charge,
            multiplicity: 1 + charge as usize,
            unrestricted: charge > 0,
            incore_mb: 128,
            conv_tol: 1e-12,
            conv_tol_dp: 1e-11,
            conv_tol_grad: Some(1e-9),
            ..MolecularMethod::default()
        };
        let setup = prepare_molecular(&NUMBERS, &AMMONIA, &method).unwrap();
        assert_eq!(setup.dispersion, Dispersion::D3Zero);
        let reference = run_molecular_reference(&setup).unwrap();
        let state = &reference.scf;
        assert!(
            state.converged,
            "NH3 charge {charge} / {}",
            state.iterations
        );
        let basis = &setup.basis;
        let [alpha, beta] = &state.density;
        let mut total = alpha.clone();
        for (entry, beta_entry) in total.as_mut_slice().iter_mut().zip(beta.as_slice()) {
            *entry += beta_entry;
        }
        let n = basis.nao_sph;
        let kinetic = kinetic_matrix(basis).frobenius_dot(&total);
        let nuclear = nuclear_matrix(basis, &setup.molecule, None).frobenius_dot(&total);
        let repulsion = setup.molecule.nuclear_repulsion();
        let full = eri_incore(basis, CoulombKernel::Coulomb, 128).unwrap();
        let long = eri_incore(basis, CoulombKernel::Erf { omega: 0.25 }, 128).unwrap();
        let mut hartree = 0.0;
        let mut exact = [0.0; 2];
        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    for d in 0..n {
                        hartree += 0.5 * total[(a, b)] * total[(c, d)] * full.get(a, b, c, d);
                        let same_spin =
                            -0.5 * (alpha[(a, c)] * alpha[(b, d)] + beta[(a, c)] * beta[(b, d)]);
                        exact[0] += 0.195728 * same_spin * full.get(a, b, c, d);
                        exact[1] += 0.804272 * same_spin * long.get(a, b, c, d);
                    }
                }
            }
        }
        let grid = setup.grid.as_ref().unwrap();
        let ao = eval_aos(basis, &grid.points, true);
        let ao_gradient = ao.grad.unwrap();
        let mut semilocal = [0.0; 3];
        for (row, &weight) in grid.weights.iter().enumerate() {
            let mut local = [0.0; 5];
            for (spin, density) in state.density.iter().enumerate() {
                let mut gradient = [0.0; 3];
                for mu in 0..n {
                    for nu in 0..n {
                        let v = density[(mu, nu)] * ao.phi[(row, mu)];
                        local[spin] += v * ao.phi[(row, nu)];
                        for (axis, drho) in gradient.iter_mut().enumerate() {
                            *drho += 2.0 * v * ao_gradient[axis][(row, nu)];
                        }
                    }
                }
                local[2 + 2 * spin] = gradient.iter().map(|x| x * x).sum();
            }
            for (sum, term) in semilocal
                .iter_mut()
                .zip(oracle::terms(local, &oracle::PARAMETERS))
            {
                *sum += weight * term;
            }
        }
        let pair = dispersion.terms(&AMMONIA);
        let expected = kinetic
            + nuclear
            + repulsion
            + hartree
            + exact.iter().sum::<f64>()
            + semilocal.iter().sum::<f64>()
            + pair.iter().sum::<f64>();
        let defect = (reference.total_energy() - expected).abs();
        assert!(defect < 2e-10, "total reconstruction defect {defect:.3e}");
        assert!((reference.e_dispersion - pair.iter().sum::<f64>()).abs() < 2e-14);
        assert!((state.e_xc - semilocal.iter().sum::<f64>()).abs() < 2e-11);
        let residual = state
            .mo_coeff
            .iter()
            .zip(&state.fock)
            .zip([state.n_alpha, state.n_beta])
            .map(|((orbitals, fock), occupied)| {
                let fmo = orbitals.transpose_matmul(&fock.matmul(orbitals));
                (0..occupied)
                    .flat_map(|i| (occupied..n).map(move |a| (i, a)))
                    .map(|i| fmo[i].abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        assert!(residual < 1e-9);
        println!("NH3 charge={charge} wB97X-D3/STO-3G: E={:.16e}, KS terms={semilocal:?}, full/LR K={exact:?}, D3 C6/C8={pair:?}, orbital={residual:.3e}, defect={defect:.3e}, iterations={}", reference.total_energy(), state.iterations);
    }
}

fn total_electronic_energy(system: &System, charge: i32) -> f64 {
    let controls = molecular_controls(charge);
    let output = run("energy", system, &controls).unwrap();
    assert_eq!(output.get("converged").and_then(Value::as_bool), Some(true));
    output.get("energy").and_then(Value::as_f64).unwrap()
}

fn molecular_controls(charge: i32) -> Controls {
    Controls::from_value(
        &Value::object()
            .with("xc", XC)
            .with("unrestricted", charge > 0)
            .with("charge", charge)
            .with("multiplicity", 1 + charge as usize)
            .with("grid_level", 1)
            .with("grid_weight_cutoff", 0.0)
            .with("conv_tol", 1e-12)
            .with("conv_tol_dp", 1e-11)
            .with("conv_tol_grad", 1e-9),
    )
    .unwrap()
}

#[test]
fn ammonia_and_ammonia_cation_have_complete_nuclear_gradient_fits() {
    let system = System::molecule(NUMBERS.to_vec(), AMMONIA.to_vec());
    let mut failures = Vec::new();
    for charge in [0, 1] {
        let output = run("gradient", &system, &molecular_controls(charge)).unwrap();
        let (shape, gradient) = output.get("gradient").unwrap().numeric_array().unwrap();
        assert_eq!(shape, [4, 3]);
        assert!(
            output
                .get("translation_residual")
                .and_then(Value::as_f64)
                .unwrap()
                < 1e-9
        );
        let names = (0..12)
            .map(|i| format!("NH3 charge {charge} dE/dR[{},{}]", i / 3, i % 3))
            .collect::<Vec<_>>();
        let stages = [
            ([0.02, 0.01, 0.005, 0.0025], Stencil::Second),
            ([0.01, 0.005, 0.0025, 0.00125], Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], Stencil::Second),
            ([0.025, 0.018, 0.012, 0.008], Stencil::Fourth),
        ];
        stencils::fit_components_staged(
            &names,
            &stages,
            |steps, stencil| {
                gradient
                    .iter()
                    .enumerate()
                    .map(|(i, analytic)| {
                        steps.map(|h| {
                            let fd = central(
                                |dx| {
                                    let mut moved = system.clone();
                                    moved.positions[i / 3][i % 3] += dx;
                                    total_electronic_energy(&moved, charge)
                                },
                                h,
                                stencil,
                            );
                            (fd - analytic).abs()
                        })
                    })
                    .collect()
            },
            &mut failures,
        );
    }
    components::assert_no_failures("ωB97X-D3 molecular gradients", &failures);
}

fn periodic_energy<S: Scalar>(model: &oracle::DispersionOracle<2>, point: [S; 6]) -> S {
    let [x0, y0, z0, x1, y1, z1] = point;
    let [c6, c8] = model.terms(&[[x0, y0, z0], [x1, y1, z1]]);
    c6 + c8
}

fn periodic_lower(model: &oracle::DispersionOracle<2>, point: [f64; 6], indices: &[usize]) -> f64 {
    match *indices {
        [] => periodic_energy(model, point),
        [a] => periodic_energy::<D1>(model, seed(point, a)).im[0],
        [a, b] => periodic_energy::<D2>(model, seed(seed(point, a), b)).im[0].im[0],
        [a, b, c] => {
            periodic_energy::<D3>(model, seed(seed(seed(point, a), b), c)).im[0].im[0].im[0]
        }
        _ => unreachable!(),
    }
}

#[test]
fn periodic_zero_damping_fit_covers_all_cartesian_components_through_order_four() {
    use dft_rs::driver::periodic::{dimensional_geometry_of, PeriodicMethod};

    let numbers = [8, 1];
    let positions = [[0.19, -0.27, 0.41], [1.56, 0.51, -0.65]];
    let lattice = [[8.0, 0.37, 0.42], [0.61, 8.8, 0.74], [0.3, -0.7, 10.1]];
    let (cn, pair) = (10.41, 12.98);
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", XC)
            .with("cn_cutoff", cn)
            .with("disp_cutoff", pair),
    )
    .unwrap();
    let method = PeriodicMethod::from_controls_with_dispersion(&controls).unwrap();
    let settings: PeriodicDispersion = method.dispersion_model_for(&numbers).unwrap().unwrap();
    assert!(settings.three_body.is_none());
    let atoms = positions.map(|[x, y, z]| Vec3::new(x, y, z));
    let point = [
        positions[0][0],
        positions[0][1],
        positions[0][2],
        positions[1][0],
        positions[1][1],
        positions[1][2],
    ];
    let mut failures = Vec::new();
    for pbc in [[true, false, false], [true, false, true], [true; 3]] {
        let system = System {
            numbers: numbers.to_vec(),
            positions: positions.to_vec(),
            cell: Some(lattice),
            pbc,
        };
        let geometry = dimensional_geometry_of(&system).unwrap();
        let model =
            oracle::DispersionOracle::new(numbers, &positions, lattice, pbc, cn, pair, 0.24);
        let energies = model.terms(&positions);
        for order in 1..=4 {
            let calculated = dimensional_dispersion_force_constant_tensor(
                &numbers, &atoms, &geometry, &settings, order,
            )
            .unwrap();
            assert!((calculated.energy - energies.iter().sum::<f64>()).abs() < 2e-14);
            assert!(calculated.permutation_residual < 1e-11);
            assert!(calculated.acoustic_residual < 1e-11);
            let indices = independent_indices(6, order);
            let names = indices
                .iter()
                .map(|i| format!("D3(0) {pbc:?} D{i:?}"))
                .collect::<Vec<_>>();
            println!("wB97X-D3 periodic {pbc:?} order {order}: C6/C8={energies:?}, permutation={:.3e}, acoustic={:.3e}, {} components", calculated.permutation_residual, calculated.acoustic_residual, indices.len());
            stencils::fit_components_staged(
                &names,
                &STAGES,
                |steps, stencil| {
                    indices
                        .iter()
                        .map(|axes| {
                            let flat = axes.iter().fold(0, |offset, i| 6 * offset + i);
                            let (&axis, lower) = axes.split_last().unwrap();
                            steps.map(|h| {
                                let fd = central(
                                    |dx| {
                                        let mut moved = point;
                                        moved[axis] += dx;
                                        periodic_lower(&model, moved, lower)
                                    },
                                    h,
                                    stencil,
                                );
                                (fd - calculated.tensor[flat]).abs()
                            })
                        })
                        .collect()
                },
                &mut failures,
            );
        }
    }
    components::assert_no_failures("ωB97X-D3 dimensional periodic pair derivatives", &failures);
}

#[test]
fn required_pair_selection_survives_aliases_and_refuses_conflicting_models() {
    use dft_rs::driver::periodic::PeriodicMethod;

    let method = MolecularMethod {
        x: Some("WB97X_D3".into()),
        c: Some(XC.into()),
        grid_level: 0,
        ..MolecularMethod::default()
    };
    assert_eq!(
        prepare_molecular(&NUMBERS, &AMMONIA, &method)
            .unwrap()
            .dispersion,
        Dispersion::D3Zero
    );
    let disabled = MolecularMethod {
        dispersion: "none".into(),
        ..method
    };
    assert!(prepare_molecular(&NUMBERS, &AMMONIA, &disabled).is_err());
    for incompatible in [
        Value::object().with("dispersion", "d3bj"),
        Value::object().with("three_body", true),
        Value::object().with("vv10", true),
    ] {
        let controls = Controls::from_value(&incompatible.with("xc", XC)).unwrap();
        assert!(PeriodicMethod::from_controls_with_dispersion(&controls).is_err());
    }
    let controls = Controls::from_value(&Value::object().with("xc", XC)).unwrap();
    assert!(PeriodicMethod::from_controls(&controls).is_err());
}
