// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832

//! Coupled cubic HF induction against reconverged monomer product energies.

#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{data, MolBasis};
use dft_rs::integrals::eri::{eri_incore, EriTensor};
use dft_rs::integrals::one_electron::{nuclear_matrix, overlap_matrix};
use dft_rs::integrals::CoulombKernel;
use dft_rs::linalg::Matrix;
use dft_rs::math::Vec3;
use dft_rs::sapt::{sapt_interaction_diagnostics, SaptConfig, SaptDiagnostics, SaptVariant};
use dft_rs::scf::{run_scf, run_scf_with_density, EriMode, ScfOptions};
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::ResolvedFunctional;
use std::cell::RefCell;
use std::collections::BTreeMap;

fn water(points: [[f64; 3]; 3]) -> Molecule {
    Molecule::new(
        points
            .into_iter()
            .zip([8, 1, 1])
            .map(|(p, z)| Atom {
                z,
                position: Vec3::new(p[0], p[1], p[2]),
            })
            .collect(),
    )
}

fn cross_field(eri: &EriTensor, density: &Matrix, offsets: [usize; 2], n: usize) -> Matrix {
    let mut field = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..density.rows {
                for l in 0..density.cols {
                    field[(i, j)] += density[(k, l)]
                        * eri.get(
                            i + offsets[0],
                            j + offsets[0],
                            k + offsets[1],
                            l + offsets[1],
                        );
                }
            }
        }
    }
    field
}

#[test]
fn water_coupled_cubic_induction_matches_reconverged_product_scf() {
    let monomers = [
        water([[0.0, 0.0, 0.0], [0.04, 0.10, 1.81], [1.72, 0.10, -0.45]]),
        water([[0.12, -0.24, 5.35], [1.45, 0.66, 6.20], [-1.00, 1.04, 6.01]]),
    ];
    let set = data::load("6-31g").unwrap();
    let f = ResolvedFunctional::hartree_fock();
    let opts = ScfOptions {
        e_tol: 5e-14,
        p_tol: 1e-12,
        max_iter: 200,
        eri_mode: EriMode::InCore { limit_mb: 256 },
        ..ScfOptions::default()
    };
    let basis = monomers
        .each_ref()
        .map(|m| MolBasis::build(m, &set).unwrap());
    let bare: [_; 2] =
        std::array::from_fn(|i| run_scf(&monomers[i], &basis[i], &opts, &f, None).unwrap());
    let dimer = Molecule::new(monomers.iter().flat_map(|m| m.atoms.clone()).collect());
    let dimer_basis = MolBasis::build(&dimer, &set).unwrap();
    let eri = eri_incore(&dimer_basis, CoulombKernel::Coulomb, 256).unwrap();
    let offset = [0, basis[0].nao_sph];
    let external = [
        nuclear_matrix(&basis[0], &monomers[1], None),
        nuclear_matrix(&basis[1], &monomers[0], None),
    ];
    let nuclear = dimer.nuclear_repulsion()
        - monomers
            .iter()
            .map(Molecule::nuclear_repulsion)
            .sum::<f64>();
    let bare_energy: f64 = bare.iter().map(|x| x.energy).sum();
    let bare_density = bare.each_ref().map(|s| s.p_total());
    let perturbation = |p: &[Matrix; 2], i: usize| {
        cross_field(
            &eri,
            &p[1 - i],
            [offset[i], offset[1 - i]],
            basis[i].nao_sph,
        )
    };
    let first = bare_density[0].frobenius_dot(&external[0])
        + bare_density[1].frobenius_dot(&external[1])
        + bare_density[0].frobenius_dot(&perturbation(&bare_density, 0))
        + nuclear;
    let baseline_field = [0, 1].map(|m| {
        let mut field = perturbation(&bare_density, m);
        for (x, y) in field.as_mut_slice().iter_mut().zip(external[m].as_slice()) {
            *x += y;
        }
        field
    });
    let metric_orbitals = [0, 1].map(|m| overlap_matrix(&basis[m]).matmul(&bare[m].mo_coeff[0]));
    let monomer_energy_change = |m: usize, c: &Matrix, delta: &Matrix| {
        // This is Tr(F0 delta P) evaluated in the stationary MO frame.
        // Compute occupied population loss from the complementary virtual
        // block to avoid subtracting nearly equal occupations.
        let w = metric_orbitals[m].transpose_matmul(c);
        let no = bare[m].n_alpha;
        let energies = &bare[m].mo_energy[0];
        let n = w.rows;
        let mut one_body = 0.0;
        for i in 0..no {
            for j in no..n {
                one_body +=
                    2.0 * (energies[j] * w[(j, i)].powi(2) - energies[i] * w[(i, j)].powi(2));
            }
        }
        let coulomb = delta.frobenius_dot(&cross_field(&eri, delta, [offset[m]; 2], n));
        let mut exchange = 0.0;
        let p = offset[m];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    for l in 0..n {
                        exchange +=
                            delta[(i, j)] * delta[(k, l)] * eri.get(p + i, p + k, p + j, p + l);
                    }
                }
            }
        }
        one_body + 0.5 * coulomb - 0.25 * exchange
    };
    let solve_product = |h: f64| {
        let mut densities = bare_density.clone();
        let mut guess = bare.each_ref().map(|r| r.density.clone());
        for _ in 0..40 {
            let mut induction = [0.0; 2];
            let mut coefficients = bare.each_ref().map(|b| b.mo_coeff[0].clone());
            let mut next = densities.clone();
            for i in 0..2 {
                let field = perturbation(&densities, i);
                let extra_hcore = Matrix::from_row_major(
                    field.rows,
                    field.cols,
                    field
                        .as_slice()
                        .iter()
                        .zip(external[i].as_slice())
                        .map(|(j, v)| h * (j + v))
                        .collect(),
                );
                let scf = run_scf_with_density(
                    &monomers[i],
                    &basis[i],
                    &ScfOptions {
                        extra_hcore: Some(extra_hcore),
                        ..opts.clone()
                    },
                    &f,
                    None,
                    Some(&guess[i]),
                )
                .unwrap();
                induction[i] = scf.energy;
                coefficients[i] = scf.mo_coeff[0].clone();
                next[i] = scf.p_total();
                guess[i] = scf.density;
            }
            let residual = next
                .iter()
                .zip(&densities)
                .flat_map(|(a, b)| a.as_slice().iter().zip(b.as_slice()))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            if residual < 1e-11 {
                let double_counting = next[0].frobenius_dot(&perturbation(&next, 0));
                let energy =
                    (induction[0] + induction[1] - bare_energy) + h * (nuclear - double_counting);
                let deltas = [0, 1].map(|m| {
                    let mut delta = next[m].clone();
                    for (x, y) in delta
                        .as_mut_slice()
                        .iter_mut()
                        .zip(bare_density[m].as_slice())
                    {
                        *x -= y;
                    }
                    delta
                });
                // The same variational energy minus its linear
                // electrostatic term, evaluated from density differences.
                let static_field = deltas
                    .iter()
                    .zip(&baseline_field)
                    .map(|(d, w)| d.frobenius_dot(w))
                    .sum::<f64>();
                let mutual = deltas[0].frobenius_dot(&perturbation(&deltas, 0));
                let detrended = monomer_energy_change(0, &coefficients[0], &deltas[0])
                    + monomer_energy_change(1, &coefficients[1], &deltas[1])
                    + h * (static_field + mutual);
                return [energy, detrended];
            }
            densities = next;
        }
        panic!("independent mutually polarized monomers failed to converge at {h}");
    };
    let energies = RefCell::new(BTreeMap::new());
    let product = |h: f64| {
        *energies
            .borrow_mut()
            .entry(h.to_bits())
            .or_insert_with(|| solve_product(h))
    };
    let calculate = |a: &Molecule, b: &Molecule| {
        sapt_interaction_diagnostics(
            a,
            b,
            &set,
            &f,
            &opts,
            None,
            &SaptConfig::default(),
            SaptVariant::Sapt0,
            SaptDiagnostics {
                relaxed_induction30: true,
                ..SaptDiagnostics::default()
            },
        )
        .unwrap()
    };
    let measure = calculate(&monomers[0], &monomers[1]);
    let swapped = calculate(&monomers[1], &monomers[0])
        .relaxed_induction30
        .unwrap();
    let relaxed = measure.relaxed_induction30.unwrap();
    assert!((relaxed.induction - swapped.induction).abs() < 1e-12);
    assert!((first - measure.e_elst).abs() < 1e-12);
    let name = ["relaxed HF induction cubic coefficient".to_string()];
    let estimate = |h: f64| {
        (stencils::Stencil::Second.first(product(h)[0], product(-h)[0], 0.0, 0.0, h) - first)
            / (h * h)
    };
    let mut failures = Vec::new();
    stencils::fit_components_staged(
        &name,
        &[
            ([0.8, 0.6, 0.4, 0.3], stencils::Stencil::Second),
            ([1.4, 1.1, 0.8, 0.6], stencils::Stencil::Second),
            ([1.0, 0.8, 0.6, 0.45], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            vec![steps.map(|h| {
                let e = estimate(h);
                let approximation = match stencil {
                    stencils::Stencil::Second => e,
                    stencils::Stencil::Fourth => (4.0 * estimate(h / 2.0) - e) / 3.0,
                };
                (approximation - relaxed.induction).abs()
            })]
        },
        &mut failures,
    );
    components::assert_no_failures("relaxed third-order HF induction", &failures);
    let [coarse, middle, fine] = [0.8, 0.4, 0.2].map(estimate);
    let richardson_error =
        ((64.0 * fine - 20.0 * middle + coarse) / 45.0 - relaxed.induction).abs();
    assert!(richardson_error < 2e-10);
    println!(
        "C1 H2O dimer relaxed induction independent Richardson residual {richardson_error:.3e} Eh"
    );
    let detrended = |h: f64| {
        stencils::Stencil::Second.first(product(h)[1], product(-h)[1], 0.0, 0.0, h) / h.powi(2)
    };
    let h = [0.8, 0.6, 0.4, 0.3];
    let errors = h.map(|h| (detrended(h) - relaxed.induction).abs());
    assert!(components::component_fit_of_order(
        "relaxed HF induction detrended energy",
        &h,
        &errors,
        2.0,
    ));
    for h in [0.4, 0.3, 0.2, 0.1, 0.06] {
        let [a, b, c] = [h, h / 2.0, h / 4.0].map(detrended);
        let reference = (64.0 * c - 20.0 * b + a) / 45.0;
        let error = (reference - relaxed.induction).abs();
        println!("C1 H2O dimer detrended Richardson base {h}: residual {error:.3e} Eh");
        if h == 0.4 {
            assert!(error < 2e-13);
        }
    }
    let cp = |keep: Option<usize>| {
        let mut atoms = dimer.atoms.clone();
        if let Some(kept) = keep {
            for (index, atom) in atoms.iter_mut().enumerate() {
                if index / 3 != kept {
                    atom.z = 0;
                }
            }
        }
        run_scf(&Molecule::new(atoms), &dimer_basis, &opts, &f, None)
            .unwrap()
            .energy
    };
    let independently_counterpoise = cp(None) - cp(Some(0)) - cp(Some(1));
    let lower_order =
        measure.e_elst + measure.e_exch + measure.e_ind_a + measure.e_ind_b + measure.e_exch_ind;
    let delta_error = (relaxed.delta_hf_order2 + lower_order - independently_counterpoise).abs();
    assert!(delta_error < 1e-12);
    println!("C1 H2O dimer 6-31G relaxed induction {:+.15e} Eh, delta HF(2) {:+.15e} Eh, counterpoise sum residual {delta_error:.3e} Eh", relaxed.induction, relaxed.delta_hf_order2);
}
