// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relaxed orbital-field reference for all transverse spinor response entries.
#[path = "support/components.rs"]
mod components;
#[path = "support/convergence.rs"]
mod convergence;
#[path = "support/stencils.rs"]
mod stencils;

use dft_rs::basis::{BasisSet, ElementBasis, MolBasis, Shell};
use dft_rs::math::Vec3;
use dft_rs::periodic::electrostatics::ChargeCompensation;
use dft_rs::periodic::features::PeriodicBoundary;
use dft_rs::periodic::magnetic_scf::StaticPeriodicField;
use dft_rs::periodic::scf_boundary::DimensionalScfOptions;
use dft_rs::periodic::spinor_scf::{
    pauli_components, periodic_pauli_operators, run_spinor_scf, spin_diagonal, BiasedSpinorBackend,
    SpinorDftBackend, SpinorScfConfig,
};
use dft_rs::periodic::{BlochMatrix, Lattice, PeriodicGeometry};
use dft_rs::response::complex::Complex;
use dft_rs::response::ResponseSolverConfig;
use dft_rs::system::{Atom, Molecule};
use dft_rs::xc::{resolve, FunctionalSpec};
use std::collections::BTreeMap;

#[test]
fn periodic_helium_hydride_spinor_orbital_current_fits_all_transverse_components() {
    fit_orbital_current("svwn5", true);
}

#[test]
fn closed_shell_periodic_spinor_current_fits_gga_and_meta_gga() {
    for name in ["pbe", "tpss"] {
        fit_orbital_current(name, false);
    }
}

fn fit_orbital_current(name: &str, noncollinear: bool) {
    let lattice = Lattice::new(
        Vec3::new(4.4, 0.17, -0.31),
        Vec3::new(0.31, 6.0, 0.26),
        Vec3::new(-0.19, 0.38, 6.5),
    );
    let atoms = [
        (2, Vec3::new(0.17, -0.21, 0.34)),
        (1, Vec3::new(1.63, 0.08, 0.16)),
    ];
    let mut molecule = Molecule::new(
        (0..2)
            .flat_map(|image| {
                atoms.map(|(z, p)| Atom {
                    z,
                    position: p + lattice.a[0] * image as f64,
                })
            })
            .collect(),
    );
    molecule.charge = 2;
    let mut set = BasisSet::new("compact helium hydride");
    for (z, exponent) in [(2, 0.9), (1, 0.72)] {
        set.insert(ElementBasis {
            z,
            shells: vec![Shell {
                l: 0,
                exps: vec![exponent],
                coefs: vec![1.0],
            }],
        });
    }
    let basis = MolBasis::build(&molecule, &set).unwrap();
    let boundary = PeriodicBoundary::FULL_3D;
    let geometry = PeriodicGeometry::new(
        Lattice::new(lattice.a[0] * 2.0, lattice.a[1], lattice.a[2]),
        boundary,
    )
    .unwrap();
    let mut options = DimensionalScfOptions::for_boundary(boundary);
    options.basis_cutoff = 6.0;
    options.grid_cutoff = 8.0;
    options.scf.xc_ao_cutoff = Some(10.0);
    options.scf.grid.level = 2;
    options.scf.grid.prune = false;
    options.scf.grid_shell = 2;
    options.scf.grid.weight_cutoff = 0.0;
    options.electrostatics.compensation = ChargeCompensation::UniformBackground;
    let k_points = geometry.k_points([1; 3]).unwrap();
    let q = geometry.reciprocal()[0];
    let spin = [
        Complex::new(0.05, 0.02),
        Complex::new(-0.01, 0.045),
        Complex::new(0.04, -0.03),
    ]
    .map(|v| if noncollinear { v } else { Complex::ZERO });
    let local_spin = periodic_pauli_operators(
        &basis,
        &geometry,
        &k_points,
        options.basis_cutoff,
        q,
        Complex::ZERO,
        spin,
        0.0,
    )
    .unwrap();
    let functional = resolve(&FunctionalSpec::Combined(name.into())).unwrap();
    let backend = SpinorDftBackend::new(
        &molecule,
        &basis,
        &geometry,
        &functional,
        &options,
        None,
        local_spin,
    )
    .unwrap();
    let config = SpinorScfConfig {
        max_iter: 200,
        energy_tolerance: 2e-14,
        density_tolerance: 2e-13,
        orbital_tolerance: 1e-12,
        ..Default::default()
    };
    let reference = run_spinor_scf(&backend, &config).unwrap();
    assert!(
        reference.converged && reference.fields.operator_converged,
        "reference SCF residual {} and grid number defect {}",
        reference.orbital_residual,
        reference.fields.xc_grid_number_defect
    );
    let density = pauli_components(&reference.density[0]).unwrap();
    let spins: Vec<_> = (0..basis.nao_sph)
        .map(|i| {
            Vec3::new(
                density[1].re[(i, i)],
                density[2].re[(i, i)],
                density[3].re[(i, i)],
            )
        })
        .collect();
    let spin_mixing = spins
        .iter()
        .flat_map(|a| spins.iter().map(move |b| a.cross(*b).norm()))
        .fold(0.0_f64, f64::max);
    if noncollinear {
        assert!(
            spin_mixing > 1e-8,
            "noncollinear AO spin defect {spin_mixing:e}"
        );
    } else {
        assert!(
            spins.iter().all(|s| s.norm() < 1e-12),
            "closed-shell spin density"
        );
    }
    let response = backend
        .orbital_susceptibility_q(
            &reference,
            2,
            &[[1, 0, 0]],
            &ResponseSolverConfig {
                tolerance: 1e-12,
                ..Default::default()
            },
        )
        .unwrap()
        .remove(0);
    let [e0, e1] = response
        .orbital
        .polarization
        .map(|[x, y, z]| Vec3::new(x, y, z));
    let diagonal = (e0 + e1) / 2.0_f64.sqrt();
    let volume = lattice.volume().abs();
    let mut fields = Vec::with_capacity(9);
    for e in [e0, e1, diagonal] {
        for phase in 0..3 {
            let field = StaticPeriodicField {
                wave: if phase == 2 { Vec3::zero() } else { q },
                vector_potential: e.to_array().map(|v| {
                    if phase == 1 {
                        Complex::new(0.0, -0.5 * v)
                    } else {
                        Complex::new(0.5 * v, 0.0)
                    }
                }),
                scalar_potential: Complex::ZERO,
                amplitude: 0.0,
            };
            let operators = field
                .operators(&basis, &geometry, &k_points, options.basis_cutoff)
                .unwrap();
            let first = spin_diagonal(&operators.first_derivative[0]);
            let second = spin_diagonal(&operators.second_derivative[0]);
            let explicit = 0.25 * reference.density[0].trace_product(&second);
            fields.push((first, second, explicit, field));
        }
    }
    let mut residual = reference.orbital_residual;
    let mut cache = BTreeMap::new();
    let mut derivative = |source: usize, h: f64| {
        *cache.entry((source, h.to_bits())).or_insert_with(|| {
            let (first, second, _, _) = &fields[source];
            // The explicit vector-potential shift cancels in the XC kinetic
            // descriptor. Reuse the Coulomb cache while relaxing each physical
            // finite-field Hamiltonian independently.
            let mut external = first.clone();
            external.scale(h);
            external.add_scaled(second, 0.5 * h * h);
            let mut guess = reference.fields.fock[0].clone();
            guess.add_scaled(&external, 1.0);
            let biased =
                BiasedSpinorBackend::new(&backend, Some(vec![external]), Some(vec![guess]))
                    .unwrap();
            let relaxed = run_spinor_scf(&biased, &config).unwrap();
            assert!(
                relaxed.converged && relaxed.fields.operator_converged,
                "field {source} h={h} residual {}",
                relaxed.orbital_residual
            );
            residual = residual.max(relaxed.orbital_residual);
            let mut d = first.clone();
            d.add_scaled(second, h);
            relaxed.density[0].trace_product(&d)
        })
    };
    let mut errors = Vec::new();
    let names: Vec<_> = ["paramagnetic", "uniform", "susceptibility"]
        .into_iter()
        .flat_map(|quantity| {
            ["xx", "yy", "xy"].map(|axis| format!("HeH+ {name} {quantity} {axis}"))
        })
        .collect();
    let flatten = |m: [[f64; 2]; 2]| [m[0][0], m[1][1], m[0][1]];
    let analytic = [
        flatten(response.paramagnetic_transverse),
        flatten(response.uniform_transverse),
        flatten(response.chi_transverse),
    ]
    .concat();
    let mut richardson = 0.0_f64;
    stencils::fit_components_staged(
        &names,
        &[
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Second),
            ([0.04, 0.02, 0.01, 0.005], stencils::Stencil::Second),
            ([0.08, 0.04, 0.02, 0.01], stencils::Stencil::Fourth),
        ],
        |steps, stencil| {
            let data = steps.map(|h| {
                let raw: [f64; 9] = std::array::from_fn(|s| {
                    let (m2, p2) = if stencil == stencils::Stencil::Fourth {
                        (derivative(s, -2.0 * h), derivative(s, 2.0 * h))
                    } else {
                        (0.0, 0.0)
                    };
                    0.25 * stencil.first(derivative(s, h), derivative(s, -h), p2, m2, h)
                        - fields[s].2
                });
                let finite: [f64; 3] = std::array::from_fn(|a| 0.5 * (raw[3 * a] + raw[3 * a + 1]));
                let uniform = [raw[2], raw[5], raw[8]];
                let chi: [f64; 3] = std::array::from_fn(|a| {
                    -4.0 * (finite[a] - 0.5 * uniform[a]) / (volume * q.norm2())
                });
                let mixed = |[a, b, c]: [f64; 3]| [a, b, c - 0.5 * (a + b)];
                [mixed(finite), mixed(uniform), mixed(chi)].concat()
            });
            analytic
                .iter()
                .enumerate()
                .map(|(i, &exact)| {
                    if stencil == stencils::Stencil::Second && steps[0] == 0.08 {
                        richardson =
                            richardson.max(((4.0 * data[3][i] - data[2][i]) / 3.0 - exact).abs());
                    }
                    std::array::from_fn(|j| (data[j][i] - exact).abs())
                })
                .collect()
        },
        &mut errors,
    );
    if functional.family == dft_rs::xc::Family::MetaGga {
        // Independently collocate the actual p+A operator for every source
        // at a finite amplitude and test the cancellation used by the oracle.
        let mut gauge_defect = 0.0_f64;
        let zero = vec![BlochMatrix::zeros(2 * basis.nao_sph, 2 * basis.nao_sph)];
        for (source, (_, _, _, field)) in fields.iter().enumerate() {
            let h = 0.04;
            let mut field = field.clone();
            field.amplitude = h;
            let with_field = SpinorDftBackend::new(
                &molecule,
                &basis,
                &geometry,
                &functional,
                &options,
                Some(&field),
                zero.clone(),
            )
            .unwrap();
            let relaxed = run_spinor_scf(&with_field, &config).unwrap();
            assert!(relaxed.converged && relaxed.fields.operator_converged);
            let operators = field
                .operators(&basis, &geometry, &k_points, options.basis_cutoff)
                .unwrap();
            let explicit = with_field
                .amplitude_derivative(&relaxed.density, &operators.first_derivative, &zero)
                .unwrap();
            gauge_defect = gauge_defect.max((explicit - derivative(source, h)).abs());
        }
        println!("HeH+ {name} explicit covariant momentum defect={gauge_defect:.3e}");
        assert!(gauge_defect < 1e-10);
    }
    println!("HeH+ {name} orbital current analytic={analytic:?} energy={:.16e} Richardson={richardson:.3e} \
        max_orbital_residual={residual:.3e} spin_mixing={spin_mixing:.3e} \
        grid_number={:.3e} coulomb={:.3e} reciprocity={:.3e} imaginary={:.3e} response_residuals={:?} ",
        reference.fields.energy, reference.fields.xc_grid_number_defect, reference.fields.coulomb_operator_defect,
        response.reciprocity_defect, response.imaginary_defect, response.orbital.response_residuals);
    components::assert_no_failures("HeH+ all transverse spinor components", &errors);
    assert!(
        richardson < 2e-7,
        "absolute extrapolation defect {richardson:e}"
    );
}
