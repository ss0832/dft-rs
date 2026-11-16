// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.
use dft_rs as accepted;
#[cfg(test)]
mod physical {
    use super::accepted;
    use accepted::driver::periodic::{prepare_periodic, PeriodicMethod};
    use accepted::driver::{Controls, System, Value};
    use accepted::periodic::optimizer::{optimize_fixed_cell_roks, PeriodicOptOptions};

    fn exercise(system: System, max_steps: usize, require_converged: bool) {
        let controls = Controls::from_value(
            &Value::object()
                .with("xc", "hf")
                .with("basis", "sto-3g")
                .with("dispersion", "none")
                .with("multiplicity", 2)
                .with("charge", if system.numbers.len() == 2 { 1 } else { 0 })
                .with("overlap_cutoff", 6.)
                .with("xc_ao_cutoff", 16.)
                .with("grid_level", 1)
                .with("grid_shell", 1)
                .with("grid_weight_cutoff", 0.)
                .with("conv_tol", 1e-13)
                .with("conv_tol_dp", 1e-11)
                .with("conv_tol_grad", 1e-10)
                .with("max_iter", 300)
                .with("guess", "core")
                .with("damping", 0.)
                .with("no_adiis", true),
        )
        .unwrap();
        let method = PeriodicMethod::from_controls(&controls).unwrap();
        let setup = prepare_periodic(&system, &method).unwrap();
        let basis = accepted::driver::method::build_basis("sto-3g", None).unwrap();
        let options = PeriodicOptOptions {
            max_steps,
            gtol: 1e-5,
            step_max: 0.15,
            ..PeriodicOptOptions::default()
        };
        let result = optimize_fixed_cell_roks(
            setup.cell.molecule(),
            setup.cell.lattice(),
            &basis,
            method.overlap_cutoff,
            &setup.functional,
            &setup.options,
            &options,
        )
        .unwrap();
        assert_eq!(result.trajectory.len(), result.iterations + 1);
        let first = result.trajectory[0].objective;
        assert!(result.objective < first - 1e-7);
        for pair in result.trajectory.windows(2) {
            assert!(pair[1].objective <= pair[0].objective);
        }
        assert!(result.scf.state.orbital_residual < 1e-10);
        assert!(result.scf.state.metric_defect < 1e-12);
        assert!(result.scf.density_change < 1e-11 && result.scf.energy_change < 1e-13);
        let problem = accepted::periodic::roks::GammaRoksProblem::new(
            &result.cell,
            &setup.functional,
            &setup.options,
        )
        .unwrap();
        let fresh = problem.nuclear_gradient(&result.scf).unwrap();
        let maximum = fresh
            .iter()
            .flat_map(|g| [g.x.abs(), g.y.abs(), g.z.abs()])
            .fold(0., f64::max);
        let trajectory = Value::Array(
            result
                .trajectory
                .iter()
                .map(|row| {
                    Value::object()
                        .with("step", row.step)
                        .with("energy", row.objective)
                        .with("max_gradient", row.grad_max)
                })
                .collect(),
        );
        let report = Value::object()
            .with("n_atoms", system.numbers.len())
            .with("n_alpha", result.scf.n_alpha)
            .with("n_beta", result.scf.n_beta)
            .with("initial_energy", first)
            .with("energy", result.objective)
            .with("iterations", result.iterations)
            .with("optimization_converged", result.converged)
            .with("gtol", options.gtol)
            .with("max_gradient", maximum)
            .with(
                "positions",
                result
                    .molecule
                    .atoms
                    .iter()
                    .map(|a| a.position.to_array())
                    .collect::<Vec<_>>(),
            )
            .with(
                "gradient",
                fresh.iter().map(|g| g.to_array()).collect::<Vec<_>>(),
            )
            .with("orbital_residual", result.scf.state.orbital_residual)
            .with("metric_defect", result.scf.state.metric_defect)
            .with("density_change", result.scf.density_change)
            .with("energy_change", result.scf.energy_change)
            .with("trajectory", trajectory);
        println!("ROKS_FIXED_CELL_RESULT {}", report.to_json());
        assert_eq!(result.converged, maximum < options.gtol);
        if require_converged {
            assert!(
                result.converged,
                "H2+ common-orbital geometry did not converge"
            );
        } else {
            assert_eq!(result.iterations, 1);
        }
    }

    #[test]
    fn h2plus_common_orbital_geometry_reaches_stationarity() {
        exercise(
            System {
                numbers: vec![1; 2],
                positions: vec![[-1.5, 0., 0.], [1.5, 0., 0.]],
                cell: Some([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]]),
                pbc: [true; 3],
            },
            30,
            true,
        );
    }

    #[test]
    fn h3_closed_open_common_orbital_armijo_step() {
        exercise(
            System {
                numbers: vec![1; 3],
                positions: vec![[0.13, -0.21, 0.07], [1.91, 0.20, -0.19], [0.4, 1.7, 0.33]],
                cell: Some([[8.5, 0., 0.], [0.31, 9.3, 0.], [0.27, -0.19, 10.1]]),
                pbc: [true; 3],
            },
            1,
            false,
        );
    }
}
