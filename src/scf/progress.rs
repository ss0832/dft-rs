// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Relative progress of a physical SCF residual, independent of its accuracy gate.

#[derive(Default)]
pub(super) struct ResidualProgress {
    reference: Option<f64>,
    best: f64,
    steps: usize,
}

impl ResidualProgress {
    /// Allow cumulative improvement over a complete history-sized window.
    /// Updating the reference on every small decrease would incorrectly
    /// classify steady geometric convergence as a permanent plateau.
    pub(super) fn has_stalled(&mut self, residual: f64, window: usize) -> bool {
        let Some(reference) = self.reference else {
            self.reference = Some(residual);
            self.best = residual;
            return false;
        };
        self.steps += 1;
        self.best = self.best.min(residual);
        if self.best < 0.7 * reference {
            self.reference = Some(self.best);
            self.steps = 0;
        }
        self.steps >= window.max(2)
    }
}

#[cfg(test)]
mod tests {
    use super::ResidualProgress;

    #[test]
    fn cumulative_geometric_convergence_is_not_a_plateau() {
        let mut monitor = ResidualProgress::default();
        for iteration in 0..200 {
            assert!(!monitor.has_stalled(0.95_f64.powi(iteration), 8));
        }
    }

    #[test]
    fn cumulative_physical_residual_reduction_does_not_start_globalization() {
        // A retained physical trajectory improves by about eighty percent
        // over six steps, although no individual reduction reaches thirty
        // percent. It must remain eligible for the local iteration.
        let residuals = [
            4.845e-4, 4.514e-4, 3.170e-4, 2.418e-4, 1.701e-4, 1.265e-4, 9.836e-5,
        ];
        let mut monitor = ResidualProgress::default();
        for residual in residuals {
            assert!(!monitor.has_stalled(residual, 6));
        }
    }

    #[test]
    fn oscillatory_plateau_recovery_is_independent_of_residual_units() {
        let residuals = [1.0, 0.9, 0.95, 0.86, 0.92, 0.89, 0.91, 0.9, 0.92];
        for scale in [1e-200, 1.0, 1e200] {
            let mut monitor = ResidualProgress::default();
            let decisions: Vec<_> = residuals
                .iter()
                .map(|r| monitor.has_stalled(r * scale, 8))
                .collect();
            assert_eq!(
                decisions,
                [false, false, false, false, false, false, false, false, true]
            );
        }
    }
}
