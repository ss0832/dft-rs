// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Initial-guess Fock matrices for the SCF.
//!
//! - `Core`: bare H_core (poor but assumption-free).
//! - `Gwh`: generalized Wolfsberg–Helmholz,
//!   `F_μν = ½·K·(H_μμ + H_νν)·S_μν` with K = 1.75 (diagonal untouched) —
//!   a much better starting spectrum for most molecules.
//! - `Sad`: superposition of spherically averaged atomic densities, built
//!   on the fly in the molecule's own basis (see [`super::sad`]). The
//!   default, because it starts with the right electron count in the right
//!   shells and so removes several SCF iterations.
//!
//! `Sad` is a DENSITY guess, so it does not go through [`initial_fock`];
//! the driver builds its Fock from the superposed density instead.

use crate::linalg::Matrix;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Guess {
    Core,
    Gwh,
    #[default]
    Sad,
}

/// One-electron guess Fock. `Sad` falls back to `Gwh` here because it is a
/// density guess handled by the driver; this keeps the function total.
pub fn initial_fock(guess: Guess, h_core: &Matrix, s: &Matrix) -> Matrix {
    match guess {
        Guess::Core => h_core.clone(),
        Guess::Gwh | Guess::Sad => {
            let n = h_core.rows;
            let mut f = Matrix::zeros(n, n);
            const K: f64 = 1.75;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        f[(i, j)] = h_core[(i, i)];
                    } else {
                        f[(i, j)] = 0.5 * K * (h_core[(i, i)] + h_core[(j, j)]) * s[(i, j)];
                    }
                }
            }
            f
        }
    }
}
