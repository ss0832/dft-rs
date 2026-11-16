// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Portable idealized regression geometries, in Bohr. These are fixed test
//! structures, not fitted equilibrium or experimental reference geometries.
use dft_rs::driver::System;

pub fn phosphine() -> System {
    let radius = 2.27;
    let height = -1.46;
    let mut positions = vec![[0.; 3]];
    for k in 0..3 {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / 3.0;
        positions.push([radius * angle.cos(), radius * angle.sin(), height]);
    }
    System::molecule(vec![15, 1, 1, 1], positions)
}

pub fn ferrocene() -> System {
    let mut numbers = vec![26];
    let mut positions = vec![[0.; 3]];
    for ring in 0..2 {
        // Ring height 3.14 Bohr, written as a quotient so that it is not
        // read as an approximation of pi.
        let height = 314.0 / 100.0;
        let z = if ring == 0 { height } else { -height };
        let offset = ring as f64 * std::f64::consts::PI / 5.0;
        for (element, radius) in [(6, 2.28), (1, 4.32)] {
            for k in 0..5 {
                let angle = offset + 2.0 * std::f64::consts::PI * k as f64 / 5.0;
                numbers.push(element);
                positions.push([radius * angle.cos(), radius * angle.sin(), z]);
            }
        }
    }
    System::molecule(numbers, positions)
}
