// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

/// Cubic, tetragonal, orthorhombic, hexagonal, rhombohedral, monoclinic
/// and triclinic magnetic cells in Bohr. Keep the original oracle metrics.
pub fn rows() -> [[[f64; 3]; 3]; 7] {
    [
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        [[5.0, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 6.0]],
        [
            [5.0, 0.0, 0.0],
            [-2.5, 4.330127018922193, 0.0],
            [0.0, 0.0, 6.0],
        ],
        [
            [5.0, 0.0, 0.0],
            [1.0, 4.898979485566356, 0.0],
            [1.0, 0.816496580927726, 4.830_458_915_396_48],
        ],
        [[5.0, 0.0, 0.0], [0.0, 5.5, 0.0], [1.2, 0.0, 6.0]],
        [[5.0, 0.0, 0.0], [0.7, 5.5, 0.0], [0.5, 0.8, 6.0]],
    ]
}
