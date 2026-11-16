<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Third-party notices

dft-rs is Apache-2.0 and contains no third-party source code. This file
indexes the provenance records for bundled *data* and cites the algorithmic
literature. It is updated at every phase close as part of the license audit.

## Rust crate dependencies

| Crate | License | Role |
|---|---|---|
| faer | MIT | dense linear algebra (pure Rust) |
| rayon | MIT OR Apache-2.0 | data parallelism |
| libm | MIT OR Apache-2.0 | portable special functions |
| pyo3 (optional) | MIT OR Apache-2.0 | Python bindings (`python` feature) |

## Bundled data (added per phase)

- `third_party/bse/NOTICE.md` — Basis Set Exchange data, including the historical
  downloads and thirty additional version-pinned sets. The upstream BSD 3-Clause
  notice, exact URLs and payload/catalogue hashes accompany the expansion.
- D2/D3 parameter tables carry their provenance directly in the headers of
  `src/dispersion/data/*.csv`; the D3 tables are published numerical data
  routed through Apache-2.0 MOPAC v23.2.5.
- `third_party/solvents/NOTICE.md` — solvent descriptors of the Minnesota
  Solvent Descriptor Database (`src/data/solvents.csv`), from the
  continuum-solvation work.

## Skala (learned functional)

`src/skala.rs` evaluates models of the Skala exchange-correlation functional
(Microsoft Research, MIT License). The architecture is implemented from the
published description of the model; no source code and no trained
parameters of that project are part of this package. A model file is
produced by the user with `tools/convert_skala_model.py`, which reads an
official checkpoint through the official Python packages.

## Reference-implementation policy

libxc, libint, dftd3/dftd4 and similar packages are referenced **only through
their published papers**; no source code or embedded data is copied or
mechanically translated. PySCF and other programs serve exclusively as
numerical oracles: frozen reference values that the test suite reproduces.

The one deliberate exception is **published measured/derived physical
constants**, which have no implementation-specific content: the D3 parameter
tables were taken via MOPAC (Apache-2.0) and the solvent descriptors via
PySCF (Apache-2.0). In both cases the primary publication is cited, the
route is recorded in the file's PROVENANCE header and in the NOTICE, and the
licenses are compatible with Apache-2.0.
