<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# dft-rs

Kohn–Sham DFT for molecules and crystals, written from
scratch in Rust.

- **Rust numerical kernels without an external integral/functional runtime
  or BLAS/LAPACK dependency.** The crate is `#![forbid(unsafe_code)]` and
  depends on `libm`, `rayon` and `faer`. Optional `pyo3` behind the `python`
  feature supplies the Python interoperation boundary.
- **Functionals implemented from the primary literature**, not transcribed
  from an existing implementation. External packages (PySCF, …) appear only
  as *numerical oracles*: frozen reference values the test suite must
  reproduce.
- **Explicit capability errors.** Unsupported model selections raise an
  error naming the missing implementation. Source implementation, compiled
  API availability and independent physical validation are separate states;
  [docs/scope.md](docs/scope.md) and the coverage ledgers record their limits.



## Finding an API

The [API guide](docs/api-guide.md) groups calculations by purpose and shows
the same example in Rust, CLI, native Python and ASE. Each interface has a
reference below.

## What it does

| | |
| --- | --- |
| **Methods** | RKS/UKS/ROKS/GKS and their HF counterparts; LDA, GGA, meta-GGA, hybrids, range-separated hybrids, VV10 functionals, separate exchange/correlation selection and weighted XC; typed double-hybrid energy recipes with independent SS/OS PT2 terms and documented response limits |
| **Basis sets** | 78 bundled Gaussian94 sets from the Basis Set Exchange (STO-3G … aug-cc-pV5Z, def2, LANL, Stuttgart, plus two RI-J fitting sets), per-element mixing, ECPs chosen independently of the orbital basis |
| **Derivatives** | analytic gradients and analytic CPKS Hessians, including meta-GGA, ECP and self-consistent VV10 terms; IR intensities from analytic APTs |
| **Excited states** | TDDFT (TDA and full RPA), oscillator and ECD rotatory strengths, CIS excited-state gradients |
| **Response** | analytic static polarizability, Raman activities, uniform external field |
| **Environment** | C-PCM / COSMO / IEF-PCM / SS(V)PE on a smooth cavity, 179 bundled solvents, SMD |
| **Interactions** | SAPT(0), SAPT(0)-S², sSAPT0, uncoupled SAPT(30), KS-FDDS, and regularized HF/KS induction |
| **Dispersion** | D2, D3(zero), D3(BJ), the ωB97X-D tail, VV10/SC-VV10 with a fully analytic Hessian, and energy-only XDM |
| **Periodic** | finite-k HF and KS-DFT, forces, stress, elastic constants, phonons, dielectric and piezoelectric response in [supported dimensional and reference sectors](docs/development-status.md#periodic-properties) |
| **Analysis** | Mulliken/Löwdin/NPA charges, Wiberg indices, Molden output, geometry optimization and P-RFO transition-state search |


## Install and run

Rust source builds require Rust 1.85 or later for the locked dependency graph.

### Command line

```bash
cargo build --release
```

```bash
./target/release/dft-rs energy water.xyz --xc b3lyp --basis def2-svp
```

```bash
./target/release/dft-rs frequencies water.xyz --xc m06-2x   # analytic CPKS Hessian
./target/release/dft-rs optimize water.xyz --xc pbe --disp d3bj --traj path.xyz
./target/release/dft-rs tddft water.xyz --xc b3lyp --nstates 5 --tddft-method rpa
./target/release/dft-rs energy water.xyz --xc b3lyp --solvent smd:water
./target/release/dft-rs sapt dimer.xyz --natoms-a 3 --xc pbe0
./target/release/dft-rs periodic chain.xyz --cell "2.9,8,8" --k-mesh 8,1,1 --guess sad --forces
./target/release/dft-rs periodic crystal.xyz --cell 8.0 --xc pbe --relax --traj crystal-opt.xyz
./target/release/fit_substituent --synthetic-check
```

Full reference: [docs/cli.md](docs/cli.md).

`dft-rs` is the canonical executable. `dft_rs_cli` remains a compatibility
alias; both names use the same implementation. `cargo run` selects `dft-rs`.

For native artifacts intended for distribution, use the standard-library
Python helper to remap private build roots and audit the resulting binaries:

```console
python tools/portable_build.py build --release --locked --lib --bins
python tools/portable_build.py build --release --locked --features python --lib --bins
```

The helper derives paths on the build computer and uses Rust's
[source-path remapping](https://doc.rust-lang.org/rustc/remap-source-paths.html).
It preserves compiler flags and checks the artifacts for private crate,
Cargo, Rustup and home roots. It never records those paths in source.
Bundled basis and model data are embedded; explicit external inputs remain
caller-supplied paths. Native artifacts target the selected operating system
and architecture, and must be built for each supported platform. Source
distributions contain no local histories, validation logs, reference inputs,
build trees or Python environments. Author name and email metadata are retained.

### Rust

```toml
[dependencies]
dft-rs = { version = "0.4.0", path = "../dft-rs" }
```

```rust
use dft_rs::driver::{run, Controls, System, Value};

fn main() -> dft_rs::Result<()> {
    let system = System::molecule(vec![1, 1], vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]);
    let controls = Controls::from_value(&Value::object()
        .with("xc", "hf").with("basis", "sto-3g")
        .with("cpu_threads", 1usize).with("memory_mb", 128usize)
        .with("log_file", "calculation.log"))?;
    let report = run("energy", &system, &controls)?;
    println!("{:?}", report.get("energy").and_then(Value::as_f64));
    Ok(())
}
```

Full reference: [docs/rust-api.md](docs/rust-api.md).

### Python / ASE

The Python distribution is named `dft-rs-python`; its import is `dft_rs` and
its console command is `dft-rs`. A wheel includes the compiled core and
needs no separate Rust executable. Once a release is published, install it
in the selected Python environment with:

```console
python -m pip install dft-rs-python
dft-rs --version
dft-rs energy molecule.xyz --xc pbe --format json
```

From a checkout, `python -m pip install .` builds and installs both the
Python package and this command. Building from source requires Rust.
For an unpublished local wheel in `dist/`, install without an index using
`python -m pip install --no-index --find-links dist dft-rs-python`.
Local packaging validation does not publish a release or reserve an index name.

The wheel installs the documentation and coverage ledger in `dft_rs/docs`.
The Python source archive includes the same files in its top-level `docs/`
directory. This command prints the installed documentation path.

```console
python -c "from importlib.resources import files; print(files('dft_rs').joinpath('docs'))"
```

```console
python -m pip install -e .
python -m pip wheel --no-deps . --wheel-dir dist
```

```python
from dft_rs.native import calculate

r = calculate([1, 1], [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
              basis="sto-3g", xc="hf", gradient=True,
              cpu_threads=1, memory_mb=128, log_file="native.log")
r["energy"], r["gradient"]        # Hartree, Hartree/Bohr
```

```python
from ase.build import molecule
from dft_rs import DftRs                     # published extra: "dft-rs-python[ase]"

atoms = molecule("H2O")
atoms.calc = DftRs(basis="def2-svp", xc="pbe0")
atoms.get_forces()                            # eV/Å, analytic
```

Full reference: [docs/python-api.md](docs/python-api.md).

## Units

Rust and native Python use Bohr for Cartesian inputs and Hartree for
energies. Reports carry per-entry units: frequencies, hyperfine couplings,
g tensors and other observables retain their documented conventional units.
XYZ coordinates and the CLI `--cell` input are Angstrom. ASE converts labelled
energy/length quantities to eV/Angstrom and preserves other unit labels.
Inspect `report["units"]` or `Report.unit(key)` for the requested quantity.

## Versioning

While the version is `0.x`, **breaking changes are allowed in any release**
and are recorded in [CHANGELOG.md](CHANGELOG.md). The one stability promise
made now is that `DftError` is `#[non_exhaustive]`: naming a new refusal is
not a breaking change, so `match` on it must carry a `_` arm. That matters
because this project prefers to add a *named* refusal over returning a
plausible number, so new variants are expected rather than exceptional.

## Documentation

| Document | Contents |
| --- | --- |
| [Implementation status](docs/development-status.md) | Current model coverage, representative results and open checks |
| [API guide](docs/api-guide.md) | Calculations by purpose, interface examples and shared controls |
| [Validation matrix](docs/validation-matrix.md) | Functional/reference, ECP and target-specific measured acceptance |
| [Molecular benchmarks](docs/molecular-property-benchmarks.md) | Physical comparison cases, computed values and retained failures |
| [Performance](docs/performance.md) | Paired benchmarking procedure and measured kernel costs |
| [Excited-state response](docs/excited-state-extension.md) | Spin sectors, nuclear derivatives and correlated TDA acceptance |
| [Scope](docs/scope.md) | Model contracts, exclusions and interface coverage |
| [Response methods](docs/response.md) | Response theory, derivative conventions and component-level evidence |
| [Periodic methods](docs/periodic.md) | Dimensional electrostatics, SCF, stress, phonons and field response |
| [Functional catalogue](docs/FUNCTIONALS.md) | Provider names, derivative capabilities and functional-level checks |
| [Theory](docs/THEORY.md) | Method derivations and historical implementation records |
| [CLI](docs/cli.md#table-2-cli), [Rust](docs/rust-api.md#table-1-rust), [Python and ASE](docs/python-api.md#table-3-native-python-and-ase) | Aligned calculation tables, signatures, controls and report units |
| [Response ledger](docs/response.md#response-coverage-ledger), [target matrix](docs/property_targets.csv) | Physical coverage, reference/provider combinations and validation status |
| [References](docs/references.md), [conventions](docs/conventions.md) | Primary literature and project rules for implementation and validation |

## Verification

The [implementation conventions](docs/conventions.md#derivative-validation)
describe the derivative fits and oracle policy. Frozen oracles live in
`tests/oracle_data/`. The periodic suite also checks the rocksalt Madelung
constant to ten digits, Ewald split independence, band folding, the neutral
molecular limit and supercell energy extensivity.

```bash
cargo test --release
cargo test --release -- --ignored
```

The ignored tests include long periodic physical checks. Use the
[release verification procedure](docs/cli.md#source-archive-verification)
for a frozen-source build and package audit. Current coverage and incomplete
scientific gates are listed in the [implementation status](docs/development-status.md).

This code was written and verified using generative AI.
