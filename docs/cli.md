<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Command-line interface

Start with the [API guide](api-guide.md) to choose a calculation, compare
interfaces, run an example and interpret units and convergence. This manual
contains the detailed contracts and reference-specific limits.

[Common commands](#common-driver-commands) · [Options](#options) · [Calculation table](#table-2-cli) · [Exit behaviour](#exit-behaviour)

The complete shared contracts are in the existing Rust manual: [requests,
controls and reports](rust-api.md#common-calculation-driver), [result
conventions](rust-api.md#result-conventions-and-physical-scope), [analytic
foundations](rust-api.md#analytic-foundation-apis), [stability and following](rust-api.md#orbital-stability-and-following-conventions),
and the subsequent magnetic, superconducting, Dirac, thermodynamic and
finite-field nuclear API sections. The [Rust calculation table](rust-api.md#table-1-rust),
[CLI table](cli.md#table-2-cli) and [native Python/ASE table](python-api.md#table-3-native-python-and-ase)
retain identical registry IDs and every earlier entry. The surface manuals
now hold all calculation descriptions, arguments, examples, units, model
limits and raw residual conventions.


The canonical executable is `dft-rs` (`dft-rs.exe` on Windows).
`dft_rs_cli` remains a compatibility alias built from the same command
implementation. `cargo run` selects the canonical executable by default.

Both CLI names accept global `--cpu-threads N` and `--memory-mb M`, including
legacy commands. For example, `dft-rs energy water.xyz --cpu-threads 4
--memory-mb 512 --format json`. Values must be positive integers; duplicate,
missing, fractional and overflowing values fail. Omission preserves existing
defaults. The options are extracted before command dispatch, so they do not
depend on selecting JSON output. `memory-mb` means MiB and limits managed
storage/workspace accounting, not whole-process RSS; see the
[resource contract](rust-api.md#errors-and-threading). `--incore-mb` remains
the existing component-specific ERI preference and cannot enlarge the shared
managed budget.

Response calculations with numerical XC integration accept `--grid-level
0..9` (default 3). Common-driver routes additionally expose
`--grid-preset fine|ultrafine|superfine`, `--grid-angular-scheme
lebedev|product` and `--grid-weight-cutoff` (default 1e-14 Bohr^3).
Molecular VV10 has independent `--vv10-grid-level 0..9` (default 3),
including legacy response commands on both CLI names. PBC k/q meshes,
open-tail cutoffs and magnetic tensor-product grids are different controls.
HF analytic integrals need no XC grid, while the Ramsey diamagnetic
expectation still uses the requested quadrature. See the
[quadrature contract](conventions.md#quadrature-grids) for all grid families,
defaults and explicit exceptions; a finer grid does not enable an unsupported
physical method.

Both CLI names also accept `--log-file PATH`, `--log-level LEVEL` and
`--log-mode console|file|both|none` before dispatch. A file alone selects
file logging; `both` retains stderr progress while saving it. Levels are
`error`, `warn`, `info` (default) and `debug`. JSON/text results on stdout
are unchanged. For example:

```console
dft-rs energy molecule.xyz --xc hf --format json --cpu-threads 2 --log-file calculation.log --log-mode both
```

The shared driver defaults to stderr without a file. The traditional CLI
commands retain stderr plus an input-stem log: `molecule.xyz` produces
`molecule.dft.log`.
Explicit logging options have the same semantics on both CLI aliases and
in the Python/Rust APIs. Files append run identities and flush every record;
failures leave partial records and an I/O failure yields a nonzero exit.
Use separate paths for concurrent calculations. See the
[shared logging contract](rust-api.md#persistent-calculation-logs).

The standalone `fit_substituent` fitter accepts the same `--cpu-threads` and
`--memory-mb` options, including for `--synthetic-check`.

Installing the `dft-rs-python` Python distribution also installs the `dft-rs` command
into that Python environment's scripts directory (`Scripts` on Windows,
`bin` on Unix). Activate the environment or add its scripts directory to
PATH. The installed command calls the same compiled Rust CLI through the
Python extension; it needs neither Cargo nor a separately installed binary.
Once a release is published, the index installation command is:

```console
python -m pip install dft-rs-python
dft-rs --version
dft-rs energy molecule.xyz --xc pbe --format json
```

For a local checkout, `python -m pip install .` builds and installs the
package and its command (a Rust toolchain is needed to build from source).
For an unpublished local wheel in `dist/`, use
`python -m pip install --no-index --find-links dist dft-rs-python`.
Local validation does not publish or register the distribution on an index.
The command preserves the Rust CLI's standard output, error output and
exit status. `dft_rs.native.cli_main(argv)` remains available for explicit
argument lists; omitting `argv` uses the process command line.

The [aligned response API tables](cli.md#table-2-cli) document the additional
`stability`, `stab-following`, `dynamic-polarizability`, `dynamic-beta`, `dynamic-gamma`, `field-taylor`, `field-mixed`, `beta`, `gamma`, `periodic-response`,
`nuclear-taylor`, `nuclear-mixed`, `molecular-taylor`, `molecular-mixed`,
`static-spectroscopy`, `force-constants`, `dielectric`, and `thermal-response` commands,
their flags, units, report diagnostics, and scope.

`dft-rs` is the third surface onto the same library, alongside
[the Rust API](rust-api.md) and [the Python API](python-api.md). It has no
dependencies of its own — the argument parser is in-tree — so the binary that
`cargo build --release` produces is the whole program.

```console
cargo build --release
./target/release/dft-rs --help
```

The release profile uses fat link-time optimization with one code
generation unit, so each final artifact optimizes the whole library in one
compiler process. Measured on Linux for v0.3.1 (peak resident memory of the
largest process): building the Python extension with
`python -m pip install .` peaks at 4.6 GiB in about 11 minutes;
`cargo build --release -j 1`, which builds the three executables one at a
time, peaks at 7.5 GiB in about 19 minutes. Without `-j 1` Cargo optimizes
the executables at the same time and needs more. The executables therefore
do not build reliably with this profile on a machine with 8 GB; reducing
that peak is planned. Until then the profile can be overridden without
editing the manifest (thin link-time optimization peaks at 3.6 GiB for the
Python extension; the executables were not measured with it):

```console
CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -j 1
```

Geometry files are XYZ, with Angstrom coordinates. CLI `--cell` lengths
are also Angstrom. Native energy/length results use Hartree/Bohr, with
per-field exceptions such as MHz hyperfine couplings, dimensionless g tensors,
ppm shifts and cm^-1 frequencies. JSON reports supply per-entry unit labels;
text summaries can additionally print eV or kcal/mol. Use those labels rather
than applying one conversion to an entire report.

```text
dft-rs <command> file.xyz [options]
```

---

## Common-driver commands

For molecular RKS, UKS, ROKS and GKS energy, gradient, Hessian, electric
moment and orbital reports,
`--xc-terms` takes a JSON array of objects. For example, pass
`[{"xc":"pbe0","weight":0.8},{"xc":"hf","weight":0.2}]` as one argument.
Use `--format json` for the report and its units.
The [weighted XC contract](rust-api.md#weighted-molecular-xc) defines
the model, scope and diagnostics.
Use `--xc pbe0` for the named global hybrid and
`molecular-multipole --max-rank 2` to include the electric quadrupole.
The same JSON argument is available for fully periodic `periodic-energy`
and `stress`. Use `--reference rks|uks|roks|gks` with the corresponding
reference controls and `--forces --stress` on `periodic-energy` to
request both derivatives. See the [periodic contract](rust-api.md#weighted-periodic-xc).

Calculations registered with the [common driver](rust-api.md#common-calculation-driver) take
`dft-rs <feature> structure.xyz [--control value ...]`, with `--format text`
(default) or `--format json` and `--save report.json`. Controls use the
driver names (`--grid-level`, `--electronic-temperature`,
`--response-tolerance`, ...); a control that the calculation does not read is
refused by name. `energy`, `gradient`, `charges`, `hessian`, `frequencies`
and `polarizability` keep their earlier text output unless `--format` or
`--save` is given, which routes them through the driver. Explicit reference
or occupation selectors also choose the common driver. Double-hybrid
controls/names select it for energy, gradient, Hessian, polarizability and
harmonic thermochemistry; harmonic projection or electronic-degeneracy
controls select it for frequencies/Gibbs reports. Output format alone is
therefore not the complete routing rule. `dipole`,
`orbitals`, `free-energy`, `molecular-multipole`, `multipole-gradient`,
`ir-raman`, `conceptual-dft`, `spin-conceptual-dft` and `feature-registry`
always use the driver. See [rust-api.md#common-calculation-driver](rust-api.md#common-calculation-driver) for every command.

Unavailable commands and unsupported feature/reference/control combinations
fail with a `not implemented` diagnostic on stderr and nonzero exit status.
Unknown control names, invalid types/shapes and missing required inputs remain
input errors; convergence and resource failures keep their own diagnostics.
Requested reference selectors are never silently replaced by the default.
Dedicated registry entries retain their documented commands even when the
generic Rust/Python driver has no adapter for them. Both CLI executable names
use this error contract and preserve machine-readable stdout on failure.

Registry entries marked `dedicated` retain their explicit command parser;
they are not automatically common-driver commands. Listing a command or a
reference does not establish every functional/reference/PBC combination.
Energy nuclear Hessians and EPR-property nuclear Hessians are separate
capabilities; see the property's model and validation limits.

Continuous-species commands must use the correct model mode and saved
record. All-electron fractional charge is different from frozen-valence
substituent replacement; the 51 fixed-operator coefficients are not the
three optimized coordinates in the four-candidate experiment. See the
[model and energy-bookkeeping contract](rust-api.md#continuous-chemical-species-model-contracts)
before interpreting energy differences or derivative reports. No named
chemical substituent is activated by a successful request-parser check.

The existing `continuous-species` command accepts
`--fragment-mode rigid-evaluate --derivative-order 0/1/2` for an explicit
`rigid-static-core-v1` primitive recipe. Supply `--fragment-model-json` as one
native-unit JSON argument, not a model filename; XYZ geometry remains in
Angstrom and record Euler angles remain radians. Order defaults to one.
Energy/gradient and capability-error checks pass on the canonical executable
and installed `dft-rs` Python console script. The package does not advertise
`python -m dft_rs` as a command. The compiled full-Hessian route remains
experimental: its final concurrent and native/ASE runtime checks were
uncompleted when continuous-species work was stopped. Two backend ordered
force-difference checks also remain unresolved and are retained alongside
their independent energy witnesses. Existing `few-evaluate` forces and
clamped parameter derivatives keep their own scope. See the linked model
contract before assigning units to mixed center/angle blocks or interpreting
these results as chemical validation.

## Commands

| Command | Produces |
| --- | --- |
| `info` | electron count, nuclear repulsion, geometry in Bohr |
| `integrals` | AO counts and S/T/V diagnostics — no SCF |
| `energy` | single-point SCF energy and its components Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider).  [ROKS doubles-only PT2 energy and residuals](#restricted-open-shell-double-hybrid-energy). |
| `charges` | Mulliken/Löwdin charges and the dipole moment  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). |
| `gradient` | energy + analytic nuclear gradient Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). The ordinary molecular report also returns `forces` as the negative Cartesian gradient in Hartree/Bohr. |
| `optimize` | L-BFGS geometry optimization |
| `saddle` | P-RFO or Householder RS-I-RFO transition-state search |
| `frequencies` | harmonic frequencies (Eckart-projected) + IR intensities |
| `hessian` | the Cartesian Hessian itself  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `polarizability` | static polarizability tensor from analytic CPKS  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). |
| `dynamic-polarizability` | complex frequency-dependent polarizability and response diagnostics |
| `stability` | channel-resolved real/complex orbital stability, eigenvalues and diagnostics |
| `stab-following` | internal negative-mode descent, re-converged state and final stability |
| `dynamic-beta` | all 27 complex beta components at two incoming frequency/damping pairs |
| `dynamic-gamma` | all 81 complex gamma components at three incoming frequency/damping pairs |
| `field-taylor` | analytic static dipole Taylor coefficients through the requested order; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `field-mixed` | analytic static mixed dipole derivative and accumulation diagnostics; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `beta` | all 27 raw static first-hyperpolarizability components, contraction vector, interchange diagnostics; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The third-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^3*Bohr^3/Hartree^2, with raw response and pair residuals. |
| `gamma` | all 81 raw static second-hyperpolarizability components, isotropic contraction, interchange diagnostics; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The fourth-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^4*Bohr^4/Hartree^3, with raw response and pair residuals. |
| `nuclear-taylor` | analytic nuclear energy Taylor coefficients, including cubic/quartic force constants |
| `nuclear-mixed` | analytic mixed nuclear energy derivative and accumulation diagnostics |
| `anharmonic` | selectable strict VPT2 or retained-polyad anharmonic bands, IR and optional static Raman; [controls and units](rust-api.md#common-calculation-driver) |
| `raman` | frequencies + IR + Raman activities |
| `tddft` | excitation energies, oscillator and rotatory strengths |
| `periodic` | finite-k periodic HF/KS-DFT energy and analytic forces |
| `periodic-response` | screened finite-q Fourier density from a unit plane-wave source |
| `dielectric` | selected inverse dielectric matrix and screened susceptibility |
| `thermal-response` | uniform static self-consistent temperature, number, and reservoir response |
| `sapt` | SAPT interaction-energy decomposition |

Two informational flags take no geometry and exit immediately:
`--list-functionals` prints the whole functional registry with descriptions,
`--list-solvents` the 179 bundled solvents with their dielectric constants.
`--version` prints the version.

`--xc blyp35` selects 0.35 full-range HF exchange + 0.65 B88 exchange +
full LYP correlation, without an implicit dispersion/nonlocal correction.
It uses the common energy/gradient driver (Hartree and Hartree/Bohr), and
appears in `--list-functionals`. Numerical acceptance and remaining response
scope are recorded in [FUNCTIONALS.md](FUNCTIONALS.md).

`--xc wc04` and `--xc wp04` follow the same route, using the explicit
VWN-RPA convention and independent fitted coefficients documented in
FUNCTIONALS.md. No implicit solvent or dispersion correction is added.

| Provider | Common features | Model and units | Raw acceptance measurements |
|---|---|---|---|
| BLYP35 | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature, no added corrections; Hartree and Hartree/Bohr | Water/methyl SCF error <=1.720e-12 Hartree; full five-molecule catalogue maximum error 9.778e-12 Hartree; 12 C1 gradient fits p=1.999584..2.001368, minimum R2=0.99999988; native/ASE/CLI maximum gradient difference 4.441e-15 Hartree/Bohr, energy difference 4.263e-14 Hartree |
| WC04 / WP04 | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature, no added corrections; explicit VWN-RPA convention; Hartree and Hartree/Bohr | Water/methyl maximum SCF errors 2.871e-12 / 2.501e-12 Hartree; all 12 C1 gradient entries per model fitted; maximum public gradient differences 1.999e-15 / 6.218e-15 Hartree/Bohr and energy differences 8.527e-14 / 7.106e-14 Hartree; WP04 five-molecule maximum error 6.435e-11 Hartree; WC04 ferrocene remains unaccepted after an interrupted nonstationary run |
| PKZB | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS; integer populations, zero temperature; original seven-variable meta-GGA, half-weighted tau, FHC input convention; Hartree and Hartree/Bohr | Maximum water/methyl SCF error 3.567e-12 Hartree; 53 local partial and 12 C1 RKS gradient fits pass; public gradient/energy differences 1.582e-15 Hartree/Bohr and 2.842e-14 Hartree; five-molecule and higher response acceptance remain pending; standard H2/LiH/water/phosphine pass, maximum error 6.196e-12 Hartree; ferrocene reference reached 180-second budget; Rust comparison unrun; 104 smooth local third partial fits pass (p300), no molecular higher-order inference |
| TPSSloc | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature; localized TPSS correlation with TPSS exchange, half-weighted tau; Hartree and Hartree/Bohr | 53 local partial and 12 C1 gradient fits pass; native/ASE/CLI maximum energy/gradient differences 5.685e-14 Hartree and 2.929e-15 Hartree/Bohr; all six p301 tests and both public cases pass after explicitly approved four-entry physical inward reference correction; original raw values, failed logs and thresholds retained; five-molecule and higher response acceptance pending |
| DSD-PBEB95-D3(BJ) | `energy`, `gradient` | Original 2013 typed global meta-GGA double hybrid; .66 HF/.34 PBE-X/.55 standard B95-C/.09 SS/.46 OS; automatic pairwise D3(BJ)(.61,0,0,6.2); all-electron spherical STO-3G RKS/UKS, integer populations; Hartree and Hartree/Bohr | p304: 46 local first/second/third partial fits, 72 structural zeros and 55 molecular term comparisons pass; all 24 independent C1 RKS/UKS gradient fits pass; both RKS/UKS public-route exact-node runs terminate successfully; earlier aggregate timeout trials retained in FUNCTIONALS.md; molecular Hessian/ECP/ROKS/GKS/excited combinations pending |
| Original omegaB2PLYP / omegaB2GP-PLYP (bounded validation) | `energy`, `gradient` | Typed range-separated GGA double hybrids; SR/full-LR exact exchange .53/1 and .65/1; omega .30/.27 Bohr^-1; ITYH SR-B88, LYP and full-Coulomb PT2; no implicit dispersion; Hartree and Hartree/Bohr | p305: 66 local derivative fits, 44 structural zeros, spin/range limits, 110 molecular term comparisons and all 48 C1 RKS/UKS gradients pass; both public energy cases and all four native/common/ASE/CLI gradient cases pass; all point kernels pass after the explicitly approved eight-entry physical-limit correction; original raw values/logs retained; standard five-molecule acceptance pending; no general excited-state acceptance |
| `hsesol`, `wb97x-2-lp`, `wb97x-2-tqz`, `xdh-pbe0` | `energy` | All-electron RKS H₂/STO-3G at 1.4 Bohr. Grid level 9, zero grid weight cutoff. Double hybrids use full-Coulomb PT2. Total energies and raw orbital residuals in Hartree. | Independent four-model totals and double-hybrid spin terms pass. Maximum total error 2.834e-9 Hartree and raw orbital residual 1.301e-16 Hartree. All three PT2 amplitude residuals are zero. [Definitions and energy values](FUNCTIONALS.md#screened-and-xdh-energy-checks) |
| TASK / mTASK | `energy`, `gradient` | All-electron spherical STO-3G RKS/UKS, integer populations, zero temperature; published exchange plus original PW92 correlation, half-weighted tau, FHC input convention; Hartree and Hartree/Bohr | 107 local derivative/parameter fits and 24 C1 RKS gradient fits pass; maximum SCF error 2.416e-13 Hartree; four native/ASE/CLI cases pass, maximum energy/gradient route differences 5.684e-14 Hartree and 2.665e-15 Hartree/Bohr; full local fifth tests pass 84 exchange plus 6 shared correlation entries at one smooth point, retaining 21 failed exploratory measurements; dense cap remains 3; five-molecule, independent UKS nuclear, ECP/ROKS/GKS and excited-state Hessian combinations remain separate |
| EDF2 (bounded validation) | `energy`, `gradient` | Global GGA hybrid, published independent weights and0.1695 HF; all-electron spherical STO-3G RKS/UKS, atomic units | New library/CLI build and portable audit pass;48 local fits, RKS/UKS terms and2 native/ASE/CLI cases pass; all12 C1 RKS gradients and78 independent common-driver Hessian fits pass; local component fifth74fits/442zeros pass; all point kernels pass after the explicitly approved six-entry physical-limit correction; original raw values/logs retained; all five standard molecular energies pass including Fe40 iterations/error1.751e-11 Hartree at unchanged gates; public Hessian/ECP/ROKS/GKS/excited combinations remain separate |
| N12-SX (bounded validation) | `energy`, `gradient`, `hessian` | Screened hybrid GGA, whole fitted local polynomial; HF0.25 short range/zero long range, omega0.11; spherical STO-3G, atomic units | Actual2a5b3ad9 Python/lib/CLI build and portable audit pass; independent full local kernels,86 first-third fits, fourth45fits/95zeros, fifth68fits/184zeros and RKS/UKS term cases pass; all12 C1 RKS gradients and78 independent Hessian fits pass; all4 RKS/UKS native/common/ASE/CLI gradient/Hessian route cases pass; four standard molecular energies pass, Fe reference active; ECP/ROKS/GKS/excited and general mixed-provider combinations remain separate |
| dlDF / B98 / tau-HCTH / hybrid tau-HCTH (bounded validation) | `energy`, `gradient`, `hessian` | dlDF global HF0.6144129 with fitted meta-GGA X/PW92-MOD C and no added dispersion; B98 fit2c HF0.219847 with original-PW92 B97 form; tau-HCTH pure/hybrid HF0/0.15 with shared kinetic exchange and independently fitted correlation | Actual84bd1e14 Python/library/CLI warning-strict build and portable audit pass; independent component points,432 first-third local fits and eight RKS/UKS term cases pass; complete interior fifth380 nonzero fits/1846 zeros and weighted composition checks pass; all270 independent C1 RKS gradient/Hessian fits for dlDF/tau-HCTH pair and all8 RKS/UKS native/common/ASE/CLI gradient cases pass; standard16/20 accepted with4 Fe cases in finite queues; no separate B98 nuclear-Hessian or universal UKS Hessian/excited/ECP/ROKS/GKS claim |
| RSX-QIDH / RSX-0DH (bounded validation) | `energy`, `gradient` | Range-separated global-LR double hybrids; QIDH SR-HF3^(-1/3), omega0.27, PT2=1/3; 0DH SR-HF0.5, omega0.33, PT2=0.125; full LR HF and full-Coulomb same/opposite-spin PT2; no dispersion | Actual8e512c3f library/CLI audit passes;110 local first-third fits,110 independent molecular terms and two public energy cases pass; all48 C1 RKS/UKS gradient fits pass, with one retained failed coarse interval resolved at unchanged gates; all four public-gradient cases pass; ferrocene/Hessian/excited/ECP/ROKS/GKS combinations remain pending |
| PBE1KCIS / mPW1KCIS (bounded validation) | `energy`, `gradient` | Global exact exchange fractions 0.22/0.15 with PBE/mPW91 and full KCIS correlation; molecular STO-3G RKS/UKS, energies in Hartree and gradients in Hartree/Bohr | Independent local fits and four molecular energy decompositions pass in the linked-density harness; raw occupied–virtual residual at most 2.603e-11 Hartree. [Model, term energies and acceptance scope](FUNCTIONALS.md#kcis-correlation-and-global-hybrids) |

### Pseudo-substituent requests and fitter

```bash
dft-rs energy model.xyz --pseudo-sub "4=Me"
fit_substituent --synthetic-check --output fitter-self-check.fit
```

`--pseudo-sub` uses zero-based `atom=name` assignments separated by semicolons
or newlines; `#` comments and `@file` includes are supported. Me, Et, tBu, Ph,
CF3 and OMe are recognized. All currently stop before basis construction/SCF
with a typed missing-parameter refusal, because no fitted named operator has
passed its required physical gates. `periodic` and `sapt` reject the option at
the command-option table.

`fit_substituent` is a separate mechanism checker. `--synthetic-check` runs
the required known-c/B recovery through weighted ridge initialization,
fixed-seed group-wise 80/20 splitting, B-eigenvalue projection and projected
L-BFGS. Its output is labelled `candidate-unaccepted` and cannot enable a
name. `--input` remains a typed refusal until the staged physical reference
evaluator (density/ESP/repulsion forces/C6 provenance/torsions) exists.

### Single points and properties

```bash
dft-rs energy water.xyz --xc b3lyp --basis def2-svp
```

```bash
dft-rs charges water.xyz --xc pbe0 --basis def2-svp --molden water.molden
```

`--dump-wfn FILE` writes a machine-readable wavefunction checkpoint (MO
energies and coefficients, density, overlap); `--molden FILE` writes a Molden
file for orbital visualization. Both work with any command that runs an SCF.

### Derivatives

```bash
dft-rs gradient water.xyz --xc pbe --basis def2-svp
```

```bash
dft-rs optimize ferrocene.xyz --xc pbe --basis "6-31g; Fe=lanl2dz" --ri-j --traj path.xyz
```

`optimize` writes `<input>.dftopt.xyz` and `saddle` writes `<input>.dftts.xyz`
next to the input. `--traj FILE` additionally streams a multi-frame XYZ **as
the walk proceeds** — appended and flushed per accepted step, with the step
number, energy and |g|max on each frame's comment line — so an interrupted run
still leaves its path on disk.

```bash
dft-rs frequencies water.xyz --xc m06-2x --basis def2-svp
```

`frequencies` and `hessian` take the analytic CPKS path whenever the method
supports it (HF, LDA, GGA, hybrids, range-separated, meta-GGA, SC-VV10, ECP)
and print the selected Hessian method. D2, D3(zero), D3(BJ), CHG, a uniform
external field, C-PCM, IEF-PCM and SS(V)PE retain the analytic path. RI-J or
another typed unsupported derivative falls back to central differences of
the complete energy-consistent gradient, which is stated rather than hidden.

```bash
dft-rs saddle guess.xyz --xc b3lyp --saddle-method rsirfo
```

`--saddle-method prfo` is the compatibility default. `rsirfo` applies the
Householder image function and solves the restricted RFO equation inside the
trust radius instead of scaling an overlong P-RFO step.

For a minimum, `frequencies` also prints ideal-gas RRHO thermochemistry.
`--thermo-temperature`, `--pressure`, and `--symmetry-number` set its standard
state. Thermochemistry is skipped explicitly for imaginary modes because no
transition-state convention is assumed silently.

### Excited states

```bash
dft-rs tddft water.xyz --xc b3lyp --basis def2-svp --nstates 8
dft-rs tddft water.xyz --xc b3lyp --nstates 5 --tddft-method rpa
dft-rs tddft water.xyz --xc b3lyp --nstates 5 --tda-spin triplet
```

`--tddft-method tda` (default) is the Tamm–Dancoff approximation; `rpa` is
full TDDFT/TDHF through the Casida reduced form. `--tda-spin` selects
`auto | singlet | triplet | spinresolved`. Output carries excitation energies
in eV and nm, oscillator strengths, and ECD rotatory strengths in both the
length and the velocity gauge — their difference is the basis-set error bar,
and for an achiral molecule both vanish.

### Polarizability and Raman

```bash
dft-rs polarizability water.xyz --xc b3lyp --basis def2-svp
dft-rs raman water.xyz --xc hf --basis sto-3g
```

The polarizability is analytic CPKS. Raman activities are `∂α/∂Q` by finite
differences along the normal modes, so `raman` costs a numerical Hessian plus
a polarizability per displacement.

### Periodic boundary conditions

```bash
dft-rs periodic diamond.xyz --cell 3.567 --xc pbe --basis sto-3g --k-mesh 4,4,4
```

```bash
dft-rs periodic chain.xyz --cell "2.9,8.0,8.0" --xc pbe0 \
    --k-mesh 8,1,1 --overlap-cutoff 12 --exchange-cutoff 12 --forces
```

```bash
dft-rs periodic crystal.xyz --cell 8.0 --xc pbe --relax \
    --opt-gtol 3e-4 --opt-max-steps 200 --traj crystal-opt.xyz
```

`--cell` is the one place where a length other than the XYZ coordinates is
entered, and it follows the same convention: **Ångström**. It takes one number
(cubic edge), three (orthorhombic), or nine (triclinic — three vectors, `;`
between them and `,` inside).

| Option | Meaning |
| --- | --- |
| `--k-mesh nx,ny,nz` | unshifted Monkhorst–Pack mesh (default `1,1,1` = Γ) |
| `--overlap-cutoff X` | Bohr; real-space cutoff of the **short-range** one-electron lattice sums |
| `--exchange-cutoff X` | Bohr; localized exact-exchange cutoff (default: the overlap cutoff) |
| `--grid-shell n` | cells of periodic images carried by the XC grid (default 2) |
| `--temperature X` | electronic temperature in Kelvin (default 0 = integer occupations) |
| `--forces` | also compute and print analytic forces |
| `--relax` | optimize atomic positions with analytic forces; lattice vectors stay fixed |
| `--opt-max-steps n` | accepted-step cap for `--relax` (default 200) |
| `--opt-gtol X` | maximum force convergence threshold in Hartree/Bohr (default `3e-4`) |
| `--opt-step-max X` | largest Cartesian step in Bohr (default `0.3`) |
| `--opt-history n` | L-BFGS history length (default 8) |
| `--traj file` | stream accepted fixed-cell geometries as multi-frame XYZ |

`--overlap-cutoff` is a genuine convergence parameter: Gaussian overlap decays
exponentially, so there is a limit to converge to. It does **not** control the
Coulomb sums, which are conditionally convergent and handled by Ewald
summation — a Coulomb sum truncated like an overlap sum converges to the wrong
number, silently and geometry-dependently. Converge the two cutoffs
independently for production work.

A positive `--temperature` switches on mesh-global Fermi–Dirac occupations and
additionally reports the entropy term, the Mermin free energy and the chemical
potential. Under smearing the analytic forces differentiate the **free**
energy, not the internal energy, and the output says which.

`--relax` rebuilds the complete immutable `Cell`, basis/ECP caches, Ewald
terms, XC grid and SCF at every trial geometry. At zero temperature it
minimizes total energy; with smearing it minimizes Mermin free energy. It is
strictly an atomic-position optimization: lattice-vector relaxation remains
refused until the electronic stress tensor is complete. Optimization controls
or `--traj` without `periodic --relax` are rejected rather than ignored.

The force block ends with the sum over atoms, which is zero by translational
invariance of the lattice sum and is printed as the cheapest available check
that no term was dropped.

The dimensional Coulomb foundation is available through
`periodic-electrostatics --pbc 1,1,0` (slab), `--pbc 1,0,0` (wire), or
any other nonempty mask. It accepts prescribed point/Gaussian charges,
including charged cells with an explicit compensation model. See
[boundary conventions and controls](periodic.md#1-boundary-conditions-and-electrostatics).
`periodic-basis --pbc 1,1,0 --k-mesh 3,3,1` supplies the actual Gaussian
overlap/kinetic matrices and fixed-lattice atomic derivatives. Optional
`--basis-grid --grid-level 3 --basis-grid-cutoff 12 --basis-grid-derivatives`
prints the open-direction grid and moving partition-weight derivatives.
See [electronic basis/grid conventions](periodic.md#2-electronic-basis-and-integration-grid) and the three
aligned [API tables](rust-api.md#common-calculation-driver) for units, complete output and cutoffs.
Complete partial-PBC electronic SCF/DFPT connections remain pending;
`periodic --pbc` identifies those missing connections. Periodic VV10/CHG
functionals (`wb97x-v`, `wb97m-v`, `b97m-v`, `wb97x-d`) still identify their
missing lattice-summed nonlocal terms.

`--disp`, `--nlc`, `--solvent` (and its raw descriptors), `--field`, `--nbo`,
`--dump-wfn` and `--molden` are refused **at the command line**, before the
XYZ is even parsed, because the `periodic` command never reads them: see
[Command availability](#command-availability) below for the full table and
the reason each one is refused.

### SAPT

```bash
dft-rs sapt dimer.xyz --natoms-a 3 --xc pbe0 --basis def2-svp
dft-rs sapt dimer.xyz --natoms-a 3 --mult-b 2 --xc pbe0    # open-shell monomer
```

The XYZ holds the dimer; `--natoms-a N` says that its first `N` atoms are
monomer A. `--charge-a/--charge-b` and `--mult-a/--mult-b` set the monomer
charges and multiplicities. Every term is printed separately in Hartree and
kcal/mol, followed by the counterpoise-corrected supermolecular interaction
energy and the δ remainder.

`--disp`/`--nlc`, `--solvent` (and its raw descriptors), `--field`, `--nbo`,
`--dump-wfn` and `--molden` are refused for `sapt` too, at the command line —
see [Command availability](#command-availability). Unlike `periodic`, `sapt`
reuses the ordinary SCF machinery, so `--incore`, `--ri-j`, `--level-shift`,
`--damping`, `--diis-size`, `--no-adiis`, `--adiis-until-error`,
`--no-soscf`, `--unrestricted`,
`--guess` and `--guess-mix` all work exactly as they do for `energy`.

### Command availability

`periodic` and `sapt` each run through their own code path instead of the
one shared by `energy`/`charges`/`gradient`/`optimize`/`saddle`/
`frequencies`/`hessian`/`polarizability`/`raman`/`tddft`, and some options
that path reads have no meaning — or no implementation — there. Giving one
of them to `periodic` or `sapt` is refused before any computation starts,
with a message naming the option and what is missing, rather than being
silently accepted and ignored.

| Option(s) | `periodic` | `sapt` | Why, when refused |
| --- | --- | --- | --- |
| `--disp`, `--nlc` | refused | refused | `periodic`: the lattice-summed empirical/nonlocal dispersion kernel is not implemented. `sapt`: a monomer dispersion correction would double-count the SAPT `E_disp(2)` term. |
| `--solvent`, `--eps`, `--refractive-index`, `--solvent-alpha/-beta/-gamma` | refused | refused | `periodic`: there is no periodic PCM. `sapt`: a solvated SAPT decomposition is not defined here. |
| `--field` | refused | refused | Neither Hamiltonian has a uniform external-field term wired in. |
| `--nbo`, `--dump-wfn`, `--molden` | refused | refused | The NAO/NPA, checkpoint and Molden bindings are not wired up for these commands yet. |
| `--incore`, `--incore-mb`, `--ri-j`, `--auxbasis` | refused | accepted | `sapt` reuses the ordinary `ScfOptions`; `periodic`'s own options struct has no in-core/density-fitting knobs. |
| `--level-shift`, `--damping`, `--diis-size` | accepted | accepted | Periodic level shifting targets the finite-k virtual projector and is automatically removed near convergence. |
| `--no-adiis`, `--adiis-until-error` | accepted | accepted | Periodic integer-occupation SCF uses complex ADIIS above the threshold, then CDIIS; smearing always uses CDIIS. |
| `--no-soscf` | refused | accepted | Periodic SOSCF is not implemented yet. |
| `--unrestricted`, `--guess-mix` | refused | accepted | `periodic`'s reference is chosen automatically from the alpha/beta electron counts and has no broken-symmetry mixing entry point. |
| `--guess sad|gwh|core` | accepted | accepted | Periodic Core is the compatibility default; SAD builds a periodic HF seed from spherical atomic densities before the finite-k occupation solve. |

Every other option — `--charge`/`--multiplicity`, `--basis`/`--ecp`,
`--xc`/`--x`/`--c`/`--omega`, `--grid-level`, `--conv-tol*`, `--max-iter`,
and the commands' own options (`--cell`, `--k-mesh`, `--overlap-cutoff`,
`--exchange-cutoff`, `--grid-shell`, `--temperature`, `--forces` for
`periodic`; `--natoms-a`, `--charge-a/-b`, `--mult-a/-b` for `sapt`) — works
normally for both. `info` and `integrals` are unaffected by this table.

This table is `OPTION_REJECTIONS` in `src/bin/dft_rs.rs`, checked by
`validate_option_table()` against every flag actually given on the command
line before the XYZ file is read; `tests/p21_cli_contract.rs` exercises it
entry by entry.

---

## Options

### System

| Option | Meaning |
| --- | --- |
| `--charge q` | total charge (default 0) |
| `--multiplicity m`, `--mult m` | spin multiplicity 2S+1 (default 1) |
| `--unrestricted`, `-u` | force UHF/UKS even for a closed shell |
| `--guess-mix deg` | rotate the β HOMO/LUMO of the guess to break spin symmetry; implies `--unrestricted` |

On its own `--unrestricted` reproduces the restricted energy to machine
precision — that identity is how the unrestricted path is gated. Combined with
`--guess-mix 45` it reaches broken-symmetry solutions for diradicals and
homolytic dissociation.

### Basis and ECP

`--basis` and `--ecp` take the **same grammar** and are **independent**: the
ECP spec is applied after the orbital basis, touches only the elements it
names, and `none` clears.

```text
--basis "def2-tzvp; H=def2-svp"            light atoms cheaper
--basis "def2-svp; Ni,Fe=lanl2dz"          ECP set on the metals only
--basis "H=sto-3g; O=./my-oxygen.g94"      no base; every element assigned
--basis @basis.spec                        read the spec from a file
--ecp "Ni=stuttgart-rsc-1997"              def2 orbitals, Stuttgart core
--ecp none                                 orbitals only, no core potential
```

48 sets are bundled (46 orbital sets plus two auxiliary fitting sets for
`--ri-j`); any `.g94` file can be given by path. AOs are spherical harmonics
throughout, in Molden's canonical `m` ordering — there is no Cartesian
(6d/10f) option.

### Functional

| Option | Meaning |
| --- | --- |
| `--xc name` | combined functional (default `hf`) |
| `--x name --c name` | exchange and correlation given separately |
| `--omega w` | override the range-separation parameter (RSH only) |
| `--xc-parameters '{"name": value, ...}'` | override named functional parameters, e.g. `'{"switch_lower": 1, "switch_upper": 1}'` for rSCAN (molecular driver calculations; an unknown name is an error) |
| `--grid-level 0..9` | quadrature resolution (default 3); radial points and Lebedev degree by period |
| `--grid-preset fine\|ultrafine\|superfine` | named per-atom (radial, angular) grid replacing the level: (75, 302), (99, 590), (150 for H–Ne and 225 beyond, 974) |
| `--grid-angular-scheme lebedev\|product` | angular rules (default `lebedev`; `product` is the Gauss–Legendre × uniform φ rule) |
| `--grid-weight-cutoff w` | drop quadrature points whose partitioned weight is below `w` Bohr^3 (default 1e-14); `0` keeps every point so the energy is smooth in the nuclear positions |

Names are matched case-insensitively with `-`, `_` and spaces ignored, so
`cam-b3lyp`, `CAM_B3LYP` and `camb3lyp` are the same entry. Run
`dft-rs --list-functionals` for the registry.

### Dispersion and nonlocal correlation

| Option | Meaning |
| --- | --- |
| `--disp model` | `d2 \| d3zero \| d3bj \| chg \| xdm \| vv10 \| vv10-post \| none` (default `auto`) |
| `--xdm-a1 value` | dimensionless Becke–Johnson XDM damping parameter; required with `--disp xdm` |
| `--xdm-a2 value` | Becke–Johnson XDM damping length in Bohr; required with `--disp xdm` |
| `--nlc model` | compatibility alias for the VV10 choices |

VV10 is offered through `--disp` because it *is* a dispersion treatment —
nonlocal correlation rather than an empirical pair sum — and choosing between
the two should be one decision. `vv10` is self-consistent, `vv10-post` a
fixed-density post-SCF correction.

Functionals that include a dispersion treatment in their **definition** (the
VV10 term of ωB97X-V/ωB97M-V/B97M-V, the CHG tail of ωB97X-D) enable it
automatically, and adding another model on top is a hard error rather than
double counting. `--disp none` still disables it, for debugging.

XDM is evaluated from the converged spin density, so it is wired after the
SCF rather than through the geometry-only `Dispersion` enum. It is available
for `energy` and `charges`; commands that require nuclear derivatives refuse
it before the SCF because the Becke–Roussel and grid-weight responses are not
implemented. dft-rs does not bundle unverified per-functional damping fits,
so both `--xdm-a1` and `--xdm-a2` must be supplied explicitly. XDM permits
`--nlc auto|none`, but cannot be combined with a VV10 mode (`vv10` or
`vv10-post`).

### Continuum solvation

| Option | Meaning |
| --- | --- |
| `--solvent NAME` | bundled solvent, optionally prefixed `smd:`, `cosmo:`, `ief:`, `ssvpe:` |
| `--solvent custom` | descriptors given directly instead of by name |
| `--eps X` | dielectric constant; alone this is enough for C-PCM/COSMO, and it overrides a named solvent |
| `--refractive-index X` | remaining SMD descriptors |
| `--solvent-alpha/-beta/-gamma X` | " |

```bash
dft-rs energy water.xyz --xc b3lyp --solvent smd:water
dft-rs energy solute.xyz --xc b3lyp --solvent custom --eps 8.93 \
    --refractive-index 1.4242 --solvent-alpha 0.10 --solvent-beta 0.05 \
    --solvent-gamma 39.15
```

The reaction field is self-consistent and participates in the CPKS/TDDFT
response kernel, so polarizabilities and Hessians see the solvent relax with
the density. `smd:` additionally switches the cavity to SMD's intrinsic
Coulomb radii and adds the G_CDS surface-tension term.

### External electric field

```bash
dft-rs energy water.xyz --xc b3lyp --field 0,0,0.005
dft-rs polarizability water.xyz --xc b3lyp --field 0,0,0.005
dft-rs optimize water.xyz --xc b3lyp --field 0,0,0.005   # relaxes in the field
```

`--field Ex,Ey,Ez` (atomic units) adds `+E·r` to the core Hamiltonian together
with the nuclear term `−Z_A R_A·E`. Everything at a fixed geometry is exact,
including `polarizability`, which differentiates with respect to the *field*
by CPKS on the already-perturbed Fock operator.

The analytic gradient carries the field as well — the geometric derivative of
the dipole integrals plus `−Z_A E` — so `gradient`, `optimize` and `saddle`
work in a field.

`frequencies`, `hessian` and `raman` retain the **analytic** Hessian in a
field. Its skeleton includes the second derivative of the dipole integrals,
while the ordinary CPKS equations carry the first-order field response.
`hessian --numerical` remains available as an independent check.

### SCF convergence

Thresholds follow PySCF's conventions, including the rule that
`--conv-tol-grad` defaults to `sqrt(--conv-tol)` — so tightening the energy
threshold alone moves the gradient threshold with it.

| Option | Default | Meaning |
| --- | --- | --- |
| `--conv-tol X` | 1e-9 | energy threshold |
| `--conv-tol-grad X` | √conv-tol | orbital-gradient threshold |
| `--conv-tol-dp X` | 1e-8 | density RMS threshold |
| `--max-iter n` | 150 | iteration cap |
| `--guess kind` | `sad` | `sad \| gwh \| core` |
| `--no-adiis` | | CDIIS only |
| `--adiis-until-error X` | 0.1 | use ADIIS above this commutator error |
| `--no-soscf` | | disable the second-order endgame |
| `--diis-size n` | 8 | DIIS subspace size |
| `--level-shift X` | | virtual level shift in Hartree |
| `--damping X` | | density damping while the error is large |

The default start is **SAD**: a superposition of spherically averaged atomic
densities, each converged on the fly in the molecule's own basis, so it ships
no data and follows any per-element basis or ECP assignment automatically.
The iteration schedule is ADIIS while the error is large → CDIIS → SOSCF near
convergence.

### Performance

| Option | Meaning |
| --- | --- |
| `--incore` | force the full ERI tensor into memory |
| `--incore-mb N` | memory budget in MiB for it (default 2048) |
| `--ri-j` | density-fit the Coulomb build |
| `--auxbasis name` | auxiliary basis for `--ri-j` (default `def2-universal-jfit`) |

The direct→in-core switch is automatic below the budget. **The cliff is
steep**, because the tensor is `n⁴` doubles: ferrocene in 6-31G with an ECP on
the metal is 132 AOs and needs 2317 MiB, 13 % over the default, and that alone
is the difference between a 64-second SCF and a geometry optimization that
completed no step in 40 minutes. Either raise `--incore-mb`, or use `--ri-j`
— for a pure GGA there is no exact exchange, so density-fitting the Coulomb
build removes the two-electron cost entirely.

### Logging

| Option | Default | Meaning |
| --- | --- | --- |
| `--log-mode m` | `console` for shared-driver commands; `both` for traditional commands | `console \| file \| both \| none`; an explicit file alone selects `file` |
| `--log-file f` | no file for shared-driver commands; `<input>.dft.log` for traditional commands | append run-local records to the selected path |
| `--log-level level` | `info` | `error \| warn \| info \| debug` |

`--cpu-threads N` applies the shared scoped thread ceiling, including nested
Rayon/faer work. Without an explicit ceiling, existing thread defaults apply;
`RAYON_NUM_THREADS` can configure Rayon's default. See the resource and logging
contracts at the start of this manual.

---

## Exit behaviour

The binary exits non-zero and prints the error to stderr on any failure. It
does not substitute an approximate result for one it cannot compute: an
unsupported analytic derivative, a periodic kernel that is not implemented,
or an under-determined solvation parameter each produce an error that names
what is missing. See [scope.md](scope.md) for the complete list of those
boundaries.

Named nonlinear optical presets use `--electric-process` and
`--process-frequencies`; their ordered frequency legs and damping conventions
are listed in [the API reference](python-api.md#nonlinear-optical-presets).
`--stab-follow` runs internal following before molecular single-point and
response properties. Geometry optimizers, SAPT, and periodic calculations
need additional state integration and currently reject this flag explicitly.

## Portable Python packaging

`pip install .` installs the canonical `dft-rs` console script through
`dft_rs.native:cli_main`. This entry point calls the native common CLI
directly. No separate Rust executable, NumPy or ASE is required to run it.
An explicit `cli_main(argv)` remains available; omitted arguments read the
process argument list. Project URLs identify the ss0832 repository.

The standard-library PEP 517/660 backend in `tools/build_backend.py`
delegates compilation to the original maturin dependency. Compiler path
remapping removes private roots from native debug/dependency data before
the artifact is produced. Generated SBOM dependency metadata stays enabled:
private build locations in that JSON are normalized consistently in IDs and
dependency edges. Native machine code is never patched. The backend audits
wheel contents, rejects remaining private paths, and recomputes every
`RECORD` hash and length after metadata normalization.

Acceptance builds an actual wheel through pip, installs it into a fresh
environment without NumPy or ASE, and invokes `dft-rs` from outside the
source tree with no source `PYTHONPATH` or external native CLI. The measured
historical `0.2.0-alpha.1` Linux ABI3 wheel retained 122 SBOM components and
123 dependency nodes; all `RECORD` hashes and the private-path audit passed.
Its H2 energy was -1.1167143251757705 Hartree and an invalid command exited
with status 1. See the [implementation status](development-status.md#api-distribution-and-performance)
for the current package metadata and regression status. The existing
source-distribution audit checks that build tools are retained while environments, logs, reference inputs
and build products are excluded. These measurements validate this build
platform; each other platform still needs its own native wheel build.
Rust compiler warnings are denied in the crate lint configuration.

### Source archive verification

After freezing the source, use the repository-owned helpers. The file and
directory names in this example are local labels. The archive version comes
from the staged package metadata.

```console
python tools/stage_release.py ../release-stage
python ../release-stage/tools/verify_release.py release-check
python ../release-stage/tools/make_release.py --zip
```

The stage must be a fresh directory outside the crate. Both staging and ZIP
creation use Cargo's include inventory, preserving public data and required
licenses while rejecting private work, caches, binaries and unreviewed links.
`SOURCE_MANIFEST.json` records each member's size and SHA256. Archive creation
checks actual membership, CRCs, hashes, Rust/Python package identity and the
shared privacy/English policy before making the complete ZIP available.
Existing stages and archives are never replaced. A manifest proves which
source bytes were packaged; it does not establish scientific acceptance.
The final ZIP additionally requires a build and the declared regression
scope from a fresh unpacked directory. Local archive creation does not
publish a package or authorize a repository tag or upload.

`--module-only` runs the declared all-target feature/Clippy matrix without
executing physical tests. Omitting it invokes the full Rust and Python
suites, including continuous-species tests. The
[implementation status](development-status.md#api-distribution-and-performance)
records which release checks passed and which are pending.
A module-only pass must not be reported as a full regression pass.


## Balanced Dirac projection and supplied-band pairing

`dft-rs no-pair-projector --dirac-hamiltonian-real ... --spectrum-separator ...`
projects supplied balanced four-component matrices without a structure file.
`dft-rs no-pair-projector atom.xyz --dirac-integrals rkb --basis sto-3g --nuclear-rms-radii ...`
generates the finite-Gaussian-nucleus, restricted-kinetic-balance molecular
operator first. Radii and displacement controls are Bohr; XYZ is Angstrom.
`--response-order 0..4` differentiates the supplied H/S directions or the
generated nuclear/radius/c path. The latter retains integral and metric
response, full one-electron SOC and negative-energy-sector coupling. It does
not run an interacting Dirac SCF calculation.

Optional `dirac_operator_real` and `dirac_operator_imaginary` are **dimensionless
covariant one-electron operator** Taylor coefficients indexed
`[order][AO row][AO column]`, including zeroth order. Supply one through
`response_order+1` coefficients; absent real/imaginary parts and omitted higher
orders are exactly zero. Every combined coefficient must be Hermitian, finite
and match the four-component dimension. These public inputs and outputs remain
dimensionless under native and ASE unit conversion. No physical operator or
physical-unit conversion is inferred from numeric matrices.

The optional outputs `projected_operator_coefficients_real/imaginary` contain
factorial-normalized `Q=S P O P S` coefficients, with a precise
`projected_operator_model` label. The low-level Rust function
`properties::no_pair::no_pair_operator_taylor(H,S,O,order,separator,cutoff)`
retains the supplied operator's units and returns `NoPairOperatorTaylor`
(`sector` diagnostics and `operator` coefficients). This is bare one-electron
no-pair projection including moving metric and electronic/negative-sector
response, not decoupled X2C picture change, interacting DKS, two-electron
picture change or a conserved-current construction.

The isolated implementation/oracle acceptance passes 128 independent complex
operator/metric component fits through fourth order: p=2.000009042105408 to
2.0013609184029173, minimum R2=0.9999998761022915, at steps
[0.08,0.04,0.02,0.01] in dimensionless path parameter. All ordered coefficients,
the identity `O=S -> SPS`, Hamiltonian equivalence and input contracts pass.
The rebuilt `tests/no_pair_operator.rs` and native/ASE atomic/physical/both-CLI integration now pass on the scoped follow-up artifact. Static getter binding and explicit resource forwarding also pass after the reproduced Python wrapper defect was corrected. This remains bare operator projection, with the limitations above.


`dft-rs scdft-pairing-kernel` consumes normal-band energies, phonon energies,
squared vertices, temperature and screened Coulomb entries. It returns the
static off-diagonal phonon/Coulomb components and optional directional
coefficients; it does not yet solve a superconducting gap. See the
[complete controls, units and limitations](rust-api.md#four-component-electronic-sector-projection-component)
and the adjacent pairing-kernel section. Both commands share the native
Rust/Python/ASE driver and JSON diagnostics.


## Complete nuclear tensors and reference constraints

The common `gradient`/`hessian` calculations accept explicit `reference`
constraints and HF aliases as described in [response.md](response.md#full-nuclear-tensors-for-generic-double-hybrids).
GKS uses complete complex spinor density blocks; the semilocal nuclear path
contains moving Becke weights, all Pauli density/gradient/tau/current terms
and independently refined angular quadrature. Additional supplied AO spin
operators are held fixed along a nuclear path; no missing SOC integral
derivative is implied. Parent nonlocal, ECP and solvent kernels retain
their documented scope. ROKS/GKS PT2 is not inferred from parent support.

For `double_hybrid_model="custom"`, `xc` selects the stationary parent
and optional `double_hybrid_target_xc` selects a distinct semilocal or
hybrid target evaluated on those orbitals. Both use the same grid settings;
a target grid is built even for an HF parent. The full parent orbital
response is retained for this nonstationary target. Named recipes fix
their target and refuse an override. Additional target nonlocal terms
remain a separately recorded provider limitation. The common driver uses
this control on Rust, native Python and ASE; the CLI spelling is
`--double-hybrid-target-xc`. For example, `xc="hf"`,
`double_hybrid_model="custom"`, `double_hybrid_target_xc="pbe"`,
`pt2_same_spin=0.37`, `pt2_opposite_spin=0.61` defines PBE + weighted PT2
on HF orbitals. All 312 total/target derivative component fits pass
(p=1.817002..2.181338, minimum R2=0.99919170). The common driver agrees
bitwise on all total/target tensor entries, and the final 36-case
native/ASE/CLI/package suite passes with warnings treated as errors.
This defines a supplied model, not a fitted named functional.

Double-hybrid `gradient` and `hessian` use the existing energy API with
`xc` or explicit custom PT2 coefficients. Every gradient component and
Hessian element is returned. `parent_`, `target_`, `pt2_`,
`pt2_same_spin_alpha_`, `pt2_opposite_spin_`, `pt2_same_spin_beta_` and
`dispersion_` prefix complete `gradient`, `hessian` and `nuclear_energy`
outputs. Unscaled spin channels reconstruct weighted `pt2_`; total equals
target + PT2 + dispersion. Units are Hartree, Hartree/Bohr and
Hartree/Bohr^2, with axes [atom][xyz] and [3*atom+axis][3*atom+axis].
Reports include orbital/amplitude response, projector, stationarity,
metric, number and reference residuals. Top-level `converged` requires
these nuclear diagnostics as well as SCF convergence. The new GKS and
correlated Hessian providers return complete Cartesian energy derivatives;
they do not yet generate the additional dipole derivatives, APT or IR
intensities supplied by the existing collinear vibrational driver. Correlated nuclear full-component
fits for four custom parents and the 24-case public suite pass, including
B2PLYP, DSD-BLYP, DSD-PBEP86 and PWPB95 nuclear tensors on native/ASE
and B2PLYP on canonical CLI JSON/plain text and legacy native entries.
Complex PBE/PBE0 nuclear and all four ROKS-class tensor fits pass;
complex TPSS remains unvalidated after failing the order-64 angular gate. `nuclear_xc_cache_bounds`
contains conservative bounds separately from the actual coefficient
changes in `nuclear_xc_angular_history`; physical residual gates are unchanged. Exact scope and failed/resolved
intervals are recorded in response.md. Native `hessian(..., reference=...,
**controls)` routes to this analytic common driver without numerical fallback.
ROKS accepts common response controls on energy/gradient requests as well
as Hessian/polarizability requests. They are validated in every case; only
the latter two require orbital-response equations. ROKS Hessian convergence
also includes its accompanying electric-response calculation.

## Table 2: CLI

[Choose a calculation by property](api-guide.md#find-a-calculation) / [Match its ID to the command](api-guide.md#choose-an-interface) / [Check input units](api-guide.md#choose-units-for-inputs-and-reports)

All commands take an XYZ file. XYZ positions use Angstrom; response quantities
use atomic units. Commands print physical values and shared report diagnostics.
The executable is `dft-rs`; `dft_rs_cli` remains a compatibility alias.

The shared molecular reference report gives the raw SCF stationarity
residual in Hartree as `scf_orbital_residual` for collinear references or
`orbital_residual` for GKS. Iteration diagnostics go to stderr.

| ID | Command | Calculation-specific controls | Output |
|---|---|---|---|
| `continuous-species` | `dft-rs continuous-species nitrogen.xyz --xc pbe --basis sto-3g --fixed-electron-count 14 --charge-directions "[[-1,1]]" --charge-coordinates "[0.25]" --charge-bounds "[[-1.5,1.5]]" --format json` | Required fixed electron count, parameter-by-atom charge-direction rows, dimensionless coordinates and bounds; optional reference charges, origin (Bohr), conservation flag and nuclear gradient. Explicit fixed-spin counts and local/steric/frame controls are validated in the stated subset. Alternate `--fragment-mode generate` uses `--fragment-spec` JSON; `--fragment-mode evaluate` uses opaque `--fragment-model-json` text and the saved model's reference. | Native atomic units; analytic clamped parameter responses and optional moving-template forces, physical total charge and diagnostics. Saved closed-shell methyl candidate generation/evaluation passes native/ASE/CLI parity; chemical accuracy remains failed. See response.md.  The common charge, fixed few-center and generated-reduced model reports also return `energy_without_internal_reference` and `classical_energy_without_internal_reference` in Hartree, plus `scf_energy_convention`; state-common constants are added only to reported totals.  Experimental `scalar-evaluate`/`scalar-optimize` add explicit full frozen joint-endpoint actions and site weight optimization; see the model contract below. Saved weight routes pass clean installed checks. `scalar-gradient`/`scalar-geometry-optimize`/`scalar-hessian` require explicit endpoint/state owners; paired RKS and explicitly spin-resolved UKS records remain distinct. Hessian reports include all ordered entries and raw residuals; the stated PBE RKS/UKS fixture and installed-selector gates pass, while real-model minima remain separate. |
| `multicomponent` | `dft-rs multicomponent mol.xyz --xc pbe --dispersion none --quantum-atoms 1 --quantum-position-constraint --species-derivative-order 2 --format json` | Legacy quantum atom/mass/exponent/lmax or `--particles JSON`; `--positron-correlation`; `--annihilation-enhancement`; optional `--annihilation-gradient-alpha`; `--quantum-position-constraint`; optional `--quantum-position-targets`; `--species-derivative-order 0/1/2`; optional `--species-direction`; `--species-polarizability`; `--species-stability none\|report\|follow`, `--species-stability-roots`; `--interspecies-correlation epc17\|epc17-2\|none`; shared electronic and response controls | Experimental simultaneous electronic/nuclear energy, constrained mean-position or unconstrained basis-center gradient/Hessian, total static alpha, disjoint energies and raw residuals; laboratory frame. Structured particle/channel and spin-averaged annihilation contracts, units and bounded acceptance are below.  Optional `quantum_mass_response` returns fixed-basis quantum-mass gradient/Hessian and raw coupled diagnostics; units and scoped acceptance are below. For relativistic joint energy use `--species-electronic-model dirac-coulomb --nuclear-rms-radii "[0.12,0.081]" --quantum-atoms "[0,1]"`. Radii are in Bohr, energies and orbital residuals in Hartree. SCF iteration energies and residuals go to stderr. For the Dirac joint reference, `--field` (Hartree/e/Bohr) returns the total dipole (e Bohr). Its `--species-polarizability` accepts `--frequency`, `--frequency-damping` and `--imaginary-frequency`. For a nonrelativistic molecular positron, `--field` perturbs the SCF and `--species-polarizability` returns the static dipole tensor (Bohr^3) and raw response residuals. [Positron benchmark and conditions](response.md#r076-multicomponent-nuclear-positron-photon-dft). With `species_electronic_model=dirac-coulomb`, `dirac_interaction=coulomb\|gaunt\|breit` selects the electron-electron operator for the coupled SCF and electric response. The current direct and exchange terms and Hermiticity defect are in Hartree. [Model and validation](response.md#r081-relativistic-quantum-nuclear-response).  In the fixed-sector relativistic quantum-nuclear model, `species_field_gradient` supplies five traceless symmetric gradient modes (Hartree/e/Bohr^2), and `species_quadrupole_polarizability` requests their coupled response. Named or explicit `origin` is supported, default zero. `electric_quadrupole` reports Buckingham moments (e Bohr^2), static/complex 5x5 response (e^2 Bohr^4/Hartree), and raw conjugate dipole-gradient 3x5/5x3 blocks (e^2 Bohr^3/Hartree); multiplying the latter moment rows by three gives Buckingham response. Eight source residuals, iterations, reference residual, mode order and signed frequency are exposed. Gaussian-product, full derivative and public acceptance are pending. The nonrelativistic route accepts `--frequencies [omega,...]` and `--frequency-damping eta` in Hartree for the signed-frequency dipole response, and `--species-quadrupole-polarizability true` for the static quadrupole and mixed tensors. [Verification status](response.md#r076-multicomponent-nuclear-positron-photon-dft). A nonrelativistic reference also accepts `--species-field-gradient` with five mode amplitudes in Hartree/e/Bohr^2. The uniform field couples at laboratory zero. |
| `casscf` | `dft-rs casscf mol.xyz --xc hf --basis sto-3g --cas 4,4 --cas-roots 2 --format json` | `--cas` (electrons, orbitals), `--cas-roots`, `--state-weights`, `--active-orbitals`, `--casci`, `--casscf-max-iter`, `--casscf-gradient-tol` (Hartree, default 1e-8; the PDFT energy follows the orbital convergence linearly)/`--casscf-energy-tol` (Hartree), `--ci-tol` (Hartree), `--ci-max-iter`, `--derivative-order 1/2` (analytic gradients/Hessians); reference SCF controls | State energies (Hartree), natural occupations, `<S^2>`, residuals; SCF iterations and CASSCF macro iterations on stderr |
| `mcpdft` | `dft-rs mcpdft mol.xyz --xc hf --basis sto-3g --cas 2,2 --on-top-functional tpbe --pdft-method cms-pdft --cas-roots 2 --format json` | The `casscf` controls plus `--on-top-functional` (`t<name>`, `ft<name>`, `tpbe0`, `ftpbe0`), `--pdft-lambda`, `--pdft-method` (`state-specific`, `l-pdft`, `cms-pdft`, `xms-pdft`), `--grid-level`, `--derivative-order 1/2` (state-specific analytic gradients/Hessians) | PDFT energies and per-state terms (Hartree), multistate rotation and effective Hamiltonian Energy acceptance: p230 (8 tests), p232 (9 frozen total-energy cases and 20 term comparisons; maximum term defect 4.418e-7 Hartree). Native gradients/Hessians use Hartree/Bohr and Hartree/Bohr^2; ASE uses eV/Angstrom and eV/Angstrom^2. Axes are [state][atom][axis] and [state][3 atom + axis][3 atom + axis]. derivative_stationarity is the largest remaining orbital/CI residual in Hartree; derivative_solve_residual is dimensionless. The transported orbital/CI Taylor path uses the common matrix-free GMRES and orthogonal retraction. Local Hessian grid convergence and open-active GGA derivative combinations require their recorded acceptance in response.md. Four CASSCF/MC-PDFT public-route cases pass, including gradients/Hessians and atomic/ASE conversions (path stationarity 3.641e-12 Hartree, solve residual 3.380e-13). MC-PDFT derivative state axes follow state_total in reference-state order, while pdft_energies are sorted; CASSCF derivatives follow cas_energies. |
| `superconducting-state` | `dft-rs superconducting-state --band-energies ... --band-weights ... --electron-number ... --chemical-potential ... --electronic-temperature ... --initial-gap-real ... --pairing-model lm2005 --phonon-energies ... --pair-couplings-squared ... --screened-pair-coulomb ...` | Hartree energies/gaps/K; squared vertices Hartree squared; Kelvin temperature; dimensionless weights/Z. `supplied` and `lm2005-off-diagonal` instead require `--mass-renormalization`; `--response-order 0..4`; common solver controls; no structure file | Coupled complex gap/number/phase and generated or supplied Z response; supplied normal data and explicit paths. `pairing_model="nambu"` evaluates full complex interband normal/anomalous density on supplied H/Delta/S, at positive temperature and supplied chemical potential. This model is grand canonical and kinematic; the existing supplied/lm2005 band providers retain canonical gap/number/phase self-consistency. Supply `--scdft-temperature-bracket '[400,1200]'` in Kelvin for the canonical linear pairing onset, using the same band and K/Z controls. The report includes `critical_temperature_k`, the final bracket, chemical potential and raw original gap/eigenvector/number residuals. For a reciprocal supplied kernel with uniform Z, `--scdft-phase-selection true` selects and checks the thermodynamic phase at one electronic temperature without an initial gap or chemical-potential guess. It returns the condensation free energy in Hartree and a dimensionless stability margin. With `--pairing-model nambu`, add `--frequencies` and optional `--frequency-damping` in Hartree for fixed-occupation retarded linear response. Provide row-major flattened `--nambu-probe-normal-real`, `--nambu-probe-normal-imaginary`, `--nambu-probe-pairing-real` or `--nambu-probe-pairing-imaginary` rows and optional symmetric `--nambu-probe-interaction` in Hartree. Probe vertices are dimensionless. Complex susceptibility is inverse Hartree, with axes frequency, observed, driven. The report includes raw residuals and stability margin. [Model and verification](response.md#r050-scdft-gap-anomalous-density). |
| `no-pair-projector` | `dft-rs no-pair-projector molecule.xyz --dirac-integrals london-rmb --nuclear-rms-radii ... --magnetic-direction 0,0,1 --response-order 1 --format json` | Generated RKB, common-origin RMB or London-RMB H/S. Optional `--magnetic-field`, `--gauge-origin`, `--magnetic-phase-order`, Gaussian radii in Bohr. Supplied mode reads Hermitian H/S matrices. | Factorial-normalized complex H/S and no-pair coefficients, gap (Hartree), raw residuals. [Models and units](rust-api.md#restricted-magnetic-balance-and-a-complex-four-component-metric). |
| `dirac-coulomb-scf` | `dft-rs dirac-coulomb-scf molecule.xyz --nuclear-rms-radii "[0.12,0.081]" --format json` | Optional basis, charge, speed of light, metric cutoff, SCF tolerances, damping and DIIS size. Nuclear RMS radii are in Bohr. | Fixed-sector finite-nucleus Dirac–Coulomb HF energy and term breakdown in Hartree, SCF iteration stderr and raw orbital, Coulomb, density and metric residuals. Use `--dirac-interaction` to select Coulomb, Gaunt or Breit. Current direct and signed exchange energies and raw Hermiticity defects are in Hartree. Use `--dirac-integrals london-rmb --magnetic-field ...` for a finite-field HF reference. The same London-RMB basis provides optional Gaunt and Breit direct and exchange corrections at finite field. |
| `two-component-relativistic` | `dft-rs two-component-relativistic molecule.xyz --relativistic-method x2c --xc wb97x-d --nuclear-rms-radii "[0.00006,0.00002]" --response-order 1 --format json` | `--relativistic-method x2c/dkh2/matrix-zora`, `--relativistic-spin scalar/spin-orbit`, one positive radius per nucleus in Bohr, optional uniform `--field` in Hartree/(e Bohr), `--electric-origin` in Bohr, `--frequencies` and `--frequency-damping` in Hartree. The field is applied in the field-free picture. | Rest-subtracted energy and raw orbital/decoupling residuals in Hartree, permanent dipole in e Bohr and response-order 1 complex polarizability in e² Bohr²/Hartree. The same SCF and kernel retain the required range exchange, CHG or VV10. [Molecular model and verification](response.md#r075-breit-and-two-component-relativistic-dft). |
| `dirac-dks` | `dft-rs dirac-dks molecule.xyz --xc wb97x-v --nuclear-rms-radii "[0.0000547,0.0000159]" --format json` | An isolated all-electron even-electron spin-orbit reference with one finite Gaussian radius per atom in Bohr. Select `coulomb`, `gaunt` or `breit` with `--dirac-interaction` for the fixed-sector two-electron operator and `--response-order 1` for electric response. Signed frequencies and damping are in Hartree. The current models report separate signed direct and exchange terms and the raw Hermiticity defect. | Energy, spectral gap and orbital residual in Hartree, dipole in e Bohr, optional complex polarizability in e² Bohr²/Hartree and raw response residuals. Use `--magnetic-field` with `--dirac-integrals london-rmb` and converge `--magnetic-phase-order` for a finite-field reference. The London-RMB option supports the current corrections with `dirac_interaction=gaunt` or `breit`. |
| `scdft-pairing-kernel` | `dft-rs scdft-pairing-kernel --band-energies=-0.016,0.007 --chemical-potential 0 --electronic-temperature 1500 --phonon-energies 0.025 --pair-couplings-squared "0.0001,0.0002,0.0002,0.0001" --screened-pair-coulomb "0.01,0.002;0.002,0.008" --band-weights 0.6,0.4` | Energies Hartree, squared weighted vertices Hartree squared, temperature Kelvin; optional band weights generate dimensionless Z. Parameter directions and `--response-order 0..4`; no structure file | Static K components and optional Z coefficients with separate raw quadrature errors; K asymmetry; no gap solve |
| `dynamic-polarizability` | `dft-rs dynamic-polarizability molecule.xyz --reference roks` | `--frequency w`, `--frequency-damping eta`, `--imaginary`; compatible `--response-omega`, `--response-damping`, `--imaginary-frequency`; `--reference rks/uks/roks/gks` and HF aliases | All nine real/imaginary components in bohr^3 and raw residuals. Frequencies in Hartree. [Collinear double-hybrid quasi-energy response](rust-api.md#frequency-dependent-double-hybrid-electric-response). `--relativistic-method dirac` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `molecular-medium-optics` | `dft-rs molecular-medium-optics molecule.xyz --frequency 0.1 --frequency-damping 0.005 --medium-number-density 0.002 --slab-thickness 1500` | Number density in Bohr⁻³, thickness in Bohr, optional local-field model, orientation, exterior indices and incidence angle in radians | Complex indices, absorption, reflection, transmission and raw residuals. [Model and scope](response.md#r080-molecular-medium-optics). With `medium_magnetic_field`, the linear orbital magnetic susceptibility correction supplies field-odd Kerr rotation and ellipticity in radians. |
| `field-taylor` | `dft-rs field-taylor molecule.xyz` | `--field-direction x,y,z` (default `0,0,1`), `--response-order n` (default 3), `--field x,y,z` for the reference field | Dipole Taylor coefficients from order zero through n and order-resolved defects; HF/PBE0 water alpha/beta/gamma verified at zero and nonzero reference field by twelve independent four-step fits; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `field-mixed` | `dft-rs field-mixed molecule.xyz --field-directions "1,0,0;0,1,0"` | Required semicolon-separated directions; `--field x,y,z` for the reference field | Unnormalized mixed dipole derivative, accumulation roundoff, evaluation count; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `beta` | `dft-rs beta molecule.xyz --xc pbe` | Shared SCF/response controls; optional `--field x,y,z` reference field; `--electric-assembly projector` (default) or `variational` (2n+1) | All 27 `beta a b c value` entries, contraction vector/norm, raw interchange measurements and residuals; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The third-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^3*Bohr^3/Hartree^2, with raw response and pair residuals. |
| `gamma` | `dft-rs gamma molecule.xyz --xc pbe` | Shared SCF/response controls; optional `--field x,y,z` reference field | All 81 `gamma a b c d value` entries, isotropic contraction, raw interchange measurements and residuals; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The fourth-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^4*Bohr^4/Hartree^3, with raw response and pair residuals. |
| `dynamic-beta` | `dft-rs dynamic-beta molecule.xyz --xc pbe --input-frequencies "0.05,0.005;0.07,0.005"` | Two incoming omega,damping pairs in Hartree; optional imaginary axis; shared SCF/response controls | All 27 `beta a b c real imag` entries, complex output frequency, original-operator residuals and full density constraints |
| `dynamic-gamma` | `dft-rs dynamic-gamma molecule.xyz --xc pbe --response-omega 0.05 --response-damping 0.005` | Three explicit incoming pairs, or repeat single-frequency defaults for third-harmonic generation | All 81 `gamma a b c d real imag` entries and the same diagnostics |
| `electric-process` | `dft-rs dynamic-beta molecule.xyz --electric-process shg --response-omega 0.05` or `dynamic-gamma ... --electric-process optical-kerr --process-frequencies 0.05` | `--electric-process`, optical magnitudes via `--process-frequencies w1,w2,...`, shared `--response-damping`; scalar `--response-omega` shorthand for one-frequency presets | Named process and all raw complex beta/gamma entries, exact DC legs and causal complex subset sums |
| `stability` | `dft-rs stability molecule.xyz` | `--stab-channels`, `--stab-roots`, `--stab-tol`, `--stab-max-iter`, `--stab-max-subspace`, `--stab-block-buffer`, `--stab-resolution`, `--stab-no-verify`, `--stab-verify-steps`; explicit `--unrestricted` for UKS/UHF singlets | Channel verdicts, eigenvalues/energy curvatures in Hartree, dominant excitations, resolution and residuals |
| `stab-following` | `dft-rs stab-following molecule.xyz`; `stability --stab-follow`; `--stab-follow` on molecular SCF/property commands | Analysis controls plus `--stab-max-descents`, `--stab-step-angle`, `--stab-min-energy-drop` | Initial/final energies, every accepted signed descent, final analysis; subsequent requested molecular property uses the final reference |
| `orbital-energy-gradients` | `dft-rs orbital-energy-gradients molecule.xyz --orbital-coordinates 5,2` | Optional Cartesian IDs `3*atom+axis` (default all); `--degeneracy-tol` in Hartree, shared SCF/response controls and optional reference `--field`; explicit `--unrestricted` | Energies, individual gradients or `undefined`, full grouped matrices and physical/reference diagnostics |
| `frontier-gap-gradient` | `dft-rs frontier-gap-gradient molecule.xyz` | The same orbital controls | Per-spin LUMO-HOMO gap/gradient or explicit absent/undefined results, plus the orbital report |
| `nuclear-taylor` | `dft-rs nuclear-taylor molecule.xyz --nuclear-direction "0,0,0;0,0,1"` | Required per-atom Bohr displacements; `--response-order n` defaults to 4 | Factorial-normalized energy coefficients, component energies and moving-basis diagnostics |
| `nuclear-mixed` | `dft-rs nuclear-mixed molecule.xyz --nuclear-directions "0,0,0;0,0,1\|0,0,0;0,0,1\|0,0,0;0,0,1"` | Atoms separated by semicolons, complete displacement paths separated by `\|`; number of paths sets derivative order | Actual mixed energy derivative, roundoff estimate and evaluation count |
| `molecular-taylor` | `dft-rs molecular-taylor molecule.xyz --nuclear-direction "0,0,0;0,0,1" --field-direction 0,0,1` | `--response-order n` (default 4), `--field x,y,z` for the reference field | Energy/component and dipole jets with physical defects |
| `force-constants` | `dft-rs force-constants molecule.xyz --response-order 4 --force-coordinates 2,5` | Order 3 (default) or 4; optional unique Cartesian IDs `3*atom+axis`, or `--nuclear-directions` for a set of displacement patterns; omit both for the full Cartesian tensor | Actual derivative tensor, component/evaluation counts, raw reverse-order comparisons and measured/absent acoustic diagnostic ; integer correlated RKS/UKS complete Cartesian/block/pattern tensors include full orbital/PT2 amplitude response and separate target/SS/OS/dispersion channels; coordinate units Hartree/Bohr^order, dimensional-pattern unit Hartree; energy permutation reuse is explicit, with raw residuals and independent acceptance tracked in response.md  ROKS/ROHF and complex GKS/GHF complete exports are connected; full acceptance remains in progress, with raw failures documented in response.md. Geometry-only D1 is accepted through the shared driver using explicit atomic C₆, radii, and scale. [Model and checks](response.md#molecular-d1-energy-and-nuclear-tensors). Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `anharmonic` | `dft-rs anharmonic minimum.xyz --xc pbe --anharmonic-raman` | Optional `--anharmonic-method strict-vpt2` (default retained-polyad), `--vibrational-modes 0,1`, `--equilibrium-tol`, `--resonance-tol`, `--resonance-coupling-factor`, `--max-vibrational-basis`, `--max-polyad-states`; shared SCF/response solver controls | `anharmonic_band index character frequency_cm harmonic_frequency_cm IR_km_mol harmonic_IR_km_mol dominant_weight`, separate transition/mechanical/electrical/mixed dipoles, optional static Raman activity, zero-point energy and polyad diagnostics |
| `molecular-mixed` | `dft-rs molecular-mixed molecule.xyz --nuclear-directions "0,0,0;0,0,1\|0,0,0;0,0,0" --field-directions "0,0,0;0,0,1"` | Equal nonzero counts of nuclear and field paths; each paired path is one mixed leg | Actual mixed energy/dipole derivatives, separate roundoff estimates |
| `static-spectroscopy` | `dft-rs static-spectroscopy molecule.xyz --nuclear-direction "0,0,0;0,0,1" --electric-order 3` | `--electric-order 1` = APT, `2` = Raman (default), `3` = hyper-Raman; optional reference `--field` | Flat Cartesian tensor, raw permutation defect, roundoff and evaluation count; common-driver integer RKS/UKS complete post-SCF target and PT2 amplitude/orbital response; nuclear paths in Bohr, atomic electric units, all ordered tensor entries; complete C1 HF+PT2 RKS/UKS static APT/Raman/hyper-Raman subset validated by all 570 independent fits and all 12 primary native/ASE/CLI checks; raw normalized amplitude, reference, metric, particle-number and stationarity diagnostics and failed-constraint warnings are returned. Use the common driver for target/PT2 energy composition; the low-level Rust entry differentiates its supplied reference functional. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). |
| `periodic-response` | `dft-rs periodic-response cell.xyz --cell "9,0,0;0,9,0;0,0,9"` | `--response-q x,y,z`, `--reciprocal-cutoff R`, `--response-modes m` (one source index), frequency controls, `--fixed-chemical-potential` | Complete complex AO `density spin k bra_AO ket_AO real imaginary` rows (hartree^-1), Fourier density, complete reciprocal-wave list, occupation consistency defect; HF/full-range exchange uses the reference cutoff and mesh; `--grid-weight-cutoff 0` is available |
| `field-response` | `dft-rs field-response cell.xyz --cell 4.7 --xc pbe` | `--response-q`, response frequency/solver controls, `--field-scalar re,im`, `--field-vector "re,im;re,im;re,im"`, reciprocal cutoff and periodic SCF controls | Complex full density/current, electric/magnetic amplitudes, shifted mesh, raw reference/response/continuity diagnostics |
| `dielectric` | `dft-rs dielectric cell.xyz --cell "9,0,0;0,9,0;0,0,9"` | Same periodic controls; `--response-modes m0,m1,...` selects both reported matrix axes | Complex screened susceptibility and inverse dielectric matrix |
| `thermal-response` | `dft-rs thermal-response cell.xyz --cell "9,0,0;0,9,0;0,0,9" --temperature 40000` | `--delta-temperature K` (default 1), `--delta-numbers a,b`, or `--fixed-chemical-potential --delta-mu a,b`; reciprocal and solver controls | Complex Fourier variations, spin number/chemical-potential changes, entropy/internal/free-energy changes and defects |
| `periodic-electrostatics` | `dft-rs periodic-electrostatics cell.xyz --cell 5,5,20 --pbc 1,1,0 --electrostatic-charges 1,-1` | Geometry/cell in Angstrom; source widths and countercharge center/width in Bohr; `--charge-compensation neutral` (default), `uniform` (3D), or `gaussian` with explicit center/width; Ewald cutoffs and quadrature controls | Same energy/components, analytic forces, charge bookkeeping, boundary/measure, potentials and raw quadrature diagnostics |
| `periodic-basis` | `dft-rs periodic-basis cell.xyz --cell "..." --pbc 1,1,0 --k-mesh 3,3,1 --basis sto-3g --overlap-cutoff 12` | `--basis-grid --grid-level 3 --basis-grid-cutoff 12 --basis-grid-derivatives` | Complete indexed matrix/derivative/grid/moving-weight rows and raw diagnostics; XYZ/cell Angstrom, explicit cutoffs Bohr |
| `periodic-gaussian-integral` | `dft-rs periodic-gaussian-integral centres.xyz --cell 5,5,20 --pbc 1,1,0 --gaussian-exponents .6,.4 --gaussian-angular "1,0,0;0,1,0"` | Three XYZ centres/two Gaussians: nuclear source last; four centres/four Gaussians: ERI; source charge/width, Ewald cutoffs, `--gaussian-tol`, `--gaussian-quadrature` | Same value/components, all centre gradients, units, model, boundary/measure, actual controls and raw quadrature/translation diagnostics |
| `periodic-coulomb` | `dft-rs periodic-coulomb cell.xyz --cell "..." --pbc 1,1,0 --basis sto-3g --charge 1 --charge-compensation gaussian --countercharge-center 0,0,0 --countercharge-width 2 --density-matrices density.csv` | Optional complete `k,row,col,real,imag` CSV; basis/k mesh/image cutoff, nuclear widths, common Ewald/compensation/quadrature controls; no SCF/XC controls | All indexed complex operators, components and centre derivatives; energy components and fixed-density gradients; particle-number, raw Hermiticity/translation/quadrature diagnostics and physical compensation |
| `periodic-xc` | `dft-rs periodic-xc cell.xyz --cell "..." --pbc 1,1,0 --xc pbe --charge 1 --alpha-density-matrices alpha.csv --beta-density-matrices beta.csv` | Complete `k,row,col,real,imag` file per spin; basis/k mesh/image cutoff, physical `--basis-grid-cutoff`, `--grid-level`, `--xc-operator-tol`, `--xc-grid-tol`; no SCF/force/temperature/Coulomb controls | All indexed complex XC potentials, weighted component energies, metric/grid/target electron counts, exact-exchange and nonlocal metadata, raw diagnostics and the explicit semilocal component model |
| `magnetic-response` | `dft-rs magnetic-response molecule.xyz --xc pbe --charge 1 --mult 2` | `--magnetic-gauge giao` or `common_origin`, `--magnetic-origin x,y,z` in Bohr; shared relative solver and molecular SCF controls | All indexed density/explicit-Fock/metric derivatives, raw number/projector/antisymmetry defects and common residual/model metadata |
| `nmr` | `dft-rs nmr molecule.xyz --xc r2scan` | Same magnetic and reference controls; orbital fixed-spin-occupation model | Complete nucleus/axis total, diamagnetic and paramagnetic entries in ppm and shared relative response/reference diagnostics |
| `periodic-field` | `dft-rs periodic-field cell.xyz --cell "..." --pbc 1,1,0 --field-wave .137,-.084,.067 --field-vector ".05,.03;-.02,.01;.01,-.04"` | Basis/charge/k mesh/image cutoff; Cartesian `--field-k-offset`, complex `--field-scalar`/`--field-vector`/`--field-frequency`; no SCF/functional controls | All analytic bare scalar/current/contact and Hamiltonian derivative entries, target meshes, physical complex mode fields and raw transversality defect |
| `dimensional-scf` | Self-consistent dimensional energy/free energy, density and physical Fock | `dft-rs dimensional-scf structure.xyz --cell "a;b;c" --pbc 1,0,0 --k-mesh 3,1,1 --xc pbe`; signed charge requires explicit `--charge-compensation gaussian --countercharge-center x,y,z --countercharge-width w` in 1D/2D | Full indexed spin/k AO density/Fock/metric and occupations; energy components, raw diagnostics, model, units and separate SCF/report convergence |
| `dimensional-field` | `dft-rs dimensional-scf structure.xyz --cell "a;b;c" --pbc 1,1,0 --field-transfer .231,0,0 --field-scalar .37,.22` | `--field-transfer f1,f2,f3` (reciprocal-lattice fractions on periodic axes, open components zero), `--field-scalar re,im`, `--field-vector "re,im;re,im;re,im"`, `--response-omega`, `--response-damping`, `--imaginary-frequency`, `--fixed-chemical-potential`, shared response solver controls | The dimensional SCF report followed by complex density/current, E/B amplitudes, shifted target mesh and raw continuity, quadrature, occupation and reference diagnostics |
| `energy` | `dft-rs energy molecule.xyz --xc pbe --format json` | Reference controls `--xc`, `--x/--c`, `--omega`, `--basis`, `--ecp`, `--charge`, `--multiplicity`, `--unrestricted`, `--guess-mix`, `--grid-level`, `--dispersion`, `--density-fit`, `--field`, `--solvent`, `--eps`, SCF controls, `--stabilize`, `--skala-model model.bin` (with `--xc skala`); `--format text\|json`, `--save report.json`. Ordinary SCF without `--format`/`--save` keeps its earlier text output; double hybrids use the same driver in text and JSON modes | The driver report as `key value` text or JSON. Double hybrids use this same entry: `xc="b2plyp"`, `xc="dsd-pbep86"`, `xc="dsd-blyp"`, `xc="pwpb95"` or `xc="wb97m2"`; the DSD names select the documented 2013 pairwise D3(BJ) recipe. SS/OS PT2 and dispersion are reported separately. Arbitrary parents accept `pt2_same_spin` and `pt2_opposite_spin`. Reports give parent, term energies and raw amplitude/Lagrangian residuals; [model and measured acceptance](response.md#double-hybrid-components-and-pt2-lagrangian).  omegaB97M(2) uses a distinct target on omegaB97M-V orbitals, scaled VV10 with b=10 and separate orbital/target energies. PWPB95 includes reoptimized PW/B95 and SOS PT2; all terms and raw residuals use the shared report. Custom parents retain their complete XC energy; `nonsemilocal_exchange_correlation_energy` (Hartree) separates a model-provider contribution from the semilocal component sum. VV10 remains in its separate nonlocal-correlation field. Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). GKS common-weight generalized PT2 energy reports the full-spin correlation, optional target correction and amplitude residual in Hartree.  [ROKS doubles-only PT2 energy and residuals](#restricted-open-shell-double-hybrid-energy). [Molecular D1, TS, MBD, VV09 and fixed-density vdW-DF controls, units and acceptance](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Molecular total-spin values and run logs](response.md#molecular-total-spin-streaming). |
| `mermin-free-energy` | `dft-rs mermin-free-energy molecule.xyz --electronic-temperature 6000 --format json` | `--xc`, `--basis`, SCF controls, optional `--gradient`; `--gibbs` also returns Gibbs RRHO from the same reference, with `--temperature K`, `--pressure Pa`, `--symmetry-number N` | Explicit Mermin free energy, internal energy, electronic entropy and occupations. For thermal ROKS/ROHF use fixed spin counts and shared spatial orbitals. For GKS/GHF use fixed total electron count and one chemical potential. Use the optional gradient with a compatible analytic provider. Equilibrium Gibbs RRHO uses the RKS/UKS route. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `gibbs-free-energy` | `dft-rs gibbs-free-energy molecule.xyz --temperature 298.15 --pressure 101325 --symmetry-number 1 --format json` | Same analytic reference Hessian; `--reference`, `--electronic-degeneracy n`; temperature K, pressure Pa | Gibbs energy in Hartree per molecule and RRHO breakdown including pV; requires a converged free molecule with positive internal frequencies. Fixed-spinor degeneracy defaults to one; provider/acceptance limits in response.md |
| `free-energy` | Compatibility alias: `dft-rs free-energy molecule.xyz --xc svwn5 --electronic-temperature 6000` | `--electronic-temperature K` (any molecular calculation accepts it), `--gradient`; reference and output controls | Free energy, internal energy, entropy, chemical potentials, occupations and optional gradient |
| `gradient` | `dft-rs gradient molecule.xyz --xc pbe --format json` | The same reference and output controls | Energy report plus `gradient` and `translation_residual` Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). The ordinary molecular report also returns `forces` as the negative Cartesian gradient in Hartree/Bohr. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). [Generalized double-hybrid gradients](api-guide.md#read-a-generalized-double-hybrid-gradient). |
| `streamed-gradient` | `dft-rs streamed-gradient mol.xyz --xc pbe0 --format json` | Reference controls | Energy and gradient with integral-direct builds [Measured restricted HF/PBE STO-3G all-component and public-route scope](validation-matrix.md#t139-integral-direct-molecular-gradient). |
| `charges` | `dft-rs charges molecule.xyz --xc pbe --format json` | The same reference and output controls | Charges, spin populations and the energy report |
| `natural-bond` | `dft-rs natural-bond molecule.xyz --xc pbe --format json` | Reference controls; `--nbo-spin auto\|total\|separate`, `--nbo-lewis-threshold` (fraction of the maximum occupancy), `--nbo-three-center true\|false`, `--nbo-perturbation-threshold` (Hartree) | Natural charges, spin populations and configurations, Wiberg indices, Lewis NBOs with hybrids, NLMOs and donor-acceptor energies (Hartree) per analysed density Use `--reference rks/uks/roks/gks`, `--nbo-spin-density true`, `--nbo-spin-density-points '[[0.34,-0.19,0.32]]'` or `--nbo-natural-orbitals true`. [Reference limits and density units](response.md#reference-selection-and-natural-density-orbitals). Use `--nbo-corresponding-orbitals true` for the collinear occupied pairs. |
| `igmh` | `dft-rs igmh dimer.xyz --xc wb97x-d --fragments 0,0,0,1,1,1 --igmh-points '[[0.31,-0.19,3.13]]' --format json` | `--igmh-points` or `--igmh-grid` with `--igmh-grid-shape`. Sampling coordinates use Bohr. `--igmh-gradient published` (default) or `hirshfeld`. `--igmh-atom-details true` returns atomic fields. | Total, inter- and intrafragment delta-g, signed density, spatial derivatives and raw residuals. [Model and measurements](response.md#r090-molecular-igmh). |
| `periodic-natural-bond` | `dft-rs periodic-natural-bond cell.xyz --cell "a;b;c" --k-mesh 3,3,3 --xc pbe --format json` | Periodic reference controls (`--pbc` for open axes), `--k-mesh` (at least 3 along periodic axes), the NBO controls of `natural-bond` | Home-cell natural charges, configurations, Wiberg bonds with cell offsets, Lewis NBOs and donor-acceptor energies per cell |
| `natural-shielding` | `dft-rs natural-shielding molecule.xyz --xc pbe --basis 6-31g* --format json` | Reference controls, `--gauge giao\|common`, `--gauge-origin`, `--nuclei`, the NBO controls of `natural-bond` | Shielding per nucleus (ppm) and its NLMO contributions |
| `natural-coupling` | `dft-rs natural-coupling molecule.xyz --xc pbe --nuclei 0,1 --nuclear-g 5.585694,1.404824 --format json` | Closed-shell reference controls, `--nuclei`, `--nuclear-g`, the NBO controls of `natural-bond` | Reduced couplings per pair, their terms and NLMO contributions; J in Hz with nuclear g factors |
| `natural-energy-decomposition` | `dft-rs natural-energy-decomposition dimer.xyz --xc pbe --fragments 0,0,0,1,1,1 --format json` | Reference controls, `--fragments` (fragment per atom), the NBO controls of `natural-bond` | Interaction energy and its charge-transfer, electrostatic, exchange-correlation and deformation terms (Hartree), per-fragment energies |
| `sapt-analysis` | `dft-rs sapt-analysis dimer.xyz --xc hf --sapt-variant sapt0 --fragments 0,0,0,1,1,1 --format json` | Reference controls, `--fragments` per atom, `--fragment-charges`, `--fragment-multiplicities`, `--sapt-variant` (`response`, `sapt0`, `sapt0-s2`, `ssapt0`, `sapt-30`, `sapt2+3`, `sapt-ks-fdds`), `--charge-transfer`, `--sapt-partition`. `--sapt-regularization-eta` (Bohr⁻²) enables R-SRS induction with `sapt0` or `sapt-ks-fdds`. `--sapt-correlation-terms` and `--sapt-relaxed-induction30` add the [correlated terms and cubic induction](response.md#correlated-sapt-terms-and-relaxed-cubic-induction) to `sapt0`. The response and KS-FDDS implementations accept `--asymptotic-shift-a`/`-b` (Hartree), `--rpa-dispersion` and `--casimir-polder-points`. | Hartree terms with separate A/B exchange-induction. Correlation and relaxed cubic induction use Hartree. `ssapt0` adds Hartree `delta_hf` and dimensionless `exchange_scale`. `sapt-30` reports [uncoupled third-order HF SAPT](response.md#uncoupled-third-order-intermolecular-sapt); `sapt2+3` reports [closed-shell HF correlation and cubic exchange](response.md#sapt23-with-full-intermolecular-exchange), including delta-HF(3). `sapt-ks-fdds` reports [KS induction and FDDS](response.md#ks-sapt-with-frequency-dependent-density-susceptibilities). See [SAPT(0) models and acceptance](response.md#sapt0-with-the-full-antisymmetrizer). |
| `dipole` | `dft-rs dipole molecule.xyz --xc pbe` | Reference controls; `--origin center_of_mass\|center_of_nuclear_charge\|zero\|x,y,z` (Bohr) | Dipole in e*Bohr and Debye, origin and convention  Double hybrids and custom PT2 parents use the same API; `parent_dipole`, `pt2_dipole_correction`, `target_dipole_correction` are e Bohr; `response_residuals` are relative, `amplitude_response_residuals` e Bohr and `response_number_residuals` electrons per atomic field. Includes noncanonical amplitude/orbital response and distinct-target exchange/VV10; integer zero-temperature reference. [Measured fits and model limits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls). [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). `--relativistic-method dirac` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `orbitals` | `dft-rs orbitals molecule.xyz --xc pbe` | Reference controls; `--coefficients`, `--density` | Orbital energies, occupations, frontier data; optional AO matrices Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `molden` | `dft-rs molden molecule.xyz --xc pbe --path molecule.molden` | `--path` (required); reference controls | Export summary |
| `orbitals-save` | `dft-rs orbitals-save molecule.xyz --xc pbe --path orbitals.json` | `--path` (required); reference controls | The saved orbital report |
| `molecular-multipole` | `dft-rs molecular-multipole molecule.xyz --max-rank 4` | Reference controls; `--max-rank n`, `--origin` | All three multipole conventions for ranks 0..n [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). |
| `quadrupole-polarizability` | `dft-rs quadrupole-polarizability molecule.xyz --format json` | Select the reference and functional. Optional `--origin`, `--frequency`, `--frequency-damping` and `--imaginary-frequency` | Five-mode Buckingham response in e²·Bohr⁴/Hartree, origin, frequency and residuals. RKS/UKS double hybrids include parent, target and PT2 spin channels at static and supported nonzero frequencies. Static GKS has a common PT2 weight. [Conventions](rust-api.md#electric-quadrupole-polarizability). `--relativistic-method dirac` also returns both mixed dipole/field-gradient blocks and residuals for all eight sources. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-polarizability` | `dft-rs multipole-polarizability molecule.xyz --multipole-rank 3 --format json` | `--multipole-rank` 1 to 5, default 3. Shared reference, functional, origin, frequency, damping and imaginary-frequency controls. | Racah response in e²·Bohr^(2 rank)/Hartree, signed frequency, mode order and raw residuals. Collinear double hybrids report parent, target and spin-resolved PT2 terms. The default center of mass follows molecular translation. [Conventions](rust-api.md#electric-multipole-polarizability). `--relativistic-method dirac` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `conceptual-dft` | `dft-rs conceptual-dft molecule.xyz --xc pbe --route integer` | `--route grand_canonical\|integer\|koopmans`, `--population mulliken\|lowdin`, `--electronic-temperature K` (selects `grand_canonical` by default), `--cation-multiplicity`, `--anion-multiplicity`, response controls | Route, global descriptors and condensed Fukui functions |
| `spin-conceptual-dft` | `dft-rs spin-conceptual-dft molecule.xyz --xc pbe --electronic-temperature 3000` | `--route grand_canonical\|integer`, `--population`, response controls | Spin chemical potential, spin hardness, charge-spin coupling and spin Fukui functions |
| `hessian` | `dft-rs hessian molecule.xyz --xc pbe --format json` | Reference and output controls; without `--format`/`--save` the earlier text command runs, except `--bfield` always selects the analytic common driver | The Hessian report Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Prescribed nonthermal populations use the full spectral moving-metric response, with `ensemble_response_residuals[3N]` (dimensionless), `ensemble_number_defects[3N][2]` (electrons/Bohr), the Hessian and APT. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc).  [Occupation-order diagnostics](rust-api.md#scf-occupation-order-diagnostics): per-spin occupied_virtual_gaps (Hartree) and negative-gap warning. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `frequencies` | `dft-rs frequencies molecule.xyz --reference rohf --xc hf --multiplicity 2 --format json` | `--vibrational-projection auto/eckart/cartesian`, `--electronic-degeneracy n`, `--temperature K`, `--pressure Pa`, `--symmetry-number n`; same analytic Hessian provider controls | cm^-1 frequencies, 1/sqrt(amu) mode patterns, km/mol IR or null without APT; fixed laboratory sources default to all 3N Cartesian modes and withhold free-molecule RRHO; source/acceptance limits in response.md |
| `ir-raman` | `dft-rs ir-raman molecule.xyz --xc pbe` | Reference controls, thermochemistry controls and `--response-tolerance`, `--response-max-iterations`, `--response-restart`, `--response-preconditioner-floor`, `--raman-frequency` (Hartree) | The frequency report plus analytic Raman activities and depolarization ratios [Measured restricted HF STO-3G dynamic Raman component and public-route scope](validation-matrix.md#t017-dynamic-raman-mode-derivatives). |
| `polarizability` | `dft-rs polarizability molecule.xyz --xc pbe --format json` | Reference and output controls | Tensor, isotropic value, anisotropy and Angstrom^3 values; named double hybrids and custom SS/OS parents use the same driver, with second-order parent orbitals, noncanonical PT2 amplitudes and distinct-target exchange/VV10 kernels (physical acceptance in progress). `parent_polarizability`, `pt2_polarizability_correction`, `target_polarizability_correction` and unscaled aa/ab/bb PT2 tensors use e^2 Bohr^2/Hartree. Six independent static components are expanded; `raw_asymmetry` is null. Pair residuals have shape [6,3] for Taylor orders 0,1,2 with units [Hartree,e Bohr,e^2 Bohr^2/Hartree]; fresh orbital residuals and projector/number/frame defects are separate. Dynamic and higher correlated properties remain pending Prescribed populations and positive electronic temperature use `response::ensemble::EnsembleSolver::polarizability` through this same entry. Reports add the population model, fresh `response_residuals[3]`, iteration counts and `number_defects[3][2]` (e Bohr/Hartree); all tensor columns are solved independently. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response). Static GKS with a common full-spin PT2 weight returns the total, parent, unweighted pair, weighted pair and optional distinct semilocal or hybrid target tensors in e^2 Bohr^2/Hartree. The six independent components use the stationary spinor energy Hessian, with raw orbital, projector, particle-number and pair diagnostics. [Conventions and measured scope](response.md#r060-molecular-multipoles-and-derivatives). The integer collinear CPKS route reports `response_absolute_residuals` in e Bohr, one maximum fresh occupied-virtual equation defect per Cartesian field. `--relativistic-method dirac` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-gradient` | `dft-rs multipole-gradient molecule.xyz --max-rank 2 --origin zero` | `--max-rank n`, `--origin`; reference controls | Moments and their analytic Cartesian gradients |
| `response-save` | `dft-rs response-save molecule.xyz --xc pbe --path response.json` | `--path`, `--nuclear false`, `--frequencies` (Hartree), `--frequency-damping`, response controls; reference controls | The response report; the file holds the same JSON The p126 strict-unit, zero-frequency static-limit and finite-frequency normal-dispersion checks pass; additional nuclear and frequency combinations remain separate. |
| `third-derivative-2n1` | `dft-rs third-derivative-2n1 mol.xyz --xc pbe --format json` | Response and reference controls | Cartesian cubic force constants `[i][j][k]` Arrays use Hartree/Bohr^3. The p104 baseline validates HF/PBE/PBE0 cubic tensors against complete mixed responses; its independent four-step SCF-energy derivative fits also pass (p=1.995439646..2.002158542, minimum R2=0.999998593470, maximum response residual 4.314e-16, maximum Richardson defect 1.965e-7 Hartree/Bohr^3). See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) |
| `cubic-fc-along` | `dft-rs cubic-fc-along mol.xyz --xc pbe --probe 0,0,1,0,0,-1 --format json` | `--probe` (one value per Cartesian coordinate); response and reference controls | Hessian derivative along the probe Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Arrays use Hartree/Bohr^3. The p104 baseline validates the PBE zero-field contraction against the complete tensor; this comparison alone does not establish an independent finite-difference derivative fit. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `quartic-fc-along` | `dft-rs quartic-fc-along mol.xyz --xc pbe --probe 0,0,1,0,0,-1 --format json` | `--probe`, `--second-probe` (default the same pattern); response and reference controls | Second Hessian derivative along the probes Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Arrays use Hartree/Bohr^4. The p104 baseline validates the PBE zero-field contraction against complete mixed responses; independent derivative and further-combination acceptance remain separate. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `excitations` | `dft-rs excitations molecule.xyz --xc b3lyp --basis def2-svp --nstates 8 --format json` | `--nstates`, `--td-method` (`tda` or `rpa`), `--td-spin` (`auto`, `singlet`, `triplet`, `spin-resolved`), `--td-tolerance`, `--td-max-iterations`, `--amplitudes`; reference controls | Excitation energies, oscillator and rotatory strengths, transition moments Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). `--td-nto true` adds paired particle/hole natural transition orbitals to the TDA states. [Conventions](response.md#natural-transition-orbitals). `--td-difference-density true` adds unrelaxed attachment and detachment in RKS/UKS/GKS. |
| `ecd-spectrum` | `dft-rs ecd-spectrum molecule.xyz --xc b3lyp --nstates 20 --broadening-ev 0.2 --format json` | `--broadening-ev` (required), `--points`, `--energy-window-ev min,max`; the `excitations` controls; reference controls | States, absorption and circular-dichroism curves, sum-rule integrals  Correlated PT2 pole and electric/magnetic transition-residue corrections use the same entry and spin controls; bounded full-component and public acceptance passes (see the detailed model scope). |
| `td-state-gradient` | `dft-rs td-state-gradient molecule.xyz --xc svwn5 --root 1 --format json` | `--root`, `--nstates`, `--td-tolerance`, `--td-max-iterations`, `--td-gradient-route z-vector` (or `path`, with `--td-method` and response controls) | Excited-state energy (Hartree) and gradient (Hartree/Bohr); native directional path subset measured, full public path-gradient acceptance pending Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `hyperfine` | `dft-rs hyperfine radical.xyz --multiplicity 2 --unrestricted --xc b3lyp --nuclear-g 1.4048236,5.5856946893,5.5856946893,5.5856946893 --format json` | `--nuclear-g` (one signed value per atom, comma separated), `--multiplicity`, `--unrestricted`; reference controls including a state-selecting `--field` when required | Contact, spin-dipolar and total tensors in MHz, principal values and column axes, raw tensor/eigensystem residuals and per-atom verdicts; isotope-independent per-g fields are always present; ECP nuclei report null all-electron results Explicit ROHF/ROKS retain common spatial orbitals. GHF/GKS require `epr_projection=single-kramers-pair` for the approximate odd-electron doublet model, including orbital-current A, spin expectation, spin-squared and metric diagnostics. These extensions are all-electron and common-origin; they do not spin-purify the determinant. Source-connected HF nuclear-A increment: `nuclear_order=0` (default) keeps static output; 1 adds gradients with null Hessian fields; 2 adds the full Hessian in `hyperfine_nuclear`. Optional zero-based `derivative_nuclei` selects target nuclei only; derivatives still span all 3N input Cartesian coordinates. Contact/dipolar/total `_per_g` fields are per signed nuclear-g convention; optional signed `nuclear_g` adds scaled fields. Native/Rust/CLI derivative units are MHz/Bohr and MHz/Bohr^2; ASE converts inverse length factors to its selected length units. Scope is all-electron integer UKS/UHF HF; PBE full6840 validation is running, and ROKS/GKS/ECP/PT2 nuclear-A are refused rather than replaced by a parent. Raw path coefficients retain derivative/k!, indexed 0..order; unavailable zero-order response diagnostics remain null. Original/inner linear residuals and reconstructed/source ratios are separate quantities. Exact adapter04 order mapping is measured; final consolidated artifact/public acceptance remains pending. |
| `nmr-shielding` | `dft-rs nmr-shielding mol.xyz --xc pbe --basis def2-svp --format json` | `--gauge giao\|common-origin`, `--gauge-origin x,y,z` (Bohr); response and reference controls | Shielding tensors and isotropic shieldings in ppm Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. With `relativistic_method="dirac"`, finite nuclear charge and magnetization radii in Bohr and optional `nuclei` select static four-component London-RMB shielding. Returns `shielding_tensors` in ppm with explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `nmr-spectrum` | `dft-rs nmr-spectrum mol.xyz --nmr-reference-elements 1,6 --nmr-reference-shieldings 31.8,188.1 --format json` | `--nmr-reference-elements`, `--nmr-reference-shieldings` (ppm), `--nmr-broadening` (ppm, full width), `--nmr-points`, `--nmr-margin` (ppm); controls of `nmr-shielding` | Chemical shifts (ppm) and the broadened spectrum of every referenced element Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless.  Exact mode: `nmr_spin_model=exact`, `nmr_couplings=true`, `nuclear_g`, optional `nuclear_spins` (twice I), `nmr_spin_max_dimension` (1024). Shared isotropic spin Hamiltonian; ppm line positions, integrated line weights, raw Hz and dimensionless eigen residuals, area defects; bounded acceptance in response.md. With `relativistic_method="dirac"`, supply finite nuclear charge and magnetic RMS radii in Bohr for the London-RMB electronic response. The spectrum uses all nuclei's shielding tensors in ppm and optional total indirect J in Hz. Raw response, eigensystem and area residuals follow the common spin Hamiltonian. See [four-component spectrum](response.md#four-component-isotropic-nuclear-spin-spectrum). |
| `spin-spin-couplings` | `dft-rs spin-spin-couplings mol.xyz --xc pbe0 --nuclear-g 5.5856946893,5.5856946893 --format json` UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `--nuclei` (atom indices, default all), `--nuclear-g` (one g factor per atom); response and reference controls | Reduced coupling tensors by term and the coupling constants `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. For the four-component total indirect tensor use `--relativistic-method dirac --nuclear-magnetic-rms-radii ...` with Bohr values. The report includes separate explicit-sector and orbital-relaxation tensors. [Four-component tests](response.md#four-component-indirect-nuclear-spin-coupling). |
| `fermi-contact-couplings` | `dft-rs fermi-contact-couplings mol.xyz --xc pbe0 --format json` UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `--nuclei`, `--nuclear-g`; response and reference controls | Fermi-contact coupling tensors and constants `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. |
| `exafs` | `dft-rs exafs cluster.xyz --exafs-absorber 0 --exafs-mfp 12 --exafs-scattering "29,1.0,0.5,0.1;29,8.0,0.3,2.0" --format json` | `--exafs-absorber`, `--exafs-s02`, `--exafs-mfp` (Bohr), `--exafs-rmax` (Bohr), `--exafs-scattering` (rows of Z, k in 1/Bohr, amplitude in Bohr, phase), `--exafs-k-range` (k_min,k_max,step in 1/Bohr), `--exafs-k-weight`, `--exafs-sigma2` (Bohr^2), `--exafs-r-points`; for double scattering `--exafs-scattering-angles` (rows of Z, scattering angle in degrees with 180 = backscattering, k, amplitude, phase), `--exafs-central-phase` (rows of k, twice the phase shift of the absorber), `--exafs-ms-sigma2` (Bohr^2) | Scattering paths, `chi(k)`, weighted signal and Fourier-transform magnitude The p121 closed-chain, right-angle and single-scattering-limit checks pass; no energy derivative is involved. |
| `xanes` | `dft-rs xanes mol.xyz --xanes-element 8 --nstates 10 --xanes-broadening 0.02 --format json` | `--xanes-element` (atomic number, K edge) or `--core-orbitals`, `--nstates`, `--td-spin`, `--xanes-broadening` (Hartree, full width), `--points`, `--amplitudes`, `--core-hole frozen` (or `delta-scf`); reference controls | Core excitation energies, oscillator strengths and the optional spectrum Relaxed core holes reuse the common physical-metric projector-IMOM SCF builder and preserve both spin labels. The p122 one-electron core-hole limit and HF water O 1s relaxation checks pass; this does not validate state-specific excited-state relaxation. |
| `nexafs` | `dft-rs nexafs mol.xyz --absorber-atoms 0 --nexafs-model cvs-tda --nexafs-broadening 0.025 --format json` | `--absorber-atoms` for the K edge or explicit `--core-orbitals-alpha` and optional `--core-orbitals-beta`, `--absorption-edge`, `--nexafs-model cvs-tda` or `relaxed-core-ion`, `--polarization`, `--line-shape`, `--nexafs-energy-window`, `--points` and reference controls | Lines and Cartesian oscillator-strength tensors, Gaussian or Lorentzian spectrum and core-hole tracking residuals. Energies and full widths are Hartree, dipoles e Bohr and oscillator density inverse Hartree. [Model limits and water results](response.md#r085-near-edge-xray-absorption-fine-structure). |
| `constrained-dft` | `dft-rs constrained-dft water.xyz --xc wb97x-d --density-constraints '[{"atoms":[0,1],"electrons":8.96}]' --optimize-geometry true --format json` | `--gradient true` for analytic forces, `--optimize-geometry true` for relaxation and `--constraint-electric-response true` for fixed-population static polarizability. The array also accepts explicit AO and spin operators. | Energy and multipliers in Hartree, gradients in Hartree/Bohr, static polarizability in e²·Bohr²/Hartree and raw constraint, orbital and response residuals. [Constraint schema and measured results](response.md#r083-constrained-density-functional-theory). |
| `dft-plus-u` | `dft-rs dft-plus-u molecule.xyz --xc wb97x-d --reference gks --hubbard-sites '[{"orbitals":[5,6],"u_eff":0.14}]' --format json` | Supply zero-based spherical AO indices. Optional `--hubbard-pairs` JSON objects with `sites` and `v` in Hartree activate intersite terms. A Liechtenstein site takes `model`, `u`, `j` and `integrals`. Supports integer RKS, UKS, ROKS and GKS. | Site and total energies and raw orbital residual in Hartree. Spin occupations in electrons. GKS reports real and imaginary spin-major occupation matrices. [Site schema and coverage](response.md#r087-molecular-dftu). |
| `pz-sic` | `dft-rs pz-sic molecule.xyz --xc wb97x-d --reference uks --format json` | Common fixed-geometry integer UKS controls. | Corrected and parent energies, occupied orbital self energies and raw orbital-gradient residual in Hartree. [Model and coverage](response.md#r088-molecular-pz-sic). |
| `flo-sic-fixed` | `dft-rs flo-sic-fixed h2plus.xyz --charge 1 --multiplicity 2 --xc wb97x-d --fod-alpha '[[0.73,0.04,0.13]]' --fod-beta '[]' --format json` | One fixed Cartesian descriptor in Bohr per occupied orbital of each real UKS spin channel. | Corrected and parent energies and parent orbital residual in Hartree, localized AO density and metric defects, and descriptor densities in electrons/Bohr³. [Model and coverage](response.md#fixed-descriptor-fermi-lowdin-sic). The common driver accepts `fod_gradient`, `fod_optimize` and `flo_self_consistent` for the real calculation. `fod_alpha_imaginary`/`fod_beta_imaginary` select complex descriptors with fixed parent orbitals. [Units, models and verification](response.md#fixed-descriptor-fermi-lowdin-sic). |
| `car-parrinello-md` | `dft-rs car-parrinello-md mol.xyz --md-steps 12 --md-timestep 2 --fictitious-mass 200 --format json` | `--md-steps`, `--md-timestep`, `--fictitious-mass`, `--nuclear-velocities` (Cartesian row per atom), `--nuclear-masses`, `--md-constraint-tolerance` and reference controls | Trajectory, forces, both kinetic energies, extended energy drift and constraint defects. Time and velocities are atomic units. Masses are electron masses, positions Bohr and energies Hartree. [Trajectory model and water tests](response.md#r084-car-parrinello-molecular-dynamics). |
| `magnetizability` | `dft-rs magnetizability mol.xyz --xc pbe --format json` | `--gauge-origin x,y,z` (Bohr); response and reference controls | Magnetizability tensors in atomic units and the isotropic value in 1e-6 cm^3/mol  Optional `response_order=1..4` and `magnetic_direction` (atomic magnetic field) add `magnetic_energy_coefficients` and actual factorial-restored `magnetic_energy_derivatives`, in Hartree along B(t)=t*direction, plus raw path/projector/stationarity residuals. Orbital zero-field collinear model; eight restricted HF/PBE/TPSS/PBE0 second/fourth magnetic energy fits pass; remaining tensor/reference/class acceptance is pending.  Optional `magnetic_directions` contains one through four Cartesian field directions. Returns the actual mixed energy derivative for B=sum_j t_j*d_j (dimensionless t_j), Hartree, plus raw response/projector/stationarity defects and polarization roundoff. Repeated and non-orthogonal directions are supported through the same London/common-origin field driver. Matching `electric_directions` optionally adds electric parts to every mixed parameter: E=sum t_j e_j, B=sum t_j b_j. `magnetic_mixed_derivative` then reports the joint energy derivative, Hartree per dimensionless parameters; the same analytic field/metric response is reused.  Static RKS/UKS double-hybrid orbital tensors use the shared complex orbital/PT2 field path for common-origin and London gauges. Parent, target, weighted PT2 and unscaled SS-alpha/OS/SS-beta tensors are reported separately with raw orbital, pair, metric and reference diagnostics. The full total energy is the same post-SCF model as the common energy driver. High-order correlated contractions remain explicitly unsupported. New tensor/public acceptance is pending. Explicit `reference=gks` or `reference=ghf` with `gauge=common-origin` selects the static all-electron spinor orbital tensor. The spinor report includes the fixed-density diamagnetic and relaxed paramagnetic terms and raw response, symmetry and imaginary defects. The finite-basis tensor depends on the origin. The rebuilt HSEsol ammonia-cation scope passes all eighteen independent four-step fits and public route parity. Maximum raw relative response residual 5.125e-11 and zero imaginary susceptibility defect. [Spinor model and acceptance](response.md#static-spinor-orbital-magnetizability-at-a-common-gauge-origin) With `relativistic_method="dirac"`, finite charge radii in Bohr and the London-RMB basis give static `magnetizability` in `e^2*bohr^2/m_e` and `magnetic_dipole` in atomic units. Reports give explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `magnetic-quadrupole` | `dft-rs magnetic-quadrupole mol.xyz --reference gks --magnetic-field 0.014,-0.019,0.023 --format json` | `--origin` (Bohr or named origin), `--magnetic-field` (Cartesian), `--magnetic-field-gradient` (symmetric traceless 3 by 3 matrix), optional spin axis and reference controls | Orbital, spin and total magnetic quadrupoles, magnetic dipole and raw SCF residual. Magnetic quantities use atomic units. Origin defaults to the centre of mass in Bohr. [Field convention and measured scope](response.md#r082-molecular-magnetic-quadrupole). |
| `franck-condon` | `dft-rs franck-condon mol.xyz --xc svwn5 --root 1 --fc-broadening 0.001 --format json` | `--root`, `--nstates`, `--fc-max-quanta`, `--fc-max-total`, `--fc-model` (vertical-gradient or vertical-hessian), `--temperature` (K), `--fc-max-hot`, `--fc-threshold`, `--fc-broadening` (Hartree, full width), `--points`; reference controls | Huang-Rhys factors, 0-0 energy, lines and the optional line shape The p103 H2 vertical-model finite-temperature normalization and zero-temperature vertical-Hessian sum rules pass; broader vibronic spectra remain pending. |
| `excited-optimization` | `dft-rs excited-optimization mol.xyz --xc svwn5 --root 1 --gtol 3e-4 --format json` | `--root`, `--nstates`, `--max-steps`, `--gtol` (Hartree/Bohr), `--step-max` (Bohr), `--history`; reference controls | Relaxed positions, energies and the optimization trajectory [Exact model, full-component acceptance and residuals](validation-matrix.md#t120-isolated-singlet-tda-optimization). Explicit `td_gradient_route=path` connects analytic nuclear-path gradients; new-route acceptance is pending the coordinated rebuild. [Spin/model limits and implementation stages](excited-state-extension.md). |
| `optical-activity` | `dft-rs optical-activity mol.xyz --frequencies 0.0773 --format json` | `--frequencies` (Hartree, comma separated), `--frequency-damping` (Hartree), `--gauge-origin x,y,z` (Bohr); response and reference controls | Rotation tensors, `beta` and the specific rotation per frequency; the model field distinguishes common-origin length gauge from modified velocity gauge with static subtraction. Nuclear optical-tensor derivative acceptance is tracked separately; periodic and correlated mixed dynamic response remain pending |
| `broken-symmetry-exchange` | `dft-rs broken-symmetry-exchange dimer.xyz --multiplicity 3 --site-a 0 --site-b 1 --format json` | `--site-a`, `--site-b` (atom indices from 0), `--unpaired` (N_A,N_B), `--multiplicity` (high-spin state); reference controls | Energies and `<S^2>` of both determinants and the exchange coupling |
| `vcd` | `dft-rs vcd mol.xyz --xc pbe --format json` | `--gauge-origin x,y,z` (Bohr); response and reference controls | Atomic axial tensors, frequencies, dipole and rotational strengths; common-origin PT2 AAT and correlated Hessian/APT: 2,025/2,250 component fits and 15 public tests pass; learned/nonlocal and London pair paths remain separate |
| `spin-susceptibility` | `dft-rs spin-susceptibility mol.xyz --unrestricted --electronic-temperature 3000 --format json` | `--unrestricted`, `--electronic-temperature` (K, positive); response and reference controls | Spin susceptibility and spin stiffness  `spin_g_factor` defaults to the free-electron value. Reports include `bare_spin_susceptibility`, `bare_spin_stiffness`, `spin_enhancement`, `spin_density_response[spin][AO][AO]`, raw response/number and reference commutator residuals. Magnetization convention M=-g(N_alpha-N_beta)/4; density response is per atomic magnetic field. Fixed total number, spin-only longitudinal response; GGA/meta-GGA/hybrid validation and remaining combinations are tracked in CSV. |
| `g-tensor` | `dft-rs g-tensor radical.xyz --multiplicity 2 --unrestricted --format json` | `--nuclear-order 0/1/2` (default zero), `--gauge london` (or static `common-origin`), `--gauge-origin x,y,z` (Bohr), `--soc-charges` (one effective charge per atom), `--multiplicity`, `--unrestricted`; response and reference controls | g tensor, isotropic g and the shift contributions in ppm The p120 nondegenerate water-cation origin/translation and one-centre-limit checks pass; the independent mixed-energy derivative fit remains pending. Explicit ROHF/ROKS use the constrained common-orbital response. GHF/GKS require `epr_projection=single-kramers-pair` and return pair-axis `g_tensor`, invariant `g_gram`, spin and angular/operator diagnostics. These extensions support the all-electron common-origin one-electron model; London response and higher multiplet projection remain unavailable. Separate `nuclear_order=1/2` uses integer all-electron HF/UHF and the one-electron Pauli London model: `g_gradient[3N,3,3]` and `g_hessian[3N,3N,3,3]` are actual derivatives in `1/bohr` and `1/bohr^2`; the Hessian is null for order one. ASE converts these to inverse Angstrom powers. Mass/gauge/spin-orbit parts and real/mixed diagnostics remain separate. Nuclear DFT, RO/spinor, ECP/ensemble, nonzero electric field and stability-following requests remain explicit errors; explicit zero field is allowed. Static order zero keeps the preceding capabilities. No two-electron SOC, picture-change or spectral-accuracy claim follows. |
| `spin-spin-zfs` | `dft-rs spin-spin-zfs triplet.xyz --xc hf --reference uhf --multiplicity 3 --nuclear-order 2 --format json` | `--nuclear-order 0/1/2` (default0), explicit `--reference uhf` only; omitted reference selects UHF. Standard basis/SCF/response/resource/log controls. | Direct spin-spin D in Hartree; nuclear gradient/Hessian in Hartree/Bohr and Hartree/Bohr^2, atom-major xyz. Unrequested blocks null. SOC, spin purification, total D, ECP, ensembles and semilocal derivatives unsupported. The installed shared artifact passes all 25 public cases, including both CLI names, partial logs, nonconvergence and unsupported-control errors. Separate backend evidence retains 810 bounded fits; SOC/full D remain excluded. |
| `roks` | `dft-rs roks radical.xyz --multiplicity 2 --xc pbe --format json` | `--multiplicity`, `--gradient true\|false`; reference controls | Restricted open-shell energy and gradient |
| `roks-hessian` | `dft-rs roks-hessian radical.xyz --multiplicity 2 --xc pbe --format json` | Response solver controls (`--response-tolerance`, ...); reference controls | Hessian, gradient, harmonic frequencies and the raw defects of the path response [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `roks-polarizability` | `dft-rs roks-polarizability molecule.xyz --multiplicity 2` or `polarizability --reference roks` | Shared molecular and response controls; constrained common orbitals; same generic Hessian as `dynamic-polarizability --reference roks` at zero frequency | Complete static Cartesian tensor, dipole and raw reference/response residuals; atomic units; see R001 |
| `dc` | `dft-rs dc mol.xyz --xc pbe --dc-buffer 8.0 --dc-smearing 0.001 --format json` | `--dc-fragments` (one fragment index per atom, comma separated), `--dc-buffer` (Bohr), `--dc-smearing` (Hartree), `--gradient`; reference controls | Divide-and-conquer free energy, internal energy, chemical potentials, subsystem sizes and gradient The molecular HF/PBE/PBE0 incomplete-buffer partition-gradient four-step fits and complete-buffer Mermin limits pass in the p102 baseline. `energy` is the electronic Mermin free energy, `internal_energy` the internal energy (Hartree); Cartesian `gradient` is its partition-relaxed derivative (Hartree/Bohr). Fixed partition membership is assumed. Public, periodic and other-class acceptance remain separate. See [raw fits](response.md#phase-b-molecular-divide-and-conquer-gradient-acceptance). |
| `mrsf` | `dft-rs mrsf ethylene.xyz --xc pbe0 --nstates 4 --format json` | `--nstates`, `--mrsf-coupling mrsf\|ci`, `--multiplicity 1\|3`; reference controls | Singlet and triplet states with energies, `<S^2>` and oscillator strengths |
| `mrsf-state` | `dft-rs mrsf-state ethylene.xyz --xc pbe0 --mrsf-state S1 --format json` | `--mrsf-state` (S0), `--mrsf-coupling`, `--spin-flip-kernel`, response solver controls; reference controls | Gradient and vertical energy of the state, raw path diagnostics |
| `mrsf-hessian` | `dft-rs mrsf-hessian ethylene.xyz --xc pbe0 --mrsf-state S1 --format json` | Controls of `mrsf-state` | Hessian, gradient and harmonic frequencies of the state |
| `spin-orbit` | `dft-rs spin-orbit radical.xyz --multiplicity 2 --unrestricted --format json` | `--spin-axes "0,0,1"` (rows of directions), `--soc-charges` (one effective charge per atom), `--spin-orbit-self-consistent`; reference controls | Spinor energies and the spin-orbit energy Set `reference=gks` (`--reference gks` in the CLI) to include the generated one-electron operator in the spin-covariant SCF. `self_consistent_energies` and `self_consistent_orbital_residuals` are paired per axis in Hartree. |
| `spinor-polarizability` | `dft-rs spinor-polarizability radical.xyz --multiplicity 2 --unrestricted --frequencies 0.05,0.1 --format json` | `--frequencies` (Hartree), `--frequency-damping` (0), `--soc-scale` (1), `--soc-charges`, `--spin-axis` (0,0,1), response solver controls; reference controls | Real and imaginary polarizability tensors and the axial vector of the antisymmetric part per frequency; energies of the spinor reference |
| `magnetic-anisotropy` | `dft-rs magnetic-anisotropy complex.xyz --multiplicity 3 --unrestricted --format json` | `--spin-axes "1,0,0;0,1,0;0,0,1"`, `--soc-charges`; reference controls | Band and anisotropy energies per spin axis |
| `magnon` | `dft-rs magnon radical.xyz --multiplicity 3 --unrestricted --xc pbe0 --nstates 5 --format json` | `--nstates`; reference controls | Spin-lowering and spin-raising excitation energies |
| `mcd` | `dft-rs mcd mol.xyz --format json` | Default reference/TDA controls; or `--mcd-model thermal-manifold --temperature K --bfield x,y,z` with `--mcd-thermal-hamiltonian`, three `--mcd-thermal-magnetic-{x,y,z}` and three `--mcd-thermal-helicity-{x,y,z}` real symmetric JSON matrices. Thermal model consumes no SCF controls. | Canonical A/B or explicitly supplied thermal-manifold C/mixing; Supplied-real-manifold public artifact acceptance passes; New source models: `thermal-complex-manifold` retains Hermitian real/imaginary quadratures; `molecular-casci-soc` generates spin-complete CASCI/SOC states in DFT/HF orbitals and fixed/isotropic thermal cross sections. CASCI is not TDDFT; scoped shared Rust/Python/ASE/both-CLI artifact02 acceptance and19 actual molecular/complex replays pass. Moving SOC, quadratic magnetic completion and measured-spectrum agreement remain open. Source follow-up adds explicit `mcd_magnetic_basis=london-linear-frozen-soc`, first-order London metric/Coulomb/optical completion, and per-state spin expectation/S-squared diagnostics; scoped linked acceptance now passes13 Rust cases and46 Python/ASE cases per CLI, with source1044-fit matrices reproduced exactly by the public core. Moving SOC and general relativistic/ROKS/GKS RPA remain separate. |
| `td-hessian` | `dft-rs td-hessian mol.xyz --xc svwn5 --root 1 --format json` | `--root`, `--nstates`, `--td-method`, `--td-spin`, `--td-tolerance`, `--td-max-iterations`; response and reference controls | Hessian (Hartree/Bohr^2), gradient (Hartree/Bohr), frequencies (cm^-1); native directional subset measured, full tensor/public acceptance pending Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `td-ecd-giao` | `dft-rs td-ecd-giao mol.xyz --xc pbe0 --td-method rpa --nstates 5 --format json` | `--td-method` (tda), `--td-spin`, `--nstates` (5), `--gauge-origin` (0,0,0; the London value does not depend on it), response solver controls; reference controls | London, common-origin and velocity rotatory strengths per state The p117 origin, translation, mirror and common-origin TDA-limit checks pass; the independent four-step London optical-rotation residue fit remains pending. |
| `roa` | `dft-rs roa mol.xyz --xc svwn5 --frequencies 0.0856 --format json` | `--frequencies` (Hartree, comma separated), `--gauge` (velocity or length), `--gauge-origin x,y,z` (Bohr); response and reference controls | Tensor derivatives, invariants and circular intensity differences per frequency and mode |
| `hypermagnetizability` | `dft-rs hypermagnetizability mol.xyz --xc pbe0 --format json` | `--gauge-origin x,y,z` (Bohr); response and reference controls | Hypermagnetizability, polarizability and magnetizability tensors Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `cotton-mouton` | `dft-rs cotton-mouton mol.xyz --xc pbe0 --temperature 298.15 --format json` | `--temperature` (K), `--gauge-origin x,y,z` (Bohr); response and reference controls | Cotton-Mouton constant with its two terms and the three tensors Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `magneto-optical` | `dft-rs magneto-optical mol.xyz --xc pbe0 --frequency 0.0856 --format json` | `--frequency` (Hartree), `--gauge` (london or common-origin), `--gauge-origin x,y,z` (Bohr); response and reference controls | Faraday tensor, its invariant and the polarizability at the frequency Optional `frequency_damping` (Hartree) uses the causal complex equation. Reports retain both parts of the molecular polarizability and the field or spin-orbit derivative, plus the raw relative and reference residuals. |
| `bfield` | `dft-rs bfield mol.xyz --xc pbe --bfield 0,0,0.01 --format json` | `--bfield` (atomic units), `--spin-zeeman`, `--bfield-series-order`, `--bfield-initial-guess auto/core/zero-field`; reference controls | Energy and orbital energies in the field  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` (Hartree/e/Bohr) adds a simultaneous uniform electric field, including nuclear coupling and restoration of the charged-system laboratory energy origin. `electric_field` records the applied field. |
| `bfield-gradient` | `dft-rs bfield-gradient mol.xyz --xc pbe --bfield 0,0,0.01 --format json` | `--bfield` (atomic units), `--spin-zeeman`, `--bfield-series-order`, `--bfield-initial-guess auto/core/zero-field`; reference controls | Energy and gradient in the field  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` adds the same electric scalar potential to the London SCF and analytic gradient. For charge Q, `gradient_translation_target=-Q*electric_field` (Hartree/Bohr); `translation_residual` measures the difference from that physical target. |
| `magnetoelectric` | `dft-rs magnetoelectric mol.xyz --xc pbe --multiplicity 2 --unrestricted --format json` | `--soc-charges`, `--gauge`, `--gauge-origin`; response and reference controls | Magnetoelectric tensor per spin axis |
| `periodic-energy` | `dft-rs periodic-energy cell.xyz --cell "0,2.04,2.04;2.04,0,2.04;2.04,2.04,0" --k-mesh 2,2,2 --xc pbe --forces` | Periodic reference controls (`--k-mesh`, `--overlap-cutoff`, `--exchange-cutoff`, `--grid-level`, `--grid-shell`, `--hartree-compact-exponent`, `--electronic-temperature`, SCF controls), `--forces`, `--stress`, `--dispersion none\|d2\|d3\|d3bj`, `--disp-cutoff`, `--cn-cutoff`, `--three-body`, `--abc-cutoff`; `--cell` Angstrom | Energy report, optional forces and stress, `dispersion_energy` 3D `dispersion="d2"` uses published functional and H-Kr atom parameters, a fixed pair cutoff in Bohr, and no ATM term. The dispersion energy is Hartree per cell, its forces Hartree/Bohr, and its stress Hartree/Bohr^3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion). Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.  VV10: `vv10`, `vv10_b`, `vv10_c`, `vv10_cutoff` (Bohr); named -V parents attach their defining term. Energy components separate semilocal/nonlocal correlation. SCF and moving Gamma kernel connected; physical/public acceptance and finite-q/frequency combinations remain pending. Explicit reference=roks/rohf source now connects an integer Gamma common-spatial-orbital reference, with closed beta subspace inside alpha, fresh physical Focks and raw orbital/metric/number diagnostics. HF/PBE source-local open-shell references and all six orbital-rotation four-step fits pass; closed-shell energies match existing periodic RKS within6.7e-16 Hartree. Core initialization/centered CDIIS, three-dimensional Gamma and zero temperature only; ROHF requires an HF provider. Finite k, thermal, shifted/damped initialization, additional nonlocal/PT2 terms and response above first order remain refused. Full-rank Gamma source now also connects analytic nuclear gradients/forces and affine strain/stress. The actual bacf85/2b60bd successor passes canonical Rust HF/PBE references, all six independent four-step orbital-rotation fits (p1.99919..2.12830, minimum R2 .999114), 11 unique public tests and two alias-CLI SCFs. Native/ASE/CLI raw common C, spin densities, physical Focks, units and residuals are retained; independent reconstruction gives common residual at most2.64e-15 Hartree. Those bacf85/2b60bd checks certify reference energy/interfaces. The later full-rank HF/PBE first-derivative subset passes all36 Cartesian/affine-strain four-step fits on the unchanged1e-13 energy,1e-11 density and1e-10 orbital gates. PBE needs a converged localized XC AO radius: xc_ao_cutoff=16 Bohr agrees with20 on this fixture; the original6-Bohr cutoff-crossing failures remain retained. Native gradient/forces use Hartree/Bohr and stress Hartree/Bohr^3; ASE converts lengths and derived units. The b895d970/80b95343 shared core passes15 unique public cases across preserved attempts and all8 native/ASE/both-CLI raw reports (288 force/gradient/stress/strain entries); maximum route difference3.97e-15 and difference from accepted analytic references2.53e-15. One false watchdog stop and one corrected metadata-stub failure remain retained. No new installed-wheel acceptance is claimed; finite-k, geometry relaxation, higher response, general GKS derivatives and material/basis/grid convergence are not inferred. Use damping=0, no_adiis=true for this core/CDIIS route. ASE construction additionally sets use_adiis=False; disabled ADIIS omits only its automatic cutoff, while explicit cutoff controls remain visible to backend validation. Conventional RKS/UKS paths retain their own existing scope. Field-free generalized increment: `reference=gks` or `ghf` on `periodic-energy` uses the complex spinor reference with optional `forces=true` and `stress=true`, without a collinear surrogate. The source-connected 3D integer all-electron/full-AO-rank stress is the complete same-gauge Coulomb/kinetic/complex-exchange/moving-XC/Pulay affine derivative divided by volume. HF/GHF and HF/SVWN5 GKS have measured all-nine-component acceptance; PBE/meta-GGA/hybrid stress and general higher response remain separate gates. `strain_derivative` and its component tensors use Hartree, `stress`/`stress_voigt` use Hartree/Bohr^3 before ASE conversion, and `pressure=-trace(stress)/3`. The ordered raw tensor and `stress_antisymmetric_residual` are retained; the Voigt view selects xx,yy,zz,yz,xz,xy without silently projecting away spatial-grid torque. Additional spin/SOC/field, finite-temperature and discarded-rank derivative sectors remain refused. See R067 in response.md for exact artifact scope, finite-grid limits and retained failures. [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free all-electron 1D/2D RKS and UKS use the dimensional finite-k reference for optional forces and stress. Field-free finite-k ROKS uses fixed integer ranks of common complex spatial orbitals on each mesh point, with separate real/imaginary matrices and raw metric, conjugation and orbital residuals. The dimensional RKS/UKS VV10 energy route supplies the reconverged oracle for all 36 force and strain fits. [Verification](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the total energy and requested forces. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `stress` | `dft-rs stress cell.xyz --cell ... --k-mesh 2,2,2 --format json` | Periodic reference controls, `--dispersion`, `--disp-cutoff`, `--cn-cutoff` | Stress tensor (Hartree/Bohr^3 and GPa), pressure, strain derivative and its terms, forces The dedicated GKS/GHF route shares the field-free 3D Gamma spinor derivative, units and diagnostics of `periodic-energy` with `stress=true`. See [analytic stress](periodic.md#11-analytic-stress-and-cell-relaxation). [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free RKS, UKS, fixed-integer ROKS and GKS also report 1D and 2D ordered affine strain with length or area normalization, active-subspace pressure and raw Coulomb/SCF residuals. The field-free scalar ECP force and affine stress include separate translated-centre terms in Hartree/Bohr and Hartree. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Field-free 3D RKS/UKS double hybrids use the relaxed Born-von Karman orbital and pair response at fixed reduced mesh, including the uniform Ewald background and monopole strain for charged cells. Ordered target and PT2 derivatives and raw solver residuals are reported. [Validation scope](response.md#r026-stress-elastic-high-elastic). Scalar ECP uses retained projector images on the first-order correlated lattice path. [ECP scope and numerical evidence](response.md#scalar-ecp-on-the-first-order-correlated-lattice-path). Neutral 3D collinear double hybrids accept a fixed Cartesian Berry bias in Hartree/e/Bohr. [Field convention and validation](response.md#correlated-stress-at-fixed-cartesian-electric-bias). Field-free RKS/UKS variational VV10 uses `vv10_cutoff` in Bohr. Conventional 3D finite-k force and stress have a verified NaH subset. The 1D RKS omegaB97X-V wire and 2D UKS omegaB97M-V slab use length or area normalization. Both pass all 18 force and ordered strain fits and the CLI, native Python and ASE comparisons. [Derivative scope and residuals](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the force and strain derivatives. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `opt-fixed-cell` | `dft-rs opt-fixed-cell cell.xyz --cell ... --gtol 3e-4 --format json` | Periodic reference controls, `--max-steps`, `--gtol`, `--step-max`, `--history` | Relaxed positions (Bohr), forces, trajectory The Gamma 3D ROKS objective includes geometry-only dispersion and its gradient. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). |
| `cell-opt` | `dft-rs cell-opt cell.xyz --cell ... --pressure-gpa 0 --stress-tol 5e-6 --format json` | Periodic reference controls, `--max-steps`, `--gtol`, `--stress-tol`, `--pressure-gpa`, `--step-max`, `--history` | Relaxed cell rows and positions (Bohr), enthalpy, stress, trajectory |
| `phonons` | `dft-rs phonons cell.xyz --cell ... --k-mesh 2,2,2 --format json` | Periodic reference controls, `--response-tolerance` and the other response controls | Gamma-point force constants (symmetrized and raw) and frequencies; `--xc hf`, hybrids and `wb97xd` are accepted for insulators Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). The collinear Gamma XC Hessian includes the meta-GGA kinetic-energy-density channel with finite-k electronic occupations. Independent TPSS force and XC operator fits and installed ECP parity are pending. RKS omegaB97X-V scalar-ECP NaH passes all 21 analytic Hessian directions at electronic Gamma. [VV10 evidence](response.md#variational-vv10-gamma-hessian). The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). |
| `phonon-bands` | `dft-rs phonon-bands cell.xyz --cell ... --k-mesh 4,4,4 --supercell 2,2,2 --path "0,0,0;0.5,0,0.5" --dos-mesh 8,8,8` | `--supercell`, `--path`, `--points`, `--labels`, `--dos-mesh`, `--dos-sigma`, `--dos-points`; periodic reference and response controls | Phonon dispersion and DOS (cm^-1)  Optional `wavevector_direction` (Cartesian 1/Bohr) with `response_order` 0..4 returns mass-weighted and unweighted Fourier coefficients along q(t)=q+t*d. `phonon_phase=cell` uses R and `atomic` uses R+tau_final-tau_initial. Raw Hermiticity is measured without projection; supercell interpolation and missing long-range/electronic flexoelectric terms remain explicit. `lattice_flexoelectric=true` with `wavevector_direction` and order 2..4 adds fixed-mass-center internal relaxation at Gamma. Born charges are computed on the primitive reference or supplied in `born_charge_rows` (3N by 3). `born_lattice_flexoelectric` is the Born-charge lattice part, e/Bohr, contracted twice with the normalized direction; electronic/dynamic-multipole terms remain separate. Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `phonon-thermo` | `dft-rs phonon-thermo cell.xyz --cell ... --k-mesh 4,4,4 --supercell 2,2,2 --q-mesh 8,8,8 --temperatures 300,600` | `--supercell`, `--q-mesh`, `--temperatures`, `--mode-cutoff-cm1`; periodic reference and response controls | Harmonic free energy, entropy, heat capacity per primitive cell Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `electron-phonon` | `dft-rs electron-phonon cell.xyz --cell ... --k-mesh 8,8,8 --supercell 2,2,2 --electronic-temperature 3000 --mu-star 0.1` | `--supercell`, `--electron-smearing`, `--phonon-smearing-cm1`, `--mode-cutoff-cm1`, `--mu-star`, `--a2f-points`, `--eliashberg-cutoff-cm1`, `--eliashberg-t-min-k`, `--gap-temperatures-k`, `--pade-points`; periodic reference and response controls | lambda, a2F, omega_log, critical temperatures (McMillan, Allen-Dynes, Eliashberg) `scdft_pairing=true` adds `scdft_inputs`, static K (Hartree) and dimensionless Z. Supply `screened_pair_coulomb` or choose `scdft_screening=rpa` (thermal head) or `rpa-neutral` (fixed-number Gamma body), with `response_reciprocal_cutoff` in inverse Bohr. Generated Coulomb screening covers the complete 3D k-point difference mesh of a spin-degenerate finite-temperature reference. The phonon contribution retains the commensurate Gamma-supercell scope. [Models and diagnostics](periodic.md#static-microscopic-scdft-coulomb-screening). |
| `quasi-harmonic` | `dft-rs quasi-harmonic cell.xyz --cell ... --k-mesh 4,4,4 --supercell 2,2,2 --q-mesh 8,8,8 --scales 0.98,0.99,1,1.01,1.02` | `--supercell`, `--q-mesh`, `--scales`, `--temperatures`, `--mode-cutoff-cm1`, `--pressure-gpa`; periodic reference and response controls | V(T), B_T, thermal expansion, heat capacities, Grueneisen parameter, Gibbs energy at the pressure, mode Grueneisen parameters |
| `elastic` | `dft-rs elastic cell.xyz --cell ... --k-mesh 4,4,4 --format json` | Periodic reference controls, `--response-tolerance` and the other response controls | Clamped- and relaxed-ion elastic constants (GPa), strain Hessian, internal strain, force constants Uniformly compensated charge includes the analytic Ewald monopole strain Hessian. The 3D collinear scalar ECP path includes affine and internal-strain terms. Charged and scalar ECP numerical acceptance is tracked in the [response ledger](response.md#r026-stress-elastic-high-elastic). Includes the geometry-only CHG and D3 strain, mixed and nuclear curvatures before ionic relaxation. The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). [Elastic report conventions and strain checks](api-guide.md#choose-elastic-constants-and-strain-diagnostics). |
| `elastic-fd` | `dft-rs elastic-fd cell.xyz --cell ... --steps 0.01,0.005 --relax-ions true --format json` | Periodic reference controls, `--steps`, `--relax-ions`, `--gtol` | Finite-difference elastic constants per step and their Richardson value (validation only) |
| `elastic-moduli` | `dft-rs elastic-moduli cell.xyz --cell ... --k-mesh 4,4,4 --ions relaxed --coefficients birch` | `--ions`, `--coefficients`; periodic reference and response controls | Voigt, Reuss and Hill moduli (GPa), wave speeds, Debye temperature |
| `christoffel` | `dft-rs christoffel cell.xyz --cell ... --directions "1,0,0;1,1,1"` | `--directions`, `--ions`, `--coefficients`; periodic reference and response controls | Phase velocities (m/s) and polarizations |
| `material-card` | `dft-rs material-card cell.xyz --cell ... --k-mesh 4,4,4 --card-path card.csv` | `--rotation`, `--card-path`, `--card-format`, `--ions`, `--coefficients`; periodic reference and response controls | The material card (and the written file) |
| `equation-of-state` | `dft-rs equation-of-state cell.xyz --cell ... --k-mesh 4,4,4 --scales 0.97,0.98,0.99,1,1.01,1.02,1.03` | `--scales`, `--relax-ions`, `--gtol`; periodic reference controls | V0, E0, B0 (GPa), B0', analytic against fitted pressures |
| `dielectric-tensor` | `dft-rs dielectric-tensor cell.xyz --cell ... --k-mesh 6,6,6 --format json` | Periodic reference and response controls | Electronic dielectric tensor; HF water continuum DFPT dilute-limit fit: p=2.970392240 in 1/L, R^2=0.999997818686, maximum raw field residual 3.722e-14; no HF Born-charge claim Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `born-charges` | `dft-rs born-charges cell.xyz --cell ... --k-mesh 6,6,6 --format json` | Periodic reference and response controls | Born charges (e), electronic dielectric tensor and raw response residuals; the common first-order-only request avoids unrelated force-constant work Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `piezoelectric-tensor` | `dft-rs piezoelectric-tensor crystal.xyz --cell "a;b;c" --k-mesh 4,4,4 --format json` | Periodic reference and response controls (`--k-mesh`, `--overlap-cutoff`, `--response-tolerance`); `--relaxed-ion false` for the clamped-ion tensor only | Clamped- and relaxed-ion proper piezoelectric tensors in e/Bohr^2 and C/m^2; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The relaxed-ion route includes CHG or D3 in the internal relaxation. |
| `berry-polarization` | `dft-rs berry-polarization cell.xyz --cell ... --k-mesh 6,6,6 --berry-mesh 12,6,6 --format json` | `--berry-mesh`; periodic reference controls | Reduced and Cartesian polarization, polarization quanta, string diagnostics |
| `crystal-ir` | `dft-rs crystal-ir cell.xyz --cell ... --k-mesh 6,6,6 --directions "1,0,0;1,1,1"` | `--directions`, `--mode-cutoff-cm1`; periodic reference and response controls | Transverse and LO frequencies, IR intensities, static dielectric tensor; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The Gamma Hessian includes CHG or D3. The report gives raw Hessian and Fock-image defects and the field/displacement solve residuals. |
| `static-dielectric` | `dft-rs static-dielectric cell.xyz --cell ... --k-mesh 6,6,6` | As `crystal-ir` | Static and electronic dielectric tensors; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Uses the same dispersion-corrected Gamma Hessian and response diagnostics as crystal IR. |
| `electronic-structure` | `dft-rs electronic-structure cell.xyz --cell ... --k-mesh 4,4,4 --coefficients true --format json` | Periodic reference controls; `--coefficients` defaults to false | SCF bands, populations; optional complex Bloch coefficients `[k][AO][band]`, overlap/Fock/per-spin density matrices `[k][AO][AO]`, per-k raw consistency residuals and spin weights. 3D collinear reference; [units, phase convention and remaining combinations](response.md#complex-bloch-orbital-coefficient-export). |
| `effective-mass` | `dft-rs effective-mass cell.xyz --cell ... --k-mesh 4,4,4 --k-point 0,0,0 --band 3 --direction 1,1,0` | `--k-point`, `--band`, `--direction`, `--spin`, `--degeneracy-tolerance`; periodic reference controls | Velocity, inverse effective-mass tensor, branch masses |
| `boltzmann-transport` | `dft-rs boltzmann-transport cell.xyz --cell ... --transport-k-mesh 24,24,24 --transport-temperatures-k 300 --transport-mu-shifts 0 --scattering-table "-1,5;0,12;1,20"` | Periodic reference controls; exactly one of `--transport-mu-shifts` and `--carrier-concentrations-cm3`; optional `--scattering-table` and `--transport-frequencies-hartree`; table excludes `--relaxation-time-fs` | Lifetime-weighted DC and optional retarded electric and thermoelectric conductivity; SI units, model and energy-origin convention in report |
| `hall-transport` | `dft-rs hall-transport cell.xyz --cell ... --transport-k-mesh 24,24,24 --transport-temperatures-k 300 --carrier-concentrations-cm3 -1e20 --scattering-table "-1,5;0,12;1,20" --transport-frequencies-hartree 0,0.02,-0.02 --format json` | Transport controls, `--hall-onsager-tolerance`; optional table plus frequencies, or legacy paired frequencies and `--relaxation-time-fs`; semilocal, hybrid and ECP band-operator controls | Physical Hall derivative with table, legacy derivative per tau squared otherwise; complex Seebeck, Peltier and electronic thermal conductivity; Hall coefficient, SI dynamic tensors and unprojected residuals |
| `coherent-transport` | `dft-rs coherent-transport cell.xyz --cell ... --transport-k-mesh 8,8,8 --transport-frequencies-hartree 0,0.02,-0.02 --format json` | `--transport-temperature-k`, `--transport-mu-shift`, `--scattering-hartree`; periodic reference controls including hybrid coefficients through the provider and ECP bases | Complex intra/interband and antisymmetric conductivity, raw residuals, S/m and exp(-i omega t) conventions |
| `quantum-hall` | `dft-rs quantum-hall cell.xyz --cell ... --topology-mesh 8,8 --topology-plane-axes 0,1 --format json` | `--topology-fixed-k`; zero-temperature insulating periodic reference controls | Occupied-multiplet Chern invariant and slice Hall conductance (S); sampled gap, raw integer and link diagnostics; refine topology mesh |
| `wannier-flow` | `dft-rs wannier-flow cell.xyz --cell ... --topology-plane-axes 0,1 --wannier-loop-divisions 12 --wannier-flow-points 12 --format json` | `--topology-plane-axes` (loop axis, flow axis), `--topology-fixed-k`, `--wannier-loop-divisions`, `--wannier-flow-points`; periodic reference controls | Centres per flow point and spin, their continuous sum and its winding |
| `periodic-cubic-fc-along` | `dft-rs periodic-cubic-fc-along cell.xyz --cell ... --k-mesh 2,1,1 --probe 0,0,1,0,0,-1 --format json` | `--probe` (one value per Cartesian coordinate of the cell); response and periodic reference controls | Derivative of the Gamma-point force constants along the probe  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc-along` | `dft-rs periodic-quartic-fc-along cell.xyz --cell ... --k-mesh 2,1,1 --probe 0,0,1,0,0,-1 --format json` | `--probe`, `--second-probe` (default the same pattern); response and periodic reference controls | Second derivative of the Gamma-point force constants along the probes  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-cubic-fc` | `dft-rs periodic-cubic-fc cell.xyz --cell ... --k-mesh 2,1,1 --format json` | response and periodic reference controls | Third derivatives for Gamma-point displacements  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc` | `dft-rs periodic-quartic-fc cell.xyz --cell ... --k-mesh 2,1,1 --format json` | response and periodic reference controls | Fourth derivatives for Gamma-point displacements  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `toec` | `dft-rs toec cell.xyz --cell ... --k-mesh 2,2,2 --format json` | `--field Ex,Ey,Ez` for neutral insulating cells; periodic reference and response controls | Clamped- and relaxed-ion third-order elastic constants in Hartree/Bohr^3 and GPa. At fixed field, `field_energy` is the stationary Berry enthalpy per primitive cell. See raw `response_residuals`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `foec` | `dft-rs foec cell.xyz --cell ... --k-mesh 2,2,2 --format json` | `--field Ex,Ey,Ez` for neutral insulating cells; periodic reference and response controls | Clamped- and relaxed-ion fourth-order elastic constants in Hartree/Bohr^3 and GPa, including second-order ionic relaxation in Bohr. At fixed field, see `field_energy` and raw `response_residuals`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `mode-gruneisen` | `dft-rs mode-gruneisen cell.xyz --cell ... --k-mesh 2,2,2 --format json` | `--gruneisen-shear` at order one; `--gruneisen-order 2` for isotropic log-volume curvature; `--dispersion`, `--disp-cutoff`, `--cn-cutoff` for geometry-only dispersion | Gamma frequencies (cm^-1), dimensionless mode parameters, and raw relative response residuals |
| `periodic-beta` | `dft-rs periodic-beta cell.xyz --cell ... --k-mesh 4,1,1 --format json` | response and periodic reference controls | Susceptibility and second-order susceptibility; nonzero HF water beta verified with all 27 unprojected index orders and a four-step fit; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-gamma` | `dft-rs periodic-gamma cell.xyz --cell ... --xc pbe --k-mesh 2,1,1 --format json` | Common periodic reference and response controls; `--use-symmetry true` by default computes 15 independent static components, `--use-symmetry false` evaluates all 81 for raw permutation checks. Static clamped ions; response_tolerance controls fresh path residual acceptance. The k mesh is represented by its Gamma supercell; finite Berry discretization is part of the model | Same tensor, chi3 normalization, shape, units and raw reference/path diagnostics as Rust. `--conv-tol-grad` can tighten the reference below the preserved 1e-10 cap; HF water also passes the unprojected 81-component four-step gamma acceptance; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-raman` | `dft-rs periodic-raman cell.xyz --cell ... --k-mesh 4,1,1 --format json` | response and periodic reference controls | Born charges and Raman tensors |
| `electrostriction` | `dft-rs electrostriction cell.xyz --cell ... --k-mesh 4,1,1 --format json` | response and periodic reference controls | Strain derivative of the susceptibility |
| `piezo-higher` | `dft-rs piezo-higher cell.xyz --cell ... --k-mesh 4,1,1 --format json` | response and periodic reference controls | Piezoelectric tensor and its strain derivative |
| `periodic-efield` | `dft-rs periodic-efield cell.xyz --cell ... --k-mesh 4,1,1 --field 0.001,0,0 --forces --format json` | `--field` (atomic units), `--forces`; periodic reference controls | Finite-field energy, dipole and forces Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Correlated `response_order=0..4`, `field_direction` in Hartree/e/Bohr per dimensionless path parameter; factorial-normalized energy coefficients in Hartree per primitive cell, unscaled SS/OS coefficients and raw orbital/pair/projector/metric/number/stationarity diagnostics. The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). |
| `dielectric-q` | `dft-rs dielectric-q cell.xyz --cell ... --k-mesh 4,1,1 --q-points 0.25,0,0,0.5,0,0 --format json` | `--q-points` (reduced, three numbers each, commensurate with the k mesh); response and periodic reference controls | Density response and dielectric function at each q |
| `orbital-susceptibility-q` | `dft-rs orbital-susceptibility-q cell.xyz --cell ... --k-mesh 4,1,1 --q-points 0.25,0,0 --format json` | Nonzero commensurate `--q-points` on an insulating 3D Gamma supercell. RKS and UKS support HF, LDA, GGA, meta-GGA and compatible hybrids. All-electron GKS supports semilocal XC with compatible exact exchange. GHF selects HF. | Diagonal transverse susceptibility in atomic, CGS and SI units, raw quadratic energies and relative `response_residuals`. GKS and GHF also return the full phase-averaged `chi_transverse` tensor in atomic units, projected energy matrices and raw static defects. [Convention and examples](api-guide.md#orbital-susceptibility-at-finite-q) |
| `periodic-dc` | `dft-rs periodic-dc cell.xyz --cell ... --dc-buffer 8 --dc-smearing 0.001 --forces --format json` | `--dc-fragments`, `--dc-buffer`, `--dc-smearing`, `--forces`, `--stress`; response and periodic reference controls | Divide-and-conquer free energy, forces |
| `dc-stress` | `dft-rs dc-stress cell.xyz --cell ... --dc-buffer 8 --format json` | `--dc-fragments`, `--dc-buffer`, `--dc-smearing`, `--forces`; response and periodic reference controls | Divide-and-conquer stress |
| `periodic-nmr-shielding` | `dft-rs periodic-nmr-shielding cell.xyz --cell ... --k-mesh 2,2,1 --q-points 0.5,0,0,0,0.5,0 --format json` | `--q-points` (reduced, commensurate with the k mesh), `--nuclei`; response and periodic reference controls | Shielding per nucleus, wave vector and polarization |
| `berry-dipole` | `dft-rs berry-dipole cell.xyz --cell ... --transport-k-mesh 8,8,8 --transport-temperature-k 300 --berry-dipole-doping 0.01 --format json` | `--transport-k-mesh`, `--transport-temperature-k`, `--transport-mu-shift` or `--berry-dipole-doping`; periodic reference controls | Berry curvature dipole tensor |
| `lab-frame-spin-spiral` | `dft-rs lab-frame-spin-spiral cell.xyz --cell ... --pbc 1,0,0 --xc svwn5 --k-mesh 4,1,1 --spiral-q 0.25,0,0 --format json` | `--spiral-q` (reduced), `--spiral-axis` (0,0,1), `--spiral-seed-field` (0.05), `--spiral-max-supercell` (12), `--nuclear-soc-scale` (0); periodic field/charge controls of `periodic-spinor-scf`; core guess, CDIIS, zero level shift | Spiral energy per cell, spin projections on both chiralities, net spin polarization, seeded-state energy and raw SCF residuals |
| `dmi` | `dft-rs dmi cell.xyz --cell ... --pbc 1,0,0 --xc svwn5 --k-mesh 4,1,1 --spiral-q 0.25,0,0 --dmi-self-consistent --format json` | Controls of `lab-frame-spin-spiral`; `--nuclear-soc-scale` (1, must be non-zero), `--dmi-self-consistent` | First-order and optional self-consistent chirality-odd spin-orbit energy, its even counterpart, rotation defect and raw residuals |
| `bands` | `dft-rs bands cell.xyz --cell ... --path "0,0,0;0.5,0,0.5;0.5,0.25,0.75" --points 30` | `--path` (reduced vertices), `--points`, `--labels`; periodic reference controls | Band energies along the path |
| `periodic-spin-response` | `dft-rs periodic-spin-response cell.xyz --cell ... --response-q 0.13,0.09,0.07 --spin-probe-field 1,0 --spin-g-factor 2 --format json` | Periodic reference, frequency and shared response controls; q Cartesian in 1/Bohr, complex longitudinal probe in atomic magnetic field | 3D collinear spin-only longitudinal response with shared adiabatic screening. Returns spin/charge electron numbers and magnetic moment per cell, susceptibility, complex density, raw dimensionless response residuals and iterations. Source +/-g B_z N(q)/4; uniform static equilibrium uses `pauli-spin-susceptibility`. Orbital/transverse/SOC response is separate; this documentation repair adds no numerical acceptance claim. |
| `periodic-field-scf` | `dft-rs periodic-field-scf cell.xyz --cell ... --pbc 1,0,0 --field-wave-indices 1,0,0 --field-vector-real 0,0.37,0 --field-vector-imag 0,0.12,0 --field-amplitude 0.4 --format json` | `--field-scalar re,im`, `--grid-cutoff`, `--ewald-alpha`, `--ewald-real-cutoff`, `--ewald-reciprocal-cutoff`, `--electrostatic-order`; signed charge compensation controls; ordinary periodic reference controls | The same complex SCF report, amplitude derivative, physical model and raw residuals as Rust; Fourier coefficients define a real field by adding the conjugate mode  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Explicit `reference=gks` or `reference=ghf` uses the shared complex spinor SCF. Nuclear spin-orbit coupling is opt-in on this entry. [Controls and validation](response.md#explicit-generalized-reference-on-static-periodic-field-scf). |
| `periodic-spinor-scf` | `dft-rs periodic-spinor-scf cell.xyz --cell ... --pbc 1,0,0 --xc svwn5 --spin-field-real 0.01,0.02,0.03 --format json` | Periodic field/charge controls; `--soc-scale` (1), `--nuclear-soc-scale` (1), `--nuclear-widths` (Bohr), `--spin-field-real/imag`; optional paired flattened `--spinor-operator-real/imag` in [k][2AO][2AO] Hartree order; total-number Fermi-Dirac occupations; core guess, CDIIS, zero level shift | Same variational spinor report as Rust; supplied external operators scale with field amplitude and nuclear zero-field SOC stays fixed while its uniform-field contact scales with amplitude; spin Zeeman includes curl A; spin populations are not separately fixed  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Joint03 bounded runtime-verified increment: explicit reference=gks/ghf selects this existing complex Bloch solver; GHF requires an HF energy provider, and ordinary/common-orbital sectors are refused. The same converged SCF mesh supplies band_energies, k_points and band_edges, with normalized band_eigen_residuals and band_metric_defects. No spin factor of two is applied. A gap is reported only for a uniform exact integer occupied prefix at zero electronic temperature; thermal or incompatible occupations leave band_edges null. This adds neither arbitrary-path interpolation nor generalized force/DFPT/Berry/ROKS support. The earlier three-dimensional two-k-point HF spin-rotation baseline passed with complex density and5.55e-17 Hartree energy difference; that earlier baseline did not execute the new report fields. Joint03 now independently checks both GKS/GHF two-k HF mesh reports, their complex density, orbital/occupation/number gates, units and occupied band-edge reconstruction. This bounded HF report proof does not establish generalized PBE or property-response accuracy. |
| `periodic-magnetic-scf` | `dft-rs periodic-magnetic-scf cell.xyz --cell ... --pbc 1,1,0 --magnetic-field ... --format json` | `--reference gks` or `ghf` with an HF provider, `--magnetic-model finite_gamma` or `crystal_bz`, `--xc`, `--k-mesh` and flux or quadrature controls | Energy in Hartree per magnetic cell, `reference_type`, flux quanta, SCF residuals, and independent spatial and XC angular diagnostics. The optional finite-Gamma nuclear report and 3D Berry electric branch have their own controls. [Shared settings and scope](api-guide.md#run-a-uniform-magnetic-flux-scf) |
| `dos` | `dft-rs dos cell.xyz --cell ... --sigma 0.005 --dos-mesh 8,8,8 --projected` | `--sigma` (Hartree, required), `--points`, `--energy-window`, `--dos-mesh`, `--projected`; periodic reference controls | Total, integrated and projected DOS. 1D/2D/3D RKS, UKS, ROKS and GKS on the stationary k mesh. ROKS uses spin-resolved semicanonical common-orbital blocks. Reports include states per cell, exact window counts and integration and projection residuals. Separate frozen sampling uses the bulk collinear route. Field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS reference energies include geometry-only dispersion. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `pauli-spin-susceptibility` | `dft-rs pauli-spin-susceptibility cell.xyz --cell ... --exchange-enhanced true --unrestricted --multiplicity 2 --temperatures-k 150000 --fermi-k-mesh 1,1,1 --spin-g-factor 2 --format json` | `--exchange-enhanced` defaults false; enhanced mode requires positive temperature and a compatible semilocal 3D collinear DFPT reference. Equal and unequal channels use the same driver; `--response-reciprocal-cutoff` and common solver controls apply | Fixed total number, with the baseline `(mu_alpha-mu_beta)/2` held fixed as spins exchange particles; this can represent a spin-biased constrained state. Bare oracle is `(g/4)^2 4 D_alpha D_beta/(D_alpha+D_beta)`. Common chemical-potential response, compressibility, per-spin DOS and raw residuals use the same units and conventions as Rust/native Python. Enhanced hybrid, finite-q and dynamic combinations remain pending Bare mode also accepts unequal collinear bands and `spin_g_factor`: frozen bands are reoccupied with a common chemical potential and zero reservoir spin bias. `spin_electrons`, `fixed_number_spin_dos`, `rigid_band_spin_magnetization`, `chemical_potential_response`, `delta_spin_numbers` and raw number defects describe that ensemble. It differs from the enhanced mode's fixed baseline bias; no self-consistent response is claimed. |
| `cli` | `dft-rs <command> ...`; `dft_rs_cli` is the Rust compatibility executable; `pip install .` installs canonical `dft-rs` | Every command and control of this table; no external native executable needed by the pip entry point | As documented per command; fresh-environment CLI and package metadata verified |
| `piezoelectric-strain` | Supplied proper stress coefficients and fixed-field compliance to d=eS; no electronic calculation | `dft-rs piezoelectric-strain --options inputs.json --format json`; required `piezoelectric_stress`, `piezoelectric_compliance`, `ion_response` | Engineering Voigt tensors; atomic input/output plus C/N and pm/V. Inputs must share axes and ion/electrical boundary convention. Joint03 bounded Rust/native/ASE/canonical and alias CLI checks pass all18 entries, SI conversions and input gates; the independent four-step fits have p=1.99814..1.99953 and minimum R2=0.999999768. Supplied tensors retain their own physical-validation responsibility. |
| `aggregate-moduli` | `dft-rs aggregate-moduli --phases "0.6,37,44;0.4,76.8,32" --format json` | `--phases`; no structure file | Aggregate bounds and estimates (GPa) |
| `rock-physics` | `dft-rs rock-physics --mineral-bulk-gpa 37 --mineral-shear-gpa 44 --mineral-density-kg-m3 2650 --porosity 0.2 --fluid-bulk-gpa 2.25 --fluid-density-kg-m3 1000 --frame critical-porosity --critical-porosity 0.4` | The physical inputs and frame parameters (`--pore-aspect-ratio` for `dem`, `kuster-toksoz` and `self-consistent`; `--mineral-conductivity-w-mk` with `--fluid-conductivity-w-mk`); no structure file | Dry and saturated moduli, Biot parameters, wave speeds |
| `feature-registry` | `dft-rs feature-registry --format json` | Output controls only; no structure file | The registry rows |

Shared response controls are `--response-tol` (default `1e-10`),
`--response-max-iter` (500), and `--response-restart` (40). Reference controls
include `--xc`, `--basis`, and the documented SCF/grid settings. The periodic
response commands default to PBE when no functional is specified. The `--cell`
vectors use Angstrom. Cartesian `--response-q` and `--reciprocal-cutoff` use
inverse Bohr; `--response-omega` and `--response-damping` use Hartree.
`--omega` is a functional range-separation parameter, a separate control.
Imaginary frequency requires a nonnegative frequency and zero damping.
Periodic response cannot be combined with force calculation or relaxation.
Thermal response additionally requires q=frequency=0, a positive reference
temperature, and compatible directions: number changes for fixed-number
calculations, chemical-potential changes for reservoir calculations. It does
not select a plane-wave source, so `--response-modes` does not apply.

```console
dft-rs dynamic-polarizability water.xyz --xc pbe --response-omega 0.05 --response-damping 0.01
dft-rs field-taylor water.xyz --xc pbe --field-direction 0,0,1 --response-order 3
dft-rs field-mixed water.xyz --xc pbe --field-directions "1,0,0;0,0,1"
dft-rs dielectric cell.xyz --cell "9,0,0;0,9,0;0,0,9" --response-q 0.04,0.01,0.02 --reciprocal-cutoff 0.75 --response-modes 0
```


## Occupied-projector delta-SCF controls

The existing commands accept the common zero-based complete occupied lists:

```text
dft-rs energy h2.xyz --xc pbe0 --basis sto-3g --occupation-mode imom --occupied-alpha '[1]' --occupied-beta '[1]' --format json
dft-rs gradient h2.xyz --xc pbe0 --basis sto-3g --occupation-mode mom --occupied-alpha '[1]' --occupied-beta '[0]' --unrestricted --field 0,0,0.0003 --format json
```

`mom` follows the previous occupied projector; `imom` retains the initial
projector. `aufbau` is the default. XYZ positions are Angstrom, electric
field is Hartree/(e Bohr), energy is Hartree and gradient is Hartree/Bohr.
The same physical Fock builds and SCF stderr progress apply to both the
initial and selected reference. Complete score, report, convergence and
scope conventions are in the [shared contract](rust-api.md#occupied-projector-delta-scf-controls).


## Fixed orbital populations

Explicit occupation controls use the shared driver for the default text
output too; neither `--format` nor `--save` is required. This also applies
to MOM/IMOM and accepts both `--key value` and `--key=value` syntax.

```text
dft-rs energy h2.xyz --xc pbe --occupation-mode=fixed --occupations-alpha='[0.8,0.2]' --occupations-beta='[0.8,0.2]'
dft-rs energy h2.xyz --xc pbe0 --occupation-mode fixed --occupations-alpha '[0.8,0.2]' --occupations-beta '[0.8,0.2]' --format json
dft-rs gradient h2.xyz --xc pbe --occupation-mode fixed --occupations-alpha '[0.8,0.2]' --occupations-beta '[0.7,0.3]' --unrestricted --field 0,0,0.0003 --format json
```

XYZ geometry is Angstrom; energies and the unequal-population gap are
Hartree; gradients are Hartree/Bohr; populations and map residuals are
dimensionless. Iteration energies and physical population residuals remain
on stderr by default. Use `energy` for prescribed populations and
`mermin-free-energy` for thermal equilibrium. Complete constraints, report
fields and unaccepted response combinations are in the
[shared fixed-population contract](rust-api.md#fixed-orbital-population-controls).

The same controls also select static electric and moving-basis nuclear
response through the ordinary commands:

```text
dft-rs polarizability h2.xyz --xc pbe0 --occupation-mode fixed --occupations-alpha '[0.8,0.2]' --occupations-beta '[0.8,0.2]' --field 0.001,-0.002,0.003 --format json
dft-rs hessian h2.xyz --xc pbe --occupation-mode fixed --occupations-alpha '[0.8,0.2]' --occupations-beta '[0.7,0.3]' --unrestricted --format json
```

`polarizability` returns the full unsymmetrized static tensor in
e^2 Bohr^2/Hartree, its raw asymmetry, three fresh response residuals and
per-spin number defects. `hessian` includes the APT, moving-metric response
residuals and number defects. `--electronic-temperature` instead selects
canonical thermal populations and their occupation response. Solver defaults,
shapes, units, the degeneracy condition, constrained-state thermochemistry
and remaining combinations are in the
[spectral-response contract](rust-api.md#fixed-population-and-thermal-static-response).

### Prescribed-population nonlinear electric response

`field-taylor`, `field-mixed`, `beta` and `gamma` use the common molecular
reference/response driver for ordinary and prescribed-population references.
For example, a two-orbital H2 basis accepts:

```sh
dft-rs field-taylor h2.xyz --xc pbe --occupation-mode fixed --occupations-alpha 0.8,0.2 --occupations-beta 0.8,0.2 --field 0,0,0.003 --field-direction 0,0,1 --response-order 4 --format json
dft-rs beta h2.xyz --xc hf --occupation-mode fixed --occupations-alpha 0.8,0.2 --occupations-beta 0.8,0.2 --electric-assembly variational
dft-rs gamma h2.xyz --xc hf --occupation-mode fixed --occupations-alpha 0.8,0.2 --occupations-beta 0.8,0.2
```

Use `--unrestricted` and separate spin populations for the unrestricted
case. XYZ coordinates are Angstrom; fields and reports use atomic units.
`--reference-field` aliases `--field`; supplying both is an error.
`--response-tolerance`/`--response-max-iterations` are the canonical solver
controls; `--response-tol`/`--response-max-iter` remain compatible aliases,
and giving both spellings is an error. The default tolerance is 1e-10,
maximum iterations 500 and GMRES restart 40.

`field-taylor` reports factorial-normalized dipole coefficients in e Bohr
for a dimensionless path parameter; `field-mixed` reports the unnormalized
mixed derivative. Full beta/gamma use e^3 Bohr^3/Hartree^2 and
e^4 Bohr^4/Hartree^3. JSON includes per-entry `units`, reference residuals,
actual SCF populations and population-map residual, plus order-resolved
fixed-spectrum/orthogonality/stationarity constraints. Nonzero raw projector
defects are retained for fractional populations; `projector_constraint`
states whether idempotency is an active constraint. Plain text retains all
ordered `beta a b c value` and `gamma a b c d value` rows from this same
report. SCF iteration energies and convergence residuals go to stderr by
default. No field derivative is obtained from a finite difference in production.

These are constrained nonthermal molecular static derivatives. Higher
thermal, correlated PT2 and periodic populations remain separate response
problems. See the measured fits and scope in
[response.md](response.md#prescribed-population-nonlinear-electric-acceptance).


### Experimental quantum-nuclear geometry controls

The `multicomponent` common driver has the following additional controls on
the canonical CLI, native Python and ASE. CLI flags use hyphens instead of
the underscores below. This extends the existing calculation; it does not
introduce a separate energy/derivative implementation.

| Control | Meaning and default |
|---|---|
| `quantum_position_constraint` | Boolean, default false. Solve `mean(r_I)=target_I` simultaneously with electronic/nuclear stationarity. Multipliers use `L=E+lambda dot (mean(r)-target)`; the physical energy excludes multiplier work. |
| `quantum_position_targets` | Optional three-coordinate rows, one per particle; defaults to each particle's Gaussian center. Native/CLI Bohr; ASE Angstrom by default. Requires the constraint flag. |
| `species_derivative_order` | Integer 0 (energy only), 1 (analytic gradient), or 2 (analytic gradient and relaxed Hessian). Defaults to 0. |
| `species_direction` | Optional dimensionless three-component rows in nuclear order followed by the added-particle Gaussian sites. A nuclear-only direction holds the appended sites fixed. If a particle's position is constrained, its expectation-position target follows its basis site. |
| `species_polarizability` | Boolean, default false. Compute the total static charge/dipole response with the complete coupled kernel. Quantum expectation positions remain fixed if the reference is position constrained. |
| `quantum_mass_response` | Boolean, default false. Return the analytic quantum-mass gradient and full relaxed mass Hessian in `mass_response`, in quantum-particle order. Gaussian bases, charges, correlation parameters and any expectation-position targets stay fixed. |
| `response_tolerance`, `response_max_iterations`, `response_restart` | Common response controls: 1e-10, 500, 40. The complete actual coupled operator is used for the final residual. |


`quantum_mass_response=true` (`--quantum-mass-response true` on the CLI)
adds `mass_response.gradient`, the flattened row-major `hessian` with
`hessian_shape=[Q,Q]`, independently assembled `raw_asymmetry`, and per-mass
coupled/reference/position residuals and particle-number defects. Masses stay
in electron masses (`m_e`) on every surface: gradients use Hartree/m_e and
Hessians Hartree/m_e^2 natively, or eV/m_e and eV/m_e^2 through ASE. The
position-response residual uses Bohr/m_e or Angstrom/m_e. This differentiates
the laboratory-frame model at fixed Gaussian bases, charges, correlation
parameters and position targets. It removes no COM kinetic energy and adds no
identical-particle statistics. The default is false and returns `mass_response=null`.
The isolated HF C1 H3+ two-isotope candidate passes all two gradient and four
ordered Hessian four-step fits: p=2.002752467556..2.004439020151, minimum
R2=0.999998680142; response residual at most 8.462e-15 and number defect at
most 3.331e-16. The scoped rebuilt artifact passes the permanent mass suite, native/ASE/both-CLI tests and default-false regressions; its source/core hashes are in `molecular-response-followup-20261101-01/final-acceptance.json`.

`geometry_derivatives` contains `gradient` and, for order 2, `hessian`.
For `N` nuclei and `Q` added particles, full gradients have shape `[N+Q,3]`.
Hessians are flattened row-major with `hessian_shape=[3(N+Q),3(N+Q)]`.
`added_particle_coordinate_order` gives the zero-based particle indices of
the appended Gaussian sites. Directional reports contain
scalars, `explicit_hessian`, `orbital_relaxation` and the source `direction`.
Native/CLI units are Hartree/Bohr and Hartree/Bohr^2; ASE returns eV/Angstrom
and eV/Angstrom^2. Response residuals, signed analytic order-zero energy
reconstruction errors and convergence flags accompany both forms. Order 1
does not expose an uncomputed off-diagonal Hessian.

`position_constraints` returns targets (Bohr), multipliers (Hartree/Bohr),
the maximum position residual/history (Bohr), iterations and its sign
convention. ASE converts these lengths and forces. `polarizability` returns
`tensor[3,3]`, `raw_asymmetry` (Bohr^3), coupled and linearized-position
residuals, iterations and convergence. ASE converts the volume unit once.
SCF, outer species and position-Newton iteration energies/residuals are
reported on stderr by the CLI. Null optional report sections mean that the
calculation was not requested; no physical zero is substituted.

For unconstrained quantum nuclei, a Gaussian-center derivative measures
finite-basis dependence and is explicitly labeled that way. Position
constraints provide nuclear-mean-coordinate derivatives, including the
analytic multiplier response. Both are laboratory-frame variational model
quantities. No exact internal vibronic spectrum, center-of-mass subtraction,
or classical fixed-nucleus interpretation is inferred. The connected subset
is all-electron, zero-temperature real collinear R/U with shared electronic
XC with optional local electron–particle correlation. The independent
added-particle center path differentiates finite-basis Gaussian parameters.
Moving-species ECP/RI/nonlocal, PT2 and periodic derivatives remain pending.
Measured acceptance and failed steps are recorded in [response.md](response.md).

### ROKS convergence fallback (acceptance pending)

Details, units and validation are in [the shared reference](rust-api.md#roks-convergence-fallback-acceptance-pending).

### Molecular reference-aware dipole response

Details, units and validation are in [the shared reference](rust-api.md#molecular-reference-aware-dipole-response).

### Collinear ECD, VCD and NMR functional providers

Details, units and validation are in [the shared reference](rust-api.md#collinear-ecd-vcd-and-nmr-functional-providers).

### Correlated orbital NMR on the existing API

The existing `nmr-shielding` and `nmr-spectrum` calculations accept the same
named/custom double-hybrid controls as `energy`: `xc`,
`double_hybrid_model`, `double_hybrid_target_xc`, `pt2_same_spin`,
`pt2_opposite_spin` and `pt2_memory_mb`. The common RKS/RHF and UKS/UHF
selectors are retained. The spectrum uses the existing reference shieldings
and Lorentzian line model; no separate correlated-spectrum API is added.

`shielding_tensors` are the mixed derivative of the complete target-plus-PT2
energy. `parent_shielding_tensors`, `target_shielding_tensors`,
`pt2_shielding_tensors`, `pt2_same_spin_alpha_shielding_tensors`,
`pt2_opposite_spin_shielding_tensors` and
`pt2_same_spin_beta_shielding_tensors` retain all `[nucleus][moment][B]`
entries in ppm. The three spin channels are unscaled; the PT2 tensor is
`c_ss*(SS_alpha+SS_beta)+c_os*OS`. The diamagnetic tensor is the complete
first derivative with respect to the explicit B/m contact operator; its
post-SCF density includes parent-orbital relaxation. The paramagnetic tensor
is the remainder. Energies/decompositions reuse the common energy report.
Amplitude, orbital-frame, projector, stationarity, reference and imaginary
energy residuals remain explicit; convergence tests inspect the raw values.

The integer zero-temperature orbital model excludes spin-Zeeman/Curie shifts.
Correlated J-coupling multiplets, nonlocal/learned field-path providers and
causal PT2 transition/VCD response remain separate work. The complex post-SCF field provider now uses the explicit
`CurrentInvariant` kinetic-density model, `tau - |j|^2/(2 rho)`, for both
the analytic energy/Fock series and its cached magnetic response kernel.
The independent finite-SCF oracle `magnetic_field_scf_with_core` uses the
same physical model; `magnetic_field_scf` and the existing finite-B nuclear
paths retain the `Mechanical` model. `FieldWavefunctionPath` and
`MagneticFieldResult` identify their model with `kinetic_density_model`.
The meta-GGA connection has independent TPSS full-component magnetic and
common-origin NMR acceptance, recorded below; other combinations are not
inferred from those measurements.


Correlated orbital NMR acceptance (`p255_correlated_nmr`, 2026-10-23)
passes all 810 common-origin Cartesian component/channel fits for restricted
C1 H4 and unrestricted C1 H5 with HF/PBE0 parents: total, parent,
SS-alpha, OS and SS-beta. The 808 second-order stencils have
p=1.807113..2.131252 and minimum R2=0.99911588; two cancellation cases
use the prescribed fourth-order mixed stencil, p=3.941553/4.022333,
minimum R2=0.99998602. All 218 failed interval attempts remain printed.
Ordinary steps are [0.04,0.03,0.02,0.015], refined through
[0.003,0.0025,0.002,0.0015]; the fourth-order intervals are
[0.04,0.03,0.02,0.015] and [0.02,0.015,0.01,0.0075].
The original restricted and unrestricted failures are retained separately;
SCF density/orbital convergence is tightened to 1e-14. No error floor,
component omission or relaxed gate is used. Maximum amplitude residual is
3.553e-15, imaginary-energy defect 1.666e-16, transported-frame defect zero
at printed precision, and the full parent-shielding limit differs by at
most 2.942e-12 ppm. The run passes in 399.15 s. This validates the named
common-origin molecular PT2 subset; full London/translation acceptance,
nonlocal/learned correlated fields, correlated transitions and spin-spin
multiplets remain separate. TPSS common-origin acceptance and the new
VCD connection are recorded below.

The complete magnetic energy Hessian acceptance (`p254_correlated_magnetic`)
passes 180 Cartesian/channel fits for restricted C1 H4 and unrestricted
C1 H5 with HF/PBE0 parents. Every total/parent/SS-alpha/OS/SS-beta entry is
computed independently: p=1.810889..2.168594, minimum R2=0.99935513.
All 243 failed interval attempts remain in the output. Resolved intervals
are [0.08,0.06,0.04,0.03], [0.02,0.015,0.01,0.0075] and
[0.005,0.00375,0.0025,0.001875] atomic units. The finite-field phase
expansion was extended from order 12 to 16 after the omitted tail of
4.706e-11 failed the unchanged 1e-12 gate. Maximum fresh amplitude/field
residuals are 1.421085471521e-14/2.026057371854e-15; the largest imaginary
energy defect is 3.885780586188e-16.

Public correlated-NMR acceptance passes all eight new cases: six native,
common-driver and ASE comparisons and two canonical CLI comparisons. The
first CLI fixture used a different Bohr conversion constant; correcting
the XYZ fixture to `dft_rs.units.BOHR_IN_ANGSTROM` resolves both failures
without changing the 1e-8 ppm gate. The companion spectroscopy, unit,
documentation and ledger checks passed in that run. These are HF/PBE0
custom PT2 and B2PLYP common-origin orbital-spectrum checks; they do not
certify PT2 ECD/VCD, coupled-spin NMR spectra or the new meta-GGA model.

The current-invariant complex XC pullback passes all 38 independent fits
of its energy and both spins' 18 independent Hermitian Fock coordinates,
through derivative order two: p=1.966887..2.001269, minimum R2=0.99998051,
steps [0.08,0.06,0.04,0.03], no failed interval. The oracle differentiates
the energy in density coordinates using forward AD and exact trigonometric
AO phases; it does not reuse the production potential pullback. TPSS full
magnetic energy Hessians (`p254_correlated_magnetic`) pass all 90 RKS/UKS
Cartesian/channel fits: p=1.811654..2.099307, minimum R2=0.99932393;
141 failed interval attempts are retained. The complete TPSS mixed NMR acceptance
(`p255_correlated_nmr`) passes all 405 RKS/UKS Cartesian/channel fits:
p=1.801623..2.147692, minimum R2=0.99939482, with all 123 failed interval
attempts retained. Every accepted TPSS stencil is second order; the
four-step intervals are printed per component. Together with the 810
HF/PBE0 fits this gives 1,215 accepted common-origin NMR entries. The
TPSS unrestricted full parent-tensor limit differs from the independent
magnetic provider by at most 2.948752353404e-13 ppm. This does not certify
London correlated shieldings, pair spin couplings, finite-temperature
response, nonlocal targets or other unmeasured combinations.

The existing `vcd` entry is now connected to
`properties::correlated_axial::correlated_molecular_axial_response` for
custom/named double hybrids. It uses the complete parent nuclear and
magnetic orbital frames and noncanonical PT2 amplitude derivatives, plus
the nuclear ket-basis half-overlap. Its harmonic force field and APT come
from the same target-plus-PT2 energy through `correlated_molecular_taylor`;
the existing nuclear tensor assembler and scalar-generic dispersion paths
are reused. The measured common-origin double-hybrid VCD subset is recorded in response.md. `p256_correlated_axial` compares every AAT
entry with independent displaced real/complex SCF determinants and direct
Slater cross-overlap sums; measured acceptance is recorded in response.md. The independent full Hessian/APT acceptance is `p257_correlated_vibrational`; public VCD cases are in `python/tests/test_correlated_vcd.py`.

`properties::pt2_axial::{PairAxialState, PairStateTangent, PairAxialOverlap}`
provide the reusable normalized first-order pair-wavefunction model. The complete double-substitution tensors avoid a fixed-word orbital limit. Nuclear and magnetic
connections are complete anti-Hermitian MO connections. The zero-field
nuclear tangent is real and the magnetic tangent purely imaginary. The
model uses `M=1+sum_c w_c <T_c|T_c>` and
`Im<d_R Psi|d_B Psi> = (I_parent+sum_c w_c I_c)/M
- M_R Im<Psi|d_B Psi>/(2 M^2)`. SS/OS weights multiply channel bilinears
and their norm; negative custom weights are an explicit signed continuation
and a nonpositive resulting norm is refused. Linear imaginary cancellation is explicitly evaluated; the cross-channel
terms are reduced by the one-body commutator selection rule described below. This is the
normalized MP1 AAT model of the
[analytic pair-wavefunction formulation](https://arxiv.org/abs/2501.06632),
with generic orbital-generating DFT references and explicit spin scaling.

`vcd` retains the controls `gauge`, `gauge_origin`, the reference controls
and generic `double_hybrid_model`, `pt2_same_spin`, `pt2_opposite_spin`,
`pt2_memory_mb`. Common origins and atom-distributed common origins are
explicit; they do not include London basis derivatives. Tensor axes remain
`[3*nucleus + nuclear_axis][magnetic_axis]`, imaginary overlap in e*bohr/hbar.
New diagnostics are `parent_atomic_axial_tensors_electronic`,
`pt2_axial_channel_bilinears` (unscaled intermediate-normalized SS-alpha,
OS, SS-beta bilinears, not additive normalized corrections),
`pt2_wavefunction_norm`, `pt2_wavefunction_norm_derivatives`,
`pt2_magnetic_connections`, `axial_connection_defects`,
`axial_cancellation_defects`, `amplitude_response_residuals`,
`orbital_frame_defects` and `vibrational_response_residuals`.
No finite differences enter these production paths. London pair AATs,
nonlocal/learned correlated field providers and genuine correlated ECD
transition moments remain separate, unfinished work.


`MolecularCorrelatedTaylor` also exports its complete `orbitals`, `occupied` and unscaled `amplitudes` Taylor data. These reuse the already computed moving-metric/nuclear path for wavefunction observables; no second orbital or amplitude solve is introduced. `vcd` returns the complete `hessian` (Hartree/Bohr^2) and `dipole_derivatives` (e, dipole-axis by nuclear-coordinate) on both ordinary and correlated routes.

Pair AAT baseline acceptance (2026-10-24), before the exact commutator
reduction, includes every common-origin Cartesian entry and every unscaled
channel for C1 H4/RKS and C1 H5/UKS:

| Parent | AAT component/channel fits | Accepted p range | Minimum R2 | Failed interval attempts retained |
|---|---:|---|---|---:|
| HF | 405 | 1.920500..2.019506 | 0.99986947 | 0 |
| PBE0 | 405 | 1.938054..2.005580 | 0.99996573 | 1 |
| TPSS | 405 | 1.904521..2.086913 | 0.99981642 | 0 |

The independent oracle converges real displaced SCF determinants and
complex finite common-origin-field determinants, transforms each PT2
amplitude by direct four-AO sums, and evaluates determinants of cross-AO
orbital overlaps. Nuclear steps are [0.02,0.015,0.01,0.0075] Bohr and
field steps are 0.03 times those numbers in atomic magnetic units; the
additional printed intervals resolve any failed fit. No element is copied
from tensor symmetry. The separate HF complete force-field/APT acceptance
passes all 450 entries (369 Hessian elements and 81 APT elements),
p=1.964973..2.135546, minimum R2=0.99976884, without failed intervals.
The expanded all-five-parent VCD acceptance passes all 2,025 AAT/channel and 2,250 Hessian/APT fits.

The production pair contraction now uses the exact fermionic identity
`Im<G_R T|G_B T> = -Im<T|G([K_R,K_B])T>/2` for anti-Hermitian orbital
connections. `G(K)T = Tr_occ(K)T - pair_action(K,T)` reuses the same
occupied-minus-virtual tensor operation as the complex PT2 amplitude
equations. Nuclear/magnetic amplitude cross terms and both contributions
to the magnetic Berry connection, `G(K_B)T + T_B`, are retained. Thus no
triple-substitution state expansion is allocated: each contraction scales
as O(N^5) with O(N^4) coefficient storage. `cancellation_defect` measures
the explicit single-space linear cancellation and the commutator's
anti-Hermitian defect; the cross-channel reduction is an exact one-body
spin-sector selection rule tested by the full Slater-overlap oracle.

`PairAxialState::with_memory_limit(dimensions, occupied, amplitudes,
memory_mb)` checks the estimated coefficient/matrix workspace before
allocation. `new` uses the existing default memory budget. The molecular
driver forwards `pt2_memory_mb`; no fixed orbital-count mask is introduced.
The exact contraction passes the full-crate/all-parent regression,
including SVWN5 and PBE parents. A preflight caught an omitted amplitude
Berry term during this optimization; it was restored without changing any
fit threshold, and a normalized two-state closed-form/four-step regression
plus a field-dependent global-phase limit were added. The failed run is
retained outside the source package.

AAT labels now explicitly include hbar: `e*bohr/hbar` (hbar=1 in atomic
units), and `pt2_magnetic_connections` is `e*bohr^2/hbar`. The latter's
length-square factor is essential to ASE conversion and to reconstructing
the normalized AAT from its reported channel/norm diagnostics in either
unit system. The numerical AAT convention is unchanged. Public acceptance
includes this reconstruction and the physical conversion of every tensor.


### Correlated TDA transition API and kernel acceptance

Details, units and validation are in [the shared reference](rust-api.md#correlated-tda-transition-api-and-kernel-acceptance).

### Correlated nuclear-moment response and learned field provider

All three existing coupling/NMR entries accept the ordinary double-hybrid
controls (`xc=b2plyp` or an explicit parent and PT2 coefficients). Reports
add `coupling_parent_tensors`, `coupling_target_tensors`,
`coupling_correlation_tensors`, `coupling_same_spin_alpha`,
`coupling_opposite_spin`, `coupling_same_spin_beta` (atomic units),
`coupling_response_converged`, `coupling_response_residuals` and
`coupling_path_residuals` (dimensionless),
`coupling_amplitude_residuals` (Hartree in alpha^2-scaled nuclear-source coordinates),
and `coupling_stationarity_defects` (Hartree). The model is a spin-free
closed-shell energy Hessian; longitudinal response is not substituted for
the transverse response required by an open shell. Full source, amplitude,
orbital and DSO relaxation is retained. No new calculation identifier is
needed. Measured scopes and raw residuals are in response.md.

The learned complex field provider evaluates the entire global network
with the tau-|j|^2/(2 rho) chain rule; orders 0..7 are connected. Its
38 independent energy/full-Fock coefficient fits and 810 full mixed-NMR
SCF/PT2 fits pass on RKS/UKS and common-origin/London gauges. Nonlocal two-endpoint magnetic
providers and London pair VCD are separate unfinished combinations.

Measured magnetic checkpoint (2026-10-24): 2025 full London PT2 NMR fits, 810 global learned mixed-NMR fits and 925 full FC/SD/PSO/DSO PT2 J fits pass. Raw residuals, failed/resolved intervals, model conventions and the bounded open combinations are in response.md.

### Explicit auxiliary field energy models

Details, units and validation are in [the shared reference](rust-api.md#explicit-auxiliary-field-energy-models).

### Complete correlated force-constant tensors

Details, units and validation are in [the shared reference](rust-api.md#complete-correlated-force-constant-tensors).

### Correlated high-order molecular exports

Details, units and validation are in [the shared reference](rust-api.md#correlated-high-order-molecular-exports).

### Common-orbital/spinor high-order scope and SCF diagnostics

On the existing `force-constants` call, `reference=roks/rohf/gks/ghf`
selects the common-orbital or full complex-spinor provider through order
four. Tensor/block/pattern selection and units follow the existing call.
`parent_tensor` and `dispersion_tensor` sum to `tensor`. Reference,
response, projector and nuclear-stationarity defects retain their raw
values, alongside the complete-Cartesian acoustic residual; a failed
gate sets `converged=false`. GKS accepts the prescribed Hermitian
`gks_operator_real`/`gks_operator_imaginary` source; a fixed AO source
does not imply a moving nuclear SOC model. Constrained/spinor PT2,
nonlocal/learned spin kernels and arbitrary-q periodic high-order tensors
remain separate. Independent all-component/public acceptance is in
progress; retained SCF and quartic failures and measured subsets are in
[response.md](response.md#common-orbitalspinor-tensors-and-scf-globalization-2026-10-26).

ROKS Newton optimization uses compensated quadratic finite-density work
when the provider is quadratic. After complex SCF stagnation, energy
globalization remains active for CDIIS proposals and backtracked convex
canonical updates. Rust providers may declare exact density-quadratic
energies with `SpinorEnergyBackend::quadratic_density_energy` (default
false); nonlinear providers retain actual energy evaluations. This
declaration changes numerical work evaluation only. Final physical
Fock, number, occupation, density, residual and energy gates are unchanged.
Globalization returns to local CDIIS after a factor-100 physical residual
reduction, retaining the same final convergence gates. The retained 12-case
SCF suite passes at 1e-14 reference tolerances; the four-route gradient/API/
ledger/Ruff suite passes nine tests. Raw numbers and failures are recorded
in response.md. CLI iteration energies, residuals and accepted fallback
steps go to stderr.

For correlated molecular nuclear outputs, `parent_nonlocal_tensor` and
`target_nonlocal_tensor` (or the corresponding gradient, Hessian and
`contracted_force_constants` prefixes) give weighted nonlocal parts
already included in parent/target totals. They use the same units as
their total derivative. Rust's
`response::nuclear::correlated_molecular_taylor_for_model` accepts the
same typed `FieldEnergyModel` as electric response. Full moving-pair
and public nuclear acceptance remains pending; no complete nonlocal
functional/reference coverage is inferred from the new fields.


The latest complex-SCF globalization uses occupied-subspace conjugate
gradients with Cayley density increments and the same energy/Fock backend;
the expanded retained set passes 18 cases at unchanged 1e-14 reference
gates. ROKS projector coefficients retain summation remainders throughout
the physical density/stationarity path. Full quartic/public acceptance is
still in progress: the preceding optimized public suite passed 12/14 and
retained two ROKS quartic convergence failures. See the dated raw results
in response.md; no broader functional/reference coverage is inferred.

The two previously failing ROKS quartic public selections now pass all
four routes with their original gates: eight reports have worst relative
response 7.860938497533e-12, raw projector
6.058451752097e-28, stationarity
9.730564927445e-12 Hartree and reference
1.783486552887e-13 Hartree. The internal rotation solve is
stricter when needed to satisfy both requested relative and absolute
residual bounds. Physical operators and public thresholds are unchanged;
complete-component/functional acceptance is tracked separately in response.md.


### Structured quantum-particle and positron correlation contract

The existing multicomponent entry accepts either legacy quantum_atoms,
quantum_masses and positron_positions, or a particles list, never both.
Each object has kind=nucleus, positron, muon, negative-muon or particle;
atom is a zero-based owner and position has three Cartesian coordinates.
A nucleus replaces its matching classical point charge. Other kinds add a
distinguishable Hartree orbital. A custom particle requires charge and mass.
Optional exponents, lmax, correlation and label override the shared defaults.
Masses are in electron masses; atomic units use Bohr and inverse Bohr squared.
ASE mode converts nested positions/exponents from Angstrom/inverse Angstrom
squared once, using the same dotted control-unit registry as scalar controls.

positron_correlation=drummond-2011 (alias qmc-2011) selects the local
dilute-positron relaxation energy; none selects the Coulomb-only model.
Each particle can instead specify its correlation explicitly. Distinct
correlation channels add their electron potentials and kernels; particle
densities within a channel are summed. The electronic functional is generic.
The model is a zero-positron-density impurity approximation, not a general
finite-positron-density or intrinsic vibronic theory. Laboratory COM kinetic
energy, fixed unselected nuclei and distinguishable-particle assumptions remain.

annihilation_enhancement accepts ipm, drummond-2011, barbiellini-ap,
barbiellini-sl and barbiellini-qmc. Optional nonnegative dimensionless
annihilation_gradient_alpha applies the gradient damping; zero is exactly
the undamped model. annihilation.positrons contains contact_density and
enhanced_contact_density (Bohr^-3 natively), rate (inverse atomic time),
rate_per_ns (ns^-1) and lifetime_ps (ps), per normalized positron. The
spin-averaged convention is lambda=pi*alpha_fs^3*int(n_+ n_- gamma).
ASE converts contact densities to Angstrom^-3; the atomic-time, ns and ps
rates/lifetimes retain their labelled units.

Independent p281 component evidence includes 16 four-step local kernel-row
fits, seven high-precision point/term oracles through density 1e-300 and five
closed Gaussian annihilation integrals (maximum enhanced-contact defect
3.119726699197e-14 Bohr^-3). These are local/component checks, not molecular
all-component derivative acceptance. All four full-crate p281 cases pass;
the two none/QMC public cases pass native Python, ASE atomic/physical units
and canonical CLI (4.16 s). The retained TPSS electronic SCF failure and
broader regression status are in response.md.


### Shared variational SCF globalization

Details, units and validation are in [the shared reference](rust-api.md#shared-variational-scf-globalization).

### Full-spin electronic nuclear J response

The existing `spin-spin-couplings` and `fermi-contact-couplings` calculations
connect the unchanged UKS/UHF reference or the self-consistent complex
GKS/GHF reference to the same full spin/AO response kernel. The current
provider covers all-electron HF, semilocal and full/range-separated hybrid
Hamiltonians with integer populations. FC, SD, PSO and explicit DSO are
retained; `spin_orbit_mixed_tensors` includes both FC/SD-PSO source orders
and is included in `reduced_couplings`. All Cartesian/spin source components
are solved independently. Reference/functional names do not select kernels.

`spin_response_constraint="fixed-spin-direction"` imposes
`Tr[S_perp1 dP]=Tr[S_perp2 dP]=0` using two orthogonal axes transverse
to the reference total spin and a Lagrange-multiplier saddle system.
The total-spin magnitude can relax. A zero/unresolved reference moment
has no spin direction and is refused with instructions to use free
response for a spin-paired state or a separate local-constraint provider
for a compensated magnet. Constraining all three spin components at a
collinear determinant would be singular: its longitudinal Jacobian is
zero despite nonzero second-order expectation changes. That draft
constraint is not used as a finite-field ensemble.
`"free"` (default) requests the unmodified unconstrained response and can be singular
at a Goldstone mode. No energy gap or Hessian eigenvalue is replaced.
The fixed-spin calculation excludes free-spin thermal Curie enhancement.
Common-orbital ROKS, open-shell PT2, ECP, PCM, learned and nonlocal spinor
response providers remain separate; no support is inferred for them.

Reports retain `[pair][M-axis][N-axis]` order and the requested nucleus
ordering. Couplings are atomic units and optional isotropic constants are Hz.
`response_residual` is the fresh relative saddle-equation residual;
`spin_response_constraint_residuals` contains the absolute derivatives of
the original spin constraints, in atomic units. Rank-revelation discards
only numerically dependent constraint projections and reports their norms
as `spin_response_discarded_projection_norms` (dimensionless), together
with `spin_response_constraint_rank`. `response_reference_fock_defect` is
Hartree; angular/cache histories have `[dimensionless, Hartree, Hartree]`
columns. The bounded all-component and public acceptance status is recorded
in response.md; code connection alone is not validation.


### Molecular ECP nuclear Taylor connection

Details, units and validation are in [the shared reference](rust-api.md#molecular-ecp-nuclear-taylor-connection).

### Finite-q full and range-separated exchange connection

Details, units and validation are in [the shared reference](rust-api.md#finite-q-full-and-range-separated-exchange-connection).

## Spin-complete pair and common-orbital provider status

| Physical calculation | Canonical CLI entry | Current integration scope |
|---|---|---|
| Generalized complex one-body PT2 path | No separate command | Lower-level Rust provider passes 41 bounded pair/physical component fits and 108 complete two-site FC source fits; additional public Ramsey combinations remain pending |
| ROKS common-orbital generic one-body response | Existing `dft-rs roks-polarizability`/`dft-rs dynamic-polarizability` electric routes reuse this engine | General longitudinal spin/orbital operators are available in Rust; full transverse ROKS J integration is pending |
| Open-shell correlated Ramsey coupling | Existing `dft-rs spin-spin-couplings` | The closed-shell correlated subset remains documented above; complete full-spin open-shell assembly, constraints and public acceptance remain required |

Input/output conventions and complete lower-level signatures are in
[rust-api.md](rust-api.md#spin-complete-pair-and-common-orbital-providers).
The [response record](response.md#spin-complete-pair-and-common-orbital-work-2026-10-30)
separates running provider tests from completed physical calculations.

High-angular nuclear scope: the existing molecular basis controls accept g,
h and i (l<=6), without a separate calculation API. The current C1 HF
acceptance has 1872 complete AE/ECP RHF/UHF gradient/Hessian fits and
776 independent scalar-ECP AO matrix comparisons. Complete cubic
fits pass all 364 directions for AE RHF/UHF g/h/i and scalar-ECP g/h RHF
(2912 directions). AE RHF g/h/i quartics each pass all 1365 directions
(4095 total); h/i include five/three resolved fourth-order stencil fits.
The g AE UHF quartic also passes all 1365 directions (1360 second-order,
five fourth-order, minimum R2=0.99926600); all 34 failed intervals are
retained. Remaining ECP/h/i-UHF quartics require separate complete runs. Independent HF terms and all twelve energy public-route
cases pass. Other higher-order/reference combinations remain in acceptance. Units and full measurements,
including the resolved integral-workspace failure and later optimization
limits, are in [response.md](response.md#c1-ghi-nuclear-component-acceptance-2026-10-30).

The ECP scalar kernel reuses exact sphere moments and repeated primitive
monomial contractions. All 5323 old/new high-angular Cartesian entries,
including raised intermediates for order four, are bitwise equal; all
228 AO/order fits, 624 nuclear regression entries and 776 independent
ECP matrix entries pass. Isolated kernel timings and bounded cache storage
are in response.md; they do not establish a whole-calculation speedup.
On the combined build containing the exact optimized ECP source, all thirteen
native/ASE atomic+physical/canonical-CLI cases pass; maximum independent
energy defect is 1.438849039914203e-13 Hartree.

The nine selected continuation models pass all 12285 independent C1 nuclear
quartic directions: 12245 second-order and 40 fourth-order error fits,
minimum R2=0.999010185261, with all 228 failed earlier intervals retained.
The initial six-model h-boundary set contributes 8190 directions. The three
additional compact C1 s/i HF models (scalar-ECP RHF, scalar-ECP UHF and
all-electron UHF) each pass all 1365 directions on the identical fixed
library, adding 4095 directions. In the i subset, 4077 second-order fits give
p=1.811921818572..2.195963003999 and 18 fourth-order fits give
p=3.888446720385..4.152374561445; minimum R2=0.999109974377 and maximum
reported linear residual9.979730888808e-13. The fixed source/optimization
artifact is identified by the Immutable20 table in response.md. This is
bounded model evidence; the later common response-frame precision extension
requires its own affected regressions. The two ECP PBE RKS/UKS quartics remain
deferred and unvalidated within the full objective. No arbitrary cumulative
basis stack, registered-functional/reference matrix or full-program
completion is inferred.

The default molecular SOSCF iteration can hand off unresolved, repeatedly
stalled projector steps to a fresh physical-Fock/CDIIS sequence. Energy,
density and gradient convergence gates are unchanged; no new control is
required. The retained i-shell UHF displacement now converges in twelve
iterations with physical commutator 5.329070518201e-15 Hartree at the
original 1e-14 energy/density/gradient controls. Its formerly failed cubic
direction passes the original four-step fit (p=1.997920, R2=0.99999993).
The combined native/ASE atomic+physical/canonical-CLI suite passes all
thirteen cases, including strict recovery; independent catalogue energies
agree within 1.438849039914e-13 Hartree. The repaired i-UHF cubic also
passes all 364 directions (363 second-order fits and one fourth-order fit,
minimum R2=0.99907621); all five failed intervals remain recorded.
Other reference/functional and scalar-ECP higher-order combinations retain
their separate acceptance scope.

## Explicit anharmonic treatment selection

`dft-rs anharmonic minimum.xyz --anharmonic-method strict-vpt2` selects standard nonresonant second-order VPT2. Both CLI names accept the same selector. The default is `retained-polyad`; unknown values and use with another command are rejected.

Strict VPT2 truncates cubic interactions at second order and quartic/Watson terms at first order, and includes normalized transition moments through second order. Reported intensities square those corrected moments; they are not an intensity expansion truncated at order two. Coupled near-resonant reported states cause an explicit error, without denominator floors or an automatic fallback. The retained-polyad method diagonalizes the contact-transformed retained Hamiltonian and can include higher-order contributions through that diagonalization. It remains an explicitly named approximation with its existing resonance controls and validity warnings.

Python and CLI reports return `anharmonic_method`, `energy_order` (the nominal construction order), `energy_treatment`, `transition_moment_order`, `intensity_treatment`, `resonance_treatment`, `eigensystem_kind` and `convergence_scope`. Strict effective-Hamiltonian matrices contain diagonal perturbative levels and identity vectors; zero eigensolver diagnostics do not claim an exact eigensolution of a coupled Hamiltonian. Numerical convergence alone does not establish perturbative validity. Strict `exterior_mixing_norm` denotes the largest first-order wavefunction-correction norm among reported harmonic states, as stated in its warnings; retained-polyad keeps the earlier exterior-transformation diagnostic. Frequencies are cm^-1, IR intensities km/mol and static Raman activities Angstrom^4/amu. A selected-mode calculation omits unselected-mode couplings in either method.

## Collisionless thermal electric response

Details, units and validation are in [the shared reference](rust-api.md#collisionless-thermal-electric-response).

### Periodic particle-number diagnostics

Details, units and validation are in [the shared reference](rust-api.md#periodic-particle-number-diagnostics).

## Constrained static electric tensors

Details, units and validation are in [the shared reference](rust-api.md#constrained-static-electric-tensors).

## Active descriptor Taylor kernels (2026-11-01)

The shared Taylor potential implementation uses two active descriptors for
LDA bodies, five for GGA bodies and seven for meta-GGA bodies. Inactive
potential arrays remain exactly zero in the unchanged seven-descriptor
output contract. The public external-body evaluator retains seven inputs;
`xc::taylor::DirectionalTaylorScalar` remains the original nongeneric
seven-descriptor public alias, preserving constructor type inference;
active descriptor types remain private to the family drivers. Parameter
Taylor paths and the independently varying directional path are retained.
No response dispatch branches on a registered functional name, and no new
CLI, native Python or ASE control is needed. Existing common-driver routes
consume the same providers.
Energy density uses Hartree/Bohr^3; density, gradient-contraction and kinetic
descriptors retain their existing atomic-unit definitions. Taylor entries
are coefficients of t^n (the nth derivative divided by n!); the
physical path fixes the units of t and the potential partials.

The permanent `src/xc/taylor_tests.rs` suite compares every energy, potential
and mixed-potential coefficient against the seven-lane evaluation through
order five, including moving parameters. A warning-strict standalone build
of the exact changed source passes five tests. Independent closed
exponential first-through-fourth derivatives cover all 468 unisolvent
directions across descriptor dimensions 2, 5 and 7, with 1390 fits across
energy, potential and mixed-potential representations. At
h=[0.08,0.04,0.02,0.01], p=2.000005591487..2.000070978831 and minimum
R2=0.999999999662. Every error is measured without a floor; no interval
fails. Separate fourth-order Richardson and closed derivative checks pass.
An isolated earlier prototype also preserves all 1008 compared coefficients
against the frozen library. These are arithmetic model contracts, not
chemical functional/reference acceptance or a whole-program speed claim.

Physical combination status and retained timeouts are recorded in
[response.md](response.md#active-descriptor-taylor-kernels-2026-11-01).
<!-- Active descriptor Taylor kernels: end -->

<!-- Reference-independent harmonic API: begin -->
### Reference Hessian harmonic postprocessing

Details, units and validation are in [the shared reference](rust-api.md#reference-hessian-harmonic-postprocessing).

## Typed physical boundaries and checked SCF states

Details, units and validation are in [the shared reference](rust-api.md#typed-physical-boundaries-and-checked-scf-states).

### Static correlated hypermagnetizability scope

Details, units and validation are in [the shared reference](rust-api.md#static-correlated-hypermagnetizability-scope).

### Trust-region augmented-Hessian SCF

Details, units and validation are in [the shared reference](rust-api.md#trust-region-augmented-hessian-scf).

### Restricted-open-shell double-hybrid energy

Details, units and validation are in [the shared reference](rust-api.md#restricted-open-shell-double-hybrid-energy).

## Experimental full frozen scalar action

Details, units and validation are in [the shared reference](rust-api.md#experimental-full-frozen-scalar-action).

### Restricted and unrestricted analytic nuclear Hessians

Details, units and validation are in [the shared reference](rust-api.md#restricted-and-unrestricted-analytic-nuclear-hessians).

### Fixed-endpoint affine scalar geometry

Details, units and validation are in [the shared reference](rust-api.md#fixed-endpoint-affine-scalar-geometry).

## SCF occupation-order diagnostics

Details, units and validation are in [the shared reference](rust-api.md#scf-occupation-order-diagnostics).
