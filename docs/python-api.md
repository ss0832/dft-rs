<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Python API

Start with the [API guide](api-guide.md) to choose a calculation, compare
interfaces, run an example and interpret units and convergence. This manual
contains the detailed contracts and reference-specific limits.

[Calculation table](#table-3-native-python-and-ase) · [Function signatures](#complete-python-and-ase-declaration-reference) · [Logging](#persistent-calculation-logs)

The common energy, gradient, Hessian, electric moment and orbital routes
and their ASE report getters
accept `xc_terms=[{"xc": "pbe0", "weight": 0.8}, {"xc": "hf", "weight": 0.2}]`
for molecular RKS, UKS, ROKS and GKS references. Pass `units="atomic"` to the ASE report getter
to compare directly with native results. See the
[weighted XC contract](rust-api.md#weighted-molecular-xc) for the
operator scope and residual units.

The same array reaches `periodic_energy`, `stress`,
`calc.get_periodic_energy_report` and the standard ASE stress calculation
through `DftRs(xc_terms=terms, ...)`. Use `reference="rks"`,
`"uks"`, `"roks"` or `"gks"` with the applicable reference controls.
See the [periodic contract](rust-api.md#weighted-periodic-xc).

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


Dynamic, nonlinear electric/nuclear molecular response and finite-q periodic density and
dielectric reports are documented in the
[aligned response API tables](python-api.md#table-3-native-python-and-ase), including
matching native/ASE methods, arguments, return keys, and units.

| Surface | Units | Purpose |
| --- | --- | --- |
| `dft_rs.Molecule`, `dft_rs.Crystal`, `dft_rs.MolecularCalculator`, `dft_rs.PeriodicCalculator` | Hartree and Bohr | Structure and calculator objects of the common driver |
| `dft_rs.native` | Hartree and Bohr | Functions over the compiled core |
| `dft_rs.DftRs` | eV and Angstrom | Optional ASE calculator |

These are input/energy conventions. Individual report fields can use MHz,
ppm, cm^-1, kelvin or dimensionless units. Read `report.unit(key)` or the
per-entry `units` mapping; ASE converts labelled energy/length quantities
and preserves other labels. Its Cartesian force is minus the energy gradient.

Registry discovery and callable signatures describe entry points. They do
not certify all reference/functional combinations or physical accuracy.

The experimental retained-center/Euler interface reuses
`dft_rs.evaluate_reduced_fragment(..., model=record,
fragment_mode="rigid-evaluate", derivative_order=1)` and
`DftRs.get_reduced_fragment` with the same controls. It requires an explicit
`rigid-static-core-v1` primitive record and molecular restricted PBE. Order
zero returns energy; order one adds center forces and Euler energy covectors;
order two requests center/center, center/angle and angle/angle Hessian blocks.
The default order is one. Native positions use bohr, ASE positions use
Angstrom, and opaque record coefficients and Euler angles remain atomic units
and radians. Common CPU/memory and run-local logging controls are forwarded.

The actual local wheel installs. Native and canonical/installed-console
energy/gradient, rejection and logging contracts pass their bounded checks;
the completed ASE cases cover rejection and nonzero unit conversion. A
successful direct ASE model evaluation was not completed. The full public-Hessian
checks were not completed before work was stopped; order two remains an
experimental compiled connection, not a passed public-runtime milestone.
All 12 backend gradient components have independent checks. Two of 144 ordered
gradient-difference Hessian checks remain unresolved, despite independent
mixed-energy witnesses for those entries. See the
[model and derivative scope](rust-api.md#continuous-chemical-species-model-contracts)
for the exact completed/cancelled coverage and distinct center/angle units.
Parameter Hessians, `few-evaluate` forces and private generated-record versions
are separate contracts; v2/v3 records are not implicitly accepted here.
Static hyperfine/g-tensor requests retain the collinear auto/UKS/UHF route
and now dispatch explicit ROKS/ROHF choices to a dedicated common-orbital
common-origin route. GKS/GHF require `epr_projection="single-kramers-pair"`
and an odd-electron doublet seed. This mean-field effective-spin approximation
does not purify spin or guarantee an isolated doublet. Dedicated London and
other unavailable combinations remain explicit errors; support and numerical
acceptance are reference/model specific. Optional hyperfine `nuclear_order=1/2`
selects the separate bounded HF-collinear nuclear-A adapter, not a general
g-tensor or all-reference nuclear Hessian. The separate g-tensor
`nuclear_order=1/2` adapter uses only integer all-electron HF/UHF with London
orbitals and the one-electron Pauli model. Its actual Cartesian derivative
shapes and inverse-length units are documented in the g-tensor row. Nonzero
electric field, stability-followed, RO/spinor and DFT nuclear-g references
remain explicit capability errors. Methyl PBE/GKS and full PBE nuclear-A
acceptance remain unfinished; the scientific batches are stopped for the
v0.3.0 API/distribution milestone.
The installed local `dft-rs-python` 0.3.0 wheel passes all 14 bounded HF
nuclear-A public cases, covering native reports, both ASE unit modes, signed
nuclear g factors, ordered targets, both CLI names and failure semantics.
The static-reference dispatch tests select the dedicated provider; they do
not certify an untested molecule, functional or nuclear response reference.
A ground-state nuclear energy Hessian is distinct from an EPR property's
nuclear Hessian. Experimental continuous-species records retain their
generation/application functional, basis and reference restrictions; see
[the current model limits](response.md). No universal transfer is claimed.
The [continuous-model contract](rust-api.md#continuous-chemical-species-model-contracts)
separates all-electron real-charge calculations from frozen-valence
substituent records, explains the 51 coefficients versus three mixture
coordinates and the E0 energy ledger, and records the failed/pending molecular
comparisons. A named `pseudo_sub` request, a generated methyl record and a
saved few-center record are distinct interfaces with distinct capabilities.

Hessian entry points also have different contracts. The legacy
`native.hessian(...)` can select its documented numerical fallback when a
functional derivative is unsupported and labels it with `method`.
`native.hessian_report(...)`, `run_feature("hessian", ...)` and the explicit
reference branch use the common analytic driver and refuse unavailable
analytic models. Do not infer an analytic method from a function name or
from the presence of a matrix alone.

The Python surface covers energies, forces, Hessians, vibrations, Raman,
geometry optimization and saddle search, NMR, charges, wavefunctions,
NAO/NPA, periodic SCF, external fields, continuum solvation, static
polarizability, TDDFT/ECD and SAPT. Analytic excited-state gradients use
`td_state_gradient` on the common driver. Each calculation has an explicit
contract and is mapped across the surfaces in the aligned API tables. [scope.md](scope.md#10-surfaces-what-is-reachable-from-where)
has the full matrix.

The public functions are grouped by intent:

| Group | Functions |
| --- | --- |
| Ground state | `calculate`, `hessian`, `vibrations` |
| Geometry | `optimize`, `saddle` |
| Spectroscopy/response | `raman`, `anharmonic`, `polarizability`, `dynamic_polarizability`, `dynamic_electric_tensor`, `dynamic_electric_process`, `stability`, `stabilize`, `orbital_energy_gradients`, `frontier_gap_gradient`, `dynamic_beta`, `dynamic_gamma`, `electric_field_taylor`, `mixed_electric_field_response`, `static_electric_tensor`, `beta`, `gamma`, `nmr`, `tddft` |
| Interaction energy | `sapt` |
| Periodic | `periodic`, `periodic_scf`, `periodic_density_response`, `periodic_dielectric`, `periodic_thermal_response`, `periodic_electrostatics`, `periodic_basis`, `periodic_capabilities`, `monkhorst_pack` |
| Discovery | `basis_sets` |

All are importable from either `dft_rs` or `dft_rs.native`. The latter
name makes the atomic-unit contract explicit; `dft_rs.DftRs` is the sole
unit-converting ASE surface.

## Persistent calculation logs

Native calculations and explicit `DftRs` report getters accept keyword-only `log_file`,
`log_mode` and `log_level`, together with the existing resource keywords.
`log_file` accepts a text path or `os.PathLike`; its parent must exist.
Files append records tagged with independent run identities. The default
creates no file. A file alone selects file logging; `log_mode="both"` also
prints progress on stderr. Levels are `error`, `warn`, `info` and `debug`.
Providing only `log_level` enables stderr progress without a file.

```python
from dft_rs import native, DftRs

report = native.energy([1, 1], [[0, 0, 0], [0, 0, 1.4]], xc="hf",
                       log_file="native.log", log_level="info", cpu_threads=2)
calculator = DftRs(xc="hf", log_file="ase.log", cpu_threads=2)
```

ASE constructor settings cover an entire getter, including its native
subcalls. Standard inherited ASE property methods use these constructor
settings and keep their ASE signatures. Explicit report getters can
override them per call. Logging keeps native unit
labels, while ASE converts returned physical results as usual. All calls
retain their original return types. Logging errors raise `OSError`,
calculation errors still propagate (an additional logging failure is an
`OSError` containing both reasons), and completed records remain available
after nonconvergence or an escaping Python exception. No stdout redirection
or process-global Python logging configuration is installed. Each record
is flushed; no final record is promised after a hard process kill. See the
[shared logging contract](rust-api.md#persistent-calculation-logs) for
levels, concurrent file ownership and I/O limits.

## Structures, calculators and reports

Global resource keywords `cpu_threads=None` and `memory_mb=None` are added by
the public native adapters, including legacy functions with explicit
signatures. For example, `energy(numbers, positions, cpu_threads=4,
memory_mb=512)` and `calculate(..., cpu_threads=4, memory_mb=512)` execute
inside the same Rust resource envelope. Calculator objects forward these
controls normally. `DftRs(cpu_threads=4, memory_mb=512)` applies them around
whole ASE calculations/getters; a getter's explicit keyword overrides its
calculator default without changing stored parameters. An older compiled
extension that lacks this envelope refuses requested limits and requires a
rebuild, rather than silently ignoring them.

Per-call resource keywords apply to the package's dedicated `DftRs.get_*`
report methods. ASE's inherited `get_potential_energy`/`get_forces` methods
retain their ASE signatures; configure resources on the calculator for those
calls.

The [Rust resource contract](rust-api.md#errors-and-threading) defines the
managed allocation scope and exclusions. `memory_mb` is a MiB budget for
registered storage and workspace allowances, **not a process RSS limit**.
Thread limits are per independent calculation. Resource diagnostics use
integer byte counts and thread counts; ASE performs no unit conversion on
these settings.

This self-contained native/object example uses Bohr and explicit method
settings so that legacy defaults cannot change the comparison:

```python
import dft_rs

numbers = [1, 1]
positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]
system = dft_rs.Molecule(numbers, positions)
calc = dft_rs.MolecularCalculator(system, xc="hf", basis="sto-3g",
                                  cpu_threads=1, memory_mb=128,
                                  log_file="object.log")
report = calc.energy()
report.require_converged()
print(report.energy, report.unit("energy"))
```

For new common-driver code, use `MolecularCalculator` or `PeriodicCalculator`
with `calc.run(feature_id, **controls)`. Named methods such as `calc.energy()`
and `calc.dynamic_polarizability(frequency=0.05)` use that same convention;
`calc.features()` lists the available method names and feature IDs.
Use canonical controls from the calculation table below. In particular,
common-driver frequency controls are `frequency`, `frequency_damping` and
`imaginary`. These methods do not translate the argument names of older
standalone helpers. Existing standalone functions retain their signatures
for compatibility; for example, `dynamic_polarizability(omega=...)` also
accepts the canonical frequency controls and reports the resolved values.
ASE users should keep `DftRs` as their unit-converting adapter.

The Python surface has three layers over the same compiled core:

1. Objects. `Molecule(numbers, positions)` and `Crystal(numbers, positions,
   cell, pbc)` are immutable structures in Bohr, with `from_angstrom`,
   `from_xyz` (plain XYZ for a molecule, extended XYZ with `Lattice="..."`
   for a crystal), `from_ase`, `to_ase`, `with_positions` and `displaced`;
   a crystal also has `from_fractional`, `volume`, `fractional_positions`,
   `scaled`, `strained`, `supercell` and `with_cell`.
   `MolecularCalculator(molecule, **controls)` and
   `PeriodicCalculator(crystal, **controls)` store the reference controls
   (`xc`, `basis`, `k_mesh`, ...). Every calculation of the common driver
   that applies to the structure is a method named like its function below
   (`calc.energy()`, `calc.gradient()`, `calc.phonons()`, ...);
   `features()` lists them, `run(feature, **controls)` runs one by its ID,
   keyword arguments override the stored controls for one call, and
   `with_structure` / `with_controls` return modified copies. A periodic calculator adds `relax(cell=False)`, which returns the relaxed
   `Crystal` and the report, and `scan_volumes`; a molecular calculator adds
   `relax()`, which returns the relaxed `Molecule` and the result of the
   dedicated optimizer. `dft_rs.calculator(structure, **controls)` picks the
   calculator by the type of the structure.
2. Driver functions. `run_feature(feature, numbers, positions, cell=...,
   **controls)` and the functions named after the calculations
   (`energy`, `gradient`, `periodic_energy`, `phonons`, ...) take the same
   controls; `run_feature(feature, structure, **controls)` accepts a
   `Molecule` or a `Crystal` in place of the arrays. `dft_rs.features(space)`
   returns the registry of calculations as data.
3. Dedicated functions (`calculate`, `hessian`, `optimize`, `tddft`,
   `periodic_scf`, ...) with their own signatures and return values, kept
   for the quantities that have no driver calculation yet and for existing
   callers.

Every driver calculation, and every dedicated function that returns a dictionary, returns a `dft_rs.Report`: a dictionary whose
entries are also attributes (`report.energy`, `report["energy"]`), with
`report.unit(key)` and `report.require_converged()`. Driver reports store
per-entry labels in the `units` dictionary. Compatibility reports may retain
a summary string in `units` and store per-entry labels in `unit_labels`;
`report.unit(key)` reads either format. Reports are in atomic units unless a key names
another unit; `dft_rs.units` holds the conversion factors of the core
(`HARTREE_IN_EV`, `BOHR_IN_ANGSTROM`, `HARTREE_PER_BOHR3_IN_GPA`, ...).

```python
import dft_rs

water = dft_rs.Molecule.from_xyz("water.xyz")
calc = dft_rs.MolecularCalculator(water, xc="b3lyp", basis="def2-svp")
print(calc.energy().energy, calc.dipole().dipole)

crystal = dft_rs.Crystal.from_xyz("silicon.extxyz")
solid = dft_rs.PeriodicCalculator(crystal, xc="pbe", basis="def2-svp", k_mesh=(4, 4, 4))
relaxed, history = solid.relax(cell=True)
stress = solid.with_structure(relaxed).stress()
print(stress.pressure_gpa)
```

## Install and build

The development source targets v0.3.0. Python distribution metadata derives
its version from Cargo rather than a second version constant. At runtime,
`dft_rs.__version__` and `dft_rs.native._core.__version__` identify the linked
extension; the feature-registry report and CLI `--version` must agree.
Acceptance rejects an older extension or CLI staged against newer source.
Historical frozen builds keep their original versions and validation scope.

The extension is a package submodule (`dft_rs.dft_rs_core`). The repository's
PEP 517/660 backend wraps maturin and audits source policy, declared MSRV,
distribution identity and private paths:

```console
python -m pip install -e .
python -m pip wheel --no-deps . --wheel-dir dist
python -m pip install "maturin>=1.8,<2.0"
python -c "from tools.build_backend import build_sdist; print(build_sdist('dist'))"
```

The Python distribution is `dft-rs-python`, with normalized artifact names
such as `dft_rs_python-0.3.0-<tags>.whl` and `dft_rs_python-0.3.0.tar.gz`.
The import remains `dft_rs`; Rust uses the `dft-rs` crate. Once a release is
published, use `python -m pip install dft-rs-python`, or
`python -m pip install "dft-rs-python[ase]"` for the optional ASE dependencies.
For a local wheel in `dist/`, use
`python -m pip install --no-index --find-links dist dft-rs-python`.
Local builds and tests do not publish to an index. Installed distribution
discovery uses `importlib.metadata.version("dft-rs-python")`.
Both wheel installation and `python -m pip install .` register the
`dft-rs` console command in the selected Python environment. It calls the
compiled CLI directly through `native.cli_main()`; an explicit argument
list remains supported, and the no-argument form reads `sys.argv[1:]`.
See [the CLI installation instructions](cli.md) for PATH and exit-status
conventions. Rust builds, binaries and integration tests use the package's
`warnings = "deny"` lint policy: compiler warnings fail the build.
Importing `dft_rs` itself does not import ASE or NumPy.

For Python development, install `python -m pip install '.[test]'` and run
`python tools/lint.py` followed by `python -W error -m pytest`. The smaller
`.[lint]` extra installs only Ruff. The lint runner checks all package,
test and tool Python files for syntax and Pyflakes correctness (`E9`, `F`),
returns a nonzero exit status for findings, and writes no cache. Explicit
paths and safe fixes are supported: `python tools/lint.py --fix PATH`.
The ordinary pytest suite also runs the lint gate.

## Native calculations

Molecular `calculate(..., stabilize=True, unrestricted=True)` follows internal
orbital instabilities before computing every returned property, including
analytic forces and the wavefunction. Its `stability` dictionary contains
initial/final diagnostics and accepted descents. The corresponding ordinary
ASE path is `DftRs(stabilize=True, unrestricted=True)`; the same report is
available as `results["stability"]`. See the
[stability conventions](rust-api.md#orbital-stability-and-following-conventions)
for explicit scope, descent defaults and reference-dependent lazy getters.

```python
from dft_rs.native import calculate

result = calculate(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]],
    basis="def2-svp",
    xc="b3lyp",
    gradient=True,
    wavefunction=True,
)

result["energy"]                  # Hartree
result["gradient"]                # Hartree/Bohr
result["dipole"]                  # atomic units
result["mulliken"]                # per-atom charges
result["e_nlc"], result["e_dispersion"]
result["wavefunction"]["mo_energy"]
result["wavefunction"]["mo_coeff"]
result["wavefunction"]["density"]
result["wavefunction"]["overlap"]
```

The following excerpt shows defaults of the dedicated `calculate` helper.
Dedicated helper signatures and common-driver controls are different
contracts; use explicit `xc` and `basis` when comparing surfaces.

```python
calculate(
    numbers,
    positions_bohr,
    basis="sto-3g",
    ecp=None,
    xc="hf",
    dispersion="auto",
    density_fit=None,
    charge=0,
    multiplicity=1,
    grid_level=3,
    vv10_grid_level=3,
    gradient=False,
    wavefunction=False,
    nbo=False,
    max_iter=150,
    conv_tol=1e-9,
    conv_tol_grad=None,
    conv_tol_dp=1e-8,
    incore_mb=2048,
    field=None,
    solvent=None,
    eps=None,
    pseudo_sub=None,
)
```

`pseudo_sub="4=Me"` uses zero-based `atom=name` assignments and the same
parser as CLI `--pseudo-sub`; Me, Et, tBu, Ph, CF3 and OMe are recognized.
Every name currently raises `NotImplementedError` before basis construction or
SCF, while malformed specs, non-H site atoms and out-of-range indices raise
`ValueError`. This is intentional: no provenance-bearing fitted operator has
passed its physical acceptance gates, and no pseudo-site Hamiltonian consumer
exists yet.

`nbo=True` returns the molecular NAO/NPA foundation: natural atomic orbital
occupations and AO coefficients, atom populations/charges/spins, the density
in the NAO basis, and Wiberg bond indices. NHO construction, Lewis-structure
search, localized NBOs, and donor-acceptor analysis are not claimed yet.

`conv_tol_grad=None` follows the PySCF convention and uses
`sqrt(conv_tol)`. `density_fit="def2-universal-jfit"` replaces Coulomb J by
RI-J while exchange K remains exact. RI-J gradients and Hessians use
geometry-rebuilt finite differences of the RI total energy because analytic
auxiliary-basis nuclear derivatives are not yet implemented.

`dispersion="auto"` enables the defining VV10 term of B97M-V, wB97X-V and
wB97M-V, and the CHG tail of wB97X-D. Explicit models are `none`, `d2`,
`d3zero`, `d3bj`, `chg`, and self-consistent `vv10`. Adding another model to
a functional with built-in dispersion is rejected to prevent double counting.

### External field and continuum solvation

```python
result = calculate(
    [8, 1, 1], water_bohr, basis="def2-svp",
    field=[0.0, 0.0, 1e-3],   # a.u.; adds +E.r to h_core and -Z_A R_A.E to E_nuc
    solvent="smd:water",       # same grammar as the CLI's --solvent
)
result["e_pcm"]       # PCM electrostatic polarization energy (<= 0), already
                       # folded into `energy` self-consistently
result["e_smd_cds"]   # SMD's non-electrostatic term, added on top of `energy`
```

`field` composes with everything `calculate` already returns: the energy,
dipole and (with `gradient=True`) the analytic gradient are all exact in the
field, because `ScfOptions::external_field` is applied once and every one of
those quantities already reads it off the perturbed Fock operator. `field`
and `solvent` are parameters of `calculate` only — `hessian`, `vibrations`,
`polarizability`, `tddft` and `sapt` do not accept them in this API (the
CLI's `--field`/`--solvent` reach further, onto `hessian`, `frequencies`,
`polarizability`, `raman` and `tddft`; that breadth is not mirrored here yet).

`solvent` reuses the CLI's `--solvent` grammar exactly rather than growing a
second one: a bundled name (`"water"`, 179 entries), optionally prefixed
`"smd:"`, `"cosmo:"`, `"ief:"` or `"ssvpe:"` to pick the continuum model
(default C-PCM), or `"custom"` together with `eps=` for a bare dielectric
constant. The Python API does not expose the CLI's per-descriptor overrides
(`--solvent-alpha`/`-beta`/`-gamma`/`--refractive-index`), so `"smd:custom"`
without a named table entry gets electrostatics only. The reaction field is
self-consistent, so it is exact at the converged density including under
`gradient=True`.

### Basis and ECP specifications

The orbital basis and ECP are independent specifications. Both accept a
bundled name, a Gaussian94 file path, or per-element assignments:

```python
calculate(z, r, basis="def2-tzvp; H=def2-svp")
calculate(z, r, basis="def2-tzvp", ecp="Ni=stuttgart-rsc-1997")
calculate(z, r, basis="lanl2dz", ecp="none")
```

Atomic numbers must be 1 through 118. A bare `Z=0` is not a valid ghost
specification because it does not say which element's basis to retain; it is
rejected rather than silently producing an ill-defined counterpoise result.

### Hessians and vibrations

```python
from dft_rs.native import hessian, vibrations

h = hessian([1, 1], [[0, 0, 0], [0, 0, 1.4]], xc="svwn5")
h["hessian"]
h["method"]       # "analytic" or "numerical"

v = vibrations([8, 1, 1], water_bohr, xc="b3lyp", basis="def2-svp")
v["frequencies"]
v["ir_intensities"]
v["n_imaginary"], v["n_projected"], v["method"]
v["thermochemistry"]  # ZPE, U/H/G corrections and entropy for a minimum
```

Supported empirical dispersion and uniform external fields remain on the
analytic Hessian path. A specifically unsupported functional or SCF
derivative falls back numerically. SCF, response, resource, and linear-algebra
failures propagate and are never hidden by fallback.

That fallback belongs to the legacy native call above. Supplying an explicit
`reference` or additional driver controls selects the common analytic driver.
`hessian_report` and ASE's lazy `get_hessian` also use the common driver and
propagate unsupported analytic combinations. The native compatibility `method`
alias is added only after checking the backend's analytic provenance; a missing
or contradictory marker is an error. ASE returns eV/Angstrom squared and caches
a copy without concealing a numerical fallback. The method matrix and its
bounded validation evidence remain in `property_targets.csv` and `response.md`.

Excited-state reports distinguish the total excited-state energy Hessian
(`hessian`) from the excitation-energy-only contribution (`excitation_hessian`)
and the ground-state contribution (`ground_hessian`). Their quantity labels
survive ASE conversion; total and excitation curvatures are not interchangeable.

`vibrations(..., temperature=298.15, pressure=101325.0,
symmetry_number=1)` controls the RRHO standard state. The electronic
degeneracy follows the molecular multiplicity.

`thermochemistry` is `None` when an imaginary mode is present. Otherwise
it is a dictionary with the following stable schema:

- metadata: `temperature` (K), `pressure` (Pa), `symmetry_number`,
  `electronic_degeneracy`, and `is_linear`;
- energy corrections (Hartree): `zero_point_energy`,
  `vibrational_thermal_energy`, `translational_energy`,
  `rotational_energy`, `internal_energy_correction`,
  `enthalpy_correction`, and `gibbs_correction`;
- entropies (Hartree/K): `entropy_translational`, `entropy_rotational`,
  `entropy_vibrational`, `entropy_electronic`, and `entropy`;
- totals (Hartree): `internal_energy`, `enthalpy`, and
  `gibbs_free_energy`.

Zero or imaginary frequencies are never silently omitted from an RRHO sum.
Transition-state thermochemistry requires an explicit convention and is
therefore left to the caller.

### Geometry and transition states

```python
from dft_rs.native import optimize, saddle

minimum = optimize(numbers, positions_bohr, xc="b3lyp")
ts = saddle(numbers, ts_guess_bohr, xc="b3lyp", method="rsirfo")
```

The saddle methods are `"prfo"` (compatibility default) and `"rsirfo"`.
RS-I-RFO uses the coupled Householder image function and an intrinsic
restricted RFO step. `step_max` is its Euclidean trust radius in Bohr; the
generalized RFO metric is solved until the step lies on that sphere. It is
not a post-hoc scale factor.

`optimize` returns final `positions`, `energy`, `converged`, `iterations`,
and `trajectory`. `saddle` adds `method` and `n_negative`. Every trajectory
frame contains `step`, `positions`, `energy`, and `grad_max`; when a path is
given, frames are flushed during the walk.

### Raman and NMR

```python
from dft_rs.native import nmr, raman

spectra = raman(numbers, positions_bohr, xc="b3lyp")
spectra["frequencies"], spectra["ir_intensities"]
spectra["raman_activities"], spectra["hessian_method"]

shielding = nmr(numbers, positions_bohr, xc="pbe", gauge="giao")
shielding["shieldings"][0]["tensor"]
shielding["shieldings"][0]["isotropic"]
```

Raman frequencies are cm^-1, IR intensities km/mol, and activities
Angstrom^4/amu. NMR total/diamagnetic/paramagnetic tensors and isotropic
values are ppm. Open-shell, ECP and unsupported nonlocal NMR combinations
raise `NotImplementedError`.

### Polarizability

```python
from dft_rs.native import polarizability

r = polarizability([8, 1, 1], water_bohr, basis="def2-svp", xc="b3lyp")
r["alpha"]        # 3x3 nested list (a.u.), symmetric to numerical precision
r["isotropic"]    # trace / 3
```

Analytic CPKS with respect to a static electric field; the gate on the Rust
side is the conjugacy identity `dE/dE_a = -mu_a`.

### TDDFT and ECD

```python
from dft_rs.native import tddft

r = tddft([8, 1, 1], water_bohr, basis="def2-svp", xc="b3lyp",
          nstates=5, method="tda", spin="auto")
r["ground_state_energy"]
for state in r["states"]:
    state["energy"], state["energy_ev"]
    state["oscillator_strength"]
    state["rotatory_strength_length"], state["rotatory_strength_velocity"]
```

`method` is `"tda"` (Tamm-Dancoff, default) or `"rpa"` (full TDDFT/TDHF, the
Casida A/B form). `spin` is `"singlet"`/`"triplet"` for a restricted
reference, `"spin_resolved"` for an unrestricted one, or `"auto"` (matches
the CLI's default: singlet unless the reference itself is unrestricted).
Rotatory strengths are atomic units; multiply by `471.4438` for the
customary `1e-40` cgs unit. The length gauge is origin-dependent in a finite
basis and the velocity gauge is not — on an achiral molecule both vanish, so
their gap elsewhere is a basis-set error bar, not noise.

### SAPT

```python
from dft_rs.native import sapt

r = sapt(
    numbers_a, positions_a,
    numbers_b, positions_b,
    basis="aug-cc-pvdz", xc="hf",
    multiplicity_a=1, multiplicity_b=1,
)
r["e_elst"], r["e_exch"], r["e_ind_a"], r["e_ind_b"]
r["e_exch_ind"], r["e_disp_coupled"], r["e_exch_disp"]
r["interaction_total"]          # sum of the seven terms above
r["e_int_supermolecular"], r["delta"]
```

`sapt` uses the default `response` model. To select a formal HF variant,
pass the combined geometry and fragment labels to `sapt_analysis` or
`calc.get_sapt_analysis`.

```python
from dft_rs import sapt_analysis

r = sapt_analysis(
    [8, 1, 1, 8, 1, 1], positions_bohr,
    xc="hf", basis="6-31g", fragments=[0, 0, 0, 1, 1, 1],
    sapt_variant="sapt0",
)
```

Unlike the CLI (one XYZ split by `--natoms-a`), the two monomers are
separate geometries here, each with its own charge and multiplicity; basis,
ECP spec and functional are shared. An ECP request raises
`NotImplementedError` because the intermolecular ECP operators are missing.
A monomer whose first-order orbital rotation leaves the small-rotation regime
raises `ValueError` because that reference is not a valid input to the
second-order decomposition. Neither case returns silent partial components.

## Common driver

`dft_rs.run_feature(feature, numbers, positions, *, cell=None, pbc=None,
**controls)` runs any calculation registered with the
[common driver](rust-api.md#common-calculation-driver); named wrappers (`energy`, `free_energy`,
`gradient`, `charges`, `dipole`, `orbitals`, `molecular_multipole`,
`multipole_gradient`, `hessian_report`, `frequencies_report`,
`vibrational_spectra`, `polarizability_report`, `conceptual_dft`,
`spin_conceptual_dft`, `feature_registry`) call it. Positions are Bohr and
reports are atomic units unless a key names another unit. The ASE
calculator's `get_feature_report(feature, atoms, **controls)` and named
`get_*_report` methods convert the geometry from Angstrom, take reference
controls from the calculator parameters and return the same report.

Ordinary integrating molecular and periodic driver calculations accept the grid controls
`grid_level=0..9` (default 3), `grid_preset="fine"|"ultrafine"|"superfine"`
(named per-atom (radial, Lebedev) grids (75, 302), (99, 590) and
(150 for H–Ne / 225 beyond, 974), replacing the level) and
`grid_angular_scheme="lebedev"|"product"` (default `lebedev`) and
`grid_weight_cutoff` (Bohr^3, default 1e-14; points with a smaller
partitioned weight are dropped, `0` keeps every point), for example
`run_feature("energy", numbers, positions, xc="pbe", grid_preset="ultrafine")`;
ASE calculators take the same keywords as calculator parameters.

These controls do not describe every numerical grid. VV10 has independent
`vv10_grid_level=3`; NEO electron/EPC integration shares the molecular grid,
while the quantum Gaussian basis has separate exponent/angular controls.
PBC k/q meshes, open-tail cutoffs and magnetic tensor-product quadrature must
be converged separately. Pure analytic HF response and supplied-data
post-processing can be grid-free; HF Ramsey diamagnetic coupling is an
exception that still needs quadrature. Legacy dedicated signatures may
expose `grid_level` without named presets; unsupported keywords are not
silently accepted. See the complete [quadrature contract](conventions.md#quadrature-grids)
for defaults, route-specific exceptions and the frozen continuous-model limits.

Calculator parameters provide the base settings; explicit per-call controls
override them. The call's `units` mode applies to the new controls and report,
while stored calculator parameters retain the calculator's own unit mode.
Integer charge, multiplicity, iteration/storage counts and k meshes reject
fractional and Boolean inputs instead of silently rounding them. NumPy integer
scalars remain supported. Controls for SCF or orbital response are inapplicable
to pure supplied-data post-processing; they do not create an electronic
calculation for such a route.

Common molecular report getters share one translation of calculator SCF
parameters. `level_shift`, SCF `damping`, `diis_size` and `use_adiis=False`
(the driver control `no_adiis=True`) reach the reference setup; ordinary
ASE energy/force requests select that common route when these parameters
are changed. A legacy getter that cannot forward them raises an explicit
unsupported-property error. Explicit additional controls such as an ADIIS
cutoff are forwarded for backend applicability validation rather than dropped.
For dynamic polarizability, the historical `damping` argument is the optical
linewidth in energy units. The wrapper-specific `scf_damping` argument is
dimensionless SCF mixing, so the two settings can be selected independently.

## Exception mapping

| Python exception | Meaning |
| --- | --- |
| `ValueError` | malformed geometry/options, unknown control names or reference selectors, or a physically invalid reference |
| `TypeError` | Python argument binding or a noniterable supplied where a vector is required, before Rust validation |
| `NotImplementedError` | unavailable calculation, reference/feature combination, common-driver adapter, derivative order, or known control unsupported in this context |
| `RuntimeError` | SCF/response convergence, resource, or linear-algebra failure |
| `OSError` | filesystem error |

These exceptions pass through the native wrappers and ASE unchanged. A
nonempty unavailable feature ID raises `NotImplementedError`; an empty ID is
an input error. Registry rows marked `dedicated` remain accessible through
their named functions, but cannot be invoked with `run_feature`. A known
control supplied to a calculation that does not use it is rejected, while a
misspelled control remains a `ValueError`. Unhandled reference selectors are
not silently replaced by an automatic collinear calculation. This capability
contract does not turn convergence failures into missing implementations or
claim support for unvalidated reference/derivative combinations.

Supply required inputs before testing capability: for example, `ecd-spectrum`
requires positive `broadening_ev` and the supported `td_method="tda"` path.
Missing required inputs remain input errors even if another option is also
unsupported. Native `run_feature(..., pbc=...)` takes three flags such as
`[True, True, True]`, not ASE's scalar `pbc=True` shorthand. With `pbc=None`,
a supplied cell defaults to three periodic axes; without a cell it defaults
to an isolated system.

## Periodic foundation

```python
from dft_rs.native import monkhorst_pack, periodic

cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
kpts = monkhorst_pack(cell, [2, 2, 2])
r = periodic(
    [1, 1],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
    cell,
    k_points=kpts,
)
```

The result contains Bloch-summed overlap, kinetic, nuclear-attraction, and
core-Hamiltonian matrices as separate real/imaginary arrays, plus Ewald
nuclear energy and stress. With `derivatives=True` it also contains nuclear
derivatives of the one-electron matrices.

Supplying a finite symmetric `nao x nao` Gamma-point `density` additionally
returns `hartree` at every requested k point and `density_electron_count`.
The Rust API accepts the general real-space `D(T)` image representation.
This low-level `periodic` surface remains useful for operator inspection. For
a periodic total energy and analytic forces, use the finite-k SCF surface:

```python
from dft_rs.native import periodic_capabilities, periodic_scf

caps = periodic_capabilities()

r = periodic_scf(
    [2],
    [[0.0, 0.0, 0.0]],
    [[6.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]],
    xc="r2scan",
    k_mesh=[3, 1, 1],
    pbc=[True, True, True],
    electronic_temperature=300.0,
    guess="sad",
    level_shift=0.2,
    damping=0.20,
    diis_size=10,
    overlap_cutoff=6.1,
    forces=True,
)
assert r["force_method"] == "analytic"
```

Fixed-cell atomic optimization uses the same entry point and force path:

```python
opt = periodic_scf(
    [1, 1],
    [[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]],
    [[18.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 18.0]],
    xc="hf",
    relax=True,
    opt_gtol=3e-4,
    opt_max_steps=100,
    trajectory="h2-periodic-opt.xyz",
)
assert opt["optimization_cell"] == "fixed"
```

The result includes `optimized_positions`, `optimization_converged`,
`optimization_iterations`, and the minimized objective. Lattice vectors do
not move; cell optimization remains unavailable until total electronic stress
is implemented. Supplying optimization controls without `relax=True` raises a
`ValueError` instead of silently ignoring them.

The SCF supports HF, global/range-separated exact exchange, and
LDA/GGA/meta-GGA XC. Zero temperature gives integer occupations; a positive
temperature enables mesh-global Fermi-Dirac occupations and returns the
internal energy, Mermin free energy, entropy term, chemical potential, and
per-spin/k-point occupations. Periodic ECP energies and analytic force
components use explicit translated-center image sums. Their derivative
matrices have direct finite-difference gates; the full multi-SCF
analytic-force/total-energy-difference comparison remains an explicitly
ignored long-running validation gate.
At positive electronic temperature, analytic forces differentiate the Mermin
free energy (reported as `force_energy="free_energy"`), not the internal
energy with its non-variational entropy omission.
`use_adiis` and `adiis_until_error` control the integer-occupation complex
finite-k ADIIS stage; finite-temperature calculations deliberately use CDIIS.
The ASE calculator forwards these settings together with `level_shift`,
`damping`, `diis_size`, and `guess`.
Partial-PBC electronic SCF, empirical dispersion, and periodic VV10/CHG remain
typed capability refusals. The independent [dimensional Coulomb](periodic.md#1-boundary-conditions-and-electrostatics)
and [electronic basis/grid](periodic.md#2-electronic-basis-and-integration-grid) drivers execute the implemented
1D/2D/3D foundations through `periodic_electrostatics` and `periodic_basis`,
also exported at package level. `DftRs.get_periodic_electrostatics` and
`DftRs.get_periodic_basis` provide the corresponding atomic-unit ASE reports.

## ASE calculator

The optional ASE dependencies are installed with `dft-rs-python[ase]` for
a published package or `python -m pip install -e ".[ase]"` for a checkout.
For the same geometry as the native example, explicitly convert Bohr:

```python
import numpy as np
from ase import Atoms
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
atoms = Atoms("H2", positions=positions * BOHR_IN_ANGSTROM)
atoms.calc = DftRs(xc="hf", basis="sto-3g", cpu_threads=1, memory_mb=128,
                   log_file="ase.log")
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Angstrom, shape (2, 3)
```

Units follow one rule: Rust and the native Python layer use atomic units,
the ASE calculator uses eV and Angstrom. Energies, forces, stress, dipole
and the Hessian are converted at the calculator, and the report getters
(`get_phonons`, `get_elastic`, `get_energy_report`, ...) convert every entry
of the driver report whose unit label names Hartree or Bohr and rewrite the
labels in `report["units"]`; entries labelled in other units (cm^-1, GPa,
K) and unlabelled entries are returned as they are, and
`report_units="atomic"` (calculator parameter or keyword of one call)
returns the driver report unchanged. The dedicated getters of low-level
operators and response tensors (`get_dimensional_scf`,
`get_dynamic_polarizability`, ...) pass their atomic-unit results through,
as their docstrings state, and controls passed to the driver (cutoffs in
Bohr, tolerances in Hartree/Bohr) keep the driver's units.

```python
from ase.build import molecule
from dft_rs import DftRs

atoms = molecule("H2O")
atoms.calc = DftRs(
    basis="def2-svp",
    xc="pbe0",
    dispersion="d3bj",
    conv_tol=1e-10,
)

energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
charges = atoms.get_charges()          # Mulliken, electron-charge units
hessian = atoms.calc.get_hessian(atoms)  # eV/Angstrom^2, cached
vib = atoms.calc.get_vibrations(atoms)   # cached
```

For periodic input, the same calculator forwards all three PBC flags to
`periodic_scf`; set `k_mesh`, `electronic_temperature`, `overlap_cutoff`,
`exchange_cutoff`, `grid_shell`, and the periodic convergence controls as
calculator parameters. Energy and
analytic forces are converted to ASE units. Pending partial-PBC electronic
SCF connections produce the same typed refusal as the Rust/Python-native API. Periodic
dipoles, Mulliken charges, Hessians, and vibrations are not reported.

Energy-only requests do not calculate gradients. If forces are requested
after an energy-only call, a second native calculation is required because
the current API does not retain an SCF object between calls. ASE caches the
new result for the unchanged geometry. Hessians and vibration analyses have
separate geometry-and-parameter caches and return copies of cached arrays.

## Errors and threading

Invalid inputs, unknown bases and functionals raise `ValueError`; filesystem
errors raise `OSError`; unsupported analytic derivatives raise
`NotImplementedError`; convergence, resource, and numerical failures raise
`RuntimeError`.

Rust kernels use Rayon internally (`RAYON_NUM_THREADS` controls its pool).
Native calls are synchronous and currently retain the Python GIL for their
duration.




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
The ordinary lazy ASE `get_hessian` also preserves explicit reference
constraints and custom correlated targets through the shared driver.
ROKS accepts common response controls on energy/gradient requests as well
as Hessian/polarizability requests. They are validated in every case; only
the latter two require orbital-response equations. ROKS Hessian convergence
also includes its accompanying electric-response calculation.

## Table 3: native Python and ASE

[Choose a calculation by property](api-guide.md#find-a-calculation) / [Find a native function or ASE getter](api-guide.md#choose-an-interface) / [Check input and output units](api-guide.md#choose-units-for-inputs-and-reports)

Native functions are exported by both `dft_rs` and `dft_rs.native`. Native
positions and lattice vectors use Bohr. ASE methods on `DftRs` convert their
`Atoms` geometry from Angstrom to Bohr and return these response reports in
the reported response units. These methods preserve these units; they do not
use the eV/Angstrom conversion of ordinary ASE energy and force properties.

The shared molecular reference report gives the raw SCF stationarity
residual as `scf_orbital_residual` for collinear references or
`orbital_residual` for GKS. Native Python and ASE with `units="atomic"`
give this residual in Hartree.

| ID | Native Python call | Corresponding ASE method | Value keys in returned dictionary |
|---|---|---|---|
| `continuous-species` | `native.continuous_species(numbers, positions, fixed_electron_count=14, charge_directions=[[-1,1]], charge_coordinates=[0.25], charge_bounds=[[-1.5,1.5]], **controls)`; `generate_methyl_fragment(..., fragment_spec=...)`; `evaluate_reduced_fragment(..., model=...)` | `DftRs.get_continuous_species(atoms, **controls)`; `DftRs.get_reduced_fragment(atoms, model=...)` | Integer template labels, real physical charges and integer electron counts remain separate. Closed-shell M1, fixed-spin/local LDA/PBE and saved static methyl candidate routes pass the stated public gates. Native atomic units; ASE converts once and preserves opaque native-unit model JSON. Candidate chemical accuracy fails; analytic alpha parameter derivatives remain Rust-only. See response.md.  The common charge, fixed few-center and generated-reduced model reports also return `energy_without_internal_reference` and `classical_energy_without_internal_reference` in Hartree, plus `scf_energy_convention`; state-common constants are added only to reported totals.  Experimental `scalar-evaluate`/`scalar-optimize` add explicit full frozen joint-endpoint actions and site weight optimization; see the model contract below. Saved weight routes pass clean installed checks. `scalar-gradient`/`scalar-geometry-optimize`/`scalar-hessian` require explicit endpoint/state owners; paired RKS and explicitly spin-resolved UKS records remain distinct. Hessian reports include all ordered entries and raw residuals; the stated PBE RKS/UKS fixture and installed-selector gates pass, while real-model minima remain separate. |
| `multicomponent` | `native.multicomponent(numbers, positions, quantum_atoms=[...], **controls)` or `particles=[{...}]`; top-level `dft_rs.multicomponent` | `DftRs.get_multicomponent(atoms, **controls)` | Simultaneous electronic/quantum-particle SCF; optional expectation-position constraints, analytic gradient/Hessian, total static alpha and coupled stability analysis or following (`species_stability`) through the common driver. Native units are atomic; ASE converts values and target positions once. Structured particle/channel and spin-averaged annihilation contracts, units and bounded acceptance are below.  Optional `quantum_mass_response` returns fixed-basis quantum-mass gradient/Hessian and raw coupled diagnostics; units and scoped acceptance are below. The `species_electronic_model="dirac-coulomb"` energy option uses `nuclear_rms_radii` in Bohr for native and Angstrom for ASE by default. The projector reference, decomposed energy and raw coupled residuals accompany the quantum density and Fock matrices. For the Dirac joint reference, `field` returns the total dipole and `species_polarizability=True` accepts `frequency`, `frequency_damping` and `imaginary_frequency`. For a nonrelativistic molecular positron, `field` perturbs the SCF and `species_polarizability=True` returns the static dipole tensor (Bohr^3 natively, Angstrom^3 in default ASE units) and raw response residuals. [Positron benchmark and conditions](response.md#r076-multicomponent-nuclear-positron-photon-dft). With `species_electronic_model=dirac-coulomb`, `dirac_interaction=coulomb\|gaunt\|breit` selects the electron-electron operator for the coupled SCF and electric response. The current direct and exchange terms and Hermiticity defect are in Hartree. [Model and validation](response.md#r081-relativistic-quantum-nuclear-response).  In the fixed-sector relativistic quantum-nuclear model, `species_field_gradient` supplies five traceless symmetric gradient modes (Hartree/e/Bohr^2), and `species_quadrupole_polarizability` requests their coupled response. Named or explicit `origin` is supported, default zero. `electric_quadrupole` reports Buckingham moments (e Bohr^2), static/complex 5x5 response (e^2 Bohr^4/Hartree), and raw conjugate dipole-gradient 3x5/5x3 blocks (e^2 Bohr^3/Hartree); multiplying the latter moment rows by three gives Buckingham response. Eight source residuals, iterations, reference residual, mode order and signed frequency are exposed. Gaussian-product, full derivative and public acceptance are pending. The nonrelativistic route accepts `frequencies`, `frequency_damping` and `species_quadrupole_polarizability` through both wrappers. Use Hartree with native or atomic ASE and eV with physical ASE. [Verification status](response.md#r076-multicomponent-nuclear-positron-photon-dft). The nonrelativistic route accepts `species_field_gradient` with five mode amplitudes. Native units are Hartree/e/Bohr^2 and physical ASE units are eV/e/Angstrom^2. |
| `casscf` | `native.casscf(numbers, positions, cas=[n_electrons, n_orbitals], **controls)`; top-level `dft_rs.casscf` | `DftRs.get_casscf(atoms, cas=[...], **controls)` | Complete-active-space SCF on the common driver; `derivative_order=1/2` adds analytic `gradients`/`hessians`; native atomic units, ASE eV and eV/Angstrom. |
| `mcpdft` | `native.mcpdft(numbers, positions, cas=[...], on_top_functional="tpbe", pdft_method="state-specific", **controls)`; top-level `dft_rs.mcpdft` | `DftRs.get_mcpdft(atoms, cas=[...], **controls)` | Multiconfiguration pair-density functional theory; hybrid fraction and multistate forms; `derivative_order=1/2` adds analytic state-specific `gradients`/`hessians`; native atomic units, ASE eV. Energy acceptance: p230 (8 tests), p232 (9 frozen total-energy cases and 20 term comparisons; maximum term defect 4.418e-7 Hartree). Native gradients/Hessians use Hartree/Bohr and Hartree/Bohr^2; ASE uses eV/Angstrom and eV/Angstrom^2. Axes are [state][atom][axis] and [state][3 atom + axis][3 atom + axis]. derivative_stationarity is the largest remaining orbital/CI residual in Hartree; derivative_solve_residual is dimensionless. The transported orbital/CI Taylor path uses the common matrix-free GMRES and orthogonal retraction. Local Hessian grid convergence and open-active GGA derivative combinations require their recorded acceptance in response.md. Four CASSCF/MC-PDFT public-route cases pass, including gradients/Hessians and atomic/ASE conversions (path stationarity 3.641e-12 Hartree, solve residual 3.380e-13). MC-PDFT derivative state axes follow state_total in reference-state order, while pdft_energies are sorted; CASSCF derivatives follow cas_energies. |
| `superconducting-state` | `superconducting_state(**inputs)` | `DftRs.get_superconducting_state(**inputs)` | No Atoms. `pairing_model="lm2005"` generates K/Z; other models require explicit Z. Native Hartree; ASE eV (squared vertices eV squared), unless `units="atomic"`; temperature Kelvin, weights/Z/electron count dimensionless. Identical common-driver coefficients and raw residuals `pairing_model="nambu"` evaluates full complex interband normal/anomalous density on supplied H/Delta/S, at positive temperature and supplied chemical potential. This model is grand canonical and kinematic; the existing supplied/lm2005 band providers retain canonical gap/number/phase self-consistency. `scdft_temperature_bracket=[400,1200]` in Kelvin selects the canonical linear pairing onset. Both Python routes report `critical_temperature_k`, final Kelvin bracket, chemical potential, and raw gap/eigenvector/number residuals. Set `scdft_phase_selection=True` with reciprocal supplied K and uniform Z to compare the canonical phases at one electronic temperature. Both routes return the selected phase, condensation free energy and raw stability margin. For the retarded Nambu path, use signed `frequencies`, optional `frequency_damping`, flattened row-major dimensionless `nambu_probe_normal_*` or `nambu_probe_pairing_*` rows and optional symmetric `nambu_probe_interaction`. This path uses fixed quasiparticle occupations and returns complex susceptibility in inverse Hartree for native Python or inverse eV for default ASE, with axes frequency, observed, driven. Both report raw residuals and stability margin. [Model and verification](response.md#r050-scdft-gap-anomalous-density). |
| `no-pair-projector` | `no_pair_projector(numbers=None, positions=None, **inputs)` | `DftRs.get_no_pair_projector(atoms=None, **inputs)` | Supplied balanced H/S need no Atoms. `dirac_integrals="london-rmb"` generates complex molecular H/S with explicit Gaussian radii, uniform `magnetic_field` and `magnetic_direction` in atomic magnetic units. ASE uses eV and Angstrom unless `units="atomic"`. [Models, units and residuals](rust-api.md#restricted-magnetic-balance-and-a-complex-four-component-metric). |
| `dirac-coulomb-scf` | `dirac_coulomb_scf(numbers, positions, **controls)` | `calc.get_dirac_coulomb_scf(atoms, **controls)` | Fixed-sector finite-nucleus Dirac–Coulomb HF. Native energies in Hartree and radii in Bohr. ASE uses eV and Angstrom unless `units="atomic"`. Raw orbital, Coulomb, metric, density and population residuals. Use `dirac_interaction` to select Coulomb, Gaunt or Breit. Current direct and signed exchange energies and raw Hermiticity defects are in Hartree. Coulomb HF also accepts `dirac_integrals="london-rmb"`, `magnetic_field` and `magnetic_phase_order`. The same London-RMB basis provides optional Gaunt and Breit direct and exchange corrections at finite field. |
| `two-component-relativistic` | `two_component_relativistic(numbers, positions, **controls)` | `calc.get_two_component_relativistic(atoms, **controls)` | Finite-Gaussian-nucleus X2C, DKH2 or matrix ZORA with scalar or spin-orbit SCF. Native reports use Hartree, Bohr, e Bohr and e² Bohr²/Hartree. ASE uses eV and Angstrom unless `units="atomic"`. Use `response_order=1` for complex electric polarizability. Energy, picture-changed operator and SCF/response residuals share the common driver. [Molecular model and verification](response.md#r075-breit-and-two-component-relativistic-dft). |
| `dirac-dks` | `dirac_dks(numbers, positions, **controls)` | `calc.get_dirac_dks(atoms, **controls)` | Fixed-sector four-component KS with `dirac_interaction="coulomb"`, `"gaunt"` or `"breit"`. Current direct and exchange terms and their raw Hermiticity defect are reported separately. Native positions and finite Gaussian radii in Bohr, energies and orbital residuals in Hartree, dipole in e Bohr and optional complex alpha in e² Bohr²/Hartree. ASE uses eV and Angstrom by default. Coulomb references accept `dirac_integrals="london-rmb"`, `magnetic_field` in atomic magnetic units and `magnetic_phase_order`. The London-RMB option supports the current corrections with `dirac_interaction=gaunt` or `breit`. |
| `scdft-pairing-kernel` | `scdft_pairing_kernel(**inputs)` | `DftRs.get_scdft_pairing_kernel(**inputs)` | No Atoms. Native Hartree/Hartree squared; ASE eV/eV squared unless `units="atomic"`; Kelvin temperature. Optional band weights enable dimensionless Z and its coefficients. Identical common-driver K/Z, conventions and raw quadrature diagnostics |
| `dynamic-polarizability` | `dynamic_polarizability(numbers, positions, *, omega=0.0, damping=0.0, imaginary=False, reference=None, unrestricted=False, **options)` | `calc.get_dynamic_polarizability(atoms, *, omega=0.0, damping=0.0, imaginary=False, **controls)` | All nine `alpha_real`/`alpha_imag` elements, isotropic parts and raw residuals in the existing atomic unit convention. [Collinear double-hybrid quasi-energy response](rust-api.md#frequency-dependent-double-hybrid-electric-response). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `molecular-medium-optics` | `molecular_medium_optics(numbers, positions, **controls)` | `calc.get_molecular_medium_optics(atoms, **controls)` | Normal complex indices, absorption coefficients and oblique s/p Jones and power data. ASE converts number density, photon energy, linewidth, thickness and inverse-length absorption. [Model and scope](response.md#r080-molecular-medium-optics). With `medium_magnetic_field`, the linear orbital magnetic susceptibility correction supplies field-odd Kerr rotation and ellipticity in radians. |
| `field-taylor` | `electric_field_taylor(numbers, positions, direction, *, order=3, reference_field=None, ...)` | `calc.get_electric_field_taylor(atoms, *, direction=(0,0,1), order=3, reference_field=None, ...)` | `dipole_coefficients`, `direction`, `reference_field`, `projector_defects`, `stationarity_defects`, `particle_number_derivatives`; HF/PBE0 water alpha/beta/gamma verified at zero and nonzero reference field by twelve independent four-step fits; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `field-mixed` | `mixed_electric_field_response(numbers, positions, directions, *, reference_field=None, ...)` | `calc.get_mixed_electric_field_response(atoms, *, directions, reference_field=None, ...)` | `dipole_derivative`, `directions`, `accumulation_roundoff`, `directional_evaluations`; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below |
| `beta` | `beta(numbers, positions, *, assembly="projector", reference_field=None, ...)` | `calc.get_beta(atoms, *, assembly="projector", reference_field=None, ...)` | Flat `tensor[27]`, `shape`, `order=2`, `beta_vector/norm`, `frobenius_norm`, raw intrinsic/energy permutation defects and counts, `accumulation_roundoff`, ordered/directional evaluation counts and physical coefficient defects; `assembly="variational"` selects three-solve 2n+1; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The third-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^3*Bohr^3/Hartree^2, with raw response and pair residuals. |
| `gamma` | `gamma(numbers, positions, *, reference_field=None, ...)` | `calc.get_gamma(atoms, *, reference_field=None, ...)` | Flat `tensor[81]`, `shape`, `order=3`, `gamma_isotropic`, `frobenius_norm`, raw intrinsic/energy permutation defects and counts, `accumulation_roundoff`, ordered/directional evaluation counts and physical coefficient defects; shared driver accepts prescribed nonthermal real collinear populations; see the nonlinear electric response contract below Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). Static `reference=gks` with equal `pt2_same_spin` and `pt2_opposite_spin` uses stationary complex spinors. The fourth-order energy paths return parent, target, target correction, and unscaled and weighted generalized PT2 tensors in e^4*Bohr^4/Hartree^3, with raw response and pair residuals. |
| `dynamic-beta` | `dynamic_beta(numbers, positions, *, input_frequencies=[[w1,eta1],[w2,eta2]], imaginary=False, ...)` | `calc.get_dynamic_beta(atoms, *, input_frequencies, imaginary=False, ...)` | Flat `tensor_real/imag[27]`, `shape`, incoming/outgoing complex frequencies, `equation_defect`, `projector_defect`, `particle_number_defect`, intrinsic comparisons, `subset_solves`, `nonlinear_kernel_evaluations` |
| `dynamic-gamma` | `dynamic_gamma(numbers, positions, *, input_frequencies=[[w1,eta1],[w2,eta2],[w3,eta3]], imaginary=False, ...)` | `calc.get_dynamic_gamma(atoms, *, input_frequencies, imaginary=False, ...)` | The same keys, flat complex parts with 81 entries, shape `[3,3,3,3]` |
| `electric-process` | `dynamic_electric_process(numbers, positions, *, process="shg", frequencies=[0.05], damping=0.005, ...)`; beta/gamma wrappers accept the same preset keywords | `calc.get_dynamic_electric_process(atoms, *, process, frequencies, damping=0.0, ...)`; `get_dynamic_beta/gamma` also accept them | Common tensor report plus canonical `process`; frequency magnitudes and damping use Hartree |
| `stability` | `native.stability(numbers, positions, n_roots=3, channels=None, tolerance=1e-7, max_iterations=200, resolution=None, ...)` | `DftRs.get_stability(atoms, **controls)` | Atomic-unit common report; `stable`, `unrestricted`, `channels`/`blocks`, `verdict`, `lowest`, `eigenvalues`, `eigenvector`, `dominant`, convergence/provenance |
| `stab-following` | `native.stabilize(numbers, positions, **controls)`; `native.stability(..., follow_unstable=True, max_descents=3, step_angle=0.3, min_energy_drop=1e-8)`; `native.calculate(..., stabilize=True, unrestricted=True)` | `DftRs.get_stab_following(atoms, **controls)`; ordinary single points with `DftRs(stabilize=True, unrestricted=True)` | Final atomic-unit energy, density, MO coefficients/energies, initial/final diagnostics, descent history, `stabilized`; ordinary ASE energy/forces/dipole use ASE units and preserve the atomic-unit diagnostic report in `results["stability"]` |
| `orbital-energy-gradients` | `orbital_energy_gradients(numbers, positions, *, coordinates=None, degeneracy_tolerance=1e-8, unrestricted=False, ...)` | `calc.get_orbital_energy_gradients(atoms, *, coordinates=None, ...)` | Atomic-unit `gradients[spin][orbital][coordinate]` or `None`, energies/occupations, coordinates, grouped matrices/energy spread/commutator and all residuals |
| `frontier-gap-gradient` | `frontier_gap_gradient(numbers, positions, *, coordinates=None, degeneracy_tolerance=1e-8, unrestricted=False, ...)` | `calc.get_frontier_gap_gradient(atoms, *, coordinates=None, ...)` | `gaps[spin]` or `None`, optional gap gradients, plus the common orbital report; grouped frontiers have `gradient=None` |
| `nuclear-taylor` | `nuclear_taylor(numbers, positions, direction, *, order=4, ...)` | `calc.get_nuclear_taylor(atoms, *, direction, order=4, ...)` | `energy_coefficients`, `energy_components`, `density_coefficients`, `overlap_coefficients`, `orthonormal_fock_coefficients`, `projector_defects`, `stationarity_defects`, `metric_defects`, `particle_number_coefficients`, reference energy/Fock errors |
| `nuclear-mixed` | `mixed_nuclear_response(numbers, positions, directions, **options)` | `calc.get_mixed_nuclear_response(atoms, *, directions, ...)` | `energy_derivative`, `directions`, `accumulation_roundoff`, `directional_evaluations` |
| `molecular-taylor` | `molecular_taylor(numbers, positions, nuclear_direction, electric_direction, *, order=4, ...)` | `calc.get_molecular_taylor(atoms, *, nuclear_direction, electric_direction, order=4, ...)` | `energy_coefficients`, `energy_components`, `dipole_coefficients`, `field_direction`, density/overlap jets and physical defects |
| `force-constants` | `analytic_force_constants(numbers, positions, *, order=3, patterns=None, coordinates=None, ...)` | `calc.get_analytic_force_constants(atoms, *, order=3, patterns=None, coordinates=None, ...)` | `tensor`, `patterns`, `order`, `independent_components`, `permutation_defect`, `permutation_comparisons`, `translation_residual` (or `None`), `translation_comparisons`, `accumulation_roundoff`, `directional_evaluations` ; integer correlated RKS/UKS complete Cartesian/block/pattern tensors include full orbital/PT2 amplitude response and separate target/SS/OS/dispersion channels; coordinate units Hartree/Bohr^order, dimensional-pattern unit Hartree; energy permutation reuse is explicit, with raw residuals and independent acceptance tracked in response.md  ROKS/ROHF and complex GKS/GHF complete exports are connected; full acceptance remains in progress, with raw failures documented in response.md. Geometry-only D1 is accepted through the shared driver using explicit atomic C₆, radii, and scale. [Model and checks](response.md#molecular-d1-energy-and-nuclear-tensors). Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `anharmonic` | `anharmonic(numbers, positions, *, method="retained-polyad", mode_ids=None, include_raman=False, ...)` | `calc.get_anharmonic(atoms, *, method="retained-polyad", mode_ids=None, include_raman=False, ...)` | `bands`, `selected_mode_ids`, `patterns`, `cubic`, `quartic`, `dipole_derivatives`, `polarizability_derivatives`, `zero_point_energy`, `harmonic_zero_point_energy`, `polyad_occupations/energies/vectors`, `effective_hamiltonian`, `rotation`, raw force/polarizability symmetry diagnostics, `basis_states`, `added_resonant_states`, `minimum_external_denominator`, `exterior_mixing_norm`, `hermiticity_defect`, `eigen_residual`, `maximum_force` |
| `molecular-mixed` | `mixed_molecular_response(numbers, positions, nuclear_directions, electric_directions, **options)` | `calc.get_mixed_molecular_response(atoms, *, nuclear_directions, electric_directions, ...)` | `energy_derivative`, `dipole_derivative`, nuclear/electric directions, energy/dipole accumulation roundoff and evaluation count |
| `static-spectroscopy` | `static_spectroscopic_derivative(numbers, positions, nuclear_direction, *, electric_order=2, ...)` | `calc.get_static_spectroscopic_derivative(atoms, *, nuclear_direction, electric_order=2, ...)` | `tensor`, `electric_order`, `nuclear_direction`, `permutation_defect`, `permutation_comparisons`, `accumulation_roundoff`, `directional_evaluations`; common-driver integer RKS/UKS complete post-SCF target and PT2 amplitude/orbital response; nuclear paths in Bohr, atomic electric units, all ordered tensor entries; complete C1 HF+PT2 RKS/UKS static APT/Raman/hyper-Raman subset validated by all 570 independent fits and all 12 primary native/ASE/CLI checks; raw normalized amplitude, reference, metric, particle-number and stationarity diagnostics and failed-constraint warnings are returned. Use the common driver for target/PT2 energy composition; the low-level Rust entry differentiates its supplied reference functional. [Constrained static tensors] Named C1 HF ROHF/GHF static subset: all434 independent fits, translation and public10 pass; other class/frequency combinations remain separate. See [measurements](#constrained-static-electric-tensors). |
| `periodic-response` | `periodic_density_response(numbers, positions, cell, *, q=(0,0,0), mode=0, ...)` | `calc.get_periodic_density_response(atoms, *, q=(0,0,0), mode=0, ...)` | `density`, `fourier_density_real`, `fourier_density_imag`, `reciprocal_waves`, `occupation_consistency_defect`; full complex AO entries and the same HF/full-range exchange physical solver as Rust/CLI, with the p292 restricted-reference scope |
| `field-response` | `periodic_field_response(numbers,positions,cell,*,q,scalar_potential=0j,vector_potential=(0j,0j,0j),...)` | `calc.get_periodic_field_response(atoms,*,q,scalar_potential=0j,vector_potential=(0j,0j,0j),...)` | Same compiled driver; complex scalar/vector observables, full spin/k AO density, explicit contact current and raw diagnostics |
| `dielectric` | `periodic_dielectric(numbers, positions, cell, *, q=(0,0,0), modes=(0,), ...)` | `calc.get_periodic_dielectric(atoms, *, q=(0,0,0), modes=(0,), ...)` | `screened_chi_real`, `screened_chi_imag`, `inverse_epsilon_real`, `inverse_epsilon_imag`, `reciprocal_waves` |
| `thermal-response` | `periodic_thermal_response(numbers, positions, cell, *, delta_temperature=1.0, delta_numbers=(0,0), delta_mu=(0,0), fixed_number=True, ...)` | `calc.get_periodic_thermal_response(atoms, *, delta_temperature=1.0, delta_numbers=(0,0), delta_mu=(0,0), fixed_number=True, ...)` | `density`, `fourier_density_real/imag`, `delta_chemical_potentials`, `delta_numbers`, `delta_entropy`, `delta_internal_energy`, `delta_free_energy`, `occupation_equation_defect`, `number_constraint_defects` |
| `periodic-electrostatics` | `periodic_electrostatics(positions, charges, cell, *, pbc=(True,True,True), widths=None, compensation="neutral", countercharge_center=None, countercharge_width=None, ...)` | `calc.get_periodic_electrostatics(atoms, *, charges, widths=None, compensation="neutral", ...)` | Common atomic-unit report: energy/components, forces, potentials, countercharge force/potential, supplied/compensating charge, dimension/measure, reciprocal vectors/k points, Ewald parameters, convergence and quadrature defects |
| `periodic-basis` | `periodic_basis(numbers, positions, cell, *, basis="sto-3g", pbc=(True,True,True), k_mesh=(1,1,1), basis_cutoff=12, grid_level=None, grid_cutoff=12, grid_derivatives=False)` | `calc.get_periodic_basis(atoms, *, basis=None, **controls)` | Atomic-unit `overlap_real/imag`, `kinetic_real/imag`, all atomic derivative arrays, optional `grid` with points/weights/owners/moving derivatives and raw defects |
| `periodic-gaussian-integral` | `periodic_gaussian_integral(positions, exponents, angular, cell, *, pbc=(True,True,True), source_charge=None, source_width=None, alpha=None, real_cutoff=None, reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11)` | `calc.get_periodic_gaussian_integral(atoms, *, exponents, angular, **controls)` | Atomic-unit value/split components, `[centre,3]` analytic gradients, measured quadrature/translation defects, model/units/warnings and all numerical controls; source knobs valid only for the three-centre nuclear case |
| `periodic-coulomb` | `periodic_coulomb_operators(numbers,positions,cell,*,basis="sto-3g",charge=0,pbc=(True,True,True),k_mesh=(1,1,1),basis_cutoff=12,density_real=None,density_imag=None,...)` | `calc.get_periodic_coulomb_operators(atoms,*,basis=None,charge=None,**controls)` | Atomic-unit complex matrices `[k,AO,AO]`, split components `[k,component,AO,AO]`, derivatives `[k,centre,axis,AO,AO]`, energy decomposition/gradients, compensation/electron-count metadata and raw defects; absent density leaves Hartree/electronic fields absent |
| `periodic-xc` | `periodic_xc_operators(numbers,positions,cell,*,density_real,density_imag=None,basis="sto-3g",xc="pbe",charge=0,pbc=(True,True,True),k_mesh=(1,1,1),...)` | `calc.get_periodic_xc_operators(atoms,*,basis=None,xc=None,charge=None,**controls)` | Input/output `[spin=2,k,AO,AO]` in native spherical order; semilocal energies/potentials in Hartree, separate raw number/operator/component diagnostics and explicit omitted exchange/nonlocal metadata |
| `magnetic-response` | `magnetic_response(numbers,positions,*,xc="hf",charge=0,multiplicity=1,gauge="giao",origin=None,unrestricted=False,...)` | `calc.get_magnetic_response(atoms,*,gauge="giao",origin=None,response_tol=1e-10,response_max_iter=500,response_restart=40)` | Common compiled `_core.nmr(...,density_response=True)` driver; full [spin,B,AO,AO] derivatives, reference matrices and raw response metadata in native units |
| `nmr` | `nmr(numbers,positions,*,gauge="giao",origin=None,unrestricted=False,response_tol=1e-10,response_max_iter=500,response_restart=40,...)` | `calc.get_nmr(atoms,...)` | Complete orbital shielding dictionary in ppm plus common reference/residual/axis/model metadata; native origin is Bohr |
| `periodic-field` | `periodic_field_operators(numbers,positions,cell,*,pbc=(True,True,True),k_mesh=(1,1,1),wave=(0,0,0),k_offset=(0,0,0),frequency=(0,0),scalar_potential=(0,0),vector_potential=None,...)` | `calc.get_periodic_field_operators(atoms,*,basis=None,charge=None,**controls)` | Native/ASE share the Rust driver; complex potentials [real,imag], wave/offset inverse Bohr, frequency Hartree, analytic vertices plus actual shifted spaces/model/raw defect |
| `dimensional-scf` | Self-consistent dimensional collinear reference and signed compensated charge | `dft_rs.dimensional_scf(numbers,positions,cell,**controls)` / `_core.dimensional_scf`; ASE `DftRs.get_dimensional_scf(atoms,**controls)` | Shared energy/component, complex matrix and diagnostic dictionary; native geometry Bohr, ASE converts Atoms geometry from Angstrom; getter energies remain Hartree per cell |
| `dimensional-field` | `dimensional_field_response(numbers,positions,cell,*,transfer_fractional=(0,0,0),scalar_potential=0j,vector_potential=(0j,0j,0j),omega=0.,damping=0.,imaginary=False,fixed_number=True,...)`; equivalently `dimensional_scf(..., field_transfer=...)["field_response"]` | `calc.get_dimensional_field_response(atoms,*,transfer_fractional,scalar_potential=0j,vector_potential=(0j,0j,0j),...)` | `transfer_fractional`, `reduced_transfer_fractional`, `wave`, `reduced_transfer`, `frequency`, `k_points`, `target_k_points`, `*_real/*_imag` for scalar/vector potential, E, B, number/charge density, paramagnetic/diamagnetic/total currents, plus `continuity_defect`, `hartree_quadrature_defect`, `occupation_consistency_defect`, `reference_operator_defect` |
| `energy` | `energy(numbers, positions, **controls)`; generic `run_feature("energy", numbers, positions, **controls)` | `calc.get_energy_report(atoms, **controls)` (calculator parameters supply reference controls) | `feature`, `version`, `model`, `functional`, `basis`, `converged`, `scf_iterations`, `unrestricted`, `n_alpha`, `n_beta`, `s_squared`, `energy`, `energy_components`, `warnings`, `units`, optional `stability_following`; `xc="skala", skala_model=path` selects a Skala model file. Double hybrids use this same entry: `xc="b2plyp"`, `xc="dsd-pbep86"`, `xc="dsd-blyp"`, `xc="pwpb95"` or `xc="wb97m2"`; the DSD names select the documented 2013 pairwise D3(BJ) recipe. SS/OS PT2 and dispersion are reported separately. Arbitrary parents accept `pt2_same_spin` and `pt2_opposite_spin`. Reports give parent, term energies and raw amplitude/Lagrangian residuals; [model and measured acceptance](response.md#double-hybrid-components-and-pt2-lagrangian).  omegaB97M(2) uses a distinct target on omegaB97M-V orbitals, scaled VV10 with b=10 and separate orbital/target energies. PWPB95 includes reoptimized PW/B95 and SOS PT2; all terms and raw residuals use the shared report. Custom parents retain their complete XC energy; `nonsemilocal_exchange_correlation_energy` (Hartree) separates a model-provider contribution from the semilocal component sum. VV10 remains in its separate nonlocal-correlation field. Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). GKS common-weight generalized PT2 energy reports the full-spin correlation, optional target correction and amplitude residual in Hartree.  [ROKS doubles-only PT2 energy and residuals](#restricted-open-shell-double-hybrid-energy). [Molecular D1, TS, MBD, VV09 and fixed-density vdW-DF controls, units and acceptance](response.md#r089-molecular-post-scf-dispersion-and-nonlocal-kernels). [Molecular total-spin values and run logs](response.md#molecular-total-spin-streaming). |
| `mermin-free-energy` | `mermin_free_energy(numbers, positions, *, electronic_temperature=0.0, gradient=False, gibbs=False, **controls)` | `calc.get_mermin_free_energy(atoms, **controls)` | Mermin free energy, internal energy, entropy, occupations and chemical potentials. Thermal ROKS/ROHF fixes both spin counts in shared spatial orbitals. GKS/GHF fixes total electron count. Native Hartree, ASE eV unless atomic units are selected. See the reference-specific gradient and equilibrium Gibbs RRHO scope. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `gibbs-free-energy` | `gibbs_free_energy(numbers, positions, **controls)` with `temperature`, `pressure`, `symmetry_number`, `electronic_degeneracy` and reference Hessian controls | `calc.get_gibbs_free_energy(atoms, **controls)` | Gibbs energy in native Hartree per molecule or ASE eV (`units="atomic"` preserves Hartree); positive internal modes, equilibrium temperature and a converged free-molecule Eckart model required; same provider/acceptance limits in response.md |
| `free-energy` | Compatibility wrapper; prefer `mermin_free_energy`: `free_energy(numbers, positions, *, electronic_temperature=0.0, gradient=False, **controls)` | `calc.get_free_energy_report(atoms, **controls)` | Energy keys plus `free_energy`, `internal_energy`, `entropy`, `entropy_term`, `electronic_temperature_kelvin`, `kt`, `chemical_potentials`, `occupations`, `ensemble`, optional `gradient`, `provenance` |
| `gradient` | `gradient(numbers, positions, **controls)` | `calc.get_gradient_report(atoms, **controls)` | Energy keys plus `gradient` (Hartree/Bohr), `translation_residual`, `provenance` Integer state selection: `occupation_mode=mom/imom`, complete `occupied_alpha/occupied_beta`, optional `occupation_tracking`; [score, units and pending combinations](rust-api.md#occupied-projector-delta-scf-controls). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). The ordinary molecular report also returns `forces` as the negative Cartesian gradient in Hartree/Bohr. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). [Generalized double-hybrid gradients](api-guide.md#read-a-generalized-double-hybrid-gradient). |
| `streamed-gradient` | `streamed_gradient(numbers, positions, **controls)` | `calc.get_streamed_gradient(atoms, **controls)` | Energy keys plus `gradient`, `translation_residual`, `integral_storage`, `avoided_tensor_megabytes` [Measured restricted HF/PBE STO-3G all-component and public-route scope](validation-matrix.md#t139-integral-direct-molecular-gradient). |
| `charges` | `charges(numbers, positions, **controls)` | `calc.get_charges_report(atoms, **controls)` | Energy keys plus `mulliken`, `lowdin`, `mulliken_spin`, `charge_sum_residual`  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). |
| `natural-bond` | `natural_bond(numbers, positions, **controls)` | `calc.get_natural_bond(atoms, **controls)` | `natural_charges`, `natural_populations`, `natural_spin_populations`, `natural_electron_configurations`, `wiberg_bond_indices`, `nao_*`, `channels` (`nbos`, `perturbation` stabilizations in Hartree, eV on ASE, `nlmos`) `nbo_spin_density=True` returns signed spin eigenweights and can evaluate `nbo_spin_density_points`. `nbo_natural_orbitals=True` returns natural spatial occupations and effectively unpaired weights. [Reference limits and units](response.md#reference-selection-and-natural-density-orbitals). `nbo_corresponding_orbitals=True` returns occupied alpha/beta pairs and complements. |
| `igmh` | `igmh(numbers, positions, **controls)` | `calc.get_igmh(atoms, **controls)` | `igmh_delta_g`, `igmh_delta_g_inter`, `igmh_delta_g_intra`, `igmh_signed_density`, density derivatives and partition residuals. Sampling coordinates are Bohr in native Python and Å by default in ASE, with the corresponding inverse powers for the fields. [Model and measurements](response.md#r090-molecular-igmh). |
| `periodic-natural-bond` | `periodic_natural_bond(numbers, positions, cell, **controls)` | `calc.get_periodic_natural_bond(atoms, **controls)` | `natural_charges`, `natural_electron_configurations`, `wiberg_bonds`, `channels` (`nbos`, `perturbation` in Hartree, eV on ASE), `translation_defect`, `unfolding_imaginary` |
| `natural-shielding` | `natural_shielding(numbers, positions, **controls)` | `calc.get_natural_shielding(atoms, **controls)` | `shieldings` (isotropic, tensor, `contributions` per NLMO, ppm), `decomposition_residual` |
| `natural-coupling` | `natural_coupling(numbers, positions, **controls)` | `calc.get_natural_coupling(atoms, **controls)` | `couplings` (reduced K terms, `coupling_hz`, `contributions` per NLMO), `decomposition_residual` |
| `natural-energy-decomposition` | `natural_energy_decomposition(numbers, positions, *, fragments, **controls)` | `calc.get_natural_energy_decomposition(atoms, **controls)` | `interaction_energy`, `charge_transfer`, `electrostatic`, `exchange_correlation`, `deformation`, `fragments` (Hartree, eV on ASE) |
| `sapt-analysis` | `sapt_analysis(numbers, positions, *, fragments, **controls)` | `calc.get_sapt_analysis(atoms, **controls)` | `sapt_variant="response"` (default) or `"sapt0"`, `"sapt0-s2"`, `"ssapt0"`, `"sapt-30"` (field-free HF), `"sapt2+3"` (closed-shell all-electron HF), or `"sapt-ks-fdds"` (KS); `electrostatic`, `exchange`, `induction`, `exchange_induction`, `exchange_induction_directions`, `dispersion`, `exchange_dispersion`, `interaction_energy`, `interaction_energy_rpa`, `third_order` (`sapt-30` or `sapt2+3`), `sapt2_plus3` ([correlated HF result](response.md#sapt23-with-full-intermolecular-exchange)), `regularized_induction` (`sapt_regularization_eta` on `sapt0` or `sapt-ks-fdds` in Bohr⁻² or Å⁻² in ASE), `correlation_terms` (`sapt_correlation_terms=True`), `relaxed_induction30` (`sapt_relaxed_induction30=True`), `delta_hf`, `exchange_scale` (dimensionless), `rpa`, `charge_transfer`, `partition` (Hartree, eV by default on ASE). See [correlated terms and cubic induction](response.md#correlated-sapt-terms-and-relaxed-cubic-induction) and [SAPT(0) models and acceptance](response.md#sapt0-with-the-full-antisymmetrizer). |
| `dipole` | `dipole(numbers, positions, *, origin="center_of_mass", **controls)` | `calc.get_dipole_report(atoms, **controls)` | Energy keys plus `dipole`, `dipole_norm`, `dipole_debye`, `dipole_norm_debye`, `origin`, `origin_convention`  Double hybrids and custom PT2 parents use the same API; `parent_dipole`, `pt2_dipole_correction`, `target_dipole_correction` are e Bohr; `response_residuals` are relative, `amplitude_response_residuals` e Bohr and `response_number_residuals` electrons per atomic field. Includes noncanonical amplitude/orbital response and distinct-target exchange/VV10; integer zero-temperature reference. [Measured fits and model limits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants). Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls). [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `orbitals` | `orbitals(numbers, positions, *, coefficients=False, density=False, **controls)` | `calc.get_orbitals_report(atoms, **controls)` | Energy keys plus `n_ao`, `channels[]` with `spin`, `energies`, `occupations`, `homo_index`, `homo`, `lumo`, `gap`, optional `coefficients`, `density`; `overlap`, `ao_order` Fixed nonthermal populations also use these common entries: `occupation_mode=fixed`, `occupations_alpha/occupations_beta`; the raw density-map residual is dimensionless and the unequal-population spectral gap is Hartree. [Constraint, units and remaining combinations](rust-api.md#fixed-orbital-population-controls).  Supports `reference=auto/rks/uks/roks` (HF aliases accepted); ROKS spin semicanonical orbital export and constraints are described in [reference-specific scope](response.md#reference-constraints-for-molecular-populations-and-orbitals).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF and GKS/GHF connections have the [reference-specific scope and diagnostics](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `molden` | `write_molden(numbers, positions, *, path, **controls)` | `calc.write_molden(atoms, path=..., **controls)` | Energy keys plus `path`, `bytes`, `n_mo`, `occupation_density_deviation`, `ao_order` |
| `orbitals-save` | `save_orbitals(numbers, positions, *, path, **controls)` | `calc.write_orbitals(atoms, path=..., **controls)` | Orbital keys with `coefficients`, `density`, `overlap`, `kt`, `chemical_potentials`, `occupation_density_deviation`, `path` |
| `molecular-multipole` | `molecular_multipole(numbers, positions, *, max_rank=4, origin="center_of_mass", **controls)` | `calc.get_molecular_multipole(atoms, **controls)` | Energy keys plus `origin`, `origin_convention`, `conventions`, `ranks[]` with `rank`, `monomials`, `cartesian`, `traceless`, `spherical` [Reference, origin and weighted XC controls](rust-api.md#static-molecular-electric-moments). |
| `quadrupole-polarizability` | `quadrupole_polarizability(numbers, positions, **controls)` | `calc.get_quadrupole_polarizability(atoms, **controls)` | Five-mode Buckingham response in e²·Bohr⁴/Hartree in native units, with origin, frequency and residuals. RKS/UKS double hybrids include parent, target and PT2 spin channels at static and supported nonzero frequencies. Static GKS has a common PT2 weight. [Conventions](rust-api.md#electric-quadrupole-polarizability). `relativistic_method="dirac"` also returns both mixed dipole/field-gradient blocks and residuals for all eight sources. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-polarizability` | `multipole_polarizability(numbers, positions, multipole_rank=3, **controls)` | `calc.get_multipole_polarizability(atoms, multipole_rank=3, **controls)` | Static and retarded Racah response for rank 1 to 5 in e²·Bohr^(2 rank)/Hartree in native units. Mode order, origin, signed frequency and raw residuals accompany the tensor. Collinear double hybrids report parent, target and PT2 terms. ASE converts units once. The default center of mass follows molecular translation. [Conventions](rust-api.md#electric-multipole-polarizability). `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `conceptual-dft` | `conceptual_dft(numbers, positions, *, route=None, population="mulliken", electronic_temperature=0.0, **controls)` | `calc.get_conceptual_dft(atoms, **controls)` | Energy keys plus `route`, `population_scheme`, `conventions`, `global`, and the route-specific descriptor keys of Table 1 |
| `spin-conceptual-dft` | `spin_conceptual_dft(numbers, positions, *, route=None, **controls)` | `calc.get_spin_conceptual_dft(atoms, **controls)` | Energy keys plus the spin-channel keys of Table 1 |
| `hessian` | `hessian_report(numbers, positions, **controls)` | `calc.get_hessian_report(atoms, **controls)` | Energy keys plus `hessian`, `dipole_derivatives`, `raw_asymmetry`, `translation_residual`, `provenance`, `symmetrized`, `axis_convention` Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Prescribed nonthermal populations use the full spectral moving-metric response, with `ensemble_response_residuals[3N]` (dimensionless), `ensemble_number_defects[3N][2]` (electrons/Bohr), the Hessian and APT. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc).  [Occupation-order diagnostics](rust-api.md#scf-occupation-order-diagnostics): per-spin occupied_virtual_gaps (Hartree) and negative-gap warning. Positive electronic temperature selects the stationary molecular Mermin derivative for supported ROKS/ROHF and GKS/GHF providers. [Finite-temperature guide](api-guide.md#set-an-electronic-temperature). |
| `frequencies` | `frequencies_report(numbers, positions, **controls)` with the same reference Hessian controls, `vibrational_projection`, `electronic_degeneracy` | `calc.get_frequencies_report(atoms, **controls)` | `frequencies_cm1`, `normal_modes_cartesian` (1/sqrt(amu)), IR (km/mol or null without APT), raw Hessian/reference residuals and optional RRHO; native Hartree/Bohr, ASE converted labels. Source/acceptance limits in response.md |
| `ir-raman` | `vibrational_spectra(numbers, positions, **controls)` | `calc.get_vibrational_spectra(atoms, **controls)` | Frequency keys plus `raman_activities`, `depolarization_ratios`, `polarizability_derivatives`, `raman_permutation_defect`, `response_converged`, and with `raman_frequency`: `dynamic_raman_activities`, `dynamic_depolarization_ratios`, `dynamic_polarizability_derivatives` [Measured restricted HF STO-3G dynamic Raman component and public-route scope](validation-matrix.md#t017-dynamic-raman-mode-derivatives). |
| `polarizability` | `polarizability_report(numbers, positions, **controls)` | `calc.get_polarizability_report(atoms, **controls)` | Energy keys plus `polarizability`, `isotropic`, `anisotropy`, `raw_asymmetry`, `polarizability_angstrom3`, `isotropic_angstrom3`; named double hybrids and custom SS/OS parents use the same driver, with second-order parent orbitals, noncanonical PT2 amplitudes and distinct-target exchange/VV10 kernels (physical acceptance in progress). `parent_polarizability`, `pt2_polarizability_correction`, `target_polarizability_correction` and unscaled aa/ab/bb PT2 tensors use e^2 Bohr^2/Hartree. Six independent static components are expanded; `raw_asymmetry` is null. Pair residuals have shape [6,3] for Taylor orders 0,1,2 with units [Hartree,e Bohr,e^2 Bohr^2/Hartree]; fresh orbital residuals and projector/number/frame defects are separate. Dynamic and higher correlated properties remain pending Prescribed populations and positive electronic temperature use `response::ensemble::EnsembleSolver::polarizability` through this same entry. Reports add the population model, fresh `response_residuals[3]`, iteration counts and `number_defects[3][2]` (e Bohr/Hartree); all tensor columns are solved independently. [Population-law contract](rust-api.md#fixed-population-and-thermal-static-response).  RKS/UKS selectors and HF aliases use the existing driver; ROKS/ROHF uses the existing constrained analytic route; [GKS/GHF nuclear paths include Coulomb/exchange and globally projected semilocal XC; full Cartesian double-hybrid tensors include PT2 amplitude/orbital response, with bounded acceptance scope](response.md#molecular-complex-spinor-reference-provider). Static GKS with a common full-spin PT2 weight returns the total, parent, unweighted pair, weighted pair and optional distinct semilocal or hybrid target tensors in e^2 Bohr^2/Hartree. The six independent components use the stationary spinor energy Hessian, with raw orbital, projector, particle-number and pair diagnostics. [Conventions and measured scope](response.md#r060-molecular-multipoles-and-derivatives). The integer collinear CPKS route reports `response_absolute_residuals` in e Bohr, one maximum fresh occupied-virtual equation defect per Cartesian field. `relativistic_method="dirac"` selects the molecular fixed-sector four-component DKS route. [Model and coverage](response.md#molecular-diraccoulomb-kohnsham). |
| `multipole-gradient` | `multipole_gradient(numbers, positions, *, max_rank=2, origin="center_of_mass", **controls)` | `calc.get_multipole_gradient(atoms, **controls)` | Energy keys plus `origin`, `origin_convention`, `ranks[]` with `rank`, `monomials`, `cartesian`, `gradient` |
| `response-save` | `response_derivatives(numbers, positions, *, path=None, nuclear=True, **controls)` | `calc.get_response_derivatives(atoms, **controls)` | Energy keys plus `field`, `polarizability`, `nuclear_density_derivatives`, `conventions`, optional `path`, and with `frequencies`: `dynamic[]` (`frequency`, `field[axis].x_plus_y_real`, `x_minus_y_real`, `polarizability_real`, ...) The p126 strict-unit, zero-frequency static-limit and finite-frequency normal-dispersion checks pass; additional nuclear and frequency combinations remain separate. |
| `third-derivative-2n1` | `third_derivative_2n1(numbers, positions, **controls)` | `calc.get_third_derivative_2n1(atoms, **controls)` | `cubic_force_constants`, `path_evaluations`, `translation_residual`, `accumulation_roundoff` Arrays use Hartree/Bohr^3. The p104 baseline validates HF/PBE/PBE0 cubic tensors against complete mixed responses; its independent four-step SCF-energy derivative fits also pass (p=1.995439646..2.002158542, minimum R2=0.999998593470, maximum response residual 4.314e-16, maximum Richardson defect 1.965e-7 Hartree/Bohr^3). See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) |
| `cubic-fc-along` | `cubic_fc_along(numbers, positions, *, probe, **controls)` | `calc.get_cubic_fc_along(atoms, **controls)` | `contracted_force_constants`, `probe`, `path_evaluations`, `accumulation_roundoff` Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Arrays use Hartree/Bohr^3. The p104 baseline validates the PBE zero-field contraction against the complete tensor; this comparison alone does not establish an independent finite-difference derivative fit. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `quartic-fc-along` | `quartic_fc_along(numbers, positions, *, probe, **controls)` | `calc.get_quartic_fc_along(atoms, **controls)` | `contracted_force_constants`, `probe`, `path_evaluations`, `accumulation_roundoff` Optional `bfield` and `field` select the shared complex London stationary nuclear path (integer occupations); `bfield_series_order` retains at least nuclear order+2. Reports include fresh `response_residuals`, `projector_defects`, `stationarity_defects`, `reference_defect`, `series_order`, `series_tail` and `accumulation_roundoff`. Intrinsic permutation reuse is declared; independent ordered asymmetry and dipole derivatives are not supplied on this magnetic path. Complex PT2 and extra Hamiltonian providers retain separate restrictions. See the externally biased nuclear scope below. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. Arrays use Hartree/Bohr^4. The p104 baseline validates the PBE zero-field contraction against complete mixed responses; independent derivative and further-combination acceptance remain separate. See [measurements](response.md#reduced-order-baseline-repeat-two-measured-route-comparison-tests) Correlated integer RKS/UKS energy paths, including static reference_field bias, are connected; parent/target/SS/OS tensors and raw residuals are returned. Measured subsets and remaining combinations are in response.md. |
| `excitations` | `excitations(numbers, positions, *, nstates=5, td_method="tda", td_spin="auto", amplitudes=False, **controls)` | `calc.get_excitations(atoms, **controls)` | Energy keys plus `td_method`, `td_spin`, `states[]`, `oscillator_strength_sum` Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). `td_nto=True` adds TDA `natural_transition_orbitals` to each state. [Conventions](response.md#natural-transition-orbitals). `td_difference_density=True` adds unrelaxed RKS/UKS/GKS difference orbitals. |
| `ecd-spectrum` | `ecd_spectrum(numbers, positions, *, broadening_ev, nstates=10, points=1000, energy_window_ev=None, **controls)` | `calc.get_ecd_spectrum(atoms, **controls)` | `states[]`, `energies_ev`, `wavenumbers_cm1`, `molar_absorptivity`, `ecd_length`, `ecd_velocity`, sum-rule integrals  Correlated PT2 pole and electric/magnetic transition-residue corrections use the same entry and spin controls; bounded full-component and public acceptance passes (see the detailed model scope). |
| `td-state-gradient` | `td_state_gradient(numbers, positions, *, root=1, **controls)`; optional `td_gradient_route="path"` | `calc.get_td_state_gradient(atoms, **controls)` | Energies (Hartree), `gradient` (Hartree/Bohr), root and route metadata, `translation_residual`; native directional path subset measured, full public path-gradient acceptance pending Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `hyperfine` | `hyperfine(numbers, positions, **controls)` or `run_feature("hyperfine", ...)`; `multiplicity`, `unrestricted`, optional signed `nuclear_g` and state-selecting `field` | `calc.get_hyperfine(atoms, **controls)` | The same contact/dipolar/total/principal fields, axis conventions, units and raw residuals as Rust and CLI; density in electrons/Bohr^3, couplings in MHz (contact also gauss), axes dimensionless; no isotope table or degenerate-state averaging Explicit ROHF/ROKS retain common spatial orbitals. GHF/GKS require `epr_projection=single-kramers-pair` for the approximate odd-electron doublet model, including orbital-current A, spin expectation, spin-squared and metric diagnostics. These extensions are all-electron and common-origin; they do not spin-purify the determinant. Source-connected HF nuclear-A increment: `nuclear_order=0` (default) keeps static output; 1 adds gradients with null Hessian fields; 2 adds the full Hessian in `hyperfine_nuclear`. Optional zero-based `derivative_nuclei` selects target nuclei only; derivatives still span all 3N input Cartesian coordinates. Contact/dipolar/total `_per_g` fields are per signed nuclear-g convention; optional signed `nuclear_g` adds scaled fields. Native/Rust/CLI derivative units are MHz/Bohr and MHz/Bohr^2; ASE converts inverse length factors to its selected length units. Scope is all-electron integer UKS/UHF HF; PBE full6840 validation is running, and ROKS/GKS/ECP/PT2 nuclear-A are refused rather than replaced by a parent. Raw path coefficients retain derivative/k!, indexed 0..order; unavailable zero-order response diagnostics remain null. Original/inner linear residuals and reconstructed/source ratios are separate quantities. Exact adapter04 order mapping is measured; final consolidated artifact/public acceptance remains pending. |
| `nmr-shielding` | `nmr_shielding(numbers, positions, **controls)` | `calc.get_nmr_shielding(atoms, **controls)` | `shielding_tensors`, `diamagnetic_tensors`, `paramagnetic_tensors`, `isotropic_shieldings`, `gauge`, `gauge_origin`, `response_residuals` Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. With `relativistic_method="dirac"`, finite nuclear charge and magnetization radii in Bohr and optional `nuclei` select static four-component London-RMB shielding. Returns `shielding_tensors` in ppm with explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `nmr-spectrum` | `nmr_spectrum(numbers, positions, **controls)` | `calc.get_nmr_spectrum(atoms, **controls)` | keys of `nmr-shielding`, `chemical_shifts`, `spectrum_elements`, `spectrum_axis`, `spectrum_intensity`, `spectrum_broadening` Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless.  Exact mode: `nmr_spin_model=exact`, `nmr_couplings=true`, `nuclear_g`, optional `nuclear_spins` (twice I), `nmr_spin_max_dimension` (1024). Shared isotropic spin Hamiltonian; ppm line positions, integrated line weights, raw Hz and dimensionless eigen residuals, area defects; bounded acceptance in response.md. With `relativistic_method="dirac"`, supply finite nuclear charge and magnetic RMS radii in Bohr for the London-RMB electronic response. The spectrum uses all nuclei's shielding tensors in ppm and optional total indirect J in Hz. Raw response, eigensystem and area residuals follow the common spin Hamiltonian. See [four-component spectrum](response.md#four-component-isotropic-nuclear-spin-spectrum). |
| `spin-spin-couplings` | `spin_spin_couplings(numbers, positions, **controls)` UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `calc.get_spin_spin_couplings(atoms, **controls)` | `coupling_pairs`, `reduced_couplings`, `diamagnetic_tensors`, `paramagnetic_tensors`, `fermi_contact_tensors`, `spin_dipole_tensors`, the isotropic lists, `coupling_constants_hz` `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Explicit VV10/learned target composition uses the same driver; new nonlocal coupled/public acceptance is in progress in p260_nonlocal_complex_response and response.md. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. Dirac references take `relativistic_method="dirac"` and Gaussian `nuclear_magnetic_rms_radii` in Bohr or ASE Angstrom. They return the indirect tensor and its explicit-sector and orbital-relaxation parts. [Four-component tests](response.md#four-component-indirect-nuclear-spin-coupling). |
| `fermi-contact-couplings` | `fermi_contact_couplings(numbers, positions, **controls)` UKS/GKS full complex Pauli response with optional fixed-spin-direction constraints is connected; all source/spin components and FC/SD-PSO mixed terms are retained. New molecular acceptance is tracked separately. | `calc.get_fermi_contact_couplings(atoms, **controls)` | `coupling_pairs`, `fermi_contact_tensors`, `fermi_contact_isotropic`, `fermi_contact_hz` `nuclei` must contain distinct atom indices. `coupling_pairs[p]=[M,N]` preserves their input order; tensor axes are `[pair][M-axis][N-axis]`, so reversing a pair requires a transpose and does not make its Cartesian tensor symmetric. The common report includes `axis_convention`; tensors retain atomic units and optional isotropic constants are Hz. Labelled-pair reciprocity and bounded public-route acceptance are measured in response.md; full functional/reference coverage remains separate. Correlated spin-free closed-shell FC/SD/PSO/DSO uses the full target/PT2 energy Hessian and relaxed DSO expectation through the same entry; independent acceptance passes all 925 component/channel fits in p259_correlated_spin_spin. The initial 17-case public/ordinary suite passes; the final rebuilt broader suite is tracked in response.md. Open-shell transverse pair response and nonlocal providers remain required. Weighted parent/target nonlocal tensors are returned separately (ppm for shielding, atomic units for coupling), already included in parent/target totals. J amplitude residuals are Hartree; frame/projector residuals are dimensionless. |
| `exafs` | `exafs(numbers, positions, **controls)` | `calc.get_exafs(atoms, **controls)` | `path_atoms`, `path_numbers`, `path_distances`, `path_sigma2`, `k`, `chi`, `chi_single`, `chi_double`, `double_scattering_paths`, `chi_weighted`, `k_weight`, `r`, `transform_magnitude` The p121 closed-chain, right-angle and single-scattering-limit checks pass; no energy derivative is involved. |
| `xanes` | `xanes(numbers, positions, **controls)` | `calc.get_xanes(atoms, **controls)` | `states`, `core_orbitals`, `core_orbital_energies`, with `core_hole = delta-scf`: `core_ionization_energy`, `core_hole_shift`, `energies_relaxed`, and with `xanes_broadening`: `spectrum_energies`, `spectrum_oscillator_density` Relaxed core holes reuse the common physical-metric projector-IMOM SCF builder and preserve both spin labels. The p122 one-electron core-hole limit and HF water O 1s relaxation checks pass; this does not validate state-specific excited-state relaxation. |
| `nexafs` | `nexafs(numbers, positions, **controls)` | `calc.get_nexafs(atoms, **controls)` | Lines and Cartesian oscillator-strength tensors, Gaussian or Lorentzian spectrum and core-hole tracking residuals. Native and ASE `units="atomic"` use Hartree, e Bohr and inverse Hartree. ASE defaults to eV, e Angstrom and inverse eV. [Model limits and water results](response.md#r085-near-edge-xray-absorption-fine-structure). |
| `constrained-dft` | `constrained_dft(numbers, positions, density_constraints=[{"atoms":[0,1],"electrons":8.96}], **controls)` | `calc.get_constrained_dft(atoms, **controls)` | `gradient`, `optimize_geometry` and `constraint_electric_response` select analytic forces, relaxation and fixed-population static polarizability. Native and ASE `units="atomic"` use Bohr and Hartree. ASE defaults to Angstrom and eV. Raw constraint, orbital and response residuals are returned. [Constraint schema and measured results](response.md#r083-constrained-density-functional-theory). |
| `dft-plus-u` | `dft_plus_u(numbers, positions, hubbard_sites=[{"orbitals":[5,6],"u_eff":0.14}], **controls)` | `calc.get_dft_plus_u(atoms, **controls)` | Fixed-geometry RKS, UKS, ROKS and GKS. Nested site inputs use Hartree. Optional `hubbard_pairs=[{"sites":[0,1],"v":0.05}]` adds the intersite correction. Pair `v` uses Hartree in native and eV in default ASE units. ASE converts the returned energies and raw orbital residual to eV by default. GKS reports real and imaginary spin-major occupation matrices in electrons. [Site schema and coverage](response.md#r087-molecular-dftu). |
| `pz-sic` | `pz_sic(numbers, positions, **controls)` | `calc.get_pz_sic(atoms, **controls)` | Real integer UKS Perdew-Zunger correction and raw orbital-gradient residual. Native uses Hartree. ASE defaults to eV. [Model and coverage](response.md#r088-molecular-pz-sic). |
| `flo-sic-fixed` | `flo_sic_fixed(numbers, positions, **controls)` | `calc.get_flo_sic_fixed(atoms, **controls)` | Frozen integer UKS determinant with fixed `fod_alpha` and `fod_beta` coordinates in Bohr for both interfaces. Native energy and parent orbital residual are Hartree. ASE defaults to physical units. [Model and coverage](response.md#fixed-descriptor-fermi-lowdin-sic). The common driver accepts `fod_gradient`, `fod_optimize` and `flo_self_consistent` for the real calculation. `fod_alpha_imaginary`/`fod_beta_imaginary` select complex descriptors with fixed parent orbitals. [Units, models and verification](response.md#fixed-descriptor-fermi-lowdin-sic). |
| `car-parrinello-md` | `car_parrinello_md(numbers, positions, **controls)` | `calc.get_car_parrinello_md(atoms, **controls)` | Trajectory, forces, both kinetic energies, extended energy drift and constraint defects. Time and velocities stay in atomic units, and masses in electron masses. Native and ASE `units="atomic"` use Bohr and Hartree for geometry and energy. ASE defaults to Angstrom and eV. [Trajectory model and water tests](response.md#r084-car-parrinello-molecular-dynamics). |
| `magnetizability` | `magnetizability(numbers, positions, **controls)` | `calc.get_magnetizability(atoms, **controls)` | `magnetizability`, `diamagnetic`, `paramagnetic`, `isotropic`, `isotropic_cgs_ppm`, `gauge_origin`, `raw_asymmetry`  Optional `response_order=1..4` and `magnetic_direction` (atomic magnetic field) add `magnetic_energy_coefficients` and actual factorial-restored `magnetic_energy_derivatives`, in Hartree along B(t)=t*direction, plus raw path/projector/stationarity residuals. Orbital zero-field collinear model; eight restricted HF/PBE/TPSS/PBE0 second/fourth magnetic energy fits pass; remaining tensor/reference/class acceptance is pending.  Optional `magnetic_directions` contains one through four Cartesian field directions. Returns the actual mixed energy derivative for B=sum_j t_j*d_j (dimensionless t_j), Hartree, plus raw response/projector/stationarity defects and polarization roundoff. Repeated and non-orthogonal directions are supported through the same London/common-origin field driver. Matching `electric_directions` optionally adds electric parts to every mixed parameter: E=sum t_j e_j, B=sum t_j b_j. `magnetic_mixed_derivative` then reports the joint energy derivative, Hartree per dimensionless parameters; the same analytic field/metric response is reused.  Static RKS/UKS double-hybrid orbital tensors use the shared complex orbital/PT2 field path for common-origin and London gauges. Parent, target, weighted PT2 and unscaled SS-alpha/OS/SS-beta tensors are reported separately with raw orbital, pair, metric and reference diagnostics. The full total energy is the same post-SCF model as the common energy driver. High-order correlated contractions remain explicitly unsupported. New tensor/public acceptance is pending. Explicit `reference=gks` or `reference=ghf` with `gauge=common-origin` selects the static all-electron spinor orbital tensor. The spinor report includes the fixed-density diamagnetic and relaxed paramagnetic terms and raw response, symmetry and imaginary defects. The finite-basis tensor depends on the origin. The rebuilt HSEsol ammonia-cation scope passes all eighteen independent four-step fits and public route parity. Maximum raw relative response residual 5.125e-11 and zero imaginary susceptibility defect. [Spinor model and acceptance](response.md#static-spinor-orbital-magnetizability-at-a-common-gauge-origin) With `relativistic_method="dirac"`, finite charge radii in Bohr and the London-RMB basis give static `magnetizability` in `e^2*bohr^2/m_e` and `magnetic_dipole` in atomic units. Reports give explicit-sector/orbital-relaxation parts and raw projector, reference and CPKS residuals. See [four-component scope](response.md#static-four-component-uniform-magnetic-response). The selected `dirac_interaction` can include magnetic Gaunt or frequency-independent Breit current terms. See [four-component current scope](response.md#uniform-magnetic-gaunt-and-breit-interactions). |
| `magnetic-quadrupole` | `magnetic_quadrupole(numbers, positions, **controls)` | `calc.get_magnetic_quadrupole(atoms, **controls)` | Orbital, spin and total magnetic quadrupoles, magnetic dipole and raw SCF residual. Magnetic quantities stay in atomic units. The default origin is the centre of mass. Explicit origin coordinates use Bohr in native and ASE `units="atomic"`, and Angstrom in default ASE mode. [Field convention and measured scope](response.md#r082-molecular-magnetic-quadrupole). |
| `franck-condon` | `franck_condon(numbers, positions, **controls)` | `calc.get_franck_condon(atoms, **controls)` | `huang_rhys`, `mode_frequencies`, `reorganization_energy`, `adiabatic_excitation_energy`, `line_energies`, `line_factors`, `line_quanta`, `franck_condon_sum` The p103 H2 vertical-model finite-temperature normalization and zero-temperature vertical-Hessian sum rules pass; broader vibronic spectra remain pending. |
| `excited-optimization` | `excited_optimization(numbers, positions, **controls)` | `calc.get_excited_optimization(atoms, **controls)` | `positions`, `excited_state_energy`, `excitation_energy`, `gradient`, `gradient_max`, `optimization_converged`, `iterations`, `trajectory_energies` [Exact model, full-component acceptance and residuals](validation-matrix.md#t120-isolated-singlet-tda-optimization). Explicit `td_gradient_route=path` connects analytic nuclear-path gradients; new-route acceptance is pending the coordinated rebuild. [Spin/model limits and implementation stages](excited-state-extension.md). |
| `optical-activity` | `optical_activity(numbers, positions, **controls)` | `calc.get_optical_activity(atoms, **controls)` | `frequencies`, `rotation_tensors`, `beta`, `specific_rotation`, `molar_mass`, `gauge_origin`; the model field distinguishes common-origin length gauge from modified velocity gauge with static subtraction. Nuclear optical-tensor derivative acceptance is tracked separately; periodic and correlated mixed dynamic response remain pending |
| `broken-symmetry-exchange` | `broken_symmetry_exchange(numbers, positions, **controls)` | `calc.get_broken_symmetry_exchange(atoms, **controls)` | `exchange_coupling`, `exchange_coupling_cm1`, `high_spin_energy`, `broken_symmetry_energy`, `high_spin_s_squared`, `broken_symmetry_s_squared`, site populations |
| `vcd` | `vcd(numbers, positions, **controls)` | `calc.get_vcd(atoms, **controls)` | `atomic_axial_tensors`, `frequencies_cm1`, `dipole_strengths`, `rotational_strengths`, `rotational_strengths_1e44_cgs`, `gauge_origin`; common-origin PT2 AAT and correlated Hessian/APT: 2,025/2,250 component fits and 15 public tests pass; learned/nonlocal and London pair paths remain separate |
| `spin-susceptibility` | `spin_susceptibility(numbers, positions, **controls)` | `calc.get_spin_susceptibility(atoms, **controls)` | `spin_susceptibility`, `spin_susceptibility_cgs_ppm`, `spin_stiffness`, `thermal_energy`, `response_residual`  `spin_g_factor` defaults to the free-electron value. Reports include `bare_spin_susceptibility`, `bare_spin_stiffness`, `spin_enhancement`, `spin_density_response[spin][AO][AO]`, raw response/number and reference commutator residuals. Magnetization convention M=-g(N_alpha-N_beta)/4; density response is per atomic magnetic field. Fixed total number, spin-only longitudinal response; GGA/meta-GGA/hybrid validation and remaining combinations are tracked in CSV. |
| `g-tensor` | `g_tensor(numbers, positions, **controls)` | `calc.get_g_tensor(atoms, **controls)` | `g_tensor`, `g_isotropic`, `g_shift_ppm`, `mass_correction_ppm`, `gauge_correction_ppm`, `spin_orbit_ppm`, `spin`, `soc_charges`, `gauge`, `gauge_origin` The p120 nondegenerate water-cation origin/translation and one-centre-limit checks pass; the independent mixed-energy derivative fit remains pending. Explicit ROHF/ROKS use the constrained common-orbital response. GHF/GKS require `epr_projection=single-kramers-pair` and return pair-axis `g_tensor`, invariant `g_gram`, spin and angular/operator diagnostics. These extensions support the all-electron common-origin one-electron model; London response and higher multiplet projection remain unavailable. Separate `nuclear_order=1/2` uses integer all-electron HF/UHF and the one-electron Pauli London model: `g_gradient[3N,3,3]` and `g_hessian[3N,3N,3,3]` are actual derivatives in `1/bohr` and `1/bohr^2`; the Hessian is null for order one. ASE converts these to inverse Angstrom powers. Mass/gauge/spin-orbit parts and real/mixed diagnostics remain separate. Nuclear DFT, RO/spinor, ECP/ensemble, nonzero electric field and stability-following requests remain explicit errors; explicit zero field is allowed. Static order zero keeps the preceding capabilities. No two-electron SOC, picture-change or spectral-accuracy claim follows. |
| `spin-spin-zfs` | `spin_spin_zfs(numbers, positions, *, nuclear_order=0, **controls)` | `calc.get_spin_spin_zfs(atoms, nuclear_order=0, **controls)` | Direct spin-spin D [3,3], optional nuclear gradient [3N,3,3] and Hessian [3N,3N,3,3]; native Hartree/Bohr, ASE eV/Angstrom. All-electron integer UHF/HF, multiplicity>=3; supply `xc="hf", unrestricted=True` with ASE. SOC and spin purification excluded; this is not total D. Unrequested blocks None, strict orders0..2. Raw residuals remain visible. The installed local wheel passes all 25 public cases on the same native artifact, including both ASE unit modes and both CLIs. Separate backend evidence retains 810 bounded HF fits; all 387 D/gradient/Hessian values agree with the retained canonical Rust report, an API-parity check only. |
| `roks` | `roks(numbers, positions, **controls)` | `calc.get_roks(atoms, **controls)` | `energy`, `energy_components`, `s_squared`, `reference_type`, `gradient`, `translation_residual` |
| `roks-hessian` | `roks_hessian(numbers, positions, **controls)` | `calc.get_roks_hessian(atoms, **controls)` | `hessian`, `gradient`, `frequencies_cm1`, `ir_intensities_km_mol`, `dipole_derivatives`, `n_imaginary`, `translation_residual`, `path_evaluations`, `response_residual` [Weighted XC controls and diagnostics](rust-api.md#weighted-molecular-xc). |
| `roks-polarizability` | `roks_polarizability(numbers, positions, **controls)` or `polarizability_report(..., reference="roks")` | `calc.get_roks_polarizability(atoms, **controls)` or `get_polarizability_report(..., reference="roks")` | Same constrained paired common-orbital zero-frequency kernel as the dynamic entry; complete Cartesian tensor, dipole and residuals; see R001 |
| `dc` | `divide_conquer(numbers, positions, **controls)` | `calc.get_divide_conquer(atoms, **controls)` | Energy keys plus `internal_energy`, `electronic_entropy`, `chemical_potentials`, `fragment_of`, `subsystem_sizes`, `density_residual`, `gradient`, `translation_residual` The molecular HF/PBE/PBE0 incomplete-buffer partition-gradient four-step fits and complete-buffer Mermin limits pass in the p102 baseline. `energy` is the electronic Mermin free energy, `internal_energy` the internal energy (Hartree); Cartesian `gradient` is its partition-relaxed derivative (Hartree/Bohr). Fixed partition membership is assumed. Public, periodic and other-class acceptance remain separate. See [raw fits](response.md#phase-b-molecular-divide-and-conquer-gradient-acceptance). |
| `mrsf` | `mrsf(numbers, positions, **controls)` | `calc.get_mrsf(atoms, **controls)` | `singlets`, `triplets` (each with `label`, `omega`, `energy`, `s_squared`, `oscillator_strength`, `transition_dipole`), `input_multiplicity`, `mrsf_coupling`, `spin_flip_asymmetry` |
| `mrsf-state` | `mrsf_state(numbers, positions, **controls)` | `calc.get_mrsf_state(atoms, **controls)` | `gradient`, `omega`, `state_energy`, `omega_reference_error`, `root_gap`, `path_evaluations` |
| `mrsf-hessian` | `mrsf_hessian(numbers, positions, **controls)` | `calc.get_mrsf_hessian(atoms, **controls)` | `hessian`, `gradient`, `frequencies_cm1`, `n_imaginary`, `translation_residual`, `omega` |
| `spin-orbit` | `spin_orbit(numbers, positions, **controls)` | `calc.get_spin_orbit(atoms, **controls)` | `spin_axes`, `band_energies`, `spin_orbit_energies`, `spinor_energies`, `soc_charges`, `self_consistent_energies` Set `reference=gks` (`--reference gks` in the CLI) to include the generated one-electron operator in the spin-covariant SCF. `self_consistent_energies` and `self_consistent_orbital_residuals` are paired per axis in Hartree. |
| `spinor-polarizability` | `spinor_polarizability(numbers, positions, *, frequencies, **controls)` | `calc.get_spinor_polarizability(atoms, *, frequencies, **controls)` | `polarizability_real`, `polarizability_imag`, `polarizability_axial`, `hermiticity_defect`, `spinor_energy`, `spin_orbit_energy`, `spin_expectation`, `response_residual` |
| `magnetic-anisotropy` | `magnetic_anisotropy(numbers, positions, **controls)` | `calc.get_magnetic_anisotropy(atoms, **controls)` | `spin_axes`, `band_energies`, `spin_orbit_energies`, `anisotropy_energies`, `soc_charges` |
| `magnon` | `magnon(numbers, positions, **controls)` | `calc.get_magnon(atoms, **controls)` | `spin_lowering_energies`, `spin_raising_energies`, `goldstone_energy`, `spin_flip_asymmetry` |
| `mcd` | `mcd(numbers, positions, **controls)` | `calc.get_mcd(atoms, **controls)` | Default canonical/legacy A/B fields; `mcd_model="thermal-manifold"` selects supplied real sublevels and normalized optical operators, `temperature` K and `bfield` au. Returns `mcd_population_c_tensor`, `mcd_state_mixing_tensor`, `mcd_thermal_populations` and finite-field normalized signal. Supplied-real-manifold public artifact acceptance passes; New source models: `thermal-complex-manifold` retains Hermitian real/imaginary quadratures; `molecular-casci-soc` generates spin-complete CASCI/SOC states in DFT/HF orbitals and fixed/isotropic thermal cross sections. CASCI is not TDDFT; scoped shared Rust/Python/ASE/both-CLI artifact02 acceptance and19 actual molecular/complex replays pass. Moving SOC, quadratic magnetic completion and measured-spectrum agreement remain open. Source follow-up adds explicit `mcd_magnetic_basis=london-linear-frozen-soc`, first-order London metric/Coulomb/optical completion, and per-state spin expectation/S-squared diagnostics; scoped linked acceptance now passes13 Rust cases and46 Python/ASE cases per CLI, with source1044-fit matrices reproduced exactly by the public core. Moving SOC and general relativistic/ROKS/GKS RPA remain separate. |
| `td-hessian` | `td_hessian(numbers, positions, *, root=1, **controls)` | `calc.get_td_hessian(atoms, **controls)` | Energy/root metadata, `gradient` (Hartree/Bohr), `hessian`, `excitation_hessian` (Hartree/Bohr^2), `frequencies_cm1`, `path_evaluations`; native directional subset measured, full tensor/public acceptance pending Full-spin `reference=ghf/gks`, `td_method=tda`, `td_spin=spinor/auto` uses the complete complex occupied-empty space, without exact total-spin labels. The isolated HF subset passes all 15 poles, 12 gradient and 144 Hessian entries and native/ASE/CLI routes with a fixed additional AO operator; full-spin RPA, moving SOC, semilocal nuclear mixed kernels and correlated pole derivatives remain pending. Full-spin Hessians currently report derivative tensors without harmonic analysis. See [bounded evidence and raw residuals](excited-state-extension.md#full-spin-hf-acceptance). `reference=roks/rohf`, `td_spin=reference-spin/auto` selects the common-orbital spin-conserving flag TDA with Berry weights (1,2,1). HF doublet/quartet all 312 gradient/Hessian fits and six native/ASE/CLI cases pass; spin-changing flag sectors and correlated pole derivatives remain unconnected. See [flag evidence](excited-state-extension.md#common-orbital-hf-acceptance). Collinear positive-SS/OS correlated TDA molecular gradients/Hessians now pass 1,248 complete component fits for custom HF-parent restricted/unrestricted C1 models and four native/ASE/CLI cases. Twelve small unrestricted Hessian entries use independently energy-validated fresh first gradients; all other entries use fresh static energies. No named semilocal/DH or ECP/ROKS/GKS correlated acceptance is inferred. See [correlated model and residuals](excited-state-extension.md#molecular-correlated-tda-nuclear-acceptance). |
| `td-ecd-giao` | `td_ecd_giao(numbers, positions, **controls)` | `calc.get_td_ecd_giao(atoms, **controls)` | `states[].rotatory_strength_london`, `rotatory_strength_london_cgs`, `rotatory_strength`, `rotatory_strength_velocity`, `transition_dipole`, `magnetic_response_residual` The p117 origin, translation, mirror and common-origin TDA-limit checks pass; the independent four-step London optical-rotation residue fit remains pending. |
| `roa` | `roa(numbers, positions, **controls)` | `calc.get_roa(atoms, **controls)` | `frequencies`, `frequencies_cm1`, `polarizability_derivatives`, `rotation_derivatives`, `quadrupole_derivatives`, the five invariant lists, `cid_forward`, `cid_backward`, `cid_right_angle_polarized`, `cid_right_angle_depolarized` |
| `hypermagnetizability` | `hypermagnetizability(numbers, positions, **controls)` | `calc.get_hypermagnetizability(atoms, **controls)` | `hypermagnetizability`, `polarizability`, `magnetizability`, `hypermagnetizability_anisotropy`, `gauge_origin`, `directional_evaluations` Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `cotton-mouton` | `cotton_mouton(numbers, positions, **controls)` | `calc.get_cotton_mouton(atoms, **controls)` | The `hypermagnetizability` keys plus `temperature`, `orientational_term`, `cotton_mouton_constant` Restricted HF H2 static tensor: all 81 ordered components agree with independently contracted fourth energy derivatives; Cotton-Mouton orientation and four-frequency static-limit fit accepted. Periodic/correlated and other-class acceptance remains pending. |
| `magneto-optical` | `magneto_optical(numbers, positions, **controls)` | `calc.get_magneto_optical(atoms, **controls)` | `frequency`, `polarizability`, `faraday_tensor`, `faraday_invariant`, `gauge`, `gauge_origin` Optional `frequency_damping` (Hartree) uses the causal complex equation. Reports retain both parts of the molecular polarizability and the field or spin-orbit derivative, plus the raw relative and reference residuals. |
| `bfield` | `magnetic_field_scf(numbers, positions, *, bfield, **controls)` | `calc.get_magnetic_field_scf(atoms, **controls)` | `field_energy`, `zeeman_energy`, `field_orbital_energies`, `bfield`, `bfield_tesla`, `series_order`, `series_tail`  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` (Hartree/e/Bohr) adds a simultaneous uniform electric field, including nuclear coupling and restoration of the charged-system laboratory energy origin. `electric_field` records the applied field. |
| `bfield-gradient` | `magnetic_field_gradient(numbers, positions, *, bfield, **controls)` | `calc.get_magnetic_field_gradient(atoms, **controls)` | `gradient`, `translation_residual`, `field_energy`, `bfield`  `bfield_initial_guess` defaults to auto; failed zero-field convergence may use a direct London-metric core seed. Requested stability following is never bypassed. `energy` is null without a converged zero-field reference; `field_energy` and `converged` describe the actual field state. `series_tail` measures omitted coefficients. Optional `field` adds the same electric scalar potential to the London SCF and analytic gradient. For charge Q, `gradient_translation_target=-Q*electric_field` (Hartree/Bohr); `translation_residual` measures the difference from that physical target. |
| `magnetoelectric` | `magnetoelectric(numbers, positions, **controls)` | `calc.get_magnetoelectric(atoms, **controls)` | `magnetoelectric_tensor`, `soc_charges`, `path_evaluations`, `response_residual` |
| `periodic-energy` | `periodic_energy(numbers, positions, cell, *, k_mesh=(1,1,1), forces=False, stress=False, **controls)` | `calc.get_periodic_energy_report(atoms, **controls)` (periodic `Atoms`) | `energy`, `free_energy`, `energy_components`, `k_points`, `k_points_reduced`, `volume`, optional `forces`, `gradient`, `translation_residual`, optional stress fields, `dispersion_energy` with `dispersion="d3bj"` 3D `dispersion="d2"` uses published functional and H-Kr atom parameters, a fixed pair cutoff in Bohr, and no ATM term. The dispersion energy is Hartree per cell, its forces Hartree/Bohr, and its stress Hartree/Bohr^3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion). Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.  VV10: `vv10`, `vv10_b`, `vv10_c`, `vv10_cutoff` (Bohr); named -V parents attach their defining term. Energy components separate semilocal/nonlocal correlation. SCF and moving Gamma kernel connected; physical/public acceptance and finite-q/frequency combinations remain pending. Explicit reference=roks/rohf source now connects an integer Gamma common-spatial-orbital reference, with closed beta subspace inside alpha, fresh physical Focks and raw orbital/metric/number diagnostics. HF/PBE source-local open-shell references and all six orbital-rotation four-step fits pass; closed-shell energies match existing periodic RKS within6.7e-16 Hartree. Core initialization/centered CDIIS, three-dimensional Gamma and zero temperature only; ROHF requires an HF provider. Finite k, thermal, shifted/damped initialization, additional nonlocal/PT2 terms and response above first order remain refused. Full-rank Gamma source now also connects analytic nuclear gradients/forces and affine strain/stress. The actual bacf85/2b60bd successor passes canonical Rust HF/PBE references, all six independent four-step orbital-rotation fits (p1.99919..2.12830, minimum R2 .999114), 11 unique public tests and two alias-CLI SCFs. Native/ASE/CLI raw common C, spin densities, physical Focks, units and residuals are retained; independent reconstruction gives common residual at most2.64e-15 Hartree. Those bacf85/2b60bd checks certify reference energy/interfaces. The later full-rank HF/PBE first-derivative subset passes all36 Cartesian/affine-strain four-step fits on the unchanged1e-13 energy,1e-11 density and1e-10 orbital gates. PBE needs a converged localized XC AO radius: xc_ao_cutoff=16 Bohr agrees with20 on this fixture; the original6-Bohr cutoff-crossing failures remain retained. Native gradient/forces use Hartree/Bohr and stress Hartree/Bohr^3; ASE converts lengths and derived units. The b895d970/80b95343 shared core passes15 unique public cases across preserved attempts and all8 native/ASE/both-CLI raw reports (288 force/gradient/stress/strain entries); maximum route difference3.97e-15 and difference from accepted analytic references2.53e-15. One false watchdog stop and one corrected metadata-stub failure remain retained. No new installed-wheel acceptance is claimed; finite-k, geometry relaxation, higher response, general GKS derivatives and material/basis/grid convergence are not inferred. Use damping=0, no_adiis=true for this core/CDIIS route. ASE construction additionally sets use_adiis=False; disabled ADIIS omits only its automatic cutoff, while explicit cutoff controls remain visible to backend validation. Conventional RKS/UKS paths retain their own existing scope. Field-free generalized increment: `reference=gks` or `ghf` on `periodic-energy` uses the complex spinor reference with optional `forces=true` and `stress=true`, without a collinear surrogate. The source-connected 3D integer all-electron/full-AO-rank stress is the complete same-gauge Coulomb/kinetic/complex-exchange/moving-XC/Pulay affine derivative divided by volume. HF/GHF and HF/SVWN5 GKS have measured all-nine-component acceptance; PBE/meta-GGA/hybrid stress and general higher response remain separate gates. `strain_derivative` and its component tensors use Hartree, `stress`/`stress_voigt` use Hartree/Bohr^3 before ASE conversion, and `pressure=-trace(stress)/3`. The ordered raw tensor and `stress_antisymmetric_residual` are retained; the Voigt view selects xx,yy,zz,yz,xz,xy without silently projecting away spatial-grid torque. Additional spin/SOC/field, finite-temperature and discarded-rank derivative sectors remain refused. See R067 in response.md for exact artifact scope, finite-grid limits and retained failures. [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free all-electron 1D/2D RKS and UKS use the dimensional finite-k reference for optional forces and stress. Field-free finite-k ROKS uses fixed integer ranks of common complex spatial orbitals on each mesh point, with separate real/imaginary matrices and raw metric, conjugation and orbital residuals. The dimensional RKS/UKS VV10 energy route supplies the reconverged oracle for all 36 force and strain fits. [Verification](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the total energy and requested forces. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `stress` | `stress(numbers, positions, cell, **controls)` | `atoms.get_stress()` (ASE Voigt order, eV/Angstrom^3, fully periodic `Atoms`; one SCF also supplies energy and forces) | `stress`, `stress_gpa`, `stress_voigt`, `pressure`, `pressure_gpa`, `strain_derivative`, `strain_derivative_terms`, `rotation_residual`, `forces`, `energy` The dedicated GKS/GHF route shares the field-free 3D Gamma spinor derivative, units and diagnostics of `periodic-energy` with `stress=true`. See [analytic stress](periodic.md#11-analytic-stress-and-cell-relaxation). [Weighted XC controls and diagnostics](rust-api.md#weighted-periodic-xc). Field-free RKS, UKS, fixed-integer ROKS and GKS also report 1D and 2D ordered affine strain with length or area normalization, active-subspace pressure and raw Coulomb/SCF residuals. The field-free scalar ECP force and affine stress include separate translated-centre terms in Hartree/Bohr and Hartree. Use `get_periodic_stress_report` for dimension-aware ASE units. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Field-free 3D RKS/UKS double hybrids use the relaxed Born-von Karman orbital and pair response at fixed reduced mesh, including the uniform Ewald background and monopole strain for charged cells. Ordered target and PT2 derivatives and raw solver residuals are reported. [Validation scope](response.md#r026-stress-elastic-high-elastic). Scalar ECP uses retained projector images on the first-order correlated lattice path. [ECP scope and numerical evidence](response.md#scalar-ecp-on-the-first-order-correlated-lattice-path). Neutral 3D collinear double hybrids accept a fixed Cartesian Berry bias in Hartree/e/Bohr. [Field convention and validation](response.md#correlated-stress-at-fixed-cartesian-electric-bias). Field-free RKS/UKS variational VV10 uses `vv10_cutoff` in Bohr. Conventional 3D finite-k force and stress have a verified NaH subset. The 1D RKS omegaB97X-V wire and 2D UKS omegaB97M-V slab use length or area normalization. Both pass all 18 force and ordered strain fits and the CLI, native Python and ASE comparisons. [Derivative scope and residuals](response.md#first-geometry-derivatives-of-variational-periodic-vv10). Field-free fixed-integer ROKS VV10 reports `nonlocal_correlation_energy` in Hartree per cell, included in `xc_energy`. [ROKS scope](response.md#first-geometry-derivatives-of-variational-periodic-vv10). The field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS routes include geometry-only dispersion in the force and strain derivatives. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `opt-fixed-cell` | `opt_fixed_cell(numbers, positions, cell, **controls)` | `calc.get_opt_fixed_cell(atoms, **controls)` | `positions`, `objective`, `forces`, `optimization_converged`, `iterations`, `trajectory` The Gamma 3D ROKS objective includes geometry-only dispersion and its gradient. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). |
| `cell-opt` | `cell_opt(numbers, positions, cell, *, pressure_gpa=0.0, **controls)` | `calc.get_cell_opt(atoms, **controls)` | `cell`, `positions`, `enthalpy`, stress fields, `forces`, `optimization_converged`, `iterations`, `trajectory` |
| `phonons` | `phonons(numbers, positions, cell, **controls)` | `calc.get_phonons(atoms, **controls)` | `force_constants`, `frequencies_cm1`, `normal_modes_cartesian`, `raw_force_constants`, `raw_asymmetry`, `acoustic_residual`, `response_solves` Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). The collinear Gamma XC Hessian includes the meta-GGA kinetic-energy-density channel with finite-k electronic occupations. Independent TPSS force and XC operator fits and installed ECP parity are pending. RKS omegaB97X-V scalar-ECP NaH passes all 21 analytic Hessian directions at electronic Gamma. [VV10 evidence](response.md#variational-vv10-gamma-hessian). The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). |
| `phonon-bands` | `phonon_bands(numbers, positions, cell, *, supercell, path, **controls)` | `calc.get_phonon_bands(atoms, *, supercell, path, **controls)` | `q_path`, `q_path_reduced`, `path_distance`, `frequencies_cm1`, optional `dos_frequencies_cm1`, `dos`; diagnostics  Optional `wavevector_direction` (Cartesian 1/Bohr) with `response_order` 0..4 returns mass-weighted and unweighted Fourier coefficients along q(t)=q+t*d. `phonon_phase=cell` uses R and `atomic` uses R+tau_final-tau_initial. Raw Hermiticity is measured without projection; supercell interpolation and missing long-range/electronic flexoelectric terms remain explicit. `lattice_flexoelectric=true` with `wavevector_direction` and order 2..4 adds fixed-mass-center internal relaxation at Gamma. Born charges are computed on the primitive reference or supplied in `born_charge_rows` (3N by 3). `born_lattice_flexoelectric` is the Born-charge lattice part, e/Bohr, contracted twice with the normalized direction; electronic/dynamic-multipole terms remain separate. Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `phonon-thermo` | `phonon_thermo(numbers, positions, cell, *, supercell, q_mesh, **controls)` | `calc.get_phonon_thermo(atoms, *, supercell, q_mesh, **controls)` | `zero_point_energy`, `internal_energy`, `free_energy`, `entropy`, `heat_capacity` and kJ/mol, J/(mol K) siblings; `temperatures`, `excluded_modes`, `imaginary_modes` Collinear periodic double hybrids use the full correlated Gamma Hessian of the force-constant supercell. The fixed-cell 3D all-electron charged route uses uniform Ewald compensation and reports primitive, force-constant and Born-von Karman charges in e. The correlated report retains orbital and PT2 residuals and force constants in Hartree/Bohr^2. Scalar ECP second-order retained-image coefficients are connected to the correlated path; independent operator, Hessian and installed acceptance are separate. [ECP second-order scope](response.md#scalar-ecp-second-order-lattice-path). Conventional scalar ECP CPKS Hessians are connected through the common driver; raw force constants, acoustic and symmetry residuals use Hartree/Bohr^2. [Separate acceptance](response.md#scalar-ecp-in-the-conventional-periodic-cpks-hessian). |
| `electron-phonon` | `electron_phonon(numbers, positions, cell, **controls)` | `calc.get_electron_phonon(atoms, **controls)` | `lambda`, `mode_lambda`, `omega_log_k`, `a2f`, `tc_allen_dynes_k`, `tc_mcmillan_k`, `tc_eliashberg_k`, `eliashberg_gaps`, `eigenvalue_derivatives` `scdft_pairing=true` adds `scdft_inputs`, static K (Hartree) and dimensionless Z. Supply `screened_pair_coulomb` or choose `scdft_screening=rpa` (thermal head) or `rpa-neutral` (fixed-number Gamma body), with `response_reciprocal_cutoff` in inverse Bohr. Generated Coulomb screening covers the complete 3D k-point difference mesh of a spin-degenerate finite-temperature reference. The phonon contribution retains the commensurate Gamma-supercell scope. [Models and diagnostics](periodic.md#static-microscopic-scdft-coulomb-screening). |
| `quasi-harmonic` | `quasi_harmonic(numbers, positions, cell, *, supercell, q_mesh, **controls)` | `calc.get_quasi_harmonic(atoms, *, supercell, q_mesh, **controls)` | `volume`, `linear_scale`, `free_energy`, `bulk_modulus_gpa`, `thermal_expansion`, `entropy_j_mol_k`, `heat_capacity_v_j_mol_k`, `heat_capacity_p_j_mol_k`, `gruneisen`, `fit_residual_rms`, `pressure_gpa`, `gibbs_energy`, `vibrational_free_energies`, `mode_gruneisen`, `mode_gruneisen_frequencies_cm1`, `mode_gruneisen_q_reduced`, `mode_gruneisen_volume`, `gruneisen_mode_average` |
| `elastic` | `elastic(numbers, positions, cell, **controls)` | `calc.get_elastic(atoms, **controls)` | `clamped_ion_gpa`, `relaxed_ion_gpa`, Birch siblings, `relaxed_ion_eigenvalues_gpa`, `strain_hessian`, `internal_strain`, `internal_relaxation`, `force_constants`, `stress`, diagnostics as in Rust Uniformly compensated charge includes the analytic Ewald monopole strain Hessian. The 3D collinear scalar ECP path includes affine and internal-strain terms. Charged and scalar ECP numerical acceptance is tracked in the [response ledger](response.md#r026-stress-elastic-high-elastic). Includes the geometry-only CHG and D3 strain, mixed and nuclear curvatures before ionic relaxation. The conventional collinear route reports the accepted SCF `orbital_residual` in Hartree alongside the coupled response residuals. [Weighted-recipe example](response.md#r026-stress-elastic-high-elastic). [Elastic report conventions and strain checks](api-guide.md#choose-elastic-constants-and-strain-diagnostics). |
| `elastic-fd` | `elastic_fd(numbers, positions, cell, *, steps=(0.01, 0.005), relax_ions=False, **controls)` | `calc.get_elastic_fd(atoms, **controls)` | `lagrangian_gpa`, `richardson_gpa`, `strain_derivative`, `reference_max_force`, `max_residual_force`, `all_converged` |
| `elastic-moduli` | `elastic_moduli(numbers, positions, cell, *, ions="relaxed", coefficients="birch", **controls)` | `calc.get_elastic_moduli(atoms, **controls)` | Moduli (GPa), `density_kg_m3`, wave speeds, `debye_temperature_k`, `born_stable` |
| `christoffel` | `christoffel(numbers, positions, cell, *, directions=None, **controls)` | `calc.get_christoffel(atoms, **controls)` | `velocities_m_s`, `polarizations`, `directions`, `density_kg_m3` |
| `material-card` | `material_card(numbers, positions, cell, *, card_path=None, rotation=None, **controls)` | `calc.get_material_card(atoms, **controls)` | `stiffness_gpa`, `compliance_per_gpa`, engineering constants, Hill moduli, `density_kg_m3`, wave speeds, `csv` |
| `equation-of-state` | `equation_of_state(numbers, positions, cell, *, scales=None, relax_ions=False, **controls)` | `calc.get_equation_of_state(atoms, **controls)` | `v0`, `v0_angstrom3`, `e0`, `b0_gpa`, `b0_prime`, `volumes`, `energies`, `analytic_pressures_gpa`, `fitted_pressures_gpa`, `pressure_fit_gap_gpa`, `residual_rms` |
| `dielectric-tensor` | `dielectric_tensor(numbers, positions, cell, **controls)` | `calc.get_dielectric_tensor(atoms, **controls)` | `epsilon_infinity`, `susceptibility`, `raw_asymmetry`, `fock_image_residual`, `field_solves`; HF water continuum DFPT dilute-limit fit: p=2.970392240 in 1/L, R^2=0.999997818686, maximum raw field residual 3.722e-14; no HF Born-charge claim Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `born-charges` | `born_charges(numbers, positions, cell, **controls)` | `calc.get_born_charges(atoms, **controls)` | Common first-order-only request: `born_charges` (e), raw `born_sum_rule`, `imaginary_residual`, `epsilon_infinity`, `susceptibility`, response solves; identical sources to polar modes Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver.; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Includes the CHG or D3 reference energy for these geometry-only terms. |
| `piezoelectric-tensor` | `piezoelectric_tensor(numbers, positions, cell, **controls)` | `calc.get_piezoelectric_tensor(atoms, **controls)` | `clamped_ion`, `relaxed_ion`, `clamped_ion_si`, `relaxed_ion_si`, `clamped_ion_full`, `relaxed_ion_full` (omitted with `relaxed_ion=False`), `born_charges`; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The relaxed-ion route includes CHG or D3 in the internal relaxation. |
| `berry-polarization` | `berry_polarization(numbers, positions, cell, **controls)` | `calc.get_berry_polarization(atoms, **controls)` | `total_reduced`, `polarization`, `polarization_c_m2`, `polarization_quanta`, `berry_phases`, `string_spread` |
| `crystal-ir` | `crystal_ir(numbers, positions, cell, *, directions=None, **controls)` | `calc.get_crystal_ir(atoms, **controls)` | `frequencies_cm1`, `mode_effective_charges`, `ir_intensities_km_mol`, `lo_frequencies_cm1`, `epsilon_static`, `epsilon_infinity`, `born_charges`; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) The Gamma Hessian includes CHG or D3. The report gives raw Hessian and Fock-image defects and the field/displacement solve residuals. |
| `static-dielectric` | `static_dielectric(numbers, positions, cell, **controls)` | `calc.get_static_dielectric(atoms, **controls)` | `epsilon_static`, `epsilon_infinity`, `born_charges`, modes; conventional 3D insulating RKS/UKS includes scalar ECP. [Acceptance](response.md#r023-born-charge-berry-polarization) Uses the same dispersion-corrected Gamma Hessian and response diagnostics as crystal IR. |
| `electronic-structure` | `electronic_structure(numbers, positions, cell, coefficients=True, **controls)` | `calc.get_electronic_structure(atoms, coefficients=True, **controls)` | `channels[]` with bands, occupations, chemical potential and edges; `mulliken`, `mulliken_spin`, `electron_count_residual`. Optional `channels[].orbitals` contains split real/imaginary C/S/F/P matrices and per-k consistency defects; `spin_weight` gives 2 for restricted and 1 for each unrestricted channel. [Units, phase convention and validated scope](response.md#complex-bloch-orbital-coefficient-export). |
| `effective-mass` | `effective_mass(numbers, positions, cell, *, k_point, band, **controls)` | `calc.get_effective_mass(atoms, *, k_point, band, **controls)` | `velocity`, `inverse_mass_tensor`, `effective_masses`, `principal_axes`, `branch_second_derivatives` |
| `boltzmann-transport` | `boltzmann_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls)`; optional `scattering_table=[[energy_hartree,lifetime_fs],...]`, `transport_frequencies_hartree` | `calc.get_boltzmann_transport(atoms, *, transport_k_mesh, transport_temperatures_k, **controls)`; same table | Common-driver DC moments and optional complex electric and thermoelectric tensors; table mode includes lifetime in SI coefficients; [limits and raw validation](periodic.md#34-energy-dependent-relaxation-time-transport) |
| `hall-transport` | `hall_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls)`; optional `scattering_table`, frequencies and `hall_onsager_tolerance=1e-10` | `calc.get_hall_transport(atoms, *, transport_k_mesh, transport_temperatures_k, **controls)`; same controls | Common-driver report with lifetime squared inside each Hall term; axes [current,electric,B], coefficient axes [electric,current,B]; raw residuals, units and model in report |
| `coherent-transport` | `coherent_transport(numbers, positions, cell, *, transport_k_mesh, transport_frequencies_hartree, **controls)` | `calc.get_coherent_transport(atoms, *, transport_k_mesh, transport_frequencies_hartree, **controls)` | Common-driver complex S/m tensors, intra/interband and intrinsic Hall decomposition, raw current/equation/Fock residuals |
| `quantum-hall` | `quantum_hall(numbers, positions, cell, **controls)` | `calc.get_quantum_hall(atoms, **controls)` | Raw Chern invariant, spin-weighted slice conductance (S), gap and link diagnostics; topology mesh and ordered plane conventions identical to Rust/CLI |
| `wannier-flow` | `wannier_flow(numbers, positions, cell, **controls)` | `calc.get_wannier_flow(atoms, **controls)` | `results` (per spin: `flow`, `centres`, `total`, `winding`, `sampled_min_gap`), `winding`, `spin_weight` |
| `periodic-cubic-fc-along` | `periodic_cubic_fc_along(numbers, positions, cell, **controls)` | `calc.get_periodic_cubic_fc_along(atoms, **controls)` | `contracted_force_constants`, `probe`, `supercell`, `path_evaluations`, `accumulation_roundoff`, `series_reference_energy_difference`, `series_reference_residual`  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc-along` | `periodic_quartic_fc_along(numbers, positions, cell, **controls)` | `calc.get_periodic_quartic_fc_along(atoms, **controls)` | `contracted_force_constants`, `probe`, `supercell`, `path_evaluations`, `accumulation_roundoff`, `series_reference_energy_difference`, `series_reference_residual`  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Every distinct path also returns its actually solved relative `response_residuals`; these are retained before any tensor permutation reuse. The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-cubic-fc` | `periodic_cubic_fc(numbers, positions, cell, **controls)` | `calc.get_periodic_cubic_fc(atoms, **controls)` | `force_constant_tensor`, `dimension`, `tensor_order`, `supercell`, `path_evaluations`, `series_reference_energy_difference`  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `periodic-quartic-fc` | `periodic_quartic_fc(numbers, positions, cell, **controls)` | `calc.get_periodic_quartic_fc(atoms, **controls)` | `force_constant_tensor`, `dimension`, `tensor_order`, `supercell`, `path_evaluations`, `series_reference_energy_difference`  Accepts `dispersion=d2/d3bj/d3/chg` with a pair cutoff in Bohr. CN and optional ATM settings belong to D3. [D2 model and verification](response.md#periodic-d2-empirical-dispersion); adds analytic dispersion third/fourth derivatives through the existing generic lattice energy. `dispersion_energy_per_primitive_cell` is Hartree; dispersion force-constant arrays and raw `dispersion_acoustic_residual`/tensor `dispersion_permutation_residual` use Hartree/Bohr^order per primitive cell. See [four-step fits](response.md#correlated-electric-dipoles-and-periodic-dispersion-force-constants); remaining electronic combinations retain their stated scope. Optional `field` selects analytic nuclear derivatives of the stationary finite-bias Berry electric enthalpy; fixed Cartesian field, insulating Gamma supercell of the k mesh. Reference residual/energy and available response residuals are reported; independent acceptance is tracked in response.md. For magnetic/joint-field nuclear derivatives use `periodic-magnetic-scf`; its accepted finite-Gamma subset and remaining combinations are documented separately. Uniformly compensated charged fixed-lattice paths are connected. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants). The driver includes the implicit CHG term for omegaB97X-D. Combined all-direction validation is pending. |
| `toec` | `toec(numbers, positions, cell, **controls)` | `calc.get_toec(atoms, **controls)` | `high_order_clamped`, `high_order_relaxed`, their `*_gpa` equivalents, `elastic_constants_clamped`, `elastic_constants_relaxed`, `internal_relaxation`, `volume`, and raw `response_residuals`. Supply `field` for neutral insulating cells to report fixed Cartesian field derivatives and `field_energy`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `foec` | `foec(numbers, positions, cell, **controls)` | `calc.get_foec(atoms, **controls)` | Same outputs and field control as `toec`, with fourth-order tensors and `second_internal_relaxation` in Bohr, indexed `[6 I + J][3 atom + axis]`. [Scope and acceptance](response.md#charged-gamma-force-constants-and-higher-elastic-constants) |
| `mode-gruneisen` | `mode_gruneisen(numbers, positions, cell, **controls)` | `calc.get_mode_gruneisen(atoms, **controls)` | `mode_gruneisen`, `mode_gruneisen_strain`, `second_log_frequency` at order two, `frequencies_cm1`, `acoustic_modes`, `force_constants`, `response_residuals`. Geometry-only `dispersion` adds `dispersion_energy_per_primitive_cell` in Hartree. |
| `periodic-beta` | `periodic_beta(numbers, positions, cell, **controls)` | `calc.get_periodic_beta(atoms, **controls)` | `susceptibility`, `dielectric_tensor`, `second_susceptibility`, `volume`, `path_evaluations`; nonzero HF water beta verified with all 27 unprojected index orders and a four-step fit; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-gamma` | `periodic_gamma(numbers, positions, cell, **controls)`; generic `run_feature("periodic-gamma", ..., cell=cell, **controls)` | `calc.get_periodic_gamma(atoms, **controls)` and generic `calc.get_feature_report("periodic-gamma", atoms, **controls)` | Shared Rust driver; native geometry Bohr, ASE geometry Angstrom. `units="atomic"` preserves the Gaussian e^2 Bohr^2/Hartree^2 tensor; ASE conversion uses e^2 Angstrom^2/eV^2. First tensor axis is polarization, remaining axes are ordered field derivatives; chi3 is one sixth of the actual derivative. `use_symmetry=True` expands 15 independent static components, False evaluates all 81; `raw_permutation_residual` is None when expansion is enabled, rather than an unmeasured zero; HF water also passes the unprojected 81-component four-step gamma acceptance; `field` / CLI `--field x,y,z` selects a self-consistent finite-bias reference through the common Berry enthalpy; Rust `lattice_field_response_at_field` / `lattice_electric_tensor_at_field` preserve zero-field compatibility wrappers. `reference_field` is Hartree/e/Bohr on native routes and converts on ASE; raw path residuals remain explicit. Biased HF/PBE/TPSS/PBE0 alpha/beta/gamma acceptance passes all twelve four-step fits; relaxed-ion biased states need ionic stationarity Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. |
| `periodic-raman` | `periodic_raman(numbers, positions, cell, **controls)` | `calc.get_periodic_raman(atoms, **controls)` | `susceptibility`, `born_charges`, `raman_tensor` |
| `electrostriction` | `electrostriction(numbers, positions, cell, **controls)` | `calc.get_electrostriction(atoms, **controls)` | `susceptibility`, `strain_susceptibility` |
| `piezo-higher` | `piezo_higher(numbers, positions, cell, **controls)` | `calc.get_piezo_higher(atoms, **controls)` | `piezoelectric_tensor`, `second_piezoelectric` |
| `periodic-efield` | `periodic_efield(numbers, positions, cell, *, field, **controls)` | `calc.get_periodic_efield(atoms, **controls)` | `field_energy`, `dipole`, `polarization`, `forces`, `field` Correlated lattice path: typed named/custom PT2 parent, full periodic Ewald integrals, every required orbital/pair order through four, integer 3D Gamma supercell model; physical acceptance tracked in response.md. Native Python, ASE and CLI use this same driver. Correlated `response_order=0..4`, `field_direction` in Hartree/e/Bohr per dimensionless path parameter; factorial-normalized energy coefficients in Hartree per primitive cell, unscaled SS/OS coefficients and raw orbital/pair/projector/metric/number/stationarity diagnostics. The three-dimensional Berry-bias route includes the lattice D2, D3/ATM or CHG correction. [Finite-field scope and acceptance](response.md#finite-electric-bias-with-empirical-dispersion). |
| `dielectric-q` | `dielectric_q(numbers, positions, cell, *, q_points, **controls)` | `calc.get_dielectric_q(atoms, **controls)` | `points` (per q: `q_reduced`, `q`, `chi`, `chi_cosine`, `chi_sine`, `inverse_dielectric`, `dielectric`), `volume` |
| `orbital-susceptibility-q` | `orbital_susceptibility_q(numbers, positions, cell, *, q_points, **controls)` | `calc.get_orbital_susceptibility_q(atoms, **controls)` | Nonzero commensurate q on an insulating 3D Gamma supercell. RKS, UKS and all-electron GKS include the meta-GGA current kernel. GKS and GHF report the full phase-averaged `chi_transverse` tensor in atomic units, energy matrices, solver residuals and raw static defects. [Convention and examples](api-guide.md#orbital-susceptibility-at-finite-q) |
| `periodic-dc` | `periodic_dc(numbers, positions, cell, **controls)` | `calc.get_periodic_dc(atoms, **controls)` | `energy`, `internal_energy`, `electronic_entropy`, `chemical_potentials`, `forces`, `stress`, `subsystem_sizes`, `density_residual`, `relaxation_residual` |
| `dc-stress` | `dc_stress(numbers, positions, cell, **controls)` | `calc.get_dc_stress(atoms, **controls)` | `stress`, `stress_gpa`, `strain_derivative`, `energy` |
| `periodic-nmr-shielding` | `periodic_nmr_shielding(numbers, positions, cell, *, q_points, **controls)` | `calc.get_periodic_nmr_shielding(atoms, **controls)` | `nuclei` (per atom: `points` with `shielding_ppm`, `paramagnetic_ppm`, `diamagnetic_ppm`, `field_direction`; `shielding_tensor_ppm`, `isotropic_ppm`) |
| `berry-dipole` | `berry_dipole(numbers, positions, cell, *, transport_k_mesh, **controls)` | `calc.get_berry_dipole(atoms, **controls)` | `berry_curvature_dipole`, `chemical_potential`, `electrons_per_cell`, `fermi_weight` |
| `lab-frame-spin-spiral` | `lab_frame_spin_spiral(numbers, positions, cell, *, spiral_q, **controls)` | `calc.get_lab_frame_spin_spiral(atoms, *, spiral_q, **controls)` | `energy`, `spiral_spin_projection`, `opposite_spin_projection`, `magnetization`, `seeded_energy`, `spiral_supercell`, `spiral_wave_vector` |
| `dmi` | `dmi(numbers, positions, cell, *, spiral_q, **controls)` | `calc.get_dmi(atoms, *, spiral_q, **controls)` | `dmi_energy`, `soc_even_energy`, `nuclear_soc_energies`, `dmi_per_wave_number`, `chirality_rotation_energy_defect`, `dmi_energy_self_consistent` |
| `bands` | `bands(numbers, positions, cell, *, path, points=20, **controls)` | `calc.get_bands(atoms, *, path, **controls)` | `k_path`, `k_path_reduced`, `path_distance`, `channels[].band_energies`, `chemical_potential` |
| `periodic-spin-response` | `periodic_spin_response(numbers, positions, cell, **controls)` | `calc.get_periodic_spin_response(atoms, **controls)` | Same common-driver 3D longitudinal spin-only response. Controls: Cartesian `response_q`, frequency, complex `spin_probe_field=[re,im]`, `spin_g_factor=2`, solver/reference controls. Native geometry Bohr, q inverse Bohr, frequency Hartree, field atomic magnetic units; ASE converts labelled controls and reports. Returns spin/charge electron numbers and magnetic moment per cell, susceptibility per field, complex density, raw response residuals/iterations and explicit model warnings. Uniform static equilibrium uses `pauli-spin-susceptibility`; orbital/transverse/SOC response is separate. No additional numerical acceptance is inferred. |
| `periodic-field-scf` | `periodic_field_scf(numbers, positions, cell, *, pbc=None, **controls)` | `calc.get_periodic_field_scf(atoms, **controls)` | Common-driver complex dimensional finite-field reference; geometry Bohr in native and converted from Angstrom in ASE; Hartree energy and analytic amplitude derivative; raw symmetry and stationarity diagnostics  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Collinear Fourier-field force and stress retain fixed reduced wave indices and Cartesian amplitudes, including current and A squared terms. Raw field-energy reconstruction defects use Hartree. The affine term uses Hartree per cell and its stress is divided by the active periodic measure. Explicit `reference=gks` or `reference=ghf` uses the shared complex spinor SCF. Nuclear spin-orbit coupling is opt-in on this entry. [Controls and validation](response.md#explicit-generalized-reference-on-static-periodic-field-scf). |
| `periodic-spinor-scf` | `periodic_spinor_scf(numbers, positions, cell, *, pbc=None, **controls)` | `calc.get_periodic_spinor_scf(atoms, **controls)` | Common variational spinor driver; native geometry Bohr, ASE Angstrom conversion; Hartree energy and amplitude derivative; one chemical potential; no orbital conjugation or separate spin-population constraint  Accepted uniform wire/in-plane-slab extension: `uniform_magnetic_field=[Bx,By,Bz]`, `magnetic_origin=[Ox,Oy,Oz]`, `field_amplitude`; wires accept every direction and ordinary slab Bloch gauges accept in-plane fields. Full continuum orbital coupling, common origin AO phase; spinor route adds `sigma.B/2`, nuclear `grad(V) cross (p+A)` and its analytic amplitude derivative. Reduced atomic field, origin in Bohr; energy/derivative in Hartree per cell. Normal slab flux, ECP/current-dependent terms and remaining response combinations are tracked separately. Rust: `UniformOpenField::{gauge_matrix,origin_phase_wave,vector_potential,operators}`, `run_dimensional_uniform_scf`, `SpinorDftBackend::with_uniform_open_field`. Joint03 bounded runtime-verified increment: explicit reference=gks/ghf selects this existing complex Bloch solver; GHF requires an HF energy provider, and ordinary/common-orbital sectors are refused. The same converged SCF mesh supplies band_energies, k_points and band_edges, with normalized band_eigen_residuals and band_metric_defects. No spin factor of two is applied. A gap is reported only for a uniform exact integer occupied prefix at zero electronic temperature; thermal or incompatible occupations leave band_edges null. This adds neither arbitrary-path interpolation nor generalized force/DFPT/Berry/ROKS support. The earlier three-dimensional two-k-point HF spin-rotation baseline passed with complex density and5.55e-17 Hartree energy difference; that earlier baseline did not execute the new report fields. Joint03 now independently checks both GKS/GHF two-k HF mesh reports, their complex density, orbital/occupation/number gates, units and occupied band-edge reconstruction. This bounded HF report proof does not establish generalized PBE or property-response accuracy. |
| `periodic-magnetic-scf` | `periodic_magnetic_scf(numbers, positions, cell, *, pbc, magnetic_field, **controls)` | `calc.get_periodic_magnetic_scf(atoms, magnetic_field=..., **controls)` | Both accept `reference="gks"` or `"ghf"` with an HF provider. Returns energy in Hartree per magnetic cell, `reference_type`, flux quanta, SCF residuals, and independent spatial and XC angular diagnostics. The optional finite-Gamma nuclear report and 3D Berry electric branch have their own controls. [Shared settings and scope](api-guide.md#run-a-uniform-magnetic-flux-scf) |
| `dos` | `dos(numbers, positions, cell, *, sigma, points=1000, energy_window=None, dos_mesh=None, projected=False, **controls)` | `calc.get_dos(atoms, *, sigma, **controls)` | `energies`, `channels[].dos`, `channels[].integrated`, `sampling_k_points`, `sigma`; with `projected` also `projection_groups`, `channels[].projected_shells`, `channels[].projected_atoms` 1D/2D/3D RKS, UKS, ROKS and GKS on the stationary k mesh. ROKS uses spin-resolved semicanonical common-orbital blocks. Reports include states per cell, exact window counts and integration and projection residuals. Separate frozen sampling uses the bulk collinear route. Field-free 1D/2D RKS and UKS and 1D/2D/3D ROKS and GKS reference energies include geometry-only dispersion. [Periodic dispersion scope and acceptance](response.md#geometry-only-dispersion-with-periodic-common-orbital-references) [Explicit periodic D1 inputs and derivative scope](response.md#periodic-d1-and-analytic-dispersion-derivatives). [Spinor VV10 energy and first geometry derivatives](response.md#first-geometry-derivatives-of-variational-periodic-vv10). |
| `pauli-spin-susceptibility` | `pauli_spin_susceptibility(numbers, positions, cell, *, exchange_enhanced=False, temperatures_k=None, fermi_k_mesh=None, **controls)`; generic `run_feature("pauli-spin-susceptibility", ..., cell=cell, **controls)` | `calc.get_pauli_spin_susceptibility(atoms, exchange_enhanced=True, unrestricted=True, multiplicity=2, **controls)` | Shared Rust driver for restricted and polarized enhanced response. Native geometry Bohr, ASE geometry Angstrom; `units="atomic"` preserves native reports. ASE units convert common chemical-potential response and spin bias to eV, compressibility/DOS to inverse eV, while magnetic field stays atomic and raw count/equation residuals stay unchanged. Bare mode retains its DOS/thermal fields and their units for both spin channels. [Model and acceptance](periodic.md#35-exchange-enhanced-static-spin-susceptibility) Bare mode also accepts unequal collinear bands and `spin_g_factor`: frozen bands are reoccupied with a common chemical potential and zero reservoir spin bias. `spin_electrons`, `fixed_number_spin_dos`, `rigid_band_spin_magnetization`, `chemical_potential_response`, `delta_spin_numbers` and raw number defects describe that ensemble. It differs from the enhanced mode's fixed baseline bias; no self-consistent response is claimed. |
| `cli` | `cli_main(argv=None)`; omitted arguments read `sys.argv[1:]`; pip console script `dft-rs` | none: an argument-list entry point has no `Atoms` method | Integer exit status; same native common driver; explicit argument lists remain supported |
| `piezoelectric-strain` | `piezoelectric_strain(*, piezoelectric_stress, piezoelectric_compliance, ion_response)` | `DftRs.get_piezoelectric_strain(...)` (static, no Atoms; native atomic input units) | Algebraic proper d=eS; atomic d, C/N and pm/V; engineering shear, matching fixed-field/clamped or relaxed inputs required. No physical reference or stability validation inferred. Joint03 bounded Rust/native/ASE/canonical and alias CLI checks pass all18 entries, SI conversions and input gates; the independent four-step fits have p=1.99814..1.99953 and minimum R2=0.999999768. Supplied tensors retain their own physical-validation responsibility. |
| `aggregate-moduli` | `aggregate_moduli(*, phases)` | `DftRs.get_aggregate_moduli(phases=...)` (no `Atoms`) | `voigt_gpa`, `reuss_gpa`, `hill_gpa`, `hashin_shtrikman_lower_gpa`, `hashin_shtrikman_upper_gpa`, `self_consistent_gpa` |
| `rock-physics` | `rock_physics(**inputs)` | `DftRs.get_rock_physics(**inputs)` (no `Atoms`) | Dry and saturated moduli, Biot parameters, wave speeds, optional Brown-Korringa tensor and thermal-conductivity bounds |
| `feature-registry` | `feature_registry()` | `DftRs.get_feature_registry()` | `features[]` with `id`, `title`, `space`, `route`, `rust`, `cli`, `python`, `ase`, `targets` |

`static_electric_tensor(..., order=2 or 3)` and
`calc.get_static_electric_tensor(..., order=2 or 3)` expose the common beta/gamma
driver. Tensor arrays are row-major with the observed dipole axis first; use
`numpy.asarray(result["tensor"]).reshape(result["shape"])` when a shaped array is
needed. All beta/gamma entries are actual derivatives without factorial scaling.
These are static, clamped-nuclei electronic tensors, including at a biased
reference field. Dynamic frequency mixing uses the separately listed complex
drivers; nuclear-relaxation contributions remain separate targets.

`response::dynamic_electric_tensor(..., &[Frequency], &config)` and native
`dynamic_electric_tensor(..., input_frequencies=...)` / ASE
`get_dynamic_electric_tensor(...)` expose the common frequency-domain alpha,
beta and gamma driver for one, two and three incoming legs. The existing paired
Casida alpha API remains compatible. Complex tensor parts are flat row-major,
observable axis first, actual mixed derivatives without factorials. Frequency
slots refer to the field axes in the same order. Each subset uses the sum of
its incoming complex frequencies; damping therefore adds, including at the
reported output. Signed real frequencies allow optical rectification,
difference-frequency mixing, electro-optic response and Kerr combinations.
The imaginary-axis form uses `[positive_frequency, 0]` pairs; the report gives
the actual imaginary complex energies. Static output-axis/field interchange
is not imposed at finite frequency. Explicit incoming pairs cannot be combined
with the scalar omega/damping defaults.

### Periodic structures: extended XYZ, `Crystal`, `PeriodicCalculator` and ASE

These are three ways to state a periodic structure once; they run the same
driver calculations as the rows of the tables above.

- CLI: without `--cell`, the comment line of an extended XYZ file supplies
  the lattice, `Lattice="ax ay az bx by bz cx cy cz"` in Angstrom, and the
  optional flags `pbc="T T F"`.
- Python: `dft_rs.Crystal(numbers, positions, cell, pbc)` is an immutable
  structure in Bohr with `from_angstrom`, `from_fractional`, `from_xyz`
  (extended XYZ), `from_ase`, `to_ase`, `volume`, `fractional_positions`,
  `scaled`, `strained`, `supercell`, `displaced`, `with_positions` and
  `with_cell`; `dft_rs.Molecule(numbers, positions)` is its molecular
  counterpart. `dft_rs.PeriodicCalculator(crystal, **controls)` and
  `dft_rs.MolecularCalculator(molecule, **controls)` store the reference
  controls; every calculation of the driver that applies to the structure
  is a method named like its function in Table 3 (`features()` lists them),
  next to `run(feature, **controls)`, `with_structure` and `with_controls`,
  and for a crystal `relax(cell=False)` (relaxed `Crystal` and report) and
  `scan_volumes`. `run_feature(feature, structure, **controls)` accepts
  either structure. Reports are `dft_rs.Report` dictionaries with attribute
  access, in the driver's atomic units; `dft_rs.units` holds the conversion
  factors of the core.
- ASE: energy, forces and stress of fully periodic `Atoms` come from one
  `periodic-energy` calculation, so the dispersion model of the calculator
  (`dispersion="d3"`, `"d3bj"`, `"chg"`) and any other driver control given
  as a calculator parameter (`three_body=True`, `disp_cutoff=...`,
  `hartree_compact_exponent=...`) apply; a calculation without a dispersion
  contribution refuses the model by name. `kpts` is accepted for `k_mesh`.
  `get_hessian` returns the Gamma-point force constants in eV/Angstrom^2,
  `get_relaxed_atoms(atoms, cell=False)` a relaxed `Atoms` in Angstrom with
  the report in `info["dft_rs_report"]`, and `get_bands` and
  `get_phonon_bands` accept an ASE `BandPath` with one continuous segment.

### Nonlinear optical presets

The same `ElectricProcess` prescription supplies all four surfaces. Optical
magnitudes are finite and nonnegative, damping is nonnegative, and the listed
order fixes whether beta or gamma is returned. Field axes follow the ordered
incoming legs. The conventional tensor label is `beta(-Omega;w1,w2)` or
`gamma(-Omega;w1,w2,w3)`; the report's output Fourier frequency is `+Omega`.
The real-frequency prescriptions follow the
[nonlinear response conventions](https://manual.q-chem.com/5.1/sec-ElectricProps.html).

| Preset | Tensor | Required optical magnitudes | Ordered incoming real frequencies | Output real frequency |
|---|---|---|---|---|
| `shg` | beta | w | w, w | 2w |
| `sfg` | beta | w1, w2 | w1, w2 | w1+w2 |
| `dfg` | beta | w1, w2 | w1, -w2 | w1-w2 |
| `optical-rectification` | beta | w | w, -w | 0 |
| `electro-optic` (alias `pockels`) | beta | w | w, 0 | w |
| `thg` | gamma | w | w, w, w | 3w |
| `efish` | gamma | w | w, w, 0 | 2w |
| `field-induced-rectification` (alias `efior`) | gamma | w | w, -w, 0 | 0 |
| `two-photon-absorption` (alias `tpa`) | gamma | w | w, w, -w | w |
| `optical-kerr` (alias `intensity-dependent-refraction`) | gamma | w | w, -w, w | w |
| `dc-kerr` (alias `quadratic-electro-optic`) | gamma | w | w, 0, 0 | w |
| `four-wave-mixing` (alias `fwm`) | gamma | w1, w2, w3 | w1, w2, -w3 | w1+w2-w3 |

Every optical leg, including a negative-frequency leg, has positive incoming
damping eta. A prescribed static leg is exactly `Frequency::Static`, including
zero damping. Thus rectification has output `0+2i eta`; DC Kerr has
`w+i eta`; optical Kerr/TPA have `w+3i eta`. The report preserves these complex
frequencies. At eta=0 the static subset uses the constrained zero-frequency
equation. Kerr naming varies between conventions; the explicit ordered legs
in this table define the requested calculation unambiguously.

Results are clamped-nuclei electronic molecular tensors. The TPA preset
provides its complex cubic response, whose absorptive part is `tensor_imag`.
Bulk susceptibilities, nonlinear refractive-index coefficients, cross-section
units and thermal orientational EFISH terms require further physical factors
and remain distinct targets. The preset does not insert angular averages,
population factors or factorials into the raw derivative tensor.

```python
from dft_rs import dynamic_beta, dynamic_gamma
shg = dynamic_beta(numbers, positions, process="shg", frequencies=[0.05], damping=0.005, xc="pbe")
kerr = dynamic_gamma(numbers, positions, process="optical-kerr", frequencies=[0.05], damping=0.005, xc="pbe")
efish = calc.get_dynamic_gamma(atoms, process="efish", frequencies=[0.05], damping=0.005, units="atomic")
```

Molecular functions accept `response_tol=1e-10` and `response_max_iter=500`;
GMRES restart currently uses the shared default of 40 in these Python calls.
Electric molecular SCF keywords include `basis="sto-3g"`, `xc="hf"`, charge,
multiplicity, grid/convergence controls, density fitting, VV10 grid resolution,
`solvent`, and `eps`. ASE supplies reference settings from the calculator.
Static field methods additionally accept `reference_field` in atomic units.

Anharmonic calls use a zero-field stationary minimum and positive projected
harmonic frequencies. IR intensities use km/mol, static Raman activities use
Angstrom^4/amu, frequencies use cm^-1, zero-point energies use Hartree, and
transition dipoles/polarizabilities use atomic units. `dipole_derivatives` and
`polarizability_derivatives` contain actual `first`, `second`, `third` nuclear
derivatives with mode indices first and Cartesian components last (3 or 9).
Patterns use electron-mass-weighted coordinates, `R_A = R_A0 + pattern_A Q`.
The force limit `equilibrium_tol=1e-4` uses Hartree/Bohr; resonance controls are
`resonance_tol=1e-8` Hartree and `resonance_coupling_factor=1.0`. Storage limits
are `max_basis_states=1024`, `max_polyad_states=256`, and at most 512 MB bounded
by `incore_mb`. Mode selection explicitly omits couplings to unselected modes.
The model is rectilinear cubic-second/quartic-first-order perturbation theory
with retained-polyad diagonalization and the same contact transformation on
dipole/static polarizability through third nuclear derivatives. Raman therefore
requires fifth energy derivatives. Equilibrium Coriolis kinetic terms and the
rotor-specific Watson pseudopotential enter the second-order Hamiltonian and
the same observable transformation. `rotation` reports Cartesian inverse
inertia, principal moments (electron mass Bohr^2), ordered-pair Coriolis
coefficients, pseudopotential (Hartree), rotor class, measured Eckart and mode
metric defects. `force_permutation_defects/comparisons` preserve the cubic and
quartic raw reverse-order measurements; `polarizability_symmetry_defect/comparisons`
measure all independently evaluated static Cartesian alpha interchanges.
A single mode has no nontrivial force-index comparison, which is reported as
zero comparisons. These are vibrational band origins with rotational levels
removed. [Model and validation](response.md#rectilinear-anharmonic-bands)
give the transformation, resonance convention and independent oracles.

Periodic functions and methods accept `omega=0.0`, `damping=0.0`,
`imaginary=False`, `fixed_number=True`, `reciprocal_cutoff=6.0`,
`response_tol=1e-10`, `response_max_iter=500`, and `response_restart=40`.
Native periodic functions also accept the reference keywords of
`periodic_scf`, including `xc="pbe"`, `electronic_temperature` in Kelvin,
`k_mesh`, and SCF/grid controls. `fixed_number=False` keeps the reference spin
chemical potentials fixed. Forces, relaxation, trajectory, and wavefunction
requests cannot be combined with these response requests.

Thermal methods use a positive reference `electronic_temperature` in Kelvin.
The temperature direction `delta_temperature` also uses Kelvin; `delta_mu`
uses Hartree and `delta_numbers` uses particles per cell per spin. The default
direction returns a derivative per Kelvin. Thermal requests accept reciprocal
and solver controls but do not accept q, frequency, or mode selection.

```python
from dft_rs.native import dynamic_polarizability, periodic_dielectric

report = dynamic_polarizability(
    [1, 1], [[0, 0, 0], [0, 0, 1.4]], xc="pbe",
    omega=0.05, damping=0.01,
)
alpha_zz = complex(report["alpha_real"][2][2], report["alpha_imag"][2][2])

dielectric = periodic_dielectric(
    [1, 1], [[0, 0, 0], [0, 0, 1.4]],
    [[9, 0, 0], [0, 9, 0], [0, 0, 9]], xc="pbe",
    q=(0.04, 0.01, 0.02), modes=(0,), reciprocal_cutoff=0.75,
)
```


## Occupied-projector delta-SCF on Python and ASE

The existing `energy`, `gradient`, `orbitals` and `run_feature` functions
accept `occupation_mode`, `occupied_alpha` and `occupied_beta` through their
ordinary controls. The matching `DftRs.get_energy_report`,
`get_gradient_report` and `get_orbitals_report` methods forward exactly the same
controls. Indices and mode are dimensionless; native geometry is Bohr,
while ASE Atoms geometry is always Angstrom. The physical score, residual,
returned occupied-first orbital order and pending combinations are defined
in the [shared determinant contract](rust-api.md#occupied-projector-delta-scf-controls).

```python
controls = dict(xc="pbe0", basis="sto-3g", occupation_mode="imom",
                occupied_alpha=[1], occupied_beta=[1])
report = dft_rs.energy([1,1], [[0,0,-0.7],[0,0,0.7]], **controls)
# atoms is the same geometry in Angstrom.
adapted = dft_rs.DftRs(units="atomic").get_energy_report(atoms, **controls)
print(report["energy"], report["occupation_tracking"])
```

Setting the same controls on `DftRs` also connects standard ASE
`Atoms.get_potential_energy()` and `Atoms.get_forces()` to the selected
reference, with its diagnostics under `calculator.results['scf']`. Combined
wavefunction/NBO export uses separate report getters and remains unconnected
to this selected-state ASE calculate route.

For single promotions with unequal spatial alpha/beta occupied spaces,
pass `unrestricted=True` and both complete lists. No spin projection is
implied. With default ASE units the energy and nested initial/difference
energies are eV; projector weights, gaps and map residuals are unchanged.


## Fixed populations on native Python and ASE

```python
controls = dict(xc="pbe0", occupation_mode="fixed",
                occupations_alpha=[0.8,0.2], occupations_beta=[0.8,0.2])
native = dft_rs.gradient([1,1], [[0,0,-0.7],[0,0,0.7]], **controls)
adapted = dft_rs.DftRs(units="atomic").get_gradient_report(atoms, **controls)
atoms.calc = dft_rs.DftRs(**controls)
forces = atoms.get_forces()
```

The ordinary report getters and `run_feature` share the common driver.
Native positions are Bohr; ASE Atoms positions are always Angstrom.
Standard ASE energy, force, charge and dipole calls honor the constraint.
With default ASE report units, energy/gap are eV and gradients eV/Angstrom;
`units="atomic"` gives Hartree and Hartree/Bohr. Populations and the raw
density-map residual do not change units. Diagnostics are also available
under `calculator.results['scf']`. Use `unrestricted=True` for unequal
alpha/beta arrays. The [shared Rust contract](rust-api.md#fixed-orbital-population-controls)
defines zero padding, number/spin constraints, the spin moment, degeneracy
and frontier conventions, physical model and pending response combinations.

Static electric response and the analytic Hessian/APT use those same
population controls, including a pre-existing electric bias:

```python
electric = dft_rs.polarizability_report(numbers, bohr_positions, **controls)
nuclear = dft_rs.hessian_report(numbers, bohr_positions, **controls)
electric_ase = dft_rs.DftRs(units="atomic").get_polarizability_report(atoms, **controls)
nuclear_ase = dft_rs.DftRs(units="atomic").get_hessian_report(atoms, **controls)
```

The native electric tensor is e^2 Bohr^2/Hartree; default ASE report units
give e^2 Angstrom^2/eV. Native Hessians are Hartree/Bohr^2 and default ASE
Hessians eV/Angstrom^2; the APT is e on both routes. Electric
`number_defects` carry e Bohr/Hartree (e Angstrom/eV on ASE), whereas Hessian
`ensemble_number_defects` carry electrons/Bohr (electrons/Angstrom on ASE).
Raw equation residuals and population diagnostics retain their meanings.
`electronic_temperature` selects the distinct canonical Fermi response.
See the [shared spectral-response contract](rust-api.md#fixed-population-and-thermal-static-response)
for all result shapes, gauges, degeneracy conditions and pending combinations.

### Named hybrid acceptance

``xc="wc04"`` and ``xc="wp04"`` use the explicit VWN-RPA convention and
independent fitted coefficients in FUNCTIONALS.md. The existing native
and ASE common-driver energy/gradient methods accept them directly, with
no implicit solvent, dispersion or nonlocal correlation correction.

``xc="blyp35"`` selects 0.35 full-range HF exchange + 0.65 B88 exchange +
full LYP correlation, with no implicit dispersion/nonlocal term, through
native ``energy``/``gradient``/``run_feature`` and ASE
``DftRs.get_gradient_report``. Native/report energy and gradient units are
Hartree and Hartree/Bohr. Numerical acceptance and unvalidated response
combinations are recorded in [FUNCTIONALS.md](FUNCTIONALS.md).

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

The normal Python release suite checks every registered native wrapper and
ASE driver getter against the existing Rust inventory, including forwarding
and a single resource envelope. Compiled discovery is compared with the source
being verified. `python tools/ledger_check.py --check` also rejects unavailable
documented public entries and unreviewed method-matrix status values. A
`validated_subset` needs an explicit scope and evidence. These guards detect
contract drift; representative physical integration and component fits are
still required to validate the claimed numerical subset.

<!-- python-public-reference:begin -->
## Complete Python and ASE declaration reference

This section describes every explicit public function, class, method and
property in `native.py`, `system.py`, the ASE adapter and the IGMH
visualization module, including
signatures and defaults. It is maintained by
`python tools/api_reference.py --write`; `--check` rejects stale or
missing entries and missing source descriptions without importing the core.

The [Python and ASE calculation table](#table-3-native-python-and-ase),
[Rust table](rust-api.md#table-1-rust) and [CLI table](cli.md#table-2-cli) describe
the matching Rust/CLI calls, units, physical scope and result conventions.
Dynamic calculator methods come from that same registry and have signature
`calculator.<registered_python_name>(**controls)`. Inherited ASE base-class
methods follow the ASE contract. Private implementation helpers are excluded.

The signatures below are the explicit source declarations. Native functions
and explicit DftRs calculation/report getters also accept keyword-only
`cpu_threads`, `memory_mb`, `log_file`, `log_mode` and
`log_level` (all default to `None`); `inspect.signature` exposes the combined
call signature. See [persistent logs](#persistent-calculation-logs) and the
[resource contract](rust-api.md#errors-and-threading) for their shared semantics.

The top-level exports in `dft_rs.__all__` are aliases of the declarations
below, plus the documented `native` and `units` modules. Their coverage is
checked explicitly; a new exported module or name requires documentation.

### `dft_rs.native.Report`

```python
class Report(dict)
```

A driver report: a dictionary whose entries are also attributes.

``report["energy"]`` and ``report.energy`` are the same value. Reports
are in atomic units unless a key names another unit; ``report.units``
names the unit of every entry.

[Source](../python/dft_rs/native.py#L118).

### `dft_rs.native.Report.unit`

```python
Report.unit(self, key)
```

The unit label of an entry, or ``None`` when it has none: from
``units`` (driver reports) or ``unit_labels`` (dedicated functions),
by the name or the dotted path of the entry.

[Source](../python/dft_rs/native.py#L135).

### `dft_rs.native.Report.require_converged`

```python
Report.require_converged(self)
```

Return the report itself, or raise when it is not converged.

[Source](../python/dft_rs/native.py#L145).

### `dft_rs.native.calculate`

```python
calculate(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, gradient: bool=False, wavefunction: bool=False, nbo: bool=False, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, field: Optional[Sequence[float]]=None, solvent: Optional[str]=None, eps: Optional[float]=None, pseudo_sub: Optional[str]=None, unrestricted: bool=False, stabilize: bool=False, grid_preset: Optional[str]=None, grid_angular_scheme: Optional[str]=None, grid_weight_cutoff: Optional[float]=None) -> dict[str, Any]
```

Run one SCF and return its results.

Parameters
----------
numbers
    Atomic numbers, 1 through 118. A bare ``0`` is rejected because it
    does not identify which element's basis should be placed at a ghost
    centre.
positions
    Cartesian coordinates in **Bohr**, shape ``(natoms, 3)``.
basis
    Bundled family name (see `basis_sets`), a path to a Gaussian94
    ``.g94`` file, or a per-element spec ``"base; SYM,SYM=source; ..."``
    — e.g. ``"def2-tzvp; H=def2-svp"`` or
    ``"def2-svp; Ni,Fe=lanl2dz"``. ``"@file"`` reads the spec from a
    file.
ecp
    Core potentials, in the SAME grammar and INDEPENDENT of ``basis``:
    applied afterwards, touching only the elements it names, with
    ``"none"`` clearing. This is what lets the orbitals and the core
    potential come from different sets —
    ``basis="def2-tzvp", ecp="Ni=stuttgart-rsc-1997"`` — or lets a set's
    bundled potential be dropped while keeping its orbitals.
xc
    Combined functional name, e.g. ``"b3lyp"``, ``"wb97m-v"``, ``"hf"``.
    Functionals whose definition includes VV10 or the CHG tail enable it
    automatically.
dispersion
    ``"auto"`` (default), ``"none"``, ``"d2"``, ``"d3zero"``,
    ``"d3bj"``, ``"chg"``, or self-consistent ``"vv10"``. Explicitly
    adding dispersion to a functional that already includes it is an
    error rather than double counting.
density_fit
    Auxiliary basis for RI-J, for example ``"def2-universal-jfit"``.
    Only Coulomb J is fitted; exchange K remains exact.
grid_level
    Quadrature resolution 0–9 (Lebedev angular rules, radial points and
    angular degree by period). Ignored for pure Hartree–Fock, which
    needs no grid.
grid_preset
    ``"fine"``, ``"ultrafine"`` or ``"superfine"``: named per-atom
    (radial, Lebedev) grids (75, 302), (99, 590), (150/225, 974) that
    replace ``grid_level``.
grid_angular_scheme
    ``"lebedev"`` (default) or ``"product"`` angular rules.
grid_weight_cutoff
    Partitioned quadrature weights below this value (Bohr^3, default
    1e-14) are dropped; ``0`` keeps every point, so the energy is a
    smooth function of the nuclear positions.
gradient
    Also compute the analytic nuclear gradient.
unrestricted
    Explicitly choose UKS/UHF, including a singlet reference.
stabilize
    Follow internal orbital instabilities before calculating every result.
    The final-state energy, gradient, charges, dipole and wavefunction share
    one reference. The stability report includes every accepted descent;
    external instabilities do not silently change the reference class.
wavefunction
    Also return MO energies and coefficients, the density, and the
    overlap matrix.
nbo
    Also return the metric-correct NAO/NPA/Wiberg foundation. This does
    not yet perform NHO construction or Lewis-structure NBO localization.
field
    Uniform external electric field ``(Ex, Ey, Ez)`` in atomic units,
    added as ``+E.r`` to the core Hamiltonian and ``-Z_A R_A.E`` to the
    nuclear energy. Composes with everything else this function returns:
    the energy, dipole, and (with ``gradient=True``) the analytic
    gradient are all exact in the field.
solvent
    Continuum solvation, using the SAME name grammar as the CLI's
    ``--solvent``: a bundled name (e.g. ``"water"``), optionally
    prefixed with ``"smd:"`` (SMD electrostatics plus its
    non-electrostatic cavity/dispersion term), ``"cosmo:"``,
    ``"ief:"``, or ``"ssvpe:"`` to pick the continuum model (default
    C-PCM), or ``"custom"`` together with ``eps`` for a bare dielectric
    constant with no named solvent. The reaction field enters the SCF
    self-consistently, so it is exact at the converged density —
    including under ``gradient=True``.
eps
    Override (or, with ``solvent="custom"``, supply) the static
    dielectric constant directly.
pseudo_sub
    Zero-based named-site assignments such as ``"4=Me"``. Me, Et, tBu,
    Ph, CF3 and OMe are recognized, but all currently raise
    ``NotImplementedError`` before SCF because no provenance-bearing
    fitted operator has passed its physical acceptance gates.

Returns
-------
dict
    Always ``energy``, ``e_elec``, ``e_nuc``, ``e_xc``, ``iterations``,
    ``converged``, ``n_alpha``, ``n_beta``, ``dipole`` (a.u.),
    ``mulliken``, ``e_nlc``, ``e_dispersion``, ``e_pcm`` (the PCM
    electrostatic polarization energy, 0 unless ``solvent`` was given)
    and ``e_smd_cds`` (SMD's non-electrostatic term, 0 unless
    ``solvent`` used the ``"smd:"`` prefix; ``energy`` already includes
    both). With ``gradient=True`` also ``gradient`` (Hartree/Bohr, shape
    ``(natoms, 3)``). With ``wavefunction=True`` also ``wavefunction``,
    containing ``mo_energy``, ``mo_coeff``, ``density`` and ``overlap``.

Examples
--------
Water at its equilibrium geometry, in Bohr:

>>> from dft_rs.native import calculate
>>> r = calculate(
...     [8, 1, 1],
...     [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]],
...     basis="def2-svp",
...     xc="b3lyp",
... )
>>> round(r["energy"], 6)  # doctest: +SKIP
-76.331...

With forces and the wavefunction:

>>> r = calculate([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
...               xc="svwn5", gradient=True, wavefunction=True)
>>> len(r["gradient"]), len(r["wavefunction"]["mo_energy"][0])  # doctest: +SKIP
(2, 2)

[Source](../python/dft_rs/native.py#L154).

### `dft_rs.native.hessian`

```python
hessian(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, reference: Optional[str]=None, **controls: Any) -> dict[str, Any]
```

Cartesian Hessian in Hartree/Bohr^2.

Analytic where the method supports it, central differences of the
analytic gradient otherwise. The returned ``method`` says which ran, so
the caller never has to guess. Supported empirical-dispersion terms stay
on the analytic path. A specifically unsupported functional derivative
uses the numerical path; other analytic failures are propagated instead
of being silently hidden by a fallback.

An explicit ``reference`` (RKS/UKS/ROKS/GKS and HF aliases), or extra
driver controls, selects the common analytic ``hessian`` driver. That
path has no numerical fallback and retains the model's stated scope.

Returns
-------
dict
    ``hessian`` (3N x 3N nested lists), ``method`` (``"analytic"`` or
    ``"numerical"``), ``energy``.

Examples
--------
>>> from dft_rs.native import hessian
>>> r = hessian([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], xc="svwn5")
>>> r["method"]  # doctest: +SKIP
'analytic'

[Source](../python/dft_rs/native.py#L346).

### `dft_rs.native.vibrations`

```python
vibrations(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, temperature: float=298.15, pressure: float=101325.0, symmetry_number: int=1) -> dict[str, Any]
```

Harmonic frequencies, IR intensities, and ideal-gas RRHO thermochemistry.

Eckart-projected. ``n_projected`` reports how many translation/rotation
modes were removed (6, or 5 for a linear molecule) and ``n_imaginary``
the saddle order — a converged minimum has zero. For a minimum,
``thermochemistry`` contains ZPE, thermal energy, enthalpy, entropy and
Gibbs quantities in Hartree (entropy in Hartree/K). It is ``None`` for a
saddle because no transition-state convention is assumed silently.

Examples
--------
>>> from dft_rs.native import vibrations
>>> v = vibrations([8, 1, 1],
...                [[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]],
...                xc="b3lyp", basis="sto-3g")
>>> v["n_imaginary"]  # doctest: +SKIP
0

[Source](../python/dft_rs/native.py#L426).

### `dft_rs.native.raman`

```python
raman(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, hessian_step: float=0.005, raman_step: float=0.005) -> dict[str, Any]
```

Frequencies, IR intensities, and Raman activities.

Normal modes use the analytic Hessian whenever the complete requested
method supports it; ``hessian_method`` reports the selected path. Raman
activities are in Angstrom^4/amu.

Returns
-------
dict
    ``frequencies`` (cm^-1), ``ir_intensities`` (km/mol),
    ``raman_activities`` (Angstrom^4/amu), ``n_imaginary``,
    ``n_projected``, and ``hessian_method``.

[Source](../python/dft_rs/native.py#L489).

### `dft_rs.native.optimize`

```python
optimize(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, max_steps: int=200, gtol: float=0.0003, step_max: float=0.3, history: int=8, trajectory: Optional[str]=None) -> dict[str, Any]
```

Optimize a molecular geometry with analytic-gradient L-BFGS.

Returns ``positions`` (Bohr), ``energy`` (Hartree), ``converged``,
``iterations``, and a ``trajectory`` list. Each trajectory item contains
``step``, ``positions``, ``energy``, and ``grad_max`` (Hartree/Bohr).
When ``trajectory`` is a path, accepted frames are flushed as XYZ while
the walk proceeds.

[Source](../python/dft_rs/native.py#L545).

### `dft_rs.native.saddle`

```python
saddle(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, method: str='prfo', max_steps: int=50, gtol: float=0.0003, step_max: float=0.2, hessian_step: float=0.005, trajectory: Optional[str]=None) -> dict[str, Any]
```

Locate a first-order saddle with analytic-first Hessians.

``method="prfo"`` selects partitioned RFO. ``method="rsirfo"`` selects
the coupled Householder image function with an intrinsic restricted step.
``step_max`` is the Euclidean trust radius for RS-I-RFO; its generalized
RFO metric is solved so the step itself satisfies the restriction.

Returns ``method``, final ``positions`` (Bohr), ``energy`` (Hartree),
``converged``, ``iterations``, ``n_negative``, and the same trajectory
schema as `optimize`.

Raises
------
ValueError
    For an unknown method or invalid numeric controls.
NotImplementedError
    When a required analytic component has no numerical fallback.

[Source](../python/dft_rs/native.py#L602).

### `dft_rs.native.polarizability`

```python
polarizability(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3) -> dict[str, Any]
```

Static dipole polarizability (a.u., 3x3) from electric-field CPKS.

Returns
-------
dict
    ``alpha`` (3x3 nested list, ``alpha[x][y] = d(mu_x)/d(E_y)``),
    ``isotropic`` (the trace over 3), and the ground-state ``energy``
    (Hartree). ``alpha`` is symmetric to numerical precision — a useful
    sanity check on the caller's side too.

Examples
--------
>>> from dft_rs.native import polarizability
>>> r = polarizability([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], xc="svwn5")
>>> r["isotropic"] > 0  # doctest: +SKIP
True

[Source](../python/dft_rs/native.py#L671).

### `dft_rs.native.dynamic_polarizability`

```python
dynamic_polarizability(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, omega: float=0.0, damping: float=0.0, imaginary: bool=False, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, solvent: Optional[str]=None, eps: Optional[float]=None, reference: Optional[str]=None, unrestricted: bool=False, scf_damping: Optional[float]=None, **options) -> dict[str, Any]
```

Paired Casida dipole response with frequencies in Hartree.

``alpha_real`` and ``alpha_imag`` are 3x3 lists in bohr^3.
Positive damping selects the retarded response; ``imaginary=True``
instead evaluates z=i*omega and requires non-negative omega and zero
damping. All three original-operator residuals accompany the tensor.
Common-driver names frequency/response_omega, frequency_damping/
response_damping and imaginary_frequency are accepted as aliases.
An alias cannot accompany a non-default value of its legacy parameter.
``scf_damping`` selects dimensionless SCF density mixing independently
of the optical ``damping`` in Hartree; it maps to the common driver's
reference ``damping`` control.
``thermal_response="collisionless"`` explicitly selects the positive-
temperature collinear Fermi-population Liouvillian. Reference populations
remain fixed; its zero-frequency limit differs from isothermal Mermin
response. The default ``"integer"`` retains integer-projector validation.
The thermal route supports local/GGA and exact-exchange parent kernels;
correlated, spinor and constrained common-orbital extensions are excluded.
For integer RKS/UKS double hybrids, nonzero frequency uses a parent,
target and spin-resolved PT2 quasi-energy response. Its real and
imaginary term tensors accompany the total dipole response.

[Source](../python/dft_rs/native.py#L727).

### `dft_rs.native.stability`

```python
stability(numbers, positions, *, n_roots=3, roots=None, channels=None, tolerance=1e-07, max_iterations=200, max_subspace=60, block_buffer=3, resolution=None, verify_lowest=True, verify_steps=24, follow_unstable=False, max_descents=3, step_angle=0.3, min_energy_drop=1e-08, unrestricted=False, **options)
```

Analytic collinear real/imaginary orbital stability in Hartree.

Restricted references include singlet/triplet blocks; unrestricted
references include independent spin variations. Numerical convergence
and physical stability are separate report fields. Each mode includes
its normalized generator, actual energy curvature and fresh residual.
Accepts the dedicated stability binding's SCF/response options, without
frequency or electric-process controls. The dynamic-only ``scf_damping``
alias is not a keyword of this legacy binding.

[Source](../python/dft_rs/native.py#L803).

### `dft_rs.native.stabilize`

```python
stabilize(numbers, positions, **options)
```

Follow internal instabilities and return the final reference and analysis.

[Source](../python/dft_rs/native.py#L837).

### `dft_rs.native.dynamic_electric_tensor`

```python
dynamic_electric_tensor(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, input_frequencies: Optional[Sequence[Sequence[float]]]=None, imaginary: bool=False, process: Optional[str]=None, frequencies: Optional[Sequence[float]]=None, damping: float=0.0, **options) -> dict[str, Any]
```

Actual complex alpha/beta/gamma derivatives at one/two/three input legs.

Each input is [omega, damping] in Hartree; imaginary-axis inputs instead
use [positive frequency, 0]. Negative real omega supports frequency mixing.
Output frequency is the complex sum. Flat tensor_real/tensor_imag have
shape [3] repeated number_of_inputs+1, observable axis first. No output
frequency or damping is independently substituted for a subset sum.
Accepts the same SCF/response options as dynamic_polarizability.

[Source](../python/dft_rs/native.py#L842).

### `dft_rs.native.dynamic_electric_process`

```python
dynamic_electric_process(numbers, positions, *, process, frequencies, damping=0.0, **options)
```

Select a physical nonlinear optical frequency preset, Hartree inputs.

Examples: process="shg", frequencies=[0.05]; process="optical-kerr",
frequencies=[0.05]. The report preserves the ordered frequency legs.

[Source](../python/dft_rs/native.py#L866).

### `dft_rs.native.dynamic_beta`

```python
dynamic_beta(numbers, positions, *, input_frequencies=None, process=None, frequencies=None, damping=0.0, **options)
```

Full 27-component beta, using explicit legs or an optical preset.

Presets: shg, sfg, dfg, optical-rectification and electro-optic.
Supply process and nonnegative frequencies (Hartree), with damping in
Hartree, or two input_frequencies=[omega, damping] pairs. The output
frequency is the sum of the incoming legs.

[Source](../python/dft_rs/native.py#L876).

### `dft_rs.native.dynamic_gamma`

```python
dynamic_gamma(numbers, positions, *, input_frequencies=None, process=None, frequencies=None, damping=0.0, **options)
```

Full 81-component gamma, using explicit legs or an optical preset.

Presets: thg, efish, field-induced-rectification,
two-photon-absorption, optical-kerr, dc-kerr and four-wave-mixing.
The first six take one nonnegative optical frequency; four-wave-mixing
takes three. Frequencies and damping use Hartree. Alternatively supply
three input_frequencies=[omega, damping] pairs. Optical legs retain
positive damping, including negative frequencies; DC legs are static.
The tensor is the ordered cubic dipole susceptibility, without optical
intensity, refractive-index, polarization or macroscopic conversion.

[Source](../python/dft_rs/native.py#L891).

### `dft_rs.native.electric_field_taylor`

```python
electric_field_taylor(numbers: Sequence[int], positions: Sequence[Sequence[float]], direction: Sequence[float], *, order: int=3, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, solvent: Optional[str]=None, eps: Optional[float]=None, reference_field: Optional[Sequence[float]]=None, **controls) -> dict[str, Any]
```

Analytic dipole series for E(t)=reference_field+t*direction.

Positions, fields and dipoles use atomic units. Coefficient n is the
nth directional derivative divided by n!, so beta and gamma along the
supplied direction are 2*coefficient[2] and 6*coefficient[3].

[Source](../python/dft_rs/native.py#L940).

### `dft_rs.native.mixed_electric_field_response`

```python
mixed_electric_field_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], directions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, solvent: Optional[str]=None, eps: Optional[float]=None, reference_field: Optional[Sequence[float]]=None, **controls) -> dict[str, Any]
```

Analytic mixed dipole derivative, one field direction per derivative.

Repeated directions are allowed. Two directions select beta, three
select gamma, with no factorial normalization in the returned vector.
All quantities and geometry use atomic units.

[Source](../python/dft_rs/native.py#L967).

### `dft_rs.native.static_electric_tensor`

```python
static_electric_tensor(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, order: int=2, assembly: str='projector', basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, solvent: Optional[str]=None, eps: Optional[float]=None, reference_field: Optional[Sequence[float]]=None, **controls) -> dict[str, Any]
```

Complete analytic static beta (order=2) or gamma (order=3), atomic units.

``tensor`` is a flat row-major array with ``shape`` [3,3,3] or [3,3,3,3].
The first axis is the measured dipole; the remaining axes are ordered
field derivatives. Entries are actual derivatives, without factorial
scaling. Ordinary variational tensors do not use symmetry filling; their
raw field and dipole/field interchange
measurements accompany the ordinary variational tensor. Nuclei are clamped;
frequencies vanish. For correlated RKS/UKS references, the common driver
differentiates the full orbital/PT2 energy and returns separate target,
unscaled SS/OS and nonlocal tensors. Scalar-energy polarization reuses
permutations; observed-leg interchange diagnostics are None with zero
comparisons. Raw absolute and relative amplitude residuals are retained.
Parent ROHF/ROKS and complex GHF/GKS use stationary total-energy
polarization with common-orbital or full spinor metric constraints.
Integer GKS also supports a common PT2 coefficient, with generalized
pair and target contributions on the same parent spinor path.
For ordinary beta, ``assembly="variational"`` uses the 2n+1 theorem and three
first-order solves; ``assembly="projector"`` is the default recurrence.

[Source](../python/dft_rs/native.py#L994).

### `dft_rs.native.beta`

```python
beta(numbers: Sequence[int], positions: Sequence[Sequence[float]], **options) -> dict[str, Any]
```

Full static first hyperpolarizability; options match static_electric_tensor.

Integer RKS/UKS double hybrids use complete orbital/PT2 energy derivatives,
including reference_field bias, with parent/target and separate SS/OS tensors.
Integer GKS accepts a common PT2 weight and returns full-spin parent,
target and pair tensors through the same driver.

[Source](../python/dft_rs/native.py#L1039).

### `dft_rs.native.gamma`

```python
gamma(numbers: Sequence[int], positions: Sequence[Sequence[float]], **options) -> dict[str, Any]
```

Full static second hyperpolarizability; options match static_electric_tensor.

Integer RKS/UKS double hybrids use complete fourth-order orbital/PT2 energy
paths, including reference_field bias, and return all 81 Cartesian entries.
Integer GKS accepts a common PT2 weight and returns full-spin parent,
target and pair tensors through the same driver.

[Source](../python/dft_rs/native.py#L1050).

### `dft_rs.native.nmr`

```python
nmr(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, gauge: str='giao', origin: Optional[Sequence[float]]=None, max_iter: int=150, density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, response_restart: int=40, unrestricted: bool=False) -> dict[str, Any]
```

Return closed/open collinear orbital GIAO shielding tensors in ppm.

``gauge="common_origin"`` is available for explicit origin studies;
production calculations should use the default London-orbital path.

The result contains ``gauge``, ``origin`` (Bohr), ``energy`` (Hartree)
and ``shieldings``. Each nucleus has ``tensor``, ``diamagnetic`` and
``paramagnetic`` 3x3 arrays plus their isotropic values, all in ppm.
ECP or nonlocal-functional combinations with missing magnetic derivatives raise
``NotImplementedError`` rather than returning an incomplete tensor.

[Source](../python/dft_rs/native.py#L1061).

### `dft_rs.native.magnetic_response`

```python
magnetic_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, gauge: str='giao', origin: Optional[Sequence[float]]=None, max_iter: int=150, density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, response_restart: int=40, unrestricted: bool=False) -> dict[str, Any]
```

Return spin orbital dP/dB = i*density_hat with raw relative residuals.

Arrays have axes [spin, external B, spherical AO, spherical AO].
Fields use reduced atomic units in (p+A)^2/2; origin uses Bohr.
The report includes overlap and fixed-density Fock derivatives, and
the zero-field overlap/Fock/density/orbitals for independent verification.
Spin Zeeman, hyperfine and thermal paramagnetic shifts are separate.

[Source](../python/dft_rs/native.py#L1119).

### `dft_rs.native.tddft`

```python
tddft(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, nstates: int=5, method: str='tda', spin: str='auto', max_iter: int=150, dispersion: str='auto', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3) -> dict[str, Any]
```

Excitation energies, oscillator strengths, and rotatory strengths (ECD).

Parameters
----------
method
    ``"tda"`` (Tamm-Dancoff, default) or ``"rpa"`` (full TDDFT/TDHF,
    the Casida A/B form).
spin
    ``"singlet"`` or ``"triplet"`` for a restricted (closed-shell)
    reference, ``"spin_resolved"`` for an unrestricted one, or
    ``"auto"`` (picks singlet/spin_resolved from the reference actually
    used — the same default the CLI applies).

Returns
-------
dict
    ``ground_state_energy`` (Hartree), ``method``, ``spin`` (the
    manifold actually solved in, as a string), and ``states``: a list of
    per-state dicts with ``energy`` (Hartree), ``energy_ev``,
    ``oscillator_strength`` (length-gauge, non-negative),
    ``transition_dipole`` (a.u., length 3),
    ``rotatory_strength_length`` and ``rotatory_strength_velocity``
    (a.u.; multiply by 471.4438 for the customary 1e-40 cgs unit),
    ``transition_angular_momentum``, ``transition_momentum``, and
    ``tamm_dancoff`` (whether this state came from TDA or full RPA).

    The length-gauge rotatory strength is origin-dependent in a finite
    basis; the velocity gauge is not. On an achiral molecule both must
    vanish, so their gap there is the basis-set error bar.

Examples
--------
>>> from dft_rs.native import tddft
>>> r = tddft([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], nstates=2)
>>> len(r["states"])  # doctest: +SKIP
2

[Source](../python/dft_rs/native.py#L1175).

### `dft_rs.native.sapt`

```python
sapt(numbers_a: Sequence[int], positions_a: Sequence[Sequence[float]], numbers_b: Sequence[int], positions_b: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge_a: int=0, multiplicity_a: int=1, charge_b: int=0, multiplicity_b: int=1, grid_level: int=3, max_iter: int=150, density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048) -> dict[str, Any]
```

Default molecular interaction-energy response decomposition for HF or KS
monomers. Use `sapt_analysis(..., sapt_variant="sapt0", xc="hf")` for the
full-antisymmetrizer SAPT(0) route.

The two monomers are given as separate geometries — unlike the CLI,
which takes one XYZ split by atom count. Each keeps its own charge and
multiplicity; the basis and functional are shared. The response model uses
Heitler-London first-order exchange and its first variation, coupled
induction, and TDA dispersion with projected exchange-dispersion.

An ECP basis raises `NotImplementedError`. A monomer whose first-order
orbital rotation leaves the small-rotation regime raises `ValueError`.

Returns
-------
dict
    ``e_monomer_a``, ``e_monomer_b``, ``e_elst``, ``e_exch``,
    ``e_ind_a``, ``e_ind_b``, ``e_exch_ind``, ``e_disp_unc`` (uncoupled,
    diagnostic), ``e_disp_sos_uncoupled`` (diagnostic — calibrates the
    spin factor against ``e_disp_unc``), ``e_disp_coupled`` (from the
    monomers' TDA spectra), ``e_exch_disp``, ``e_int_supermolecular``
    (counterpoise-corrected), ``delta`` (supermolecular minus the
    reported total — the higher-order remainder), and
    ``interaction_total`` = ``e_elst + e_exch + e_ind_a + e_ind_b +
    e_exch_ind + e_disp_coupled + e_exch_disp`` (all Hartree).

Examples
--------
>>> from dft_rs.native import sapt
>>> r = sapt([1], [[0.0, 0.0, 0.0]], [1], [[0.0, 0.0, 3.5]])
>>> r["interaction_total"] == (
...     r["e_elst"] + r["e_exch"] + r["e_ind_a"] + r["e_ind_b"]
...     + r["e_exch_ind"] + r["e_disp_coupled"] + r["e_exch_disp"]
... )  # doctest: +SKIP
True

[Source](../python/dft_rs/native.py#L1257).

### `dft_rs.native.nuclear_taylor`

```python
nuclear_taylor(numbers: Sequence[int], positions: Sequence[Sequence[float]], direction: Sequence[Sequence[float]], *, order: int=4, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='hf', charge: int=0, multiplicity: int=1, grid_level: int=3, max_iter: int=150, dispersion: str='none', density_fit: Optional[str]=None, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, incore_mb: int=2048, vv10_grid_level: int=3, response_tol: float=1e-10, response_max_iter: int=500, reference_field: Optional[Sequence[float]]=None) -> dict[str, Any]
```

Analytic energy coefficients along R(t) = R(0) + t direction.

Positions and per-atom displacements use Bohr; energy coefficients use
Hartree. Coefficient n is the directional derivative divided by n!.
Order four includes cubic and quartic force constants. The report also
contains component energies, density/overlap coefficients and number,
metric, projector and stationarity diagnostics. The grid topology and
reference AO rank are fixed; AO values and Becke weights move analytically.

Current physical kernels cover all-electron exact-integral HF, semilocal,
global hybrid and range-separated hybrid models without additional
nonlocal, empirical or solvent terms. A uniform reference field is allowed.

[Source](../python/dft_rs/native.py#L1336).

### `dft_rs.native.orbital_energy_gradients`

```python
orbital_energy_gradients(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, coordinates: Optional[Sequence[int]]=None, degeneracy_tolerance: float=1e-08, unrestricted: bool=False, **options: Any) -> dict[str, Any]
```

Relaxed orbital gradients in Hartree/Bohr, including moving-basis terms.

Cartesian IDs are 3*atom+axis; None requests all coordinates. Nuclear
driver calculation keywords are accepted, including reference_field.
Degenerate/near-degenerate groups return full projected derivative
matrices and None for their individual gradients. Their actual energy
spread is retained; separately sorted directional spectra must not be
combined into multivariate orbital branches.

[Source](../python/dft_rs/native.py#L1372).

### `dft_rs.native.frontier_gap_gradient`

```python
frontier_gap_gradient(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, coordinates: Optional[Sequence[int]]=None, degeneracy_tolerance: float=1e-08, unrestricted: bool=False, **options: Any) -> dict[str, Any]
```

Per-spin neutral-reference LUMO-HOMO gaps and relaxed gradients.

This is an orbital gap, not a charged-state fundamental gap. Missing
occupied/virtual spaces yield None; grouped frontier levels have no
unique Cartesian gradient. The common orbital report retains their
projected derivative matrices for explicit directional calculations.

[Source](../python/dft_rs/native.py#L1398).

### `dft_rs.native.mixed_nuclear_response`

```python
mixed_nuclear_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], directions: Sequence[Sequence[Sequence[float]]], **options: Any) -> dict[str, Any]
```

Actual mixed energy derivative along per-atom Bohr displacements.

Three/four directions select cubic/quartic force constants. No factorial
is applied to the returned derivative. Calculation keywords are those
of nuclear_taylor except order; the direction count sets the order.

[Source](../python/dft_rs/native.py#L1422).

### `dft_rs.native.molecular_taylor`

```python
molecular_taylor(numbers: Sequence[int], positions: Sequence[Sequence[float]], nuclear_direction: Sequence[Sequence[float]], electric_direction: Sequence[float], *, order: int=4, **options: Any) -> dict[str, Any]
```

Energy and dipole jets along simultaneous nuclear and electric paths.

Geometry and nuclear paths are in Bohr; electric fields are in atomic
units. All coefficients are divided by n!. Calculation keywords are
those of nuclear_taylor, including an optional reference_field.

[Source](../python/dft_rs/native.py#L1441).

### `dft_rs.native.mixed_molecular_response`

```python
mixed_molecular_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], nuclear_directions: Sequence[Sequence[Sequence[float]]], electric_directions: Sequence[Sequence[float]], **options: Any) -> dict[str, Any]
```

Actual mixed energy and dipole derivatives for paired nuclear/field paths.

The two direction lists must have equal nonzero lengths. No factorial
is applied. Roundoff estimates and physical response residuals accompany
both observables. Zero nuclear or electric entries select pure legs.

[Source](../python/dft_rs/native.py#L1460).

### `dft_rs.native.static_spectroscopic_derivative`

```python
static_spectroscopic_derivative(numbers: Sequence[int], positions: Sequence[Sequence[float]], nuclear_direction: Sequence[Sequence[float]], *, electric_order: int=2, **options: Any) -> dict[str, Any]
```

Directional analytic APT (1), Raman (2), or hyper-Raman (3).

The flat tensor has Cartesian shape (3,) repeated electric_order times,
with the output dipole index first. Nuclear directions use Bohr. All
27 hyper-Raman entries are evaluated in their own field-slot order;
permutation_defect measures raw entries before any symmetrization.
The common driver accepts generic integer RKS/UKS post-SCF targets,
including PT2 orbital/amplitude response through energy order three
for Raman and four for hyper-Raman. Tensor units follow the supplied
Bohr nuclear path and the electric indices; per-entry unit_labels are
authoritative. Parent ROHF/ROKS and complex GHF/GKS share the stationary
total-energy connection; correlated ROKS/GKS and dynamic spectra remain
separate. Provider-specific acceptance is recorded in response.md.

[Source](../python/dft_rs/native.py#L1480).

### `dft_rs.native.analytic_force_constants`

```python
analytic_force_constants(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, order: int=3, patterns: Optional[Sequence[Sequence[Sequence[float]]]]=None, coordinates: Optional[Sequence[int]]=None, **options: Any) -> dict[str, Any]
```

Analytic cubic/quartic pattern, Cartesian block, or full Cartesian tensor.

Omit patterns and coordinates for all 3N coordinates. Coordinate IDs
are 3*atom+axis; patterns are atom-by-Cartesian Bohr displacements.
The actual derivative tensor has no factorial normalization. Ordinary
references compare reverse orders before permutation filling. Correlated
integer RKS/UKS paths include full orbital and noncanonical PT2 amplitude
response with separate parent/target, unscaled SS/OS and dispersion
tensors. ROKS/ROHF common-orbital and complex GKS/GHF paths use the
same complete tensor assembly with parent/dispersion channels and raw
reference/projector diagnostics; acceptance is tracked in response.md.
Energy-polarization reuse is explicit: permutation_defect is
None and comparisons zero rather than an inferred measurement. The acoustic
translation residual is measured only for a complete Cartesian block;
an unmeasured residual is None with zero comparisons.

[Source](../python/dft_rs/native.py#L1509).

### `dft_rs.native.anharmonic`

```python
anharmonic(numbers: Sequence[int], positions: Sequence[Sequence[float]], *, mode_ids: Optional[Sequence[int]]=None, include_raman: bool=False, equilibrium_tol: float=0.0001, resonance_tol: float=1e-08, resonance_coupling_factor: float=1.0, max_basis_states: int=1024, max_polyad_states: int=256, method: str='retained-polyad', **options: Any) -> dict[str, Any]
```

Analytic molecular rectilinear anharmonic IR and optional static Raman.

Positions use Bohr. The reference must be a stationary minimum. Force
fields and first/second/third dipole or polarizability derivatives use
analytic response, including fifth energy order for Raman. The default
``retained-polyad`` preserves contact-transformed polyad diagonalization.
``strict-vpt2`` uses the nonresonant second-order expansion and rejects
coupled near-resonant reported states. Intensities square transition
moments corrected through second order; they are not order-truncated
intensity expansions. Returned metadata identifies the chosen treatment.
A selected-mode Hamiltonian omits couplings to unselected modes explicitly.
Equilibrium Coriolis and rotor-specific Watson kinetic terms are included
consistently at VPT2 order, including the observable transformation.

[Source](../python/dft_rs/native.py#L1541).

### `dft_rs.native.periodic`

```python
periodic(numbers: Sequence[int], positions: Sequence[Sequence[float]], cell: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, charge: int=0, multiplicity: int=1, k_points: Optional[Sequence[Sequence[float]]]=None, overlap_cutoff: float=20.0, derivatives: bool=False, density: Optional[Sequence[Sequence[float]]]=None) -> dict[str, Any]
```

Periodic one-electron matrices, optional Hartree, nuclear energy and stress.

Parameters
----------
cell
    3x3 lattice in **Bohr**, rows being the lattice vectors.
positions
    Cartesian coordinates in **Bohr**; they need not lie inside the cell.
k_points
    Cartesian reciprocal vectors (1/Bohr). Omit for the Gamma point;
    `monkhorst_pack` builds a regular mesh.
overlap_cutoff
    Real-space cutoff (Bohr) for the SHORT-range one-electron lattice
    sums. This is a genuine convergence parameter — Gaussian overlap
    decays exponentially, so there is a limit to converge to. It does
    **not** control the Coulomb sums, which are conditionally
    convergent and handled by Ewald summation instead; conflating the
    two converges the electrostatics to the wrong number.
derivatives
    Also return the nuclear derivatives of every one-electron matrix.
    Off by default: they cost considerably more than the matrices, and
    a caller who only wants a band structure has no use for them.
density
    Optional finite, symmetric ``nao x nao`` Gamma-point density matrix.
    When supplied, the result also contains ``hartree`` at every k point
    and ``density_electron_count``. General real-space ``D(T)`` blocks
    are available through the Rust API.

Returns
-------
dict
    ``nao``, ``n_electrons``, ``volume``, ``net_charge``,
    ``nuclear_repulsion``, ``charged_cell_correction``,
    ``nuclear_energy``, ``nuclear_stress`` (3x3), ``image_counts``,
    ``background_shift``, ``k_points``, and ``overlap`` / ``kinetic`` /
    ``nuclear`` / ``core_hamiltonian`` — one entry per k point, each a
    dict of ``real`` and ``imag`` matrices (the core keeps its linear
    algebra real, so Hermitian matrices come back as a pair).

    With ``derivatives=True``, also ``derivatives``: one entry per k
    point, each a dict with the same four operator keys, indexed
    ``[atom][axis]`` down to the same ``real``/``imag`` pair.

``background_shift`` is the ``G = 0`` constant that ``nuclear`` carries
times the overlap. It pairs with the background already inside
``nuclear_repulsion``: neither is meaningful alone, and anything
comparing ``nuclear`` against a molecular reference has to account for
it.

Notes
-----
**Charged cells.** When ``charge`` is non-zero the Makov-Payne
correction is applied automatically and also reported separately as
``charged_cell_correction``. The neutralising background makes the
energy finite, but it cannot remove the charge's interaction with its
own periodic images, and that artefact decays only as ``1/L`` — on a
molecular ion it is routinely larger than the basis-set error.

``nuclear_repulsion`` does **not** approach the molecular value in a
large cell, and should not: nuclei carry net positive charge, so the
periodic electrostatic energy exists only alongside the neutralising
background, whose contribution cancels against the matching one in the
electronic energy. Only the total is physical.

**Scope.** This function is the low-level operator API: one-electron
matrices, their nuclear derivatives, nuclear terms, and Hartree for a
supplied density. Use `periodic_scf` for finite-k exact exchange,
XC-SCF, a periodic total energy, and analytic forces.

Examples
--------
>>> from dft_rs.native import periodic, monkhorst_pack
>>> cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
>>> r = periodic([1, 1], [[0, 0, 0], [0, 0, 1.4]], cell,
...              k_points=monkhorst_pack(cell, [2, 2, 2]))  # doctest: +SKIP
>>> r["nuclear_stress"]  # doctest: +SKIP

[Source](../python/dft_rs/native.py#L1571).

### `dft_rs.native.periodic_scf`

```python
periodic_scf(numbers: Sequence[int], positions: Sequence[Sequence[float]], cell: Sequence[Sequence[float]], *, basis: str='sto-3g', ecp: Optional[str]=None, xc: str='pbe', charge: int=0, multiplicity: int=1, pbc: Sequence[bool]=(True, True, True), k_mesh: Sequence[int]=(1, 1, 1), overlap_cutoff: float=20.0, exchange_cutoff: Optional[float]=None, grid_level: int=3, grid_shell: int=2, max_iter: int=100, conv_tol: float=1e-09, conv_tol_grad: Optional[float]=None, conv_tol_dp: float=1e-08, forces: bool=False, wavefunction: bool=False, electronic_temperature: float=0.0, level_shift: float=0.0, damping: float=0.25, diis_size: int=8, use_adiis: bool=True, adiis_until_error: float=0.1, guess: str='core', relax: bool=False, opt_max_steps: int=200, opt_gtol: float=0.0003, opt_step_max: float=0.3, opt_history: int=8, trajectory: Optional[str]=None, response: Optional[dict[str, Any]]=None, grid_preset: Optional[str]=None, grid_angular_scheme: Optional[str]=None, grid_weight_cutoff: Optional[float]=None) -> dict[str, Any]
```

Run finite-k periodic HF or generalized Kohn-Sham SCF.

All inputs and outputs use atomic units: positions and lattice vectors
are in Bohr, energies in Hartree, and analytic forces in Hartree/Bohr.
``pbc`` explicitly records the three periodic axes. Partial boundaries
are represented by the API but rejected until their dimensional Ewald
kernels are available. ``k_mesh`` is an unshifted,
time-reversal-complete uniform mesh. Exact
exchange, LDA/GGA/meta-GGA XC (including the kinetic-density ``tau``
channel), Ewald Hartree, and analytic electronic/nuclear forces all use
that same finite-k wavefunction. A positive ``electronic_temperature``
(Kelvin) enables Fermi-Dirac occupations and returns both internal
``energy`` and Mermin ``free_energy``.
Under smearing, analytic forces are derivatives of ``free_energy``;
``force_energy`` records this convention in the result.
``level_shift``, ``damping``, ``diis_size``, ``use_adiis``, and
``adiis_until_error`` expose the corresponding periodic SCF convergence
controls instead of silently fixing them in the native wrapper. ADIIS is
used only with integer occupations; finite-temperature runs use CDIIS.
``guess`` selects ``"core"`` (the compatibility-preserving default),
``"gwh"``, or the periodic ``"sad"`` path. SAD builds a periodic HF
seed from on-the-fly spherical atomic densities before applying the
requested finite-k occupations and target functional.
With ``relax=True``, atomic positions are optimized against analytic
periodic forces while lattice vectors remain fixed. ``opt_max_steps``,
``opt_gtol``, ``opt_step_max``, and ``opt_history`` control that L-BFGS
optimization. ``trajectory`` streams accepted frames to an XYZ file;
cell-vector optimization remains unavailable until total electronic
stress is implemented.

``response`` is an advanced dictionary of compiled-core response controls
used by the ASE adapter. Prefer the explicit public functions
``periodic_density_response``, ``periodic_field_response``,
``periodic_dielectric`` and ``periodic_thermal_response`` for response calculations and their unit
contracts. Those reports are also documented in ``docs/python-api.md#table-3-native-python-and-ase``.

Functionals whose definition includes nonlocal VV10 or CHG dispersion
are rejected until that nonlocal term has a periodic implementation;
returning only their semilocal part would not be the named functional.

Returns ``energy``, its electronic/nuclear/XC components, convergence
data, and Cartesian ``k_points``. With ``forces=True`` it also returns
``forces``, ``electronic_forces``, ``nuclear_forces``, and
``force_method='analytic'``. With ``wavefunction=True`` it returns
complex Bloch matrices as separate ``real`` and ``imag`` arrays.

[Source](../python/dft_rs/native.py#L1677).

### `dft_rs.native.periodic_field_response`

```python
periodic_field_response(numbers, positions, cell, *, q=(0, 0, 0), scalar_potential=0j, vector_potential=(0j, 0j, 0j), omega=0.0, damping=0.0, imaginary=False, fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10, response_max_iter=500, response_restart=40, **scf_options)
```

Screened orbital density and current for periodic electric/magnetic fields.

Geometry uses Bohr, q inverse Bohr, scalar electron potential V=-phi
Hartree, and vector potential reduced momentum. Fields use exp(i q.r-i z.t),
E=i q V+i z A, B=i q cross A. The complex density/current is integrated per
cell; charge observables have the opposite electron-number sign. Explicit
diamagnetic current is returned separately. Bra k+q is independently built.
The current backend uses 3D all-electron collinear references with
reciprocal Hartree and adiabatic LDA/GGA density screening. Matching
HF/hybrid parents also connect the full/range-separated exchange kernel;
the independent field/current acceptance remains the documented LDA/GGA
subset. Other functional/internal-space kernels,
dimensional screening, uniform Berry fields and magnetic flux SCF remain
tracked work. A continuity defect is reported without gauge projection.

[Source](../python/dft_rs/native.py#L1804).

### `dft_rs.native.periodic_density_response`

```python
periodic_density_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], cell: Sequence[Sequence[float]], *, q: Sequence[float]=(0, 0, 0), mode: int=0, omega: float=0.0, damping: float=0.0, imaginary: bool=False, fixed_number: bool=True, reciprocal_cutoff: float=6.0, response_tol: float=1e-10, response_max_iter: int=500, response_restart: int=40, **scf_options) -> dict[str, Any]
```

Screened density response to a unit plane-wave scalar potential.

Geometry uses Bohr; Cartesian q and reciprocal_cutoff use inverse Bohr;
omega and damping use Hartree. The source is mode ``mode`` of the complete
spherical set |G+q| <= reciprocal_cutoff, sorted by magnitude and then
Cartesian components. The uniform zero mode is omitted. Returned Bloch
densities retain independent (k+q,k) indices. The Fourier density is per
volume (bohr^-3) per unit potential (Hartree). ``scf_options`` are the
arguments of periodic_scf, including electronic_temperature in Kelvin.

Current physical kernels support 3D collinear semilocal KS references,
including meta-GGA, plus full/range-separated exact exchange. The
reference exchange cutoff and finite-mesh pair weights are retained.
Complete exchange-kernel fits, 360 Gamma charge/spin-source component
fits and eighteen mesh/frequency dense cases pass for the stated
restricted-reference HF/PBE0/CAM-B3LYP subset in response.md. Learned/nonlocal/correlated kernels remain
separate requirements; no missing defining term is silently omitted.

[Source](../python/dft_rs/native.py#L1837).

### `dft_rs.native.periodic_dielectric`

```python
periodic_dielectric(numbers: Sequence[int], positions: Sequence[Sequence[float]], cell: Sequence[Sequence[float]], *, q: Sequence[float]=(0, 0, 0), modes: Sequence[int]=(0,), omega: float=0.0, damping: float=0.0, imaginary: bool=False, fixed_number: bool=True, reciprocal_cutoff: float=6.0, response_tol: float=1e-10, response_max_iter: int=500, response_restart: int=40, **scf_options) -> dict[str, Any]
```

Selected inverse dielectric entries, epsilon^-1 = I + v*screened_chi.

Units, reference controls, and reciprocal mode ordering match
periodic_density_response. ``modes`` selects both reported axes; all
reciprocal modes within the cutoff still participate in screening.
v=4*pi/|G+q|^2 multiplies the density row. chi is the fully screened density
response to external potential, in bohr^-3/Hartree.

[Source](../python/dft_rs/native.py#L1874).

### `dft_rs.native.periodic_thermal_response`

```python
periodic_thermal_response(numbers: Sequence[int], positions: Sequence[Sequence[float]], cell: Sequence[Sequence[float]], *, delta_temperature: float=1.0, delta_numbers: Sequence[float]=(0.0, 0.0), delta_mu: Sequence[float]=(0.0, 0.0), fixed_number: bool=True, reciprocal_cutoff: float=6.0, response_tol: float=1e-10, response_max_iter: int=500, response_restart: int=40, **scf_options) -> dict[str, Any]
```

Uniform static self-consistent Mermin thermodynamic response.

The positive reference electronic_temperature and delta_temperature use
Kelvin; delta_mu uses Hartree and delta_numbers is per spin per cell.
Fixed-number requests use delta_numbers; reservoir requests use delta_mu.
Returned density/entropy variations are dimensionless/in k_B units;
internal/free-energy and chemical-potential variations use Hartree.
A unit temperature direction therefore returns derivatives per Kelvin.
All geometry is in Bohr. No finite-temperature differences are used.

[Source](../python/dft_rs/native.py#L1901).

### `dft_rs.native.periodic_basis`

```python
periodic_basis(numbers, positions, cell, *, basis='sto-3g', pbc=(True, True, True), k_mesh=(1, 1, 1), basis_cutoff=12.0, grid_level=None, grid_cutoff=12.0, grid_derivatives=False)
```

Gaussian Bloch S/T and analytic atomic derivatives, in atomic units.

Positions/cell/cutoffs use Bohr. Optional atom-centred quadrature covers
one periodic cell and open directions; weights have the 3D Bohr^3 measure.
This foundation is independent of electron occupations and functional.
It is not a self-consistent energy calculation.

[Source](../python/dft_rs/native.py#L1930).

### `dft_rs.native.dimensional_scf`

```python
dimensional_scf(numbers, positions, cell, *, basis='sto-3g', xc='pbe', charge=0, multiplicity=1, pbc=(True, True, True), k_mesh=(1, 1, 1), **options)
```

Self-consistent dimensional HF/DFT in a common compensated charge gauge.

Geometry and explicit widths/cutoffs are Bohr, energy is Hartree per cell.
A charged wire/slab requires Gaussian countercharge parameters. Raw fresh
physical Fock, occupations, number and quadrature diagnostics are returned.
This is the zero-field collinear driver; ECP/nonlocal tails remain pending.

[Source](../python/dft_rs/native.py#L1946).

### `dft_rs.native.dimensional_field_response`

```python
dimensional_field_response(numbers, positions, cell, *, transfer_fractional=(0.0, 0.0, 0.0), scalar_potential=0j, vector_potential=(0j, 0j, 0j), omega=0.0, damping=0.0, imaginary=False, fixed_number=True, response_tol=1e-10, response_max_iter=500, response_restart=40, **scf_options)
```

Screened orbital fields in 1D/2D/3D; geometry Bohr, frequency Hartree.

transfer_fractional labels reciprocal lattice vectors on periodic axes;
open coordinates must be zero. Integer transfers retain the microscopic
source wave but use the exact uniform Bloch sector and number constraint.
V=-phi and reduced A use exp(i wave.r-i z.t); E=i wave V+i z A,
B=i wave cross A. Density/current are complex per-cell quantities.
All-electron collinear LDA/GGA density screening is implemented here;
current XC, spin, SOC, Berry and magnetic flux reference extensions remain
tracked. Raw continuity, quadrature and stationary-reference defects are
returned independently of response convergence.

[Source](../python/dft_rs/native.py#L1961).

### `dft_rs.native.periodic_field_operators`

```python
periodic_field_operators(numbers, positions, cell, *, basis='sto-3g', charge=0, pbc=(True, True, True), k_mesh=(1, 1, 1), basis_cutoff=12.0, wave=(0.0, 0.0, 0.0), k_offset=(0.0, 0.0, 0.0), frequency=(0.0, 0.0), scalar_potential=(0.0, 0.0), vector_potential=None)
```

Analytic scalar/current/contact Fourier vertices for 1D/2D/3D.

Coordinates/cutoff use Bohr and Cartesian wave/k offsets inverse Bohr.
Complex potentials use [real,imag], and frequency [Hartree,damping].
The first derivative connects (k+q,k), second order (k+2q,k).
This operator report is independent of SCF occupations or functional.

[Source](../python/dft_rs/native.py#L1986).

### `dft_rs.native.periodic_xc_operators`

```python
periodic_xc_operators(numbers, positions, cell, *, density_real, density_imag=None, basis='sto-3g', xc='pbe', charge=0, pbc=(True, True, True), k_mesh=(1, 1, 1), basis_cutoff=12.0, grid_cutoff=12.0, grid_level=3, operator_tolerance=1e-11, grid_tolerance=1e-05)
```

Semilocal energy/components and analytic spin/Bloch XC potentials.

Geometry/cutoffs use Bohr; densities use [spin=2,k,AO,AO] spherical AO
ordering and uniform k weights. Exact-exchange coefficients and required
nonlocal tails are reported separately; this is the semilocal component,
without SCF stationarity or Coulomb compensation inferred from charge.

[Source](../python/dft_rs/native.py#L2007).

### `dft_rs.native.periodic_coulomb_operators`

```python
periodic_coulomb_operators(numbers, positions, cell, *, basis='sto-3g', charge=0, pbc=(True, True, True), k_mesh=(1, 1, 1), basis_cutoff=12.0, density_real=None, density_imag=None, nuclear_widths=None, compensation='neutral', countercharge_center=None, countercharge_width=None, alpha=None, real_cutoff=None, reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11)
```

Contracted spherical nuclear/Hartree operators and fixed-density gradients.

All geometry and explicit widths/cutoffs use Bohr. Densities are uniformly
weighted [k,AO,AO] Hermitian matrices; their imaginary parts are retained
even without time reversal. No density requests nuclear operators only.
Formal charge fixes the physical compensation and target electron count.
Charged wires/slabs require a Gaussian countercharge; uniform is 3D only.
The report includes every split component, raw defects, actual electron
count, and fixed AO coefficient gradients. These are Coulomb components
before SCF stationarity and Pulay/charge constraints, not SCF forces.

[Source](../python/dft_rs/native.py#L2025).

### `dft_rs.native.periodic_gaussian_integral`

```python
periodic_gaussian_integral(positions, exponents, angular, cell, *, pbc=(True, True, True), source_charge=None, source_width=None, alpha=None, real_cutoff=None, reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11)
```

Normalized Cartesian primitive Coulomb integral and centre gradients.

Three positions mean two Gaussian centres followed by a nuclear source;
provide two exponents and angular triples. Four positions mean an ERI
(ab|cd); provide four exponents and angular triples, without source knobs.
Geometry/widths/cutoffs use Bohr, exponents Bohr^-2, source charge e.
Nuclear defaults are charge 1 and point width 0. Values are Hartree and
gradients Hartree/Bohr in input centre order. The 3D Green function has
zero cell average; isolated wire/slab gauges match periodic_electrostatics.
These are primitive Green integrals before Bloch or density contraction.
A neutral electronic calculation must combine all charge contributions.

[Source](../python/dft_rs/native.py#L2053).

### `dft_rs.native.periodic_electrostatics`

```python
periodic_electrostatics(positions: Sequence[Sequence[float]], charges: Sequence[float], cell: Sequence[Sequence[float]], *, pbc: Sequence[bool]=(True, True, True), widths: Optional[Sequence[float]]=None, compensation: str='neutral', countercharge_center: Optional[Sequence[float]]=None, countercharge_width: Optional[float]=None, alpha: Optional[float]=None, real_cutoff: Optional[float]=None, reciprocal_cutoff: Optional[float]=None, quadrature_order: int=96, tolerance: float=1e-11, k_mesh: Sequence[int]=(1, 1, 1)) -> dict[str, Any]
```

Dimensional Coulomb energy and analytic forces for prescribed charges.

All lengths use Bohr, charges e, energy Hartree and forces Hartree/Bohr.
A zero source width is a point with its bare self energy excluded.
Positive widths define normalized exp(-r^2/width^2) Gaussian charges
with finite self energy. Nonperiodic vectors may be zero and have no
influence on images, electrostatics or k points.

Charged 3D sources can select compensation="uniform". Charged 1D/2D
sources use compensation="gaussian" with an explicit center and width;
this physical countercharge is repeated only in periodic directions.
Its center force/potential, finite self energy, and the potential gauge
are reported. This is the static Coulomb foundation, not an electronic
SCF energy. Quadrature defects measure two different Fourier rules;
real/reciprocal cutoffs must also be converged.

[Source](../python/dft_rs/native.py#L2079).

### `dft_rs.native.periodic_capabilities`

```python
periodic_capabilities() -> dict[str, bool]
```

Report executable periodic kernels and staged feature boundaries.

[Source](../python/dft_rs/native.py#L2116).

### `dft_rs.native.monkhorst_pack`

```python
monkhorst_pack(cell: Sequence[Sequence[float]], mesh: Sequence[int], *, pbc: Sequence[bool]=(True, True, True)) -> list[list[float]]
```

Regular k-point mesh for a lattice, Cartesian in 1/Bohr.

>>> from dft_rs.native import monkhorst_pack
>>> len(monkhorst_pack([[5,0,0],[0,5,0],[0,0,5]], [2, 2, 2]))
8

[Source](../python/dft_rs/native.py#L2121).

### `dft_rs.native.basis_sets`

```python
basis_sets() -> list[str]
```

Names of the bundled basis-set families.

>>> from dft_rs.native import basis_sets
>>> "def2-svp" in basis_sets()  # doctest: +SKIP
True

[Source](../python/dft_rs/native.py#L2134).

### `dft_rs.native.functionals`

```python
functionals() -> list[dict]
```

The functional registry, one dict per canonical name.

Keys: ``name``, ``class`` (``exchange``, ``correlation``, ``combined``
or ``double-hybrid``), ``family`` (``lda``, ``gga``, ``meta-gga``),
``description``, ``exact_exchange`` and ``long_range_exchange``
(fractions of the full exchange), ``omega`` (range separation,
1/Bohr, 0 when unused), ``nonlocal`` (``vv10``, ``chg`` or None),
``pt2_same_spin`` and ``pt2_opposite_spin`` (double hybrids, else
None), ``distinct_target`` (an xDH target evaluated on the parent
orbitals) and ``dispersion`` (the recipe adds its own D3(BJ) term).
Every combined and double-hybrid name is accepted as ``xc``.
Named functional parameters of the parts can be overridden on the
driver calculations with ``xc_parameters={"name": value}``, for example
``xc="rscan", xc_parameters={"switch_lower": 1, "switch_upper": 1}``.

>>> from dft_rs.native import functionals
>>> [f["pt2_same_spin"] for f in functionals() if f["name"] == "b2plyp"]  # doctest: +SKIP
[0.27]

[Source](../python/dft_rs/native.py#L2144).

### `dft_rs.native.run_feature`

```python
run_feature(feature: str, numbers: Any=None, positions: Optional[Sequence[Sequence[float]]]=None, *, cell: Optional[Sequence[Sequence[float]]]=None, pbc: Optional[Sequence[bool]]=None, **controls: Any) -> Report
```

Run any calculation registered with the common driver.

The structure is given as ``numbers`` and ``positions`` (Bohr), with
``cell`` rows (Bohr) and ``pbc`` for a periodic one, or as a single
object with those attributes (`dft_rs.Molecule`,
`dft_rs.Crystal`) in place of ``numbers``.
``feature_registry()["features"]`` lists the registered IDs.

[Source](../python/dft_rs/native.py#L2181).

### `dft_rs.native.energy`

```python
energy(numbers, positions, **controls) -> dict[str, Any]
```

Molecular total energy (Hartree) with every energy component.

Named double hybrids use this same entry: ``xc='b2plyp'``,
``'dsd-pbep86'`` or ``'dsd-blyp'`` (2013 D3(BJ) recipes), ``'pwpb95'``
or ``'wb97m2'``. The latter uses a distinct target on omegaB97M-V
orbitals and reports reference and target nonlocal terms separately. Any existing
parent accepts ``pt2_same_spin`` and ``pt2_opposite_spin`` coefficients.
Reports separate parent, dispersion, SS/OS PT2 and amplitude residuals.

[Source](../python/dft_rs/native.py#L2217).

### `dft_rs.native.mermin_free_energy`

```python
mermin_free_energy(numbers, positions, **controls) -> dict[str, Any]
```

Electronic Mermin free energy A=E-T_e*S_e in Hartree.

A Fermi-Dirac reference at
``electronic_temperature`` (Kelvin), with entropy, chemical potentials,
and occupations. Use ``gradient=True`` for the analytic nuclear
gradient with a supported provider. ROKS and ROHF share spatial
orbitals and fix both spin electron counts. GKS and GHF fix the total
electron count, with one chemical potential and one occupation per
spinor.
The primary result is mermin_free_energy; electronic entropy is in k_B.
Nuclear translation, rotation, vibration and pressure are excluded.
For RKS and UKS, set gibbs=True to reuse the analytic Hessian for an
additional gibbs_free_energy and term-by-term RRHO thermochemistry.
temperature defaults to the positive electronic_temperature, otherwise
298.15 K; pressure is Pa and symmetry_number is dimensionless. Positive
electronic and nuclear temperatures must agree for equilibrium Gibbs.

[Source](../python/dft_rs/native.py#L2230).

### `dft_rs.native.gibbs_free_energy`

```python
gibbs_free_energy(numbers, positions, **controls) -> dict[str, Any]
```

Ideal-gas RRHO Gibbs free energy G=H-T*S in Hartree per molecule.

Uses the analytic molecular Hessian and the same thermochemistry as
frequencies_report. Controls: temperature (Kelvin, 298.15), pressure
(Pa, 101325), symmetry_number (1), plus reference controls. Includes
zero-point, vibrational, rotational, translational, electronic-degeneracy
and pV contributions. A minimum with real frequencies is required.
Finite electronic temperature uses the occupation/metric Mermin Hessian;
nuclear temperature defaults to that electronic temperature and must
agree with it. The electronic entropy is included exactly once.
Results include gibbs_free_energy and the full thermochemistry report.
ROHF/ROKS and complex GHF/GKS reuse their analytic Hessian provider.
electronic_degeneracy defaults to multiplicity for collinear states and
one for a fixed spinor. Free-molecule Eckart modes, positive internal
frequencies and a converged reference are required; fixed laboratory
fields or additional AO operators do not define this ideal-gas model.
Correlated-reference and acceptance limits remain those in response.md.

[Source](../python/dft_rs/native.py#L2251).

### `dft_rs.native.free_energy`

```python
free_energy(numbers, positions, **controls) -> dict[str, Any]
```

Compatibility wrapper for mermin_free_energy, the electronic A=E-T_e*S_e.

Prefer mermin_free_energy or gibbs_free_energy to state the thermodynamic
potential explicitly. This wrapper retains the free-energy feature ID.

[Source](../python/dft_rs/native.py#L2273).

### `dft_rs.native.gradient`

```python
gradient(numbers, positions, **controls) -> dict[str, Any]
```

Analytic gradient (Hartree/Bohr); reference selects RKS/UKS/ROKS/GKS constraints within the documented model scope.

[Source](../python/dft_rs/native.py#L2282).

### `dft_rs.native.charges`

```python
charges(numbers, positions, **controls) -> dict[str, Any]
```

Mulliken/Loewdin charges; reference selects auto/rks/uks/roks/gks and HF aliases.

[Source](../python/dft_rs/native.py#L2287).

### `dft_rs.native.natural_bond`

```python
natural_bond(numbers, positions, **controls) -> dict[str, Any]
```

Natural bond orbital analysis of the converged reference.

Natural atomic orbitals and natural population analysis (charges, spin
populations, electron configurations), Wiberg bond indices in the NAO
basis, the natural Lewis structure with its hybrids, antibonds and
Rydberg orbitals, natural localized molecular orbitals and second-order
donor-acceptor energies (Hartree). ``nbo_spin`` is ``"auto"`` (one
Lewis structure per spin for open shells), ``"total"`` or
``"separate"``; ``nbo_lewis_threshold`` fixes the occupancy threshold as
a fraction of the maximum occupancy, ``nbo_three_center`` adds
three-centre bonds and ``nbo_perturbation_threshold`` (Hartree) is the
smallest reported stabilization.

Set ``nbo_natural_orbitals=True`` for spatial occupations and effectively
unpaired weights. ``nbo_spin_density=True`` gives signed spin-density
orbitals, and ``nbo_spin_density_points`` samples them at Bohr coordinates.
``nbo_corresponding_orbitals=True`` pairs the occupied alpha and beta
frames of an integer-occupied collinear SCF reference.

[Source](../python/dft_rs/native.py#L2292).

### `dft_rs.native.igmh`

```python
igmh(numbers, positions, **controls) -> dict[str, Any]
```

Sample molecular IGMH, using the spin-summed SCF density.

``igmh_points`` are Cartesian triples in Bohr, or supply ``igmh_grid``
as origin and three step vectors in Bohr with ``igmh_grid_shape``.
``fragments`` gives one integer label per atom. The default
``igmh_gradient="published"`` follows the authors' erratum.
``"hirshfeld"`` uses the spatial gradient of the Hirshfeld density.
Results include total, inter- and intrafragment delta-g, signed density
and raw partition residuals. ``igmh_atom_details=True`` adds atomic fields.

[Source](../python/dft_rs/native.py#L2315).

### `dft_rs.native.periodic_natural_bond`

```python
periodic_natural_bond(numbers, positions, cell, **controls) -> dict[str, Any]
```

Natural bond orbital analysis of a periodic reference.

The Bloch overlap, spin densities and Fock matrices of the k mesh are
unfolded onto its Born-von Karman supercell; natural charges,
configurations, Wiberg bonds (with cell offsets of the partners), the
natural Lewis structure per cell and second-order donor-acceptor
energies (Hartree) are reported for the atoms of the home cell. Open
axes (``pbc``) use the dimensional SCF; use at least three k points
along every periodic axis so that an atom and its image do not share
supercell orbitals.

[Source](../python/dft_rs/native.py#L2329).

### `dft_rs.native.natural_shielding`

```python
natural_shielding(numbers, positions, **controls) -> dict[str, Any]
```

Natural chemical shielding analysis: the NMR shielding of every
nucleus (ppm) as a sum over the natural localized molecular orbitals of
the natural Lewis structure, with diamagnetic and paramagnetic parts.
``gauge`` (``"giao"``), ``gauge_origin``, ``nuclei`` and the
``natural_bond`` controls apply.

[Source](../python/dft_rs/native.py#L2344).

### `dft_rs.native.natural_coupling`

```python
natural_coupling(numbers, positions, **controls) -> dict[str, Any]
```

Natural J-coupling analysis of a closed-shell molecule: reduced
indirect spin-spin couplings (atomic units; Hz with ``nuclear_g``) of
the ``nuclei`` pairs as sums over natural localized molecular orbitals,
per Fermi-contact, spin-dipole, paramagnetic and diamagnetic term.

[Source](../python/dft_rs/native.py#L2353).

### `dft_rs.native.natural_energy_decomposition`

```python
natural_energy_decomposition(numbers, positions, *, fragments, **controls) -> dict[str, Any]
```

Natural energy decomposition of the interaction of molecular
fragments (``fragments``: the fragment index 0, 1, ... of every atom):
charge transfer from the deletion of inter-fragment NBO Fock couplings,
electrostatics of the deformed fragments, exchange-correlation remainder
and deformation against the isolated fragments in the complex basis
(Hartree; their sum is the interaction energy).

[Source](../python/dft_rs/native.py#L2361).

### `dft_rs.native.sapt_analysis`

```python
sapt_analysis(numbers, positions, *, fragments, **controls) -> dict[str, Any]
```

Intermolecular perturbation analysis of fragments 0 and 1.

`sapt_variant="response"` uses HF/KS monomer response with Heitler-London
exchange. `"sapt0"`, `"sapt0-s2"` and `"ssapt0"` select field-free
all-electron Hartree-Fock monomers and full, single-pair and scaled
single-pair exchange, respectively. `"sapt-30"` uses uncoupled HF
monomers and returns the third-order energy contributions.
`"sapt2+3"` uses closed-shell RHF monomers, the full antisymmetrizer,
intramonomer correlation and delta-HF(3). It includes the correlated
and relaxed-induction reports by default.
`"sapt-ks-fdds"` uses coupled adiabatic KS density response and
uncoupled KS exchange-dispersion. Directional
induction and other energies use Hartree. sSAPT0 also returns `delta_hf`
and the dimensionless `exchange_scale`. The common driver accepts
`fragment_charges` and `fragment_multiplicities`. The coupled selectors
accept `charge_transfer` and `sapt_partition`. With `"sapt0"` or
`"sapt-ks-fdds"`,
`sapt_regularization_eta` enables Gaussian-regularized second-order
induction (eta in inverse Bohr squared). For `"sapt0"`,
`sapt_correlation_terms=True` adds mixed-order fixed-orbital RHF
coefficients and relaxed MP2/MP3 electrostatics;
`sapt_relaxed_induction30=True` adds coupled cubic HF induction and
delta-HF(2). These optional terms are reported in Hartree.
The response and KS-FDDS
modes accept
`asymptotic_shift_a`/`_b` (Hartree), `rpa_dispersion` and
`casimir_polder_points`.

[Source](../python/dft_rs/native.py#L2372).

### `dft_rs.native.dipole`

```python
dipole(numbers, positions, **controls) -> dict[str, Any]
```

Molecular dipole (e*Bohr and Debye) about ``origin`` (default
``"center_of_mass"``; also ``"center_of_nuclear_charge"``, ``"zero"`` or
Bohr coordinates).

[Source](../python/dft_rs/native.py#L2404).

### `dft_rs.native.orbitals`

```python
orbitals(numbers, positions, **controls) -> dict[str, Any]
```

Orbital energies and occupations for reference=auto/rks/uks/roks/gks (HF aliases accepted).
``coefficients=True`` and ``density=True`` add AO matrices and overlap.
ROKS returns spin semicanonical occupied/virtual blocks of its common
spatial determinant, rather than unconstrained spin Fock eigenvectors.

[Source](../python/dft_rs/native.py#L2411).

### `dft_rs.native.write_molden`

```python
write_molden(numbers, positions, *, path, **controls) -> dict[str, Any]
```

Write a Molden file of the converged reference to ``path``.

[Source](../python/dft_rs/native.py#L2419).

### `dft_rs.native.save_orbitals`

```python
save_orbitals(numbers, positions, *, path, **controls) -> dict[str, Any]
```

Write the orbital report (coefficients, densities, occupations) as
JSON to ``path``; refused when occupations do not reproduce the density.

[Source](../python/dft_rs/native.py#L2424).

### `dft_rs.native.molecular_multipole`

```python
molecular_multipole(numbers, positions, **controls) -> dict[str, Any]
```

Raw Cartesian, Buckingham traceless and Racah spherical multipoles up
to ``max_rank`` (default 4, at most 8) about ``origin``.

[Source](../python/dft_rs/native.py#L2430).

### `dft_rs.native.molecular_medium_optics`

```python
molecular_medium_optics(numbers, positions, **controls) -> dict[str, Any]
```

Molecular dielectric, normal indices and anisotropic slab scattering.

[Source](../python/dft_rs/native.py#L2436).

### `dft_rs.native.quadrupole_polarizability`

```python
quadrupole_polarizability(numbers, positions, **controls) -> dict[str, Any]
```

Analytic electric quadrupole response in five traceless field-gradient modes.

The origin is in Bohr and the retarded frequency and damping are in Hartree.
Supported collinear double hybrids include the complex parent, target
and spin-resolved PT2 quasi-energy terms.

[Source](../python/dft_rs/native.py#L2441).

### `dft_rs.native.multipole_polarizability`

```python
multipole_polarizability(numbers, positions, **controls) -> dict[str, Any]
```

Racah electric multipole response for rank 1 to 5 (default 3).

Rows are real spherical moments and columns their conjugate fields, in
mode order 0, +1, -1, and so on. Origin is in Bohr, frequency and damping
are in Hartree, and the response is in e² Bohr^(2 rank)/Hartree.
Supported collinear double hybrids return complex parent, target
and spin-resolved PT2 quasi-energy terms.
The default center of mass moves with the molecule, making the result
invariant to a rigid coordinate translation. An explicit different
expansion center physically mixes lower multipole ranks.

[Source](../python/dft_rs/native.py#L2451).

### `dft_rs.native.conceptual_dft`

```python
conceptual_dft(numbers, positions, **controls) -> dict[str, Any]
```

Conceptual DFT descriptors. ``route`` is ``"grand_canonical"``
(analytic, needs ``electronic_temperature`` > 0 K and is then the
default), ``"integer"`` (N, N-1, N+1 total energies; default at zero
temperature) or ``"koopmans"``.

[Source](../python/dft_rs/native.py#L2466).

### `dft_rs.native.spin_conceptual_dft`

```python
spin_conceptual_dft(numbers, positions, **controls) -> dict[str, Any]
```

Spin-channel conceptual DFT with M = N_alpha - N_beta.

[Source](../python/dft_rs/native.py#L2474).

### `dft_rs.native.hessian_report`

```python
hessian_report(numbers, positions, **controls) -> dict[str, Any]
```

Analytic Cartesian Hessian in Hartree/Bohr^2.

With positive electronic_temperature, supported ROKS and GKS providers
differentiate the stationary Mermin free energy and return thermal
response and number diagnostics. Other report fields, including the
atomic polar tensor and raw ordered asymmetry, depend on the provider.
The complex London route accepts fixed laboratory electric and magnetic
fields with integer occupations. See the API guide for provider scope.

[Source](../python/dft_rs/native.py#L2479).

### `dft_rs.native.frequencies_report`

```python
frequencies_report(numbers, positions, **controls) -> dict[str, Any]
```

Harmonic frequencies (cm^-1), IR intensities (km/mol), Cartesian
normal modes and RRHO thermochemistry (``temperature`` K, ``pressure``
Pa, ``symmetry_number``). ROHF/ROKS and complex GHF/GKS use the same
analytic Hessian provider as hessian_report, retaining its diagnostics.
vibrational_projection is auto, eckart or cartesian: auto retains all
Cartesian modes under a fixed laboratory field or extra AO operator.
Missing atomic polar tensors give null IR intensities. Fixed-spinor
electronic_degeneracy defaults to one. RRHO is withheld when its free
molecule or positive-mode assumptions fail. See response.md for the
correlated-reference and bounded acceptance scope.

[Source](../python/dft_rs/native.py#L2492).

### `dft_rs.native.vibrational_spectra`

```python
vibrational_spectra(numbers, positions, **controls) -> dict[str, Any]
```

Frequencies with analytic IR intensities and analytic Raman
activities (Angstrom^4/amu) and depolarization ratios.

[Source](../python/dft_rs/native.py#L2506).

### `dft_rs.native.polarizability_report`

```python
polarizability_report(numbers, positions, **controls) -> dict[str, Any]
```

Static dipole polarizability (Bohr^3 and Angstrom^3) from CPKS.

[Source](../python/dft_rs/native.py#L2512).

### `dft_rs.native.multipole_gradient`

```python
multipole_gradient(numbers, positions, **controls) -> dict[str, Any]
```

Analytic nuclear derivatives of Cartesian multipoles up to
``max_rank`` (default 2) about ``origin``.

[Source](../python/dft_rs/native.py#L2517).

### `dft_rs.native.response_derivatives`

```python
response_derivatives(numbers, positions, **controls) -> dict[str, Any]
```

First-order response export: orbital rotations U^F and densities
dP/dF for each field axis and nuclear density derivatives dP/dR;
``path`` also writes the report as JSON.

[Source](../python/dft_rs/native.py#L2523).

### `dft_rs.native.third_derivative_2n1`

```python
third_derivative_2n1(numbers, positions, **controls) -> dict[str, Any]
```

Cartesian third derivatives of the energy (Hartree/Bohr^3) from
first-order responses only, by the 2n+1 rule.

[Source](../python/dft_rs/native.py#L2532).

### `dft_rs.native.cubic_fc_along`

```python
cubic_fc_along(numbers, positions, **controls) -> dict[str, Any]
```

Cubic force constants contracted with the displacement pattern
``probe`` (one value per Cartesian coordinate): the derivative of the
Hessian along it. Optional ``bfield`` and ``field`` use the shared
complex London stationary nuclear response about a biased reference;
the matrix has units Hartree/Bohr^3 for dimensionless probe entries.
Integer RKS/UKS correlated requests include full orbital/PT2 response,
reference_field electric bias and separately reported parent/target/SS/OS
and dispersion matrices. Magnetic PT2 and additional Hamiltonian providers
remain separate.

[Source](../python/dft_rs/native.py#L2538).

### `dft_rs.native.quartic_fc_along`

```python
quartic_fc_along(numbers, positions, **controls) -> dict[str, Any]
```

Quartic force constants contracted with the displacement patterns
``probe`` and ``second_probe`` (default the same pattern). Optional
``bfield`` and ``field`` select complex London nuclear response with
fixed laboratory fields; Hartree/Bohr^4 for dimensionless probe entries.
Integer RKS/UKS correlated requests include full orbital/PT2 response,
reference_field electric bias and separately reported parent/target/SS/OS
and dispersion matrices. Magnetic PT2 and additional Hamiltonian providers
remain separate.

[Source](../python/dft_rs/native.py#L2551).

### `dft_rs.native.streamed_gradient`

```python
streamed_gradient(numbers, positions, **controls) -> dict[str, Any]
```

Analytic gradient with integral-direct Coulomb and exchange builds:
no integral tensor is stored.

[Source](../python/dft_rs/native.py#L2563).

### `dft_rs.native.excitations`

```python
excitations(numbers, positions, **controls) -> dict[str, Any]
```

Vertical excitation energies and transition strengths.
Collinear references use their TDA/RPA models. ``reference="roks"`` or
``"rohf"`` selects reference-spin common-orbital flag TDA; its HF doublet
and quartet routes have full derivative/public acceptance.
``reference="gks"``/``"ghf"`` selects full complex spinor TDA with transverse channels
and no exact total-spin label. Full-spin RPA and correlated flag/spinor
propagators remain unconnected. ``amplitudes=True`` includes amplitudes.
``td_nto=True`` returns particle–hole pairs for supported TDA states.
``td_difference_density=True`` returns unrelaxed attachment and detachment
orbitals for RKS, UKS and GKS TDA.

[Source](../python/dft_rs/native.py#L2569).

### `dft_rs.native.ecd_spectrum`

```python
ecd_spectrum(numbers, positions, **controls) -> dict[str, Any]
```

Gaussian-broadened absorption and circular-dichroism spectra
(L mol^-1 cm^-1) of the TDA states; ``broadening_ev`` (standard
deviation in eV) is required.

[Source](../python/dft_rs/native.py#L2583).

### `dft_rs.native.td_state_gradient`

```python
td_state_gradient(numbers, positions, **controls) -> dict[str, Any]
```

Analytic total excited-state gradient (Hartree/Bohr).
``root`` is one based. The collinear Z-vector route covers restricted
singlet HF/pure LDA; ``td_gradient_route="path"`` selects compatible
analytic nuclear paths. ROKS/ROHF uses the common-orbital reference-spin
flag model; GKS/GHF uses full-spin TDA with a fixed additional AO source.
HF flag doublet/quartet and isolated spinor subsets have complete Cartesian
acceptance. Moving semilocal spinor kernels are implemented with molecular
verification pending; correlated pole nuclear derivatives are unconnected.

[Source](../python/dft_rs/native.py#L2590).

### `dft_rs.native.hyperfine`

```python
hyperfine(numbers, positions, **controls) -> dict[str, Any]
```

Contact, spin-dipolar and total hyperfine tensors of an open shell.

``multiplicity`` above 1 is required. Couplings are reported in MHz per
nuclear g; an optional signed ``nuclear_g`` list, one value per atom,
also supplies isotope-scaled MHz values and contact couplings in gauss.
There is no built-in isotope table. Auto/UKS/UHF retain the collinear route.
ROKS/ROHF select the dedicated common-orbital static route. GKS/GHF require
``epr_projection="single-kramers-pair"`` and an odd-electron doublet seed;
this explicit mean-field approximation does not spin-purify the state or
guarantee an isolated doublet. Dedicated routes use a common-origin model
and retain their functional/operator restrictions in the report.
Optional ``nuclear_order=1/2`` selects a separate HF-collinear nuclear-A
adapter, with its own convergence and support restrictions. Static tensors
alone do not establish nuclear derivatives, vibrational averaging or spectra.

[Source](../python/dft_rs/native.py#L2602).

### `dft_rs.native.nmr_shielding`

```python
nmr_shielding(numbers, positions, **controls) -> dict[str, Any]
```

Nuclear magnetic shielding tensors in ppm from the coupled-perturbed
magnetic response; ``gauge`` is "giao" (default) or "common-origin" with
``gauge_origin``.

[Source](../python/dft_rs/native.py#L2621).

### `dft_rs.native.nmr_spectrum`

```python
nmr_spectrum(numbers, positions, **controls) -> dict[str, Any]
```

Isotropic NMR spectrum: one integrated area per observed nucleus at
its chemical shift. ``nmr_reference_elements`` and
``nmr_reference_shieldings`` (ppm) give the reference of each plotted
element. With ``nmr_couplings=True`` and ``nuclear_g`` (one g factor per
atom) lines use first-order multiplets by default. Set
``nmr_spin_model="exact"`` to diagonalize the full isotropic scalar-coupling
spin Hamiltonian, including flip-flops and coherent isotope-group detection.
``nuclear_spins`` contains twice I; ``nmr_spin_max_dimension`` (default 1024)
bounds the untruncated Hilbert dimension. All active partner shieldings
enter the Hamiltonian; reference shieldings affect only the plotted axis.
Line positions are ppm, weights
are integrated areas, and the spin eigensystem residual is Hz. The common
high-temperature population factor is removed; quadrupolar, anisotropic
and relaxation terms are excluded.

[Source](../python/dft_rs/native.py#L2628).

### `dft_rs.native.spin_spin_couplings`

```python
spin_spin_couplings(numbers, positions, **controls) -> dict[str, Any]
```

Indirect nuclear spin-spin coupling tensors. UKS/GKS use full complex
Pauli response; ``spin_response_constraint`` selects free (default) or fixed-spin-direction
response. ROKS and correlated open-shell providers
remain separate. Complete mixed spin/orbital terms are retained:
reduced coupling tensors with their diamagnetic and paramagnetic
spin-orbit, Fermi-contact and spin-dipole parts between the atoms
``nuclei`` (default all), and with ``nuclear_g`` the constants in Hz.

[Source](../python/dft_rs/native.py#L2646).

### `dft_rs.native.fermi_contact_couplings`

```python
fermi_contact_couplings(numbers, positions, **controls) -> dict[str, Any]
```

Fermi-contact part of the indirect nuclear spin-spin couplings.

[Source](../python/dft_rs/native.py#L2657).

### `dft_rs.native.exafs`

```python
exafs(numbers, positions, **controls) -> dict[str, Any]
```

Single-scattering EXAFS of the atom ``exafs_absorber`` from the
tabulated backscattering amplitudes and phases of ``exafs_scattering``
(rows of atomic number, k, amplitude, phase); no electronic-structure
calculation is involved.

[Source](../python/dft_rs/native.py#L2662).

### `dft_rs.native.xanes`

```python
xanes(numbers, positions, **controls) -> dict[str, Any]
```

Core-level absorption from Tamm-Dancoff response in the
core-valence-separated space of the K shell of ``xanes_element`` (or of
``core_orbitals``); the core hole is not relaxed.

[Source](../python/dft_rs/native.py#L2670).

### `dft_rs.native.nexafs`

```python
nexafs(numbers, positions, **controls) -> dict[str, Any]
```

Polarized molecular near-edge absorption in a finite orbital basis.

[Source](../python/dft_rs/native.py#L2677).

### `dft_rs.native.constrained_dft`

```python
constrained_dft(numbers, positions, **controls) -> dict[str, Any]
```

Variational AO population or spin constraints with optional analytic forces,
geometry optimization and static electric response. Supply fragment atoms
and electrons in the entries of density_constraints.

[Source](../python/dft_rs/native.py#L2682).

### `dft_rs.native.dft_plus_u`

```python
dft_plus_u(numbers, positions, **controls) -> dict[str, Any]
```

Fixed-geometry molecular DFT+U. Supply Hubbard site parameters in Hartree.

[Source](../python/dft_rs/native.py#L2689).

### `dft_rs.native.pz_sic`

```python
pz_sic(numbers, positions, **controls) -> dict[str, Any]
```

Perdew-Zunger SIC on real integer unrestricted orbitals at fixed geometry.

[Source](../python/dft_rs/native.py#L2694).

### `dft_rs.native.flo_sic_fixed`

```python
flo_sic_fixed(numbers, positions, **controls) -> dict[str, Any]
```

Fermi-Lowdin SIC on frozen UKS orbitals with fixed descriptors in Bohr.

[Source](../python/dft_rs/native.py#L2699).

### `dft_rs.native.car_parrinello_md`

```python
car_parrinello_md(numbers, positions, **controls) -> dict[str, Any]
```

Microcanonical dynamics of nuclei and real or complex occupied orbitals.

[Source](../python/dft_rs/native.py#L2704).

### `dft_rs.native.magnetic_quadrupole`

```python
magnetic_quadrupole(numbers, positions, **controls) -> dict[str, Any]
```

Orbital and Pauli magnetic quadrupole at the selected field origin.

[Source](../python/dft_rs/native.py#L2709).

### `dft_rs.native.magnetizability`

```python
magnetizability(numbers, positions, **controls) -> dict[str, Any]
```

Magnetizability tensor from the coupled-perturbed magnetic response
with a common gauge origin (``gauge_origin``, default the centre of
nuclear charge); the result depends on the origin in a finite basis.
gauge="london" uses London orbitals. magnetic_directions optionally
contracts a mixed energy derivative of order 1..4; matching optional
electric_directions add electric parts to the same dimensionless
parameters. The magnetic_mixed fields report this joint derivative.
Spin Zeeman and current-dependent correlation are excluded on this path.

[Source](../python/dft_rs/native.py#L2714).

### `dft_rs.native.franck_condon`

```python
franck_condon(numbers, positions, **controls) -> dict[str, Any]
```

Franck-Condon progression of the singlet TDA state ``root`` in the
vertical-gradient model (displaced ground-state harmonic modes, zero
temperature): Huang-Rhys factors, 0-0 energy and line factors.

[Source](../python/dft_rs/native.py#L2727).

### `dft_rs.native.excited_optimization`

```python
excited_optimization(numbers, positions, **controls) -> dict[str, Any]
```

Relax an excited-state geometry by L-BFGS on its analytic gradient.
The default is the restricted singlet TDA Z-vector model. Explicit
``td_gradient_route="path"`` selects compatible nuclear-path models,
including the tested HF triplet route. Reports identify the actual
method, spin channel and gradient route. General spinor optimization
is not connected; state following retains the squared-amplitude
heuristic and does not certify crossings or degenerate surfaces.

[Source](../python/dft_rs/native.py#L2734).

### `dft_rs.native.optical_activity`

```python
optical_activity(numbers, positions, **controls) -> dict[str, Any]
```

Optical rotation at the photon energies ``frequencies``: the electric
dipole - magnetic dipole polarizability of the response in the length
gauge about ``gauge_origin``, ``beta`` and the specific rotation.

[Source](../python/dft_rs/native.py#L2745).

### `dft_rs.native.broken_symmetry_exchange`

```python
broken_symmetry_exchange(numbers, positions, **controls) -> dict[str, Any]
```

Heisenberg exchange coupling J (H = -2 J S_A . S_B) of the sites
``site_a`` and ``site_b`` from the high-spin determinant (the
``multiplicity`` of the controls) and the broken-symmetry determinant.

[Source](../python/dft_rs/native.py#L2752).

### `dft_rs.native.vcd`

```python
vcd(numbers, positions, **controls) -> dict[str, Any]
```

Vibrational circular dichroism: atomic axial tensors with a common
gauge origin and the rotational strengths of the harmonic modes.

[Source](../python/dft_rs/native.py#L2759).

### `dft_rs.native.spin_susceptibility`

```python
spin_susceptibility(numbers, positions, **controls) -> dict[str, Any]
```

Static spin susceptibility chi = g_e^2 mu_B^2 / eta of an unrestricted
reference with a positive ``electronic_temperature``, from the ensemble
response to a spin transfer at fixed electron number.

[Source](../python/dft_rs/native.py#L2765).

### `dft_rs.native.spin_spin_zfs`

```python
spin_spin_zfs(numbers, positions, *, nuclear_order=0, **controls) -> dict[str, Any]
```

Direct electron spin-spin D in H_D=S.D.S; SOC and spin projection excluded.

Integer all-electron UHF/HF only, multiplicity >= 3. Omitted reference
selects the dedicated UHF route; explicit reference must be ``"uhf"``.
Strict integer ``nuclear_order=0/1/2`` selects D [3,3] in Hartree,
dD/dR [3N,3,3] in Hartree/Bohr and d2D/dR2 [3N,3N,3,3] in
Hartree/Bohr^2. Unrequested derivatives are None. Coordinates are
atom-major xyz. Raw SCF/response diagnostics and managed workspace
accounting are retained; nonconvergence raises an error. This is not
total ZFS, a vibrational correction or experimental validation.

[Source](../python/dft_rs/native.py#L2772).

### `dft_rs.native.g_tensor`

```python
g_tensor(numbers, positions, **controls) -> dict[str, Any]
```

Electronic g tensor of an unrestricted open-shell reference:
relativistic mass correction, gauge correction and the orbital Zeeman /
one-electron spin-orbit cross term, with a common ``gauge_origin`` and
optional effective ``soc_charges``. The tensor is dimensionless; shift
contributions are in ppm. Auto/UKS/UHF retain this collinear route.
ROKS/ROHF use the dedicated common-origin common-orbital route. GKS/GHF
require ``epr_projection="single-kramers-pair"`` and an odd-electron doublet
seed; the returned effective-spin model is an explicit mean-field
approximation, not spin purification or an isolated-doublet guarantee.
Dedicated London-orbital and other unavailable combinations are errors.
``nuclear_order=1/2`` selects the separate all-electron integer HF/UHF
one-electron Pauli London model. It returns actual Cartesian
``g_gradient[3N,3,3]`` and ``g_hessian[3N,3N,3,3]`` in inverse Bohr and
inverse Bohr squared; the Hessian is null for order one. Mass, gauge
and spin-orbit parts and raw response diagnostics remain separate.
This nuclear route rejects DFT, RO/spinor references, nonzero electric
fields, stability-followed references, ECPs and ensemble occupations.
Explicit zero electric field is allowed. Order zero preserves static
routing. Neither route is a complete many-electron relativistic model.

[Source](../python/dft_rs/native.py#L2787).

### `dft_rs.native.roks`

```python
roks(numbers, positions, **controls) -> dict[str, Any]
```

Restricted open-shell Kohn-Sham energy of the state of
``multiplicity`` with one set of spatial orbitals, and its analytic
gradient (``gradient=False`` skips it).

[Source](../python/dft_rs/native.py#L2810).

### `dft_rs.native.roks_hessian`

```python
roks_hessian(numbers, positions, **controls) -> dict[str, Any]
```

Analytic Hessian of the restricted open-shell Kohn-Sham energy of the
state of ``multiplicity``, from the second-order Taylor coefficients of
the energy along nuclear paths, with its gradient and the harmonic
frequencies.

[Source](../python/dft_rs/native.py#L2817).

### `dft_rs.native.roks_polarizability`

```python
roks_polarizability(numbers, positions, **controls) -> dict[str, Any]
```

Dipole moment and static polarizability of the restricted open-shell
Kohn-Sham state of ``multiplicity``, from the Taylor series of the
energy along field paths.

[Source](../python/dft_rs/native.py#L2825).

### `dft_rs.native.divide_conquer`

```python
divide_conquer(numbers, positions, **controls) -> dict[str, Any]
```

Divide-and-conquer Kohn-Sham energy and gradient: subsystems of
fragments (``dc_fragments``, one fragment index per atom; default one
atom per fragment) with buffers of radius ``dc_buffer`` (Bohr), a common
Fermi level with smearing ``dc_smearing`` (Hartree).

[Source](../python/dft_rs/native.py#L2832).

### `dft_rs.native.mrsf`

```python
mrsf(numbers, positions, **controls) -> dict[str, Any]
```

Mixed-reference spin-flip TDA: ``nstates`` singlets and triplets on
the restricted open-shell triplet of the molecule (the input names the
singlet or the triplet); energies ``omega`` are relative to that
triplet.

[Source](../python/dft_rs/native.py#L2840).

### `dft_rs.native.mrsf_state`

```python
mrsf_state(numbers, positions, **controls) -> dict[str, Any]
```

Analytic gradient of the total energy of one mixed-reference
spin-flip state (``mrsf_state="S0"``, ``"S1"``, ``"T0"``, ...) on the
restricted open-shell triplet, from Taylor series along nuclear paths.

[Source](../python/dft_rs/native.py#L2848).

### `dft_rs.native.mrsf_hessian`

```python
mrsf_hessian(numbers, positions, **controls) -> dict[str, Any]
```

Analytic Hessian, gradient and harmonic frequencies of the total
energy of one mixed-reference spin-flip state (``mrsf_state``).

[Source](../python/dft_rs/native.py#L2855).

### `dft_rs.native.spin_orbit`

```python
spin_orbit(numbers, positions, **controls) -> dict[str, Any]
```

One-shot spin-orbit coupling: the one-electron spin-orbit operator
(nuclear or effective ``soc_charges``) diagonalized in the Kohn-Sham
spin orbitals for the spin axis ``spin_axes`` (default z).

Select reference="gks" for self-consistency with the full spin-covariant
functional. Read the paired self_consistent_energies and
self_consistent_orbital_residuals, both in Hartree.

[Source](../python/dft_rs/native.py#L2861).

### `dft_rs.native.spinor_polarizability`

```python
spinor_polarizability(numbers, positions, *, frequencies, **controls) -> dict[str, Any]
```

Dynamic polarizability of the self-consistent two-component state
with the one-electron spin-orbit operator (``soc_charges``, ``soc_scale``)
in its Fock matrix, at each of ``frequencies`` (Hartree, optional
``frequency_damping``): ``polarizability_real`` / ``polarizability_imag``
as ``[frequency][a][b]`` and the axial vector of the antisymmetric
imaginary part. ``spin_axis`` is the spin axis of the collinear start.

[Source](../python/dft_rs/native.py#L2873).

### `dft_rs.native.magnetic_anisotropy`

```python
magnetic_anisotropy(numbers, positions, **controls) -> dict[str, Any]
```

Magnetic anisotropy energies from the band energies of the spin axes
``spin_axes`` (default x, y, z) with the one-shot spin-orbit coupling
(magnetic force theorem).

[Source](../python/dft_rs/native.py#L2883).

### `dft_rs.native.magnon`

```python
magnon(numbers, positions, **controls) -> dict[str, Any]
```

Transverse spin excitations: the lowest ``nstates`` spin-lowering and
spin-raising spin-flip TDA energies of an unrestricted reference
(collinear kernel, exact exchange only).

[Source](../python/dft_rs/native.py#L2890).

### `dft_rs.native.mcd`

```python
mcd(numbers, positions, **controls) -> dict[str, Any]
```

Magnetic circular dichroism of a closed-shell molecule: Faraday B
terms (and A terms of degenerate states) of the lowest ``nstates`` states
from the complete manifold of singlet Tamm-Dancoff states.

``mcd_model="thermal-manifold"`` selects a separate canonical ensemble
of supplied real symmetric sublevel Hamiltonian, magnetic moments and
normalized optical-helicity matrices. ``temperature`` is Kelvin and
``bfield`` is atomic magnetic flux density. It reports the population
C response separately from interlevel mixing and finite-field saturation.
It does not generate SOC/optical sublevels from the atomic geometry or
provide an absolute absorption spectrum. SCF controls do not apply.

``mcd_model="thermal-complex-manifold"`` retains complex Hermitian
operators: optional ``*_imag`` matrices specify imaginary quadratures.
``mcd_model="molecular-casci-soc"`` generates common DFT/HF orbitals,
a spin-complete active-space Coulomb/SOC Hamiltonian, magnetic/electric
operators, canonical many-electron populations and absolute broadened
cross sections. This is CASCI in DFT/HF orbitals, not TDDFT. Explicit
active-space, fixed-origin linear-Zeeman and one-electron-SOC assumptions
and convergence limits are returned. ``mcd_orientation="isotropic"``
performs a streamed angular average for light parallel to the field.

``mcd_magnetic_basis="london-linear-frozen-soc"`` selects first-order
London orbital, overlap, Coulomb and optical-operator completion. The
common gauge origin cancels; zero-field SOC is frozen in the transported
orthonormal active frame. Moving SOC and quadratic magnetic terms are
absent. The default ``common-origin`` model retains its prior semantics.
Per-state spin expectation and spin squared are returned; SOC eigenstates
need not have a pure total spin or integer multiplicity.

``faraday_a_terms`` is the canonical line-shape-derivative coefficient;
``a_terms`` preserves the legacy opposite first-line-moment sign.
``b_terms`` includes ground-state permanent-dipole subtraction. Sum full
degenerate levels and apply orientation/helicity factors separately, as
described by the returned ``mcd_convention`` and unit labels.

[Source](../python/dft_rs/native.py#L2897).

### `dft_rs.native.td_hessian`

```python
td_hessian(numbers, positions, **controls) -> dict[str, Any]
```

Analytic Hessian (Hartree/Bohr squared) of one isolated state.
The total surface is SCF energy plus the selected excitation pole. Use
``reference="roks"``/``"rohf"`` for reference-spin common-orbital flag TDA,
or ``"gks"``/``"ghf"`` for full complex spinor TDA. HF doublet/quartet flag
and isolated full-spin HF Cartesian subsets are accepted. These reports
currently contain derivative tensors without the collinear harmonic
postprocessing. Semilocal spinor molecular verification and correlated pole
derivatives remain pending; degenerate single-root derivatives are refused.

[Source](../python/dft_rs/native.py#L2937).

### `dft_rs.native.td_ecd_giao`

```python
td_ecd_giao(numbers, positions, **controls) -> dict[str, Any]
```

Length-gauge rotatory strengths with London orbitals of the TDA or
RPA states (``td_method``, ``nstates``): ``rotatory_strength_london``
per state next to the common-origin (``gauge_origin``) and velocity
values, in atomic units and in 1e-40 erg cm^3.

[Source](../python/dft_rs/native.py#L2949).

### `dft_rs.native.roa`

```python
roa(numbers, positions, **controls) -> dict[str, Any]
```

Raman optical activity: normal-mode derivatives of the electric dipole
polarizability, the optical rotation tensor and the dipole-quadrupole
polarizability at the photon energies ``frequencies`` (Hartree), their
invariants and the circular intensity differences; ``gauge`` is
``velocity`` (default, independent of the origin) or ``length``.

[Source](../python/dft_rs/native.py#L2957).

### `dft_rs.native.hypermagnetizability`

```python
hypermagnetizability(numbers, positions, **controls) -> dict[str, Any]
```

Static hypermagnetizability ``eta[a][b][c][d] = d2 alpha_ab/(dB_c dB_d)``
with the static polarizability and the magnetizability, common gauge
origin ``gauge_origin`` (Bohr; default the centre of nuclear charge).

[Source](../python/dft_rs/native.py#L2966).

### `dft_rs.native.cotton_mouton`

```python
cotton_mouton(numbers, positions, **controls) -> dict[str, Any]
```

Cotton-Mouton constant per molecule at ``temperature`` (K) from the
hypermagnetizability anisotropy and the orientational term of the
polarizability and magnetizability anisotropies.

[Source](../python/dft_rs/native.py#L2973).

### `dft_rs.native.magneto_optical`

```python
magneto_optical(numbers, positions, **controls) -> dict[str, Any]
```

Magneto-optical tensor at the photon energy ``frequency`` (Hartree).

An open-shell reference (``magneto_optical_source="spin-orbit"``, the
automatic choice) gives the first-order change of the polarizability
under one-electron spin-orbit coupling for the spins along each axis,
``magneto_optical_tensor[a][b][k]``. A closed-shell reference
(``"field"``) gives the Faraday tensor ``Im d alpha_ab(w)/dB_c`` of an
external field, with London orbitals (``gauge="london"``, default) or a
common gauge origin.

[Source](../python/dft_rs/native.py#L2980).

### `dft_rs.native.magnetic_field_scf`

```python
magnetic_field_scf(numbers, positions, *, bfield, **controls) -> dict[str, Any]
```

Self-consistent field in the finite uniform magnetic field ``bfield``
(atomic units, 1 a.u. = 2.35052e5 T) with London orbitals; the spin
Zeeman term is included unless ``spin_zeeman=False``. Optional field
(Hartree/e/Bohr) includes the scalar potential and nuclear coupling, with
the charged-system energy referenced to the original laboratory origin.

[Source](../python/dft_rs/native.py#L2993).

### `dft_rs.native.magnetic_field_gradient`

```python
magnetic_field_gradient(numbers, positions, *, bfield, **controls) -> dict[str, Any]
```

Self-consistent field in the finite uniform magnetic field ``bfield``
(atomic units) with London orbitals and the analytic gradient of its
energy (``gradient``, Hartree/Bohr).

[Source](../python/dft_rs/native.py#L3003).

### `dft_rs.native.magnetoelectric`

```python
magnetoelectric(numbers, positions, **controls) -> dict[str, Any]
```

Linear magnetoelectric response of an open-shell reference to first
order in the one-electron spin-orbit coupling (``soc_charges``):
``magnetoelectric_tensor[a][b][k]`` is the electric dipole component
``a`` induced per unit magnetic field ``b`` with the spins along ``k``.

[Source](../python/dft_rs/native.py#L3010).

### `dft_rs.native.periodic_energy`

```python
periodic_energy(numbers, positions, cell, **controls) -> dict[str, Any]
```

Finite-k periodic energy per cell (Hartree; Bohr geometry and lattice
rows), with analytic forces when ``forces=True``.

[Source](../python/dft_rs/native.py#L3018).

### `dft_rs.native.stress`

```python
stress(numbers, positions, cell, **controls) -> dict[str, Any]
```

Analytic stress of the finite-k periodic energy with forces.

The field-free spinor path uses the periodic length, area or volume as
its normalization. In 3D the stress is also given in GPa. The Voigt
order is (xx, yy, zz, yz, xz, xy). Commensurate scalar and vector
Fourier controls use fixed reduced wave indices and fixed Cartesian
amplitudes for the collinear strain derivative.

[Source](../python/dft_rs/native.py#L3024).

### `dft_rs.native.opt_fixed_cell`

```python
opt_fixed_cell(numbers, positions, cell, **controls) -> dict[str, Any]
```

Fixed-cell relaxation of the atoms with analytic finite-k forces
(``max_steps``, ``gtol`` in Hartree/Bohr, ``step_max`` in Bohr).

[Source](../python/dft_rs/native.py#L3035).

### `dft_rs.native.cell_opt`

```python
cell_opt(numbers, positions, cell, **controls) -> dict[str, Any]
```

Variable-cell relaxation with analytic forces and stress at the
hydrostatic ``pressure_gpa``; returns the relaxed ``cell`` rows and
``positions`` in Bohr.

[Source](../python/dft_rs/native.py#L3041).

### `dft_rs.native.phonons`

```python
phonons(numbers, positions, cell, **controls) -> dict[str, Any]
```

Gamma-point harmonic phonons from analytic coupled-perturbed force
constants (Hartree/Bohr^2; frequencies in cm^-1, negative = imaginary).
Optional field (Hartree/e/Bohr) selects the stationary finite-bias Berry
electric enthalpy on the Gamma supercell of the insulating k mesh.
This path reports its reference and response residuals explicitly.

[Source](../python/dft_rs/native.py#L3048).

### `dft_rs.native.phonon_bands`

```python
phonon_bands(numbers, positions, cell, *, supercell, path, **controls) -> dict[str, Any]
```

Phonon dispersion (cm^-1) along reduced q vertices ``path`` from
analytic force constants of a ``supercell``, optional DOS.
lattice_flexoelectric=True with wavevector_direction and response_order
2..4 adds fixed-mass-center internal relaxation at Gamma. Born charges
are computed on the primitive reference, or supplied as born_charge_rows
(3N by 3, atom/polarization then displacement). born_lattice_flexoelectric
is the Born-charge lattice contribution in e/Bohr, contracted twice with
the normalized Cartesian direction; it excludes clamped-ion and dynamic
multipole terms. Long-wave coefficient entries carry their order/units.

[Source](../python/dft_rs/native.py#L3057).

### `dft_rs.native.phonon_thermo`

```python
phonon_thermo(numbers, positions, cell, *, supercell, q_mesh, **controls) -> dict[str, Any]
```

Harmonic lattice free energy, entropy and heat capacity per primitive
cell from analytic supercell force constants on a uniform ``q_mesh``.

[Source](../python/dft_rs/native.py#L3071).

### `dft_rs.native.electron_phonon`

```python
electron_phonon(numbers, positions, cell, **controls) -> dict[str, Any]
```

Electron-phonon coupling lambda, the Eliashberg function,
McMillan/Allen-Dynes critical temperatures and the critical temperature of
the isotropic imaginary-axis Eliashberg equations (``tc_eliashberg_k``;
controls ``eliashberg_cutoff_cm1``, ``eliashberg_t_min_k``) and, for ``gap_temperatures_k``, the nonlinear gap
``Delta(T)`` with a Pade continuation to the real axis (``eliashberg_gaps``,
``pade_points``) from analytic responses. ``scdft_pairing=True`` returns
``scdft_inputs`` and Gamma-supercell K/Z. Supply ``screened_pair_coulomb``
or select ``scdft_screening="rpa"`` for a thermal head and ``"rpa-neutral"``
for the fixed-number Gamma body. The generated models use the complete
k-point difference mesh and ``response_reciprocal_cutoff`` in inverse Bohr.
SCDFT uses positive temperature and spin-degenerate normal frames.

[Source](../python/dft_rs/native.py#L3077).

### `dft_rs.native.quasi_harmonic`

```python
quasi_harmonic(numbers, positions, cell, *, supercell, q_mesh, **controls) -> dict[str, Any]
```

Quasi-harmonic V(T), isothermal bulk modulus, thermal expansion,
heat capacities and Grueneisen parameter from analytic supercell phonons
on isotropically scaled cells (``scales``).

[Source](../python/dft_rs/native.py#L3093).

### `dft_rs.native.elastic`

```python
elastic(numbers, positions, cell, **controls) -> dict[str, Any]
```

Clamped- and relaxed-ion elastic constants (Voigt, GPa and
Hartree/Bohr^3) from analytic strain, internal-strain and force-constant
responses.

[Source](../python/dft_rs/native.py#L3100).

### `dft_rs.native.elastic_fd`

```python
elastic_fd(numbers, positions, cell, **controls) -> dict[str, Any]
```

Validation route: Lagrangian elastic constants (GPa) from central
differences of the analytic stress, clamped or relaxed ions.

[Source](../python/dft_rs/native.py#L3107).

### `dft_rs.native.elastic_moduli`

```python
elastic_moduli(numbers, positions, cell, **controls) -> dict[str, Any]
```

Voigt, Reuss and Hill moduli (GPa), isotropic wave speeds (m/s) and
the Debye temperature (K) from the analytic elastic tensor.

[Source](../python/dft_rs/native.py#L3113).

### `dft_rs.native.christoffel`

```python
christoffel(numbers, positions, cell, **controls) -> dict[str, Any]
```

Christoffel phase velocities (m/s) and polarizations along
``directions`` from the analytic elastic tensor.

[Source](../python/dft_rs/native.py#L3119).

### `dft_rs.native.material_card`

```python
material_card(numbers, positions, cell, **controls) -> dict[str, Any]
```

Continuum material card from the analytic elastic tensor: stiffness,
compliance, engineering constants, Hill moduli, density and wave speeds
(``card_path`` writes JSON or CSV).

[Source](../python/dft_rs/native.py#L3125).

### `dft_rs.native.equation_of_state`

```python
equation_of_state(numbers, positions, cell, **controls) -> dict[str, Any]
```

Third-order Birch-Murnaghan fit of isotropically scaled cells
(``scales``), with analytic pressures; V0 in Bohr^3, B0 in GPa.

[Source](../python/dft_rs/native.py#L3132).

### `dft_rs.native.dielectric_tensor`

```python
dielectric_tensor(numbers, positions, cell, **controls) -> dict[str, Any]
```

Electronic (clamped-ion) dielectric tensor ``epsilon_infinity`` by
periodic density-functional perturbation theory.

[Source](../python/dft_rs/native.py#L3138).

### `dft_rs.native.born_charges`

```python
born_charges(numbers, positions, cell, **controls) -> dict[str, Any]
```

Born effective charges (elementary charges) and the electronic
dielectric tensor by periodic density-functional perturbation theory.

[Source](../python/dft_rs/native.py#L3144).

### `dft_rs.native.piezoelectric_tensor`

```python
piezoelectric_tensor(numbers, positions, cell, **controls) -> dict[str, Any]
```

Proper piezoelectric tensor of an insulating crystal: clamped-ion
(strain response of the polarization) and relaxed-ion (Born charges with
the internal relaxation), in e/bohr^2 and C/m^2.

[Source](../python/dft_rs/native.py#L3150).

### `dft_rs.native.berry_polarization`

```python
berry_polarization(numbers, positions, cell, **controls) -> dict[str, Any]
```

Berry-phase polarization of an insulating cell (e/Bohr^2 and C/m^2),
defined modulo the polarization quanta; ``berry_mesh`` defaults to the
SCF k mesh.

[Source](../python/dft_rs/native.py#L3157).

### `dft_rs.native.crystal_ir`

```python
crystal_ir(numbers, positions, cell, **controls) -> dict[str, Any]
```

Gamma transverse modes, mode effective charges, infrared intensities
(km/mol) and LO frequencies along ``directions``.

[Source](../python/dft_rs/native.py#L3164).

### `dft_rs.native.static_dielectric`

```python
static_dielectric(numbers, positions, cell, **controls) -> dict[str, Any]
```

Static (relaxed-ion) dielectric tensor ``epsilon_static`` with the
electronic tensor, Born charges and Gamma modes.

[Source](../python/dft_rs/native.py#L3170).

### `dft_rs.native.electronic_structure`

```python
electronic_structure(numbers, positions, cell, **controls) -> dict[str, Any]
```

SCF bands and populations; ``coefficients=True`` adds complex Bloch
orbitals and F/S/P matrices with raw residuals, in the SCF parent model.

[Source](../python/dft_rs/native.py#L3176).

### `dft_rs.native.effective_mass`

```python
effective_mass(numbers, positions, cell, *, k_point, band, **controls) -> dict[str, Any]
```

Analytic band velocity and inverse effective-mass tensor of ``band``
(zero-based) at the reduced ``k_point``; for a degenerate band give a
Cartesian ``direction`` for the branch masses.

[Source](../python/dft_rs/native.py#L3182).

### `dft_rs.native.boltzmann_transport`

```python
boltzmann_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls) -> dict[str, Any]
```

Scalar-relaxation-time, rigid-band Boltzmann transport: sigma/tau,
Seebeck and electronic thermal conductivity per temperature and chemical
potential (``transport_mu_shifts`` in Hartree or
``carrier_concentrations_cm3``). Optional ``scattering_table`` rows
[absolute band energy in Hartree, positive lifetime in fs] include the
energy-dependent lifetime and return conductivity in S/m and electronic
thermal conductivity in W/(m K), without division by tau. Interpolation
is log-linear with constant endpoint tails. Frequencies may accompany
the table; do not also give ``relaxation_time_fs``.

[Source](../python/dft_rs/native.py#L3189).

### `dft_rs.native.hall_transport`

```python
hall_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls) -> dict[str, Any]
```

Ordinary weak-field Hall transport of isolated rigid bands, with
constant relaxation time. Returns ``d sigma/dB / tau**2`` in S/(m T s^2)
and ``d rho/dB`` in m^3/C, axes (current, electric field, magnetic field)
for conductivity and (electric field, current, magnetic field) for
resistivity. Give chemical-potential shifts or carrier concentrations.
The electron parabolic-band coefficient R_yxz is negative. Optional
``hall_onsager_tolerance`` (1e-10) controls the raw symmetry diagnostic
and warnings, without changing the tensors. For causal finite-frequency
intraband tensors, give ``transport_frequencies_hartree`` (signed angular
frequencies) and positive ``relaxation_time_fs``. The extra
``frequency_results`` contain SI tensors with last axis [real, imaginary],
including Seebeck (V/K), Peltier (V), open-circuit electronic heat
conductivity (W/(m K)) and unprojected inverse/Kelvin residuals.
Alternatively give ``scattering_table`` rows [energy_hartree, lifetime_fs]
for energy-dependent scattering, optionally with frequencies, without a
constant lifetime. Then ``hall_conductivity_derivative`` includes tau(E)^2
and has units S/(m T). Semilocal, full/range-separated hybrid and ECP
references use the same band operator and scattering provider.

[Source](../python/dft_rs/native.py#L3210).

### `dft_rs.native.coherent_transport`

```python
coherent_transport(numbers, positions, cell, *, transport_k_mesh, transport_frequencies_hartree, **controls) -> dict[str, Any]
```

Covariant independent-particle intra/interband conductivity, including
exact degenerate blocks. SI S/m tensors have last axis [real, imaginary].
Defaults: transport_temperature_k=300, scattering_hartree=0.001,
transport_mu_shift=0 relative to the reference level. Includes physical
AO position connections; exp(-i omega t), signed angular frequencies.

[Source](../python/dft_rs/native.py#L3236).

### `dft_rs.native.quantum_hall`

```python
quantum_hall(numbers, positions, cell, **controls) -> dict[str, Any]
```

Non-Abelian Chern invariant and integer Hall conductance (S) of a
positive-gap occupied Bloch slice. topology_mesh=[8,8],
topology_plane_axes=[0,1], topology_fixed_k=0 by default. Zero-temperature
insulating reference; refine the mesh to verify the gap and invariant.
The zero-field collinear reference has no orbital TR-breaking field/SOC.

[Source](../python/dft_rs/native.py#L3247).

### `dft_rs.native.wannier_flow`

```python
wannier_flow(numbers, positions, cell, **controls) -> dict[str, Any]
```

Hybrid Wannier charge centres of the occupied bands along the loop
axis as a function of the transverse wave vector along the flow axis
(``topology_plane_axes=[loop, flow]``, ``wannier_loop_divisions``,
``wannier_flow_points``); the winding of their sum is the Chern number.

[Source](../python/dft_rs/native.py#L3256).

### `dft_rs.native.periodic_cubic_fc_along`

```python
periodic_cubic_fc_along(numbers, positions, cell, **controls) -> dict[str, Any]
```

Cubic force constants of a cell (Hartree/Bohr^3 per primitive cell)
contracted with the Gamma-point displacement pattern ``probe`` (one
value per Cartesian coordinate of the cell): the derivative of the
Gamma-point force constants along it, from first-order responses of the
Gamma-point supercell of the k mesh. Optional field (Hartree/e/Bohr)
differentiates the stationary finite-bias Berry electric enthalpy at
fixed Cartesian electric bias.

[Source](../python/dft_rs/native.py#L3264).

### `dft_rs.native.periodic_quartic_fc_along`

```python
periodic_quartic_fc_along(numbers, positions, cell, **controls) -> dict[str, Any]
```

Quartic force constants of a cell (Hartree/Bohr^4 per primitive cell)
contracted with the Gamma-point displacement patterns ``probe`` and
``second_probe`` (default the same pattern). Optional field
(Hartree/e/Bohr) selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/native.py#L3275).

### `dft_rs.native.periodic_cubic_fc`

```python
periodic_cubic_fc(numbers, positions, cell, **controls) -> dict[str, Any]
```

Complete cubic force constants of a cell (Hartree/Bohr^3 per primitive
cell) for Gamma-point displacements, from first-order responses of the
Gamma-point supercell of the k mesh. Optional field (Hartree/e/Bohr)
selects the stationary finite-bias Berry electric enthalpy.

[Source](../python/dft_rs/native.py#L3283).

### `dft_rs.native.periodic_quartic_fc`

```python
periodic_quartic_fc(numbers, positions, cell, **controls) -> dict[str, Any]
```

Complete quartic force constants of a cell (Hartree/Bohr^4 per
primitive cell) for Gamma-point displacements. Optional field
(Hartree/e/Bohr) selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/native.py#L3291).

### `dft_rs.native.toec`

```python
toec(numbers, positions, cell, **controls) -> dict[str, Any]
```

Third-order Lagrangian (Brugger) elastic constants at clamped and
relaxed ions (``high_order_clamped``, ``high_order_relaxed`` in
Hartree/Bohr^3, also in GPa), Voigt order xx, yy, zz, yz, xz, xy.

[Source](../python/dft_rs/native.py#L3298).

### `dft_rs.native.foec`

```python
foec(numbers, positions, cell, **controls) -> dict[str, Any]
```

Fourth-order Lagrangian (Brugger) elastic constants at clamped and
relaxed ions, with the second-order relaxation of the atoms.

[Source](../python/dft_rs/native.py#L3305).

### `dft_rs.native.mode_gruneisen`

```python
mode_gruneisen(numbers, positions, cell, **controls) -> dict[str, Any]
```

Mode Grueneisen parameters of the Gamma-point phonons from the strain
derivative of the force constants (``gruneisen_shear=True`` adds the
three shear strains). ``gruneisen_order=2`` selects analytic isotropic
log-volume first/second derivatives, with ``relaxed_ion=False`` by default.
Second order returns ``second_log_frequency = -d gamma/d ln(V)`` and
undefined acoustic/unstable coefficients as null. Relaxed ions require
``gruneisen_force_tolerance`` internal stationarity (Hartree/Bohr in atomic
units). The second-order route does not include shear derivatives.

[Source](../python/dft_rs/native.py#L3311).

### `dft_rs.native.periodic_beta`

```python
periodic_beta(numbers, positions, cell, **controls) -> dict[str, Any]
```

Clamped-ion electric susceptibility of a cell and its field derivative
``second_susceptibility`` = d2P/dE2 (chi2 is half of it), from the
single-point Berry phase of the Gamma-point supercell of the k mesh.

[Source](../python/dft_rs/native.py#L3323).

### `dft_rs.native.periodic_gamma`

```python
periodic_gamma(numbers, positions, cell, **controls) -> dict[str, Any]
```

Static clamped-ion d3P/dE3 from the Berry-phase electric enthalpy.
``tensor``/``third_susceptibility`` are flat [3,3,3,3] actual derivatives;
``chi3`` divides them by six. The Gaussian susceptibility derivative
unit is e^2 Bohr^2/Hartree^2. ``use_symmetry=True`` evaluates 15 independent
static components and expands their permutations. Set it to False to
evaluate all 81 components and measure the raw permutation residual;
that residual is None when symmetry expansion is used. Path residuals
are always freshly checked with the response operator.
Second-order internal relaxation and dynamic response remain separate.

[Source](../python/dft_rs/native.py#L3330).

### `dft_rs.native.periodic_raman`

```python
periodic_raman(numbers, positions, cell, **controls) -> dict[str, Any]
```

Born charges and the displacement derivative of the susceptibility
(``raman_tensor``, Bohr^-1) of a periodic cell.

[Source](../python/dft_rs/native.py#L3344).

### `dft_rs.native.electrostriction`

```python
electrostriction(numbers, positions, cell, **controls) -> dict[str, Any]
```

Strain derivative of the electric susceptibility of a periodic cell
at clamped ions and fixed reduced field.

[Source](../python/dft_rs/native.py#L3350).

### `dft_rs.native.piezo_higher`

```python
piezo_higher(numbers, positions, cell, **controls) -> dict[str, Any]
```

Proper clamped-ion piezoelectric tensor of a periodic cell and its
strain derivative.

[Source](../python/dft_rs/native.py#L3356).

### `dft_rs.native.periodic_efield`

```python
periodic_efield(numbers, positions, cell, *, field, **controls) -> dict[str, Any]
```

Self-consistent field of a periodic cell in the finite uniform
electric field ``field`` (atomic units), with the Berry-phase dipole and,
with ``forces=True``, analytic forces.

[Source](../python/dft_rs/native.py#L3362).

### `dft_rs.native.dielectric_q`

```python
dielectric_q(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]
```

Self-consistent density response ``chi(q, q)`` and inverse dielectric
function at the reduced wave vectors ``q_points`` (flat list, three
numbers each), which must be commensurate with the k mesh.

[Source](../python/dft_rs/native.py#L3369).

### `dft_rs.native.orbital_susceptibility_q`

```python
orbital_susceptibility_q(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]
```

Orbital magnetic susceptibility ``chi(q)`` of a periodic cell at the
reduced wave vectors ``q_points`` commensurate with the k mesh, for two
transverse polarizations of the vector potential. GKS and GHF also return
the phase-averaged ``chi_transverse`` tensor in the two magnetic-field axes.
See the API guide for the reference and functional limits.

[Source](../python/dft_rs/native.py#L3376).

### `dft_rs.native.periodic_dc`

```python
periodic_dc(numbers, positions, cell, **controls) -> dict[str, Any]
```

Divide-and-conquer Kohn-Sham free energy of a periodic cell at Gamma
(``dc_fragments``, ``dc_buffer`` in Bohr by the minimum image,
``dc_smearing`` in Hartree), with ``forces=True`` and ``stress=True``
for the derivatives including the response of the partition.

[Source](../python/dft_rs/native.py#L3386).

### `dft_rs.native.dc_stress`

```python
dc_stress(numbers, positions, cell, **controls) -> dict[str, Any]
```

Stress of the divide-and-conquer free energy of a periodic cell at
Gamma (Hartree/Bohr^3 and GPa).

[Source](../python/dft_rs/native.py#L3394).

### `dft_rs.native.periodic_nmr_shielding`

```python
periodic_nmr_shielding(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]
```

Nuclear magnetic shielding in a periodic cell at the reduced wave
vectors ``q_points`` commensurate with the k mesh (``nuclei`` selects
atoms of the cell); wave vectors along two axes give the full tensor.
The bulk magnetization (G = 0) term is not included.

[Source](../python/dft_rs/native.py#L3400).

### `dft_rs.native.berry_dipole`

```python
berry_dipole(numbers, positions, cell, *, transport_k_mesh, **controls) -> dict[str, Any]
```

Berry curvature dipole of the Fermi surface on ``transport_k_mesh`` at
``transport_temperature_k``, with the chemical potential shifted by
``transport_mu_shift`` (Hartree) or fixed by ``berry_dipole_doping``
(electrons per cell).

[Source](../python/dft_rs/native.py#L3408).

### `dft_rs.native.lab_frame_spin_spiral`

```python
lab_frame_spin_spiral(numbers, positions, cell, *, spiral_q, **controls) -> dict[str, Any]
```

Energy of a flat spin spiral of reduced wave vector ``spiral_q`` by a
two-component SCF in the smallest supercell that holds it (``k_mesh``
must be divisible by it). The spins turn about ``spiral_axis``; the state
is seeded by a Zeeman field of strength ``spiral_seed_field`` and then
converged again without it. ``nuclear_soc_scale`` (default 0) adds the
one-electron nuclear spin-orbit term. Positions/cell are Bohr, energies
Hartree per cell of the structure; pbc selects physical periodic axes.

[Source](../python/dft_rs/native.py#L3416).

### `dft_rs.native.dmi`

```python
dmi(numbers, positions, cell, *, spiral_q, **controls) -> dict[str, Any]
```

Chirality-odd spin-orbit energy of the flat spin spiral of reduced wave
vector ``spiral_q``: half the difference of the nuclear spin-orbit energy
of the two chiralities, to first order in ``nuclear_soc_scale`` on the
states converged without it, and with ``dmi_self_consistent=True`` also
from SCF runs with the spin-orbit term. Hartree per cell of the structure.

[Source](../python/dft_rs/native.py#L3427).

### `dft_rs.native.bands`

```python
bands(numbers, positions, cell, *, path, **controls) -> dict[str, Any]
```

Band structure along ``path`` (reduced k vertices).

[Source](../python/dft_rs/native.py#L3436).

### `dft_rs.native.periodic_spin_response`

```python
periodic_spin_response(numbers, positions, cell, **controls) -> dict[str, Any]
```

Longitudinal spin-only Zeeman response, H'_alpha/beta=+/-g B_z N(q)/4.

response_q is Cartesian inverse Bohr; frequency and frequency_damping
use Hartree, with imaginary=True for the imaginary axis. spin_probe_field
is [real, imaginary] magnetic atomic units (default [1,0]); spin_g_factor
defaults to 2. Results are per cell; complex values end in [real,imag].
Orbital fields, transverse spin, SOC and spin relaxation are separate.
The uniform static equilibrium Pauli limit uses pauli_spin_susceptibility
with exchange_enhanced=True; it is refused here instead of independently
fixing both spin populations. Geometry and lattice rows are Bohr.

[Source](../python/dft_rs/native.py#L3440).

### `dft_rs.native.periodic_field_scf`

```python
periodic_field_scf(numbers, positions, cell, **controls) -> dict[str, Any]
```

Complex self-consistent collinear 1D/2D/3D reference in a real static
periodic field. field_wave_indices=[1,0,0], field_vector_real/imag=[0,0,0],
field_scalar=[0,0], field_amplitude=1. A(r)=lambda*(Aq*exp(iqr)+conjugate).
Includes both paramagnetic and A^2 terms and complex variational exchange.
Energy/free energy and analytic amplitude derivative are Hartree per cell.
Zero net magnetic flux; spin Zeeman, SOC and uniform flux are separate models.
Native geometry is Bohr; use pbc to select physical periodic axes.
Optional forces and stress differentiate the commensurate Fourier field
at fixed reduced wave indices and fixed Cartesian amplitudes.

[Source](../python/dft_rs/native.py#L3454).

### `dft_rs.native.periodic_spinor_scf`

```python
periodic_spinor_scf(numbers, positions, cell, **controls) -> dict[str, Any]
```

General complex two-component 1D/2D/3D SCF with one chemical potential.
HF/local spin-density providers, Pauli SOC of the prescribed periodic
scalar potential, periodic Zeeman fields and general Hermitian spin/SOC
AO operators are variational. spin_field_real/imag are Fourier amplitudes.
spinor_operator_real/imag are flat [k][up AOs,down AOs]^2 Hartree arrays.
All fields scale with field_amplitude, including the kinetic-momentum SOC
contact. nuclear_soc_scale=1 includes analytic one-electron nuclear SOC
with the same dimensional Green function as SCF; nuclear_widths supplies
finite Gaussian nuclei. Nonlocal-core/two-electron SOC, scalar relativistic
terms and uniform-flux magnetic cells remain tracked. Nuclear SOC with a
vector potential additionally requires its nuclear kinetic-momentum contact.
Positions/cell are Bohr; pbc selects physical periodic axes.

[Source](../python/dft_rs/native.py#L3467).

### `dft_rs.native.periodic_magnetic_scf`

```python
periodic_magnetic_scf(numbers, positions, cell, **controls) -> dict[str, Any]
```

Uniform-flux London spinor SCF on a physical 2D/3D magnetic cell.
magnetic_field is in atomic units; each face flux divided by 2*pi must
be integer. The finite cell uses one Gamma sector and the zero-average
periodic Coulomb interaction for Hartree and full/range exchange.
Includes uniform Zeeman and nuclear Pauli SOC with kinetic-momentum
contact. Nuclear widths and gauge origin are Bohr. General Hermitian
spinor_operator_real/imag arrays are flat [up AOs,down AOs]^2 Hartree.
HF, spin-covariant LDA/GGA/meta-GGA and full/range exchange share the
spinor solver. GGA/meta-GGA reports include xc_diagnostics for angular
refinement, alongside the spatial quadrature defects.
magnetic_model selects finite_gamma or crystal_bz; the latter integrates
magnetic Bloch sectors and requires its own mesh/cutoff convergence.
  Optional field is a fixed Cartesian electric bias, Hartree/(e Bohr),
  on the integer-occupation 3D finite_gamma single-point Berry branch.
  Its energy is reported in energy_components.orbital_field and its
  dipole in electric_dipole (e Bohr). Uniform background compensation is
  described with the local insulating branch in electric_field_model and
  centred at half the sum of lattice vectors. Nuclear response keeps
  both electric and magnetic fields fixed. Magnetic crystal strings and
  open-boundary electric coupling retain their independent providers.
Wires use periodic_spinor_scf with uniform_magnetic_field. Functional,
boundary and response combinations retain their documented restrictions.
Optional nuclear_order=1..4 selects relaxed nuclear response on the
integer-occupation 2D/3D finite_gamma exchange/SOC provider.
nuclear_direction gives a scalar Taylor path (3N dimensionless entries);
otherwise orders 1/2 return gradient/Hessian, and orders 3/4 need probe
and optional second_probe for contracted tensors. See docs/python-api.md#table-3-native-python-and-ase for
units, raw residuals, acceptance scope and remaining providers.

[Source](../python/dft_rs/native.py#L3483).

### `dft_rs.native.dos`

```python
dos(numbers, positions, cell, *, sigma, **controls) -> dict[str, Any]
```

Gaussian-broadened electronic DOS on the physical periodic axes.

The native geometry is in Bohr and ``sigma`` is in Hartree.
``projected=True`` adds Mulliken weights by atom and angular momentum.
ROKS uses separate spin spectra within its common-orbital subspaces.

[Source](../python/dft_rs/native.py#L3515).

### `dft_rs.native.pauli_spin_susceptibility`

```python
pauli_spin_susceptibility(numbers, positions, cell, **controls) -> dict[str, Any]
```

Bare equal/unequal-channel rigid-band thermodynamics reoccupies the
frozen bands with a common chemical potential and zero reservoir spin
bias, at fixed total particle number. It includes the common-potential
magnetic derivative, heat capacity and thermally averaged spin DOS.
``exchange_enhanced=True`` selects self-consistent static spin response.
The enhanced semilocal mode holds the baseline reference spin bias and
converges a separate thermal reference for each ``temperatures_k`` and
uses ``fermi_k_mesh`` as its SCF/response mesh. Spin populations exchange
particles at fixed total number to first order. ``spin_g_factor=2``;
no orbital or SOC contribution. Reports units and raw residuals.

[Source](../python/dft_rs/native.py#L3525).

### `dft_rs.native.cli_main`

```python
cli_main(argv: Optional[Sequence[str]]=None) -> int
```

Run the ``dft-rs`` command line and return its process exit status.

Explicit ``argv`` omits the program name. The installed console script
calls this without arguments, using ``sys.argv[1:]``. The compiled core
supplies the parser and calculations; no external executable is needed.

[Source](../python/dft_rs/native.py#L3539).

### `dft_rs.native.piezoelectric_strain`

```python
piezoelectric_strain(*, piezoelectric_stress, piezoelectric_compliance, ion_response) -> dict[str, Any]
```

Convert supplied proper e and fixed-field S to d=eS without SCF.

Inputs are 3x6 e/Bohr^2 and 6x6 Bohr^3/Hartree, in engineering Voigt
order xx,yy,zz,yz,xz,xy with identical axes and ion_response='clamped'
or 'relaxed'. Returns d in e Bohr/Hartree, C/N and pm/V. Neither the
physical reference nor mechanical stability is validated by this conversion.

[Source](../python/dft_rs/native.py#L3552).

### `dft_rs.native.aggregate_moduli`

```python
aggregate_moduli(*, phases) -> dict[str, Any]
```

Voigt, Reuss, Hill, Hashin-Shtrikman and self-consistent moduli (GPa)
of an isotropic aggregate; ``phases`` rows are [fraction, K, G].

[Source](../python/dft_rs/native.py#L3566).

### `dft_rs.native.scdft_pairing_kernel`

```python
scdft_pairing_kernel(**inputs) -> dict[str, Any]
```

Static LM2005 singlet pairing and phonon mass on supplied normal
bands. Energies and chemical potential are Hartree, squared weighted
electron-phonon vertices are Hartree squared, temperature is Kelvin.
Supplying band_weights also generates dimensionless diagonal Z; K
excludes these final-band weights and Z includes them once. Optional
energy, chemical-potential, temperature, phonon, vertex, screened-Coulomb
and weight directions give Taylor coefficients through order four.
Reports separate K/Z quadrature errors; no gap or density backreaction
is inferred. The mass uses the particle-hole-symmetric approximation.

[Source](../python/dft_rs/native.py#L3572).

### `dft_rs.native.superconducting_state`

```python
superconducting_state(**inputs) -> dict[str, Any]
```

Static superconducting gap or full Nambu density response, orders 0..4.

Supply normal band_energies (Hartree), band_weights,
electronic_temperature (Kelvin), electron_number,
initial_gap_real/imaginary and initial chemical_potential (Hartree).
pairing_model="supplied" uses a reciprocal pairing_kernel (Hartree).
"lm2005-off-diagonal" computes K from phonon_energies, weighted
pair_couplings_squared and screened_pair_coulomb at the responding mu.
Both of these models require explicit mass_renormalization Z. "lm2005"
generates K and particle-hole-symmetric phonon Z from the same vertices,
temperature, band weights and responding mu; explicit Z is then rejected.
Direction controls define a dimensionless Taylor path. The same AD
equations solve number and global phase; physical residuals and the
Goldstone defect and mass coefficients are returned. Supplied normal
orbitals/phonons may have explicit parameter paths. A zero seed selects
the normal stationary branch.

scdft_phase_selection=True selects the canonical phase of the supplied
reciprocal kernel with uniform Z at one temperature. Omit the gap and
chemical-potential seeds. The selected state includes the condensation
free energy in Hartree, entropy divided by Boltzmann's constant and
the dimensionless variational stability margin.

pairing_model="nambu" instead evaluates the grand-canonical full matrix
density of supplied nambu_normal_real/imaginary and antisymmetric
nambu_pairing_real/imaginary (Hartree), with optional dimensionless
nambu_overlap_real/imaginary (identity by default). Spin is already in
the matrix indices. Corresponding nambu_*_direction_real/imaginary and
chemical_potential_direction define the same Taylor path. Positive
electronic_temperature is required. Reports include full, normal and
anomalous densities, number coefficients and particle/hole diagnostics.
Supply signed frequencies and optional frequency_damping (Hartree) for
collisionless retarded linear response at fixed quasiparticle occupations.
Each nambu_probe_normal_real/imaginary or nambu_probe_pairing_real/imaginary
row holds one row-major dimensionless probe matrix. Set optional symmetric
nambu_probe_interaction (Hartree) for separable induced fields. Bare and
induced susceptibilities have axes frequency, observed probe, driven probe
and units inverse Hartree. Both paths use supplied normal/pairing operators.

[Source](../python/dft_rs/native.py#L3586).

### `dft_rs.native.dirac_coulomb_scf`

```python
dirac_coulomb_scf(numbers, positions, **controls) -> dict[str, Any]
```

Finite-nucleus no-pair four-component Hartree-Fock in atomic units.

Optional dirac_interaction selects coulomb, gaunt or breit. The positive
sector is fixed by the nuclear core and the current energies are reported
as separate direct and signed exchange terms.

[Source](../python/dft_rs/native.py#L3629).

### `dft_rs.native.two_component_relativistic`

```python
two_component_relativistic(numbers, positions, **controls) -> dict[str, Any]
```

Finite-nucleus X2C, DKH2 or matrix ZORA with scalar or spin-orbit SCF.

Give positions and nuclear_rms_radii in Bohr. The common picture
transforms one-electron observables and the interaction density.
response_order=1 returns the uniform electric polarizability, with
optional frequencies and positive damping in Hartree.

[Source](../python/dft_rs/native.py#L3639).

### `dft_rs.native.dirac_dks`

```python
dirac_dks(numbers, positions, **controls) -> dict[str, Any]
```

Fixed-sector four-component Dirac–Coulomb KS in atomic units.

[Source](../python/dft_rs/native.py#L3650).

### `dft_rs.native.no_pair_projector`

```python
no_pair_projector(numbers=None, positions=None, **inputs) -> dict[str, Any]
```

Electronic-sector projection of balanced four-component H/S.
Hamiltonian and separator are Hartree, overlap dimensionless. Optional
H/S path directions yield analytic projector response through order four.
With dirac_integrals="rkb", molecular numbers/positions (Bohr), basis
and explicit nuclear_rms_radii (Bohr) generate finite-nucleus balanced
integrals and full one-electron SOC. Nuclear displacement/radius/c paths
retain all metric/projector coefficients. dirac_integrals="rmb" instead
includes magnetic_field and magnetic_direction (atomic units) and fixed
gauge_origin (Bohr). The small basis is sigma.(p+A)/(2c); orbital, spin,
diamagnetic and complex metric responses are retained. This common-origin
Gaussian basis has no London phases. Neither mode is interacting SCF.

[Source](../python/dft_rs/native.py#L3655).

### `dft_rs.native.rock_physics`

```python
rock_physics(**inputs) -> dict[str, Any]
```

Dry frame, Gassmann fluid substitution and Biot parameters from
supplied mineral and fluid data (no defaults for physical inputs).

``frame`` is ``critical-porosity``, ``hertz-mindlin``,
``hashin-shtrikman``, ``given``, or one of the spheroidal-pore models
``dem``, ``kuster-toksoz`` and ``self-consistent`` (these need
``pore_aspect_ratio``). ``mineral_conductivity_w_mk`` together with
``fluid_conductivity_w_mk`` adds ``thermal_conductivity_w_mk``.

[Source](../python/dft_rs/native.py#L3671).

### `dft_rs.native.continuous_species`

```python
continuous_species(numbers, positions, *, fixed_electron_count, charge_directions, charge_coordinates, charge_bounds, **controls) -> dict[str, Any]
```

Fixed-template real nuclear-charge collinear model, in atomic units.

``numbers`` remain integer basis/grid template identifiers. Physical charges
are ``physical_nuclear_charges + charge_directions.T @ charge_coordinates``;
reference charges default to the integer labels. Coordinates are dimensionless
with explicit bounds. ``fixed_electron_count`` independently fixes occupations.
Explicit ``fixed_spin_electron_counts=[alpha, beta]`` selects UKS with fixed
non-negative integer spin populations (alpha >= beta), summing to the total.
Reports analytic energy, parameter gradient/Hessian, dipole Jacobian and
static polarizability for LDA/PBE at clamped geometry. Optional
``charge_nuclear_gradient=True`` differentiates moving AO/grid templates.
Optional ``local_potential_*`` controls add fixed-exponent Gaussian
electronic potentials; amplitudes and their parameter directions use
Hartree, and exponents use Bohr^-2. ``steric_*`` supplies an independent
classical overlap with an explicit binary pair-exclusion mask. Atom or
three-anchor frames determine center motion. ``fragment_internal_energy``
is a constant in Hartree, included in every SCF iteration and final total.
``charge_probe_points`` returns positive electron density and the physical
electrostatic potential. ``charge_dynamic_response=True`` also evaluates
the existing paired frequency response, with the ordinary frequency controls.
Third-order polarizability parameter derivatives are explicitly unsupported.

[Source](../python/dft_rs/native.py#L3683).

### `dft_rs.native.generate_methyl_fragment`

```python
generate_methyl_fragment(numbers, positions, *, fragment_spec, **controls) -> dict[str, Any]
```

Generate a fixed one-center methyl candidate through the common driver.

Positions and optional construction ESP probes use Bohr. ``fragment_spec``
declares explicit/retained atom indices, the three port anchors, bonds and
one electronic center. The returned ``model_json`` is opaque native-unit
data; its presence does not establish chemical accuracy. Frozen and active
electron counts, projection residuals and candidate status are reported.

[Source](../python/dft_rs/native.py#L3713).

### `dft_rs.native.evaluate_reduced_fragment`

```python
evaluate_reduced_fragment(numbers, positions, *, model, fragment_mode='evaluate', **controls) -> dict[str, Any]
```

Evaluate a saved reduced candidate in atomic units.

``fragment_mode="evaluate"`` preserves the original rigid methyl record.
``fragment_mode="few-evaluate"`` selects a versioned fixed primitive8 /
fittedGaussian3 PBE record: RKS CoreSp/CoreSpH/CoreSpHH or UKS CoreSp.
Supply retained integer labels followed by the saved H-labelled main ionic
role; auxiliary basis sites are internal and are not force coordinates.
The new mode returns energy, analytic primary-coordinate forces and
candidate/accounting/convergence metadata, without dipole or response.
``model`` is opaque native-unit JSON text or a dict. The record fixes its
reference and spin sector; no fitting or chemical acceptance is inferred.
Resource controls retain the common driver envelope.

``fragment_mode="rigid-evaluate"`` selects an explicit versioned
``rigid-static-core-v1`` primitive recipe. Numbers/positions identify
retained point-ion centers; the record supplies the separate full/frozen
electron ledger and fixed body dictionary. ``derivative_order=0/1/2``
requests energy, analytic center/Euler gradient, or relaxed Hessian.
Center and angle blocks carry distinct units; angles remain radians.
This bounded experimental PBE restricted route has no internal bond
relaxation or private generated-model loader. Its tested scope and the
two unresolved direct gradient-difference checks remain explicit in the
report/manual; chemical transfer is not established.

``"scalar-evaluate"`` and ``"scalar-optimize"`` consume the experimental
``frozen-scalar-action-v1`` record. Supply its first-state raw geometry
and flattened per-site simplex ``fragment_weights`` in saved site order.
Full joint endpoint scalar evaluators retain intersite interactions;
this is not a compressed potential. Optimization changes weights only,
with ``fragment_weight_strategy="joint"``, ``"one-pass"`` or ``"cyclic"``.
The default weight tolerance is 1e-4 Hartree; SCF gates are independent.

``"scalar-gradient"``, ``"scalar-geometry-optimize"`` and ``"scalar-hessian"`` require
explicit ``fragment_state`` and ``fragment_endpoint`` identities and an
endpoint with ``geometry_transport="fixed-internal-affine-v1"`` plus
``geometry_atom_owners``. Each full atom translates with its declared raw
center while its saved internal offset remains fixed. These modes return
analytic raw-center gradients/forces or optimize those coordinates with
the common geometry optimizer. They do not mix endpoint weights, rotate
fragments, relax internal offsets, or supply model thermochemistry.
``scalar-hessian`` returns full constrained curvature and raw response
residuals. The separate ``frozen-scalar-collinear-action-v1`` record
explicitly declares alpha/beta frozen orbitals and active electron counts;
the original record stays paired RKS. These saved evaluators currently
support local and gradient functionals only; chemical acceptance is PBE.

[Source](../python/dft_rs/native.py#L3726).

### `dft_rs.native.multicomponent`

```python
multicomponent(numbers, positions, *, quantum_atoms=None, **controls) -> dict[str, Any]
```

Experimental simultaneous electronic/quantum-nuclear SCF, in atomic units.

``quantum_atoms`` gives zero-based nuclei to replace by normalized quantum
Hartree orbitals. ``interspecies_correlation='epc17'`` (default) couples both
total densities self-consistently. ``quantum_masses`` are electron masses;
``quantum_exponents`` are inverse Bohr squared. Alternatively ``particles``
is a list of objects with ``kind`` (``nucleus``, ``positron``, ``muon``,
``negative-muon`` or ``particle``), ``atom`` or ``position`` in Bohr,
and optional ``mass``, ``exponents``, ``lmax``, ``correlation`` and ``label``.
A custom ``particle`` also needs ``charge`` in elementary charges and
``mass`` in electron masses. Do not mix ``particles`` with legacy particle
controls. ``positron_positions`` adds distinguishable positrons, with
``positron_correlation='drummond-2011'`` for the dilute-positron local model
or ``'none'`` for a Coulomb-only model. ``annihilation_enhancement`` selects
``ipm``, ``drummond-2011`` or ``barbiellini-ap/sl/qmc``; the reported
spin-averaged two-photon rates are in inverse atomic time and inverse ns.
Optional ``annihilation_gradient_alpha`` damps the contact enhancement.
Ordinary electronic XC is reused.
Select ``dispersion='none'``. The energy includes laboratory center-of-mass
motion; unselected nuclei stay clamped. This is not an internal-frame exact
vibronic solution. ``species_derivative_order=1`` or ``2`` requests the
analytic gradient or relaxed Hessian; optional ``species_direction``
contracts them. ``quantum_position_constraint=True`` makes quantum
coordinates nuclear expectation positions, with optional explicit
``quantum_position_targets`` in Bohr. Without that constraint a quantum
Gaussian center is a basis parameter. ``species_polarizability=True``
returns the total-charge static response, including nuclear motion or
its constrained response. All calculations use the common driver.
``species_electronic_model="dirac-coulomb"`` selects fixed-sector
four-component HF energy with distinguishable quantum nuclei.
Supply one finite Gaussian ``nuclear_rms_radii`` value in Bohr per
atom. This model defaults to Coulomb-only interspecies coupling.
The uniform electric field uses atomic units and returns the total
dipole. The optional polarizability uses the same fixed sector and
accepts signed frequency and damping in Hartree.
``species_quadrupole_polarizability=True`` adds the electric quadrupole
and its response with the dipole–gradient and conjugate-moment–field
cross blocks. ``species_field_gradient`` applies five symmetric
traceless mode amplitudes in Hartree/e/Bohr². The shared moment
origin defaults to laboratory zero and can be set with ``origin``.

[Source](../python/dft_rs/native.py#L3780).

### `dft_rs.native.casscf`

```python
casscf(numbers, positions, *, cas, **controls) -> dict[str, Any]
```

Complete-active-space SCF through the common driver, in atomic units.

``cas=[active electrons, active orbitals]``; the remaining electrons fill
doubly occupied core orbitals. The reference SCF (``xc``, normally
``hf``) supplies starting orbitals; ``active_orbitals`` names them
explicitly. ``cas_roots`` lowest states of the reference multiplicity are
averaged with ``state_weights`` (equal by default); ``casci=True`` keeps
the starting orbitals. Energies are Hartree.

[Source](../python/dft_rs/native.py#L3827).

### `dft_rs.native.mcpdft`

```python
mcpdft(numbers, positions, *, cas, **controls) -> dict[str, Any]
```

Multiconfiguration pair-density functional theory, in atomic units.

Takes the ``casscf`` controls plus ``on_top_functional`` (``t<name>`` or
``ft<name>`` for a registered local or gradient-corrected functional,
``tpbe0``/``ftpbe0``; default ``tpbe``), ``pdft_lambda`` (hybrid fraction
of the active-space energy) and ``pdft_method`` (``state-specific``,
``l-pdft``, ``cms-pdft`` or ``xms-pdft``). Energies are Hartree.

[Source](../python/dft_rs/native.py#L3840).

### `dft_rs.native.feature_registry`

```python
feature_registry() -> dict[str, Any]
```

The registry of public calculations as data.

[Source](../python/dft_rs/native.py#L3852).

### `dft_rs.system.Molecule`

```python
class Molecule
Molecule(numbers: tuple, positions: tuple)
```

Integer atomic numbers (1..118) and Cartesian positions in Bohr.

[Source](../python/dft_rs/system.py#L108).

### `dft_rs.system.Molecule.from_angstrom`

```python
@classmethod
Molecule.from_angstrom(cls, numbers, positions)
```

Cartesian positions given in Angstrom.

[Source](../python/dft_rs/system.py#L123).

### `dft_rs.system.Molecule.from_xyz`

```python
@classmethod
Molecule.from_xyz(cls, path)
```

An XYZ file (Angstrom).

[Source](../python/dft_rs/system.py#L129).

### `dft_rs.system.Molecule.from_ase`

```python
@classmethod
Molecule.from_ase(cls, atoms)
```

An ASE ``Atoms`` object (Angstrom); its cell is ignored.

[Source](../python/dft_rs/system.py#L135).

### `dft_rs.system.Molecule.to_ase`

```python
Molecule.to_ase(self)
```

An ASE ``Atoms`` object (Angstrom); needs ASE.

[Source](../python/dft_rs/system.py#L141).

### `dft_rs.system.Molecule.natoms`

```python
@property
Molecule.natoms(self)
```

Number of atoms in this immutable structure.

[Source](../python/dft_rs/system.py#L148).

### `dft_rs.system.Molecule.angstrom_positions`

```python
Molecule.angstrom_positions(self)
```

Cartesian positions converted from Bohr to Angstrom.

[Source](../python/dft_rs/system.py#L152).

### `dft_rs.system.Molecule.with_positions`

```python
Molecule.with_positions(self, positions)
```

The same atoms at other Cartesian positions (Bohr).

[Source](../python/dft_rs/system.py#L157).

### `dft_rs.system.Molecule.displaced`

```python
Molecule.displaced(self, atom, axis, step)
```

One atom moved by ``step`` Bohr along a Cartesian axis (0, 1, 2).

[Source](../python/dft_rs/system.py#L161).

### `dft_rs.system.Crystal`

```python
class Crystal
Crystal(numbers: tuple, positions: tuple, cell: tuple, pbc: tuple = (True, True, True))
```

Integer atomic numbers (1..118), positions and lattice rows in Bohr.

[Source](../python/dft_rs/system.py#L167).

### `dft_rs.system.Crystal.from_angstrom`

```python
@classmethod
Crystal.from_angstrom(cls, numbers, positions, cell, pbc=(True, True, True))
```

Cartesian positions and lattice rows given in Angstrom.

[Source](../python/dft_rs/system.py#L195).

### `dft_rs.system.Crystal.from_fractional`

```python
@classmethod
Crystal.from_fractional(cls, numbers, fractional, cell, pbc=(True, True, True))
```

Three-component fractional coordinates with lattice rows in Bohr.

[Source](../python/dft_rs/system.py#L206).

### `dft_rs.system.Crystal.from_xyz`

```python
@classmethod
Crystal.from_xyz(cls, path)
```

An extended XYZ file (Angstrom) whose comment line carries
``Lattice="ax ay az bx by bz cx cy cz"`` and optionally ``pbc="T T F"``.

[Source](../python/dft_rs/system.py#L218).

### `dft_rs.system.Crystal.from_ase`

```python
@classmethod
Crystal.from_ase(cls, atoms)
```

An ASE ``Atoms`` object (Angstrom).

[Source](../python/dft_rs/system.py#L238).

### `dft_rs.system.Crystal.to_ase`

```python
Crystal.to_ase(self)
```

An ASE ``Atoms`` object (Angstrom); needs ASE.

[Source](../python/dft_rs/system.py#L247).

### `dft_rs.system.Crystal.natoms`

```python
@property
Crystal.natoms(self)
```

Number of atoms in the periodic unit cell.

[Source](../python/dft_rs/system.py#L261).

### `dft_rs.system.Crystal.volume`

```python
@property
Crystal.volume(self)
```

Cell volume in Bohr^3.

[Source](../python/dft_rs/system.py#L266).

### `dft_rs.system.Crystal.fractional_positions`

```python
Crystal.fractional_positions(self)
```

Positions in units of the lattice rows.

[Source](../python/dft_rs/system.py#L270).

### `dft_rs.system.Crystal.angstrom_positions`

```python
Crystal.angstrom_positions(self)
```

Unit-cell Cartesian positions in Angstrom.

[Source](../python/dft_rs/system.py#L279).

### `dft_rs.system.Crystal.angstrom_cell`

```python
Crystal.angstrom_cell(self)
```

Lattice vectors as rows in Angstrom.

[Source](../python/dft_rs/system.py#L284).

### `dft_rs.system.Crystal.with_positions`

```python
Crystal.with_positions(self, positions)
```

The same cell with other Cartesian positions (Bohr).

[Source](../python/dft_rs/system.py#L291).

### `dft_rs.system.Crystal.with_cell`

```python
Crystal.with_cell(self, cell, *, scale_atoms=False)
```

Another cell (Bohr rows); ``scale_atoms`` keeps fractional coordinates.

[Source](../python/dft_rs/system.py#L295).

### `dft_rs.system.Crystal.displaced`

```python
Crystal.displaced(self, atom, axis, step)
```

One atom moved by ``step`` Bohr along a Cartesian axis (0, 1, 2).

[Source](../python/dft_rs/system.py#L301).

### `dft_rs.system.Crystal.scaled`

```python
Crystal.scaled(self, factor)
```

Isotropic scaling of the cell and the positions by a linear factor.

[Source](../python/dft_rs/system.py#L305).

### `dft_rs.system.Crystal.strained`

```python
Crystal.strained(self, strain)
```

Homogeneous strain ``x -> (1 + strain) x`` of cell and positions.

[Source](../python/dft_rs/system.py#L317).

### `dft_rs.system.Crystal.supercell`

```python
Crystal.supercell(self, repeats)
```

An ``n1 x n2 x n3`` supercell in the atom order of the driver:
the primitive atom index runs fastest, then ``n1``, ``n2``, ``n3``.
Repeats must be three positive integers; fractional values are rejected.

[Source](../python/dft_rs/system.py#L333).

### `dft_rs.system.features`

```python
features(space=None)
```

The registered calculations as a list of dictionaries (``id``,
``title``, ``space``, ``route``, ``python``, ``ase``, ...).

``space`` filters by ``"molecule"``, ``"periodic"`` or ``"data"``; a
calculation defined for molecules and crystals is listed for both.

[Source](../python/dft_rs/system.py#L359).

### `dft_rs.system.Calculator`

```python
class Calculator
```

A structure with reference controls (``xc``, ``basis``, ...).

``run(feature, **controls)`` runs a calculation of the common driver on
the structure. Each calculation that applies to the structure is also a
method named like its function in `dft_rs.native`; ``features()``
lists them. Use `MolecularCalculator` or
`PeriodicCalculator`, or `calculator` to pick by structure.

[Source](../python/dft_rs/system.py#L389).

### `dft_rs.system.Calculator.__init__`

```python
Calculator.__init__(self, structure, **controls: Any)
```

Bind a compatible immutable structure to common-driver controls.

[Source](../python/dft_rs/system.py#L402).

### `dft_rs.system.Calculator.with_structure`

```python
Calculator.with_structure(self, structure)
```

The same controls on another structure.

[Source](../python/dft_rs/system.py#L415).

### `dft_rs.system.Calculator.with_controls`

```python
Calculator.with_controls(self, **controls: Any)
```

The same structure with updated controls (``None`` removes one).

[Source](../python/dft_rs/system.py#L419).

### `dft_rs.system.Calculator.features`

```python
@classmethod
Calculator.features(cls)
```

Method names and the driver calculations they run.

[Source](../python/dft_rs/system.py#L425).

### `dft_rs.system.Calculator.run`

```python
Calculator.run(self, feature: str, **controls: Any)
```

Run a registered calculation; ``controls`` override the stored ones.

[Source](../python/dft_rs/system.py#L429).

### `dft_rs.system.MolecularCalculator`

```python
class MolecularCalculator(Calculator)
```

A `Molecule` with reference controls: ``energy``, ``gradient``,
``charges``, ``dipole``, ``hessian_report``, ``excitations``, ...

[Source](../python/dft_rs/system.py#L453).

### `dft_rs.system.MolecularCalculator.molecule`

```python
@property
MolecularCalculator.molecule(self)
```

The immutable molecule bound to this calculator.

[Source](../python/dft_rs/system.py#L461).

### `dft_rs.system.MolecularCalculator.with_molecule`

```python
MolecularCalculator.with_molecule(self, molecule)
```

The same controls on another molecule.

[Source](../python/dft_rs/system.py#L465).

### `dft_rs.system.MolecularCalculator.relax`

```python
MolecularCalculator.relax(self, **controls: Any)
```

Relax the geometry with analytic gradients.

This is the dedicated ``dft_rs.native.optimize`` route: next to the
reference controls it understands (``xc``, ``basis``, ``charge``,
``multiplicity``, ``grid_level``, ...) it takes ``max_steps``,
``gtol`` (Hartree/Bohr), ``step_max`` (Bohr), ``history`` and
``trajectory``. Returns the relaxed `Molecule` and the result
(``positions``, ``energy``, ``converged``, ``iterations``,
``trajectory``) as a `dft_rs.Report`.

[Source](../python/dft_rs/system.py#L469).

### `dft_rs.system.PeriodicCalculator`

```python
class PeriodicCalculator(Calculator)
```

A `Crystal` with reference controls: ``periodic_energy``,
``stress``, ``phonons``, ``phonon_bands``, ``quasi_harmonic``,
``elastic``, ``bands``, ...

[Source](../python/dft_rs/system.py#L485).

### `dft_rs.system.PeriodicCalculator.crystal`

```python
@property
PeriodicCalculator.crystal(self)
```

The immutable crystal bound to this calculator.

[Source](../python/dft_rs/system.py#L494).

### `dft_rs.system.PeriodicCalculator.with_crystal`

```python
PeriodicCalculator.with_crystal(self, crystal)
```

The same controls on another crystal.

[Source](../python/dft_rs/system.py#L498).

### `dft_rs.system.PeriodicCalculator.relax`

```python
PeriodicCalculator.relax(self, *, cell=False, **controls: Any)
```

Relax the atoms (``cell=False``) or atoms and cell (``cell=True``).

Returns the relaxed `Crystal` and the optimization report.

[Source](../python/dft_rs/system.py#L502).

### `dft_rs.system.PeriodicCalculator.scan_volumes`

```python
PeriodicCalculator.scan_volumes(self, scales: Sequence[float], feature: str='periodic-energy', **controls: Any)
```

Run one calculation on isotropically scaled copies of the crystal.

Returns ``[(volume, report), ...]`` in the order of ``scales``.

[Source](../python/dft_rs/system.py#L515).

### `dft_rs.system.calculator`

```python
calculator(structure, **controls: Any)
```

The calculator of a structure: molecular for a `Molecule`,
periodic for a `Crystal`.

[Source](../python/dft_rs/system.py#L526).

### `dft_rs.ase_calculator.DftRs`

```python
class DftRs(Calculator)
```

ASE interface to dft-rs.

Parameters mirror `dft_rs.native.calculate`. In particular,
``ecp`` is independent of ``basis``, ``density_fit`` selects an RI-J
auxiliary basis, and ``dispersion='auto'`` enables built-in VV10/CHG
terms for functionals that define them.

Examples
--------
>>> from ase.build import molecule
>>> from dft_rs import DftRs
>>> atoms = molecule("H2O")
>>> atoms.calc = DftRs(basis="def2-svp", xc="b3lyp")
>>> energy = atoms.get_potential_energy()  # eV
>>> forces = atoms.get_forces()            # eV/Angstrom
>>> hessian = atoms.calc.get_hessian(atoms)  # eV/Angstrom^2

[Source](../python/dft_rs/ase_calculator.py#L348).

### `dft_rs.ase_calculator.DftRs.__init__`

```python
DftRs.__init__(self, **kwargs)
```

Initialize the ASE calculator and empty derivative-result caches.

Keyword arguments are ASE calculator parameters and documented DFT
reference controls. Dimensional inputs use eV/Angstrom by default;
units="atomic" selects Hartree/Bohr for controls and report getters.

guess=None preserves the reference driver default (SAD for molecular
common-driver calculations, core for periodic calculations). An
explicit guess is forwarded to the selected common reference.

[Source](../python/dft_rs/ase_calculator.py#L410).

### `dft_rs.ase_calculator.DftRs.set`

```python
DftRs.set(self, **kwargs)
```

Set parameters without discarding invalid integers equal to defaults.

[Source](../python/dft_rs/ase_calculator.py#L427).

### `dft_rs.ase_calculator.DftRs.get_periodic_basis`

```python
DftRs.get_periodic_basis(self, atoms=None, *, basis=None, **options)
```

Return dimensional Gaussian S/T and optional quadrature in atomic units.

Reference geometry is converted from Angstrom. Explicit basis/grid
cutoffs remain Bohr; this getter preserves ordinary SCF results.

[Source](../python/dft_rs/ase_calculator.py#L687).

### `dft_rs.ase_calculator.DftRs.get_dimensional_scf`

```python
DftRs.get_dimensional_scf(self, atoms=None, *, basis=None, xc=None, charge=None, multiplicity=None, **options)
```

Return zero-field dimensional SCF; convert only Atoms geometry to Bohr.

[Source](../python/dft_rs/ase_calculator.py#L702).

### `dft_rs.ase_calculator.DftRs.get_dimensional_field_response`

```python
DftRs.get_dimensional_field_response(self, atoms=None, *, basis=None, xc=None, charge=None, multiplicity=None, **options)
```

Return screened dimensional orbital field response; convert only Atoms geometry to Bohr.

[Source](../python/dft_rs/ase_calculator.py#L738).

### `dft_rs.ase_calculator.DftRs.get_periodic_field_operators`

```python
DftRs.get_periodic_field_operators(self, atoms=None, *, basis=None, charge=None, **options)
```

Bare scalar/vector Fourier vertices; convert Atoms geometry to Bohr.

[Source](../python/dft_rs/ase_calculator.py#L774).

### `dft_rs.ase_calculator.DftRs.get_periodic_xc_operators`

```python
DftRs.get_periodic_xc_operators(self, atoms=None, *, basis=None, xc=None, charge=None, **options)
```

Return semilocal components and spin/Bloch XC potentials in Hartree.

Atoms geometry is converted to Bohr; explicit cutoffs and supplied
AO density matrices retain native units and ordering.

[Source](../python/dft_rs/ase_calculator.py#L785).

### `dft_rs.ase_calculator.DftRs.get_periodic_coulomb_operators`

```python
DftRs.get_periodic_coulomb_operators(self, atoms=None, *, basis=None, charge=None, **options)
```

Return contracted Coulomb components and fixed-density derivatives.

Atoms geometry is converted to Bohr. Explicit widths/cutoffs remain
atomic-unit inputs; the density uses the native spherical AO ordering.

[Source](../python/dft_rs/ase_calculator.py#L800).

### `dft_rs.ase_calculator.DftRs.get_periodic_gaussian_integral`

```python
DftRs.get_periodic_gaussian_integral(self, atoms=None, *, exponents, angular, **options)
```

Primitive Coulomb integral, converting Atoms geometry to Bohr.

Exponents use Bohr^-2; explicit source widths/cutoffs use Bohr.
Three positions select nuclear attraction with the source last;
four select an ERI. Atomic numbers do not specify primitive charges.

[Source](../python/dft_rs/ase_calculator.py#L816).

### `dft_rs.ase_calculator.DftRs.get_periodic_electrostatics`

```python
DftRs.get_periodic_electrostatics(self, atoms=None, *, charges, widths=None, compensation='neutral', countercharge_center=None, countercharge_width=None, **options)
```

Prescribed-charge Coulomb report for Atoms.pbc, in atomic units.

Atom positions/cell are converted from Angstrom. Widths and the
countercharge center/width are explicit Bohr quantities. Charges
are supplied by the caller; no electronic density is inferred.
This getter does not alter ordinary self-consistent ASE results.

[Source](../python/dft_rs/ase_calculator.py#L832).

### `dft_rs.ase_calculator.DftRs.get_periodic_field_response`

```python
DftRs.get_periodic_field_response(self, atoms=None, *, q=(0, 0, 0), scalar_potential=0j, vector_potential=(0j, 0j, 0j), omega=0.0, damping=0.0, imaginary=False, fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Screened per-cell number/charge/current report in reduced atomic units.

[Source](../python/dft_rs/ase_calculator.py#L853).

### `dft_rs.ase_calculator.DftRs.get_periodic_density_response`

```python
DftRs.get_periodic_density_response(self, atoms=None, *, q=(0, 0, 0), mode=0, omega=0.0, damping=0.0, imaginary=False, fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Return the native finite-q density report in atomic units.

Geometry and cell are converted from ASE Angstrom to Bohr. Cartesian
q and reciprocal_cutoff are already inverse Bohr; frequency is Hartree.

[Source](../python/dft_rs/ase_calculator.py#L875).

### `dft_rs.ase_calculator.DftRs.get_periodic_thermal_response`

```python
DftRs.get_periodic_thermal_response(self, atoms=None, *, delta_temperature=1.0, delta_numbers=(0.0, 0.0), delta_mu=(0.0, 0.0), fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Return static thermodynamic path variations in native units.

Geometry is converted from Angstrom; temperature directions use Kelvin,
chemical potentials Hartree, and number directions particles per cell.
The calculator needs positive electronic_temperature in Kelvin.

[Source](../python/dft_rs/ase_calculator.py#L896).

### `dft_rs.ase_calculator.DftRs.get_periodic_dielectric`

```python
DftRs.get_periodic_dielectric(self, atoms=None, *, q=(0, 0, 0), modes=(0,), omega=0.0, damping=0.0, imaginary=False, fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Return epsilon^-1=I+v*screened_chi in native atomic units.

[Source](../python/dft_rs/ase_calculator.py#L919).

### `dft_rs.ase_calculator.DftRs.calculate`

```python
DftRs.calculate(self, atoms=None, properties=('energy',), system_changes=all_changes)
```

Populate the ASE result dictionary for the requested properties.

[Source](../python/dft_rs/ase_calculator.py#L936).

### `dft_rs.ase_calculator.DftRs.get_hessian`

```python
DftRs.get_hessian(self, atoms=None)
```

Return the Cartesian Hessian in eV/Angstrom^2, computed lazily.

For fully periodic ``Atoms`` these are the Gamma-point force constants
of the cell (analytic coupled-perturbed response). Molecular calls
preserve the selected RKS/UKS/ROKS/GKS reference and double-hybrid
target through the common analytic driver; unsupported combinations
raise the driver's named error without a numerical fallback.

[Source](../python/dft_rs/ase_calculator.py#L1039).

### `dft_rs.ase_calculator.DftRs.get_dynamic_polarizability`

```python
DftRs.get_dynamic_polarizability(self, atoms=None, *, omega=0.0, damping=0.0, imaginary=False, response_tol=1e-10, response_max_iter=500, solvent=None, eps=None, **controls)
```

Molecular alpha in bohr^3; omega/damping in Hartree.

Return the native report unchanged so convergence diagnostics and
both parts of the complex tensor retain the same conventions.

[Source](../python/dft_rs/ase_calculator.py#L1072).

### `dft_rs.ase_calculator.DftRs.get_electric_field_taylor`

```python
DftRs.get_electric_field_taylor(self, atoms=None, *, direction=(0.0, 0.0, 1.0), order=3, response_tol=1e-10, response_max_iter=500, solvent=None, eps=None, reference_field=None, **controls)
```

Analytic dipole Taylor coefficients on the shared reference and solver.

Fixed populations use isospectral rotations. Default ASE units give
field directions in eV/(e Angstrom) and dipoles in e Angstrom;
units="atomic" gives the native report. Coefficients include 1/n!.

[Source](../python/dft_rs/ase_calculator.py#L1122).

### `dft_rs.ase_calculator.DftRs.get_mixed_electric_field_response`

```python
DftRs.get_mixed_electric_field_response(self, atoms=None, *, directions, response_tol=1e-10, response_max_iter=500, solvent=None, eps=None, reference_field=None, **controls)
```

Static mixed dipole derivative from the common reference/response driver.

Directions carry the field units of the selected interface; the path
parameter is dimensionless. No factorial scaling is applied.

[Source](../python/dft_rs/ase_calculator.py#L1136).

### `dft_rs.ase_calculator.DftRs.get_static_electric_tensor`

```python
DftRs.get_static_electric_tensor(self, atoms=None, *, order=2, assembly='projector', response_tol=1e-10, response_max_iter=500, solvent=None, eps=None, reference_field=None, **controls)
```

Complete independently ordered static beta/gamma on the shared driver.

assembly="variational" selects the three-solve 2n+1 beta contraction.
Fixed populations retain every differently populated orbital pair.
Per-entry unit_labels are authoritative on both unit modes.
Parent ROHF/ROKS and full complex GHF/GKS use the common stationary
energy contraction. Integer GKS also supports a common PT2 weight
and reports parent, target and generalized pair tensors.

[Source](../python/dft_rs/ase_calculator.py#L1148).

### `dft_rs.ase_calculator.DftRs.get_beta`

```python
DftRs.get_beta(self, atoms=None, **options)
```

Complete static first hyperpolarizability, without factorial scaling.

[Source](../python/dft_rs/ase_calculator.py#L1167).

### `dft_rs.ase_calculator.DftRs.get_gamma`

```python
DftRs.get_gamma(self, atoms=None, **options)
```

Complete static second hyperpolarizability, without factorial scaling.

[Source](../python/dft_rs/ase_calculator.py#L1171).

### `dft_rs.ase_calculator.DftRs.get_stability`

```python
DftRs.get_stability(self, atoms=None, *, solvent=None, eps=None, **controls)
```

Collinear real/imaginary orbital stability; atomic-unit report.

[Source](../python/dft_rs/ase_calculator.py#L1175).

### `dft_rs.ase_calculator.DftRs.get_stab_following`

```python
DftRs.get_stab_following(self, atoms=None, **controls)
```

Follow internal orbital instabilities and return the final atomic-unit report.

[Source](../python/dft_rs/ase_calculator.py#L1188).

### `dft_rs.ase_calculator.DftRs.get_dynamic_electric_tensor`

```python
DftRs.get_dynamic_electric_tensor(self, atoms=None, *, input_frequencies=None, process=None, frequencies=None, damping=0.0, tensor_order=None, imaginary=False, solvent=None, eps=None, response_tol=1e-10, response_max_iter=500)
```

Complex alpha/beta/gamma; frequencies/damping in eV, or atomic units.

Select units="atomic" for Hartree inputs and atomic tensor outputs.

[Source](../python/dft_rs/ase_calculator.py#L1192).

### `dft_rs.ase_calculator.DftRs.get_dynamic_electric_process`

```python
DftRs.get_dynamic_electric_process(self, atoms=None, *, process, frequencies, damping=0.0, **options)
```

Nonlinear optical preset; frequencies/damping in eV, or units="atomic".

[Source](../python/dft_rs/ase_calculator.py#L1210).

### `dft_rs.ase_calculator.DftRs.get_dynamic_beta`

```python
DftRs.get_dynamic_beta(self, atoms=None, *, input_frequencies=None, process=None, frequencies=None, damping=0.0, **options)
```

Full beta from explicit legs or shg/sfg/dfg/rectification/electro-optic.

Frequencies/damping use eV, or Hartree with units="atomic".

[Source](../python/dft_rs/ase_calculator.py#L1215).

### `dft_rs.ase_calculator.DftRs.get_dynamic_gamma`

```python
DftRs.get_dynamic_gamma(self, atoms=None, *, input_frequencies=None, process=None, frequencies=None, damping=0.0, **options)
```

Full gamma from explicit legs or a cubic optical preset.

Presets: thg, efish, field-induced-rectification,
two-photon-absorption, optical-kerr, dc-kerr and four-wave-mixing.
Frequencies/damping use eV, or Hartree with units="atomic". The
result is the ordered molecular cubic dipole susceptibility.

[Source](../python/dft_rs/ase_calculator.py#L1226).

### `dft_rs.ase_calculator.DftRs.get_orbital_energy_gradients`

```python
DftRs.get_orbital_energy_gradients(self, atoms=None, *, coordinates=None, degeneracy_tolerance=1e-08, unrestricted=False, reference_field=None, response_tol=1e-10, response_max_iter=500)
```

Relaxed orbital gradients/projected degenerate blocks in atomic units.

[Source](../python/dft_rs/ase_calculator.py#L1240).

### `dft_rs.ase_calculator.DftRs.get_frontier_gap_gradient`

```python
DftRs.get_frontier_gap_gradient(self, atoms=None, *, coordinates=None, degeneracy_tolerance=1e-08, unrestricted=False, reference_field=None, response_tol=1e-10, response_max_iter=500)
```

Per-spin neutral-reference orbital gaps and Hartree/Bohr gradients.

[Source](../python/dft_rs/ase_calculator.py#L1251).

### `dft_rs.ase_calculator.DftRs.get_magnetic_response`

```python
DftRs.get_magnetic_response(self, atoms=None, *, gauge='giao', origin=None, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Spin orbital dP/dB in reduced atomic units; origin in Bohr.

[Source](../python/dft_rs/ase_calculator.py#L1277).

### `dft_rs.ase_calculator.DftRs.get_nmr`

```python
DftRs.get_nmr(self, atoms=None, *, gauge='giao', origin=None, response_tol=1e-10, response_max_iter=500, response_restart=40)
```

Orbital shieldings in ppm with raw residuals; origin in Bohr.

[Source](../python/dft_rs/ase_calculator.py#L1285).

### `dft_rs.ase_calculator.DftRs.get_nuclear_taylor`

```python
DftRs.get_nuclear_taylor(self, atoms=None, *, direction, order=4, response_tol=1e-10, response_max_iter=500)
```

Analytic nuclear energy coefficients and diagnostics in atomic units.

ASE positions are converted from Angstrom. Direction displacements
are explicitly in Bohr, as in the native response report.

[Source](../python/dft_rs/ase_calculator.py#L1293).

### `dft_rs.ase_calculator.DftRs.get_mixed_nuclear_response`

```python
DftRs.get_mixed_nuclear_response(self, atoms=None, *, directions, response_tol=1e-10, response_max_iter=500)
```

Mixed nuclear energy derivative; directions in Bohr, result in Hartree.

[Source](../python/dft_rs/ase_calculator.py#L1311).

### `dft_rs.ase_calculator.DftRs.get_molecular_taylor`

```python
DftRs.get_molecular_taylor(self, atoms=None, *, nuclear_direction, electric_direction, order=4, reference_field=None, response_tol=1e-10, response_max_iter=500)
```

Analytic molecular jets; nuclear paths in Bohr and fields in atomic units.

[Source](../python/dft_rs/ase_calculator.py#L1337).

### `dft_rs.ase_calculator.DftRs.get_mixed_molecular_response`

```python
DftRs.get_mixed_molecular_response(self, atoms=None, *, nuclear_directions, electric_directions, reference_field=None, response_tol=1e-10, response_max_iter=500)
```

Actual mixed energy/dipole derivatives with physical residuals.

[Source](../python/dft_rs/ase_calculator.py#L1348).

### `dft_rs.ase_calculator.DftRs.get_static_spectroscopic_derivative`

```python
DftRs.get_static_spectroscopic_derivative(self, atoms=None, *, nuclear_direction, electric_order=2, reference_field=None, response_tol=1e-10, response_max_iter=500)
```

Directional APT/Raman/hyper-Raman on the common molecular reference.

Integer RKS/UKS correlated targets include full orbital and PT2
amplitude response through energy order three/four. Nuclear paths
use Angstrom by default, Bohr in atomic mode. All ordered entries
and raw permutation/response diagnostics are retained. Parent
ROHF/ROKS and full complex GHF/GKS share the stationary energy
connection; their constrained PT2 and dynamic routes remain separate.

[Source](../python/dft_rs/ase_calculator.py#L1359).

### `dft_rs.ase_calculator.DftRs.get_analytic_force_constants`

```python
DftRs.get_analytic_force_constants(self, atoms=None, *, order=3, patterns=None, coordinates=None, reference_field=None, response_tol=1e-10, response_max_iter=500, **controls)
```

Complete analytic cubic/quartic Cartesian, block or pattern tensor.

Integer RKS/UKS double hybrids include full orbital/PT2 amplitude
response and separate target/SS/OS/dispersion tensors. Controls and
results follow the calculator unit mode and common report labels.

[Source](../python/dft_rs/ase_calculator.py#L1379).

### `dft_rs.ase_calculator.DftRs.get_anharmonic`

```python
DftRs.get_anharmonic(self, atoms=None, *, mode_ids=None, include_raman=False, equilibrium_tol=0.0001, resonance_tol=1e-08, resonance_coupling_factor=1.0, max_basis_states=1024, max_polyad_states=256, response_tol=1e-10, response_max_iter=500, method='retained-polyad')
```

Analytic rectilinear anharmonic bands, IR km/mol and Raman Angstrom^4/amu.

[Source](../python/dft_rs/ase_calculator.py#L1397).

### `dft_rs.ase_calculator.DftRs.get_vibrations`

```python
DftRs.get_vibrations(self, atoms=None)
```

Return frequencies (cm^-1), IR intensities, and projection metadata.

[Source](../python/dft_rs/ase_calculator.py#L1410).

### `dft_rs.ase_calculator.DftRs.get_feature_report`

```python
DftRs.get_feature_report(self, feature, atoms=None, **controls)
```

Run a registered driver calculation for ``atoms``.

[Source](../python/dft_rs/ase_calculator.py#L1489).

### `dft_rs.ase_calculator.DftRs.get_energy_report`

```python
DftRs.get_energy_report(self, atoms=None, **controls)
```

Total energy and components in Hartree.

[Source](../python/dft_rs/ase_calculator.py#L1515).

### `dft_rs.ase_calculator.DftRs.get_free_energy_report`

```python
DftRs.get_free_energy_report(self, atoms=None, **controls)
```

Compatibility getter for electronic Mermin free energy; see get_mermin_free_energy.

[Source](../python/dft_rs/ase_calculator.py#L1519).

### `dft_rs.ase_calculator.DftRs.get_mermin_free_energy`

```python
DftRs.get_mermin_free_energy(self, atoms=None, **controls)
```

Electronic A=E-T_e*S_e; electronic_temperature in K, result eV by default.

units="atomic" returns Hartree. Nuclear RRHO and pV are excluded.
ROKS and ROHF fix both spin counts. GKS and GHF fix the total count.
Set gradient=True for the analytic nuclear derivative with a
supported provider.
RKS and UKS also accept gibbs=True for equilibrium RRHO
thermochemistry on the same reference. Temperature is K and pressure is Pa.

[Source](../python/dft_rs/ase_calculator.py#L1523).

### `dft_rs.ase_calculator.DftRs.get_gibbs_free_energy`

```python
DftRs.get_gibbs_free_energy(self, atoms=None, **controls)
```

Ideal-gas RRHO G=H-T*S from the analytic Hessian, eV per molecule.

temperature is Kelvin, pressure is Pa; units="atomic" returns
Hartree. Uses the same harmonic model as get_frequencies_report.
Constrained ROHF/ROKS/GHF/GKS reuse their analytic Hessian provider.
electronic_degeneracy defaults to one for fixed spinors. Free-molecule
Eckart modes, positive internal frequencies and convergence are required;
provider restrictions and bounded acceptance remain in response.md.

[Source](../python/dft_rs/ase_calculator.py#L1535).

### `dft_rs.ase_calculator.DftRs.get_gradient_report`

```python
DftRs.get_gradient_report(self, atoms=None, **controls)
```

Analytic nuclear gradient in Hartree/Bohr.

[Source](../python/dft_rs/ase_calculator.py#L1547).

### `dft_rs.ase_calculator.DftRs.get_charges_report`

```python
DftRs.get_charges_report(self, atoms=None, **controls)
```

Mulliken/Loewdin charges; reference selects auto/rks/uks/roks/gks and HF aliases.

[Source](../python/dft_rs/ase_calculator.py#L1551).

### `dft_rs.ase_calculator.DftRs.get_natural_bond`

```python
DftRs.get_natural_bond(self, atoms=None, **controls)
```

Natural bond orbital analysis: natural charges and configurations,
Wiberg indices, the natural Lewis structure, NLMOs and second-order
donor-acceptor energies (eV in ASE mode, Hartree in atomic mode).
Optional natural-orbital controls include ``nbo_natural_orbitals``,
``nbo_spin_density`` and ``nbo_corresponding_orbitals``.
``nbo_spin_density_points`` uses Angstrom in ASE mode and Bohr in
atomic mode.

[Source](../python/dft_rs/ase_calculator.py#L1555).

### `dft_rs.ase_calculator.DftRs.get_igmh`

```python
DftRs.get_igmh(self, atoms=None, **controls)
```

Sample molecular IGMH fields on points or an affine grid.

Coordinates and step vectors use Angstrom in ASE mode and Bohr in
atomic mode. The report converts density and gradient units accordingly.

[Source](../python/dft_rs/ase_calculator.py#L1565).

### `dft_rs.ase_calculator.DftRs.get_periodic_natural_bond`

```python
DftRs.get_periodic_natural_bond(self, atoms=None, **controls)
```

Natural bond orbital analysis of the periodic cell on the
Born-von Karman supercell of the k mesh (energies in eV in ASE mode,
Hartree in atomic mode).

[Source](../python/dft_rs/ase_calculator.py#L1573).

### `dft_rs.ase_calculator.DftRs.get_natural_shielding`

```python
DftRs.get_natural_shielding(self, atoms=None, **controls)
```

Natural chemical shielding analysis (ppm) over natural localized
molecular orbitals.

[Source](../python/dft_rs/ase_calculator.py#L1579).

### `dft_rs.ase_calculator.DftRs.get_natural_coupling`

```python
DftRs.get_natural_coupling(self, atoms=None, **controls)
```

Natural J-coupling analysis of a closed-shell molecule over
natural localized molecular orbitals.

[Source](../python/dft_rs/ase_calculator.py#L1584).

### `dft_rs.ase_calculator.DftRs.get_natural_energy_decomposition`

```python
DftRs.get_natural_energy_decomposition(self, atoms=None, **controls)
```

Natural energy decomposition of fragment interactions
(``fragments`` per atom; eV in ASE mode, Hartree in atomic mode).

[Source](../python/dft_rs/ase_calculator.py#L1589).

### `dft_rs.ase_calculator.DftRs.get_sapt_analysis`

```python
DftRs.get_sapt_analysis(self, atoms=None, **controls)
```

Intermolecular perturbation analysis with `fragments` per atom.

`sapt_variant` accepts `"response"`, `"sapt0"`, `"sapt0-s2"`,
`"ssapt0"`, `"sapt-30"`, `"sapt2+3"` and `"sapt-ks-fdds"`. Energies use eV by default or Hartree
with `units="atomic"`. For optional `sapt_regularization_eta` with
`"sapt0"` or `"sapt-ks-fdds"`, use inverse Angstrom squared or inverse Bohr squared
in atomic mode. The optional `sapt_correlation_terms` and
`sapt_relaxed_induction30` reports follow the same energy units.

[Source](../python/dft_rs/ase_calculator.py#L1594).

### `dft_rs.ase_calculator.DftRs.get_dipole_report`

```python
DftRs.get_dipole_report(self, atoms=None, **controls)
```

Dipole in e*Bohr and Debye about the reported origin.

[Source](../python/dft_rs/ase_calculator.py#L1605).

### `dft_rs.ase_calculator.DftRs.get_orbitals_report`

```python
DftRs.get_orbitals_report(self, atoms=None, **controls)
```

Orbital energies and occupations; reference selects auto/rks/uks/roks/gks and HF aliases.

[Source](../python/dft_rs/ase_calculator.py#L1609).

### `dft_rs.ase_calculator.DftRs.write_molden`

```python
DftRs.write_molden(self, atoms=None, *, path, **controls)
```

Write a Molden file of the converged reference.

[Source](../python/dft_rs/ase_calculator.py#L1613).

### `dft_rs.ase_calculator.DftRs.write_orbitals`

```python
DftRs.write_orbitals(self, atoms=None, *, path, **controls)
```

Write the JSON orbital report.

[Source](../python/dft_rs/ase_calculator.py#L1617).

### `dft_rs.ase_calculator.DftRs.get_molecular_multipole`

```python
DftRs.get_molecular_multipole(self, atoms=None, **controls)
```

Cartesian, traceless and spherical multipoles (e*Bohr^rank).

[Source](../python/dft_rs/ase_calculator.py#L1621).

### `dft_rs.ase_calculator.DftRs.get_molecular_medium_optics`

```python
DftRs.get_molecular_medium_optics(self, atoms=None, **controls)
```

Molecular dielectric and polarization-resolved slab optics.

[Source](../python/dft_rs/ase_calculator.py#L1625).

### `dft_rs.ase_calculator.DftRs.get_quadrupole_polarizability`

```python
DftRs.get_quadrupole_polarizability(self, atoms=None, **controls)
```

Electric quadrupole response in five symmetric traceless gradient modes.

[Source](../python/dft_rs/ase_calculator.py#L1629).

### `dft_rs.ase_calculator.DftRs.get_multipole_polarizability`

```python
DftRs.get_multipole_polarizability(self, atoms=None, **controls)
```

Static or frequency-resolved Racah multipole response for rank 1 to 5.

[Source](../python/dft_rs/ase_calculator.py#L1633).

### `dft_rs.ase_calculator.DftRs.get_conceptual_dft`

```python
DftRs.get_conceptual_dft(self, atoms=None, **controls)
```

Conceptual DFT descriptors (Hartree-based).

[Source](../python/dft_rs/ase_calculator.py#L1637).

### `dft_rs.ase_calculator.DftRs.get_spin_conceptual_dft`

```python
DftRs.get_spin_conceptual_dft(self, atoms=None, **controls)
```

Spin-channel conceptual DFT descriptors.

[Source](../python/dft_rs/ase_calculator.py#L1641).

### `dft_rs.ase_calculator.DftRs.get_hessian_report`

```python
DftRs.get_hessian_report(self, atoms=None, **controls)
```

Analytic Hessian report in Hartree/Bohr^2 (atomic units).

Positive electronic_temperature selects the Mermin Hessian for
supported ROKS and GKS providers. Read the thermal response and
number diagnostics from the report.
Optional ``bfield`` and ``field`` select the complex London nuclear
response with fixed laboratory fields. Residuals are returned before
any intrinsic tensor permutation reuse; correlated PT2 and extra
Hamiltonian providers retain their documented restrictions.

[Source](../python/dft_rs/ase_calculator.py#L1645).

### `dft_rs.ase_calculator.DftRs.get_frequencies_report`

```python
DftRs.get_frequencies_report(self, atoms=None, **controls)
```

Frequencies, optional IR, normal modes and thermochemistry.

Constrained references reuse their analytic Hessian provider and raw
residuals. vibrational_projection=auto retains all Cartesian modes
under a fixed laboratory source; eckart/cartesian override projection.
IR is null without an atomic polar tensor. electronic_degeneracy
defaults to one for fixed spinors. See response.md for the precise
correlated-reference and bounded acceptance scope.

[Source](../python/dft_rs/ase_calculator.py#L1658).

### `dft_rs.ase_calculator.DftRs.get_vibrational_spectra`

```python
DftRs.get_vibrational_spectra(self, atoms=None, **controls)
```

Analytic IR intensities and Raman activities.

[Source](../python/dft_rs/ase_calculator.py#L1670).

### `dft_rs.ase_calculator.DftRs.get_polarizability_report`

```python
DftRs.get_polarizability_report(self, atoms=None, **controls)
```

Static polarizability report (Bohr^3 and Angstrom^3).

[Source](../python/dft_rs/ase_calculator.py#L1674).

### `dft_rs.ase_calculator.DftRs.get_multipole_gradient`

```python
DftRs.get_multipole_gradient(self, atoms=None, **controls)
```

Analytic multipole nuclear derivatives.

[Source](../python/dft_rs/ase_calculator.py#L1678).

### `dft_rs.ase_calculator.DftRs.get_response_derivatives`

```python
DftRs.get_response_derivatives(self, atoms=None, **controls)
```

U^F, dP/dF and dP/dR in atomic units (AO basis).

[Source](../python/dft_rs/ase_calculator.py#L1682).

### `dft_rs.ase_calculator.DftRs.get_third_derivative_2n1`

```python
DftRs.get_third_derivative_2n1(self, atoms=None, **controls)
```

Cartesian third derivatives of the energy by the 2n+1 rule.

[Source](../python/dft_rs/ase_calculator.py#L1688).

### `dft_rs.ase_calculator.DftRs.get_cubic_fc_along`

```python
DftRs.get_cubic_fc_along(self, atoms=None, **controls)
```

Cubic force constants contracted with a displacement pattern.

Optional ``bfield``/``field`` use complex London nuclear response.

[Source](../python/dft_rs/ase_calculator.py#L1692).

### `dft_rs.ase_calculator.DftRs.get_quartic_fc_along`

```python
DftRs.get_quartic_fc_along(self, atoms=None, **controls)
```

Quartic force constants contracted with two displacement patterns.

Optional ``bfield``/``field`` use complex London nuclear response.

[Source](../python/dft_rs/ase_calculator.py#L1699).

### `dft_rs.ase_calculator.DftRs.get_streamed_gradient`

```python
DftRs.get_streamed_gradient(self, atoms=None, **controls)
```

Analytic gradient without a stored integral tensor.

[Source](../python/dft_rs/ase_calculator.py#L1706).

### `dft_rs.ase_calculator.DftRs.get_excitations`

```python
DftRs.get_excitations(self, atoms=None, **controls)
```

Collinear TDA/RPA, common-orbital reference-spin TDA or
full-spin GKS TDA poles. Spinor states have no exact total-spin label.
Request TDA transition orbitals with ``td_nto=True``.
``td_difference_density=True`` gives unrelaxed attachment and
detachment orbitals for RKS, UKS and GKS.

[Source](../python/dft_rs/ase_calculator.py#L1710).

### `dft_rs.ase_calculator.DftRs.get_ecd_spectrum`

```python
DftRs.get_ecd_spectrum(self, atoms=None, **controls)
```

Broadened absorption and circular-dichroism spectra.

[Source](../python/dft_rs/ase_calculator.py#L1718).

### `dft_rs.ase_calculator.DftRs.get_td_state_gradient`

```python
DftRs.get_td_state_gradient(self, atoms=None, **controls)
```

Analytic total excited-state gradient (Hartree/Bohr).
HF common-orbital doublet/quartet and full-spin HF subsets are accepted.
Moving semilocal spinor verification and correlated pole derivatives remain pending.

[Source](../python/dft_rs/ase_calculator.py#L1722).

### `dft_rs.ase_calculator.DftRs.get_hyperfine`

```python
DftRs.get_hyperfine(self, atoms=None, **controls)
```

Static contact, spin-dipolar and total hyperfine tensors.

MHz/per-nuclear-g and optional isotope-scaled MHz/gauss labels are
preserved. Native reference/model restrictions apply, including the
explicit single-Kramers-pair approximation for GKS/GHF. The default
report is static; nuclear_order=1/2 selects the separate bounded
HF-collinear nuclear-A adapter. This is not vibrational averaging.

[Source](../python/dft_rs/ase_calculator.py#L1728).

### `dft_rs.ase_calculator.DftRs.get_nmr_shielding`

```python
DftRs.get_nmr_shielding(self, atoms=None, **controls)
```

Nuclear magnetic shielding tensors (ppm).

[Source](../python/dft_rs/ase_calculator.py#L1739).

### `dft_rs.ase_calculator.DftRs.get_nmr_spectrum`

```python
DftRs.get_nmr_spectrum(self, atoms=None, **controls)
```

Isotropic NMR spectrum (ppm); nmr_spin_model="exact" retains scalar spin mixing.
Uses the shared native controls, transition areas and spin residuals.

[Source](../python/dft_rs/ase_calculator.py#L1743).

### `dft_rs.ase_calculator.DftRs.get_spin_spin_couplings`

```python
DftRs.get_spin_spin_couplings(self, atoms=None, **controls)
```

Ramsey tensors, including UKS/GKS full-spin mixed terms.
Selects free response by default; fixed-spin-direction clamps orientation.
ROKS and open-shell PT2 remain separate providers.

[Source](../python/dft_rs/ase_calculator.py#L1749).

### `dft_rs.ase_calculator.DftRs.get_fermi_contact_couplings`

```python
DftRs.get_fermi_contact_couplings(self, atoms=None, **controls)
```

Fermi-contact part of the indirect nuclear spin-spin couplings.

[Source](../python/dft_rs/ase_calculator.py#L1756).

### `dft_rs.ase_calculator.DftRs.get_exafs`

```python
DftRs.get_exafs(self, atoms=None, **controls)
```

Single-scattering EXAFS from tabulated scattering functions.

[Source](../python/dft_rs/ase_calculator.py#L1760).

### `dft_rs.ase_calculator.DftRs.get_xanes`

```python
DftRs.get_xanes(self, atoms=None, **controls)
```

Core-level absorption from core-valence-separated TDA.

[Source](../python/dft_rs/ase_calculator.py#L1764).

### `dft_rs.ase_calculator.DftRs.get_nexafs`

```python
DftRs.get_nexafs(self, atoms=None, **controls)
```

Polarized molecular near-edge absorption.

[Source](../python/dft_rs/ase_calculator.py#L1768).

### `dft_rs.ase_calculator.DftRs.get_constrained_dft`

```python
DftRs.get_constrained_dft(self, atoms=None, **controls)
```

AO population or spin constraints, with optional forces, relaxation
and static electric response.

[Source](../python/dft_rs/ase_calculator.py#L1772).

### `dft_rs.ase_calculator.DftRs.get_dft_plus_u`

```python
DftRs.get_dft_plus_u(self, atoms=None, **controls)
```

Molecular DFT+U. Nested Hubbard site parameters are always in Hartree.

[Source](../python/dft_rs/ase_calculator.py#L1777).

### `dft_rs.ase_calculator.DftRs.get_pz_sic`

```python
DftRs.get_pz_sic(self, atoms=None, **controls)
```

Perdew-Zunger SIC with occupied orbital optimization.

[Source](../python/dft_rs/ase_calculator.py#L1781).

### `dft_rs.ase_calculator.DftRs.get_flo_sic_fixed`

```python
DftRs.get_flo_sic_fixed(self, atoms=None, **controls)
```

Return frozen UKS Fermi-Lowdin SIC with fixed descriptors in Bohr.

[Source](../python/dft_rs/ase_calculator.py#L1785).

### `dft_rs.ase_calculator.DftRs.get_car_parrinello_md`

```python
DftRs.get_car_parrinello_md(self, atoms=None, **controls)
```

Microcanonical molecular extended-Lagrangian trajectory.

[Source](../python/dft_rs/ase_calculator.py#L1789).

### `dft_rs.ase_calculator.DftRs.get_magnetic_quadrupole`

```python
DftRs.get_magnetic_quadrupole(self, atoms=None, **controls)
```

Orbital and Pauli magnetic quadrupole.

[Source](../python/dft_rs/ase_calculator.py#L1793).

### `dft_rs.ase_calculator.DftRs.get_magnetizability`

```python
DftRs.get_magnetizability(self, atoms=None, **controls)
```

Magnetizability tensor with a common gauge origin.

[Source](../python/dft_rs/ase_calculator.py#L1797).

### `dft_rs.ase_calculator.DftRs.get_franck_condon`

```python
DftRs.get_franck_condon(self, atoms=None, **controls)
```

Franck-Condon progression in the vertical-gradient model.

[Source](../python/dft_rs/ase_calculator.py#L1801).

### `dft_rs.ase_calculator.DftRs.get_excited_optimization`

```python
DftRs.get_excited_optimization(self, atoms=None, **controls)
```

Relax a supported excited surface with its analytic gradient.
``td_gradient_route="path"`` enables compatible paths; general spinor
optimization and robust crossing/subspace tracking remain pending.

[Source](../python/dft_rs/ase_calculator.py#L1805).

### `dft_rs.ase_calculator.DftRs.get_optical_activity`

```python
DftRs.get_optical_activity(self, atoms=None, **controls)
```

Optical rotation tensors and specific rotation.

[Source](../python/dft_rs/ase_calculator.py#L1811).

### `dft_rs.ase_calculator.DftRs.get_broken_symmetry_exchange`

```python
DftRs.get_broken_symmetry_exchange(self, atoms=None, **controls)
```

Heisenberg exchange coupling from a broken-symmetry determinant.

[Source](../python/dft_rs/ase_calculator.py#L1815).

### `dft_rs.ase_calculator.DftRs.get_vcd`

```python
DftRs.get_vcd(self, atoms=None, **controls)
```

Vibrational circular dichroism rotational strengths.

[Source](../python/dft_rs/ase_calculator.py#L1819).

### `dft_rs.ase_calculator.DftRs.get_spin_susceptibility`

```python
DftRs.get_spin_susceptibility(self, atoms=None, **controls)
```

Static spin susceptibility of a Fermi-Dirac ensemble.

[Source](../python/dft_rs/ase_calculator.py#L1823).

### `dft_rs.ase_calculator.DftRs.get_spin_spin_zfs`

```python
DftRs.get_spin_spin_zfs(self, atoms=None, *, nuclear_order=0, **controls)
```

Direct spin-spin D and optional analytic HF/UHF nuclear derivatives.

Requires multiplicity >= 3; SOC and spin purification are excluded.
Native units are Hartree/Bohr; ASE mode converts D, its gradient and
Hessian to eV, eV/Angstrom and eV/Angstrom^2. The atom-major xyz
ordering and raw diagnostic labels are preserved. Unrequested
derivatives are None. See native.spin_spin_zfs for model restrictions.

[Source](../python/dft_rs/ase_calculator.py#L1827).

### `dft_rs.ase_calculator.DftRs.get_g_tensor`

```python
DftRs.get_g_tensor(self, atoms=None, **controls)
```

Static electronic g tensor (dimensionless), with ppm shift terms.

Uses the native open-shell reference/model restrictions. ROKS/ROHF
have a dedicated common-origin route; GKS/GHF require an explicit
single-Kramers-pair approximation. ``nuclear_order=1/2`` is a
separate integer all-electron HF/UHF London route, with Cartesian
gradient/Hessian shapes [3N,3,3]/[3N,3N,3,3]; the Hessian is null
for order one. Default ASE units are inverse Angstrom and inverse
Angstrom squared; atomic mode retains inverse Bohr units. DFT,
RO/spinor nuclear response, nonzero electric field and stability
following remain unsupported. This is not experimental-spectrum
validation or a complete many-electron SOC model.

[Source](../python/dft_rs/ase_calculator.py#L1838).

### `dft_rs.ase_calculator.DftRs.get_roks`

```python
DftRs.get_roks(self, atoms=None, **controls)
```

Restricted open-shell Kohn-Sham energy and gradient.

[Source](../python/dft_rs/ase_calculator.py#L1854).

### `dft_rs.ase_calculator.DftRs.get_roks_hessian`

```python
DftRs.get_roks_hessian(self, atoms=None, **controls)
```

Analytic Hessian and harmonic frequencies of the restricted
open-shell energy.

[Source](../python/dft_rs/ase_calculator.py#L1858).

### `dft_rs.ase_calculator.DftRs.get_roks_polarizability`

```python
DftRs.get_roks_polarizability(self, atoms=None, **controls)
```

Dipole moment and static polarizability of the restricted
open-shell reference.

[Source](../python/dft_rs/ase_calculator.py#L1863).

### `dft_rs.ase_calculator.DftRs.get_divide_conquer`

```python
DftRs.get_divide_conquer(self, atoms=None, **controls)
```

Divide-and-conquer Kohn-Sham energy and gradient.

[Source](../python/dft_rs/ase_calculator.py#L1868).

### `dft_rs.ase_calculator.DftRs.get_mrsf`

```python
DftRs.get_mrsf(self, atoms=None, **controls)
```

Mixed-reference spin-flip singlets and triplets.

[Source](../python/dft_rs/ase_calculator.py#L1872).

### `dft_rs.ase_calculator.DftRs.get_mrsf_state`

```python
DftRs.get_mrsf_state(self, atoms=None, **controls)
```

Analytic gradient of one mixed-reference spin-flip state.

[Source](../python/dft_rs/ase_calculator.py#L1876).

### `dft_rs.ase_calculator.DftRs.get_mrsf_hessian`

```python
DftRs.get_mrsf_hessian(self, atoms=None, **controls)
```

Analytic Hessian and harmonic frequencies of one mixed-reference
spin-flip state.

[Source](../python/dft_rs/ase_calculator.py#L1880).

### `dft_rs.ase_calculator.DftRs.get_spin_orbit`

```python
DftRs.get_spin_orbit(self, atoms=None, **controls)
```

One-electron spin-orbit coupling at the selected molecular reference.

With reference="gks", the generated operator enters the
self-consistent spin-covariant GKS equations.

[Source](../python/dft_rs/ase_calculator.py#L1885).

### `dft_rs.ase_calculator.DftRs.get_spinor_polarizability`

```python
DftRs.get_spinor_polarizability(self, atoms=None, *, frequencies, **controls)
```

Dynamic polarizability of the self-consistent spin-orbit spinor
state at ``frequencies``.

[Source](../python/dft_rs/ase_calculator.py#L1893).

### `dft_rs.ase_calculator.DftRs.get_magnetic_anisotropy`

```python
DftRs.get_magnetic_anisotropy(self, atoms=None, **controls)
```

Magnetic anisotropy energies of several spin axes.

[Source](../python/dft_rs/ase_calculator.py#L1898).

### `dft_rs.ase_calculator.DftRs.get_magnon`

```python
DftRs.get_magnon(self, atoms=None, **controls)
```

Transverse spin excitation energies (spin-flip TDA).

[Source](../python/dft_rs/ase_calculator.py#L1902).

### `dft_rs.ase_calculator.DftRs.get_mcd`

```python
DftRs.get_mcd(self, atoms=None, **controls)
```

Faraday A and B terms of magnetic circular dichroism.

[Source](../python/dft_rs/ase_calculator.py#L1906).

### `dft_rs.ase_calculator.DftRs.get_td_hessian`

```python
DftRs.get_td_hessian(self, atoms=None, **controls)
```

Analytic isolated-state excited Hessian (Hartree/Bohr squared).
HF flag doublet/quartet and full-spin HF subsets have complete Cartesian
acceptance. Their reports contain tensors without harmonic postprocessing;
semilocal spinor verification and correlated pole derivatives remain pending.

[Source](../python/dft_rs/ase_calculator.py#L1910).

### `dft_rs.ase_calculator.DftRs.get_td_ecd_giao`

```python
DftRs.get_td_ecd_giao(self, atoms=None, **controls)
```

Rotatory strengths with London orbitals in the length gauge.

[Source](../python/dft_rs/ase_calculator.py#L1917).

### `dft_rs.ase_calculator.DftRs.get_roa`

```python
DftRs.get_roa(self, atoms=None, **controls)
```

Raman optical activity invariants and circular intensity differences.

[Source](../python/dft_rs/ase_calculator.py#L1921).

### `dft_rs.ase_calculator.DftRs.get_hypermagnetizability`

```python
DftRs.get_hypermagnetizability(self, atoms=None, **controls)
```

Static hypermagnetizability, polarizability and magnetizability.

[Source](../python/dft_rs/ase_calculator.py#L1925).

### `dft_rs.ase_calculator.DftRs.get_cotton_mouton`

```python
DftRs.get_cotton_mouton(self, atoms=None, **controls)
```

Cotton-Mouton constant per molecule at a temperature.

[Source](../python/dft_rs/ase_calculator.py#L1929).

### `dft_rs.ase_calculator.DftRs.get_magneto_optical`

```python
DftRs.get_magneto_optical(self, atoms=None, **controls)
```

Magneto-optical tensor at a photon energy (spin-orbit or external field).

[Source](../python/dft_rs/ase_calculator.py#L1933).

### `dft_rs.ase_calculator.DftRs.get_magnetic_field_scf`

```python
DftRs.get_magnetic_field_scf(self, atoms=None, **controls)
```

Self-consistent field in a finite uniform magnetic field.

[Source](../python/dft_rs/ase_calculator.py#L1937).

### `dft_rs.ase_calculator.DftRs.get_magnetic_field_gradient`

```python
DftRs.get_magnetic_field_gradient(self, atoms=None, **controls)
```

Energy and analytic gradient in a finite uniform magnetic field.

[Source](../python/dft_rs/ase_calculator.py#L1941).

### `dft_rs.ase_calculator.DftRs.get_magnetoelectric`

```python
DftRs.get_magnetoelectric(self, atoms=None, **controls)
```

Linear magnetoelectric response to first order in spin-orbit coupling.

[Source](../python/dft_rs/ase_calculator.py#L1945).

### `dft_rs.ase_calculator.DftRs.get_periodic_energy_report`

```python
DftRs.get_periodic_energy_report(self, atoms=None, **controls)
```

Periodic energy per cell (Hartree) and optional forces (Hartree/Bohr).

[Source](../python/dft_rs/ase_calculator.py#L2003).

### `dft_rs.ase_calculator.DftRs.get_periodic_stress_report`

```python
DftRs.get_periodic_stress_report(self, atoms=None, **controls)
```

Affine stress normalized by the active periodic measure.

Collinear commensurate Fourier fields keep reduced wave indices
and Cartesian amplitudes fixed in the derivative.

[Source](../python/dft_rs/ase_calculator.py#L2007).

### `dft_rs.ase_calculator.DftRs.get_opt_fixed_cell`

```python
DftRs.get_opt_fixed_cell(self, atoms=None, **controls)
```

Fixed-cell relaxation report (Bohr positions, Hartree energies).

[Source](../python/dft_rs/ase_calculator.py#L2015).

### `dft_rs.ase_calculator.DftRs.get_cell_opt`

```python
DftRs.get_cell_opt(self, atoms=None, **controls)
```

Variable-cell relaxation report: relaxed ``cell`` rows and
``positions`` in Bohr, enthalpy in Hartree, stress in Hartree/Bohr^3.

[Source](../python/dft_rs/ase_calculator.py#L2019).

### `dft_rs.ase_calculator.DftRs.get_relaxed_atoms`

```python
DftRs.get_relaxed_atoms(self, atoms=None, *, cell=False, **controls)
```

Relax the atoms (``cell=False``) or atoms and cell (``cell=True``)
and return a new ``Atoms`` in Angstrom; the optimization report (ASE
units) is stored in ``relaxed.info["dft_rs_report"]``.

[Source](../python/dft_rs/ase_calculator.py#L2024).

### `dft_rs.ase_calculator.DftRs.get_phonons`

```python
DftRs.get_phonons(self, atoms=None, **controls)
```

Gamma-point phonons from analytic force constants (cm^-1).
Optional field selects the stationary finite-bias Berry enthalpy;
it is eV/Angstrom by default, atomic field units with units='atomic'.

[Source](../python/dft_rs/ase_calculator.py#L2041).

### `dft_rs.ase_calculator.DftRs.get_phonon_bands`

```python
DftRs.get_phonon_bands(self, atoms=None, *, supercell, path, **controls)
```

Phonon dispersion from analytic supercell force constants (cm^-1).
lattice_flexoelectric adds fixed-mass-center internal relaxation and
the Born-charge lattice contribution (e/Angstrom by default), with
order-dependent coefficient units. See native.phonon_bands for scope.

[Source](../python/dft_rs/ase_calculator.py#L2047).

### `dft_rs.ase_calculator.DftRs.get_phonon_thermo`

```python
DftRs.get_phonon_thermo(self, atoms=None, *, supercell, q_mesh, **controls)
```

Harmonic lattice thermodynamics per primitive cell (Hartree, K).

[Source](../python/dft_rs/ase_calculator.py#L2055).

### `dft_rs.ase_calculator.DftRs.get_electron_phonon`

```python
DftRs.get_electron_phonon(self, atoms=None, **controls)
```

Electron-phonon coupling, a2F and critical temperatures (K):
McMillan, Allen-Dynes and isotropic Eliashberg (``tc_eliashberg_k``), and the
nonlinear gap with Pade continuation for ``gap_temperatures_k``.
``scdft_pairing`` adds Gamma-vertex inputs and static K/Z. For generated
Coulomb screening use ``scdft_screening="rpa"`` or ``"rpa-neutral"``.
Default units are eV, eV squared for squared vertices and inverse Angstrom.

[Source](../python/dft_rs/ase_calculator.py#L2059).

### `dft_rs.ase_calculator.DftRs.get_quasi_harmonic`

```python
DftRs.get_quasi_harmonic(self, atoms=None, *, supercell, q_mesh, **controls)
```

Quasi-harmonic thermodynamics on scaled cells (Bohr^3, GPa, 1/K).

[Source](../python/dft_rs/ase_calculator.py#L2069).

### `dft_rs.ase_calculator.DftRs.get_elastic`

```python
DftRs.get_elastic(self, atoms=None, **controls)
```

Clamped- and relaxed-ion elastic constants (Voigt, GPa).

[Source](../python/dft_rs/ase_calculator.py#L2073).

### `dft_rs.ase_calculator.DftRs.get_elastic_fd`

```python
DftRs.get_elastic_fd(self, atoms=None, **controls)
```

Validation route: elastic constants from stress differences (GPa).

[Source](../python/dft_rs/ase_calculator.py#L2077).

### `dft_rs.ase_calculator.DftRs.get_elastic_moduli`

```python
DftRs.get_elastic_moduli(self, atoms=None, **controls)
```

Voigt, Reuss and Hill moduli (GPa), wave speeds and Debye temperature.

[Source](../python/dft_rs/ase_calculator.py#L2081).

### `dft_rs.ase_calculator.DftRs.get_christoffel`

```python
DftRs.get_christoffel(self, atoms=None, **controls)
```

Christoffel phase velocities (m/s) along ``directions``.

[Source](../python/dft_rs/ase_calculator.py#L2085).

### `dft_rs.ase_calculator.DftRs.get_material_card`

```python
DftRs.get_material_card(self, atoms=None, **controls)
```

Continuum material card (GPa, kg/m^3, m/s); ``card_path`` writes it.

[Source](../python/dft_rs/ase_calculator.py#L2089).

### `dft_rs.ase_calculator.DftRs.get_equation_of_state`

```python
DftRs.get_equation_of_state(self, atoms=None, **controls)
```

Birch-Murnaghan equation of state (V0 in Bohr^3, B0 in GPa).

[Source](../python/dft_rs/ase_calculator.py#L2093).

### `dft_rs.ase_calculator.DftRs.get_dielectric_tensor`

```python
DftRs.get_dielectric_tensor(self, atoms=None, **controls)
```

Electronic dielectric tensor by periodic DFPT.

[Source](../python/dft_rs/ase_calculator.py#L2097).

### `dft_rs.ase_calculator.DftRs.get_born_charges`

```python
DftRs.get_born_charges(self, atoms=None, **controls)
```

Born effective charges (e) and the electronic dielectric tensor.

[Source](../python/dft_rs/ase_calculator.py#L2101).

### `dft_rs.ase_calculator.DftRs.get_piezoelectric_tensor`

```python
DftRs.get_piezoelectric_tensor(self, atoms=None, **controls)
```

Proper clamped- and relaxed-ion piezoelectric tensors.

[Source](../python/dft_rs/ase_calculator.py#L2105).

### `dft_rs.ase_calculator.DftRs.get_berry_polarization`

```python
DftRs.get_berry_polarization(self, atoms=None, **controls)
```

Berry-phase polarization (modulo the polarization quanta).

[Source](../python/dft_rs/ase_calculator.py#L2109).

### `dft_rs.ase_calculator.DftRs.get_crystal_ir`

```python
DftRs.get_crystal_ir(self, atoms=None, **controls)
```

Crystal infrared intensities and LO-TO splitting (cm^-1, km/mol).

[Source](../python/dft_rs/ase_calculator.py#L2113).

### `dft_rs.ase_calculator.DftRs.get_static_dielectric`

```python
DftRs.get_static_dielectric(self, atoms=None, **controls)
```

Static (relaxed-ion) dielectric tensor.

[Source](../python/dft_rs/ase_calculator.py#L2117).

### `dft_rs.ase_calculator.DftRs.get_electronic_structure`

```python
DftRs.get_electronic_structure(self, atoms=None, **controls)
```

SCF bands and populations; coefficients=True adds complex Bloch orbitals and F/S/P matrices.

[Source](../python/dft_rs/ase_calculator.py#L2121).

### `dft_rs.ase_calculator.DftRs.get_effective_mass`

```python
DftRs.get_effective_mass(self, atoms=None, *, k_point, band, **controls)
```

Band velocity and effective-mass tensor (atomic units, m_e).

[Source](../python/dft_rs/ase_calculator.py#L2125).

### `dft_rs.ase_calculator.DftRs.get_boltzmann_transport`

```python
DftRs.get_boltzmann_transport(self, atoms=None, *, transport_k_mesh, transport_temperatures_k, **controls)
```

Boltzmann transport in SI; optional scattering_table includes tau(E).

[Source](../python/dft_rs/ase_calculator.py#L2129).

### `dft_rs.ase_calculator.DftRs.get_hall_transport`

```python
DftRs.get_hall_transport(self, atoms=None, *, transport_k_mesh, transport_temperatures_k, **controls)
```

Ordinary weak-field Hall tensors in SI units for isolated rigid
bands. Optional scattering_table includes energy-dependent lifetime.
Axes: electric/current and magnetic x,y,z.

[Source](../python/dft_rs/ase_calculator.py#L2139).

### `dft_rs.ase_calculator.DftRs.get_coherent_transport`

```python
DftRs.get_coherent_transport(self, atoms=None, *, transport_k_mesh, transport_frequencies_hartree, **controls)
```

Covariant intra/interband conductivity (S/m), including degeneracy.

[Source](../python/dft_rs/ase_calculator.py#L2148).

### `dft_rs.ase_calculator.DftRs.get_quantum_hall`

```python
DftRs.get_quantum_hall(self, atoms=None, **controls)
```

Non-Abelian occupied-slice Chern invariant and Hall conductance (S).

[Source](../python/dft_rs/ase_calculator.py#L2154).

### `dft_rs.ase_calculator.DftRs.get_wannier_flow`

```python
DftRs.get_wannier_flow(self, atoms=None, **controls)
```

Hybrid Wannier charge centres and their flow.

[Source](../python/dft_rs/ase_calculator.py#L2158).

### `dft_rs.ase_calculator.DftRs.get_periodic_cubic_fc_along`

```python
DftRs.get_periodic_cubic_fc_along(self, atoms=None, **controls)
```

Cubic force constants contracted with a Gamma displacement pattern.
Optional field selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/ase_calculator.py#L2162).

### `dft_rs.ase_calculator.DftRs.get_periodic_quartic_fc_along`

```python
DftRs.get_periodic_quartic_fc_along(self, atoms=None, **controls)
```

Quartic force constants contracted with two Gamma displacement patterns.
Optional field selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/ase_calculator.py#L2167).

### `dft_rs.ase_calculator.DftRs.get_periodic_cubic_fc`

```python
DftRs.get_periodic_cubic_fc(self, atoms=None, **controls)
```

Cubic force constants for Gamma-point displacements.
Optional field selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/ase_calculator.py#L2172).

### `dft_rs.ase_calculator.DftRs.get_periodic_quartic_fc`

```python
DftRs.get_periodic_quartic_fc(self, atoms=None, **controls)
```

Quartic force constants for Gamma-point displacements.
Optional field selects the stationary finite-bias Berry enthalpy.

[Source](../python/dft_rs/ase_calculator.py#L2177).

### `dft_rs.ase_calculator.DftRs.get_toec`

```python
DftRs.get_toec(self, atoms=None, **controls)
```

Third-order Lagrangian elastic constants at clamped and relaxed ions.

[Source](../python/dft_rs/ase_calculator.py#L2182).

### `dft_rs.ase_calculator.DftRs.get_foec`

```python
DftRs.get_foec(self, atoms=None, **controls)
```

Fourth-order Lagrangian elastic constants at clamped and relaxed ions.

[Source](../python/dft_rs/ase_calculator.py#L2186).

### `dft_rs.ase_calculator.DftRs.get_mode_gruneisen`

```python
DftRs.get_mode_gruneisen(self, atoms=None, **controls)
```

Gamma phonon Gruneisen response; order two uses log-volume curvature.

Set ``gruneisen_order=2`` for ``-d gamma/d ln(V)``. Acoustic and
unstable coefficients are undefined. See ``mode_gruneisen`` for
clamped/relaxed-ion and stationarity controls.

[Source](../python/dft_rs/ase_calculator.py#L2190).

### `dft_rs.ase_calculator.DftRs.get_periodic_beta`

```python
DftRs.get_periodic_beta(self, atoms=None, **controls)
```

Electric susceptibility of a cell and its field derivative.

[Source](../python/dft_rs/ase_calculator.py#L2199).

### `dft_rs.ase_calculator.DftRs.get_periodic_raman`

```python
DftRs.get_periodic_raman(self, atoms=None, **controls)
```

Born charges and the displacement derivative of the susceptibility.

[Source](../python/dft_rs/ase_calculator.py#L2203).

### `dft_rs.ase_calculator.DftRs.get_periodic_gamma`

```python
DftRs.get_periodic_gamma(self, atoms=None, **controls)
```

Static clamped-ion third-order electric susceptibility of a cell.

[Source](../python/dft_rs/ase_calculator.py#L2207).

### `dft_rs.ase_calculator.DftRs.get_electrostriction`

```python
DftRs.get_electrostriction(self, atoms=None, **controls)
```

Strain derivative of the electric susceptibility of a cell.

[Source](../python/dft_rs/ase_calculator.py#L2211).

### `dft_rs.ase_calculator.DftRs.get_piezo_higher`

```python
DftRs.get_piezo_higher(self, atoms=None, **controls)
```

Piezoelectric tensor of a cell and its strain derivative.

[Source](../python/dft_rs/ase_calculator.py#L2215).

### `dft_rs.ase_calculator.DftRs.get_periodic_efield`

```python
DftRs.get_periodic_efield(self, atoms=None, **controls)
```

Self-consistent field of a cell in a finite uniform electric field.

[Source](../python/dft_rs/ase_calculator.py#L2219).

### `dft_rs.ase_calculator.DftRs.get_dielectric_q`

```python
DftRs.get_dielectric_q(self, atoms=None, **controls)
```

Density response and inverse dielectric function at commensurate wave vectors.

[Source](../python/dft_rs/ase_calculator.py#L2223).

### `dft_rs.ase_calculator.DftRs.get_orbital_susceptibility_q`

```python
DftRs.get_orbital_susceptibility_q(self, atoms=None, **controls)
```

Orbital magnetic susceptibility at commensurate wave vectors.

[Source](../python/dft_rs/ase_calculator.py#L2227).

### `dft_rs.ase_calculator.DftRs.get_periodic_dc`

```python
DftRs.get_periodic_dc(self, atoms=None, **controls)
```

Divide-and-conquer Kohn-Sham energy and forces of a cell at Gamma.

[Source](../python/dft_rs/ase_calculator.py#L2231).

### `dft_rs.ase_calculator.DftRs.get_dc_stress`

```python
DftRs.get_dc_stress(self, atoms=None, **controls)
```

Stress of the divide-and-conquer free energy of a cell.

[Source](../python/dft_rs/ase_calculator.py#L2235).

### `dft_rs.ase_calculator.DftRs.get_periodic_nmr_shielding`

```python
DftRs.get_periodic_nmr_shielding(self, atoms=None, **controls)
```

Nuclear magnetic shielding in a periodic cell at commensurate wave vectors.

[Source](../python/dft_rs/ase_calculator.py#L2239).

### `dft_rs.ase_calculator.DftRs.get_berry_dipole`

```python
DftRs.get_berry_dipole(self, atoms=None, **controls)
```

Berry curvature dipole of the Fermi surface.

[Source](../python/dft_rs/ase_calculator.py#L2243).

### `dft_rs.ase_calculator.DftRs.get_lab_frame_spin_spiral`

```python
DftRs.get_lab_frame_spin_spiral(self, atoms=None, *, spiral_q, **controls)
```

Energy of a commensurate flat spin spiral (two-component SCF in
its supercell); ``spiral_q`` in reduced reciprocal coordinates.

[Source](../python/dft_rs/ase_calculator.py#L2247).

### `dft_rs.ase_calculator.DftRs.get_dmi`

```python
DftRs.get_dmi(self, atoms=None, *, spiral_q, **controls)
```

Chirality-odd spin-orbit energy of a commensurate spin spiral.

[Source](../python/dft_rs/ase_calculator.py#L2252).

### `dft_rs.ase_calculator.DftRs.get_bands`

```python
DftRs.get_bands(self, atoms=None, *, path, **controls)
```

Band structure along reduced k vertices ``path`` (Hartree).

[Source](../python/dft_rs/ase_calculator.py#L2256).

### `dft_rs.ase_calculator.DftRs.get_periodic_spin_response`

```python
DftRs.get_periodic_spin_response(self, atoms=None, **controls)
```

Longitudinal spin-only finite-q/frequency Zeeman response per cell.

[Source](../python/dft_rs/ase_calculator.py#L2260).

### `dft_rs.ase_calculator.DftRs.get_periodic_field_scf`

```python
DftRs.get_periodic_field_scf(self, atoms=None, **controls)
```

Complex finite periodic-field SCF and optional force and stress reports.

[Source](../python/dft_rs/ase_calculator.py#L2264).

### `dft_rs.ase_calculator.DftRs.get_periodic_spinor_scf`

```python
DftRs.get_periodic_spinor_scf(self, atoms=None, **controls)
```

Complex spinor SCF with total-number occupations and raw residuals.

[Source](../python/dft_rs/ase_calculator.py#L2268).

### `dft_rs.ase_calculator.DftRs.get_periodic_magnetic_scf`

```python
DftRs.get_periodic_magnetic_scf(self, atoms=None, **controls)
```

Uniform-flux London reference on a physical 2D/3D magnetic cell.

magnetic_model selects finite_gamma or crystal_bz. Wires use the
separate periodic spinor reference with uniform_magnetic_field.
GGA and meta-GGA SCF reports include angular and spatial XC defects.
nuclear_order=1..4 selects the relaxed finite_gamma exchange/SOC
nuclear provider. nuclear_direction selects a scalar path; otherwise
gradient/Hessian or probe-contracted cubic/quartic tensors are returned
in nuclear_response. Input directions/probes are dimensionless.

[Source](../python/dft_rs/ase_calculator.py#L2272).

### `dft_rs.ase_calculator.DftRs.get_dos`

```python
DftRs.get_dos(self, atoms=None, *, sigma, **controls)
```

Density of states with Gaussian broadening ``sigma`` (Hartree).

[Source](../python/dft_rs/ase_calculator.py#L2285).

### `dft_rs.ase_calculator.DftRs.get_pauli_spin_susceptibility`

```python
DftRs.get_pauli_spin_susceptibility(self, atoms=None, **controls)
```

Bare Pauli thermodynamics, or ``exchange_enhanced=True`` static
semilocal longitudinal spin response with a thermal SCF per temperature.
The common report includes units, assumptions and raw residuals.

[Source](../python/dft_rs/ase_calculator.py#L2289).

### `dft_rs.ase_calculator.DftRs.get_dirac_coulomb_scf`

```python
DftRs.get_dirac_coulomb_scf(self, atoms=None, **controls)
```

Finite-nucleus no-pair energy with raw SCF diagnostics.

Select the Coulomb, Gaunt or Breit interaction with dirac_interaction.

[Source](../python/dft_rs/ase_calculator.py#L2295).

### `dft_rs.ase_calculator.DftRs.get_two_component_relativistic`

```python
DftRs.get_two_component_relativistic(self, atoms=None, **controls)
```

Picture-changed finite-nucleus X2C, DKH2 or matrix ZORA report.

[Source](../python/dft_rs/ase_calculator.py#L2302).

### `dft_rs.ase_calculator.DftRs.get_dirac_dks`

```python
DftRs.get_dirac_dks(self, atoms=None, **controls)
```

Fixed-sector four-component Dirac–Coulomb KS and electric response.

[Source](../python/dft_rs/ase_calculator.py#L2306).

### `dft_rs.ase_calculator.DftRs.get_no_pair_projector`

```python
@staticmethod
DftRs.get_no_pair_projector(atoms=None, **inputs)
```

Four-component H/S projector, optionally generated from Atoms.
Default Hamiltonian and separator are eV, overlap dimensionless;
``units="atomic"`` uses Hartree. Balanced input is the provider's
responsibility for supplied matrices. With dirac_integrals="rkb",
Atoms supplies Angstrom geometry and explicit nuclear_rms_radii
use Angstrom (Bohr with units="atomic"). dirac_integrals="rmb"
additionally retains the magnetic small basis and complex metric;
magnetic_field/direction are atomic magnetic units and gauge_origin
is Angstrom, or Bohr with units="atomic". No interacting SCF is inferred.

[Source](../python/dft_rs/ase_calculator.py#L2311).

### `dft_rs.ase_calculator.DftRs.get_superconducting_state`

```python
@staticmethod
DftRs.get_superconducting_state(**inputs)
```

Canonical band gap or supplied-operator full Nambu density response.

No Atoms is needed. Energies/gaps and their path directions are eV,
squared vertices eV squared and temperature Kelvin by default.
units="atomic" uses Hartree. pairing_model="lm2005" generates
phonon K and dimensionless Z from the supplied vertices and band
weights; supplied and lm2005-off-diagonal require explicit Z.
scdft_phase_selection=True checks the canonical phase for a supplied
reciprocal kernel and uniform Z without starting guesses. The
condensation free energy is in eV (Hartree with units="atomic").
pairing_model="nambu" accepts full normal and antisymmetric pairing
matrices in eV (Hartree with units="atomic"), with dimensionless
overlap and positive temperature at supplied chemical potential.
Supply signed frequencies and optional frequency_damping for retarded
collisionless response. Dimensionless probe rows are flattened
matrices in row-major order. Optional symmetric nambu_probe_interaction
uses the energy unit. Bare and induced susceptibilities are inverse eV
(inverse Hartree with units="atomic") at fixed quasiparticle occupations.

[Source](../python/dft_rs/ase_calculator.py#L2336).

### `dft_rs.ase_calculator.DftRs.get_scdft_pairing_kernel`

```python
@staticmethod
DftRs.get_scdft_pairing_kernel(**inputs)
```

Pairing K and optional phonon Z on supplied bands; no Atoms.
Default energies are eV and squared vertices eV squared; temperature
is Kelvin. ``units="atomic"`` uses Hartree and Hartree squared.
Supply band_weights to generate dimensionless Z and its response.

[Source](../python/dft_rs/ase_calculator.py#L2365).

### `dft_rs.ase_calculator.DftRs.get_piezoelectric_strain`

```python
@staticmethod
DftRs.get_piezoelectric_strain(*, piezoelectric_stress, piezoelectric_compliance, ion_response)
```

Supplied atomic-unit proper e/S to d; no Atoms or SCF, SI outputs included.

Voigt shear is engineering strain. Both tensors must have the same
clamped/relaxed-ion and fixed-electric-field convention. This static
data method retains native atomic units, like the explicit SI outputs.

[Source](../python/dft_rs/ase_calculator.py#L2380).

### `dft_rs.ase_calculator.DftRs.get_aggregate_moduli`

```python
@staticmethod
DftRs.get_aggregate_moduli(*, phases)
```

Aggregate bounds and estimates (GPa); no ``Atoms`` needed.

[Source](../python/dft_rs/ase_calculator.py#L2393).

### `dft_rs.ase_calculator.DftRs.get_rock_physics`

```python
@staticmethod
DftRs.get_rock_physics(**inputs)
```

Dry frames (including spheroidal pores), fluid substitution, Biot
parameters and thermal-conductivity bounds; no ``Atoms`` needed.

[Source](../python/dft_rs/ase_calculator.py#L2398).

### `dft_rs.ase_calculator.DftRs.get_multicomponent`

```python
DftRs.get_multicomponent(self, atoms=None, **controls)
```

Experimental simultaneous electronic/quantum-particle SCF report.

``quantum_atoms`` contains zero-based quantum nuclei. Geometry and
Gaussian exponents cross the common ASE/atomic unit boundary once;
particle masses use electron masses. The laboratory energy retains
center-of-mass motion. ``species_derivative_order=1``/``2`` returns
the analytic gradient/Hessian. ``quantum_position_constraint=True``
differentiates nuclear expectation positions; otherwise a quantum
center derivative measures finite-basis dependence. Target positions
use Angstrom in ASE mode and Bohr in atomic mode. The same boundary
converts gradients, Hessians and ``species_polarizability`` once.
For ``species_electronic_model="dirac-coulomb"``, supply
``nuclear_rms_radii`` in Angstrom or Bohr in atomic mode.
This route returns the joint fixed-sector energy and raw residuals.
``field`` adds the total dipole. ``species_polarizability=True``
enables static or frequency-resolved coupled dipole response.
``species_quadrupole_polarizability=True`` adds quadrupole and
mixed electric response. ``species_field_gradient`` applies the
five gradient modes in eV/e/Angstrom², or Hartree/e/Bohr² in
atomic mode. ``origin`` sets the shared dipole and quadrupole
origin in the selected length unit or by name.

[Source](../python/dft_rs/ase_calculator.py#L2403).

### `dft_rs.ase_calculator.DftRs.get_continuous_species`

```python
DftRs.get_continuous_species(self, atoms=None, **controls)
```

Fixed-template real nuclear-charge RKS model and analytic sensitivities.

Supply fixed_electron_count, charge_directions, charge_coordinates and
charge_bounds. Charges are in e and parameter coordinates dimensionless;
charge_origin uses Angstrom by default and Bohr with units='atomic'.
Optional charge_nuclear_gradient returns template nuclear forces.
Local Gaussian amplitudes, steric strengths and fragment_internal_energy
use eV by default; exponents use Angstrom^-2 and environment hardness
Angstrom^-3. Atomic mode uses Hartree and Bohr. Explicit pair masks and
fixed integer spin populations are dimensionless. Optional probes and
dynamic response use the common report's unit labels.

[Source](../python/dft_rs/ase_calculator.py#L2428).

### `dft_rs.ase_calculator.DftRs.get_reduced_fragment`

```python
DftRs.get_reduced_fragment(self, atoms=None, *, model, fragment_mode='evaluate', **controls)
```

Evaluate saved candidate energy/forces with standard ASE conversion.

``model`` is opaque native-unit JSON text or a dict and is never unit
converted. Atoms are the retained labels followed by the saved ionic
role label. Atoms positions always use Angstrom, including atomic
output mode. Field/origin controls use ASE units by default, atomic units
in atomic mode. ``fragment_mode="evaluate"`` preserves the original
record; ``"few-evaluate"`` uses a fixed PBE primitive record and returns
energy/primary forces and metadata without dipole or response. The returned candidate status does not establish chemical accuracy
or an internal-polarization model. This getter does not define masses
for reduced-coordinate dynamics or relax frozen internal coordinates.

``"rigid-evaluate"`` instead requires an explicit ``rigid-static-core-v1``
primitive recipe and atoms at its retained point-ion centers.
``derivative_order=0/1/2`` requests energy, analytic gradient or relaxed
Hessian. Center gradients and Hessians convert to eV/Angstrom and
eV/Angstrom^2; center/angle blocks convert to eV/Angstrom/radian, and
angle blocks to eV/radian^2. Euler gradients are energy covectors,
not Cartesian forces or world-frame torques. Recipe angles remain
native radians and opaque model coefficients are never converted.
The static PBE restricted route is experimental; private generated
schemas and frozen internal-coordinate relaxation are unsupported.

``"scalar-evaluate"`` and ``"scalar-optimize"`` use the explicit
frozen-scalar-action-v1 record and its first-state raw geometry.
Flattened ``fragment_weights`` contain one simplex per saved site.
The full joint endpoint evaluators retain site cross interactions;
this is not a compressed potential. Weight gradients and residuals
with Hartree dimensions convert to eV. Native-unit saved models are
opaque. Only weights change; nuclear geometry optimization and
reaction enthalpies are not implemented by these modes.

``"scalar-gradient"``, ``"scalar-geometry-optimize"`` and ``"scalar-hessian"`` select one
explicit ``fragment_state`` and ``fragment_endpoint``. The saved
endpoint must declare ``fixed-internal-affine-v1`` transport and
full-atom owners. Only the raw-center coordinates move, with fixed
internal offsets and no fragment rotation or weight interpolation.
Gradients/forces convert to eV/Angstrom and positions to Angstrom.
The geometry convergence flag is returned without inferring a
vibrational minimum or reaction enthalpy.

[Source](../python/dft_rs/ase_calculator.py#L2443).

### `dft_rs.ase_calculator.DftRs.get_casscf`

```python
DftRs.get_casscf(self, atoms=None, **controls)
```

Complete-active-space SCF report through the common driver.

``cas=[active electrons, active orbitals]`` is required. Energies and
energy tolerances are eV in ASE mode and Hartree in atomic mode.

[Source](../python/dft_rs/ase_calculator.py#L2505).

### `dft_rs.ase_calculator.DftRs.get_mcpdft`

```python
DftRs.get_mcpdft(self, atoms=None, **controls)
```

Multiconfiguration pair-density functional theory report.

Takes the ``casscf`` controls plus ``on_top_functional``,
``pdft_lambda`` and ``pdft_method``; energies are eV in ASE mode and
Hartree in atomic mode.

[Source](../python/dft_rs/ase_calculator.py#L2513).

### `dft_rs.ase_calculator.DftRs.get_feature_registry`

```python
@staticmethod
DftRs.get_feature_registry()
```

The registry of public calculations as data.

[Source](../python/dft_rs/ase_calculator.py#L2523).

### `dft_rs.igmh_visualization.write_igmh_cubes`

```python
write_igmh_cubes(report, prefix, numbers, positions, *, field='inter', position_unit='bohr', grid=None)
```

Write matching interaction and signed-density CUBE files. Return their paths.

Use a report obtained with igmh_grid and igmh_grid_shape. The optional
grid argument repeats the original origin and step vectors, in the report's
coordinate units. Supply it when any grid axis has one sample.

The two filenames append -delta-g-{field}.cube and -signed-density.cube
to prefix. Draw an isosurface of the first file and colour it with the
second. The exported positions and fields use atomic units. Supply the
molecular geometry in Bohr or set position_unit="angstrom", for example
for positions from an ASE Atoms object. Choose field="inter", "intra" or
"total" for the corresponding delta-g field.

[Source](../python/dft_rs/igmh_visualization.py#L167).

### `dft_rs.igmh_visualization.write_igmh_slice`

```python
write_igmh_slice(report, path, *, axis=2, index=None, field='inter', isovalue=0.005, color_limit=0.05)
```

Write and return a standalone SVG of a two-dimensional IGMH grid slice.

Axis and index select the fixed grid coordinate, counting from zero.
The default index selects the middle sample of that axis. Sample cells
at or above the delta-g isovalue use a blue to green to red signed-density
scale. Isovalue and color_limit use electrons/Bohr^4 and electrons/Bohr^3,
respectively, also when the input report uses ASE units. The preview
uses sample-cell shading on the selected slice.

[Source](../python/dft_rs/igmh_visualization.py#L228).

### `dft_rs.igmh_visualization.write_igmh_viewer`

```python
write_igmh_viewer(report, path, numbers, positions, *, field='inter', position_unit='bohr', bonds=None, isovalue=0.005, color_limit=0.05, igmh_limit=None, nci_limit=None, surface_density_min=None, surface_density_max=None, surface_density_field='density', surface_color_range=None, scatter_density_limit=None)
```

Write a standalone rotatable IGMH scene and an IGMH/NCI scatter plot.

Supply a three-dimensional affine-grid report with at least two points
on each axis. Geometry uses Bohr or position_unit="angstrom". Optional
bonds are zero-based atom-index pairs. By default, links for common
nonmetals are inferred within each fragment from their covalent radii.

The scene contours delta-g for field="inter", "intra" or "total" and
colours it by signed density. Isovalue and igmh_limit use electrons/Bohr^4.
Color_limit sets the signed colour scale. Scatter_density_limit sets the
horizontal scatter window; it defaults to color_limit. Sampled molecular
densities below 1e-8 electrons/Bohr^3 are outside that window. The NCI
ordinate shows the full range of the dimensionless reduced density gradient
by default. Supply nci_limit to start with a custom upper limit, or select
the low-gradient detail range in the browser.

Surface_density_min/max set clipping bounds on the selected surface
density field. Use surface_density_field="signed_density" to select
signed density. Surface_color_range sets an independent pair of signed
colour endpoints and defaults to (-color_limit, color_limit). The saved
file allows decimal adjustments to these controls. Density arguments use
electrons/Bohr^3. The HTML embeds the complete indexed mesh, scatter
samples and JavaScript. Viewer arrays use 32-bit floats and indices.
The source report retains its original precision.

[Source](../python/dft_rs/igmh_visualization.py#L528).

### Unit conversion constants

`dft_rs.units.X_IN_Y` is the value of one X expressed in Y; native
calculations retain atomic units. The optional ASE adapter uses the same
package CODATA 2018 factors at its boundary, independently of ASE defaults.
The Rust CODATA constants are described
in [the Rust reference](rust-api.md#errors-and-threading).

| Name | Definition |
|---|---|
| `dft_rs.units.BOHR_IN_ANGSTROM` | `0.529177210903` |
| `dft_rs.units.ANGSTROM_IN_BOHR` | `1.0 / BOHR_IN_ANGSTROM` |
| `dft_rs.units.BOHR_IN_METER` | `5.29177210903e-11` |
| `dft_rs.units.HARTREE_IN_EV` | `27.211386245988` |
| `dft_rs.units.HARTREE_IN_KCAL_MOL` | `627.509474063` |
| `dft_rs.units.HARTREE_IN_KJ_MOL` | `2625.4996394799` |
| `dft_rs.units.HARTREE_IN_CM1` | `219474.6313632` |
| `dft_rs.units.HARTREE_IN_JOULE` | `4.3597447222071e-18` |
| `dft_rs.units.HARTREE_PER_BOHR3_IN_GPA` | `HARTREE_IN_JOULE / (BOHR_IN_METER * BOHR_IN_METER * BOHR_IN_METER) * 1e-09` |
| `dft_rs.units.AU_DIPOLE_IN_DEBYE` | `2.541746473` |

`dft_rs.native.BOHR_PER_ANGSTROM` equals `dft_rs.units.ANGSTROM_IN_BOHR`;
`dft_rs.native.HARTREE_IN_EV` equals `dft_rs.units.HARTREE_IN_EV`.

<!-- python-public-reference:end -->

### Prescribed-population nonlinear electric response

`electric_field_taylor`, `mixed_electric_field_response`,
`static_electric_tensor`, `beta` and `gamma` call the common driver and
accept its additional reference/response keyword controls. Their existing
explicit arguments and default values remain available. For example:

```python
import dft_rs

options = dict(xc="pbe", basis="sto-3g", dispersion="none", grid_level=4,
               occupation_mode="fixed", occupations_alpha=[0.8, 0.2],
               occupations_beta=[0.8, 0.2], field=[0., 0., .003])
report = dft_rs.electric_field_taylor(
    [1, 1], [[0., 0., -.8], [.2, .1, .8]], [0., 0., 1.], order=4, **options)
beta_zzz = 2 * report["dipole_coefficients"][2][2]
gamma_zzzz = 6 * report["dipole_coefficients"][3][2]
```

Native geometry is Bohr, field is Hartree/(e Bohr), and each path coefficient
is a dipole in e Bohr divided by its derivative factorial. `field-mixed`
returns actual mixed directional derivatives. Full beta/gamma return
e^3 Bohr^3/Hartree^2 and e^4 Bohr^4/Hartree^3. `response_tolerance` and
`response_max_iterations` select the shared solver; the previous
`response_tol`/`response_max_iter` arguments remain available. `field` and
`reference_field` are aliases for the reference bias; do not supply both.

ASE `get_electric_field_taylor`, `get_mixed_electric_field_response`,
`get_static_electric_tensor`, `get_beta` and `get_gamma` accept the same
reference keywords, inherit calculator reference controls and cross the
unit boundary once. Default ASE fields/directions are eV/(e Angstrom),
dipoles are e Angstrom, beta is e^3 Angstrom^3/eV^2 and gamma is
e^4 Angstrom^4/eV^3; `units="atomic"` selects the native conventions.
The compatibility `units` string remains a summary; `unit_labels` contains
the authoritative dotted-path labels. Direct `run_feature` and
`get_feature_report` return those labels in the `units` mapping.

Reports preserve the physical non-idempotency instead of replacing it by
zero. `projector_constraint=False` and the `fixed_population` diagnostics
identify the field Taylor constraint. Full tensors carry
`fixed_population_defect`, raw ordered-slot interchange measurements and
the independently reevaluated response residuals. `assembly="variational"`
selects three-solve 2n+1 beta on the same fixed spectrum. These constrained
nonthermal derivatives exclude a thermal redistribution of occupations;
periodic, magnetic, moving-nuclear and correlated PT2 population response
must be checked separately. Acceptance data are in
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

| Physical calculation | Native Python / ASE | Current integration scope |
|---|---|---|
| Generalized complex one-body PT2 path | New low-level path bindings pending | Implemented Rust provider passes 41 bounded pair/physical component fits and 108 complete two-site FC source fits; new public coupling support remains pending |
| ROKS common-orbital electric susceptibility | Existing polarizability and dynamic-polarizability wrappers/getters | Reuses the constrained generic source engine; general longitudinal magnetic source bindings and transverse ROKS J assembly remain required |
| Open-shell correlated Ramsey coupling | Existing `spin_spin_couplings` / `get_spin_spin_couplings` | Closed-shell correlated scope remains unchanged until full-spin physical assembly, constraints and public acceptance pass |

The complete new Rust signatures, units, factorial normalization, raw defects
and restrictions are in [rust-api.md](rust-api.md#spin-complete-pair-and-common-orbital-providers).
Acceptance state is recorded in [response.md](response.md#spin-complete-pair-and-common-orbital-work-2026-10-30).

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

`dft_rs.anharmonic(..., method="strict-vpt2")` and `DftRs.get_anharmonic(..., method="strict-vpt2")` select standard nonresonant second-order VPT2. The default `method="retained-polyad"` preserves earlier results. An unknown method raises `ValueError` before SCF.

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

### Dimensionless operator in the no-pair sector

`dirac_operator_real/imaginary` supplies coefficient-major `[order][row][column]` dimensionless covariant Hermitian operators to `no_pair_projector` or the static `DftRs.get_no_pair_projector`. The result `projected_operator_coefficients_real/imaginary` is the factorial-normalized bare projection `S P O P S`; atomic and physical ASE modes preserve its dimensionless units. Explicit CPU/memory controls remain accepted on the static getter. The rebuilt128-fit Rust suite and native/ASE/both-CLI routes pass. This is not interacting relativistic DFT, X2C decoupling, two-electron picture change or a conserved current.

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
