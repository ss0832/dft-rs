<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 ss0832 -->

# Choosing and using an API

Choose a calculation by feature ID, then choose the interface that will
supply its structure and read its report.

| What you have | Start here |
| --- | --- |
| An XYZ file and a molecule | [Water dipole and Hessian](#molecular-example-water-dipole-and-hessian) |
| A unit cell and periodic axes | [Stress and Gamma phonons](#periodic-example-stress-and-gamma-phonons) |
| A calculation name or physical quantity | [Calculation finder](#find-a-calculation) |
| An existing Python script using `hessian`, `vibrations` or `polarizability` | [Choose a Python entry point](#choose-a-python-entry-point) |
| A functional, reference or external field to check | [Applicability and errors](#check-applicability-and-errors) and [verified scope](development-status.md) |

- [Find a calculation](#find-a-calculation)
- [Choose an orbital or density analysis](#choose-an-orbital-or-density-analysis)
- [Choose an interface](#choose-an-interface)
- [Choose a Python entry point](#choose-a-python-entry-point)
- [Choose units for inputs and reports](#choose-units-for-inputs-and-reports)
- [Run the same small calculation](#run-the-same-small-calculation)
- [Molecular example: water dipole and Hessian](#molecular-example-water-dipole-and-hessian)
- [Periodic example: stress and Gamma phonons](#periodic-example-stress-and-gamma-phonons)
  - [Periodic reference and functional support](#periodic-reference-and-functional-support)
- [Run a uniform magnetic-flux SCF](#run-a-uniform-magnetic-flux-scf)
- [Set the reference and functional](#set-the-reference-and-functional)
- [Set an electronic temperature](#set-an-electronic-temperature)
- [Choose a molecular relativistic Hamiltonian](#choose-a-molecular-relativistic-hamiltonian)
- [Read units, tensors and convergence](#read-units-tensors-and-convergence)
- [Read a complex quadrupole response](#read-a-complex-quadrupole-response)
- [Read a generalized double-hybrid gradient](#read-a-generalized-double-hybrid-gradient)
- [Choose a Hessian and minimum check](#choose-a-hessian-and-minimum-check)
- [Resolve similar names](#resolve-similar-names)
- [Check applicability and errors](#check-applicability-and-errors)

## Find a calculation

These are entry points to the detailed references. Read the selected row's
model restrictions and the linked coverage evidence before combining options.
The complete, aligned catalogues are in the [Rust table](rust-api.md#table-1-rust),
[CLI table](cli.md#table-2-cli) and [Python/ASE table](python-api.md#table-3-native-python-and-ase).

### Molecular structure, electronic response and spectra

| Goal | Feature IDs to look up | Select explicitly |
| --- | --- | --- |
| Molecular energy, forces and structure curvature | `energy`, `gradient`, `hessian`, `orbitals` | Reference, charge, multiplicity, basis and functional |
| Vibrations and anharmonicity | `frequencies`, `ir-raman`, `force-constants`, `anharmonic` | Full tensor versus directional response; harmonic versus anharmonic treatment |
| Electric moments | `dipole`, `molecular-multipole` | Multipole rank, origin and Cartesian/spherical convention |
| Electric response | `polarizability`, `dynamic-polarizability`, `beta`, `gamma`, `dynamic-beta`, `dynamic-gamma` | Static/retarded/imaginary frequency, damping and index ordering |
| Higher electric multipole response | `quadrupole-polarizability`, `multipole-polarizability` | Source/output ranks, mode normalization and origin |
| Mixed fields and derivatives | `field-taylor`, `field-mixed`, `nuclear-taylor`, `nuclear-mixed`, `molecular-mixed`, `static-spectroscopy` | Perturbation directions, order and Taylor normalization |
| Constrained electronic states | `constrained-dft`, `energy` / `gradient` with MOM or IMOM | Choose a charge/spin constraint operator or a tracked orbital occupation |
| Strong-correlation and self-interaction methods | `mcpdft`, `dft-plus-u`, `pz-sic` | Choose an active space, explicit Hubbard AO subspaces, or real occupied orbitals for a self-interaction correction |
| Car-Parrinello molecular dynamics | `car-parrinello-md` | Real RKS/UKS/ROKS or complex GKS, timestep, fictitious orbital mass and initial nuclear velocities |
| Excitations and optical spectra | `excitations`, `ecd-spectrum`, `optical-activity`, `vcd`, `mcd` | Response approximation, frequencies and broadening |
| Orbital and density analysis | `orbitals`, `natural-bond`, `excitations`, `periodic-natural-bond`, `wannier-flow` | Density source, spin reference, and ground-state, transition or difference density |
| Core-level spectra | `xanes`, `nexafs` | Absorber or core orbitals, CVS-TDA or relaxed-core-ion model where available, polarization and linewidth |
| Nonlinear optical processes | `electric-process` | Use a Pockels or Kerr frequency preset on the documented `dynamic-beta` or `dynamic-gamma` interface |
| Optical properties of a medium | `molecular-medium-optics`, `magneto-optical`, `cotton-mouton` | Number density, orientation and optical boundary conditions |
| Magnetic and electron-spin properties | `magnetic-quadrupole`, `magnetizability`, `hypermagnetizability`, `hyperfine`, `g-tensor`, `spin-spin-zfs`, `broken-symmetry-exchange` | Gauge, origin, spin model and nuclear/isotope data |
| NMR | `nmr-shielding`, `spin-spin-couplings`, `nmr-spectrum` | Shielding references, nuclear g factors/spins, coupling terms and spectral spin model |

### Periodic systems

| Goal | Feature IDs to look up | Select explicitly |
| --- | --- | --- |
| Periodic energy, forces and stress | `periodic-energy`, `stress`, `opt-fixed-cell`, `cell-opt` | Cell, periodic axes, k mesh, occupations and stress convention |
| Periodic elastic and anharmonic tensors | `elastic`, `toec`, `foec`, `periodic-cubic-fc`, `periodic-quartic-fc` | Homogeneous strain or repeated Gamma displacement, clamped/relaxed ions and model restrictions |
| Periodic vibrations and thermal properties | `phonons`, `phonon-bands`, `phonon-thermo`, `quasi-harmonic` | q sampling, temperature, harmonic model and dimensional electrostatics |
| Periodic electric/strain response | `born-charges`, `dielectric-tensor`, `static-dielectric`, `dielectric-q`, `piezoelectric-tensor`, `crystal-ir` | Clamped/relaxed ions, q/direction, dimension and electrical boundary conditions |
| Periodic field reference | `periodic-efield`, `periodic-field-scf`, `periodic-magnetic-scf` | Berry-branch, Fourier-field or commensurate uniform-flux model and charge compensation |
| Bands and electronic thermodynamics | `bands`, `dos`, `electronic-structure`, `mermin-free-energy`, `thermal-response` | Fermi occupations, electronic free energy, and the ensemble used for thermal DFPT |

### Specialized electronic models

| Goal | Feature IDs to look up | Select explicitly |
| --- | --- | --- |
| Relativistic and multicomponent models | `two-component-relativistic`, `dirac-dks`, `dirac-coulomb-scf`, `no-pair-projector`, `multicomponent` | Hamiltonian, electronic sector, quantum particles and operator transformations |
| Superconducting DFT | `superconducting-state`, `scdft-pairing-kernel` | Supplied normal bands, pairing/mass model, particle number and phase convention |
| Continuous chemical species | `continuous-species` | Record format, fixed electron space, construction functional and fitted domain |

For the full implemented/validated/pending matrix use
[scope](scope.md), the [response ledger](response.md#response-coverage-ledger)
and [property targets](property_targets.csv). A successful spectrum
post-processing test does not validate the electronic response supplying it.
The reference-specific dispatch and acceptance fixtures are listed in
[ROKS/GKS coverage by property and periodicity](response.md#public-roks-and-gks-calculation-coverage).

For static GKS orbital magnetizability, use `reference="gks"` and
`gauge="common-origin"` with `dft_rs.magnetizability`,
`calc.get_magnetizability`, or Rust `driver::run("magnetizability", ...)`.
The corresponding CLI controls are `--reference gks --gauge common-origin`.
Supply `gauge_origin` when comparing runs at a specific common origin.
The atomic-unit interfaces use Bohr for the origin. ASE uses Å unless
`units="atomic"` is selected.
[Spinor orbital model and acceptance](response.md#static-spinor-orbital-magnetizability-at-a-common-gauge-origin)

## Choose an orbital or density analysis

| Analysis | Feature and controls | Scope |
| --- | --- | --- |
| Canonical molecular orbitals and AO density | `orbitals` with `coefficients` and `density` | Select the electronic reference explicitly. Complex spinor reports give real and imaginary parts. |
| NAO, NPA, Wiberg index, NBO, hybrid composition and NLMO | `natural-bond` with `nbo_density` and `nbo_spin` | Choose the SCF, relaxed PT2 or CASSCF density as supported. The spinor Lewis path requires real spatial blocks in a common spin frame. |
| Natural spatial orbitals and effectively unpaired electrons | `natural-bond` with `nbo_natural_orbitals=true` | Works on the selected density. For a UKS density these are unrestricted natural orbitals. |
| Natural spin-density orbitals and point samples | `natural-bond` with `nbo_spin_density=true` and optional `nbo_spin_density_points` | Signed spin-density weights and molecular point values. Use lengths in Bohr in atomic-unit mode. |
| Real-space interaction fields | `igmh` with `fragments` and `igmh_points` or an affine grid | Molecular SCF charge density and inter- or intrafragment interaction fields. |
| Corresponding occupied orbitals | `natural-bond` with `nbo_corresponding_orbitals=true` | Integer-occupied collinear SCF references. Returns the paired alpha–beta overlaps and both unpaired complements. |
| TDA natural transition orbitals | `excitations` with `td_nto=true` | One set of particle–hole pairs per reported transition and spin channel. |
| Unrelaxed TDA attachment and detachment orbitals | `excitations` with `td_difference_density=true` | RKS, UKS and full-spin GKS. The occupations are the signed eigenvalues of the corresponding state-density difference blocks. |
| Resonance structures and orbital decompositions | `natural-bond` with `nbo_resonance=true`, `natural-shielding`, `natural-coupling`, `natural-energy-decomposition` | Natural Lewis structures, NLMO decomposition, or explicit atom fragments as appropriate. |
| Periodic populations and hybrid Wannier centers | `periodic-natural-bond` for unfolded BvK analysis and `wannier-flow` for center flow | Model and mesh restrictions apply. |

The NTO and natural difference orbital controls can be used together.
The first reads the directed transition density and the second gives the
unrelaxed TDA change in the state density. Orbital gauges within degenerate
subspaces can vary while their weighted density is invariant.
[Normalization, scope and numerical checks](response.md#natural-transition-orbitals)

Further orbital analyses have distinct input requirements. NOCV and
OVOCV require a defined fragment reference and deformation density.
IAO and IBO require a
projected atomic reference and occupied-space construction. Dyson
orbitals require a transition between states with different particle
numbers. These public routes remain open.

### Map an interaction with IGMH

Assign one fragment label per atom. For a water dimer ordered OHH–OHH,
use `0,0,0,1,1,1`. The interfragment field measures gradient cancellation
between the two water densities. The signed density colours the surface
using the middle eigenvalue of the molecular density Hessian.

~~~bash
dft-rs igmh dimer.xyz --xc wb97x-d --fragments 0,0,0,1,1,1 --igmh-points '[[0.31,-0.19,3.13]]' --format json
~~~

In native Python, pass `igmh_points` in Bohr to `dft_rs.igmh`. In ASE, pass
the same physical points in Angstrom to `calc.get_igmh`. For a volume, use
`igmh_grid=[origin, step_x, step_y, step_z]` and
`igmh_grid_shape=[nx, ny, nz]`. Reshape `igmh_delta_g_inter` and
`igmh_signed_density` to that shape in C order for paired isosurface and
colour grids. The last index changes fastest. The XYZ geometries supplied
to the CLI use Angstrom and its sampling controls use Bohr.
[Model, units and verification](response.md#r090-molecular-igmh)

Use the same report to write a standalone slice and the two congruent
CUBE files for a three-dimensional view. The `igmh_visualization`
module uses the report's unit labels to convert the fields to atomic
units. Select an isosurface from the delta-g file and colour it with
the signed-density file.

~~~python
from dft_rs import igmh
from dft_rs.igmh_visualization import write_igmh_cubes, write_igmh_slice

# numbers and positions_bohr describe the same dimer as the CLI example.
report = igmh(
    numbers, positions_bohr, xc="wb97x-d", fragments=[0, 0, 0, 1, 1, 1],
    igmh_grid=[[-3, -2, -1], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
    igmh_grid_shape=[31, 21, 41],
)
write_igmh_cubes(report, "dimer", numbers, positions_bohr)
write_igmh_slice(report, "dimer-inter.svg", axis=1, isovalue=0.005)
~~~

An ASE report works with the same functions. Pass `atoms.numbers`,
`atoms.positions` and `position_unit="angstrom"` when writing its CUBE
files. Pass the original `grid` to the CUBE writer when an axis
contains a single point.

For a rotatable scene and an interaction scatter plot, export the same
three-dimensional grid from Python.

~~~python
from dft_rs.igmh_visualization import write_igmh_viewer

write_igmh_viewer(
    report, "dimer-igmh.html", numbers, positions_bohr,
    isovalue=0.01, color_limit=0.05,
    surface_density_min=0.001, surface_density_max=0.05,
)
~~~

Open the generated file in a browser. Drag to rotate, scroll to zoom
and adjust the opacity to see atoms behind the surface. Use the scatter
selector to switch between the selected IGMH `delta-g` field and the
dimensionless NCI reduced gradient. The full NCI range is the default.
Select the detail range to inspect the low-gradient branch (`s <= 2`).
The graph reports the number of shown samples and, for NCI, the largest
sampled reduced gradient. Both plots use signed density on the horizontal
axis. Set `scatter_density_limit` to control that window independently
of `color_limit`.

Choose **Molecular density ρ** or **Signed density** in the surface
filter, then adjust the decimal minimum and maximum in electrons/Bohr³.
The signed field accepts negative values. Clear either bound for an open
interval on that side. The viewer clips each triangle using linear
interpolation and remembers both fields' ranges. Adjust **Colour minimum**
and **Colour maximum** to map the visible signed values to the colour
ramp independently of the surface filter, for example from -0.025 to
-0.002 electrons/Bohr³. The complete indexed mesh, scatter samples and
JavaScript are embedded in one offline HTML file. Viewer arrays use 32-bit
numbers and the calculation report retains its original precision.

Supply zero-based `bonds` for explicit connectivity. The default marks
short covalent links between common nonmetals within each fragment. The
view needs at least two samples along each grid axis.

## Choose an interface

Start with the feature ID in the table above. Find its `route` in the
registry or look up the ID in the reference for your interface. Keep the
same reference, functional and model controls when comparing interfaces.

| Interface | Start here | Structure input |
| --- | --- | --- |
| Rust common driver | `driver::run(id, &system, &controls)` | Bohr; cell vectors are rows |
| Canonical CLI | `dft-rs <id> structure.xyz --format json ...` | XYZ and `--cell` lengths in Angstrom |
| Native Python | `dft_rs.run_feature(id, numbers, positions, **controls)` or the named wrapper | Bohr; `cell` rows in Bohr |
| ASE | `calculator.get_feature_report(id, atoms, **controls)` or the named getter | `Atoms` in Angstrom, including when report units are atomic |

The registry returns `id`, `space`, `route`, `rust`, `cli`, `python`,
`ase` and `targets`. Use the recorded names and the corresponding
signatures. For example, `ir-raman` uses `vibrational_spectra` in Python
and `get_vibrational_spectra` in ASE.

```python
import dft_rs

for entry in dft_rs.feature_registry()["features"]:
    if entry["space"] == "molecule":
        print(entry["id"], entry["route"], entry["python"], entry["ase"])
```

```console
dft-rs feature-registry --format json
```

| Registry entry | Rust | CLI | Native Python | ASE |
| --- | --- | --- | --- | --- |
| `energy` (`driver`) | `driver::run("energy", ...)` | `dft-rs energy` | `run_feature("energy", ...)` or `energy(...)` | `get_feature_report("energy", atoms)` or `get_energy_report(atoms)` |
| `ir-raman` (`driver`) | `driver::run("ir-raman", ...)` | `dft-rs ir-raman` | `run_feature("ir-raman", ...)` or `vibrational_spectra(...)` | `get_feature_report("ir-raman", atoms)` or `get_vibrational_spectra(atoms)` |
| `electric-process` (`dedicated`) | `response::dynamic_electric_process` | `dft-rs dynamic-beta ... --electric-process pockels --response-omega 0.05` | `dynamic_electric_process(..., process="pockels", frequencies=[0.05])` | `get_dynamic_electric_process(atoms, process="pockels", frequencies=[0.05])` |
| `feature-registry` (`driver`, `data`) | `driver::run("feature-registry", ...)` with `System::default()` | `dft-rs feature-registry --format json` | `feature_registry()` | Use the module-level registry |

A `driver` route accepts the generic Rust/Python/ASE call in the
applicable space. A `dedicated` route requires its named entry and its
own control names. For `data` routes, check the table for required arrays
or operators and whether a structure is used. [Shared requests](rust-api.md#common-calculation-driver)

The pip distribution is `dft-rs-python`, the Python import is `dft_rs`, and
the installed command is `dft-rs`. Local wheel/source installation and
toolchain requirements are in [CLI installation](cli.md).

## Choose a Python entry point

Use `dft_rs.run_feature` when the feature ID is a string in a script or a
data file. Use the named wrapper when the calculation is decided in the
code. ASE offers the same choice through `get_feature_report` and its
named getters.

| Calculation | Common driver feature ID | Native report wrapper | ASE report getter |
| --- | --- | --- | --- |
| Total energy and decomposition | `energy` | `energy` | `get_energy_report` |
| Analytic Cartesian Hessian | `hessian` | `hessian_report` | `get_hessian_report` |
| Harmonic vibrations and thermochemistry | `frequencies` | `frequencies_report` | `get_frequencies_report` |
| IR and Raman | `ir-raman` | `vibrational_spectra` | `get_vibrational_spectra` |
| Static dipole polarizability | `polarizability` | `polarizability_report` | `get_polarizability_report` |
| Complex dipole polarizability at a frequency | `dynamic-polarizability` | `dynamic_polarizability` | `get_dynamic_polarizability` |

The older native `hessian`, `vibrations`, `raman` and `polarizability`
functions have their own fixed signatures and return contracts. For
scripts that pass settings such as `reference="gks"` or reuse the same
controls for several features, start with the report entries above and
check the property-specific reference scope.

Read calculator parameters first and per-call keywords second. The
per-call keywords take precedence. Use the [unit table](#choose-units-for-inputs-and-reports)
when moving controls or geometry between native Python and ASE.

## Choose units for inputs and reports

| Interface | Geometry and cell | Controls and results |
| --- | --- | --- |
| Rust and native Python | Bohr. Periodic cells use three lattice vectors as rows. | Atomic units for dimensionful common-driver controls. Read `units` per result field. |
| CLI | XYZ positions and `--cell` lengths in Angstrom. | Each option uses its documented unit. For example `--frequency` is Hartree and `--origin` is Bohr on the quadrupole route. |
| ASE common reports | `Atoms` positions and cell in Angstrom for either report mode. | Use `units="atomic"` to compare convertible per-call controls and report fields directly with native Python. The default physical mode converts according to each field's unit label. |

Dimensionless controls such as a reduced k point, integer spin
multiplicity or `xc_terms` weight keep the same meaning across
interfaces. [Report fields and units](rust-api.md#result-conventions-and-physical-scope)

For a numerical comparison across interfaces, write the CLI geometry and
cell with enough significant digits to reconstruct the same coordinates
in Bohr. A low-precision XYZ export can move a response tensor by more
than a tight comparison tolerance.

## Run the same small calculation

This all-electron H2/STO-3G HF example uses a bond length of 1.4 Bohr.
It is a compact interface example, not a converged basis-set benchmark.

### Native Python

```python
import dft_rs

positions = [[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]  # Bohr
controls = dict(xc="hf", reference="rks", basis="sto-3g", dispersion="none")
report = dft_rs.run_feature("energy", [1, 1], positions, **controls)
report.require_converged()
print(report["energy"], report.unit("energy"))
print(report["model"])
```

The named `dft_rs.energy([1, 1], positions, **controls)` uses this same driver.

### ASE

```python
from ase import Atoms
from dft_rs import DftRs
from dft_rs.units import BOHR_IN_ANGSTROM

atoms = Atoms("H2", positions=[
    [0.0, 0.0, -0.7 * BOHR_IN_ANGSTROM],
    [0.0, 0.0,  0.7 * BOHR_IN_ANGSTROM],
])
calculator = DftRs(xc="hf", reference="rks", basis="sto-3g", dispersion="none")
report = calculator.get_energy_report(atoms, units="atomic")
print(report["energy"], report["units"]["energy"])
```

`units="atomic"` selects atomic report units and atomic units for
convertible per-call controls. Keep `Atoms.positions` and `Atoms.cell` in
Angstrom. Standard ASE energy, force and stress getters retain ASE units.

### CLI

Save the equivalent geometry as `h2.xyz`:

```text
2
H2, Angstrom coordinates
H 0.0 0.0 -0.3704240476321
H 0.0 0.0  0.3704240476321
```

```console
dft-rs energy h2.xyz --xc hf --reference rks --basis sto-3g --dispersion none --format json
```

JSON goes to stdout; SCF energies and convergence residuals go to stderr
by default. Logging controls are in the [CLI logging reference](cli.md#logging).

### Rust

```rust
use dft_rs::driver::{run, Controls, System, Value};

fn main() -> dft_rs::Result<()> {
    let system = System::molecule(
        vec![1, 1],
        vec![[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]],
    );
    let controls = Controls::from_value(
        &Value::object()
            .with("xc", "hf")
            .with("reference", "rks")
            .with("basis", "sto-3g")
            .with("dispersion", "none"),
    )?;
    let report = run("energy", &system, &controls)?;
    println!("{:?}", report.get("energy").and_then(Value::as_f64));
    Ok(())
}
```

Use the same system with another applicable feature ID and that feature's
controls. Consult the exact signature for dedicated APIs.

## Molecular example: water dipole and Hessian

Use one bent water geometry for a dipole and an analytic Cartesian
Hessian at RKS/PBE/STO-3G. These are fixed-geometry calculations. The
Hessian can be evaluated before optimization. For equilibrium vibrational
analysis, optimize at the same level before calling `frequencies`.
Save the following Angstrom geometry as `water.xyz`.

```text
3
Bent water, Angstrom coordinates
O  0.0                0.0                0.0
H  0.957281574523527  0.0                0.0
H -0.239188099328156  0.927118473502056  0.0
```

```console
dft-rs dipole water.xyz --xc pbe --reference rks --basis sto-3g --grid-level 3 --dispersion none --origin center_of_mass --format json
dft-rs hessian water.xyz --xc pbe --reference rks --basis sto-3g --grid-level 3 --dispersion none --format json
```

`--format json` selects the common-driver Hessian report in the CLI.
Specify `origin` on the dipole call. Use the Hessian's Cartesian indexing
`3*atom + axis` in the same atom order as the input.

```python
import dft_rs

numbers = [8, 1, 1]
positions = [[0.0, 0.0, 0.0], [1.809, 0.0, 0.0], [-0.452, 1.752, 0.0]]  # Bohr
settings = dict(xc="pbe", reference="rks", basis="sto-3g",
                grid_level=3, dispersion="none")
dipole = dft_rs.dipole(numbers, positions, origin="center_of_mass", **settings)
hessian = dft_rs.hessian_report(numbers, positions, **settings)
print(dipole["dipole"], dipole["origin"], hessian["hessian"][0][0])
```

In ASE, load the same geometry with `ase.io.read("water.xyz")` and pass
`settings` to `DftRs(**settings)`. Call
`calc.get_dipole_report(atoms, origin="center_of_mass", units="atomic")`
and `calc.get_hessian_report(atoms, units="atomic")`. In Rust, use the
[molecular System example](#rust) with the water positions above and
call `run("dipole", ...)` and `run("hessian", ...)`. The dipole report
uses e Bohr and the Hessian uses Hartree/Bohr^2 in atomic mode.

## Periodic example: stress and Gamma phonons

Both requests below use a skewed three-dimensionally repeated H2 cell,
an electronic `4,1,1` k mesh, neutral integer RKS/SVWN5/STO-3G, and
zero electronic temperature. All three lattice directions are periodic.
The **phonon wave vector** is Gamma (q = 0), so all images of each atom
move together. The **electronic k mesh** still has four points. The
cell and positions follow the [Gamma phonon fixture](periodic.md#12-gamma-point-force-constants-and-phonons).
Use this input to compare APIs. The settings use the shared defaults of
charge 0, multiplicity 1, zero electronic temperature and no added
dispersion. Evaluate k-mesh, basis, real-space cutoff and grid convergence
for material predictions.

### Periodic reference and functional support

| Reference or model | `stress` | `phonons` (Gamma) |
| --- | --- | --- |
| **RKS / UKS, conventional 3D** | Analytic finite-k stress for HF, semilocal LDA/GGA/meta-GGA, and compatible full- and range-separated exchange. Representative inputs are `hf`, `svwn5`, `pbe`, `tpss` and `pbe0`. | The conventional CPKS route includes those operator classes, including the meta-GGA kinetic-density term. The exact-exchange and scalar ECP terms are connected. The PBE and TPSS scalar-ECP fixtures pass all 81 ordered relaxed entries for each of RKS and UKS at electronic Gamma. Finite electronic k acceptance remains open. |
| **RKS / UKS, extra terms** | Scalar ECP and compatible weighted XC or Skala with a model file have 3D routes. Named or custom double hybrids use a separate 3D, zero-temperature lattice path. | Scalar ECP and the 3D Skala displacement/kernel route are connected. Compatible weighted XC uses the same semilocal/exchange operators. The double-hybrid path uses the Gamma point of the supercell representing the electronic k mesh. These extensions have separate acceptance records. |
| **ROKS / ROHF** | Common-orbital, fixed-integer stress in supported semilocal and exact-exchange models. Field-free variational VV10 is connected for ROKS. Select `xc="hf"` for ROHF. The VV10 force and stress scans across wires, slabs and bulk cells have separate acceptance gates. | The common driver rejects this reference for `phonons`. |
| **GKS / GHF** | Field-free spinor affine stress with zero-temperature occupations and compatible semilocal or exact-exchange operators. GKS also includes variational VV10 in 1D, 2D and 3D. Select `xc="hf"` for GHF. PT2, learned functionals and additional spin/SOC operators require separate providers. | The common driver rejects this reference for `phonons`. |

For self-consistent VV10 references such as omegaB97X-V and omegaB97M-V,
`phonons` uses the stationary lattice Taylor Hessian at electronic Gamma
of the supercell representing the k mesh. It reports per-primitive-cell
force constants and response residuals. All 21 independent quadratic
directions pass for the scalar-ECP RKS omegaB97X-V NaH fixture at
electronic Gamma. Conventional 3D RKS/UKS
`stress` and `periodic-energy` with `forces=true` include VV10 first
geometry derivatives at finite electronic k. The RKS NaH case passes
all 15 ordered strain and force fits. Field-free 1D/2D RKS/UKS use
the same pair kernel with active-axis images and length or area
normalization. A finite-k RKS omegaB97X-V wire and UKS omegaB97M-V slab
each pass all nine force and nine ordered strain fits. Both also pass
CLI, native Python and ASE comparisons. The pair cutoff is
`vv10_cutoff` in Bohr. [VV10 acceptance record](response.md#first-geometry-derivatives-of-variational-periodic-vv10)

The GKS VV10 route has matching public force and ordered stress reports
for a wire, slab and bulk cell. Its independent finite-k relaxed-energy
derivative fits are tracked separately.
[Spinor VV10 scope and evidence](response.md#first-geometry-derivatives-of-variational-periodic-vv10)

OmegaB97X-D includes the geometry-only CHG pair term in stress,
Gamma phonons, SOEC and the conventional polar routes.
For SOEC and relaxed-ion piezoelectricity, its strain, mixed and
nuclear curvatures enter before internal relaxation. For partial
PBC, `stress` can use the dimensional driver with length (1D) or area
(2D) normalization. The conventional `phonons` route and its ASE getter
require all three periodic axes. Applied fields use separate model
restrictions. [Stress scope and measurements](response.md#r026-stress-elastic-high-elastic)
and [Gamma phonon scope and measurements](response.md#r020-dynamical-matrix-phonons)

The 3D `periodic-quartic-fc` and `periodic-quartic-fc-along` routes
include D2, D3(zero), D3(BJ), their ATM variants and the CHG term for
Gamma nuclear displacements with supported references and parameters.
The 3D `toec` and `foec` strain paths also accept these empirical models
and D2.
The energy of the selected pair and ATM images follows the same Lagrangian
deformation and ion motion as the electronic energy. Its mixed derivatives
enter the internal relaxation equations. See the current acceptance status.
[Higher elastic constants](response.md#charged-gamma-force-constants-and-higher-elastic-constants)
[Periodic dispersion tests](response.md#seven-crystal-system-dispersion-acceptance)
[D2 model and verification](response.md#periodic-d2-empirical-dispersion)

### Periodic CLI

Save `cell.xyz` with the following positions and lattice rows in
Angstrom. The extended XYZ header supplies the cell and PBC to both
commands. With ordinary XYZ, give the same rows using
`--cell "1.905037959250800,0,0;0.158753163270900,3.704240476321000,0;0,0.211670884361200,3.704240476321000" --pbc 1,1,1`.

```text
2
Lattice="1.905037959250800 0 0 0.158753163270900 3.704240476321000 0 0 0.211670884361200 3.704240476321000" pbc="T T T"
H 0.0                0.0                0.0
H 0.767306955809350  0.052917721090300  0.026458860545150
```

```console
dft-rs stress cell.xyz --xc svwn5 --reference rks --basis sto-3g --k-mesh 4,1,1 --overlap-cutoff 14 --grid-level 3 --grid-shell 3 --grid-weight-cutoff 0 --conv-tol 1e-12 --conv-tol-dp 1e-10 --max-iter 200 --forces=false --format json
dft-rs phonons cell.xyz --xc svwn5 --reference rks --basis sto-3g --k-mesh 4,1,1 --overlap-cutoff 14 --grid-level 3 --grid-shell 3 --grid-weight-cutoff 0 --conv-tol 1e-12 --conv-tol-dp 1e-10 --max-iter 200 --response-tolerance 1e-10 --format json
```

### Periodic native Python and ASE

Native Python takes Cartesian positions and lattice rows in Bohr. ASE
takes them in Angstrom. `units="atomic"` keeps the ASE report and its
dimensionful controls in the same units as the native call.

```python
import dft_rs
from ase.io import read

numbers = [1, 1]
positions = [[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]]
cell = [[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]
pbc = [True, True, True]
settings = dict(
    xc="svwn5", reference="rks", basis="sto-3g", k_mesh=[4, 1, 1],
    overlap_cutoff=14.0, grid_level=3, grid_shell=3, grid_weight_cutoff=0.0,
    conv_tol=1e-12, conv_tol_dp=1e-10, max_iter=200,
)
stress = dft_rs.stress(numbers, positions, cell, pbc=pbc, forces=False, **settings)
phonons = dft_rs.phonons(
    numbers, positions, cell, pbc=pbc, response_tolerance=1e-10, **settings,
)

atoms = read("cell.xyz", format="extxyz")
calc = dft_rs.DftRs(units="atomic", **settings)
ase_stress = calc.get_periodic_stress_report(atoms, forces=False)
ase_phonons = calc.get_phonons(atoms, response_tolerance=1e-10)

print(stress["stress"], ase_stress["stress"])
print(phonons["frequencies_cm1"], ase_phonons["frequencies_cm1"])
```

### Periodic Rust

```rust
use dft_rs::driver::{run, Controls, System, Value};

fn main() -> dft_rs::Result<()> {
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.0, 0.0, 0.0], [1.45, 0.1, 0.05]],
        cell: Some([[3.6, 0.0, 0.0], [0.3, 7.0, 0.0], [0.0, 0.4, 7.0]]),
        pbc: [true; 3],
    };
    let settings = Value::object()
        .with("xc", "svwn5")
        .with("reference", "rks")
        .with("basis", "sto-3g")
        .with("k_mesh", [4, 1, 1])
        .with("overlap_cutoff", 14.0)
        .with("grid_level", 3)
        .with("grid_shell", 3)
        .with("grid_weight_cutoff", 0.0)
        .with("conv_tol", 1e-12)
        .with("conv_tol_dp", 1e-10)
        .with("max_iter", 200);
    let stress = run(
        "stress",
        &system,
        &Controls::from_value(&settings.clone().with("forces", false))?,
    )?;
    let phonons = run(
        "phonons",
        &system,
        &Controls::from_value(&settings.with("response_tolerance", 1e-10))?,
    )?;
    println!("{:?}", stress.get("stress"));
    println!("{:?}", phonons.get("frequencies_cm1"));
    Ok(())
}
```

### Compare the periodic reports

| Calculation | Fields to compare in this example |
| --- | --- |
| Stress | `stress` is the 3 by 3 Cartesian tensor in Hartree/Bohr^3. `stress_voigt` is ordered xx, yy, zz, yz, xz, xy. Positive diagonal components denote tension. `pressure = -tr(stress)/3`. Check `orbital_residual` and `rotation_residual`. |
| Gamma phonons | `force_constants` and `raw_force_constants` are 6 by 6 in Hartree/Bohr^2, with indices `3*atom + axis`. `frequencies_cm1` uses negative values for imaginary modes. Check `raw_asymmetry`, `acoustic_residual` and each `response_solves[*].relative_residual`. |

The dedicated stress route includes forces by default. The examples
disable them explicitly. Response solver controls belong on the phonon
request in this example. To get forces and stress in one SCF, use
`periodic-energy` with `forces=true` and `stress=true`. For this 3D cell,
standard ASE `atoms.get_stress()` returns Voigt stress in eV/Angstrom^3
when a calculator is attached. Use the report getter above for raw
diagnostics and atomic-unit comparisons. For phonon dispersion use
`phonon-bands` with its explicit supercell and reduced q path. Directional
LO-TO response belongs to the documented `crystal-ir` route.

### Choose elastic constants and strain diagnostics

Use `elastic` for supported three-dimensional collinear references.
The matching entries are `dft-rs elastic`, `dft_rs.elastic`,
`DftRs.get_elastic` and `driver::run("elastic", ...)`. Reuse the
geometry and reference controls of the stress calculation and add the
response tolerance. Check `orbital_residual` and each
`strain_solves[*].relative_residual` before interpreting the tensor.

| Quantity | Report field and convention |
| --- | --- |
| Raw curvature of energy per cell | `strain_hessian[3*a+b][3*c+d]` is in Hartree. It differentiates the Cartesian chart `F = I + eps`. |
| Material constants | `clamped_ion_gpa` and `relaxed_ion_gpa` use Lagrangian strain and Voigt order xx, yy, zz, yz, xz, xy. Relaxed ions include the internal displacement induced by strain. |
| Stress-strain constants | `clamped_ion_birch_gpa` and `relaxed_ion_birch_gpa` include the reference-stress contribution. |

When checking the raw Hessian against separate `stress` reports at
displaced cells, use the unsymmetrized `strain_derivative` rather than
the volume-normalized `stress`. The derivative of a displaced cell's
local component `[a][b]` with respect to reference strain `eps[c][d]`
is `H[3*a+b][3*c+d] + delta(b,c) * strain_derivative_at_zero[a][d]`.
The extra term comes from composing the two Cartesian deformations.
Audit the AO image and grid cutoffs over all displaced cells.
[Model restrictions and charged-cell measurements](response.md#r026-stress-elastic-high-elastic)

## Run a uniform magnetic-flux SCF

Choose the API by the field and the quantity to be computed.

| Field or observable | Use | Geometry and reference |
| --- | --- | --- |
| Uniform field on a wire, or a uniform field in the plane of a slab | `periodic-spinor-scf` with `uniform_magnetic_field` | Wire in any field direction or slab with zero normal field. Ordinary Bloch translations and a complex spinor reference. |
| Uniform field through a slab or bulk face | `periodic-magnetic-scf` with `magnetic_field` | Flux-compatible slab or bulk. Select the finite magnetic cell or a magnetic Bloch mesh. |
| Real sinusoidal vector potential | `periodic-field-scf` for a collinear reference, or `periodic-spinor-scf` for a generalized reference | Wave must be periodic in the supplied cell. The potential is `2 * field_amplitude * Re(A_q exp(i q.r))`. Its coefficient `A_q` comes from `field_vector_real` and `field_vector_imag`. |
| Linear orbital susceptibility at nonzero q | [`orbital-susceptibility-q`](#orbital-susceptibility-at-finite-q) | Insulating 3D Gamma supercell with q commensurate with the mesh. Collinear and spinor references have distinct provider limits. |

`periodic-magnetic-scf` uses continuum London orbitals on a physical 2D
slab or 3D bulk cell. Supply `magnetic_field` in atomic units and choose
a field for which the flux through each periodic face divided by
`2*pi` is an integer. Read `flux_quanta` to confirm the sector.

| Interface | Entry |
| --- | --- |
| Rust | `driver::run("periodic-magnetic-scf", &system, &controls)` |
| CLI | `dft-rs periodic-magnetic-scf cell.xyz --cell ... --pbc 1,1,0 --magnetic-field ... --reference gks --magnetic-model crystal_bz --k-mesh 2,1,1 --format json` |
| Native Python | `periodic_magnetic_scf(numbers, positions_bohr, cell_bohr, **controls)` |
| ASE | `DftRs(units="atomic").get_periodic_magnetic_scf(atoms, **controls)` with `Atoms` geometry in Angstrom |

Use `magnetic_model="finite_gamma"` with `k_mesh=[1,1,1]` for the finite
cell. Use `magnetic_model="crystal_bz"` for magnetic Bloch integration.
Both expose HF, LDA, GGA and meta-GGA spinor SCF with compatible
full/range exchange. Omit `reference` to use the default GKS label,
or set it explicitly to `gks`. Use `reference="ghf"` for an HF
energy provider. Check `reference_type` in the result. Use `xc`
for a named functional or `xc_terms` with explicit weights for a
compatible composition. ECP, dispersion, nonlocal XC and moving
semilocal nuclear response have separate provider requirements.
Use `nuclear_order`
with the compatible exchange-only finite-Gamma provider. A 3D insulating
finite-Gamma reference can also take a fixed Cartesian `field` in
Hartree/(e Bohr).

| Control | Default and interpretation |
| --- | --- |
| `magnetic_origin`, `current_points` | Explicit coordinates in Bohr. Current samples at point nuclei are undefined. |
| `magnetic_grid_order` | Starts at `16 + 4*grid_level` with spatial refinement up to 128. LDA and GGA use the tensor-product rule. Meta-GGA uses four times as many radial points and splits the first `1e-5` Bohr into Gauss intervals at Gaussian-product centers. With those centers, the high-order ladder compares degree-131 Lebedev and degree-151 spherical product rules. |
| `magnetic_open_order` | Starts at 64 for the slab normal integral. Valid initial orders are multiples of four from 32 through 512. |
| `magnetic_reciprocal_cutoff` | 32 inverse Bohr. |
| `magnetic_exchange_auxiliary_width` | 0.5 inverse Bohr for crystal exchange. Zero selects raw punctured quadrature. |
| `magnetic_operator_tolerance` | `1e-10` for the raw integral and quadrature convergence checks. |
| `nuclear_soc_scale`, `spinor_amplitude` | Both default to one. The latter scales the supplied Hermitian spinor operator. |
| `charge_compensation` | A charged slab uses explicit Gaussian compensation. Uniform background compensation requires 3D. |

Read `scf_converged`, `orbital_residual` and `particle_number_error`
for the stationary spinor solve. The reported `converged` also
requires `backend_converged`, which includes the spatial energy,
Fock and number checks in `xc_grid_refinement_history`. Each row
gives the fine grid order, energy difference in Hartree per cell,
Fock difference in Hartree and number error in electrons. GGA and meta-GGA
also report `xc_diagnostics.angular_refinement_history`, with rows
containing angular order, energy defect in Hartree per magnetic cell,
and operator defect in Hartree. Converge reciprocal cutoff, spatial
quadrature and the k mesh independently for your cell.

The Gaussian-product centers refine the integration region where
overlapping AO images contribute. They enter only the quadrature.
Gradient, kinetic-density and current terms use one shared evaluation
of the London orbitals and their three kinetic-momentum components.
Large grids perform this evaluation by block when a
complete collocation cache would exceed 512 MiB or half of the
remaining configured memory budget. The same quadrature and spatial
convergence criteria apply to both memory paths.
The reported spatial defects are measured at the returned SCF density.
Rust diagnostic queries at an identical density can reuse the retained defects.
On a skew H2 slab with M06-L, CLI, native Python and ASE give
`1.939727501367524` Hartree per cell at finite Gamma and
`1.9397275013680506` Hartree per cell on a `2 x 1 x 1` magnetic Bloch
mesh. The spatial Fock defects are about `2.85e-11` Hartree.
[Measured slab energies and Fock checks](response.md#covariant-semilocal-scf-on-uniform-flux-cells)
[Fixed-flux nuclear response](rust-api.md#nuclear-response-on-a-finite-uniform-flux-periodic-reference) · [Joint electric and magnetic bias](rust-api.md#simultaneous-electric-and-magnetic-bias-on-a-finite-periodic-cell)

### Orbital susceptibility at finite q

Use `orbital-susceptibility-q` for a nonzero reduced wave vector
commensurate with the k mesh, for example `q_points=[0.25, 0, 0]`
with `k_mesh=[4, 1, 1]`. The Gamma supercell uses sinusoidal vector
potentials in each of two transverse directions.

| Reference | Functional and field scope | Result |
| --- | --- | --- |
| `rks` / `uks` | Real collinear HF, LDA, GGA, meta-GGA and compatible hybrids. Meta-GGA includes the current derivative of the gauge-invariant kinetic density. | Two diagonal transverse responses. |
| `gks` / `ghf` | Complex spinor semilocal LDA, GGA, current-corrected meta-GGA and compatible full- and long-range exact exchange, or HF. A fixed local spin texture can use `field_wave_indices`, `spin_field_real` and `spin_field_imag` on the supercell. A uniformly compensated charged cell is allowed. | The two diagonal responses and the symmetric `chi_transverse` matrix in `field_direction` axes. |

For the spinor route, use three periodic axes, integer insulating
occupations and an all-electron basis. Finite temperature, nonlocal XC,
explicit orbital bias, nuclear SOC and nonlocal ECP require separate
response vertices. Local spin controls define the stationary reference;
each orbital probe leaves the spin bias fixed.

The spinor meta-GGA kernel retains all spin components of the
paramagnetic current. In each projected spin channel the explicit vector
potential cancels from `tau_A - |j_A|^2/(2 rho)`. The response uses that
kinetic-density derivative together with the orbital current vertex.

The periodic atom-centred grid uses angular pruning by default. A large
`xc_grid_number_defect` can persist when the radial and angular levels
increase, because pruning retains low angular orders on selected radial
shells. Set `grid_prune=false` (`--grid-prune=false` in the CLI) to
use the full angular order on every shell. `grid_weight_cutoff` separately
controls the removal of points with small partition weights.

For `A = a e cos(q.r)` or its sine counterpart, the mean squared
magnetic field is `a^2 q^2/2`. The method subtracts half the
uniform-potential paramagnetic term from the sine/cosine mean
to impose the diamagnetic sum rule in a finite basis.
`points[*].chi` is the volume susceptibility in atomic units and
`chi_cgs` and `chi_si` are dimensionless. The two rows of
`points[*].response_residuals` correspond to the transverse
polarizations. Read cosine, sine and uniform-potential residuals in
that order in each row.
For spinors, `paramagnetic_transverse` and `uniform_transverse` hold the
quadratic energy coefficients per primitive cell. The report gives the
maximum imaginary and transpose defects of the raw static response
before forming the symmetric tensor.
[Reference scope and transverse spectral validation](response.md#commensurate-wave-vectors)

This compact TPSS example uses a `2,1,1` mesh and the reduced wave vector
`[0.5, 0, 0]`. Save the one-primitive basis as `compact-h.gbs`.

```text
H 0
S 1 1.0
0.7 1.0
****
```

Save the skew H2 cell as `orbital-q.xyz` in Angstrom.

```text
2
Lattice="1.905037959250800 0.100543670071570 -0.121710758507690 0.164044935379930 3.704240476321000 0.142877846943810 -0.116418986398660 0.216962656470230 3.704240476321000" pbc="T T T"
H 0.058209493199330 -0.068793037417390 0.089960125853510
H 0.814932904790620  0.089960125853510 -0.063501265308360
```

```console
dft-rs orbital-susceptibility-q orbital-q.xyz --xc tpss --reference rks --basis ./compact-h.gbs --k-mesh 2,1,1 --q-points 0.5,0,0 --overlap-cutoff 12 --grid-level 2 --grid-shell 2 --grid-weight-cutoff 0 --conv-tol 1e-13 --conv-tol-dp 1e-12 --conv-tol-grad 1e-11 --response-tolerance 1e-11 --cpu-threads 1 --format json
```

```python
import dft_rs
from ase.io import read

positions = [[0.11, -0.13, 0.17], [1.54, 0.17, -0.12]]
cell = [[3.6, 0.19, -0.23], [0.31, 7.0, 0.27], [-0.22, 0.41, 7.0]]
response = dict(
    xc="tpss", reference="rks", basis="./compact-h.gbs", k_mesh=[2, 1, 1],
    q_points=[0.5, 0.0, 0.0], overlap_cutoff=12.0, grid_level=2,
    grid_shell=2, grid_weight_cutoff=0.0, conv_tol=1e-13, conv_tol_dp=1e-12,
    conv_tol_grad=1e-11, response_tolerance=1e-11, cpu_threads=1,
)
native_q = dft_rs.orbital_susceptibility_q(
    [1, 1], positions, cell, pbc=[True, True, True], **response,
)
atoms = read("orbital-q.xyz", format="extxyz")
ase_q = dft_rs.DftRs(units="atomic").get_orbital_susceptibility_q(atoms, **response)
print(native_q["points"][0]["chi"], ase_q["points"][0]["chi"])
```

```rust
use dft_rs::driver::{run, Controls, System, Value};

fn main() -> dft_rs::Result<()> {
    let system = System {
        numbers: vec![1, 1],
        positions: vec![[0.11, -0.13, 0.17], [1.54, 0.17, -0.12]],
        cell: Some([[3.6, 0.19, -0.23], [0.31, 7.0, 0.27], [-0.22, 0.41, 7.0]]),
        pbc: [true; 3],
    };
    let settings = Value::parse_json(r#"{
        "xc": "tpss", "reference": "rks", "basis": "./compact-h.gbs",
        "k_mesh": [2, 1, 1], "q_points": [0.5, 0.0, 0.0],
        "overlap_cutoff": 12.0, "grid_level": 2, "grid_shell": 2,
        "grid_weight_cutoff": 0.0, "conv_tol": 1e-13, "conv_tol_dp": 1e-12,
        "conv_tol_grad": 1e-11, "response_tolerance": 1e-11, "cpu_threads": 1
    }"#)?;
    let report = run("orbital-susceptibility-q", &system, &Controls::from_value(&settings)?)?;
    println!("{}", report.to_json());
    Ok(())
}
```

The two `chi` values are approximately `-6.00170949e-4` and
`-5.46320549e-4` atomic units for this basis, cell and grid.

## Set the reference and functional

Reference and functional are separate choices. `reference="rks"` selects
a restricted reference; `xc="pbe0"` selects the named global hybrid.
UKS, common-orbital ROKS and complex-spinor GKS have distinct stationarity
conditions. With `xc="hf"` they select the corresponding HF model where
the requested property supports it. Charge and multiplicity still have to
define a valid electron configuration.

| Setting | Python / Rust control | CLI spelling |
| --- | --- | --- |
| Reference | `reference="uks"`, `charge=0`, `multiplicity=2` | `--reference uks --charge 0 --multiplicity 2` |
| Named functional | `xc="pbe0"` | `--xc pbe0` |
| Separate exchange/correlation | `x="pbe"`, `c="pbe"` | `--x pbe --c pbe` |
| Explicit weighted composition | `xc_terms=[{"xc": "pbe0", "weight": 0.8}, {"xc": "hf", "weight": 0.2}]` | `--xc-terms` followed by one JSON argument |
| Numerical quadrature | `grid_level`, `grid_weight_cutoff`, `grid_prune` on periodic atom-centred grids | `--grid-level`, `--grid-weight-cutoff`, `--grid-prune` |
| SCF tolerance | `conv_tol`, `conv_tol_dp`, `conv_tol_grad` | Corresponding hyphenated flags |
| Response tolerance | `response_tolerance` on applicable routes | `--response-tolerance` |
| Resources | `cpu_threads`, `memory_mb` | `--cpu-threads`, `--memory-mb` |

Use `dft-rs --list-functionals`, `dft_rs.functionals()` or Rust
`dft_rs::xc::catalogue::catalogue()` to inspect the registered names and
their recipe terms. Use a `combined` or `double-hybrid` entry as `xc`,
an `exchange` entry as `x`, and a `correlation` entry as `c`. Then look
up the property and reference in the response ledger. That entry gives
the numerical checks for the requested model and boundary conditions.
[Functional catalogue](FUNCTIONALS.md) · [Response coverage ledger](response.md#response-coverage-ledger)

Choose one of `xc`, `x`/`c`, or `xc_terms`. Composition weights are used
as supplied; they are not normalized. The current weighted XC resolver
supports local parts, full-range exchange and one distinct long-range
exchange kernel. Weighted learned, nonlocal and double-hybrid terms have
separate missing operators; they are not enabled by putting their names
in an array. A supported custom PT2 parent is a separate control contract.
See [weighted molecular XC](rust-api.md#weighted-molecular-xc),
[weighted periodic XC](rust-api.md#weighted-periodic-xc) and the
[functional catalogue](FUNCTIONALS.md).

Passing an unsupported reference or provider must produce a capability
error. Testing representative functionals or every registered local kernel
does not establish every complete molecular/periodic response combination.

## Set an electronic temperature

Set `electronic_temperature` in Kelvin. For a molecule, select
`mermin-free-energy` to read both the internal energy `E` and the
stationary electronic free energy `A = E - T_e S_e`. For a cell,
`periodic-energy` reports the corresponding values per cell.

| Reference | Molecular finite-temperature entry | Occupation constraint |
| --- | --- | --- |
| RKS / UKS | `mermin-free-energy` | Fixed electron counts in the two collinear spin channels |
| ROKS / ROHF | `mermin-free-energy` or the thermal fields in `energy` | Shared spatial orbitals and fixed electron counts in the two spin channels |
| GKS / GHF | `mermin-free-energy` or the thermal fields in `energy` | Fixed total electron count, one chemical potential and one occupation per spinor |

Use `gradient` and `hessian` for isothermal nuclear derivatives of `A`
with ROKS or GKS. The analytic Hessian relaxes the density and occupations
at fixed temperature and electron counts. The electronic temperature is
distinct from the nuclear temperature in RRHO thermochemistry.

For example, use the same open-shell XYZ geometry, charge, multiplicity,
temperature and reference in each interface. This example uses the
hydrogen molecular ion:

```bash
dft-rs hessian hydrogen.xyz --reference roks --xc wb97x-d --basis sto-3g \
  --charge 1 --multiplicity 2 --electronic-temperature 120000 --format json
```

```python
from dft_rs import DftRs, hessian_report

settings = dict(reference="roks", xc="wb97x-d", basis="sto-3g", charge=1,
                multiplicity=2, electronic_temperature=120000.0)
native = hessian_report(numbers, positions_bohr, **settings)
ase = DftRs(units="atomic").get_hessian_report(atoms, **settings)
```

```rust
use dft_rs::driver::{run, Controls, System, Value};

let system = System::molecule(numbers, positions_bohr);
let settings = Value::object()
    .with("reference", "roks")
    .with("xc", "wb97x-d")
    .with("basis", "sto-3g")
    .with("charge", 1)
    .with("multiplicity", 2)
    .with("electronic_temperature", 120_000.0);
let hessian = run("hessian", &system, &Controls::from_value(&settings)?)?;
```

Use an XYZ file and ASE atoms in Angstrom and native Python or Rust
positions in Bohr. Read `units` for the gradient in Hartree/Bohr and
the Hessian in Hartree/Bohr².
Read `response_residuals` and `thermal_number_defects` from the Taylor
report. For ROKS, `nuclear_stationarity_defects` combines the orbital
gradient and the diagonal Fock-level equations. For GKS,
`thermal_stationarity_defects` measures the spinor density against the
Fermi matrix. The nuclear response requires a matching analytic
moving-geometry provider. Thermal PT2 and the periodic extension of this
ROKS/GKS molecular Taylor response remain open.
[Finite-temperature verification and scope](response.md#molecular-finite-temperature-roks-and-gks)

## Choose a molecular relativistic Hamiltonian

| Model | Feature and controls | Reference |
| --- | --- | --- |
| Scalar X2C, DKH2 or matrix ZORA | `two-component-relativistic`, `relativistic_method="x2c"` / `"dkh2"` / `"matrix-zora"`, `relativistic_spin="scalar"` | Even-electron RKS |
| Variational spin–orbit X2C, DKH2 or matrix ZORA | `two-component-relativistic`, same method selection and `relativistic_spin="spin-orbit"` | Even-electron complex GKS |
| Four-component no-pair Dirac–Coulomb KS | `dirac-dks`, `dirac_interaction="coulomb"` | Even-electron complex GKS in the fixed positive nuclear-core sector |

All three models use isolated all-electron molecules and require one
positive Gaussian RMS nuclear radius per atom. Supply radii in Bohr through
Rust, the CLI and native Python. ASE takes radii in Angstrom by default and
in Bohr with `units="atomic"`. Set `response_order=1` for electric
dipole polarizability and supply signed frequencies and damping for the
retarded response. The report contains the orbital and decoupling
residuals in Hartree and dimensionless residuals for the electric solve.
The four-component report also contains the spectral gap between the
negative and electronic sectors, in Hartree.

Use `relativistic_method="dirac"` with the existing molecular dipole,
polarizability, dynamic polarizability, quadrupole polarizability or
multipole polarizability entry to select the same positive-energy sector.
The CLI uses `--relativistic-method dirac`. For quadrupole response the
provider returns the quadrupole block and both mixed electric blocks.
The five `electric_field_gradient` values represent the traceless
gradient modes in the report. Use the same origin for the perturbation
and moment when comparing finite-field values.
[Model and coverage](response.md#r075-breit-and-two-component-relativistic-dft)

## Read units, tensors and convergence

Read `units` (or the dedicated API's `unit_labels`) per field. Native
coordinates and energies use Bohr/Hartree, but frequency, ppm, MHz, Kelvin
and other report quantities retain their documented units.

- An energy gradient is `dE/dR`. Force is its negative. Cartesian Hessian
  indices are `3*atom + axis`, with zero-based atoms and axes x/y/z.
- Taylor coefficients contain `1/n!`. For an energy series, the third and
  fourth derivatives are `6*c[3]` and `24*c[4]`. An explicitly returned mixed
  derivative is already a derivative. Directional output is not a full tensor.
- A dynamic tensor can have real and imaginary parts. Read the signed
  frequency, damping and Fourier convention; do not drop its imaginary part.
- Molecular multipoles depend on the chosen expansion origin. A
  molecule-attached origin makes a translated calculation reproducible;
  it does not make every multipole invariant under an arbitrary origin change.
- Inspect the property's response, reference, stationarity, particle-number
  and frame diagnostics when provided. Field names and residual units vary
  by route. `Report.require_converged()` checks the top-level flag only;
  it does not certify stability, basis/grid convergence or every response flag.
- `memory_mb` is a managed-memory budget in MiB, not a whole-process RSS cap.

See [shared result conventions](rust-api.md#result-conventions-and-physical-scope)
and the selected property's detailed row for tensor normalization, units
and exact diagnostics.

## Read a complex quadrupole response

Use the `quadrupole-polarizability` driver route for the five Buckingham
modes. The CLI reads the geometry in Angstrom:

```console
dft-rs quadrupole-polarizability molecule.xyz --xc svwn5 --reference gks --frequency 0.02 --frequency-damping 0.005 --format json
```

With atomic numbers and Cartesian positions in Bohr, the native call is:

```python
report = dft_rs.quadrupole_polarizability(
    numbers, positions, xc="svwn5", reference="gks",
    frequency=0.02, frequency_damping=0.005,
)
c00 = complex(
    report["quadrupole_polarizability_real"][0][0],
    report["quadrupole_polarizability_imag"][0][0],
)
print(c00, report["mode_order"][0], report["response_residuals"])
```

ASE uses `get_quadrupole_polarizability(atoms, units="atomic", ...)`.
Rust uses `driver::run("quadrupole-polarizability", ...)` with the same
named controls. A geometry's own tensor depends on its basis, reference
and origin. The following values are from the recorded test fixture.

The retarded Buckingham tensor evaluates the response at complex photon
energy `z = omega + i eta`. In a recorded GKS/SVWN5 calculation at
`omega = 0.02` Hartree, reducing the broadening changes the same component:

| Broadening eta (Hartree) | C[0,0] (e^2 Bohr^4/Hartree) |
|---:|---:|
| 0.005 | 23.49094819276935 + 0.625364356028947 i |
| 0.0005 | 23.637355263701 + 0.065055596366601 i |
| 0 | 23.638885450557 - 4.618e-14 i |

The imaginary part is the out-of-phase response and depends on the
broadening and proximity to poles. At this frequency it approaches zero
with the broadening. The largest relative response residual across these
runs and the separate static solve is 9.681e-11. For real perturbation
operators, the retarded signed-frequency check pairs C(-omega + i eta)
with the conjugate of C(omega + i eta). Use the same origin and mode
normalization for both requests. [Measured response and conventions](response.md#r060-molecular-multipoles-and-derivatives)

## Read a generalized double-hybrid gradient

The molecular gradient driver accepts an integer GKS reference with
equal same- and opposite-spin PT2 weights. Select a registered recipe
such as B2PLYP, or use a custom parent and target with matching
`pt2_same_spin` and `pt2_opposite_spin` values. Unequal weights require
a spin-covariant pair model and return an unsupported-combination error.
The registered unequal-weight recipes are DSD-PBEP86, DSD-BLYP,
DSD-PBEB95, omegaB97X-2(LP), omegaB97X-2(TQZ), PWPB95, revDSD-PBEP86
and xDH-PBE0. The ROKS double-hybrid energy uses a common-orbital
reference. Its PT2 nuclear and electric derivatives require the
corresponding common-orbital response.

For a doublet NH3 cation, pass B2PLYP, GKS, charge +1 and multiplicity 2
through the common gradient entry on each interface. Native Python
positions use Bohr, and the XYZ file and ASE positions use Angstrom.

| Interface | Call |
| --- | --- |
| Rust | `driver::run("gradient", &system, &controls)` with `reference="gks"`, `xc="b2plyp"`, `charge=1` and `multiplicity=2` |
| CLI | `dft-rs gradient ammonia.xyz --reference gks --xc b2plyp --charge 1 --multiplicity 2 --format json` |
| Native Python | `dft_rs.gradient(numbers, positions, reference="gks", xc="b2plyp", charge=1, multiplicity=2)` |
| ASE | `calc.get_gradient_report(atoms, units="atomic", reference="gks", xc="b2plyp", charge=1, multiplicity=2)` |

The total `gradient` equals `parent_gradient` plus the target correction
`target_gradient`, weighted `pt2_gradient` and `dispersion_gradient`.
The unweighted spinor-pair derivative is `generalized_pt2_gradient`.
`forces` is the negative total gradient. These Cartesian arrays are
in Hartree/Bohr in atomic reports and eV/Angstrom in ASE physical reports.
Jointly check `converged`, `response_converged`, dimensionless
`response_residuals` and `spinor_nuclear_pair_diagnostics`.
The latter reports absolute and relative pair residuals for each
Cartesian perturbation, with units tagged by Taylor order.
[Measured B2PLYP gradient and remaining scope](response.md#screened-and-double-hybrid-checkpoints)

## Choose a Hessian and minimum check

| Intended calculation | Shared route | Coordinates and interpretation |
| --- | --- | --- |
| Ordinary molecular force constants | `hessian`; native `hessian_report`; ASE `get_hessian_report` | All nuclear Cartesian coordinates; RKS/RHF and UKS/UHF share the analytic implementation. Read the provider-specific coverage. |
| Double-hybrid molecular force constants | The same `hessian` route with a named recipe or explicit custom PT2 weights | The parent orbitals, target energy and PT2 amplitudes all respond. The report separates total, parent, target, SS-alpha, OS and SS-beta contributions. |
| Ordinary molecular vibrations and thermal corrections | `frequencies` | Run at the optimized geometry with the same method, charge and spin. Inspect imaginary modes before using the RRHO result. |
| Saved continuous-species force constants | `continuous-species`, `fragment_mode="scalar-hessian"`; native `evaluate_reduced_fragment`; ASE `get_reduced_fragment` | One explicit saved state and endpoint, in its raw coordinate chart. Frozen internal offsets are retained. This is constrained model curvature. |
| Saved substituent weight optimization | `continuous-species`, `fragment_mode="scalar-optimize"` | Optimizes endpoint weights, not molecular coordinates. Its convergence does not imply a stationary geometry. |

For the ordinary stationary RKS/UKS Hessian, `orbital_response_residuals` reports the
fresh CPKS equation defect for every Cartesian perturbation, in Hartree/Bohr.
The Hessian uses Hartree/Bohr^2 before mass weighting; ASE physical reports
convert these to eV/Angstrom and eV/Angstrom^2. `raw_asymmetry` precedes
symmetrization. Retain these measurements together with the SCF residual.

For an RKS or UKS double hybrid, `response_residuals` instead contains dimensionless
original orbital-equation residuals; `amplitude_response_residuals` is in
Hartree. Read the separate metric, projector, number and physical
stationarity defects as well. All returned gradients and Hessians use the
same Cartesian ordering; the SS-alpha, OS and SS-beta tensors are unscaled.
Their explicit SS/OS coefficients form `pt2_gradient` and `pt2_hessian`.
The total also includes the target and declared dispersion terms. Converging
the parent SCF energy alone does not converge or optimize the total
double-hybrid energy.

Set `reference="rks"` for a paired reference or `reference="uks"` with
the intended `charge` and `multiplicity` for independent spin blocks. These
controls pass through `hessian_report`, ASE `get_hessian_report` and CLI
`--reference`, `--charge`, `--multiplicity`. `xc="hf"` selects the
corresponding Hartree-Fock case. A selected provider still needs its full
nuclear-response contract; a registered energy name does not establish
Hessian acceptance. The common analytic report refuses missing kernels.
See [ordinary acceptance](response.md#restricted-and-unrestricted-hessian-residual-checkpoint)
and [open-shell PT2 component acceptance](response.md#open-shell-correlated-nuclear-component-acceptance)
for their distinct tested scopes.

The saved-model Hessian additionally requires `fragment_state` and
`fragment_endpoint`, the declared `geometry_atom_owners`, and the
`fixed-internal-affine-v1` transport contract. Its `raw_hessian` and
`directional_response` retain ordered entries and response diagnostics.
Paired RKS uses `frozen-scalar-action-v1`; UKS requires explicit alpha/beta
blocks and electron counts in `frozen-scalar-collinear-action-v1`. Changing
only a reference string does not convert an RKS record into a UKS model.
These saved evaluators currently admit LDA/GGA providers; their chemical
examples use PBE. See the [measured coverage](response.md) for acceptance.

A saved-model stationary geometry does not establish an ordinary all-atom
minimum. Frozen internal vibrations are absent, and fixed laboratory
offsets do not define a freely rotating molecular frame. Consequently,
ordinary six-mode projection and full molecular RRHO thermochemistry must
not be applied to this constrained Hessian as if it described every atom.

## Resolve similar names

| Names | Difference |
| --- | --- |
| `hessian_report` / `run_feature("hessian", ...)` and legacy `hessian` | The common report requests an analytic route and refuses missing kernels. The legacy function can use its documented, labelled numerical fallback. |
| `polarizability_report` and `dynamic_polarizability` | Static common report versus frequency-dependent response; use the documented static limit when comparing. |
| Pockels, DC Kerr and optical Kerr | Pockels uses `dynamic-beta` with optical and DC inputs. DC Kerr uses `dynamic-gamma` with one optical and two DC inputs. Optical Kerr uses three signed optical inputs. The presets construct these frequencies from the supplied optical magnitude. |
| `nmr-shielding`, `spin-spin-couplings` and `nmr-spectrum` | Electronic shielding tensor, nuclear J tensor, and a spectrum assembled using isotope/reference/broadening inputs. |
| `nmr_spin_model="first-order"` and `"exact"` | First-order line splitting versus diagonalization of the finite isotropic spin Hamiltonian. Neither adds missing electronic shielding/J kernels. |
| `spin-spin-couplings` and `spin-spin-zfs` | Indirect nuclear J coupling versus electronic zero-field splitting. |
| `broken-symmetry-exchange` and nuclear J | Two-site HDVV electronic exchange uses the documented `H=-2J S_A.S_B` convention; it is a different physical coupling. |
| `mermin-free-energy` and `gibbs-free-energy` | Electronic finite-temperature free energy versus molecular thermochemistry; temperature and ensemble differ. |
| `thermal-response` | Uniform static finite-temperature DFPT for a 3D collinear reference. The user chooses a temperature, number, or chemical-potential perturbation and the ensemble. Use `periodic_thermal_response` in native Python, `get_periodic_thermal_response` in ASE, or `PeriodicDensitySolver::solve_thermal` in Rust. [CLI controls](cli.md) [Validation](response.md#thermal-and-coulomb-acceptance-checkpoint) |
| `cubic-fc-along` / `quartic-fc-along` and `force-constants` | Contracted directions versus complete requested force-constant tensors. |
| `xanes` and `nexafs` | `xanes` offers the existing CVS-based core spectrum and core-hole shift. `nexafs` reports Cartesian oscillator-strength tensors from CVS-TDA or the per-hole relaxed-ion model. The latter uses projected, nonorthogonal determinant transition dipoles. [NEXAFS scope](response.md#r085-near-edge-xray-absorption-fine-structure) |
| `constrained-dft` and MOM/IMOM | Constrained DFT adjusts multipliers to reach target expectation values. MOM/IMOM follows a selected determinant on the ordinary `energy` and `gradient` routes. [Occupation controls](rust-api.md#occupied-projector-delta-scf-controls) |
| `dft-plus-u` and `pz-sic` | The Hubbard route adds an energy term that depends on selected site occupation matrices in RKS, UKS or ROKS. PZ-SIC subtracts orbital self-interactions and optimizes the occupied UKS orbitals. [Hubbard site controls](response.md#r087-molecular-dftu) [PZ-SIC scope](response.md#r088-molecular-pz-sic) |
| `stress`, `elastic`, `toec` and `foec` | First through fourth strain derivatives on their supported routes. `elastic` reports both Lagrangian and Birch coefficients, which coincide at zero stress. The third- and fourth-order routes report Lagrangian coefficients. |
| `phonons`, `phonon-bands` and `crystal-ir` | Gamma-point normal modes, supercell force-constant interpolation and infrared response with directional LO-TO terms in supported sectors. [Periodic models](periodic.md#14-phonon-dispersion-and-lattice-thermodynamics) |
| `superconducting-state` | The band route solves a supplied-band canonical gap, pairing onset or phase problem. The Nambu route computes equilibrium densities or retarded probe response on supplied full complex operators. [SCDFT models and coverage](response.md#r050-scdft-gap-anomalous-density) |

For retarded response on a supplied two-orbital Nambu Hamiltonian, use
`frequencies` for the signed energies `hbar*omega`. This native Python input
uses Hartree and two dimensionless probes, one normal and one pairing.

```python
from dft_rs import superconducting_state

report = superconducting_state(
    pairing_model="nambu",
    nambu_normal_real=[[-0.11, 0.03], [0.03, 0.09]],
    nambu_pairing_real=[[0.0, 0.027], [-0.027, 0.0]],
    chemical_potential=-0.013,
    electronic_temperature=300.0,
    nambu_probe_normal_real=[[0.29, 0.0, 0.0, -0.11], [0.0, 0.0, 0.0, 0.0]],
    nambu_probe_pairing_real=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.16, -0.16, 0.0]],
    frequencies=[0.067, -0.067],
    frequency_damping=0.008,
)
diagonal_imaginary = report["nambu_susceptibility_imaginary"][0][0][0]
```

In ASE the same inputs go to `DftRs.get_superconducting_state(units="atomic", ...)`.
For default ASE units, express the energy inputs in eV. For CLI, use
`dft-rs superconducting-state --format json`, spell control names with
hyphens, and give the rows and frequencies as JSON arrays. Rust exposes
`response::nambu::nambu_causal_response` with `NambuReference`, `NambuProbe`
and the common frequency controls. All routes report the matrix in
frequency, observed, driven order with inverse energy units and raw
numerical residuals.

## Check applicability and errors

For a reproducible calculation, retain the feature ID, complete controls,
geometry/cell, basis and core model, frequency/gauge/origin conventions,
package version, output units and raw residuals.

A registry row describes reachability. The [coverage ledger](response.md#response-coverage-ledger)
distinguishes implementation from executed validation and records failures.
The bounded [constrained DFT](response.md#r083-constrained-density-functional-theory),
[Car-Parrinello MD](response.md#r084-car-parrinello-molecular-dynamics) and
[NEXAFS](response.md#r085-near-edge-xray-absorption-fine-structure) entries
list their supported models and pending extensions.

Constrained DFT and Car-Parrinello MD use the
[shared molecular functional controls](#set-the-reference-and-functional).

For constrained DFT, select RKS, UKS, ROKS or GKS. Supply one
`density_constraints` entry per independent fragment. For example,
`[{"atoms":[0,1],"electrons":8.96}]` fixes the combined Mulliken
electron population of atoms 0 and 1 to 8.96. Atom indices are zero-based
and fractional population targets are permitted. The total number of
electrons is specified separately by the molecular charge.
Rust code can call `scf::constrained::run_collinear` to reuse the
collinear SCF state and multiplier response. GKS uses the same
multiplier solver with the spinor SCF and response backend.
`gradient=true` requests the analytic energy gradient including
the moving fragment projector. `optimize_geometry=true` relaxes the
nuclei on that constrained surface. `constraint_electric_response=true`
requests the static polarizability with the electron populations held
fixed, also when a static electric bias is specified.

Car-Parrinello MD accepts all-electron, integer-occupation RKS, UKS
and ROKS with real orbitals, and GKS with complex spinors. The shared
RATTLE integrator uses the instantaneous symmetric Löwdin AO frame.
Real-reference off-shell derivatives and trajectories have measured
acceptance. Complex derivative and trajectory acceptance and correlated
off-shell PT2 forces remain open.

The scientific regressions and the clean-install CLI verifier require
both `xc="wb97x-d"` (CHG dispersion) and `xc="wb97x-v"` (variational
VV10) for these methods. See the linked ledger entries for the
validated reference and model combinations.

Unsupported physics raises a named Rust capability error and Python
`NotImplementedError`. Malformed inputs, SCF/response failures and resource
errors retain distinct diagnostics; the CLI returns a nonzero exit status.
Unknown or inapplicable controls are rejected, so check the selected route's
signature before copying settings from another calculation.


## Difficult molecular SCF

Select `scf_optimizer="trah"` (CLI `--scf-optimizer trah`) to request
the trust-region augmented-Hessian orbital solver. Use `"auto"` for
automatic recovery and `"legacy"` to reproduce the preceding algorithm.
Inspect the physical SCF residual separately from the printed inner AH
residual. Parent-SCF convergence does not establish PT2 accuracy or ground-state
stability. See the detailed [solver controls](rust-api.md#trust-region-augmented-hessian-scf)
and [measured acceptance](response.md).
