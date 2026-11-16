// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Dependency-free command-line frontend for dft-rs.
//!
//! XYZ input/output is Ångström; printed energies are Hartree (with eV and
//! kcal/mol conversions in summaries). Keeping this parser in-tree avoids a
//! CLI dependency for the core library.

use dft_rs::math::Vec3;
use dft_rs::Molecule;

fn parse_vector(value: &str, flag: &str) -> Result<[f64; 3], String> {
    let values = value
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid {flag}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != 3 || values.iter().any(|x| !x.is_finite()) {
        return Err(format!(
            "{flag} requires three finite comma-separated values"
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn parse_field_complex(
    value: &str,
    flag: &str,
) -> Result<dft_rs::response::complex::Complex, String> {
    let v = value
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("invalid {flag}"))?;
    if v.len() != 2 || v.iter().any(|x| !x.is_finite()) {
        return Err(format!("{flag} requires finite real,imag"));
    }
    Ok(dft_rs::response::complex::Complex::new(v[0], v[1]))
}

fn print_ao_density_rows(density: &[Vec<dft_rs::periodic::BlochMatrix>; 2]) {
    println!("# density spin k bra_AO ket_AO real imaginary; units as in report");
    for (spin, points) in density.iter().enumerate() {
        for (k, matrix) in points.iter().enumerate() {
            for i in 0..matrix.rows() {
                for j in 0..matrix.cols() {
                    println!(
                        "density {spin} {k} {i} {j} {:.14e} {:.14e}",
                        matrix.re[(i, j)],
                        matrix.im[(i, j)]
                    );
                }
            }
        }
    }
}

fn print_response_metadata<T>(report: &dft_rs::response::ResponseReport<T>) {
    println!("Functional: {}", report.functional);
    println!("Shape: {:?}", report.shape);
    println!("Units: {}", report.units);
    println!("Model: {}", report.model);
    println!("Axes: {}", report.axis_convention);
    println!("Converged: {}", report.converged);
    println!("Response residuals: {:?}", report.residuals);
    println!("Response iterations: {:?}", report.iterations);
    println!(
        "Reference eigen residual: {:.3e}",
        report.reference.eigen_residual
    );
    println!(
        "Reference stationarity residual: {:.3e}",
        report.reference.orbital_residual
    );
    println!(
        "Reference metric defect: {:.3e}",
        report.reference.orthonormality_defect
    );
    println!(
        "Reference projector defect: {:.3e}",
        report.reference.projector_defect
    );
    println!(
        "Reference particle-number error: {:.3e}",
        report.reference.particle_number_error
    );
    for warning in &report.warnings {
        println!("Warning: {warning}");
    }
}

fn cli_usage() -> &'static str {
    r#"Usage: dft-rs <command> file.xyz [options]

Commands
  mermin-free-energy electronic A=E-T_e*S_e; --electronic-temperature in K
                     --gibbs also returns equilibrium ideal-gas RRHO G
  gibbs-free-energy  molecular ideal-gas RRHO G=H-T*S from the analytic Hessian
                     --temperature K, --pressure Pa, --symmetry-number N
  free-energy        compatibility alias for mermin-free-energy
  info            parse the geometry and print system information
  integrals       build the AO basis and print S/T/V diagnostics
  energy          single-point SCF energy
  gradient        energy + analytic nuclear gradient
  charges         Mulliken/Loewdin charges and dipole
  optimize        L-BFGS geometry optimization
  saddle          P-RFO or RS-I-RFO transition-state search
  frequencies     harmonic frequencies / IR intensities
  hessian         Cartesian Hessian
  dimensional-scf  self-consistent compensated 1D/2D/3D collinear DFT
                   --field-transfer f1,f2,f3 adds screened scalar/vector response
  periodic-field  analytic dimensional scalar/current/contact Fourier vertices
  magnetic-response  spin orbital magnetic density derivatives (analytic London CPKS)
  nmr             orbital NMR shielding tensors in ppm
  polarizability  static polarizability tensor (analytic CPKS)
  dynamic-polarizability  analytic complex dipole response (paired Casida)
  stability       collinear orbital stability; optional --stab-follow
  stab-following  stability with internal instability following enabled
  dynamic-beta    complete complex first hyperpolarizability (27 entries)
  dynamic-gamma   complete complex second hyperpolarizability (81 entries)
  field-taylor    analytic nonlinear dipole coefficients along a field direction
  field-mixed     analytic mixed dipole derivative (beta, gamma or higher)
  beta            complete analytic static first hyperpolarizability (27 entries)
  gamma           complete analytic static second hyperpolarizability (81 entries)
  orbital-energy-gradients relaxed analytic orbital eigenvalue gradients
  frontier-gap-gradient per-spin neutral-reference LUMO-HOMO gap gradients
  nuclear-taylor analytic nuclear energy coefficients, including cubic/quartic
  nuclear-mixed  analytic mixed nuclear energy derivative (force constants)
  molecular-taylor analytic simultaneous nuclear/electric energy and dipole jets
  molecular-mixed analytic mixed nuclear/electric energy and dipole derivatives
  static-spectroscopy analytic directional APT, Raman or hyper-Raman tensor
  force-constants analytic cubic/quartic pattern, coordinate block or full tensor
  anharmonic     analytic rectilinear anharmonic IR and optional static Raman
  raman           frequencies + IR + Raman activities
  tddft           TDA/RPA excitation energies, oscillator and rotatory strengths
  periodic-electrostatics  dimensional prescribed-charge Coulomb energy/forces
  periodic-basis       dimensional Gaussian Bloch S/T, derivatives and quadrature
  periodic-gaussian-integral normalized Cartesian nuclear or electron-repulsion integral
  periodic-coulomb     contracted spherical nuclear/Hartree operators for supplied density
  periodic-xc          semilocal DFT component energy and spin/Bloch potentials
  periodic        finite-k periodic HF/KS-DFT (needs --cell; --forces adds the
                  analytic Hellmann-Feynman + Pulay forces)
  field-response  screened finite-q orbital electric/magnetic density and current
  periodic-response  screened finite-q density response to a unit plane wave
  dielectric      inverse dielectric matrix from screened density response
  thermal-response uniform static Mermin temperature/number/reservoir response
  sapt            SAPT interaction decomposition; the XYZ is the dimer and
                  --natoms-a N marks monomer A (--charge-a/--charge-b for ions,
                  --mult-a/--mult-b for open-shell monomers)

Periodic boundary conditions (the `periodic` command)
  --cell <spec>         lattice in ANGSTROM, matching the XYZ convention:
                          --cell 5.43                 cubic edge
                          --cell "4.0,4.0,6.5"        orthorhombic
                          --cell "a1x,a1y,a1z; a2x,a2y,a2z; a3x,a3y,a3z"
  --k-mesh <nx,ny,nz>   unshifted Monkhorst-Pack mesh (default 1,1,1 = Gamma)
  --basis-grid         include dimensional atom-centred quadrature (periodic-basis)
  --basis-grid-derivatives include moving grid-weight derivatives
  --basis-grid-cutoff X physical radial/Becke cutoff in Bohr (default 12)
  --pbc <0|1,0|1,0|1> periodic axes (default 1,1,1; open axes need one k point)
  --electrostatic-charges <q,...> prescribed charges for periodic-electrostatics
  --electrostatic-widths <w,...> Gaussian widths in Bohr (default zero: points)
  --charge-compensation <m> neutral | uniform (3D) | gaussian
  --countercharge-center <x,y,z> fixed Gaussian center in Bohr
  --countercharge-width <w> normalized Gaussian width in Bohr
  --ewald-alpha <a>    Ewald splitting parameter in inverse Bohr
  --ewald-real-cutoff <r> real-space Coulomb cutoff in Bohr
  --ewald-reciprocal-cutoff <g> reciprocal Coulomb cutoff in inverse Bohr
  --electrostatic-tol <e> absolute Fourier quadrature tolerance (default 1e-11)
  --electrostatic-quadrature <n> initial Fourier rule order (default 96)
  --gaussian-exponents <a,b,...> primitive exponents in inverse Bohr squared
  --gaussian-angular <lx,ly,lz;...> Cartesian powers for each Gaussian centre
  --gaussian-source-charge <q> nuclear source charge (default 1)
  --gaussian-source-width <w> nuclear Gaussian width in Bohr (default point: 0)
  --gaussian-tol <e> primitive quadrature/translation tolerance (default 1e-11)
  --gaussian-quadrature <n> initial primitive Fourier rule order (default 96)
  --density-matrices <csv> complete k,row,col,real,imag density rows (periodic-coulomb)
  --nuclear-widths <w,...> physical Gaussian nuclear widths in Bohr (default points)
  --alpha-density-matrices <csv> complete alpha density for periodic-xc
  --beta-density-matrices <csv> complete beta density for periodic-xc
  --xc-operator-tol <value> raw XC operator/component/particle tolerance (1e-11)
  --xc-grid-tol <value> grid electron-count tolerance against the metric (1e-5)
  --overlap-cutoff X    real-space cutoff in BOHR for the short-range
                        one-electron lattice sums (default 20). This is a
                        convergence parameter with a limit; it does NOT
                        control the conditionally convergent Coulomb sums,
                        which use Ewald summation.
  --exchange-cutoff X   localized exact-exchange cutoff in Bohr
                        (default: the overlap cutoff)
  --grid-shell <n>      cells of periodic images carried by the XC grid
                        (default 2)
  --temperature X       electronic temperature in Kelvin. 0 (default) means
                        integer occupations; positive enables mesh-global
                        Fermi-Dirac occupations, and the reported energy to
                        differentiate becomes the Mermin free energy.
  --response-q x,y,z    Cartesian transfer in inverse Bohr (default 0,0,0)
  --reciprocal-cutoff X full spherical Hartree reciprocal cutoff, inverse Bohr
                        (default 6; converge separately from AO cutoffs)
  --response-modes i,j  reciprocal mode indices (default 0; one for
                        periodic-response, selected matrix axes for dielectric)
  --fixed-chemical-potential  use grand-canonical uniform static response
  --forces              also compute analytic forces
  --relax               optimize atomic positions with analytic periodic
                        forces while keeping all lattice vectors fixed;
                        `--traj` streams accepted geometries. Cell relaxation
                        needs the still-staged electronic stress tensor.
  --opt-max-steps <n>   fixed-cell periodic optimization step cap (default 200)
  --opt-gtol <x>        periodic max-force convergence threshold in Ha/Bohr
                        (default 3e-4)
  --opt-step-max <x>    largest accepted Cartesian step in Bohr (default 0.3)
  --opt-history <n>     periodic L-BFGS history length (default 8)

Continuum solvation
  --solvent NAME        bundled solvent (179 entries; `--list-solvents`),
                        optionally prefixed `smd:`, `cosmo:`, `ief:` or `ssvpe:`
                        (the IEF family brings in the D matrix and a
                        general K; energies, analytic gradients and analytic
                        Hessians work for all four continuum models)
                        (`smd:` also switches the cavity to SMD's intrinsic
                        Coulomb radii and adds the G_CDS surface-tension term)
  --solvent custom      descriptors given directly instead of by name
  --eps X               dielectric constant (alone this is enough for
                        C-PCM/COSMO, and it overrides a named solvent)
  --refractive-index X  --solvent-alpha X  --solvent-beta X
  --solvent-gamma X     remaining SMD descriptors

Options
  --charge <q>          total charge (default 0)
  --multiplicity <m>    spin multiplicity 2S+1 (default 1)
  --guess <kind>        SCF starting point: sad (superposition of spherically
                        averaged atomic densities, built on the fly in this
                        basis) | gwh | core. Default: sad for molecular
                        commands, core for periodic (preserves its legacy
                        initial state).
  --unrestricted        force UKS/UHF even for a closed shell. On its own
                        this reproduces the restricted energy exactly;
                        combine with --guess-mix to reach a broken-symmetry
                        solution (diradicals, homolytic dissociation).
  --guess-mix <deg>     rotate the beta HOMO/LUMO of the guess by this
                        angle to break spin symmetry (default 0; 45 is a
                        common choice). Implies --unrestricted.
  --basis <spec>        basis set: a bundled name, a .g94 path, or a
                        per-element spec (default sto-3g). The spec is
                        "base; SYM,SYM=source; ..." — e.g.
                          --basis "def2-tzvp; H=def2-svp"
                          --basis "def2-svp; Ni,Fe=lanl2dz"
                          --basis "H=sto-3g; O=./my-oxygen.g94"
                        "@file" reads the spec from a file (# comments).
  --ecp <spec>          core potentials, SAME grammar and independent of
                        --basis, applied after it. Only the elements named
                        are touched; "none" clears. e.g.
                          --ecp "Ni=stuttgart-rsc-1997"   ECP from another set
                          --ecp none                      orbitals, no ECP
  --pseudo-sub <spec>   zero-based pseudo-substituent sites, e.g. "4=Me".
                        Me/Et/tBu/Ph/CF3/OMe are parsed and validated but
                        currently return a typed missing-parameter refusal;
                        no request is silently treated as ordinary hydrogen.
  --xc <name>           combined functional (svwn5/svwn3/spw92/pbe/blyp/b3lyp/
                        b3lyp5/o3lyp/pbe0/cam-b3lyp/lc-blyp/lc-pbe/hf; default hf;
                        `--list-functionals` prints the whole registry)
  --x <name> --c <name> exchange + correlation given separately
  --omega <w>           override the range-separation parameter ω (RSH only)
  --disp <model>        dispersion correction (default none):
                        empirical: d2 | d3zero | d3bj | chg (ωB97X-D tail)
                        density-dependent: xdm (energy correction for the
                        energy/charges commands; requires --xdm-a1/--xdm-a2)
                        nonlocal:  vv10 (self-consistent) | vv10-post (post-SCF)
                        (-V functionals auto-enable SC-VV10 unless overridden)
  --xdm-a1 <value>      dimensionless XDM BJ damping parameter a1
  --xdm-a2 <value>      XDM BJ damping length a2 in Bohr
  --nlc <model>         compatibility alias for the VV10 choices (XDM permits
                        auto/none only):
                        auto | vv10 | vv10-post | none
  --nstates <n>         number of TDA excited states (tddft command; default 5)
  --response-omega <w>  response frequency in Hartree (dynamic-polarizability)
  --input-frequencies <pairs> incoming omega,damping pairs, separated by ';'
                           dynamic-beta needs 2, dynamic-gamma needs 3
                           omitted: repeat response-omega/response-damping
  --electric-process <p> shg,sfg,dfg,optical-rectification,electro-optic,
                           thg,efish,optical-kerr,dc-kerr,four-wave-mixing,
                           field-induced-rectification,two-photon-absorption
  --stab-follow        follow internal instabilities before molecular properties
  --stab-channels <ids> comma-separated stability channel IDs (default automatic)
  --stab-roots <n>     lowest eigenvalues per channel (default 3)
  --stab-tol <value>   absolute eigen-residual tolerance (default 1e-7 Hartree)
  --stab-max-iter <n>  Davidson iteration cap (default 200)
  --stab-max-subspace n projected subspace cap (default 60)
  --stab-block-buffer n retained degeneracy buffer (default 3)
  --stab-resolution r verdict resolution (default 10*stab-tol)
  --stab-no-verify     disable independent lowest-root verification
  --stab-verify-steps n Rayleigh descent iterations per probe (default 24)
  --stab-max-descents n maximum accepted internal descents (default 3)
  --stab-step-angle r  orbital rotation angle in radians (default 0.3)
  --stab-min-energy-drop e required energy decrease in Hartree (default 1e-8)
  --process-frequencies <w> optical magnitudes separated by ',' (Hartree)
                           one-frequency presets also accept response-omega
  --response-restart n GMRES restart length (default 40)
  --delta-temperature K thermal-response temperature direction (default 1 Kelvin)
  --delta-numbers a,b   thermal-response per-spin particle-number direction
  --delta-mu a,b        thermal-response reservoir direction in Hartree
  --response-damping <d> non-negative retarded damping in Hartree (default 0)
  --imaginary-frequency evaluate z=i*omega, with non-negative omega and no damping
  --field-wave <qx,qy,qz>  Cartesian inverse Bohr (periodic-field)
  --field-k-offset <kx,ky,kz>  independent shifted ket mesh in inverse Bohr
  --field-scalar <real,imag>  electron potential energy, Hartree
  --field-vector <re,im;re,im;re,im>  reduced atomic vector potential
  --field-frequency <omega,damping>  Hartree, nonnegative damping
  --magnetic-gauge <g>  giao (default) or common_origin for molecular magnetic commands
  --magnetic-origin <x,y,z>  vector-potential origin in Bohr (default zero)
  --response-tol <t>    original-operator relative residual (default 1e-10)
  --response-max-iter <n> response iteration cap (default 500)
  --field-direction <Ex,Ey,Ez> field-taylor direction in atomic units (default 0,0,1)
  --electric-order <n> static-spectroscopy electric rank: 1=APT, 2=Raman, 3=hyper-Raman
  --orbital-coordinates <i,j,...> selected Cartesian orbital-gradient coordinates
  --degeneracy-tol <e> grouped orbital energy span in Hartree (default 1e-8)
  --force-coordinates <i,j,...> force-constants Cartesian block (IDs 3*atom+axis)
  --anharmonic-raman  include third nuclear polarizability derivatives
  --anharmonic-method <retained-polyad|strict-vpt2> treatment (default: retained-polyad)
  --vibrational-modes <i,j,...> select harmonic modes (default: all)
  --equilibrium-tol <f> maximum force for anharmonic expansion (default: 1e-4)
  --resonance-tol <f> absolute polyad energy tolerance (default: 1e-8 Hartree)
  --resonance-coupling-factor <f> coupling-relative resonance window (default: 1)
  --max-vibrational-basis <n> oscillator storage limit (default: 1024)
  --max-polyad-states <n> retained oscillator states (default: 256)
  --response-order <n> highest field-taylor order (default 3)
  --nuclear-direction <spec> per-atom Bohr displacements: "0,0,0;0,0,1"
  --nuclear-directions <spec> mixed displacement paths, separated by |:
                         "0,0,0;0,0,1|0,0,0;0,0,1|0,0,0;0,0,1"
                         nuclear-taylor defaults to order 4
  --electric-assembly <m> beta assembly: projector (default) or variational (2n+1)
  --field-directions <spec> field-mixed directions, separated by semicolons:
                           "1,0,0;0,1,0" is beta with x and y fields
  --tda-spin <mode>     auto | singlet | triplet | spinresolved (tddft command)
  --tddft-method <m>    tda (default) | rpa — full TDDFT/TDHF via the Casida form
  --grid-level <0..9>   quadrature grid resolution (default 3)
  --vv10-grid-level <0..9> independent molecular VV10 quadrature (default 3)
  --grid-weight-cutoff x nonnegative quadrature weight cutoff (default 1e-14)
  --field <Ex,Ey,Ez>    uniform external electric field in a.u.; adds +E.r to
                        H_core and the nuclear term -Z_A R_A.E. Analytic
                        gradients and Hessians carry both, so derivative,
                        vibration, optimization, saddle and Raman workflows
                        all run analytically in the field.
  --thermo-temperature <K>
                        RRHO temperature for frequencies (default 298.15 K)
  --pressure <Pa>       ideal-gas RRHO pressure (default 101325 Pa)
  --symmetry-number <n> external rotational symmetry number (default 1)
  --saddle-method <m>  prfo (default) | rsirfo (Householder image function
                        with an intrinsic restricted RFO step)
  --traj <file>         stream the optimize/saddle path as a multi-frame XYZ,
                        flushed per accepted step rather than buffered

SCF convergence (PySCF-compatible defaults)
  --conv-tol X          energy threshold (PySCF conv_tol; default 1e-9)
  --conv-tol-grad X     orbital-gradient threshold (default sqrt(conv-tol))
  --conv-tol-dp X       density RMS threshold (default 1e-8)
  --max-iter <n>        iteration cap (default 150)
  --no-adiis            disable ADIIS (CDIIS only)
  --adiis-until-error X use ADIIS above this commutator error (default 0.1;
                        integer occupations only for periodic calculations)
  --no-soscf            disable the second-order endgame
  --diis-size <n>       DIIS subspace size (default 8)
  --level-shift X       virtual level shift in Hartree
  --damping X           density damping while the error is large
  --incore              store the full ERI tensor instead of direct recompute
  --cpu-threads <N>     per-calculation worker ceiling, including nested Rayon/faer
  --memory-mb <N>       shared managed allocation budget in MiB (not process RSS)
  --incore-mb <N>       memory budget in MiB for holding the ERI tensor
                        (default 2048). The Direct→in-core switch is
                        automatic below this; raising it is worth checking
                        when a run seems slow, because the tensor is n^4
                        doubles and the cliff is steep — ferrocene/6-31G
                        (132 AOs) needs 2317 MiB, 13% over the default.
  --ri-j                density-fit the Coulomb build (default aux: def2-universal-jfit)
  --auxbasis <name>     auxiliary basis for --ri-j
  --nbo                 add the natural population analysis, Wiberg indices,
                        natural Lewis structure and second-order
                        donor-acceptor energies to the `charges` output
  --dump-wfn <file>     write a machine-readable wavefunction checkpoint
  --molden <file>       write a Molden file (orbitals for visualization)
  --log-mode <m>        progress logging: console | file | both | none (default both)
  --log-file <file>     log destination (default <input>.dft.log)
  --log-level <level>   error | warn | info | debug (default info)
  --list-functionals    print the functional registry and exit
  --list-solvents       print the bundled solvent table and exit
  --version             print the version and exit
"#
}

#[derive(Debug, Default)]
struct Cli {
    command: String,
    path: String,
    charge: i32,
    multiplicity: usize,
    basis: String,
    ecp: Option<String>,
    pseudo_sub: Option<String>,
    trajectory: Option<String>,
    dump_wfn: Option<String>,
    molden: Option<String>,
    xc: Option<String>,
    x_func: Option<String>,
    c_func: Option<String>,
    omega: Option<f64>,
    disp: String,
    xdm_a1: Option<f64>,
    xdm_a2: Option<f64>,
    nlc: String,
    log_mode: String,
    log_file: Option<String>,
    grid_level: usize,
    vv10_grid_level: usize,
    grid_weight_cutoff: Option<f64>,
    incore: bool,
    incore_mb: Option<usize>,
    density_fit: Option<String>,
    nstates: usize,
    tda_spin: String,
    tddft_method: String,
    stab_follow: bool,
    stabilize_options: dft_rs::response::StabilizeOptions,
    response_omega: f64,
    input_frequencies: Option<Vec<[f64; 2]>>,
    electric_process: Option<dft_rs::response::ElectricProcess>,
    process_frequencies: Option<Vec<f64>>,
    response_damping: f64,
    imaginary_frequency: bool,
    field_options: dft_rs::periodic::field::PeriodicFieldOptions,
    magnetic_gauge: dft_rs::properties::magnetic::Gauge,
    magnetic_origin: [f64; 3],
    response_tol: f64,
    response_max_iter: usize,
    response_restart: usize,
    response_q: [f64; 3],
    field_transfer: Option<[f64; 3]>,
    response_reciprocal_cutoff: f64,
    response_modes: Vec<usize>,
    response_fixed_number: bool,
    delta_temperature: f64,
    delta_numbers: [f64; 2],
    delta_mu: [f64; 2],
    response_order: usize,
    electric_order: usize,
    force_coordinates: Option<Vec<usize>>,
    orbital_options: dft_rs::response::OrbitalGradientOptions,
    anharmonic_config: dft_rs::response::anharmonic::AnharmonicConfig,
    anharmonic_raman: bool,
    vibrational_modes: Option<Vec<usize>>,
    field_direction: [f64; 3],
    mixed_field_directions: Option<Vec<[f64; 3]>>,
    nuclear_direction: Option<Vec<[f64; 3]>>,
    mixed_nuclear_directions: Option<Vec<Vec<[f64; 3]>>>,
    saddle_method: String,
    natoms_a: usize,
    charge_a: i32,
    charge_b: i32,
    mult_a: usize,
    mult_b: usize,
    unrestricted: bool,
    guess_mix: Option<f64>,
    guess: Option<String>,
    solvent: Option<String>,
    eps: Option<f64>,
    refractive_index: Option<f64>,
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
    list_solvents: bool,
    list_functionals: bool,
    conv_tol: Option<f64>,
    conv_tol_grad: Option<f64>,
    conv_tol_dp: Option<f64>,
    max_iter: Option<usize>,
    no_adiis: bool,
    adiis_until_error: Option<f64>,
    no_soscf: bool,
    diis_size: Option<usize>,
    level_shift: Option<f64>,
    damping: Option<f64>,
    field: Option<[f64; 3]>,
    nbo: bool,
    /// Lattice vectors in Bohr, already converted from the Angstrom spec.
    cell: Option<[[f64; 3]; 3]>,
    periodic_axes: [bool; 3],
    basis_grid: bool,
    basis_grid_derivatives: bool,
    basis_grid_cutoff: f64,
    electrostatic_charges: Option<Vec<f64>>,
    electrostatic_widths: Option<Vec<f64>>,
    electrostatic_options: dft_rs::periodic::electrostatics::ElectrostaticOptions,
    gaussian_exponents: Option<Vec<f64>>,
    nuclear_widths: Option<Vec<f64>>,
    density_matrices: Option<String>,
    spin_density_matrices: [Option<String>; 2],
    xc_operator_tolerance: f64,
    xc_grid_tolerance: f64,
    gaussian_angular: Option<Vec<[usize; 3]>>,
    gaussian_source_charge: Option<f64>,
    gaussian_source_width: Option<f64>,
    gaussian_options: dft_rs::periodic::gaussian_coulomb::GaussianIntegralOptions,
    k_mesh: [usize; 3],
    overlap_cutoff: f64,
    exchange_cutoff: Option<f64>,
    grid_shell: i64,
    temperature: f64,
    thermo_temperature: f64,
    pressure: f64,
    symmetry_number: usize,
    forces: bool,
    relax: bool,
    opt_max_steps: usize,
    opt_gtol: f64,
    opt_step_max: f64,
    opt_history: usize,
}

/// Complete uniformly weighted density matrices, with no implicit missing entries.
fn read_coulomb_density(
    path: &str,
    nk: usize,
    nao: usize,
) -> Result<Vec<dft_rs::periodic::BlochMatrix>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read density CSV: {e}"))?;
    let mut lines = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'));
    if lines.next().map(str::trim) != Some("k,row,col,real,imag") {
        return Err("density CSV requires header k,row,col,real,imag".into());
    }
    let expected = nk
        .checked_mul(nao)
        .and_then(|n| n.checked_mul(nao))
        .ok_or("density CSV shape overflow")?;
    let records: Vec<_> = lines.collect();
    if records.len() != expected {
        return Err(format!(
            "density CSV requires exactly {expected} complete matrix entries"
        ));
    }
    let mut density: Vec<_> = (0..nk)
        .map(|_| dft_rs::periodic::BlochMatrix::zeros(nao, nao))
        .collect();
    let mut seen = vec![false; expected];
    for record in records {
        let parts: Vec<_> = record.split(',').map(str::trim).collect();
        if parts.len() != 5 {
            return Err("density CSV row requires five entries".into());
        }
        let k = parts[0]
            .parse::<usize>()
            .map_err(|_| "invalid density k index")?;
        let i = parts[1]
            .parse::<usize>()
            .map_err(|_| "invalid density row index")?;
        let j = parts[2]
            .parse::<usize>()
            .map_err(|_| "invalid density column index")?;
        let re = parts[3]
            .parse::<f64>()
            .map_err(|_| "invalid density real value")?;
        let im = parts[4]
            .parse::<f64>()
            .map_err(|_| "invalid density imaginary value")?;
        if k >= nk || i >= nao || j >= nao || !re.is_finite() || !im.is_finite() {
            return Err("density CSV indices and values must fit the finite matrix shape".into());
        }
        let index = (k * nao + i) * nao + j;
        if seen[index] {
            return Err("density CSV contains a duplicate matrix entry".into());
        }
        seen[index] = true;
        density[k].re[(i, j)] = re;
        density[k].im[(i, j)] = im;
    }
    Ok(density)
}

/// Single source of truth for which `--option` a given command refuses.
/// Anything not listed here is accepted (or simply unused) by that command;
/// everything listed here is rejected by [`validate_option_table`] before
/// any computation starts, with a message naming the option and what is
/// missing -- never a silent no-op that still prints a plausible number.
///
/// `sapt` reuses the ordinary `ScfOptions`, so it honors `--incore`,
/// `--ri-j`, `--level-shift`, `--damping`, `--diis-size`, `--no-adiis`,
/// `--adiis-until-error`, `--no-soscf`, `--unrestricted`, `--guess` and
/// `--guess-mix` exactly like
/// the energy-family commands; only the dispersion/solvent/field/analysis
/// options below are rejected for it. `periodic` has its own
/// `PeriodicScfOptions` (src/periodic/scf.rs) that exposes ADIIS, damping,
/// DIIS sizing and level shifting but not the remaining molecular-only knobs.
pub const OPTION_REJECTIONS: &[(&str, &str, &str)] = &[
    (
        "periodic",
        "--pseudo-sub",
        "--pseudo-sub is not available for periodic: no periodic pseudo-site Hamiltonian exists",
    ),
    (
        "sapt",
        "--pseudo-sub",
        "--pseudo-sub is not available for sapt: no monomer pseudo-site Hamiltonian exists",
    ),
    // Dispersion / nonlocal correlation.
    (
        "periodic",
        "--disp",
        "--disp is not available for periodic: the lattice-summed empirical/nonlocal \
         dispersion kernel is not implemented for periodic systems \
         (PeriodicCapabilities::vv10/chg are false; see src/periodic/features.rs:90-111)",
    ),
    (
        "periodic",
        "--nlc",
        "--nlc is not available for periodic: the lattice-summed nonlocal-correlation \
         (VV10) kernel is not implemented for periodic systems \
         (PeriodicCapabilities::vv10 is false; see src/periodic/features.rs:90-111)",
    ),
    (
        "sapt",
        "--disp",
        "--disp is not available for sapt: a monomer empirical/nonlocal dispersion \
         correction would double-count the SAPT E_disp(2) term",
    ),
    (
        "sapt",
        "--nlc",
        "--nlc is not available for sapt: a monomer nonlocal-correlation (VV10) \
         correction would double-count the SAPT E_disp(2) term",
    ),
    (
        "periodic",
        "--xdm-a1",
        "--xdm-a1 is not available for periodic: XDM has no lattice-summed kernel",
    ),
    (
        "periodic",
        "--xdm-a2",
        "--xdm-a2 is not available for periodic: XDM has no lattice-summed kernel",
    ),
    (
        "sapt",
        "--xdm-a1",
        "--xdm-a1 is not available for sapt: XDM would double-count the SAPT E_disp(2) term",
    ),
    (
        "sapt",
        "--xdm-a2",
        "--xdm-a2 is not available for sapt: XDM would double-count the SAPT E_disp(2) term",
    ),
    // Continuum solvation (the named-solvent flag and every raw descriptor
    // that can drive the PCM without a name).
    (
        "periodic",
        "--solvent",
        "--solvent is not available for periodic: there is no periodic PCM",
    ),
    (
        "periodic",
        "--eps",
        "--eps is not available for periodic: there is no periodic PCM",
    ),
    (
        "periodic",
        "--refractive-index",
        "--refractive-index is not available for periodic: there is no periodic PCM",
    ),
    (
        "periodic",
        "--solvent-alpha",
        "--solvent-alpha is not available for periodic: there is no periodic PCM",
    ),
    (
        "periodic",
        "--solvent-beta",
        "--solvent-beta is not available for periodic: there is no periodic PCM",
    ),
    (
        "periodic",
        "--solvent-gamma",
        "--solvent-gamma is not available for periodic: there is no periodic PCM",
    ),
    (
        "sapt",
        "--solvent",
        "--solvent is not available for sapt: a solvated SAPT decomposition is not \
         defined here",
    ),
    (
        "sapt",
        "--eps",
        "--eps is not available for sapt: a solvated SAPT decomposition is not \
         defined here",
    ),
    (
        "sapt",
        "--refractive-index",
        "--refractive-index is not available for sapt: a solvated SAPT decomposition \
         is not defined here",
    ),
    (
        "sapt",
        "--solvent-alpha",
        "--solvent-alpha is not available for sapt: a solvated SAPT decomposition is \
         not defined here",
    ),
    (
        "sapt",
        "--solvent-beta",
        "--solvent-beta is not available for sapt: a solvated SAPT decomposition is \
         not defined here",
    ),
    (
        "sapt",
        "--solvent-gamma",
        "--solvent-gamma is not available for sapt: a solvated SAPT decomposition is \
         not defined here",
    ),
    // External electric field.
    (
        "periodic",
        "--field",
        "--field is not available for periodic: the periodic Hamiltonian has no \
         uniform external-field term",
    ),
    (
        "sapt",
        "--field",
        "--field is not available for sapt: the SAPT monomer Hamiltonians have no \
         uniform external-field term",
    ),
    // Post-SCF analysis / export bindings.
    (
        "periodic",
        "--nbo",
        "--nbo is not available for periodic: the NAO/NPA analysis binding is not \
         wired up for this command yet",
    ),
    (
        "sapt",
        "--nbo",
        "--nbo is not available for sapt: the NAO/NPA analysis binding is not wired \
         up for this command yet",
    ),
    (
        "periodic",
        "--dump-wfn",
        "--dump-wfn is not available for periodic: the wavefunction-checkpoint \
         binding is not wired up for this command yet",
    ),
    (
        "sapt",
        "--dump-wfn",
        "--dump-wfn is not available for sapt: the wavefunction-checkpoint binding \
         is not wired up for this command yet",
    ),
    (
        "periodic",
        "--molden",
        "--molden is not available for periodic: the Molden-export binding is not \
         wired up for this command yet",
    ),
    (
        "sapt",
        "--molden",
        "--molden is not available for sapt: the Molden-export binding is not wired \
         up for this command yet",
    ),
    // Molecular-SCF knobs that PeriodicScfOptions does not expose: the
    // periodic SCF builds its own ERI/DIIS pipeline (src/periodic/scf.rs)
    // and reads none of these from the CLI today.
    (
        "periodic",
        "--incore",
        "--incore is not available for periodic: the periodic SCF has no in-core \
         ERI mode, only its own direct/lattice build",
    ),
    (
        "periodic",
        "--incore-mb",
        "--incore-mb is not available for periodic: the periodic SCF has no in-core \
         ERI mode, only its own direct/lattice build",
    ),
    (
        "periodic",
        "--ri-j",
        "--ri-j is not available for periodic: density fitting is not wired into \
         the periodic Coulomb/exchange build",
    ),
    (
        "periodic",
        "--auxbasis",
        "--auxbasis is not available for periodic: density fitting is not wired \
         into the periodic Coulomb/exchange build",
    ),
    (
        "periodic",
        "--no-soscf",
        "--no-soscf is not available for periodic: the periodic SCF has no \
         second-order SOSCF endgame to disable",
    ),
    (
        "periodic",
        "--unrestricted",
        "--unrestricted is not available for periodic: the periodic SCF reference \
         is chosen automatically from the alpha/beta electron counts \
         (PeriodicScfOptions::reference stays Auto) and cannot be forced from the \
         CLI yet",
    ),
    (
        "periodic",
        "--guess-mix",
        "--guess-mix is not available for periodic: the periodic SCF has no \
         guess-mixing / broken-symmetry entry point",
    ),
];

/// Reject any `(command, option)` pair from [`OPTION_REJECTIONS`] that was
/// actually given on the command line. Anything not in the table passes
/// through unchanged -- this only narrows what is accepted, it never widens
/// it, so commands outside the table keep their existing behaviour.
pub fn validate_option_table(
    command: &str,
    flags_given: &std::collections::BTreeSet<String>,
) -> Result<(), String> {
    for &(rejected_command, flag, reason) in OPTION_REJECTIONS {
        if command == rejected_command && flags_given.contains(flag) {
            return Err(reason.to_owned());
        }
    }
    Ok(())
}

/// Resolve the density-dependent XDM CLI mode without inventing empirical
/// damping parameters. XDM is currently an energy-only correction: its
/// analytic nuclear gradient needs both the implicit Becke-Roussel root
/// response and the Becke/Hirshfeld grid-weight response, neither of which is
/// implemented yet. Refuse derivative/property commands before starting an
/// SCF instead of accepting `--disp xdm` and silently omitting it.
pub fn resolve_xdm_cli(
    command: &str,
    dispersion: &str,
    a1: Option<f64>,
    a2: Option<f64>,
    nlc: &str,
) -> Result<Option<dft_rs::dispersion::xdm::XdmDampingParams>, String> {
    let is_xdm = dispersion.eq_ignore_ascii_case("xdm");
    if !is_xdm {
        if a1.is_some() || a2.is_some() {
            return Err("--xdm-a1/--xdm-a2 require `--disp xdm`".to_owned());
        }
        return Ok(None);
    }

    if !matches!(command, "energy" | "charges") {
        return Err(format!(
            "--disp xdm is not available for {command}: XDM currently has no analytic nuclear \
             gradient; use the `energy` or `charges` command"
        ));
    }
    if !matches!(nlc.to_ascii_lowercase().as_str(), "auto" | "none") {
        return Err(
            "--disp xdm cannot be combined with --nlc: that would double-count dispersion"
                .to_owned(),
        );
    }
    let a1 = a1.ok_or_else(|| {
        "--disp xdm requires --xdm-a1 and --xdm-a2; dft-rs does not invent or bundle \
         per-functional XDM damping fits"
            .to_owned()
    })?;
    let a2 = a2.ok_or_else(|| {
        "--disp xdm requires --xdm-a1 and --xdm-a2; dft-rs does not invent or bundle \
         per-functional XDM damping fits"
            .to_owned()
    })?;
    if !(a1.is_finite() && a1 >= 0.0) {
        return Err("--xdm-a1 must be finite and non-negative".to_owned());
    }
    if !(a2.is_finite() && a2 > 0.0) {
        return Err("--xdm-a2 must be finite and positive (in Bohr)".to_owned());
    }
    Ok(Some(dft_rs::dispersion::xdm::XdmDampingParams { a1, a2 }))
}

/// Parse `--cell`: one number (cubic), three (orthorhombic), or nine
/// (triclinic, three vectors separated by `;`). Angstrom in, Bohr out — the
/// same convention as the XYZ file, so a user never mixes unit systems
/// within one command line.
fn parse_cell(spec: &str) -> Result<[[f64; 3]; 3], String> {
    let values: Vec<f64> = spec
        .split([';', ',', ' '])
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<f64>()
                .map_err(|_| format!("--cell: `{}` is not a number", s.trim()))
        })
        .collect::<Result<_, _>>()?;
    let a = dft_rs::constants::ANGSTROM_TO_BOHR;
    let rows = match values.len() {
        1 => [
            [values[0], 0.0, 0.0],
            [0.0, values[0], 0.0],
            [0.0, 0.0, values[0]],
        ],
        3 => [
            [values[0], 0.0, 0.0],
            [0.0, values[1], 0.0],
            [0.0, 0.0, values[2]],
        ],
        9 => [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
        ],
        n => {
            return Err(format!(
                "--cell takes 1 (cubic), 3 (orthorhombic) or 9 (triclinic) numbers, got {n}"
            ))
        }
    };
    Ok(rows.map(|row| row.map(|v| v * a)))
}

fn parse_args_from(args: Vec<String>) -> Result<Cli, String> {
    let mut args = args.into_iter();
    let command = args.next().ok_or_else(|| cli_usage().to_owned())?;
    if command == "--help" || command == "-h" {
        return Err(String::new());
    }
    if command == "--list-solvents" {
        return Ok(Cli {
            command: "list-solvents".to_owned(),
            list_solvents: true,
            ..Cli::default()
        });
    }
    if command == "--list-functionals" {
        return Ok(Cli {
            command: "list-functionals".to_owned(),
            list_functionals: true,
            ..Cli::default()
        });
    }
    match command.as_str() {
        "info"
        | "integrals"
        | "energy"
        | "gradient"
        | "charges"
        | "optimize"
        | "saddle"
        | "frequencies"
        | "hessian"
        | "polarizability"
        | "magnetic-response"
        | "nmr"
        | "dynamic-polarizability"
        | "dynamic-beta"
        | "dynamic-gamma"
        | "stability"
        | "stab-following"
        | "field-taylor"
        | "field-mixed"
        | "beta"
        | "gamma"
        | "nuclear-taylor"
        | "nuclear-mixed"
        | "molecular-taylor"
        | "molecular-mixed"
        | "static-spectroscopy"
        | "force-constants"
        | "anharmonic"
        | "orbital-energy-gradients"
        | "frontier-gap-gradient"
        | "raman"
        | "tddft"
        | "sapt"
        | "periodic"
        | "periodic-electrostatics"
        | "periodic-basis"
        | "periodic-gaussian-integral"
        | "periodic-coulomb"
        | "periodic-xc"
        | "periodic-field"
        | "dimensional-scf"
        | "field-response"
        | "periodic-response"
        | "dielectric"
        | "thermal-response" => {}
        _ => {
            return Err(format!(
                "{}\n\n{}",
                dft_rs::DftError::Unsupported {
                    feature: format!("command `{command}`"),
                    missing:
                        "a registered CLI implementation; use --help to list available commands"
                            .into(),
                },
                cli_usage()
            ))
        }
    }
    let path = args
        .next()
        .ok_or_else(|| format!("missing XYZ file\n\n{}", cli_usage()))?;
    let mut charge = 0;
    let mut multiplicity = 1;
    let mut basis = "sto-3g".to_owned();
    let mut ecp: Option<String> = None;
    let mut pseudo_sub: Option<String> = None;
    let mut trajectory: Option<String> = None;
    let mut dump_wfn: Option<String> = None;
    let mut molden: Option<String> = None;
    let mut xc: Option<String> = None;
    let mut x_func: Option<String> = None;
    let mut c_func: Option<String> = None;
    let mut omega: Option<f64> = None;
    let mut disp = "auto".to_owned();
    let mut xdm_a1: Option<f64> = None;
    let mut xdm_a2: Option<f64> = None;
    let mut nlc = "auto".to_owned();
    let mut log_mode = "both".to_owned();
    let mut log_file: Option<String> = None;
    let mut grid_level = 3usize;
    let mut vv10_grid_level = 3usize;
    let mut grid_weight_cutoff = None;
    let mut nstates = 5usize;
    let mut tda_spin = "auto".to_owned();
    let mut tddft_method = "tda".to_owned();
    let mut saddle_method = "prfo".to_owned();
    let mut incore = false;
    let mut incore_mb: Option<usize> = None;
    let mut ri_j = false;
    let mut auxbasis: Option<String> = None;
    let mut natoms_a = 0usize;
    let mut charge_a = 0i32;
    let mut charge_b = 0i32;
    let mut mult_a = 1usize;
    let mut mult_b = 1usize;
    let mut unrestricted = false;
    let mut guess_mix: Option<f64> = None;
    let mut guess: Option<String> = None;
    let mut solvent: Option<String> = None;
    let mut eps: Option<f64> = None;
    let mut refractive_index: Option<f64> = None;
    let mut alpha: Option<f64> = None;
    let mut beta: Option<f64> = None;
    let mut gamma: Option<f64> = None;
    let mut list_solvents = false;
    let mut list_functionals = false;
    let mut conv_tol: Option<f64> = None;
    let mut conv_tol_grad: Option<f64> = None;
    let mut conv_tol_dp: Option<f64> = None;
    let mut max_iter_opt: Option<usize> = None;
    let mut no_adiis = false;
    let mut adiis_until_error: Option<f64> = None;
    let mut no_soscf = false;
    let mut diis_size: Option<usize> = None;
    let mut level_shift: Option<f64> = None;
    let mut damping: Option<f64> = None;
    let mut field: Option<[f64; 3]> = None;
    let mut cell: Option<[[f64; 3]; 3]> = None;
    let mut periodic_axes = [true; 3];
    let mut basis_grid = false;
    let mut basis_grid_derivatives = false;
    let mut basis_grid_cutoff: f64 = 12.0;
    let mut electrostatic_charges = None;
    let mut electrostatic_widths = None;
    let mut electrostatic_options =
        dft_rs::periodic::electrostatics::ElectrostaticOptions::default();
    let mut compensation_id = "neutral".to_owned();
    let mut gaussian_exponents = None;
    let mut nuclear_widths = None;
    let mut density_matrices = None;
    let mut spin_density_matrices = [None, None];
    let mut xc_operator_tolerance = 1e-11;
    let mut xc_grid_tolerance = 1e-5;
    let mut gaussian_angular = None;
    let mut gaussian_source_charge = None;
    let mut gaussian_source_width = None;
    let mut gaussian_options =
        dft_rs::periodic::gaussian_coulomb::GaussianIntegralOptions::default();
    let mut countercharge_center = None;
    let mut countercharge_width = None;
    let mut k_mesh = [1usize; 3];
    let mut overlap_cutoff = 20.0f64;
    if matches!(
        command.as_str(),
        "periodic-basis"
            | "periodic-coulomb"
            | "periodic-xc"
            | "periodic-field"
            | "dimensional-scf"
    ) {
        overlap_cutoff = 12.0;
    }
    let mut exchange_cutoff: Option<f64> = None;
    let mut grid_shell = 2i64;
    let mut temperature = 0.0f64;
    let mut thermo_temperature = 298.15f64;
    let mut pressure = 101_325.0f64;
    let mut symmetry_number = 1usize;
    let mut forces = false;
    let mut relax = false;
    let mut opt_max_steps = 200usize;
    let mut opt_gtol = 3e-4f64;
    let mut opt_step_max = 0.3f64;
    let mut opt_history = 8usize;
    let mut nbo = false;
    let mut flags_given: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut stab_follow = command == "stab-following";
    let mut stabilize_options = dft_rs::response::StabilizeOptions::default();
    let mut response_omega = 0.0;
    let mut input_frequencies = None;
    let mut electric_process = None;
    let mut process_frequencies = None;
    let mut response_damping = 0.0;
    let mut imaginary_frequency = false;
    let mut field_options = dft_rs::periodic::field::PeriodicFieldOptions::default();
    let mut magnetic_gauge = dft_rs::properties::magnetic::Gauge::Giao;
    let mut magnetic_origin = [0.0; 3];
    let mut response_tol = 1e-10;
    let mut response_max_iter = 500;
    let mut response_restart = 40;
    let mut response_q = [0.0; 3];
    let mut field_transfer = None;
    let mut response_reciprocal_cutoff: f64 = 6.0;
    let mut response_modes = vec![0];
    let mut response_fixed_number = true;
    let mut delta_temperature: f64 = 1.0;
    let mut delta_numbers = [0.0; 2];
    let mut delta_mu = [0.0; 2];
    let mut response_order = 3;
    let mut electric_assembly = "projector".to_owned();
    let mut electric_order = 2;
    let mut force_coordinates = None;
    let mut orbital_options = dft_rs::response::OrbitalGradientOptions::default();
    let mut anharmonic_config = dft_rs::response::anharmonic::AnharmonicConfig::default();
    let mut anharmonic_raman = false;
    let mut vibrational_modes = None;
    let mut field_direction = [0.0, 0.0, 1.0];
    let mut mixed_field_directions = None;
    let mut nuclear_direction = None;
    let mut mixed_nuclear_directions = None;
    while let Some(flag) = args.next() {
        // Record the flag under its canonical long form (independent of the
        // value it takes) so validate_option_table() can see exactly what
        // the user typed, never confusing a default with an explicit ask.
        flags_given.insert(if flag == "-u" {
            "--unrestricted".to_owned()
        } else {
            flag.clone()
        });
        if matches!(flag.as_str(), "--stab-follow" | "--stab-following") {
            stab_follow = true;
            continue;
        }
        if flag == "--stab-no-verify" {
            stabilize_options.analysis.verify_lowest = false;
            continue;
        }
        if flag == "--forces" {
            forces = true;
            continue;
        }
        if flag == "--basis-grid" {
            basis_grid = true;
            continue;
        }
        if flag == "--basis-grid-derivatives" {
            basis_grid_derivatives = true;
            continue;
        }
        if flag == "--anharmonic-raman" {
            anharmonic_raman = true;
            continue;
        }
        if flag == "--imaginary-frequency" {
            imaginary_frequency = true;
            continue;
        }
        if flag == "--fixed-chemical-potential" {
            response_fixed_number = false;
            continue;
        }
        if flag == "--relax" {
            relax = true;
            continue;
        }
        if flag == "--nbo" {
            nbo = true;
            continue;
        }
        if flag == "--incore" {
            incore = true;
            continue;
        }
        if flag == "--ri-j" {
            ri_j = true;
            continue;
        }
        if flag == "--list-solvents" {
            list_solvents = true;
            continue;
        }
        if flag == "--list-functionals" {
            list_functionals = true;
            continue;
        }
        if flag == "--no-adiis" {
            no_adiis = true;
            continue;
        }
        if flag == "--no-soscf" {
            no_soscf = true;
            continue;
        }
        if flag == "--unrestricted" || flag == "-u" {
            unrestricted = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--charge" => {
                charge = value
                    .parse()
                    .map_err(|_| format!("invalid charge: {value}"))?
            }
            "--multiplicity" | "--mult" => {
                multiplicity = value
                    .parse()
                    .map_err(|_| format!("invalid multiplicity: {value}"))?;
                if multiplicity == 0 {
                    return Err("multiplicity must be >= 1".to_owned());
                }
            }
            "--conv-tol" => {
                conv_tol = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --conv-tol: {value}"))?,
                )
            }
            "--conv-tol-grad" => {
                conv_tol_grad = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --conv-tol-grad: {value}"))?,
                )
            }
            "--conv-tol-dp" => {
                conv_tol_dp = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --conv-tol-dp: {value}"))?,
                )
            }
            "--max-iter" => {
                max_iter_opt = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --max-iter: {value}"))?,
                )
            }
            "--opt-max-steps" => {
                opt_max_steps = value
                    .parse()
                    .map_err(|_| format!("invalid --opt-max-steps: {value}"))?
            }
            "--opt-gtol" => {
                opt_gtol = value
                    .parse()
                    .map_err(|_| format!("invalid --opt-gtol: {value}"))?
            }
            "--opt-step-max" => {
                opt_step_max = value
                    .parse()
                    .map_err(|_| format!("invalid --opt-step-max: {value}"))?
            }
            "--opt-history" => {
                opt_history = value
                    .parse()
                    .map_err(|_| format!("invalid --opt-history: {value}"))?
            }
            "--diis-size" => {
                diis_size = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --diis-size: {value}"))?,
                )
            }
            "--adiis-until-error" => {
                let parsed = value
                    .parse::<f64>()
                    .map_err(|_| format!("invalid --adiis-until-error: {value}"))?;
                if !(parsed.is_finite() && parsed >= 0.0) {
                    return Err("--adiis-until-error must be finite and non-negative".to_owned());
                }
                adiis_until_error = Some(parsed);
            }
            "--level-shift" => {
                level_shift = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --level-shift: {value}"))?,
                )
            }
            "--damping" => {
                damping = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --damping: {value}"))?,
                )
            }
            "--basis" => basis = value,
            "--ecp" => ecp = Some(value),
            "--pseudo-sub" => pseudo_sub = Some(value),
            "--traj" | "--trajectory" => trajectory = Some(value),
            "--auxbasis" => auxbasis = Some(value),
            "--disp" => disp = value,
            "--xdm-a1" => {
                xdm_a1 = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --xdm-a1: {value}"))?,
                )
            }
            "--xdm-a2" => {
                xdm_a2 = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --xdm-a2: {value}"))?,
                )
            }
            "--nlc" => nlc = value,
            "--log-mode" => log_mode = value,
            "--log-file" => log_file = Some(value),
            "--nstates" => {
                nstates = value
                    .parse()
                    .map_err(|_| format!("invalid --nstates: {value}"))?
            }
            "--tda-spin" => tda_spin = value,
            "--tddft-method" => tddft_method = value,
            "--saddle-method" => saddle_method = value.to_ascii_lowercase(),
            "--natoms-a" => {
                natoms_a = value
                    .parse()
                    .map_err(|_| format!("invalid --natoms-a: {value}"))?
            }
            "--charge-a" => {
                charge_a = value
                    .parse()
                    .map_err(|_| format!("invalid --charge-a: {value}"))?
            }
            "--charge-b" => {
                charge_b = value
                    .parse()
                    .map_err(|_| format!("invalid --charge-b: {value}"))?
            }
            "--mult-a" => {
                mult_a = value
                    .parse()
                    .map_err(|_| format!("invalid --mult-a: {value}"))?
            }
            "--mult-b" => {
                mult_b = value
                    .parse()
                    .map_err(|_| format!("invalid --mult-b: {value}"))?
            }
            "--incore-mb" => {
                incore_mb = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --incore-mb: {value}"))?,
                )
            }
            "--guess" => guess = Some(value),
            "--guess-mix" => {
                guess_mix = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --guess-mix: {value}"))?,
                )
            }
            "--solvent" => solvent = Some(value),
            "--eps" => {
                eps = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --eps: {value}"))?,
                )
            }
            "--refractive-index" => {
                refractive_index = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --refractive-index: {value}"))?,
                )
            }
            "--solvent-alpha" => {
                alpha = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --solvent-alpha: {value}"))?,
                )
            }
            "--solvent-beta" => {
                beta = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --solvent-beta: {value}"))?,
                )
            }
            "--solvent-gamma" => {
                gamma = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --solvent-gamma: {value}"))?,
                )
            }
            "--dump-wfn" => dump_wfn = Some(value),
            "--molden" => molden = Some(value),
            "--xc" => xc = Some(value),
            "--x" => x_func = Some(value),
            "--c" => c_func = Some(value),
            "--omega" => {
                omega = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid omega: {value}"))?,
                )
            }
            "--stab-roots" => {
                stabilize_options.analysis.n_roots =
                    value.parse().map_err(|_| "invalid --stab-roots")?
            }
            "--stab-tol" => {
                stabilize_options.analysis.tolerance =
                    value.parse().map_err(|_| "invalid --stab-tol")?
            }
            "--stab-max-iter" => {
                stabilize_options.analysis.max_iterations =
                    value.parse().map_err(|_| "invalid --stab-max-iter")?
            }
            "--stab-max-subspace" => {
                stabilize_options.analysis.max_subspace =
                    value.parse().map_err(|_| "invalid --stab-max-subspace")?
            }
            "--stab-block-buffer" => {
                stabilize_options.analysis.block_buffer =
                    value.parse().map_err(|_| "invalid --stab-block-buffer")?
            }
            "--stab-resolution" => {
                stabilize_options.analysis.resolution =
                    Some(value.parse().map_err(|_| "invalid --stab-resolution")?)
            }
            "--stab-verify-steps" => {
                stabilize_options.analysis.verify_steps =
                    value.parse().map_err(|_| "invalid --stab-verify-steps")?
            }
            "--stab-channels" => {
                stabilize_options.analysis.channels = value
                    .split(',')
                    .map(|c| {
                        dft_rs::response::StabilityChannel::from_id(c.trim())
                            .map_err(|e| e.to_string())
                    })
                    .collect::<Result<_, _>>()?
            }
            "--stab-max-descents" => {
                stabilize_options.max_descents =
                    value.parse().map_err(|_| "invalid --stab-max-descents")?
            }
            "--stab-step-angle" => {
                stabilize_options.step_angle =
                    value.parse().map_err(|_| "invalid --stab-step-angle")?
            }
            "--stab-min-energy-drop" => {
                stabilize_options.min_energy_drop = value
                    .parse()
                    .map_err(|_| "invalid --stab-min-energy-drop")?
            }
            "--response-omega" => {
                response_omega = value
                    .parse()
                    .map_err(|_| "invalid --response-omega".to_owned())?
            }
            "--input-frequencies" => {
                let pairs = value
                    .split(';')
                    .map(|pair| {
                        let values = pair
                            .split(',')
                            .map(|v| {
                                v.trim()
                                    .parse::<f64>()
                                    .map_err(|_| "invalid --input-frequencies".to_owned())
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        if values.len() != 2 {
                            return Err("each input frequency requires omega,damping".to_owned());
                        }
                        Ok([values[0], values[1]])
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                input_frequencies = Some(pairs);
            }
            "--electric-process" => {
                electric_process = Some(
                    value
                        .parse::<dft_rs::response::ElectricProcess>()
                        .map_err(|e| e.to_string())?,
                );
            }
            "--process-frequencies" => {
                process_frequencies = Some(
                    value
                        .split(',')
                        .map(|w| {
                            w.trim()
                                .parse::<f64>()
                                .map_err(|_| "invalid --process-frequencies".to_owned())
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "--field-transfer" => {
                field_transfer = Some(parse_vector(&value, "--field-transfer")?);
            }
            "--response-q" => {
                response_q = parse_vector(&value, "--response-q")?;
            }
            "--delta-temperature" => {
                delta_temperature = value
                    .parse()
                    .map_err(|_| "invalid --delta-temperature".to_owned())?;
            }
            "--delta-numbers" | "--delta-mu" => {
                let entries = value
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| format!("invalid {flag}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if entries.len() != 2 || entries.iter().any(|x| !x.is_finite()) {
                    return Err(format!("{flag} requires two finite comma-separated values"));
                }
                if flag == "--delta-numbers" {
                    delta_numbers = [entries[0], entries[1]];
                } else {
                    delta_mu = [entries[0], entries[1]];
                }
            }
            "--reciprocal-cutoff" => {
                response_reciprocal_cutoff = value
                    .parse()
                    .map_err(|_| "invalid --reciprocal-cutoff".to_owned())?;
            }
            "--response-modes" => {
                response_modes = value
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| "invalid --response-modes".to_owned())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--response-restart" => {
                response_restart = value
                    .parse()
                    .map_err(|_| "invalid --response-restart".to_owned())?;
            }
            "--response-damping" => {
                response_damping = value
                    .parse()
                    .map_err(|_| "invalid --response-damping".to_owned())?
            }
            "--field-wave" => {
                let v = parse_vector(&value, &flag)?;
                field_options.wave = Vec3::new(v[0], v[1], v[2]);
            }
            "--field-k-offset" => {
                let v = parse_vector(&value, &flag)?;
                field_options.k_offset = Vec3::new(v[0], v[1], v[2]);
            }
            "--field-scalar" => {
                field_options.scalar_potential = parse_field_complex(&value, &flag)?;
            }
            "--field-frequency" => {
                let z = parse_field_complex(&value, &flag)?;
                field_options.frequency = dft_rs::response::Frequency::Retarded {
                    omega: z.re,
                    damping: z.im,
                };
            }
            "--field-vector" => {
                let v = value
                    .split(';')
                    .map(|s| parse_field_complex(s, &flag))
                    .collect::<Result<Vec<_>, _>>()?;
                field_options.vector_potential = v
                    .try_into()
                    .map_err(|_| "--field-vector requires three complex components".to_owned())?;
            }
            "--magnetic-gauge" => {
                magnetic_gauge = dft_rs::properties::magnetic::Gauge::from_name(&value)
                    .map_err(|e| e.to_string())?;
            }
            "--magnetic-origin" => {
                magnetic_origin = parse_vector(&value, "--magnetic-origin")?;
            }
            "--response-tol" => {
                response_tol = value
                    .parse()
                    .map_err(|_| "invalid --response-tol".to_owned())?
            }
            "--response-max-iter" => {
                response_max_iter = value
                    .parse()
                    .map_err(|_| "invalid --response-max-iter".to_owned())?
            }
            "--electric-assembly" => electric_assembly = value,
            "--orbital-coordinates" => {
                let coordinates: Vec<usize> = value
                    .split(',')
                    .map(|v| {
                        v.trim()
                            .parse()
                            .map_err(|_| "invalid --orbital-coordinates")
                    })
                    .collect::<Result<_, _>>()?;
                let mut sorted = coordinates.clone();
                sorted.sort_unstable();
                if sorted.is_empty() || sorted.windows(2).any(|w| w[0] == w[1]) {
                    return Err("orbital coordinates must be nonempty and unique".into());
                }
                orbital_options.coordinates = Some(coordinates);
            }
            "--degeneracy-tol" => {
                orbital_options.degeneracy_tolerance =
                    value.parse().map_err(|_| "invalid --degeneracy-tol")?;
                if !orbital_options.degeneracy_tolerance.is_finite()
                    || orbital_options.degeneracy_tolerance < 0.0
                {
                    return Err("degeneracy tolerance must be finite and non-negative".into());
                }
            }
            "--force-coordinates" => {
                let coordinates: Vec<usize> = value
                    .split(',')
                    .map(|s| s.trim().parse().map_err(|_| "invalid --force-coordinates"))
                    .collect::<Result<_, _>>()?;
                let mut sorted = coordinates.clone();
                sorted.sort_unstable();
                if sorted.is_empty() || sorted.windows(2).any(|w| w[0] == w[1]) {
                    return Err("--force-coordinates requires nonempty unique Cartesian IDs".into());
                }
                force_coordinates = Some(coordinates);
            }
            "--vibrational-modes" => {
                vibrational_modes = Some(
                    value
                        .split(',')
                        .map(|s| {
                            s.trim()
                                .parse::<usize>()
                                .map_err(|_| "invalid --vibrational-modes")
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "--anharmonic-method" => {
                anharmonic_config.method =
                    dft_rs::response::anharmonic::AnharmonicMethod::parse(&value)
                        .map_err(|e| e.to_string())?;
            }
            "--equilibrium-tol" => {
                anharmonic_config.equilibrium_tolerance =
                    value.parse().map_err(|_| "invalid --equilibrium-tol")?
            }
            "--resonance-tol" => {
                anharmonic_config.resonance_tolerance =
                    value.parse().map_err(|_| "invalid --resonance-tol")?
            }
            "--resonance-coupling-factor" => {
                anharmonic_config.resonance_coupling_factor = value
                    .parse()
                    .map_err(|_| "invalid --resonance-coupling-factor")?
            }
            "--max-vibrational-basis" => {
                anharmonic_config.max_basis_states = value
                    .parse()
                    .map_err(|_| "invalid --max-vibrational-basis")?
            }
            "--max-polyad-states" => {
                anharmonic_config.max_polyad_states =
                    value.parse().map_err(|_| "invalid --max-polyad-states")?
            }
            "--electric-order" => {
                electric_order = value.parse().map_err(|_| "invalid --electric-order")?;
                if !(1..=3).contains(&electric_order) {
                    return Err("--electric-order must be 1, 2 or 3".into());
                }
            }
            "--response-order" => {
                response_order = value
                    .parse()
                    .map_err(|_| "invalid --response-order".to_owned())?
            }
            "--field-direction" => {
                let parts: Vec<f64> = value
                    .split([',', ' '])
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| "invalid --field-direction".to_owned())
                    })
                    .collect::<Result<_, _>>()?;
                if parts.len() != 3 || parts.iter().any(|x| !x.is_finite()) {
                    return Err("--field-direction takes three finite components".to_owned());
                }
                field_direction = [parts[0], parts[1], parts[2]];
            }
            "--field-directions" => {
                let directions: Vec<_> = value
                    .split(';')
                    .map(|v| {
                        let parts: Vec<f64> = v
                            .split([',', ' '])
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| {
                                s.trim()
                                    .parse::<f64>()
                                    .map_err(|_| "invalid --field-directions".to_owned())
                            })
                            .collect::<Result<_, _>>()?;
                        if parts.len() != 3 || parts.iter().any(|x| !x.is_finite()) {
                            return Err(
                                "each field direction takes three finite components".to_owned()
                            );
                        }
                        Ok([parts[0], parts[1], parts[2]])
                    })
                    .collect::<Result<_, _>>()?;
                mixed_field_directions = Some(directions);
            }
            "--nuclear-direction" | "--nuclear-directions" => {
                let parse_path = |path: &str| -> Result<Vec<[f64; 3]>, String> {
                    path.split(';')
                        .map(|atom| {
                            let c = atom
                                .split(',')
                                .map(|x| {
                                    x.trim()
                                        .parse::<f64>()
                                        .map_err(|_| "invalid nuclear direction".to_owned())
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            if c.len() != 3 || c.iter().any(|x| !x.is_finite()) {
                                return Err(
                                    "each nuclear displacement requires three finite components"
                                        .into(),
                                );
                            }
                            Ok([c[0], c[1], c[2]])
                        })
                        .collect()
                };
                if flag == "--nuclear-direction" {
                    nuclear_direction = Some(parse_path(&value)?);
                } else {
                    mixed_nuclear_directions =
                        Some(value.split('|').map(parse_path).collect::<Result<_, _>>()?);
                }
            }
            "--grid-weight-cutoff" => {
                let cutoff: f64 = value
                    .parse()
                    .map_err(|_| format!("invalid grid weight cutoff: {value}"))?;
                if !cutoff.is_finite() || cutoff < 0.0 {
                    return Err("grid weight cutoff must be finite and nonnegative".into());
                }
                grid_weight_cutoff = Some(cutoff);
            }
            "--grid-level" => {
                grid_level = value
                    .parse()
                    .map_err(|_| format!("invalid grid level: {value}"))?;
                if grid_level > 9 {
                    return Err("grid level must be 0..=9".to_owned());
                }
            }
            "--vv10-grid-level" => {
                vv10_grid_level = value
                    .parse()
                    .map_err(|_| format!("invalid VV10 grid level: {value}"))?;
                if vv10_grid_level > 9 {
                    return Err("VV10 grid level must be 0..=9".to_owned());
                }
            }
            "--field" => {
                let parts: Vec<f64> = value
                    .split([',', ' '])
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<f64>()
                            .map_err(|_| format!("invalid --field component: {s}"))
                    })
                    .collect::<Result<_, _>>()?;
                if parts.len() != 3 {
                    return Err("--field takes three components: Ex,Ey,Ez (a.u.)".to_owned());
                }
                field = Some([parts[0], parts[1], parts[2]]);
            }
            "--cell" => cell = Some(parse_cell(&value)?),
            "--basis-grid-cutoff" => {
                basis_grid_cutoff = value.parse().map_err(|_| "invalid --basis-grid-cutoff")?;
            }
            "--pbc" => {
                let flags: Vec<_> = value
                    .split(',')
                    .map(|v| match v.trim() {
                        "0" => Ok(false),
                        "1" => Ok(true),
                        _ => Err("--pbc requires three flags 0 or 1"),
                    })
                    .collect::<Result<_, _>>()?;
                periodic_axes = flags
                    .try_into()
                    .map_err(|_| "--pbc requires three flags 0 or 1")?;
            }
            "--electrostatic-charges" | "--electrostatic-widths" => {
                let values: Vec<f64> = value
                    .split(',')
                    .map(|v| v.trim().parse().map_err(|_| format!("invalid {flag}")))
                    .collect::<Result<_, _>>()?;
                if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                    return Err(format!("{flag} requires finite nonempty values"));
                }
                if flag == "--electrostatic-charges" {
                    electrostatic_charges = Some(values);
                } else {
                    electrostatic_widths = Some(values);
                }
            }
            "--charge-compensation" => compensation_id = value,
            "--density-matrices" => density_matrices = Some(value),
            "--alpha-density-matrices" => spin_density_matrices[0] = Some(value),
            "--beta-density-matrices" => spin_density_matrices[1] = Some(value),
            "--xc-operator-tol" => {
                xc_operator_tolerance =
                    value.parse().map_err(|_| "invalid XC operator tolerance")?
            }
            "--xc-grid-tol" => {
                xc_grid_tolerance = value.parse().map_err(|_| "invalid XC grid tolerance")?
            }
            "--nuclear-widths" => {
                nuclear_widths = Some(
                    value
                        .split(',')
                        .map(|v| {
                            v.trim()
                                .parse::<f64>()
                                .map_err(|_| "invalid --nuclear-widths")
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            "--gaussian-exponents" => {
                gaussian_exponents = Some(
                    value
                        .split(',')
                        .map(|v| {
                            v.trim()
                                .parse::<f64>()
                                .map_err(|_| "invalid --gaussian-exponents")
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "--gaussian-angular" => {
                gaussian_angular = Some(
                    value
                        .split(';')
                        .map(|triple| {
                            let parts = triple
                                .split(',')
                                .map(|v| {
                                    v.trim()
                                        .parse::<usize>()
                                        .map_err(|_| "invalid --gaussian-angular")
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            parts
                                .try_into()
                                .map_err(|_| "--gaussian-angular requires lx,ly,lz triples")
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "--gaussian-source-charge" => {
                gaussian_source_charge = Some(
                    value
                        .parse()
                        .map_err(|_| "invalid --gaussian-source-charge")?,
                )
            }
            "--gaussian-source-width" => {
                gaussian_source_width = Some(
                    value
                        .parse()
                        .map_err(|_| "invalid --gaussian-source-width")?,
                )
            }
            "--gaussian-tol" => {
                gaussian_options.tolerance = value.parse().map_err(|_| "invalid --gaussian-tol")?
            }
            "--gaussian-quadrature" => {
                gaussian_options.quadrature_order =
                    value.parse().map_err(|_| "invalid --gaussian-quadrature")?
            }
            "--countercharge-center" => {
                let values: Vec<f64> = value
                    .split(',')
                    .map(|v| {
                        v.trim()
                            .parse()
                            .map_err(|_| "invalid --countercharge-center")
                    })
                    .collect::<Result<_, _>>()?;
                let xyz: [f64; 3] = values
                    .try_into()
                    .map_err(|_| "--countercharge-center requires three Bohr coordinates")?;
                countercharge_center = Some(Vec3::new(xyz[0], xyz[1], xyz[2]));
            }
            "--countercharge-width" => {
                countercharge_width =
                    Some(value.parse().map_err(|_| "invalid --countercharge-width")?)
            }
            "--ewald-alpha" => {
                electrostatic_options.alpha =
                    Some(value.parse().map_err(|_| "invalid --ewald-alpha")?)
            }
            "--ewald-real-cutoff" => {
                electrostatic_options.real_cutoff =
                    Some(value.parse().map_err(|_| "invalid --ewald-real-cutoff")?)
            }
            "--ewald-reciprocal-cutoff" => {
                electrostatic_options.reciprocal_cutoff = Some(
                    value
                        .parse()
                        .map_err(|_| "invalid --ewald-reciprocal-cutoff")?,
                )
            }
            "--electrostatic-tol" => {
                electrostatic_options.tolerance =
                    value.parse().map_err(|_| "invalid --electrostatic-tol")?
            }
            "--electrostatic-quadrature" => {
                electrostatic_options.quadrature_order = value
                    .parse()
                    .map_err(|_| "invalid --electrostatic-quadrature")?
            }
            "--k-mesh" | "--kmesh" => {
                let parts: Vec<usize> = value
                    .split([',', 'x', ' '])
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .map_err(|_| format!("invalid --k-mesh component: {s}"))
                    })
                    .collect::<Result<_, _>>()?;
                if parts.len() != 3 || parts.contains(&0) {
                    return Err("--k-mesh takes three positive integers: nx,ny,nz".to_owned());
                }
                k_mesh = [parts[0], parts[1], parts[2]];
            }
            "--overlap-cutoff" => {
                overlap_cutoff = value
                    .parse()
                    .map_err(|_| format!("invalid --overlap-cutoff: {value}"))?
            }
            "--exchange-cutoff" => {
                exchange_cutoff = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --exchange-cutoff: {value}"))?,
                )
            }
            "--grid-shell" => {
                grid_shell = value
                    .parse()
                    .map_err(|_| format!("invalid --grid-shell: {value}"))?;
                if grid_shell < 1 {
                    return Err("--grid-shell must be at least 1".to_owned());
                }
            }
            "--temperature" => {
                temperature = value
                    .parse()
                    .map_err(|_| format!("invalid --temperature: {value}"))?;
                if !(temperature.is_finite() && temperature >= 0.0) {
                    return Err("--temperature must be finite and non-negative".to_owned());
                }
            }
            "--thermo-temperature" => {
                thermo_temperature = value
                    .parse()
                    .map_err(|_| format!("invalid --thermo-temperature: {value}"))?;
                if !(thermo_temperature.is_finite() && thermo_temperature > 0.0) {
                    return Err("--thermo-temperature must be finite and positive".to_owned());
                }
            }
            "--pressure" => {
                pressure = value
                    .parse()
                    .map_err(|_| format!("invalid --pressure: {value}"))?;
                if !(pressure.is_finite() && pressure > 0.0) {
                    return Err("--pressure must be finite and positive".to_owned());
                }
            }
            "--symmetry-number" => {
                symmetry_number = value
                    .parse()
                    .map_err(|_| format!("invalid --symmetry-number: {value}"))?;
                if symmetry_number == 0 {
                    return Err("--symmetry-number must be at least 1".to_owned());
                }
            }
            _ => return Err(format!("unknown option: {flag}\n\n{}", cli_usage())),
        }
    }
    if !matches!(command.as_str(), "magnetic-response" | "nmr")
        && ["--magnetic-gauge", "--magnetic-origin"]
            .iter()
            .any(|f| flags_given.contains(*f))
    {
        return Err("magnetic gauge controls require magnetic-response or nmr".into());
    }
    if flags_given.contains("--vv10-grid-level")
        && matches!(
            command.as_str(),
            "periodic"
                | "periodic-scf"
                | "periodic-response"
                | "field-response"
                | "dielectric"
                | "thermal-response"
                | "dimensional-scf"
                | "dimensional-field"
                | "periodic-basis"
                | "periodic-xc"
                | "periodic-electrostatics"
                | "periodic-coulomb"
                | "periodic-gaussian-integral"
                | "periodic-field"
                | "sapt"
        )
    {
        return Err("--vv10-grid-level requires a molecular SCF/response command".into());
    }
    if matches!(command.as_str(), "magnetic-response" | "nmr") {
        let allowed = [
            "--basis",
            "--ecp",
            "--xc",
            "--x",
            "--c",
            "--charge",
            "--multiplicity",
            "--mult",
            "--grid-level",
            "--grid-weight-cutoff",
            "--vv10-grid-level",
            "--max-iter",
            "--incore",
            "--incore-mb",
            "--unrestricted",
            "--conv-tol",
            "--conv-tol-grad",
            "--conv-tol-dp",
            "--diis-size",
            "--adiis-until-error",
            "--soscf-from-error",
            "--no-adiis",
            "--no-soscf",
            "--magnetic-gauge",
            "--magnetic-origin",
            "--log-mode",
            "--log-file",
            "--response-tol",
            "--response-max-iter",
            "--response-restart",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} is not a supported molecular magnetic control"
            ));
        }
    }
    let periodic_commands = [
        "periodic",
        "periodic-response",
        "field-response",
        "dielectric",
        "thermal-response",
        "periodic-electrostatics",
        "periodic-basis",
        "periodic-gaussian-integral",
        "periodic-coulomb",
        "periodic-xc",
        "periodic-field",
        "dimensional-scf",
    ];
    if flags_given.contains("--pbc") && !periodic_commands.contains(&command.as_str()) {
        return Err("--pbc requires a periodic command".into());
    }
    if command == "periodic-field" {
        let allowed = [
            "--cell",
            "--pbc",
            "--basis",
            "--charge",
            "--k-mesh",
            "--kmesh",
            "--overlap-cutoff",
            "--field-wave",
            "--field-k-offset",
            "--field-scalar",
            "--field-vector",
            "--field-frequency",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} does not apply to bare periodic field operators"
            ));
        }
        if cell.is_none() {
            return Err("periodic-field requires --cell".into());
        }
        field_options
            .frequency
            .validate()
            .map_err(|e| e.to_string())?;
    } else if command == "field-response"
        || (command == "dimensional-scf" && field_transfer.is_some())
    {
        if [
            "--field-wave",
            "--field-k-offset",
            "--field-frequency",
            "--response-modes",
            "--response-q",
            "--reciprocal-cutoff",
        ]
        .iter()
        .any(|f| {
            flags_given.contains(*f)
                && (command == "dimensional-scf"
                    || !matches!(*f, "--response-q" | "--reciprocal-cutoff"))
        }) {
            return Err(
                "field-response uses --response-q and response frequency controls; \
                offset and source-mode controls do not apply"
                    .into(),
            );
        }
    } else if [
        "--field-wave",
        "--field-k-offset",
        "--field-scalar",
        "--field-vector",
        "--field-frequency",
    ]
    .iter()
    .any(|f| flags_given.contains(*f))
    {
        return Err("Fourier field controls require periodic-field or field-response".into());
    }
    if command == "periodic-basis" {
        let allowed = [
            "--cell",
            "--pbc",
            "--k-mesh",
            "--kmesh",
            "--basis",
            "--overlap-cutoff",
            "--basis-grid",
            "--basis-grid-derivatives",
            "--basis-grid-cutoff",
            "--grid-level",
            "--grid-weight-cutoff",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!("{flag} does not apply to periodic basis operators"));
        }
        if cell.is_none() {
            return Err("periodic-basis requires --cell".into());
        }
        if !basis_grid
            && (basis_grid_derivatives
                || flags_given.contains("--grid-level")
                || flags_given.contains("--grid-weight-cutoff")
                || flags_given.contains("--basis-grid-cutoff"))
        {
            return Err("grid controls require periodic-basis --basis-grid".into());
        }
        if !overlap_cutoff.is_finite()
            || overlap_cutoff <= 0.0
            || !basis_grid_cutoff.is_finite()
            || basis_grid_cutoff <= 0.0
        {
            return Err("periodic basis cutoffs must be finite and positive".into());
        }
    } else if !matches!(command.as_str(), "periodic-xc" | "dimensional-scf")
        && flags_given.iter().any(|f| f.starts_with("--basis-grid"))
    {
        return Err("basis grid controls require periodic-basis".into());
    }
    if command == "periodic-gaussian-integral" {
        let allowed = [
            "--cell",
            "--pbc",
            "--gaussian-exponents",
            "--gaussian-angular",
            "--gaussian-source-charge",
            "--gaussian-source-width",
            "--gaussian-tol",
            "--gaussian-quadrature",
            "--ewald-alpha",
            "--ewald-real-cutoff",
            "--ewald-reciprocal-cutoff",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} does not apply to a primitive Gaussian integral"
            ));
        }
        if cell.is_none() || gaussian_exponents.is_none() || gaussian_angular.is_none() {
            return Err(
                "periodic-gaussian-integral requires --cell, --gaussian-exponents and \
                --gaussian-angular"
                    .into(),
            );
        }
        gaussian_options.alpha = electrostatic_options.alpha;
        gaussian_options.real_cutoff = electrostatic_options.real_cutoff;
        gaussian_options.reciprocal_cutoff = electrostatic_options.reciprocal_cutoff;
    } else if flags_given.iter().any(|f| f.starts_with("--gaussian-")) {
        return Err("Gaussian primitive controls require periodic-gaussian-integral".into());
    }
    if command == "dimensional-scf" {
        let mut allowed = vec![
            "--cell",
            "--pbc",
            "--k-mesh",
            "--kmesh",
            "--basis",
            "--xc",
            "--x",
            "--c",
            "--charge",
            "--multiplicity",
            "--unrestricted",
            "--overlap-cutoff",
            "--basis-grid-cutoff",
            "--grid-level",
            "--grid-weight-cutoff",
            "--nuclear-widths",
            "--charge-compensation",
            "--countercharge-center",
            "--countercharge-width",
            "--ewald-alpha",
            "--ewald-real-cutoff",
            "--ewald-reciprocal-cutoff",
            "--electrostatic-tol",
            "--electrostatic-quadrature",
            "--xc-operator-tol",
            "--xc-grid-tol",
            "--temperature",
            "--max-iter",
            "--conv-tol",
            "--conv-tol-dp",
            "--conv-tol-grad",
            "--guess",
            "--exchange-cutoff",
            "--no-adiis",
            "--adiis-until-error",
            "--damping",
            "--level-shift",
            "--diis-size",
            "--log-mode",
            "--log-file",
        ];
        if field_transfer.is_some() {
            allowed.extend([
                "--field-transfer",
                "--field-scalar",
                "--field-vector",
                "--response-omega",
                "--response-damping",
                "--imaginary-frequency",
                "--response-tol",
                "--response-max-iter",
                "--response-restart",
                "--fixed-chemical-potential",
            ]);
        }
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!("{flag} is not a dimensional SCF control"));
        }
        if cell.is_none() {
            return Err("dimensional-scf requires --cell".into());
        }
    }
    if field_transfer.is_some() && command != "dimensional-scf" {
        return Err("--field-transfer requires dimensional-scf".into());
    }
    if let Some(fractional) = field_transfer {
        if (0..3).any(|a| (!periodic_axes[a] && fractional[a] != 0.0) || fractional[a].abs() > 1e12)
        {
            return Err(
                "fractional field transfer must use periodic axes and remain in finite \
                coordinate range"
                    .into(),
            );
        }
    }
    if command == "periodic-xc" {
        let allowed = [
            "--cell",
            "--pbc",
            "--k-mesh",
            "--kmesh",
            "--basis",
            "--charge",
            "--xc",
            "--x",
            "--c",
            "--overlap-cutoff",
            "--basis-grid-cutoff",
            "--grid-level",
            "--grid-weight-cutoff",
            "--alpha-density-matrices",
            "--beta-density-matrices",
            "--xc-operator-tol",
            "--xc-grid-tol",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} does not apply to supplied-density semilocal XC operators"
            ));
        }
        if cell.is_none() || spin_density_matrices.iter().any(Option::is_none) {
            return Err(
                "periodic-xc requires --cell, --alpha-density-matrices and --beta-density-matrices"
                    .into(),
            );
        }
    } else if spin_density_matrices.iter().any(Option::is_some)
        || (command != "dimensional-scf"
            && (flags_given.contains("--xc-operator-tol") || flags_given.contains("--xc-grid-tol")))
    {
        return Err("spin density files and XC tolerance controls require periodic-xc".into());
    }
    if command == "periodic-coulomb" {
        let allowed = [
            "--cell",
            "--pbc",
            "--k-mesh",
            "--kmesh",
            "--basis",
            "--charge",
            "--overlap-cutoff",
            "--density-matrices",
            "--nuclear-widths",
            "--charge-compensation",
            "--countercharge-center",
            "--countercharge-width",
            "--ewald-alpha",
            "--ewald-real-cutoff",
            "--ewald-reciprocal-cutoff",
            "--electrostatic-tol",
            "--electrostatic-quadrature",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} does not apply to supplied-density Coulomb operators"
            ));
        }
        if cell.is_none() {
            return Err("periodic-coulomb requires --cell".into());
        }
    } else if density_matrices.is_some()
        || (command != "dimensional-scf" && nuclear_widths.is_some())
    {
        return Err("density matrices and nuclear widths require periodic-coulomb".into());
    }
    if command == "periodic-electrostatics" {
        let allowed = [
            "--cell",
            "--pbc",
            "--k-mesh",
            "--kmesh",
            "--electrostatic-charges",
            "--electrostatic-widths",
            "--charge-compensation",
            "--countercharge-center",
            "--countercharge-width",
            "--ewald-alpha",
            "--ewald-real-cutoff",
            "--ewald-reciprocal-cutoff",
            "--electrostatic-tol",
            "--electrostatic-quadrature",
            "--log-mode",
            "--log-file",
        ];
        if let Some(flag) = flags_given.iter().find(|f| !allowed.contains(&f.as_str())) {
            return Err(format!(
                "{flag} does not apply to prescribed-charge periodic electrostatics"
            ));
        }
        if cell.is_none() || electrostatic_charges.is_none() {
            return Err(
                "periodic-electrostatics requires --cell and --electrostatic-charges".into(),
            );
        }
    }
    if matches!(
        command.as_str(),
        "periodic-electrostatics" | "periodic-coulomb" | "dimensional-scf"
    ) {
        use dft_rs::periodic::electrostatics::ChargeCompensation;
        electrostatic_options.compensation = match compensation_id.as_str() {
            "neutral" if countercharge_center.is_none() && countercharge_width.is_none() => {
                ChargeCompensation::RequireNeutral
            }
            "uniform" if countercharge_center.is_none() && countercharge_width.is_none() => {
                ChargeCompensation::UniformBackground
            }
            "gaussian" => ChargeCompensation::GaussianCountercharge {
                center: countercharge_center
                    .ok_or("Gaussian compensation requires --countercharge-center")?,
                width: countercharge_width
                    .ok_or("Gaussian compensation requires --countercharge-width")?,
            },
            _ => {
                return Err(
                    "charge compensation must be neutral, uniform or gaussian; center/width \
                belong only to gaussian"
                        .into(),
                )
            }
        };
    } else if command != "periodic-gaussian-integral"
        && flags_given.iter().any(|f| {
            f.starts_with("--electrostatic-")
                || f.starts_with("--ewald-")
                || f.starts_with("--countercharge-")
                || f == "--charge-compensation"
        })
    {
        return Err("electrostatic controls require periodic-electrostatics".into());
    }
    if matches!(
        command.as_str(),
        "periodic" | "periodic-response" | "dielectric" | "thermal-response" | "field-response"
    ) && cell.is_none()
    {
        return Err("the periodic command needs --cell (see --help)".to_owned());
    }
    if relax && command != "periodic" {
        return Err("--relax is only available for the periodic command".to_owned());
    }
    if flags_given.contains("--saddle-method") && command != "saddle" {
        return Err("--saddle-method is only available for the saddle command".to_owned());
    }
    if !matches!(
        saddle_method.as_str(),
        "prfo" | "p-rfo" | "rsirfo" | "rs-i-rfo"
    ) {
        return Err("--saddle-method must be prfo or rsirfo".to_owned());
    }
    let opt_controls_given = [
        "--opt-max-steps",
        "--opt-gtol",
        "--opt-step-max",
        "--opt-history",
    ]
    .iter()
    .any(|flag| flags_given.contains(*flag));
    if opt_controls_given && !(command == "periodic" && relax) {
        return Err(
            "--opt-max-steps/--opt-gtol/--opt-step-max/--opt-history require periodic --relax"
                .to_owned(),
        );
    }
    let thermo_controls_given = ["--thermo-temperature", "--pressure", "--symmetry-number"]
        .iter()
        .any(|flag| flags_given.contains(*flag));
    if thermo_controls_given && command != "frequencies" {
        return Err(
            "--thermo-temperature/--pressure/--symmetry-number are only available for frequencies"
                .to_owned(),
        );
    }
    if trajectory.is_some()
        && command != "optimize"
        && command != "saddle"
        && !(command == "periodic" && relax)
    {
        return Err("--traj requires optimize, saddle, or periodic --relax".to_owned());
    }
    let adiis_controls_given = no_adiis || adiis_until_error.is_some();
    if adiis_controls_given && matches!(command.as_str(), "info" | "integrals") {
        return Err(format!(
            "--no-adiis/--adiis-until-error are SCF controls and are not available for {command}"
        ));
    }
    if adiis_controls_given && (list_functionals || list_solvents) {
        return Err(
            "--no-adiis/--adiis-until-error cannot be combined with catalog list flags".to_owned(),
        );
    }
    if no_adiis && adiis_until_error.is_some() {
        return Err(
            "--no-adiis cannot be combined with --adiis-until-error because ADIIS is disabled"
                .to_owned(),
        );
    }
    if xc.is_some() && (x_func.is_some() || c_func.is_some()) {
        return Err("--xc cannot be combined with --x/--c".to_owned());
    }
    if x_func.is_some() != c_func.is_some() {
        return Err(
            "--x and --c must be given together (use --c none for exchange-only)".to_owned(),
        );
    }
    if stab_follow
        && !matches!(
            command.as_str(),
            "stability"
                | "stab-following"
                | "energy"
                | "charges"
                | "gradient"
                | "hessian"
                | "frequencies"
                | "polarizability"
                | "magnetic-response"
                | "nmr"
                | "dynamic-polarizability"
                | "dynamic-beta"
                | "dynamic-gamma"
                | "field-taylor"
                | "field-mixed"
                | "beta"
                | "gamma"
                | "nuclear-taylor"
                | "nuclear-mixed"
                | "molecular-taylor"
                | "molecular-mixed"
                | "static-spectroscopy"
                | "force-constants"
                | "anharmonic"
                | "orbital-energy-gradients"
                | "frontier-gap-gradient"
                | "raman"
                | "tddft"
        )
    {
        return Err(
            "stab-follow requires a molecular SCF/property command with an explicit \
            reference state"
                .into(),
        );
    }
    if !stab_follow
        && !matches!(command.as_str(), "stability" | "stab-following")
        && flags_given.iter().any(|f| f.starts_with("--stab-"))
    {
        return Err("stability controls require stability or --stab-follow".into());
    }
    if matches!(command.as_str(), "stability" | "stab-following") {
        if flags_given.contains("--response-restart") {
            return Err("stability uses Davidson, not GMRES restart".into());
        }
        for (response, stab) in [
            ("--response-tol", "--stab-tol"),
            ("--response-max-iter", "--stab-max-iter"),
        ] {
            if flags_given.contains(response) && flags_given.contains(stab) {
                return Err(format!("{response} cannot be combined with {stab}"));
            }
        }
        if flags_given.contains("--response-tol") {
            stabilize_options.analysis.tolerance = response_tol;
        }
        if flags_given.contains("--response-max-iter") {
            stabilize_options.analysis.max_iterations = response_max_iter;
        }
    }
    if stab_follow || matches!(command.as_str(), "stability" | "stab-following") {
        stabilize_options.validate().map_err(|e| e.to_string())?;
    }
    // Command x option contract: refuse any option a command silently
    // ignores, before any parsing of the geometry or basis, let alone an
    // SCF, ever starts.
    let response_controls = [
        "--response-omega",
        "--response-damping",
        "--imaginary-frequency",
    ];
    if flags_given.contains("--input-frequencies")
        && !matches!(command.as_str(), "dynamic-beta" | "dynamic-gamma")
    {
        return Err("--input-frequencies requires dynamic-beta or dynamic-gamma".into());
    }
    if (electric_process.is_some() || process_frequencies.is_some())
        && !matches!(command.as_str(), "dynamic-beta" | "dynamic-gamma")
    {
        return Err("electric process presets require dynamic-beta or dynamic-gamma".into());
    }
    if process_frequencies.is_some() && electric_process.is_none() {
        return Err("--process-frequencies requires --electric-process".into());
    }
    if let Some(process) = electric_process {
        let order = if command == "dynamic-beta" { 2 } else { 3 };
        if process.order() != order {
            return Err(format!(
                "{} requires dynamic-{}",
                process.name(),
                if process.order() == 2 {
                    "beta"
                } else {
                    "gamma"
                }
            ));
        }
        if input_frequencies.is_some() || imaginary_frequency {
            return Err(
                "electric process presets cannot be combined with explicit input pairs \
                or imaginary frequency"
                    .into(),
            );
        }
        if process_frequencies.is_some() && flags_given.contains("--response-omega") {
            return Err("process frequencies cannot be combined with response-omega".into());
        }
        let magnitudes = process_frequencies
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![response_omega]);
        process
            .inputs(&magnitudes, response_damping)
            .map_err(|e| e.to_string())?;
    }
    if matches!(command.as_str(), "dynamic-beta" | "dynamic-gamma") {
        if input_frequencies.is_some()
            && (flags_given.contains("--response-omega")
                || flags_given.contains("--response-damping"))
        {
            return Err(
                "explicit input frequencies cannot be combined with single-frequency defaults"
                    .into(),
            );
        }
        let order = if command == "dynamic-beta" { 2 } else { 3 };
        let pairs = input_frequencies
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![[response_omega, response_damping]; order]);
        if pairs.len() != order {
            return Err(format!("{command} requires {order} input frequency pairs"));
        }
        for [omega, damping] in pairs {
            let f = if imaginary_frequency {
                if damping != 0.0 {
                    return Err("imaginary input frequencies require zero damping".into());
                }
                dft_rs::response::Frequency::Imaginary { frequency: omega }
            } else {
                dft_rs::response::Frequency::Retarded { omega, damping }
            };
            f.validate().map_err(|e| e.to_string())?;
        }
    }
    if !matches!(
        command.as_str(),
        "dynamic-polarizability"
            | "dynamic-beta"
            | "dynamic-gamma"
            | "field-response"
            | "dimensional-scf"
            | "periodic-response"
            | "dielectric"
            | "thermal-response"
    ) && response_controls
        .iter()
        .any(|flag| flags_given.contains(*flag))
    {
        return Err(
            "response frequency controls require dynamic-polarizability, \
            periodic-response, or dielectric"
                .to_owned(),
        );
    }
    if !matches!(command.as_str(), "field-taylor" | "molecular-taylor")
        && flags_given.contains("--field-direction")
    {
        return Err("--field-direction requires field-taylor or molecular-taylor".to_owned());
    }
    if !matches!(
        command.as_str(),
        "field-taylor" | "nuclear-taylor" | "molecular-taylor" | "force-constants"
    ) && flags_given.contains("--response-order")
    {
        return Err("--response-order requires a Taylor command".to_owned());
    }
    if !matches!(
        command.as_str(),
        "nuclear-taylor" | "molecular-taylor" | "static-spectroscopy"
    ) && flags_given.contains("--nuclear-direction")
    {
        return Err(
            "--nuclear-direction requires nuclear-taylor, molecular-taylor or static-spectroscopy"
                .into(),
        );
    }
    if !matches!(
        command.as_str(),
        "nuclear-mixed" | "molecular-mixed" | "force-constants"
    ) && flags_given.contains("--nuclear-directions")
    {
        return Err("--nuclear-directions requires nuclear-mixed or molecular-mixed".into());
    }
    if command != "static-spectroscopy" && flags_given.contains("--electric-order") {
        return Err("--electric-order requires static-spectroscopy".into());
    }
    if !matches!(
        command.as_str(),
        "orbital-energy-gradients" | "frontier-gap-gradient"
    ) && (flags_given.contains("--orbital-coordinates")
        || flags_given.contains("--degeneracy-tol"))
    {
        return Err(
            "orbital controls require orbital-energy-gradients or frontier-gap-gradient".into(),
        );
    }
    if command != "force-constants" && force_coordinates.is_some() {
        return Err("--force-coordinates requires force-constants".into());
    }
    if flags_given.contains("--electric-assembly") && command != "beta" {
        return Err("--electric-assembly requires beta".into());
    }
    if !matches!(electric_assembly.as_str(), "projector" | "variational") {
        return Err("--electric-assembly must be projector or variational".into());
    }
    let anharmonic_controls = [
        "--anharmonic-method",
        "--anharmonic-raman",
        "--vibrational-modes",
        "--equilibrium-tol",
        "--resonance-tol",
        "--resonance-coupling-factor",
        "--max-vibrational-basis",
        "--max-polyad-states",
    ];
    if command != "anharmonic" && anharmonic_controls.iter().any(|f| flags_given.contains(*f)) {
        return Err("anharmonic controls require the anharmonic command".into());
    }
    if command == "anharmonic" {
        if flags_given.contains("--field") {
            return Err("anharmonic currently uses a zero-field stationary reference".into());
        }
        anharmonic_config.tolerance = response_tol;
        anharmonic_config.memory_mb = incore_mb.unwrap_or(512).min(512);
        anharmonic_config.validate().map_err(|e| e.to_string())?;
        if let Some(ids) = &vibrational_modes {
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            if ids.is_empty() || sorted.windows(2).any(|w| w[0] == w[1]) {
                return Err(
                    "--vibrational-modes requires nonempty unique harmonic mode IDs".into(),
                );
            }
        }
    }
    if command == "force-constants"
        && (!(3..=4).contains(&response_order)
            || force_coordinates.is_some() && mixed_nuclear_directions.is_some())
    {
        return Err(
            "force-constants requires --response-order 3 or 4 and either \
            --force-coordinates or --nuclear-directions"
                .into(),
        );
    }
    if matches!(
        command.as_str(),
        "nuclear-taylor"
            | "nuclear-mixed"
            | "molecular-taylor"
            | "molecular-mixed"
            | "static-spectroscopy"
            | "force-constants"
            | "anharmonic"
            | "orbital-energy-gradients"
            | "frontier-gap-gradient"
    ) {
        if matches!(
            command.as_str(),
            "nuclear-taylor" | "molecular-taylor" | "static-spectroscopy"
        ) {
            if nuclear_direction.is_none() {
                return Err("this molecular response command requires --nuclear-direction".into());
            }
            if !flags_given.contains("--response-order") {
                response_order = 4;
            }
        } else if !matches!(
            command.as_str(),
            "force-constants" | "anharmonic" | "orbital-energy-gradients" | "frontier-gap-gradient"
        ) && mixed_nuclear_directions.is_none()
        {
            return Err("mixed nuclear response requires --nuclear-directions".into());
        }
        if command == "molecular-mixed"
            && (mixed_field_directions.is_none()
                || mixed_field_directions.as_ref().map(Vec::len)
                    != mixed_nuclear_directions.as_ref().map(Vec::len))
        {
            return Err(
                "molecular-mixed requires equally many --nuclear-directions and --field-directions"
                    .into(),
            );
        }
        if !matches!(disp.as_str(), "auto" | "none")
            || [
                "--ecp",
                "--density-fit",
                "--nlc",
                "--solvent",
                "--eps",
                "--refractive-index",
                "--solvent-alpha",
                "--solvent-beta",
                "--solvent-gamma",
                "--xdm-a1",
                "--xdm-a2",
            ]
            .iter()
            .any(|f| flags_given.contains(*f))
        {
            return Err(
                "nuclear response currently requires all-electron exact-integral models \
                without nonlocal, empirical or solvent terms"
                    .into(),
            );
        }
    }
    if !matches!(
        command.as_str(),
        "field-taylor"
            | "field-mixed"
            | "beta"
            | "gamma"
            | "nuclear-taylor"
            | "nuclear-mixed"
            | "molecular-taylor"
            | "molecular-mixed"
            | "static-spectroscopy"
            | "force-constants"
            | "anharmonic"
            | "orbital-energy-gradients"
            | "frontier-gap-gradient"
            | "magnetic-response"
            | "nmr"
            | "dynamic-polarizability"
            | "dynamic-beta"
            | "dynamic-gamma"
            | "stability"
            | "stab-following"
            | "field-response"
            | "dimensional-scf"
            | "periodic-response"
            | "dielectric"
            | "thermal-response"
    ) && [
        "--response-tol",
        "--response-max-iter",
        "--response-restart",
    ]
    .iter()
    .any(|f| flags_given.contains(*f))
    {
        return Err("response solver controls require a response command".to_owned());
    }
    if matches!(
        command.as_str(),
        "beta"
            | "gamma"
            | "field-taylor"
            | "nuclear-taylor"
            | "nuclear-mixed"
            | "molecular-taylor"
            | "molecular-mixed"
            | "static-spectroscopy"
            | "force-constants"
            | "anharmonic"
            | "orbital-energy-gradients"
            | "frontier-gap-gradient"
            | "magnetic-response"
            | "nmr"
    ) {
        dft_rs::response::ResponseSolverConfig {
            tolerance: response_tol,
            max_iterations: response_max_iter,
            gmres_restart: response_restart,
            ..Default::default()
        }
        .validate()
        .map_err(|e| e.to_string())?;
        if response_order == usize::MAX {
            return Err("response-order overflow".to_owned());
        }
    }
    if !matches!(command.as_str(), "field-mixed" | "molecular-mixed")
        && flags_given.contains("--field-directions")
    {
        return Err("--field-directions requires field-mixed or molecular-mixed".to_owned());
    }
    if command == "field-mixed" {
        if mixed_field_directions.is_none() {
            return Err("field-mixed requires --field-directions".to_owned());
        }
        dft_rs::response::ResponseSolverConfig {
            tolerance: response_tol,
            max_iterations: response_max_iter,
            gmres_restart: response_restart,
            ..Default::default()
        }
        .validate()
        .map_err(|e| e.to_string())?;
    }
    if matches!(
        command.as_str(),
        "dynamic-polarizability"
            | "dynamic-beta"
            | "dynamic-gamma"
            | "field-response"
            | "dimensional-scf"
            | "periodic-response"
            | "dielectric"
            | "thermal-response"
    ) {
        use dft_rs::response::{Frequency, ResponseSolverConfig};
        if imaginary_frequency && response_damping != 0.0 {
            return Err("--imaginary-frequency requires zero --response-damping".to_owned());
        }
        let frequency = if imaginary_frequency {
            Frequency::Imaginary {
                frequency: response_omega,
            }
        } else {
            Frequency::Retarded {
                omega: response_omega,
                damping: response_damping,
            }
        };
        frequency.validate().map_err(|e| e.to_string())?;
        ResponseSolverConfig {
            tolerance: response_tol,
            max_iterations: response_max_iter,
            gmres_restart: response_restart,
            ..Default::default()
        }
        .validate()
        .map_err(|e| e.to_string())?;
    }
    if !matches!(
        command.as_str(),
        "periodic-response"
            | "dielectric"
            | "thermal-response"
            | "field-response"
            | "dimensional-scf"
    ) && [
        "--response-q",
        "--reciprocal-cutoff",
        "--response-modes",
        "--fixed-chemical-potential",
    ]
    .iter()
    .any(|f| flags_given.contains(*f))
    {
        return Err("finite-q density controls require periodic-response or dielectric".to_owned());
    }
    if matches!(
        command.as_str(),
        "periodic-response"
            | "dielectric"
            | "thermal-response"
            | "field-response"
            | "dimensional-scf"
    ) && command != "dimensional-scf"
    {
        if forces {
            return Err("--forces requires periodic, not a density-response command".to_owned());
        }
        if !response_reciprocal_cutoff.is_finite()
            || response_reciprocal_cutoff <= 0.0
            || response_modes.is_empty()
            || response_modes
                .iter()
                .enumerate()
                .any(|(i, x)| response_modes[..i].contains(x))
        {
            return Err(
                "response modes must be nonempty and unique; reciprocal cutoff must be \
                finite and positive"
                    .to_owned(),
            );
        }
        if command == "periodic-response" && response_modes.len() != 1 {
            return Err("periodic-response requires exactly one source mode".to_owned());
        }
    }
    if command != "thermal-response"
        && ["--delta-temperature", "--delta-numbers", "--delta-mu"]
            .iter()
            .any(|f| flags_given.contains(*f))
    {
        return Err("thermodynamic directions require thermal-response".into());
    }
    if command == "thermal-response"
        && (temperature <= 0.0
            || response_q != [0.0; 3]
            || response_omega != 0.0
            || response_damping != 0.0
            || flags_given.contains("--response-modes")
            || !delta_temperature.is_finite()
            || (response_fixed_number && delta_mu != [0.0; 2])
            || (!response_fixed_number && delta_numbers != [0.0; 2]))
    {
        return Err(
            "thermal-response requires positive reference temperature, q=0, zero \
            frequency, and ensemble-compatible directions"
                .into(),
        );
    }
    validate_option_table(
        if matches!(
            command.as_str(),
            "dynamic-polarizability"
                | "dynamic-beta"
                | "dynamic-gamma"
                | "stability"
                | "stab-following"
                | "field-taylor"
                | "field-mixed"
                | "beta"
                | "gamma"
                | "nuclear-taylor"
                | "nuclear-mixed"
                | "molecular-taylor"
                | "molecular-mixed"
                | "static-spectroscopy"
                | "force-constants"
                | "anharmonic"
                | "orbital-energy-gradients"
                | "frontier-gap-gradient"
        ) {
            "polarizability"
        } else if matches!(
            command.as_str(),
            "periodic-response" | "dielectric" | "thermal-response" | "field-response"
        ) {
            "periodic"
        } else {
            &command
        },
        &flags_given,
    )?;
    Ok(Cli {
        command,
        path,
        charge,
        multiplicity,
        basis,
        ecp,
        pseudo_sub,
        trajectory,
        dump_wfn,
        molden,
        xc,
        x_func,
        c_func,
        omega,
        disp,
        xdm_a1,
        xdm_a2,
        nlc,
        log_mode,
        log_file,
        grid_level,
        vv10_grid_level,
        grid_weight_cutoff,
        incore,
        incore_mb,
        density_fit: if ri_j || auxbasis.is_some() {
            Some(auxbasis.unwrap_or_else(|| "def2-universal-jfit".to_owned()))
        } else {
            None
        },
        nstates,
        tda_spin,
        tddft_method,
        stab_follow,
        stabilize_options,
        response_omega,
        input_frequencies,
        electric_process,
        process_frequencies,
        response_damping,
        imaginary_frequency,
        field_options,
        magnetic_gauge,
        magnetic_origin,
        response_tol,
        response_max_iter,
        response_restart,
        response_q,
        field_transfer,
        response_reciprocal_cutoff,
        response_modes,
        response_fixed_number,
        delta_temperature,
        delta_numbers,
        delta_mu,
        response_order,
        electric_order,
        force_coordinates,
        orbital_options,
        anharmonic_config,
        anharmonic_raman,
        vibrational_modes,
        field_direction,
        mixed_field_directions,
        nuclear_direction,
        mixed_nuclear_directions,
        saddle_method,
        natoms_a,
        charge_a,
        charge_b,
        mult_a,
        mult_b,
        // Mixing the guess without an unrestricted reference would be
        // silently ignored, so asking for one implies the other.
        unrestricted: unrestricted || guess_mix.is_some_and(|v| v != 0.0),
        guess_mix,
        guess,
        solvent,
        eps,
        refractive_index,
        alpha,
        beta,
        gamma,
        list_solvents,
        list_functionals,
        conv_tol,
        conv_tol_grad,
        conv_tol_dp,
        max_iter: max_iter_opt,
        no_adiis,
        adiis_until_error,
        no_soscf,
        diis_size,
        level_shift,
        damping,
        field,
        nbo,
        cell,
        periodic_axes,
        basis_grid,
        basis_grid_derivatives,
        basis_grid_cutoff,
        electrostatic_charges,
        electrostatic_widths,
        electrostatic_options,
        gaussian_exponents,
        nuclear_widths,
        density_matrices,
        spin_density_matrices,
        xc_operator_tolerance,
        xc_grid_tolerance,
        gaussian_angular,
        gaussian_source_charge,
        gaussian_source_width,
        gaussian_options,
        k_mesh,
        overlap_cutoff,
        exchange_cutoff,
        grid_shell,
        temperature,
        thermo_temperature,
        pressure,
        symmetry_number,
        forces,
        relax,
        opt_max_steps,
        opt_gtol,
        opt_step_max,
        opt_history,
    })
}

/// Resolve `--basis` (which may be a per-element spec) and then apply
/// `--ecp` on top. Two independent specs, applied in that order, so the
/// core potential is never an accident of which set the orbitals came from.
fn resolve_basis(cli: &Cli) -> Result<dft_rs::basis::BasisSet, dft_rs::error::DftError> {
    let mut set = dft_rs::basis::mixed::load_spec(&cli.basis)?;
    if let Some(spec) = &cli.ecp {
        dft_rs::basis::mixed::apply_ecp_spec(&mut set, spec)?;
    }
    Ok(set)
}

/// Resolve the shared named-site grammar and refuse before calculation until a
/// nonzero site table has an actual Hamiltonian consumer. With no molecule
/// (catalog-list early returns), name resolution still occurs so the option is
/// never silently ignored.
pub fn resolve_pseudo_sub_cli(
    spec: &str,
    molecule: Option<&Molecule>,
) -> Result<(), dft_rs::error::DftError> {
    if let Some(molecule) = molecule {
        let _table = dft_rs::pseudo_sub::build_named_site_table(spec, molecule)?;
    } else {
        for request in dft_rs::pseudo_sub::parse_named_sites(spec)? {
            let _site = request.build()?;
        }
    }
    Err(dft_rs::error::DftError::Unsupported {
        feature: "pseudo-substituent calculation".to_string(),
        missing: "the PseudoSiteTable Hamiltonian/gradient consumer; a fitted name must \
            not become a silent ordinary-hydrogen calculation"
            .to_string(),
    })
}

fn print_stability(
    report: &dft_rs::response::ResponseReport<dft_rs::response::MolecularStability>,
) {
    print_response_metadata(report);
    println!(
        "stable {} classification {} unrestricted {}",
        report.value.stable, report.value.classification, report.value.unrestricted
    );
    for block in &report.value.blocks {
        println!(
            "stability_channel {} {} resolution {:.12e} converged {} matvec_count {}",
            block.name, block.classification, block.resolution, block.converged, block.matvec_count
        );
        for (r, mode) in block.modes.iter().enumerate() {
            println!(
                "stability_mode {} {r} {:.12e} energy_curvature {:.12e} absolute_residual {:.12e}",
                block.name, mode.eigenvalue, mode.energy_curvature, mode.absolute_residual
            );
        }
        println!("dominant {} {:?}", block.name, block.dominant);
    }
}
fn run_property_scf(
    cli: &Cli,
    mol: &dft_rs::system::Molecule,
    mb: &dft_rs::basis::MolBasis,
    opts: &dft_rs::scf::ScfOptions,
    functional: &dft_rs::xc::ResolvedFunctional,
    grid: Option<&dft_rs::grid::MolecularGrid>,
) -> dft_rs::Result<dft_rs::scf::ScfResult> {
    let state = dft_rs::scf::run_scf(mol, mb, opts, functional, grid)?;
    if !cli.stab_follow {
        return Ok(state);
    }
    let report = dft_rs::response::stabilize(
        mol,
        mb,
        functional,
        grid,
        &state,
        opts,
        &cli.stabilize_options,
    )?;
    println!(
        "stabilized {} following_converged {} initial_energy {:.12e} final_energy {:.12e}",
        report.stabilized, report.converged, state.energy, report.final_state.energy
    );
    for d in &report.descents {
        println!(
            "stability_descent {} eigenvalue {:.12e} theta {:.12e} energy_before \
            {:.12e} energy_after {:.12e}",
            d.channel.id(),
            d.eigenvalue,
            d.theta,
            d.energy_before,
            d.energy_after
        );
    }
    for warning in &report.warnings {
        println!("{warning}");
    }
    Ok(report.final_state)
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    // Resolve XDM before every early-returning command, including the two
    // top-level catalog listings. Otherwise a value-dependent request such as
    // `gradient ... --disp xdm --list-solvents` would print the catalog and
    // silently ignore XDM.
    let xdm_damping = resolve_xdm_cli(&cli.command, &cli.disp, cli.xdm_a1, cli.xdm_a2, &cli.nlc)?;
    if (cli.list_functionals || cli.list_solvents) && cli.pseudo_sub.is_some() {
        resolve_pseudo_sub_cli(cli.pseudo_sub.as_deref().unwrap(), None)?;
    }
    if cli.list_functionals {
        use dft_rs::xc::{resolve, FunctionalSpec};
        println!(
            "{} combined functionals (--xc):",
            dft_rs::xc::COMBINED_NAMES.len()
        );
        for &name in dft_rs::xc::COMBINED_NAMES {
            let f = resolve(&FunctionalSpec::Combined(name.into()))?;
            println!("  {name:<14} {}", f.description);
        }
        let double_hybrids: Vec<_> = dft_rs::xc::catalogue::catalogue()?
            .into_iter()
            .filter(|r| r.class == dft_rs::xc::catalogue::FunctionalClass::DoubleHybrid)
            .collect();
        println!(
            "\n{} double hybrids (--xc; PT2 same-spin/opposite-spin coefficients):",
            double_hybrids.len()
        );
        for r in &double_hybrids {
            let (ss, os) = r.pt2.unwrap_or((0.0, 0.0));
            println!("  {:<14} ss {ss:<7} os {os:<7} {}", r.name, r.description);
        }
        println!(
            "\n{} exchange (--x) / {} correlation (--c) components:",
            dft_rs::xc::EXCHANGE_NAMES.len(),
            dft_rs::xc::CORRELATION_NAMES.len()
        );
        println!("  --x  {}", dft_rs::xc::EXCHANGE_NAMES.join(" "));
        println!("  --c  {} none", dft_rs::xc::CORRELATION_NAMES.join(" "));
        println!(
            "\nNames are matched case-insensitively with `-`, `_` and spaces \
             ignored,\nso `cam-b3lyp`, `CAM_B3LYP` and `camb3lyp` are the same entry; \
             aliases\n(`lda`, `pbe1pbe`, `hcth`, …) are accepted but not listed."
        );
        return Ok(());
    }
    if cli.list_solvents {
        let names = dft_rs::pcm::solvent_names();
        println!("{} bundled solvents (name, eps):", names.len());
        for name in names {
            if let Some(s) = dft_rs::pcm::solvent_by_name(name) {
                println!("  {:<40} {:>10.4}", name, s.eps);
            }
        }
        println!(
            "\nShort names: dmso dmf dmac thf dcm ccl4 meoh etoh mecn ether etoac\n\
             Prefix `smd:` for SMD, `cosmo:`/`ief:`/`ssvpe:` for the model.\n\
             Descriptors can also be given directly, with or without a name:\n  \
             --solvent custom --eps 8.93 --refractive-index 1.4242 \\\n    \
             --solvent-alpha 0.10 --solvent-beta 0.05 --solvent-gamma 39.15"
        );
        return Ok(());
    }
    dft_rs::ensure_parallelism();
    // Progress logging (default: console + <input>.dft.log).
    let mode = dft_rs::logger::LogMode::parse(&cli.log_mode)
        .ok_or_else(|| format!("unknown log mode: {}", cli.log_mode))?;
    let default_log = std::path::Path::new(&cli.path).with_extension("dft.log");
    let log_path = cli
        .log_file
        .as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or(default_log);
    if !dft_rs::logger::is_active() {
        dft_rs::logger::init(mode, Some(&log_path))?;
    }
    dft_rs::logger::log(&format!(
        "dft-rs {} | command: {} | input: {} | units=atomic",
        env!("CARGO_PKG_VERSION"),
        cli.command,
        cli.path
    ));
    let molecule = Molecule::from_xyz_file(&cli.path)?
        .with_charge(cli.charge)
        .with_multiplicity(cli.multiplicity);
    if let Some(spec) = &cli.pseudo_sub {
        resolve_pseudo_sub_cli(spec, Some(&molecule))?;
    }
    match cli.command.as_str() {
        "periodic-basis" => {
            use dft_rs::periodic::{
                periodic_basis, Lattice, PeriodicBasisOptions, PeriodicGeometry,
            };
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let rows = cli.cell.expect("validated cell");
            let geometry = PeriodicGeometry::new(
                Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let mb = dft_rs::basis::MolBasis::build(&molecule, &resolve_basis(&cli)?)?;
            let report = periodic_basis(
                &molecule,
                &mb,
                &geometry,
                &PeriodicBasisOptions {
                    k_mesh: cli.k_mesh,
                    basis_cutoff: cli.overlap_cutoff,
                    grid_cutoff: cli.basis_grid_cutoff,
                    grid: cli.basis_grid.then(|| dft_rs::grid::GridOptions {
                        level: cli.grid_level,
                        prune: false,
                        weight_cutoff: cli.grid_weight_cutoff.unwrap_or(0.0),
                        ..Default::default()
                    }),
                    grid_derivatives: cli.basis_grid_derivatives,
                },
            )?;
            println!("# {}", report.model);
            println!("# {}", report.units);
            println!(
                "dimension {} pbc {:?} measure {:.12e} nao {}",
                geometry.dimension(),
                cli.periodic_axes,
                geometry.measure(),
                mb.nao_sph
            );
            for (index, k) in report.k_points.iter().enumerate() {
                println!("k_point {index} {:.12e} {:.12e} {:.12e}", k.x, k.y, k.z);
            }
            for (name, matrices, derivatives) in [
                ("overlap", &report.overlap, &report.overlap_derivatives),
                ("kinetic", &report.kinetic, &report.kinetic_derivatives),
            ] {
                for (k, m) in matrices.iter().enumerate() {
                    for i in 0..mb.nao_sph {
                        for j in 0..mb.nao_sph {
                            println!(
                                "{name} {k} {i} {j} {:.12e} {:.12e}",
                                m.re[(i, j)],
                                m.im[(i, j)]
                            );
                        }
                    }
                    for (atom, axes) in derivatives[k].iter().enumerate() {
                        for (axis, m) in axes.iter().enumerate() {
                            for i in 0..mb.nao_sph {
                                for j in 0..mb.nao_sph {
                                    println!("{name}_derivative {k} {atom} {axis} {i} {j} {:.12e} {:.12e}",m.re[(i,j)],m.im[(i,j)]);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(grid) = &report.grid {
                println!(
                    "grid_points {} grid_overlap_defect {:.12e}",
                    grid.len(),
                    report.grid_overlap_defect.unwrap()
                );
                for (point, &r) in grid.points.iter().enumerate() {
                    println!(
                        "grid {point} {} {:.12e} {:.12e} {:.12e} {:.12e} {:.12e}",
                        grid.atom_of[point],
                        r.x,
                        r.y,
                        r.z,
                        grid.weights[point],
                        grid.raw_weights[point]
                    );
                    if let Some(derivatives) = &report.moving_weight_derivatives {
                        for (atom, g) in derivatives[point].iter().enumerate() {
                            println!(
                                "moving_weight_derivative {point} {atom} {:.12e} {:.12e} {:.12e}",
                                g.x, g.y, g.z
                            );
                        }
                    }
                }
            }
            println!(
                "minimum_overlap_eigenvalue {:.12e} hermiticity_defect {:.12e} \
                translation_defect {:.12e}",
                report.minimum_overlap_eigenvalue,
                report.hermiticity_defect,
                report.translation_defect
            );
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "dimensional-scf" => {
            use dft_rs::periodic::scf_boundary::{run_dimensional_scf, DimensionalScfOptions};
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let boundary = dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                boundary,
            )?;
            let mb = dft_rs::basis::MolBasis::build(&molecule, &resolve_basis(&cli)?)?;
            let spec = match (&cli.xc, &cli.x_func, &cli.c_func) {
                (Some(name), _, _) => dft_rs::xc::FunctionalSpec::Combined(name.clone()),
                (None, Some(x), Some(c)) => dft_rs::xc::FunctionalSpec::Split {
                    x: x.clone(),
                    c: c.clone(),
                },
                _ => dft_rs::xc::FunctionalSpec::Combined("pbe".into()),
            };
            let functional = dft_rs::xc::resolve(&spec)?;
            let mut options = DimensionalScfOptions::for_boundary(boundary);
            options.basis_cutoff = cli.overlap_cutoff;
            options.grid_cutoff = cli.basis_grid_cutoff;
            options.operator_tolerance = cli.xc_operator_tolerance;
            options.grid_tolerance = cli.xc_grid_tolerance;
            options.nuclear_widths = cli.nuclear_widths.clone().unwrap_or_default();
            options.electrostatics = cli.electrostatic_options.clone();
            options.scf.k_mesh = cli.k_mesh;
            options.scf.grid.level = cli.grid_level;
            options.scf.reference = if cli.unrestricted {
                dft_rs::scf::Reference::Uhf
            } else {
                dft_rs::scf::Reference::Auto
            };
            options.scf.occupation = if cli.temperature == 0.0 {
                dft_rs::periodic::occupations::OccupationScheme::Insulator
            } else {
                dft_rs::periodic::occupations::OccupationScheme::FermiDirac {
                    temperature_kelvin: cli.temperature,
                }
            };
            options.scf.guess = match cli.guess.as_deref().unwrap_or("core") {
                "core" => dft_rs::scf::Guess::Core,
                "gwh" => dft_rs::scf::Guess::Gwh,
                "sad" => dft_rs::scf::Guess::Sad,
                _ => return Err("guess must be core, gwh or sad".into()),
            };
            options.scf.max_iter = cli.max_iter.unwrap_or(100);
            options.scf.e_tol = cli.conv_tol.unwrap_or(1e-9);
            options.scf.p_tol = cli.conv_tol_dp.unwrap_or(1e-8);
            options.scf.conv_tol_grad = cli
                .conv_tol_grad
                .unwrap_or_else(|| options.scf.e_tol.sqrt());
            options.scf.exchange_cutoff = cli.exchange_cutoff;
            options.scf.use_adiis = !cli.no_adiis;
            options.scf.adiis_until_error = cli.adiis_until_error.unwrap_or(0.1);
            options.scf.damping = cli.damping.unwrap_or(0.25);
            options.scf.level_shift = cli.level_shift.unwrap_or(0.0);
            options.scf.diis_size = cli.diis_size.unwrap_or(8);
            let r = run_dimensional_scf(&molecule, &mb, &geometry, &functional, &options)?;

            if let Some(transfer_fractional) = cli.field_transfer {
                use dft_rs::dfpt::field_dimensional::{
                    dimensional_field_response, DimensionalFieldResponseOptions,
                };
                let frequency = if cli.imaginary_frequency {
                    dft_rs::response::Frequency::Imaginary {
                        frequency: cli.response_omega,
                    }
                } else {
                    dft_rs::response::Frequency::Retarded {
                        omega: cli.response_omega,
                        damping: cli.response_damping,
                    }
                };
                let request = DimensionalFieldResponseOptions {
                    transfer_fractional,
                    frequency,
                    fixed_number: cli.response_fixed_number,
                    scalar_potential: cli.field_options.scalar_potential,
                    vector_potential: cli.field_options.vector_potential,
                    solver: dft_rs::response::ResponseSolverConfig {
                        tolerance: cli.response_tol,
                        max_iterations: cli.response_max_iter,
                        gmres_restart: cli.response_restart,
                        ..Default::default()
                    },
                };
                let report = dimensional_field_response(
                    &molecule,
                    &mb,
                    &geometry,
                    &functional,
                    &r,
                    &options,
                    &request,
                )?;
                println!(
                    "field_response_model {}\nfield_response_units {}\nfield_response_converged {}",
                    report.model, report.units, report.converged
                );
                println!("field_response_axis_convention {}", report.axis_convention);
                for (i, v) in report.residuals.iter().enumerate() {
                    println!("field_response_residual {i} {v:.15e}");
                }
                for (i, v) in report.iterations.iter().enumerate() {
                    println!("field_response_iterations {i} {v}");
                }
                for (name, values) in [
                    ("vector_potential", report.value.vector_potential),
                    ("electric_field", report.value.electric_field),
                    ("magnetic_field", report.value.magnetic_field),
                    (
                        "paramagnetic_number_current",
                        report.value.paramagnetic_number_current,
                    ),
                    (
                        "diamagnetic_number_current",
                        report.value.diamagnetic_number_current,
                    ),
                    ("number_current", report.value.number_current),
                    ("charge_current", report.value.charge_current),
                ] {
                    for (a, z) in values.into_iter().enumerate() {
                        println!("{name} {a} {:.15e} {:.15e}", z.re, z.im);
                    }
                }
                for (name, z) in [
                    ("scalar_potential", report.value.scalar_potential),
                    ("number_density", report.value.number_density),
                    ("charge_density", report.value.charge_density),
                ] {
                    println!("{name} {:.15e} {:.15e}", z.re, z.im);
                }
                for (name, v) in [
                    ("continuity_defect", report.value.continuity_defect),
                    ("reference_electrons", report.value.reference_electrons),
                    (
                        "occupation_consistency_defect",
                        report.value.occupation_consistency_defect,
                    ),
                    (
                        "hartree_quadrature_defect",
                        report.value.hartree_quadrature_defect,
                    ),
                    (
                        "reference_operator_defect",
                        report.value.reference_operator_defect,
                    ),
                    ("temperature_hartree", report.value.temperature_hartree),
                ] {
                    println!("{name} {v:.15e}");
                }
                for (name, v) in [
                    ("transfer_fractional", report.value.transfer_fractional),
                    (
                        "reduced_transfer_fractional",
                        report.value.reduced_transfer_fractional,
                    ),
                ] {
                    println!("{name} {:.15e} {:.15e} {:.15e}", v[0], v[1], v[2]);
                }
                let z = report.value.frequency.energy()?;
                println!("frequency {:.15e} {:.15e}", z.re, z.im);
                for (name, v) in [
                    ("wave", report.value.wave),
                    ("reduced_transfer", report.value.reduced_transfer),
                ] {
                    println!("{name} {:.15e} {:.15e} {:.15e}", v.x, v.y, v.z);
                }
                for (name, points) in [
                    ("field_k_point", &report.value.k_points),
                    ("field_target_k_point", &report.value.target_k_points),
                ] {
                    for (ik, k) in points.iter().enumerate() {
                        println!("{name} {ik} {:.15e} {:.15e} {:.15e}", k.x, k.y, k.z);
                    }
                }
                for spin in 0..2 {
                    for (ik, m) in report.value.density[spin].iter().enumerate() {
                        for i in 0..m.rows() {
                            for j in 0..m.cols() {
                                println!(
                                    "field_density {spin} {ik} {i} {j} {:.15e} {:.15e}",
                                    m.re[(i, j)],
                                    m.im[(i, j)]
                                );
                            }
                        }
                    }
                }
                for warning in report.warnings {
                    println!("field_response_warning {warning}");
                }
            }
            let s = &r.reference;
            let d = &r.reference_diagnostics;
            println!(
                "model {}\nunits {}\nconverged {}\nscf_converged {}\niterations {}",
                r.model, r.units, r.converged, s.converged, s.iterations
            );
            println!(
                "dimension {} formal_charge {} compensating_charge {:.15e}",
                geometry.dimension(),
                molecule.charge,
                r.compensating_charge
            );
            for (name, value) in [
                ("energy", s.energy),
                ("free_energy", s.free_energy),
                ("mermin_free_energy", s.free_energy),
                ("entropy_term", s.entropy_term),
                ("source_energy", s.e_nuc),
                ("electronic_energy", s.e_elec),
                ("semilocal_energy", s.e_xc),
                ("kinetic_energy", r.kinetic_energy),
                ("electron_nuclear_energy", r.electron_nuclear_energy),
                ("ecp_energy", r.ecp_energy),
                ("external_field_energy", r.external_field_energy),
                ("hartree_energy", r.hartree_energy),
                ("full_exchange_energy", r.full_exchange_energy),
                ("long_range_exchange_energy", r.long_range_exchange_energy),
                ("orbital_residual", d.orbital_residual),
                ("eigen_residual", d.eigen_residual),
                ("orthonormality_defect", d.orthonormality_defect),
                ("projector_defect", d.projector_defect),
                ("particle_number_error", d.particle_number_error),
                ("occupation_consistency", r.occupation_consistency),
                ("component_energy_defect", r.component_energy_defect),
                ("final_fock_change", r.final_fock_change),
                ("grid_number_defect", r.grid_number_defect),
                ("coulomb_operator_defect", r.coulomb_operator_defect),
                ("coulomb_derivative_defect", r.coulomb_derivative_defect),
                ("coulomb_source_defect", r.coulomb_source_defect),
            ] {
                println!("{name} {value:.15e}");
            }
            for (part, energy) in r.semilocal_component_energies.iter().enumerate() {
                println!("semilocal_component_energy {part} {energy:.15e}");
            }
            for spin in 0..2 {
                for ik in 0..s.k_points.len() {
                    for (name, matrices) in [
                        ("density", &s.density),
                        ("fock", &s.fock),
                        ("energy_weighted_density", &s.energy_weighted_density),
                    ] {
                        let m = &matrices[spin][ik];
                        for i in 0..m.rows() {
                            for j in 0..m.cols() {
                                println!(
                                    "{name} {spin} {ik} {i} {j} {:.15e} {:.15e}",
                                    m.re[(i, j)],
                                    m.im[(i, j)]
                                );
                            }
                        }
                    }
                    for (band, occ) in s.occupations[spin][ik].iter().enumerate() {
                        println!("occupation {spin} {ik} {band} {occ:.15e}");
                    }
                }
            }
            for (name, matrices) in [("overlap", &s.overlap), ("h_core", &s.h_core)] {
                for (ik, m) in matrices.iter().enumerate() {
                    for i in 0..m.rows() {
                        for j in 0..m.cols() {
                            println!(
                                "{name} {ik} {i} {j} {:.15e} {:.15e}",
                                m.re[(i, j)],
                                m.im[(i, j)]
                            );
                        }
                    }
                }
            }
            for warning in r.warnings {
                println!("warning {warning}");
            }
        }
        "periodic-field" => {
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let mb = dft_rs::basis::MolBasis::build(&molecule, &resolve_basis(&cli)?)?;
            let mut controls = cli.field_options.clone();
            controls.k_mesh = cli.k_mesh;
            controls.basis_cutoff = cli.overlap_cutoff;
            let report = dft_rs::periodic::field::periodic_field_operators(
                &molecule, &mb, &geometry, &controls,
            )?;
            println!(
                "model {}\nunits {}\naxes {}",
                report.model, report.units, report.axis_convention
            );
            println!(
                "dimension {} formal_charge {} shape {:?}",
                geometry.dimension(),
                molecule.charge,
                [report.k_points.len(), mb.nao_sph, mb.nao_sph]
            );
            for (name, matrices) in [
                ("density_vertex", &report.density_vertex),
                ("first_order", &report.first_order),
                ("second_order", &report.second_order),
                ("contact_vertex", &report.contact_vertex),
            ] {
                for (k, m) in matrices.iter().enumerate() {
                    for i in 0..m.rows() {
                        for j in 0..m.cols() {
                            println!(
                                "{name} {k} {i} {j} {:.12e} {:.12e}",
                                m.re[(i, j)],
                                m.im[(i, j)]
                            );
                        }
                    }
                }
            }
            for a in 0..3 {
                for (k, m) in report.current_vertex[a].iter().enumerate() {
                    for i in 0..m.rows() {
                        for j in 0..m.cols() {
                            println!(
                                "current_vertex {a} {k} {i} {j} {:.12e} {:.12e}",
                                m.re[(i, j)],
                                m.im[(i, j)]
                            );
                        }
                    }
                }
                println!(
                    "electric_field {a} {:.12e} {:.12e}",
                    report.electric_field[a].re, report.electric_field[a].im
                );
                println!(
                    "magnetic_field {a} {:.12e} {:.12e}",
                    report.magnetic_field[a].re, report.magnetic_field[a].im
                );
            }
            for (k, v) in report.k_points.iter().enumerate() {
                let q = report.target_k_points[k];
                let qq = report.second_target_k_points[k];
                println!("k_point {k} {:.12e} {:.12e} {:.12e}", v.x, v.y, v.z);
                println!("target_k_point {k} {:.12e} {:.12e} {:.12e}", q.x, q.y, q.z);
                println!(
                    "second_target_k_point {k} {:.12e} {:.12e} {:.12e}",
                    qq.x, qq.y, qq.z
                );
            }
            println!(
                "transversality_defect {:.12e}",
                report.transversality_defect
            );
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "periodic-xc" => {
            use dft_rs::periodic::xc_boundary::{periodic_xc_operators, PeriodicXcOptions};
            use dft_rs::xc::{resolve, FunctionalSpec};
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let mb = dft_rs::basis::MolBasis::build(&molecule, &resolve_basis(&cli)?)?;
            let nk = geometry.k_points(cli.k_mesh)?.len();
            let density = [
                read_coulomb_density(
                    cli.spin_density_matrices[0]
                        .as_deref()
                        .expect("validated alpha file"),
                    nk,
                    mb.nao_sph,
                )?,
                read_coulomb_density(
                    cli.spin_density_matrices[1]
                        .as_deref()
                        .expect("validated beta file"),
                    nk,
                    mb.nao_sph,
                )?,
            ];
            let spec = match (&cli.xc, &cli.x_func, &cli.c_func) {
                (Some(name), _, _) => FunctionalSpec::Combined(name.clone()),
                (None, Some(x), Some(c)) => FunctionalSpec::Split {
                    x: x.clone(),
                    c: c.clone(),
                },
                _ => FunctionalSpec::Combined("pbe".into()),
            };
            let functional = resolve(&spec)?;
            let report = periodic_xc_operators(
                &molecule,
                &mb,
                &geometry,
                &functional,
                &density,
                &PeriodicXcOptions {
                    vv10: None,
                    k_mesh: cli.k_mesh,
                    basis_cutoff: cli.overlap_cutoff,
                    grid_cutoff: cli.basis_grid_cutoff,
                    grid: dft_rs::grid::GridOptions {
                        level: cli.grid_level,
                        weight_cutoff: cli.grid_weight_cutoff.unwrap_or(1e-14),
                        ..Default::default()
                    },
                    operator_tolerance: cli.xc_operator_tolerance,
                    grid_tolerance: cli.xc_grid_tolerance,
                },
            )?;
            println!("units {}", report.units);
            println!("model {}", report.model);
            println!(
                "functional {} family {:?}",
                functional.description, functional.family
            );
            println!(
                "dimension {} measure {:.15e}",
                geometry.dimension(),
                geometry.measure()
            );
            println!(
                "pbc {} {} {}",
                cli.periodic_axes[0], cli.periodic_axes[1], cli.periodic_axes[2]
            );
            println!("shape 2 {nk} {} {}", mb.nao_sph, mb.nao_sph);
            println!("energy {:.15e}", report.energy);
            for (i, (energy, label)) in report
                .component_energies
                .iter()
                .zip(&report.component_labels)
                .enumerate()
            {
                println!("component_energy {i} {energy:.15e} {label}");
            }
            for spin in 0..2 {
                for k in 0..nk {
                    for i in 0..mb.nao_sph {
                        for j in 0..mb.nao_sph {
                            let m = &report.potential[spin][k];
                            println!(
                                "potential {spin} {k} {i} {j} {:.15e} {:.15e}",
                                m.re[(i, j)],
                                m.im[(i, j)]
                            );
                        }
                    }
                }
            }
            for (k, point) in report.k_points.iter().enumerate() {
                println!(
                    "k_point {k} {:.15e} {:.15e} {:.15e}",
                    point.x, point.y, point.z
                );
            }
            for (name, value) in [
                ("electron_count", report.electron_count),
                ("grid_electron_count", report.grid_electron_count),
                ("expected_electrons", report.expected_electrons),
                ("charge_residual", report.charge_residual),
                ("grid_number_defect", report.grid_number_defect),
                ("hermiticity_defect", report.hermiticity_defect),
                ("component_energy_defect", report.component_energy_defect),
            ] {
                println!("{name} {value:.15e}");
            }
            println!(
                "exact_exchange_coefficients {:.15e} {:.15e} {:.15e}",
                report.hybrid.cx, report.hybrid.clr, report.hybrid.omega
            );
            println!(
                "required_nonlocal {}",
                report.required_nonlocal.unwrap_or("none")
            );
            println!(
                "basis_cutoff {:.15e} grid_cutoff {:.15e} grid_level {} grid_points {}",
                cli.overlap_cutoff, cli.basis_grid_cutoff, cli.grid_level, report.grid_points
            );
            println!(
                "operator_tolerance {:.15e} grid_tolerance {:.15e}",
                cli.xc_operator_tolerance, cli.xc_grid_tolerance
            );
            println!(
                "formal_charge {} converged {}",
                molecule.charge, report.converged
            );
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "periodic-coulomb" => {
            use dft_rs::periodic::coulomb_boundary::{
                periodic_coulomb_operators, PeriodicCoulombOptions,
            };
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let mb = dft_rs::basis::MolBasis::build(&molecule, &resolve_basis(&cli)?)?;
            let k_points = geometry.k_points(cli.k_mesh)?;
            let density = cli
                .density_matrices
                .as_ref()
                .map(|path| read_coulomb_density(path, k_points.len(), mb.nao_sph))
                .transpose()?;
            let report = periodic_coulomb_operators(
                &molecule,
                &mb,
                &geometry,
                density.as_deref(),
                &PeriodicCoulombOptions {
                    k_mesh: cli.k_mesh,
                    basis_cutoff: cli.overlap_cutoff,
                    nuclear_widths: cli.nuclear_widths.clone().unwrap_or_default(),
                    electrostatics: cli.electrostatic_options.clone(),
                },
            )?;
            println!("units {}", report.units);
            println!("model {}", report.model);
            println!(
                "dimension {} measure {:.15e} centre_count {}",
                geometry.dimension(),
                geometry.measure(),
                report.centre_count
            );
            println!(
                "pbc {} {} {}",
                cli.periodic_axes[0], cli.periodic_axes[1], cli.periodic_axes[2]
            );
            println!("shape {} {} {}", k_points.len(), mb.nao_sph, mb.nao_sph);
            println!(
                "formal_charge {} compensating_charge {:.15e}",
                report.formal_charge, report.compensating_charge
            );
            println!(
                "alpha {:.15e} real_cutoff {:.15e} reciprocal_cutoff {:.15e} basis_cutoff {:.15e}",
                report.params.alpha, report.params.rcut, report.params.gcut, cli.overlap_cutoff
            );
            match &report.compensation {
                dft_rs::periodic::electrostatics::ChargeCompensation::RequireNeutral => {
                    println!("compensation neutral")
                }
                dft_rs::periodic::electrostatics::ChargeCompensation::UniformBackground => {
                    println!("compensation uniform")
                }
                dft_rs::periodic::electrostatics::ChargeCompensation::GaussianCountercharge {
                    center,
                    width,
                } => {
                    println!("compensation gaussian");
                    println!(
                        "countercharge_center {:.15e} {:.15e} {:.15e}",
                        center.x, center.y, center.z
                    );
                    println!("countercharge_width {:.15e}", width);
                }
            }
            for (index, k) in k_points.iter().enumerate() {
                println!("k_point {index} {:.15e} {:.15e} {:.15e}", k.x, k.y, k.z);
            }
            for (name, matrices, components, derivatives) in [
                (
                    "nuclear",
                    Some(&report.nuclear),
                    Some(&report.nuclear_components),
                    Some(&report.nuclear_derivatives),
                ),
                (
                    "hartree",
                    report.hartree.as_ref(),
                    report.hartree_components.as_ref(),
                    report.hartree_derivatives.as_ref(),
                ),
            ] {
                if let (Some(matrices), Some(components), Some(derivatives)) =
                    (matrices, components, derivatives)
                {
                    for (k, m) in matrices.iter().enumerate() {
                        for i in 0..mb.nao_sph {
                            for j in 0..mb.nao_sph {
                                println!(
                                    "{name} {k} {i} {j} {:.15e} {:.15e}",
                                    m.re[(i, j)],
                                    m.im[(i, j)]
                                );
                                for (component, part) in components[k].iter().enumerate() {
                                    println!(
                                        "{name}_component {k} {component} {i} {j} {:.15e} {:.15e}",
                                        part.re[(i, j)],
                                        part.im[(i, j)]
                                    );
                                }
                            }
                        }
                    }
                    for (k, centres) in derivatives.iter().enumerate() {
                        for (centre, axes) in centres.iter().enumerate() {
                            for (axis, m) in axes.iter().enumerate() {
                                for i in 0..mb.nao_sph {
                                    for j in 0..mb.nao_sph {
                                        println!("{name}_derivative {k} {centre} {axis} {i} {j} {:.15e} {:.15e}",m.re[(i,j)],m.im[(i,j)]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let c = &report.source_energy_components;
            println!("source_energy {:.15e}", report.source_energy);
            println!(
                "source_energy_components real_space {:.15e} reciprocal_nonzero {:.15e} \
                reciprocal_zero {:.15e} point_self {:.15e}",
                c.real_space, c.reciprocal_nonzero, c.reciprocal_zero, c.point_self
            );
            for (name, components) in [
                (
                    "electron_nuclear_components",
                    report.electron_nuclear_components,
                ),
                (
                    "hartree_energy_components",
                    report.hartree_energy_components,
                ),
            ] {
                if let Some(c) = components {
                    println!("{name} {:.15e} {:.15e} {:.15e}", c[0], c[1], c[2]);
                } else {
                    println!("{name} none");
                }
            }
            for (name, value) in [
                ("total_coulomb_energy", report.total_coulomb_energy),
                ("electron_count", report.electron_count),
                ("charge_residual", report.charge_residual),
            ] {
                if let Some(value) = value {
                    println!("{name} {value:.15e}");
                } else {
                    println!("{name} none");
                }
            }
            println!("expected_electrons {:.15e}", report.expected_electrons);
            for (centre, g) in report.source_gradients.iter().enumerate() {
                println!(
                    "source_gradient {centre} {:.15e} {:.15e} {:.15e}",
                    g.x, g.y, g.z
                );
            }
            if let Some(gradients) = &report.energy_gradients {
                for (centre, g) in gradients.iter().enumerate() {
                    println!(
                        "energy_gradient {centre} {:.15e} {:.15e} {:.15e}",
                        g.x, g.y, g.z
                    );
                }
            }
            println!(
                "converged {} quadrature_order {}",
                report.converged, report.quadrature_order
            );
            for (name, value) in [
                (
                    "quadrature_operator_defect",
                    report.quadrature_operator_defect,
                ),
                (
                    "quadrature_derivative_defect",
                    report.quadrature_derivative_defect,
                ),
                ("quadrature_source_defect", report.quadrature_source_defect),
                ("hermiticity_defect", report.hermiticity_defect),
                ("translation_defect", report.translation_defect),
                (
                    "real_space_density_imaginary_max",
                    report.real_space_density_imaginary_max,
                ),
                (
                    "hartree_potential_imaginary_defect",
                    report.hartree_potential_imaginary_defect,
                ),
            ] {
                println!("{name} {value:.15e}");
            }
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "periodic-gaussian-integral" => {
            use dft_rs::periodic::gaussian_coulomb::{
                periodic_gaussian_integral, CartesianGaussian, GaussianNucleus,
            };
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let nuclear = molecule.len() == 3;
            let count = if nuclear { 2 } else { 4 };
            let exponents = cli
                .gaussian_exponents
                .as_ref()
                .expect("validated exponents");
            let angular = cli
                .gaussian_angular
                .as_ref()
                .expect("validated angular powers");
            if !matches!(molecule.len(), 3 | 4)
                || exponents.len() != count
                || angular.len() != count
            {
                return Err(
                    "provide three XYZ centres with two exponents/angular triples for \
                    nuclear attraction, or four centres with four for electron repulsion"
                        .into(),
                );
            }
            if !nuclear
                && (cli.gaussian_source_charge.is_some() || cli.gaussian_source_width.is_some())
            {
                return Err("source controls apply only to nuclear attraction".into());
            }
            let gaussians: Vec<_> = (0..count)
                .map(|i| CartesianGaussian {
                    center: molecule.atoms[i].position,
                    exponent: exponents[i],
                    angular: angular[i],
                })
                .collect();
            let source = nuclear.then(|| GaussianNucleus {
                center: molecule.atoms[2].position,
                charge: cli.gaussian_source_charge.unwrap_or(1.0),
                width: cli.gaussian_source_width.unwrap_or(0.0),
            });
            let report = periodic_gaussian_integral(
                &geometry,
                &gaussians,
                source.as_ref(),
                &cli.gaussian_options,
            )?;
            println!("value {:.15e} kind {}", report.value, report.kind);
            println!("units {}", report.units);
            println!("model {}", report.model);
            println!(
                "dimension {} measure {:.15e} pbc {} {} {}",
                geometry.dimension(),
                geometry.measure(),
                cli.periodic_axes[0],
                cli.periodic_axes[1],
                cli.periodic_axes[2]
            );
            println!(
                "components real_space {:.15e} reciprocal_nonzero {:.15e} \
                reciprocal_zero {:.15e} point_self {:.15e}",
                report.components.real_space,
                report.components.reciprocal_nonzero,
                report.components.reciprocal_zero,
                report.components.point_self
            );
            println!(
                "alpha {:.15e} real_cutoff {:.15e} reciprocal_cutoff {:.15e}",
                report.params.alpha, report.params.rcut, report.params.gcut
            );
            println!("shape {} 3", report.center_gradients.len());
            for (center, g) in report.center_gradients.iter().enumerate() {
                println!(
                    "center_gradient {center} {:.15e} {:.15e} {:.15e}",
                    g.x, g.y, g.z
                );
            }
            println!(
                "converged {} quadrature_order {} quadrature_value_defect {:.15e} \
                quadrature_gradient_defect {:.15e} translation_defect {:.15e}",
                report.converged,
                report.quadrature_order,
                report.quadrature_value_defect,
                report.quadrature_gradient_defect,
                report.translation_defect
            );
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "periodic-electrostatics" => {
            use dft_rs::periodic::electrostatics::{periodic_electrostatics, ElectrostaticSource};
            let rows = cli.cell.expect("validated cell");
            let vector = |v: [f64; 3]| Vec3::new(v[0], v[1], v[2]);
            let geometry = dft_rs::periodic::PeriodicGeometry::new(
                dft_rs::periodic::Lattice::new(vector(rows[0]), vector(rows[1]), vector(rows[2])),
                dft_rs::periodic::features::PeriodicBoundary::new(cli.periodic_axes),
            )?;
            let points = geometry.k_points(cli.k_mesh)?;
            let charges = cli
                .electrostatic_charges
                .as_ref()
                .expect("validated charges");
            let widths = cli
                .electrostatic_widths
                .clone()
                .unwrap_or_else(|| vec![0.0; charges.len()]);
            if charges.len() != molecule.len() || widths.len() != charges.len() {
                return Err("prescribed charges and widths require one value per XYZ atom".into());
            }
            let sources: Vec<_> = molecule
                .atoms
                .iter()
                .zip(charges)
                .zip(widths)
                .map(|((atom, &charge), width)| ElectrostaticSource {
                    position: atom.position,
                    charge,
                    width,
                })
                .collect();
            let report = periodic_electrostatics(&geometry, &sources, &cli.electrostatic_options)?;
            println!("# {}", report.model);
            println!("# {}; {}", report.units, report.potential_convention);
            println!(
                "dimension {} pbc {:?} measure {:.12e}",
                geometry.dimension(),
                cli.periodic_axes,
                geometry.measure()
            );
            println!("energy {:.12e}", report.energy);
            println!(
                "components real_space {:.12e} reciprocal_nonzero {:.12e} \
                reciprocal_zero {:.12e} point_self {:.12e}",
                report.components.real_space,
                report.components.reciprocal_nonzero,
                report.components.reciprocal_zero,
                report.components.point_self
            );
            println!(
                "supplied_charge {:.12e} compensating_charge {:.12e}",
                report.supplied_charge, report.compensating_charge
            );
            println!(
                "alpha {:.12e} real_cutoff {:.12e} reciprocal_cutoff {:.12e}",
                report.params.alpha, report.params.rcut, report.params.gcut
            );
            for (i, f) in report.forces.iter().enumerate() {
                println!("force {i} {:.12e} {:.12e} {:.12e}", f.x, f.y, f.z);
            }
            for (i, p) in report.site_potentials.iter().enumerate() {
                println!("site_potential {i} {p:.12e}");
            }
            if let Some(f) = report.compensation_force {
                println!("compensation_force {:.12e} {:.12e} {:.12e}", f.x, f.y, f.z);
            }
            if let Some(p) = report.compensation_potential {
                println!("compensation_potential {p:.12e}");
            }
            for (i, g) in geometry.reciprocal().iter().enumerate() {
                println!(
                    "reciprocal_vector {i} {:.12e} {:.12e} {:.12e}",
                    g.x, g.y, g.z
                );
            }
            for (i, k) in points.iter().enumerate() {
                println!("k_point {i} {:.12e} {:.12e} {:.12e}", k.x, k.y, k.z);
            }
            println!(
                "converged {} quadrature_order {} quadrature_energy_defect {:.12e} \
                quadrature_force_defect {:.12e} force_sum_defect {:.12e}",
                report.converged,
                report.quadrature_order,
                report.quadrature_energy_defect,
                report.quadrature_force_defect,
                report.force_sum_defect
            );
            for warning in report.warnings {
                println!("warning {warning}");
            }
        }
        "info" => {
            let (n_alpha, n_beta) = molecule.n_alpha_beta()?;
            println!("Atoms:               {}", molecule.len());
            println!("Charge:              {}", molecule.charge);
            println!("Multiplicity:        {}", molecule.multiplicity);
            println!("Electrons:           {}", molecule.n_electrons()?);
            println!("  alpha / beta:      {n_alpha} / {n_beta}");
            println!(
                "Nuclear repulsion:   {:.12} Hartree",
                molecule.nuclear_repulsion()
            );
            println!("# atom  element      x (Bohr)          y (Bohr)          z (Bohr)");
            for (i, atom) in molecule.atoms.iter().enumerate() {
                println!(
                    "{:>5} {:>5} {:>18.10} {:>18.10} {:>18.10}",
                    i + 1,
                    dft_rs::z_to_symbol(atom.z).unwrap_or("X"),
                    atom.position.x,
                    atom.position.y,
                    atom.position.z
                );
            }
        }
        "periodic" | "periodic-response" | "dielectric" | "thermal-response" | "field-response" => {
            use dft_rs::grid::GridOptions;
            use dft_rs::periodic::features::{PeriodicBoundary, PeriodicCapabilities};
            use dft_rs::periodic::forces::analytic_forces;
            use dft_rs::periodic::occupations::OccupationScheme;
            use dft_rs::periodic::scf::{run_periodic_scf, PeriodicScfOptions};
            use dft_rs::periodic::{Cell, Lattice};
            use dft_rs::xc::FunctionalSpec;

            let rows = cli.cell.expect("--cell is required and checked at parse");
            let lattice = Lattice::new(
                dft_rs::math::Vec3::new(rows[0][0], rows[0][1], rows[0][2]),
                dft_rs::math::Vec3::new(rows[1][0], rows[1][1], rows[1][2]),
                dft_rs::math::Vec3::new(rows[2][0], rows[2][1], rows[2][2]),
            );
            let boundary = PeriodicBoundary::new(cli.periodic_axes);
            boundary.validate_k_mesh(cli.k_mesh)?;
            // Dimensional source kernels are available; every electronic
            // operator must use them before a partial-PBC SCF can run.
            PeriodicCapabilities::current().validate_boundary(boundary)?;
            if !(lattice.volume().is_finite() && lattice.volume().abs() > 1e-12) {
                return Err("--cell must define a finite non-zero volume".into());
            }

            let spec = match (&cli.xc, &cli.x_func, &cli.c_func) {
                (Some(name), _, _) => FunctionalSpec::Combined(name.clone()),
                (None, Some(x), Some(c)) => FunctionalSpec::Split {
                    x: x.clone(),
                    c: c.clone(),
                },
                _ if cli.command == "periodic-response"
                    || cli.command == "dielectric"
                    || cli.command == "thermal-response" =>
                {
                    FunctionalSpec::Combined("pbe".to_owned())
                }
                _ => FunctionalSpec::Combined("hf".to_owned()),
            };
            let mut functional = dft_rs::xc::resolve(&spec)?;
            if let Some(w) = cli.omega {
                functional = functional.with_omega(w)?;
            }
            if let Some(term) = functional.required_nonlocal_term() {
                return Err(format!(
                    "periodic SCF for this functional is unavailable because its defining \
                     {term} nonlocal term is not yet periodic; returning only the semilocal \
                     part would not be the named functional"
                )
                .into());
            }
            let set = resolve_basis(&cli)?;
            let guess = match cli.guess.as_deref().map(str::to_ascii_lowercase) {
                None => dft_rs::scf::guess::Guess::Core,
                Some(name) if name == "sad" => dft_rs::scf::guess::Guess::Sad,
                Some(name) if name == "gwh" => dft_rs::scf::guess::Guess::Gwh,
                Some(name) if name == "core" => dft_rs::scf::guess::Guess::Core,
                Some(other) => {
                    return Err(format!("unknown --guess: {other} (sad|gwh|core)").into())
                }
            };
            let options = PeriodicScfOptions {
                k_mesh: cli.k_mesh,
                boundary,
                occupation: if cli.temperature == 0.0 {
                    OccupationScheme::Insulator
                } else {
                    OccupationScheme::FermiDirac {
                        temperature_kelvin: cli.temperature,
                    }
                },
                guess,
                use_adiis: !cli.no_adiis,
                adiis_until_error: cli.adiis_until_error.unwrap_or(0.1),
                max_iter: cli.max_iter.unwrap_or(100),
                e_tol: cli.conv_tol.unwrap_or(1e-9),
                p_tol: cli.conv_tol_dp.unwrap_or(1e-8),
                conv_tol_grad: cli
                    .conv_tol_grad
                    .unwrap_or_else(|| cli.conv_tol.unwrap_or(1e-9).sqrt()),
                level_shift: cli.level_shift.unwrap_or(0.0),
                damping: cli.damping.unwrap_or(0.25),
                diis_size: cli.diis_size.unwrap_or(8),
                exchange_cutoff: cli.exchange_cutoff,
                grid: GridOptions {
                    level: cli.grid_level,
                    weight_cutoff: cli.grid_weight_cutoff.unwrap_or(1e-14),
                    ..GridOptions::default()
                },
                grid_shell: cli.grid_shell,
                ..PeriodicScfOptions::default()
            };
            let cell = Cell::new(molecule.clone(), lattice, &set, cli.overlap_cutoff)?;
            if cli.relax {
                let optimized = dft_rs::periodic::optimizer::optimize_fixed_cell(
                    &molecule,
                    cell.lattice(),
                    &set,
                    cli.overlap_cutoff,
                    &functional,
                    &options,
                    &dft_rs::periodic::optimizer::PeriodicOptOptions {
                        max_steps: cli.opt_max_steps,
                        gtol: cli.opt_gtol,
                        step_max: cli.opt_step_max,
                        history: cli.opt_history,
                        trajectory: cli.trajectory.as_ref().map(std::path::PathBuf::from),
                        dispersion: None,
                    },
                )?;
                if !optimized.converged {
                    return Err(format!(
                        "fixed-cell periodic optimization did not converge after {} accepted \
                            steps (final objective {:.12} Hartree)",
                        optimized.iterations, optimized.objective
                    )
                    .into());
                }
                let input = std::path::Path::new(&cli.path);
                let stem = input
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("periodic-optimized");
                let output = input.with_file_name(format!("{stem}.pbcopt.xyz"));
                std::fs::write(
                    &output,
                    optimized
                        .molecule
                        .to_xyz_string("dft-rs fixed-cell periodic optimized geometry; Angstrom"),
                )?;
                println!(
                    "Converged: {} after {} fixed-cell steps",
                    optimized.converged, optimized.iterations
                );
                println!("Final objective: {:.12} Hartree", optimized.objective);
                println!("Optimized geometry: {}", output.display());
                println!(
                    "# lattice vectors were held fixed; cell relaxation needs electronic stress"
                );
                if cli.forces {
                    let final_forces =
                        analytic_forces(&optimized.cell, &functional, &optimized.scf, &options)?;
                    println!("# final analytic forces (Hartree/Bohr)");
                    for (i, (atom, force)) in optimized
                        .molecule
                        .atoms
                        .iter()
                        .zip(&final_forces.total)
                        .enumerate()
                    {
                        println!(
                            "{:>5} {:>5} {:>18.10} {:>18.10} {:>18.10}",
                            i + 1,
                            dft_rs::z_to_symbol(atom.z).unwrap_or("X"),
                            force.x,
                            force.y,
                            force.z
                        );
                    }
                }
                if let Some(path) = &cli.trajectory {
                    println!(
                        "Trajectory: {} ({} frames)",
                        path,
                        optimized.trajectory.len()
                    );
                }
                return Ok(());
            }
            let result = run_periodic_scf(&cell, &functional, &options)?;

            if cli.command != "periodic" {
                use dft_rs::dfpt::density::{PeriodicDensitySolver, PeriodicResponseOptions};
                use dft_rs::response::{Frequency, ResponseSolverConfig};
                let request = PeriodicResponseOptions {
                    q: dft_rs::math::Vec3::new(
                        cli.response_q[0],
                        cli.response_q[1],
                        cli.response_q[2],
                    ),
                    frequency: if cli.imaginary_frequency {
                        Frequency::Imaginary {
                            frequency: cli.response_omega,
                        }
                    } else {
                        Frequency::Retarded {
                            omega: cli.response_omega,
                            damping: cli.response_damping,
                        }
                    },
                    fixed_number: cli.response_fixed_number,
                    reciprocal_cutoff: cli.response_reciprocal_cutoff,
                    solver: ResponseSolverConfig {
                        tolerance: cli.response_tol,
                        max_iterations: cli.response_max_iter,
                        gmres_restart: cli.response_restart,
                        ..Default::default()
                    },
                };
                if cli.command == "field-response" {
                    use dft_rs::dfpt::field::{
                        periodic_field_response, PeriodicFieldResponseOptions,
                    };
                    let report = periodic_field_response(
                        &cell,
                        &functional,
                        &result,
                        &options,
                        &PeriodicFieldResponseOptions {
                            response: request,
                            scalar_potential: cli.field_options.scalar_potential,
                            vector_potential: cli.field_options.vector_potential,
                        },
                    )?;
                    print_response_metadata(&report);
                    for (key, values) in [
                        ("vector_potential", report.value.vector_potential),
                        ("electric_field", report.value.electric_field),
                        ("magnetic_field", report.value.magnetic_field),
                        (
                            "paramagnetic_number_current",
                            report.value.paramagnetic_number_current,
                        ),
                        (
                            "diamagnetic_number_current",
                            report.value.diamagnetic_number_current,
                        ),
                        ("number_current", report.value.number_current),
                        ("charge_current", report.value.charge_current),
                    ] {
                        for (axis, z) in values.into_iter().enumerate() {
                            println!("{key} {axis} {:.14e} {:.14e}", z.re, z.im);
                        }
                    }
                    for (key, z) in [
                        ("scalar_potential", report.value.scalar_potential),
                        ("number_density", report.value.number_density),
                        ("charge_density", report.value.charge_density),
                    ] {
                        println!("{key} {:.14e} {:.14e}", z.re, z.im);
                    }
                    println!("continuity_defect {:.14e}", report.value.continuity_defect);
                    println!(
                        "reference_electrons {:.14e}",
                        report.value.reference_electrons
                    );
                    println!(
                        "occupation_consistency_defect {:.14e}",
                        report.value.density_response.occupation_consistency_defect
                    );
                    println!("reference_energy {:.14e}", result.energy);
                    println!("reference_mermin_free_energy {:.14e}", result.free_energy);
                    println!("reference_free_energy {:.14e}", result.free_energy);
                    let q = report.value.density_response.q;
                    println!("q {:.14e} {:.14e} {:.14e}", q.x, q.y, q.z);
                    let z = report.value.density_response.frequency.energy()?;
                    println!("frequency {:.14e} {:.14e}", z.re, z.im);
                    for (key, points) in [
                        ("k_point", &report.value.k_points),
                        ("target_k_point", &report.value.target_k_points),
                        (
                            "reciprocal_wave",
                            &report.value.density_response.reciprocal_waves,
                        ),
                    ] {
                        for (k, p) in points.iter().enumerate() {
                            println!("{key} {k} {:.14e} {:.14e} {:.14e}", p.x, p.y, p.z);
                        }
                    }
                    print_ao_density_rows(&report.value.density_response.density);
                    for (m, z) in report
                        .value
                        .density_response
                        .fourier_density
                        .iter()
                        .enumerate()
                    {
                        println!("fourier_density {m} {:.14e} {:.14e}", z.re, z.im);
                    }
                    return Ok(());
                }
                let solver =
                    PeriodicDensitySolver::new(&cell, &functional, &result, &options, &request)?;
                println!(
                    "q (bohr^-1): {:.12} {:.12} {:.12}",
                    request.q.x, request.q.y, request.q.z
                );
                let z = request.frequency.energy()?;
                println!("Frequency (Hartree): {:.12} {:.12}", z.re, z.im);
                if cli.command == "thermal-response" {
                    use dft_rs::dfpt::density::ThermalSource;
                    use dft_rs::dfpt::thermodynamic::Ensemble;
                    let source = ThermalSource {
                        delta_temperature_hartree: cli.delta_temperature
                            * dft_rs::periodic::occupations::BOLTZMANN_HARTREE_PER_K,
                        ensembles: std::array::from_fn(|s| {
                            if request.fixed_number {
                                Ensemble::FixedNumber {
                                    delta_number: cli.delta_numbers[s],
                                }
                            } else {
                                Ensemble::FixedChemicalPotential {
                                    delta_mu: cli.delta_mu[s],
                                }
                            }
                        }),
                    };
                    let report = solver.solve_thermal(source)?;
                    print_response_metadata(&report);
                    print_ao_density_rows(&report.value.density);
                    println!(
                        "Temperature direction (Kelvin): {:.14e}",
                        cli.delta_temperature
                    );
                    println!("Delta entropy (k_B): {:.14e}", report.value.delta_entropy);
                    println!(
                        "Delta internal energy (Hartree): {:.14e}",
                        report.value.delta_internal_energy
                    );
                    println!(
                        "Delta free energy (Hartree): {:.14e}",
                        report.value.delta_free_energy
                    );
                    println!(
                        "Occupation equation defect: {:.14e}",
                        report.value.occupation_equation_defect
                    );
                    println!(
                        "Occupation consistency defect: {:.14e}",
                        report.value.occupation_consistency_defect
                    );
                    println!("# spin delta_number delta_mu_Hartree number_constraint_defect");
                    for s in 0..2 {
                        let mu = report.value.delta_chemical_potentials[s]
                            .map_or("undefined".to_owned(), |v| format!("{v:.14e}"));
                        println!(
                            "{s} {:.14e} {mu} {:.14e}",
                            report.value.delta_numbers[s],
                            report.value.number_constraint_defects[s]
                        );
                    }
                    println!(
                        "# G_x G_y G_z Fourier_density_variation_real \
                        Fourier_density_variation_imag (bohr^-3)"
                    );
                    for (g, rho) in report
                        .value
                        .reciprocal_waves
                        .iter()
                        .zip(&report.value.fourier_density)
                    {
                        println!(
                            "{:.12} {:.12} {:.12} {:.14e} {:.14e}",
                            g.x, g.y, g.z, rho.re, rho.im
                        );
                    }
                } else if cli.command == "dielectric" {
                    let report = solver.dielectric(&cli.response_modes)?;
                    print_response_metadata(&report);
                    println!(
                        "# row column inverse_epsilon_real inverse_epsilon_imag \
                        screened_chi_real screened_chi_imag"
                    );
                    for i in 0..cli.response_modes.len() {
                        for j in 0..cli.response_modes.len() {
                            let eps = report.value.inverse_epsilon[i][j];
                            let chi = report.value.screened_chi[i][j];
                            println!(
                                "{i} {j} {:.14e} {:.14e} {:.14e} {:.14e}",
                                eps.re, eps.im, chi.re, chi.im
                            );
                        }
                    }
                    println!("# reciprocal waves (bohr^-1)");
                    for g in &report.value.reciprocal_waves {
                        println!("{:.12} {:.12} {:.12}", g.x, g.y, g.z);
                    }
                } else {
                    let report = solver.solve(&solver.plane_wave_source(cli.response_modes[0])?)?;
                    print_response_metadata(&report);
                    print_ao_density_rows(&report.value.density);
                    println!(
                        "Occupation consistency defect: {:.3e}",
                        report.value.occupation_consistency_defect
                    );
                    println!(
                        "# G+q_x G+q_y G+q_z Fourier_density_real Fourier_density_imag \
                        (bohr^-3/hartree)"
                    );
                    for (g, rho) in report
                        .value
                        .reciprocal_waves
                        .iter()
                        .zip(&report.value.fourier_density)
                    {
                        println!(
                            "{:.12} {:.12} {:.12} {:.14e} {:.14e}",
                            g.x, g.y, g.z, rho.re, rho.im
                        );
                    }
                }
                return Ok(());
            }

            println!("Cell volume:         {:.6} Bohr^3", cell.lattice().volume());
            println!(
                "k mesh:              {}x{}x{}  ({} points)",
                cli.k_mesh[0],
                cli.k_mesh[1],
                cli.k_mesh[2],
                result.k_points.len()
            );
            println!("Converged:           {}", result.converged);
            println!("Iterations:          {}", result.iterations);
            println!("E_nuc (Ewald):       {:.12} Hartree", result.e_nuc);
            println!("E_elec:              {:.12} Hartree", result.e_elec);
            println!("E_xc:                {:.12} Hartree", result.e_xc);
            println!("Total energy:        {:.12} Hartree /cell", result.energy);
            if cli.temperature > 0.0 {
                println!("Entropy term T*S:    {:.12} Hartree", result.entropy_term);
                println!(
                    "Mermin free energy:  {:.12} Hartree /cell",
                    result.free_energy
                );
                for (spin, mu) in result.chemical_potential.iter().enumerate() {
                    if let Some(mu) = mu {
                        println!("Chemical potential ({spin}): {mu:.12} Hartree");
                    }
                }
            }
            if let Some(count) = result.grid_electron_count {
                println!("Grid electron count: {count:.8}");
            }
            if cli.forces {
                let f = analytic_forces(&cell, &functional, &result, &options)?;
                // At finite temperature the analytic force differentiates the
                // Mermin free energy, not the internal energy: the entropy is
                // not stationary with respect to the occupations, so the two
                // are genuinely different derivatives.
                let differentiated = if cli.temperature > 0.0 {
                    "Mermin free energy"
                } else {
                    "total energy"
                };
                println!("# analytic forces -d({differentiated})/dR (Hartree/Bohr)");
                println!("# atom  element             Fx                Fy                Fz");
                let mut sum = dft_rs::math::Vec3::zero();
                for (i, (atom, force)) in molecule.atoms.iter().zip(&f.total).enumerate() {
                    println!(
                        "{:>5} {:>5} {:>18.10} {:>18.10} {:>18.10}",
                        i + 1,
                        dft_rs::z_to_symbol(atom.z).unwrap_or("X"),
                        force.x,
                        force.y,
                        force.z
                    );
                    sum += *force;
                }
                // Newton's third law inside the cell: an exact lattice sum
                // has no net force on the unit cell, so this line is the
                // cheapest available check that no term was dropped.
                println!(
                    "sum of forces:       {:.3e} {:.3e} {:.3e} (exactly zero by translation)",
                    sum.x, sum.y, sum.z
                );
            }
        }
        "sapt" => {
            use dft_rs::grid::GridOptions;
            use dft_rs::scf::ScfOptions;
            use dft_rs::system::Molecule;
            use dft_rs::xc::FunctionalSpec;
            if cli.natoms_a == 0 || cli.natoms_a >= molecule.len() {
                return Err(
                    "sapt needs --natoms-a N (the first N atoms of the XYZ form monomer A)".into(),
                );
            }
            let set = resolve_basis(&cli)?;
            let spec = match (&cli.xc, &cli.x_func, &cli.c_func) {
                (Some(name), _, _) => FunctionalSpec::Combined(name.clone()),
                (None, Some(x), Some(c)) => FunctionalSpec::Split {
                    x: x.clone(),
                    c: c.clone(),
                },
                _ => FunctionalSpec::Combined("hf".to_owned()),
            };
            let mut functional = dft_rs::xc::resolve(&spec)?;
            if let Some(w) = cli.omega {
                functional = functional.with_omega(w)?;
            }
            let mut opts = ScfOptions::default();
            if let Some(mb) = cli.incore_mb {
                opts.incore_mb = mb;
            }
            if cli.incore {
                opts.eri_mode = dft_rs::scf::EriMode::InCore {
                    limit_mb: opts.incore_mb,
                };
            }
            opts.density_fit = cli.density_fit.clone();
            // SCF convergence. PySCF's rule: conv_tol_grad defaults to
            // sqrt(conv_tol), so overriding the energy threshold alone
            // moves the gradient threshold with it -- which is what a user
            // asking for tighter convergence means.
            if let Some(v) = cli.conv_tol {
                opts.e_tol = v;
                opts.conv_tol_grad = v.sqrt();
            }
            if let Some(v) = cli.conv_tol_grad {
                opts.conv_tol_grad = v;
            }
            if let Some(v) = cli.conv_tol_dp {
                opts.p_tol = v;
            }
            if let Some(v) = cli.max_iter {
                opts.max_iter = v;
            }
            if let Some(v) = cli.diis_size {
                opts.diis_size = v;
            }
            if let Some(v) = cli.level_shift {
                opts.level_shift = v;
            }
            if let Some(v) = cli.damping {
                opts.damping = v;
            }
            if cli.no_adiis {
                opts.use_adiis = false;
            }
            if let Some(v) = cli.adiis_until_error {
                opts.adiis_until_error = v;
            }
            if cli.no_soscf {
                opts.use_soscf = false;
            }
            if cli.unrestricted {
                opts.reference = dft_rs::Reference::Uhf;
            }
            if let Some(v) = cli.guess_mix {
                opts.guess_mix = v;
            }
            if let Some(g) = &cli.guess {
                opts.guess = match g.to_ascii_lowercase().as_str() {
                    "sad" => dft_rs::scf::guess::Guess::Sad,
                    "gwh" => dft_rs::scf::guess::Guess::Gwh,
                    "core" => dft_rs::scf::guess::Guess::Core,
                    other => return Err(format!("unknown --guess: {other} (sad|gwh|core)").into()),
                };
            }
            let mol_a = Molecule::new(molecule.atoms[..cli.natoms_a].to_vec())
                .with_charge(cli.charge_a)
                .with_multiplicity(cli.mult_a);
            let mol_b = Molecule::new(molecule.atoms[cli.natoms_a..].to_vec())
                .with_charge(cli.charge_b)
                .with_multiplicity(cli.mult_b);
            let gopts = GridOptions {
                level: cli.grid_level,
                weight_cutoff: cli.grid_weight_cutoff.unwrap_or(1e-14),
                ..GridOptions::default()
            };
            let res = dft_rs::sapt::sapt_interaction(
                &mol_a,
                &mol_b,
                &set,
                &functional,
                &opts,
                Some(&gopts),
            )?;
            const KCAL: f64 = 627.5094740631;
            let shells = if cli.mult_a > 1 || cli.mult_b > 1 {
                "open-shell monomers (spin-resolved)"
            } else {
                "closed-shell monomers"
            };
            println!("SAPT interaction energy decomposition ({shells})");
            println!("  E(monomer A)      = {:>18.10} Hartree", res.e_monomer_a);
            println!("  E(monomer B)      = {:>18.10} Hartree", res.e_monomer_b);
            println!(
                "  E_elst(1)         = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_elst,
                res.e_elst * KCAL
            );
            println!(
                "  E_exch(1)  [HL]   = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_exch,
                res.e_exch * KCAL
            );
            println!(
                "  E_ind(2)   A<-B   = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_ind_a,
                res.e_ind_a * KCAL
            );
            println!(
                "  E_ind(2)   B<-A   = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_ind_b,
                res.e_ind_b * KCAL
            );
            println!(
                "  E_disp(2)  [unc]  = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_disp_unc,
                res.e_disp_unc * KCAL
            );
            println!(
                "  E_exch-ind(2)     = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_exch_ind,
                res.e_exch_ind * KCAL
            );
            println!(
                "  E_disp(2)  [TDA]  = {:>18.10} Hartree  ({:>10.4} kcal/mol)  [coupled]",
                res.e_disp_coupled,
                res.e_disp_coupled * KCAL
            );
            println!(
                "  E_exch-disp(2)    = {:>18.10} Hartree  ({:>10.4} kcal/mol)",
                res.e_exch_disp,
                res.e_exch_disp * KCAL
            );
            println!(
                "  uncoupled calib.  = {:>18.10} Hartree  ({:>10.4} kcal/mol)  [diagnostic]",
                res.interaction_total_uncoupled_calibration(),
                res.interaction_total_uncoupled_calibration() * KCAL
            );
            println!(
                "  SAPT sum          = {:>18.10} Hartree  ({:>10.4} kcal/mol)  [coupled]",
                res.interaction_total(),
                res.interaction_total() * KCAL
            );
            println!(
                "  supermolecular    = {:>18.10} Hartree  ({:>10.4} kcal/mol)  [CP-corrected]",
                res.e_int_supermolecular,
                res.e_int_supermolecular * KCAL
            );
            println!(
                "  delta             = {:>18.10} Hartree  ({:>10.4} kcal/mol)  \
                    [higher-order remainder]",
                res.delta,
                res.delta * KCAL
            );
        }
        "energy"
        | "charges"
        | "gradient"
        | "optimize"
        | "frequencies"
        | "hessian"
        | "saddle"
        | "polarizability"
        | "magnetic-response"
        | "nmr"
        | "dynamic-polarizability"
        | "dynamic-beta"
        | "dynamic-gamma"
        | "stability"
        | "stab-following"
        | "field-taylor"
        | "field-mixed"
        | "beta"
        | "gamma"
        | "nuclear-taylor"
        | "nuclear-mixed"
        | "molecular-taylor"
        | "molecular-mixed"
        | "static-spectroscopy"
        | "force-constants"
        | "anharmonic"
        | "orbital-energy-gradients"
        | "frontier-gap-gradient"
        | "raman"
        | "tddft" => {
            use dft_rs::basis::MolBasis;
            use dft_rs::dispersion::{dispersion_energy, Dispersion};
            use dft_rs::grid::{GridOptions, MolecularGrid};
            use dft_rs::properties;
            use dft_rs::scf::ScfOptions;
            use dft_rs::xc::FunctionalSpec;
            // Dispersion handling. VV10 is a dispersion method (nonlocal
            // correlation) selectable through --disp alongside the
            // empirical models; --nlc stays as a compatibility alias.
            // Named density and geometry terms follow the functional,
            // including the shared equal-x/c spelling of combined entries.
            let method = dft_rs::driver::method::MolecularMethod {
                xc: cli.xc.clone(),
                x: cli.x_func.clone(),
                c: cli.c_func.clone(),
                ..Default::default()
            };
            let func_name_for_disp = method.dispersion_functional_name();
            let (_, builtin) = dft_rs::xc::built_in_dispersion(&func_name_for_disp);
            let dispersion_request = cli.disp.to_ascii_lowercase();
            if let Some(term) = builtin {
                let defining_d3 = term == dft_rs::xc::RequiredNonlocalTerm::D3Zero;
                if dispersion_request != "auto" && (defining_d3 || dispersion_request != "none") {
                    return Err(format!(
                        "`{func_name_for_disp}` includes {} by definition. Select --disp auto.",
                        term.name()
                    )
                    .into());
                }
                if defining_d3 && !matches!(cli.nlc.as_str(), "auto" | "none") {
                    return Err(
                        "the defining D3 pair correction cannot be combined with --nlc".into(),
                    );
                }
            }
            let (dispersion, nlc_from_disp) = match dispersion_request.as_str() {
                "auto" => (
                    builtin
                        .and_then(dft_rs::xc::RequiredNonlocalTerm::pair_dispersion)
                        .unwrap_or_default(),
                    None,
                ),
                "vv10" | "vv10-scf" => (Dispersion::None, Some("vv10")),
                "vv10-post" => (Dispersion::None, Some("vv10-post")),
                "xdm" => (Dispersion::None, None),
                other => (
                    Dispersion::parse(other)
                        .ok_or_else(|| format!("unknown dispersion model: {}", cli.disp))?,
                    None,
                ),
            };
            let nlc_request = cli.nlc.to_ascii_lowercase();
            let nlc_mode = match (nlc_from_disp, nlc_request.as_str()) {
                (Some(m), "auto") => m.to_string(),
                (Some(_), _) => return Err("--disp vv10* cannot be combined with --nlc".into()),
                (None, m) => m.to_owned(),
            };
            let set = resolve_basis(&cli)?;
            let mb = MolBasis::build(&molecule, &set)?;
            let mut opts = ScfOptions {
                vv10_grid_level: cli.vv10_grid_level,
                vv10_grid_weight_cutoff: cli.grid_weight_cutoff.unwrap_or(1e-14),
                ..ScfOptions::default()
            };
            if let Some(mb) = cli.incore_mb {
                opts.incore_mb = mb;
            }
            if cli.incore {
                opts.eri_mode = dft_rs::scf::EriMode::InCore {
                    limit_mb: opts.incore_mb,
                };
            }
            opts.density_fit = cli.density_fit.clone();
            // SCF convergence. PySCF's rule: conv_tol_grad defaults to
            // sqrt(conv_tol), so overriding the energy threshold alone
            // moves the gradient threshold with it -- which is what a user
            // asking for tighter convergence means.
            if let Some(v) = cli.conv_tol {
                opts.e_tol = v;
                opts.conv_tol_grad = v.sqrt();
            }
            if let Some(v) = cli.conv_tol_grad {
                opts.conv_tol_grad = v;
            }
            if let Some(v) = cli.conv_tol_dp {
                opts.p_tol = v;
            }
            if let Some(v) = cli.max_iter {
                opts.max_iter = v;
            }
            if let Some(v) = cli.diis_size {
                opts.diis_size = v;
            }
            if let Some(v) = cli.level_shift {
                opts.level_shift = v;
            }
            if let Some(v) = cli.damping {
                opts.damping = v;
            }
            if cli.no_adiis {
                opts.use_adiis = false;
            }
            if let Some(v) = cli.adiis_until_error {
                opts.adiis_until_error = v;
            }
            if cli.no_soscf {
                opts.use_soscf = false;
            }
            if cli.unrestricted {
                opts.reference = dft_rs::Reference::Uhf;
            }
            if let Some(v) = cli.guess_mix {
                opts.guess_mix = v;
            }
            if let Some(g) = &cli.guess {
                opts.guess = match g.to_ascii_lowercase().as_str() {
                    "sad" => dft_rs::scf::guess::Guess::Sad,
                    "gwh" => dft_rs::scf::guess::Guess::Gwh,
                    "core" => dft_rs::scf::guess::Guess::Core,
                    other => return Err(format!("unknown --guess: {other} (sad|gwh|core)").into()),
                };
            }
            // A uniform external field enters H_core as +E.r together with
            // the matching nuclear term -Z_A R_A.E. Both move with the
            // nuclei. `assemble_gradient` and `analytic_hessian_full`
            // differentiate both terms, so gradients, vibrations, geometry
            // optimization, transition-state search and Raman all work in a
            // field without switching derivative models.
            if let Some(e) = cli.field {
                opts.external_field = Some(dft_rs::math::Vec3::new(e[0], e[1], e[2]));
            }
            // Continuum solvation. `--solvent water|smd-water` selects the
            // model; `--eps` overrides the dielectric constant.
            // Continuum solvation. The solvent may be named from the
            // bundled descriptor table (179 entries) or specified purely
            // by its descriptors; `--eps` alone is enough for the
            // electrostatic models and overrides a named solvent's value.
            let mut smd = false;
            let mut p = dft_rs::pcm::PcmOptions::default();
            let mut want_pcm = false;
            if let Some(spec) = &cli.solvent {
                want_pcm = true;
                let mut name = spec.as_str();
                if let Some(rest) = name.strip_prefix("smd:") {
                    smd = true;
                    name = rest;
                } else if let Some(rest) = name.strip_prefix("cosmo:") {
                    p.model = dft_rs::pcm::PcmModel::Cosmo;
                    name = rest;
                } else if let Some(rest) = name.strip_prefix("ief:") {
                    p.model = dft_rs::pcm::PcmModel::IefPcm;
                    name = rest;
                } else if let Some(rest) = name.strip_prefix("ssvpe:") {
                    p.model = dft_rs::pcm::PcmModel::Ssvpe;
                    name = rest;
                }
                // SMD's electrostatics uses its own intrinsic Coulomb radii,
                // unscaled -- not the scaled Bondi set plain C-PCM uses.
                if smd {
                    p.smd_radii = true;
                    p.vdw_scale = 1.0;
                }
                if name.eq_ignore_ascii_case("custom") {
                    if cli.eps.is_none() {
                        return Err("--solvent custom needs --eps (and --refractive-index / \
                             --alpha / --beta / --gamma for the SMD term)"
                            .into());
                    }
                } else {
                    match dft_rs::pcm::solvent_by_name(name) {
                        Some(s) => {
                            p.eps = s.eps;
                            p.descriptors = Some(s);
                        }
                        None => {
                            return Err(format!(
                                "unknown solvent `{name}` ({} bundled; try `--list-solvents`, \
                                 or `--solvent custom --eps X` to give descriptors directly)",
                                dft_rs::pcm::solvent_names().len()
                            )
                            .into())
                        }
                    }
                }
            }
            // Descriptor overrides work with or without a named solvent.
            if let Some(e) = cli.eps {
                p.eps = e;
                want_pcm = true;
            }
            if let Some(d) = &mut p.descriptors {
                if let Some(v) = cli.refractive_index {
                    d.n = v;
                }
                if let Some(v) = cli.alpha {
                    d.alpha = v;
                }
                if let Some(v) = cli.beta {
                    d.beta = v;
                }
                if let Some(v) = cli.gamma {
                    d.gamma = v;
                }
            } else if cli.refractive_index.is_some()
                || cli.alpha.is_some()
                || cli.beta.is_some()
                || cli.gamma.is_some()
            {
                p.descriptors = Some(dft_rs::pcm::Solvent {
                    eps: p.eps,
                    n: cli.refractive_index.unwrap_or(1.0),
                    alpha: cli.alpha.unwrap_or(0.0),
                    beta: cli.beta.unwrap_or(0.0),
                    gamma: cli.gamma.unwrap_or(0.0),
                    phi: 0.0,
                    psi: 0.0,
                });
                want_pcm = true;
            }
            if want_pcm {
                opts.pcm = Some(p);
            }
            // Resolve the functional BEFORE any SCF work (fails fast on
            // unknown names and on component-wise XC-integrated names).
            let spec = match (&cli.xc, &cli.x_func, &cli.c_func) {
                (Some(name), _, _) => FunctionalSpec::Combined(name.clone()),
                (None, Some(x), Some(c)) => FunctionalSpec::Split {
                    x: x.clone(),
                    c: c.clone(),
                },
                _ => FunctionalSpec::Combined("hf".to_owned()),
            };
            let mut functional = dft_rs::xc::resolve(&spec)?;
            if let Some(w) = cli.omega {
                functional = functional.with_omega(w)?;
            }
            let gopts = GridOptions {
                level: cli.grid_level,
                weight_cutoff: cli.grid_weight_cutoff.unwrap_or(1e-14),
                ..GridOptions::default()
            };
            let grid = if functional.parts.is_empty() {
                None
            } else {
                Some(MolecularGrid::build(&molecule, &gopts)?)
            };
            // VV10 mode: self-consistent (default for -V functionals) or
            // post-SCF ("vv10-post"), or off.
            let builtin_vv10 = dft_rs::nlc::Vv10Params::for_functional(&func_name_for_disp);
            let is_v_functional = builtin_vv10.is_some();
            let vv10_params = builtin_vv10.unwrap_or(dft_rs::nlc::Vv10Params::ORIGINAL);
            // -V functionals carry VV10 as part of their DEFINITION: it is
            // enabled automatically, and an explicit VV10 request on top is
            // a specification conflict (hard error). `none` stays allowed
            // for semilocal-only debugging.
            if is_v_functional && matches!(nlc_mode.as_str(), "vv10" | "vv10-scf" | "vv10-post") {
                return Err(format!(
                    "`{func_name_for_disp}` includes VV10 by definition (enabled \
                     automatically); do not request it again via --disp/--nlc. \
                     Use --nlc none to disable it for debugging."
                )
                .into());
            }
            let vv10_post = match nlc_mode.as_str() {
                "auto" => {
                    if is_v_functional {
                        opts.vv10 = Some(vv10_params);
                    }
                    false
                }
                "vv10" | "vv10-scf" => {
                    opts.vv10 = Some(vv10_params);
                    false
                }
                "vv10-post" => true,
                "none" => false,
                other => return Err(format!("unknown nlc model: {other}").into()),
            };
            if cli.command == "gradient" {
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let mut grad = if opts.density_fit.is_some() {
                    dft_rs::gradient::numerical_gradient(
                        &molecule,
                        &set,
                        &opts,
                        &functional,
                        if functional.parts.is_empty() {
                            None
                        } else {
                            Some(&gopts)
                        },
                        1e-4,
                    )?
                } else {
                    dft_rs::gradient::assemble_gradient(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                    )?
                };
                let mut energy = scf.energy;
                if dispersion != Dispersion::None {
                    energy += dispersion_energy(&molecule, &dispersion, &func_name_for_disp)?;
                    let gd = dft_rs::dispersion::dispersion_gradient(
                        &molecule,
                        &dispersion,
                        &func_name_for_disp,
                    )?;
                    for (g, d) in grad.iter_mut().zip(gd) {
                        *g += d;
                    }
                }
                // VV10: self-consistent mode is already inside the SCF and
                // assemble_gradient (exact); post-SCF mode adds the fixed-
                // density energy + gradient here (the density-response
                // cross term is neglected in post mode).
                if opts.vv10.is_some() {
                    println!(
                        "# self-consistent VV10 included (E_nlc = {:.12})",
                        scf.e_nlc
                    );
                } else if vv10_post {
                    use dft_rs::nlc::{vv10_energy, vv10_gradient};
                    let nlc_grid = opts.nonlocal_grid(&molecule)?;
                    let p = scf.p_total();
                    energy += vv10_energy(&mb, &nlc_grid, &p, vv10_params)?;
                    let gv = vv10_gradient(&molecule, &mb, &nlc_grid, &p, vv10_params)?;
                    for (g, d) in grad.iter_mut().zip(gv) {
                        *g += d;
                    }
                    println!("# post-SCF VV10 energy and fixed-density gradient included");
                }
                println!("Energy: {:.12} Hartree", energy);
                println!(
                    "# atom       dE/dx              dE/dy              dE/dz     (Hartree/Bohr)"
                );
                for (i, g) in grad.iter().enumerate() {
                    println!(
                        "{:>5} {:>18.10e} {:>18.10e} {:>18.10e}",
                        i + 1,
                        g.x,
                        g.y,
                        g.z
                    );
                }
                return Ok(());
            }
            if cli.command == "optimize" {
                let res = dft_rs::optimizer::optimize(
                    &molecule,
                    &set,
                    &opts,
                    &functional,
                    &func_name_for_disp,
                    if functional.parts.is_empty() {
                        None
                    } else {
                        Some(&gopts)
                    },
                    &dispersion,
                    &dft_rs::optimizer::OptOptions {
                        trajectory: cli.trajectory.as_ref().map(std::path::PathBuf::from),
                        ..dft_rs::optimizer::OptOptions::default()
                    },
                )?;
                let input = std::path::Path::new(&cli.path);
                let stem = input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("optimized");
                let output = input.with_file_name(format!("{stem}.dftopt.xyz"));
                std::fs::write(
                    &output,
                    res.molecule
                        .to_xyz_string("dft-rs optimized geometry; Angstrom"),
                )?;
                println!(
                    "Converged: {} after {} steps",
                    res.converged, res.iterations
                );
                println!("Final energy: {:.12} Hartree", res.energy);
                println!("Optimized geometry: {}", output.display());
                if let Some(t) = &cli.trajectory {
                    println!(
                        "Trajectory: {} ({} frames, written as the walk proceeded)",
                        t,
                        res.trajectory.len()
                    );
                }
                return Ok(());
            }
            if cli.command == "frequencies" || cli.command == "hessian" {
                // Analytic CPKS Hessian when the method supports it
                // (HF/LC-HF, LDA, GGA, meta-GGA, and SC-VV10 through the
                // nonlocal skeleton/f_nlc and ECP terms), with the empirical
                // dispersion second derivative (D2/D3(zero)/D3(BJ)/CHG) added
                // on top via its own dual-seeded `dispersion_hessian` — the
                // same energy-consistent-correction pattern the gradient
                // path already uses for `dispersion_gradient`. A uniform
                // external field is analytic too; RI-J still routes to the
                // numerical Hessian through a typed capability refusal.
                let analytic_ok = opts.density_fit.is_none();
                // Try the analytic path and fall back on a "not implemented"
                // refusal, rather than deciding in advance which
                // combinations are supported. The advance guess was
                // incomplete combinations can otherwise abort instead of
                // using a complete numerical Hessian. Keying on the error
                // variant means a newly-refused combination degrades
                // gracefully without duplicating a capability table here.
                let equilibrium_scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let analytic = if analytic_ok {
                    match dft_rs::hessian::analytic_hessian_full(
                        &molecule,
                        &mb,
                        &opts,
                        &functional,
                        &equilibrium_scf,
                        grid.as_ref(),
                    ) {
                        Ok(full) => {
                            println!("# analytic CPKS Hessian");
                            let mut hessian = full.hessian;
                            if dispersion != Dispersion::None {
                                let hd = dft_rs::dispersion::dispersion_hessian(
                                    &molecule,
                                    &dispersion,
                                    &func_name_for_disp,
                                )?;
                                for (t, v) in hessian.as_mut_slice().iter_mut().zip(hd.as_slice()) {
                                    *t += *v;
                                }
                            }
                            Some(dft_rs::hessian::HessianResult {
                                hessian,
                                dipole_derivatives: full.dipole_derivatives,
                            })
                        }
                        Err(e) if e.is_unsupported() => {
                            eprintln!("dft-rs: {e}");
                            eprintln!(
                                "dft-rs: falling back to the numerical Hessian of the analytic \
                                 gradients (correct, but slower)"
                            );
                            None
                        }
                        Err(e) => return Err(e.into()),
                    }
                } else {
                    None
                };
                let res = if let Some(res) = analytic {
                    res
                } else {
                    dft_rs::hessian::numerical_hessian(
                        &molecule,
                        &set,
                        &opts,
                        &functional,
                        &func_name_for_disp,
                        if functional.parts.is_empty() {
                            None
                        } else {
                            Some(&gopts)
                        },
                        &dispersion,
                        5e-3,
                    )?
                };
                if cli.command == "hessian" {
                    println!("# Cartesian Hessian (Hartree/Bohr^2)");
                    for i in 0..res.hessian.rows {
                        let row: Vec<String> = (0..res.hessian.cols)
                            .map(|j| format!("{:.10e}", res.hessian[(i, j)]))
                            .collect();
                        println!("{}", row.join(" "));
                    }
                    return Ok(());
                }
                let vib = dft_rs::properties::vibrational::vibrational_analysis(
                    &molecule,
                    &res.hessian,
                    Some(&res.dipole_derivatives),
                )?;
                println!("# mode   frequency (cm^-1)     IR intensity (km/mol)");
                for (i, (f, ir)) in vib
                    .frequencies_cm
                    .iter()
                    .zip(&vib.ir_intensities)
                    .enumerate()
                {
                    println!("{:>5} {:>18.2} {:>22.4}", i + 1, f, ir);
                }
                println!(
                    "# projected translation/rotation modes: {}",
                    vib.n_projected
                );
                println!(
                    "# imaginary modes: {} ({})",
                    vib.n_imaginary,
                    match vib.n_imaginary {
                        0 => "minimum",
                        1 => "transition state",
                        _ => "higher-order saddle",
                    }
                );
                if vib.n_imaginary == 0 {
                    let electronic_energy = equilibrium_scf.energy
                        + dft_rs::dispersion::dispersion_energy(
                            &molecule,
                            &dispersion,
                            &func_name_for_disp,
                        )?;
                    let thermo = dft_rs::properties::thermochemistry::thermochemistry(
                        &molecule,
                        &vib.frequencies_cm,
                        electronic_energy,
                        dft_rs::properties::thermochemistry::ThermochemistryOptions {
                            temperature_kelvin: cli.thermo_temperature,
                            pressure_pascal: cli.pressure,
                            symmetry_number: cli.symmetry_number,
                            electronic_degeneracy: cli.multiplicity,
                        },
                    )?;
                    let s_cal_mol_k =
                        thermo.entropy * dft_rs::constants::HARTREE_TO_KCAL_MOL * 1000.0;
                    println!(
                        "# RRHO thermochemistry: T = {:.2} K, p = {:.3} Pa, symmetry = {}, \
                            degeneracy = {}",
                        thermo.temperature_kelvin,
                        thermo.pressure_pascal,
                        thermo.symmetry_number,
                        thermo.electronic_degeneracy
                    );
                    println!(
                        "Zero-point energy:       {:>18.10} Hartree",
                        thermo.zero_point_energy
                    );
                    println!(
                        "Internal-energy correction:{:>17.10} Hartree",
                        thermo.internal_energy_correction
                    );
                    println!(
                        "Enthalpy correction:      {:>18.10} Hartree",
                        thermo.enthalpy_correction
                    );
                    println!(
                        "Entropy:                   {:>18.10e} Hartree/K ({:.6} cal/mol/K)",
                        thermo.entropy, s_cal_mol_k
                    );
                    println!(
                        "Gibbs correction:         {:>18.10} Hartree",
                        thermo.gibbs_correction
                    );
                    println!(
                        "Total enthalpy:           {:>18.10} Hartree",
                        thermo.enthalpy
                    );
                    println!(
                        "Total Gibbs free energy:  {:>18.10} Hartree",
                        thermo.gibbs_free_energy
                    );
                } else {
                    println!(
                        "# RRHO thermochemistry skipped: {} imaginary mode(s); no \
                            transition-state convention was requested",
                        vib.n_imaginary
                    );
                }
                return Ok(());
            }
            if cli.command == "tddft" {
                use dft_rs::response::tddft::{TdaSolver, TdaSpin};
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let mode = match cli.tda_spin.as_str() {
                    "singlet" => TdaSpin::Singlet,
                    "triplet" => TdaSpin::Triplet,
                    "spinresolved" => TdaSpin::SpinResolved,
                    "auto" => {
                        if scf.unrestricted {
                            TdaSpin::SpinResolved
                        } else {
                            TdaSpin::Singlet
                        }
                    }
                    other => return Err(format!("unknown --tda-spin: {other}").into()),
                };
                let solver = TdaSolver::new(
                    &molecule,
                    &mb,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                    mode,
                )?;
                let states = match cli.tddft_method.as_str() {
                    "tda" => solver.solve(cli.nstates)?,
                    "rpa" | "tddft" | "tdhf" => solver.solve_rpa(cli.nstates)?,
                    other => return Err(format!("unknown --tddft-method: {other}").into()),
                };
                println!(
                    "# {} excitations ({mode:?})",
                    if cli.tddft_method == "tda" {
                        "TDA"
                    } else {
                        "RPA"
                    }
                );
                // Rotatory strengths in the customary 1e-40 cgs unit.
                const R_AU_TO_CGS: f64 = 471.443_8;
                println!(
                    "# state   E (Hartree)      E (eV)      lambda (nm)          f   \
                     R_len (1e-40 cgs)  R_vel (1e-40 cgs)"
                );
                for (k, st) in states.iter().enumerate() {
                    let ev = st.energy * dft_rs::constants::HARTREE_TO_EV;
                    let nm = if ev > 0.0 {
                        1239.841984 / ev
                    } else {
                        f64::INFINITY
                    };
                    println!(
                        "{:>7} {:>13.8} {:>11.4} {:>14.2} {:>12.6} {:>18.4} {:>18.4}",
                        k + 1,
                        st.energy,
                        ev,
                        nm,
                        st.oscillator_strength,
                        st.rotatory_strength * R_AU_TO_CGS,
                        st.rotatory_strength_velocity * R_AU_TO_CGS
                    );
                }
                // The SIGN of R is the ECD observable — it is what tells
                // enantiomers apart — so it is worth saying plainly that the
                // length gauge is origin-dependent in a finite basis while
                // the velocity gauge is not. Their difference is the
                // error bar; for an achiral molecule both should vanish.
                println!(
                    "# R_len is length-gauge (origin-dependent in a finite basis, \
                     origin here = coordinate origin);"
                );
                println!(
                    "# R_vel is velocity-gauge (origin-independent). Their gap \
                     measures the basis-set error."
                );
                return Ok(());
            }
            if matches!(
                cli.command.as_str(),
                "nuclear-taylor"
                    | "nuclear-mixed"
                    | "molecular-taylor"
                    | "molecular-mixed"
                    | "static-spectroscopy"
                    | "force-constants"
                    | "anharmonic"
                    | "orbital-energy-gradients"
                    | "frontier-gap-gradient"
            ) {
                if dispersion != Dispersion::None
                    || opts.vv10.is_some()
                    || functional.required_nonlocal_kind().is_some()
                {
                    return Err(
                        "nuclear Taylor response requires moving-geometry kernels for the \
                        selected correction before it can be calculated"
                            .into(),
                    );
                }
                let check_path = |d: &[[f64; 3]]| -> Result<(), Box<dyn std::error::Error>> {
                    if d.len() != molecule.len() {
                        return Err(
                            "nuclear direction must contain one Cartesian displacement per atom"
                                .into(),
                        );
                    }
                    Ok(())
                };
                if let Some(d) = &cli.nuclear_direction {
                    check_path(d)?;
                }
                if let Some(paths) = &cli.mixed_nuclear_directions {
                    for d in paths {
                        check_path(d)?;
                    }
                }
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let config = dft_rs::response::ResponseSolverConfig {
                    tolerance: cli.response_tol,
                    max_iterations: cli.response_max_iter,
                    gmres_restart: cli.response_restart,
                    ..Default::default()
                };
                if matches!(
                    cli.command.as_str(),
                    "orbital-energy-gradients" | "frontier-gap-gradient"
                ) {
                    if cli.command == "frontier-gap-gradient" {
                        let report = dft_rs::response::frontier_gap_gradient(
                            &molecule,
                            &mb,
                            &functional,
                            grid.as_ref(),
                            &scf,
                            &opts,
                            &cli.orbital_options,
                            &config,
                        )?;
                        print_response_metadata(&report);
                        print_orbital_gradients(&report.value.orbitals);
                        for (spin, gap) in report.value.gaps.iter().enumerate() {
                            if let Some(gap) = gap {
                                println!(
                                    "frontier_gap {spin} {} {} {:.12e}",
                                    gap.homo, gap.lumo, gap.gap
                                );
                                if let Some(gradient) = &gap.gradient {
                                    println!("gap_gradient {spin} {:?}", gradient);
                                } else {
                                    println!("gap_gradient {spin} undefined");
                                }
                            } else {
                                println!("frontier_gap {spin} absent");
                            }
                        }
                    } else {
                        let report = dft_rs::response::orbital_energy_gradients(
                            &molecule,
                            &mb,
                            &functional,
                            grid.as_ref(),
                            &scf,
                            &opts,
                            &cli.orbital_options,
                            &config,
                        )?;
                        print_response_metadata(&report);
                        print_orbital_gradients(&report.value);
                    }
                } else if cli.command == "anharmonic" {
                    let report = dft_rs::response::anharmonic::molecular::molecular_anharmonic(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        cli.vibrational_modes.as_deref(),
                        cli.anharmonic_raman,
                        &cli.anharmonic_config,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    let value = &report.value;
                    let spectrum = &value.spectrum;
                    println!("anharmonic_method {}", spectrum.method.as_str());
                    println!("energy_order 2");
                    println!("energy_treatment {}", spectrum.method.energy_treatment());
                    println!("transition_moment_order 2");
                    println!(
                        "intensity_treatment {}",
                        spectrum.method.intensity_treatment()
                    );
                    println!(
                        "resonance_treatment {}",
                        spectrum.method.resonance_treatment()
                    );
                    println!("eigensystem_kind {}", spectrum.method.eigensystem_kind());
                    println!("convergence_scope numerical diagnostics; perturbation validity is separate");
                    println!("selected_mode_ids {:?}", value.selected_mode_ids);
                    println!("energy {:.16e}", scf.energy);
                    println!("n_projected {}", value.n_projected);
                    println!(
                        "harmonic_frequencies_cm {:?}",
                        value.harmonic_frequencies_cm
                    );
                    println!("patterns {:?}", value.patterns);
                    println!("cubic {:?}", value.polynomial.cubic);
                    println!("quartic {:?}", value.polynomial.quartic);
                    for (rank, values) in [
                        (1, &value.polynomial.dipole.first),
                        (2, &value.polynomial.dipole.second),
                        (3, &value.polynomial.dipole.third),
                    ] {
                        println!("dipole_derivatives_{rank} {:?}", values);
                    }
                    if let Some(alpha) = &value.polynomial.polarizability {
                        for (rank, values) in
                            [(1, &alpha.first), (2, &alpha.second), (3, &alpha.third)]
                        {
                            println!("polarizability_derivatives_{rank} {:?}", values);
                        }
                    }
                    println!("polyad_occupations {:?}", spectrum.polyad_occupations);
                    println!("polyad_energies {:?}", spectrum.polyad_energies);
                    for i in 0..spectrum.polyad_energies.len() {
                        println!(
                            "effective_hamiltonian {i} {:?}",
                            (0..spectrum.effective_hamiltonian.cols)
                                .map(|j| spectrum.effective_hamiltonian[(i, j)])
                                .collect::<Vec<_>>()
                        );
                        println!(
                            "polyad_vectors {i} {:?}",
                            (0..spectrum.polyad_vectors.cols)
                                .map(|j| spectrum.polyad_vectors[(i, j)])
                                .collect::<Vec<_>>()
                        );
                    }
                    println!("maximum_force {:.12e}", value.maximum_force);
                    if let Some(rotation) = &value.polynomial.rotation {
                        println!("rotor_class {:?}", rotation.class);
                        println!("inverse_inertia {:?}", rotation.inverse_inertia);
                        println!("principal_moments {:?}", rotation.principal_moments);
                        println!("coriolis {:?}", rotation.coriolis);
                        println!(
                            "watson_pseudopotential {:.16e}",
                            rotation.watson_pseudopotential
                        );
                        println!("eckart_defect {:?}", rotation.eckart_defect);
                        println!("mode_metric_defect {:?}", rotation.mode_metric_defect);
                    }
                    println!(
                        "force_permutation_defects {:?}",
                        value.force_permutation_defects
                    );
                    println!(
                        "force_permutation_comparisons {:?}",
                        value.force_permutation_comparisons
                    );
                    println!(
                        "polarizability_symmetry_defect {:?}",
                        value.polarizability_symmetry_defect
                    );
                    println!(
                        "polarizability_symmetry_comparisons {}",
                        value.polarizability_symmetry_comparisons
                    );
                    println!("zero_point_energy {:.16e}", spectrum.zero_point_energy);
                    println!(
                        "harmonic_zero_point_energy {:.16e}",
                        spectrum.harmonic_zero_point_energy
                    );
                    println!("basis_states {}", spectrum.basis_states);
                    println!("added_resonant_states {}", spectrum.added_resonant_states);
                    println!(
                        "minimum_external_denominator {:?}",
                        spectrum.minimum_external_denominator
                    );
                    println!(
                        "exterior_mixing_norm {:.12e}",
                        spectrum.exterior_mixing_norm
                    );
                    println!("hermiticity_defect {:.12e}", spectrum.hermiticity_defect);
                    println!("eigen_residual {:.12e}", spectrum.eigen_residual);
                    for (i, band) in spectrum.bands.iter().enumerate() {
                        println!(
                            "anharmonic_band {i} {} {:.12e} {:.12e} {:.12e} {:.12e} {:.12e}",
                            band.character,
                            band.frequency_cm,
                            band.harmonic_frequency_cm,
                            band.ir_intensity,
                            band.double_harmonic_ir_intensity,
                            band.dominant_weight
                        );
                        println!("dominant_occupation {i} {:?}", band.dominant_occupation);
                        for (name, vector) in [
                            ("transition_dipole", band.transition_dipole),
                            ("double_harmonic_dipole", band.double_harmonic_dipole),
                            ("mechanical_dipole", band.mechanical_dipole),
                            ("electrical_dipole", band.electrical_dipole),
                            ("mixed_dipole", band.mixed_dipole),
                        ] {
                            println!(
                                "{name} {i} {:.12e} {:.12e} {:.12e}",
                                vector[0], vector[1], vector[2]
                            );
                        }
                        if let Some(activity) = band.raman_activity {
                            println!(
                                "raman_activity {i} {:.12e} {:.12e}",
                                activity,
                                band.double_harmonic_raman_activity.unwrap()
                            );
                            println!(
                                "transition_polarizability {i} {:?}",
                                band.transition_polarizability.unwrap()
                            );
                        }
                    }
                    for warning in &report.warnings {
                        println!("warning {warning}");
                    }
                } else if cli.command == "force-constants" {
                    let report = if let Some(patterns) = &cli.mixed_nuclear_directions {
                        dft_rs::response::force_constants::analytic_force_constants(
                            &molecule,
                            &mb,
                            &functional,
                            grid.as_ref(),
                            &scf,
                            &opts,
                            patterns,
                            cli.response_order,
                            &config,
                        )
                    } else if let Some(coordinates) = &cli.force_coordinates {
                        dft_rs::response::force_constants::cartesian_force_constant_block(
                            &molecule,
                            &mb,
                            &functional,
                            grid.as_ref(),
                            &scf,
                            &opts,
                            coordinates,
                            cli.response_order,
                            &config,
                        )
                    } else {
                        dft_rs::response::force_constants::cartesian_force_constants(
                            &molecule,
                            &mb,
                            &functional,
                            grid.as_ref(),
                            &scf,
                            &opts,
                            cli.response_order,
                            &config,
                        )
                    }?;
                    print_response_metadata(&report);
                    println!(
                        "order {} independent_components {} directional_evaluations {}",
                        report.value.order,
                        report.value.independent_components,
                        report.value.directional_evaluations
                    );
                    for (ix, v) in report.value.tensor.iter().enumerate() {
                        println!("tensor {ix} {v:.12e}");
                    }
                    println!(
                        "permutation_defect {:.12e} permutation_comparisons {} \
                        translation_residual {:?} translation_comparisons {}",
                        report.value.permutation_defect,
                        report.value.permutation_comparisons,
                        report.value.translation_residual,
                        report.value.translation_comparisons
                    );
                    println!(
                        "accumulation_roundoff {:?}",
                        report.value.accumulation_roundoff
                    );
                } else if cli.command == "static-spectroscopy" {
                    let report = dft_rs::response::nuclear::static_spectroscopic_derivative(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        cli.nuclear_direction
                            .as_ref()
                            .expect("validated nuclear direction"),
                        cli.electric_order,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    println!("electric_order {}", report.value.electric_order);
                    println!(
                        "permutation_comparisons {}",
                        report.value.permutation_comparisons
                    );
                    for (ix, v) in report.value.tensor.iter().enumerate() {
                        println!("tensor {ix} {v:.12e}");
                    }
                    println!(
                        "permutation_defect {:.12e} directional_evaluations {}",
                        report.value.permutation_defect, report.value.directional_evaluations
                    );
                    println!(
                        "accumulation_roundoff {:?}",
                        report.value.accumulation_roundoff
                    );
                } else if cli.command == "molecular-mixed" {
                    let paths: Vec<_> =
                        cli.mixed_nuclear_directions
                            .as_ref()
                            .expect("validated nuclear directions")
                            .iter()
                            .cloned()
                            .zip(
                                cli.mixed_field_directions
                                    .as_ref()
                                    .expect("validated field directions")
                                    .iter()
                                    .copied(),
                            )
                            .map(|(nuclear, electric)| {
                                dft_rs::response::nuclear::MolecularDirection { nuclear, electric }
                            })
                            .collect();
                    let report = dft_rs::response::nuclear::mixed_molecular_response(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        &paths,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    println!("energy_derivative {:.12e}", report.value.energy_derivative);
                    println!(
                        "dipole_derivative {:.12e} {:.12e} {:.12e}",
                        report.value.dipole_derivative[0],
                        report.value.dipole_derivative[1],
                        report.value.dipole_derivative[2]
                    );
                    println!(
                        "accumulation_roundoff {:.12e} dipole_accumulation_roundoff {:?} \
                        directional_evaluations {}",
                        report.value.energy_accumulation_roundoff,
                        report.value.dipole_accumulation_roundoff,
                        report.value.directional_evaluations
                    );
                } else if cli.command == "nuclear-mixed" {
                    let report = dft_rs::response::nuclear::mixed_nuclear_response(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        cli.mixed_nuclear_directions
                            .as_ref()
                            .expect("validated nuclear directions"),
                        &config,
                    )?;
                    print_response_metadata(&report);
                    println!("energy_derivative {:.12e}", report.value.energy_derivative);
                    println!(
                        "accumulation_roundoff {:.12e} directional_evaluations {}",
                        report.value.accumulation_roundoff, report.value.directional_evaluations
                    );
                } else {
                    let report = dft_rs::response::nuclear::molecular_taylor(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        &dft_rs::response::nuclear::MolecularDirection {
                            nuclear: cli
                                .nuclear_direction
                                .clone()
                                .expect("validated nuclear direction"),
                            electric: if cli.command == "molecular-taylor" {
                                cli.field_direction
                            } else {
                                [0.0; 3]
                            },
                        },
                        cli.response_order,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    for (n, e) in report.value.energy_components.iter().enumerate() {
                        println!("energy_coefficient {n} {:.12e}", e.total());
                        println!(
                            "energy_components {n} nuclear {:.12e} nuclear_field {:.12e} \
                            one_electron {:.12e} hartree {:.12e} exchange {:.12e} \
                            long_range_exchange {:.12e} semilocal_xc {:.12e}",
                            e.nuclear_repulsion,
                            e.nuclear_field,
                            e.one_electron,
                            e.hartree,
                            e.exchange,
                            e.long_range_exchange,
                            e.semilocal_xc
                        );
                        let mu = report.value.dipole_coefficients[n];
                        println!(
                            "dipole_coefficient {n} {:.12e} {:.12e} {:.12e}",
                            mu[0], mu[1], mu[2]
                        );
                    }
                    println!(
                        "projector_defects {:?} stationarity_defects {:?} metric_defects {:?} \
                        particle_number_coefficients {:?}",
                        report.value.projector_defects,
                        report.value.stationarity_defects,
                        report.value.metric_defects,
                        report.value.particle_number_coefficients
                    );
                    println!(
                        "reference_energy_error {:.12e} reference_fock_error {:.12e}",
                        report.value.reference_energy_error, report.value.reference_fock_error
                    );
                }
                return Ok(());
            }
            if matches!(cli.command.as_str(), "stability" | "stab-following") {
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let report = dft_rs::response::stability_from_state(
                    &molecule,
                    &mb,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                    &cli.stabilize_options.analysis,
                )?;
                print_stability(&report);
                println!("energy {:.12e}", scf.energy);
                return Ok(());
            }
            if matches!(cli.command.as_str(), "dynamic-beta" | "dynamic-gamma") {
                use dft_rs::response::{Frequency, ResponseSolverConfig};
                let order = if cli.command == "dynamic-beta" { 2 } else { 3 };
                let inputs = cli
                    .input_frequencies
                    .clone()
                    .unwrap_or_else(|| vec![[cli.response_omega, cli.response_damping]; order]);
                let frequencies: Vec<_> = if let Some(process) = cli.electric_process {
                    process.inputs(
                        &cli.process_frequencies
                            .clone()
                            .unwrap_or_else(|| vec![cli.response_omega]),
                        cli.response_damping,
                    )?
                } else {
                    inputs
                        .iter()
                        .map(|&[omega, damping]| {
                            if cli.imaginary_frequency {
                                Frequency::Imaginary { frequency: omega }
                            } else {
                                Frequency::Retarded { omega, damping }
                            }
                        })
                        .collect()
                };
                let config = ResponseSolverConfig {
                    tolerance: cli.response_tol,
                    max_iterations: cli.response_max_iter,
                    gmres_restart: cli.response_restart,
                    ..Default::default()
                };
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let mut report = dft_rs::response::dynamic_electric_tensor(
                    &molecule,
                    &mb,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                    &frequencies,
                    &config,
                )?;
                report.value.process = cli.electric_process;
                if let Some(process) = report.value.process {
                    println!("process {}", process.name());
                }
                println!("# {} ({})", report.model, report.units);
                println!(
                    "# shape {:?}; functional {}; reference {:?}",
                    report.shape, report.functional, report.reference
                );
                println!(
                    "# {}; input frequencies {:?}; output frequency {:?}",
                    report.axis_convention, frequencies, report.value.output_frequency
                );
                let label = if order == 2 { "beta" } else { "gamma" };
                for (flat, value) in report.value.tensor.iter().enumerate() {
                    let mut code = flat;
                    let mut axes = vec![0; order + 1];
                    for a in axes.iter_mut().rev() {
                        *a = code % 3;
                        code /= 3;
                    }
                    println!(
                        "{label} {} {:.12e} {:.12e}",
                        axes.iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(" "),
                        value.re,
                        value.im
                    );
                }
                println!(
                    "converged {} residuals {:?} iterations {:?}",
                    report.converged, report.residuals, report.iterations
                );
                println!(
                    "projector_defect {:.12e}; equation_defect {:.12e}; \
                    particle_number_defect {:.12e}",
                    report.value.projector_defect,
                    report.value.equation_defect,
                    report.value.particle_number_defect
                );
                println!(
                    "intrinsic_permutation_defect {:.12e}; intrinsic_permutation_comparisons \
                    {}; subset_solves {}; nonlinear_kernel_evaluations {}",
                    report.value.intrinsic_permutation_defect,
                    report.value.intrinsic_permutation_comparisons,
                    report.value.subset_solves,
                    report.value.nonlinear_kernel_evaluations
                );
                for warning in report.warnings {
                    println!("# warning: {warning}");
                }
                return Ok(());
            }
            if cli.command == "dynamic-polarizability" {
                use dft_rs::response::{Frequency, ResponseSolverConfig};
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let frequency = if cli.imaginary_frequency {
                    Frequency::Imaginary {
                        frequency: cli.response_omega,
                    }
                } else {
                    Frequency::Retarded {
                        omega: cli.response_omega,
                        damping: cli.response_damping,
                    }
                };
                let config = ResponseSolverConfig {
                    tolerance: cli.response_tol,
                    max_iterations: cli.response_max_iter,
                    gmres_restart: cli.response_restart,
                    ..Default::default()
                };
                let report = dft_rs::response::dynamic_polarizability(
                    &molecule,
                    &mb,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                    frequency,
                    &config,
                )?;
                println!("# {} ({})", report.model, report.units);
                println!("# frequency {:?}; {}", frequency, report.axis_convention);
                println!(
                    "# shape {:?}; functional {}; reference {:?}",
                    report.shape, report.functional, report.reference
                );
                for a in 0..3 {
                    for b in 0..3 {
                        println!(
                            "alpha {a} {b} {:.12e} {:.12e}",
                            report.value.alpha[a][b].re, report.value.alpha[a][b].im
                        );
                    }
                }
                println!(
                    "isotropic {:.12e} {:.12e}",
                    report.value.isotropic.re, report.value.isotropic.im
                );
                println!(
                    "converged {} residuals {:?} iterations {:?}",
                    report.converged, report.residuals, report.iterations
                );
                return Ok(());
            }
            if matches!(cli.command.as_str(), "magnetic-response" | "nmr") {
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let config = dft_rs::response::ResponseSolverConfig {
                    tolerance: cli.response_tol,
                    max_iterations: cli.response_max_iter,
                    gmres_restart: cli.response_restart,
                    ..Default::default()
                };
                let o = Vec3::new(
                    cli.magnetic_origin[0],
                    cli.magnetic_origin[1],
                    cli.magnetic_origin[2],
                );
                println!("energy {:.12e}", scf.energy);
                println!(
                    "gauge {:?} origin_bohr {:?}",
                    cli.magnetic_gauge, cli.magnetic_origin
                );
                println!(
                    "unrestricted {} occupied {:?}",
                    scf.unrestricted,
                    [scf.n_alpha, scf.n_beta]
                );
                if cli.command == "magnetic-response" {
                    let report = properties::magnetic::magnetic_density_response(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        cli.magnetic_gauge,
                        o,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    println!(
                        "projector_defect {:.12e} particle_number_defect {:.12e} \
                        antisymmetry_defect {:.12e}",
                        report.value.projector_defect,
                        report.value.particle_number_defect,
                        report.value.antisymmetry_defect
                    );
                    println!("solver_methods {:?}", report.value.solver_methods);
                    for s in 0..2 {
                        for a in 0..3 {
                            for i in 0..mb.nao_sph {
                                for j in 0..mb.nao_sph {
                                    println!(
                                        "density_hat {s} {a} {i} {j} {:.12e}",
                                        report.value.density_hat[s][a][(i, j)]
                                    );
                                    println!(
                                        "explicit_fock_hat {s} {a} {i} {j} {:.12e}",
                                        report.value.explicit_fock_hat[s][a][(i, j)]
                                    );
                                    if s == 0 {
                                        println!(
                                            "overlap_hat {a} {i} {j} {:.12e}",
                                            report.value.overlap_hat[a][(i, j)]
                                        );
                                    }
                                }
                            }
                        }
                    }
                } else {
                    let report = properties::magnetic::molecular_nmr_response(
                        &molecule,
                        &mb,
                        &functional,
                        grid.as_ref(),
                        &scf,
                        &opts,
                        cli.magnetic_gauge,
                        o,
                        &config,
                    )?;
                    print_response_metadata(&report);
                    for (atom, v) in report.value.iter().enumerate() {
                        for i in 0..3 {
                            for j in 0..3 {
                                println!(
                                    "shielding {atom} {i} {j} {:.12e} {:.12e} {:.12e}",
                                    v.tensor[i][j], v.diamagnetic[i][j], v.paramagnetic[i][j]
                                );
                            }
                        }
                        println!("isotropic {atom} {:.12e}", v.isotropic());
                    }
                }
                return Ok(());
            }
            if cli.command == "polarizability" {
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let alpha = dft_rs::response::polarizability::polarizability(
                    &molecule,
                    &mb,
                    &functional,
                    grid.as_ref(),
                    &scf,
                    &opts,
                )?;
                println!("# static polarizability α (a.u.)");
                for row in &alpha {
                    println!("{:>14.6} {:>14.6} {:>14.6}", row[0], row[1], row[2]);
                }
                let iso = (alpha[0][0] + alpha[1][1] + alpha[2][2]) / 3.0;
                println!("isotropic: {:.6} a.u.", iso);
                return Ok(());
            }
            if cli.command == "raman" {
                let scf =
                    run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
                let (res, hessian_method) = dft_rs::hessian::hessian_with_fallback(
                    &molecule,
                    &set,
                    &mb,
                    &opts,
                    &functional,
                    &func_name_for_disp,
                    if functional.parts.is_empty() {
                        None
                    } else {
                        Some(&gopts)
                    },
                    grid.as_ref(),
                    &scf,
                    &dispersion,
                    5e-3,
                )?;
                println!("# {} Hessian", hessian_method.as_str());
                let vib = dft_rs::properties::vibrational::vibrational_analysis(
                    &molecule,
                    &res.hessian,
                    Some(&res.dipole_derivatives),
                )?;
                let act = dft_rs::response::polarizability::raman_activities(
                    &molecule,
                    &set,
                    &opts,
                    &functional,
                    if functional.parts.is_empty() {
                        None
                    } else {
                        Some(&gopts)
                    },
                    &vib.modes,
                    5e-3,
                )?;
                println!("# mode  freq (cm^-1)   IR (km/mol)   Raman activity (A^4/amu)");
                for (i, ((f, ir), ra)) in vib
                    .frequencies_cm
                    .iter()
                    .zip(&vib.ir_intensities)
                    .zip(&act)
                    .enumerate()
                {
                    println!("{:>5} {:>12.2} {:>13.4} {:>24.4}", i + 1, f, ir, ra);
                }
                return Ok(());
            }
            if cli.command == "saddle" {
                let res = dft_rs::saddle::optimize_saddle(
                    &molecule,
                    &set,
                    &opts,
                    &functional,
                    &func_name_for_disp,
                    if functional.parts.is_empty() {
                        None
                    } else {
                        Some(&gopts)
                    },
                    &dispersion,
                    &dft_rs::saddle::SaddleOptions {
                        method: match cli.saddle_method.as_str() {
                            "rsirfo" | "rs-i-rfo" => dft_rs::saddle::SaddleMethod::RsIrfo,
                            _ => dft_rs::saddle::SaddleMethod::Prfo,
                        },
                        trajectory: cli.trajectory.as_ref().map(std::path::PathBuf::from),
                        ..dft_rs::saddle::SaddleOptions::default()
                    },
                )?;
                let input = std::path::Path::new(&cli.path);
                let stem = input
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("saddle");
                let output = input.with_file_name(format!("{stem}.dftts.xyz"));
                std::fs::write(
                    &output,
                    res.molecule
                        .to_xyz_string("dft-rs transition-state geometry; Angstrom"),
                )?;
                println!(
                    "Converged: {} after {} {} steps",
                    res.converged,
                    res.iterations,
                    res.method.as_str()
                );
                println!("Final energy: {:.12} Hartree", res.energy);
                println!("Negative Hessian eigenvalues: {}", res.n_negative);
                println!("Geometry: {}", output.display());
                return Ok(());
            }
            let scf = run_property_scf(&cli, &molecule, &mb, &opts, &functional, grid.as_ref())?;
            let e_disp = if let Some(damping) = xdm_damping {
                let e = dft_rs::dispersion::xdm::xdm_dispersion_energy(
                    &molecule,
                    &mb,
                    [&scf.density[0], &scf.density[1]],
                    &gopts,
                    &damping,
                )?;
                println!(
                    "Dispersion (XDM; a1={}, a2={} Bohr): {:.12} Hartree",
                    damping.a1, damping.a2, e
                );
                e
            } else if dispersion != Dispersion::None {
                let e = dispersion_energy(&molecule, &dispersion, &func_name_for_disp)?;
                println!("Dispersion ({}):    {:.12} Hartree", cli.disp, e);
                e
            } else {
                0.0
            };
            // VV10 nonlocal correlation: self-consistent (already inside
            // the SCF energy) or post-SCF add-on.
            let e_nlc = if opts.vv10.is_some() {
                println!("VV10 NLC (SC):       {:.12} Hartree", scf.e_nlc);
                0.0 // already contained in scf.energy
            } else if vv10_post {
                use dft_rs::nlc::vv10_energy;
                let nlc_grid = opts.nonlocal_grid(&molecule)?;
                let e = vv10_energy(&mb, &nlc_grid, &scf.p_total(), vv10_params)?;
                println!("VV10 NLC (post-SCF): {:.12} Hartree", e);
                e
            } else {
                0.0
            };
            println!("Functional:          {}", functional.description);
            if let Some(ne) = scf.grid_electron_count {
                println!(
                    "Grid: level {}, {} points, ∫ρ = {:.8} ({} electrons)",
                    cli.grid_level,
                    grid.as_ref().map(|g| g.len()).unwrap_or(0),
                    ne,
                    scf.n_alpha + scf.n_beta
                );
            }
            println!(
                "Reference:           {}",
                if scf.unrestricted {
                    "UKS/UHF"
                } else {
                    "RKS/RHF"
                }
            );
            println!("Basis:               {} ({} AOs)", set.name, mb.nao_sph);
            if mb.n_core_electrons > 0 {
                println!(
                    "ECP:                 {} atom(s), {} core electrons replaced",
                    mb.ecps.len(),
                    mb.n_core_electrons
                );
            }
            println!("SCF converged in {} iterations", scf.iterations);
            // SMD adds its non-electrostatic term on top of the SCF total.
            let g_cds = if smd {
                let s = match &opts.pcm.as_ref().and_then(|o| o.descriptors) {
                    Some(d)
                        if !cli
                            .solvent
                            .as_deref()
                            .unwrap_or("")
                            .eq_ignore_ascii_case("smd:water") =>
                    {
                        dft_rs::pcm::SmdSolvent::NonAqueous(*d)
                    }
                    _ => dft_rs::pcm::SmdSolvent::Water,
                };
                dft_rs::pcm::smd_cds_energy(&molecule, s, 2048)?
            } else {
                0.0
            };
            println!(
                "Total energy:        {:.12} Hartree",
                scf.energy + e_disp + e_nlc + g_cds
            );
            println!("Electronic energy:   {:.12} Hartree", scf.e_elec);
            println!("Nuclear repulsion:   {:.12} Hartree", scf.e_nuc);
            if scf.e_xc != 0.0 {
                println!("XC energy:           {:.12} Hartree", scf.e_xc);
            }
            if scf.e_pcm != 0.0 {
                println!("PCM polarization:    {:.12} Hartree", scf.e_pcm);
            }
            if g_cds != 0.0 {
                println!("SMD G_CDS:           {:.12} Hartree", g_cds);
            }
            if scf.unrestricted {
                println!("<S^2>:               {:.6}", scf.s_squared);
            }
            let (homo, lumo) = scf.homo_lumo();
            if let (Some(h), Some(l)) = (homo, lumo) {
                println!("HOMO / LUMO:         {:.6} / {:.6} Hartree", h, l);
            }
            if let Some(path) = &cli.dump_wfn {
                dft_rs::properties::checkpoint::dump_wavefunction(
                    std::path::Path::new(path),
                    &molecule,
                    &mb,
                    &scf,
                )?;
                println!("Wavefunction checkpoint: {path}");
            }
            if let Some(path) = &cli.molden {
                dft_rs::properties::molden::write_molden(
                    std::path::Path::new(path),
                    &molecule,
                    &set,
                    &mb,
                    &scf,
                )?;
                println!("Molden file: {path}");
            }
            if cli.command == "charges" {
                let mull = properties::mulliken_charges(&molecule, &mb, &scf);
                let low = properties::lowdin_charges(&molecule, &mb, &scf)?;
                println!("# atom  element   Mulliken (e)      Loewdin (e)");
                for (i, atom) in molecule.atoms.iter().enumerate() {
                    println!(
                        "{:>5} {:>5} {:>15.8} {:>16.8}",
                        i + 1,
                        dft_rs::z_to_symbol(atom.z).unwrap_or("X"),
                        mull[i],
                        low[i]
                    );
                }
                let mu = properties::dipole_debye(&molecule, &mb, &scf);
                println!(
                    "Dipole: {:.8} {:.8} {:.8} D; |mu| = {:.8} D",
                    mu.x,
                    mu.y,
                    mu.z,
                    mu.norm()
                );
                if cli.nbo {
                    use dft_rs::properties::natural_bond::{
                        natural_bond_analysis, NboOptions, NboSpin,
                    };
                    let analysis = natural_bond_analysis(
                        &molecule,
                        &mb,
                        &scf,
                        NboSpin::Auto,
                        &NboOptions::default(),
                    )?;
                    let nao = &analysis.nao;
                    let symbol = |i: usize| dft_rs::z_to_symbol(molecule.atoms[i].z).unwrap_or("X");
                    println!("# Natural population analysis");
                    println!("# atom  element   natural charge (e)   spin population (e)   configuration");
                    for (i, charge) in nao.atomic_charges.iter().enumerate() {
                        println!(
                            "{:>5} {:>5} {:>18.8} {:>21.8}   {}",
                            i + 1,
                            symbol(i),
                            charge,
                            nao.spin_populations[i],
                            nao.configuration_text[i]
                        );
                    }
                    println!("# Wiberg bond indices in the NAO basis (pairs above 0.05)");
                    let w = &nao.wiberg_bond_indices;
                    for i in 0..molecule.len() {
                        for j in 0..i {
                            if w[(i, j)] > 0.05 {
                                println!(
                                    "{:>5}-{:<5} {:>5}-{:<5} {:>12.6}",
                                    i + 1,
                                    symbol(i),
                                    j + 1,
                                    symbol(j),
                                    w[(i, j)]
                                );
                            }
                        }
                    }
                    for channel in &analysis.channels {
                        println!(
                            "# Natural Lewis structure of the {} density: {:.6} non-Lewis electrons",
                            channel.spin, channel.non_lewis_occupancy
                        );
                        for orbital in channel
                            .nbos
                            .iter()
                            .filter(|b| b.kind.is_lewis() || b.occupancy > 0.01)
                        {
                            println!(
                                "  {:<24} occupancy {:>10.6}  energy {}",
                                orbital.label,
                                orbital.occupancy,
                                orbital
                                    .energy
                                    .map_or("-".to_string(), |e| format!("{e:.6} Hartree"))
                            );
                        }
                        println!(
                            "# Second-order donor-acceptor energies of the {} density (Hartree)",
                            channel.spin
                        );
                        for pair in &channel.perturbation {
                            println!(
                                "  {:<24} -> {:<24} {:>12.6}",
                                channel.nbos[pair.donor].label,
                                channel.nbos[pair.acceptor].label,
                                pair.stabilization
                            );
                        }
                    }
                }
            } else {
                properties::print_mo_table(&scf);
            }
        }
        "integrals" => {
            use dft_rs::basis::MolBasis;
            use dft_rs::integrals::{kinetic_matrix, nuclear_matrix, overlap_matrix};
            use dft_rs::linalg::Orthogonalizer;
            let set = resolve_basis(&cli)?;
            let mb = MolBasis::build(&molecule, &set)?;
            println!("Basis:               {}", set.name);
            println!("Shells:              {}", mb.shells.len());
            println!("Spherical AOs:       {}", mb.nao_sph);
            if mb.n_core_electrons > 0 {
                println!(
                    "ECP:                 {} atom(s), {} core electrons replaced",
                    mb.ecps.len(),
                    mb.n_core_electrons
                );
            }
            println!("Cartesian AOs:       {}", mb.nao_cart);
            println!("Max angular momentum: {}", mb.max_l);
            let s = overlap_matrix(&mb);
            let t = kinetic_matrix(&mb);
            let v = nuclear_matrix(&mb, &molecule, None);
            let orth = Orthogonalizer::canonical(&s, 1e-7)?;
            println!(
                "Overlap: min kept eigenvalue {:.3e}, {} function(s) dropped",
                orth.min_kept_eigenvalue, orth.n_dropped
            );
            let tr = |m: &dft_rs::linalg::Matrix| -> f64 { (0..m.rows).map(|i| m[(i, i)]).sum() };
            println!(
                "Tr S = {:.10}   Tr T = {:.10}   Tr V = {:.10}",
                tr(&s),
                tr(&t),
                tr(&v)
            );
        }
        other => {
            // Unreachable in practice: parse_args() already restricts
            // `command` to exactly the set matched above (see the
            // `match command.as_str()` validation there), so this only
            // fires if that invariant is ever broken by a future edit.
            // Keep this list in sync with that one rather than naming an
            // arbitrary subset of commands.
            return Err(format!(
                "command `{other}` is not implemented; recognized commands: info, \
                 integrals, energy, gradient, charges, optimize, saddle, frequencies, \
                 hessian, polarizability, raman, tddft, sapt, periodic"
            )
            .into());
        }
    }
    Ok(())
}

/// Commands whose plain-text form predates the common driver. They keep that
/// output; explicit occupation controls, `--format` or `--save` use the driver.
const LEGACY_TEXT_COMMANDS: &[&str] = &[
    "force-constants",
    "energy",
    "gradient",
    "charges",
    "hessian",
    "frequencies",
    "polarizability",
    "raman",
];

/// Whether this argument list is executed by the common driver.
pub(crate) fn uses_common_driver(args: &[String]) -> bool {
    let Some(command) = args.first() else {
        return false;
    };
    let Some(spec) = dft_rs::driver::find(command) else {
        return false;
    };
    if spec.route != dft_rs::driver::Route::Driver {
        return false;
    }
    if args.iter().enumerate().any(|(i, arg)| {
        let (key, value) = arg.split_once('=').map_or(
            (arg.as_str(), args.get(i + 1).map(String::as_str)),
            |(key, value)| (key, Some(value)),
        );
        matches!(key, "--disp" | "--dispersion")
            && value.is_some_and(dft_rs::driver::handles_molecular_dispersion_energy)
    }) {
        return true;
    }
    // Occupation constraints belong to the shared reference builder. They
    // must not fall through to the legacy text parser, which has no selector.
    if args.iter().any(|arg| {
        matches!(
            arg.split('=').next(),
            Some(
                "--occupation-mode"
                    | "--scf-optimizer"
                    | "--reference"
                    | "--dispersion-free-polarizabilities"
                    | "--dispersion-free-c6"
                    | "--dispersion-free-radii"
                    | "--dispersion-radius-scale"
                    | "--gks-operator-real"
                    | "--gks-operator-imaginary"
                    | "--spin-axis"
                    | "--xc-terms"
                    | "--occupied-alpha"
                    | "--occupied-beta"
                    | "--occupations-alpha"
                    | "--occupations-beta"
            )
        )
    }) {
        return true;
    }
    // Orbital-dependent functional resolution belongs to the energy driver,
    // including the default text route. Keep the legacy ordinary-SCF text.
    if matches!(
        command.as_str(),
        "energy"
            | "gradient"
            | "hessian"
            | "polarizability"
            | "force-constants"
            | "frequencies"
            | "gibbs-free-energy"
    ) && args.iter().enumerate().any(|(i, arg)| {
        let (key, value) = arg.split_once('=').map_or(
            (arg.as_str(), args.get(i + 1).map(String::as_str)),
            |(key, value)| (key, Some(value)),
        );
        matches!(
            key,
            "--double-hybrid-model"
                | "--double-hybrid-target-xc"
                | "--pt2-same-spin"
                | "--pt2-opposite-spin"
        ) || key == "--xc"
            && value.is_some_and(|name| {
                dft_rs::xc::double_hybrid::DoubleHybridFunctional::named(name).is_some()
            })
    }) {
        return true;
    }
    if matches!(command.as_str(), "frequencies" | "gibbs-free-energy")
        && args.iter().any(|arg| {
            matches!(
                arg.split('=').next(),
                Some("--vibrational-projection" | "--electronic-degeneracy")
            )
        })
    {
        return true;
    }
    !LEGACY_TEXT_COMMANDS.contains(&command.as_str())
        || command == "hessian"
            && args
                .iter()
                .any(|a| a == "--bfield" || a.starts_with("--bfield="))
        || args.iter().any(|a| {
            a == "--format"
                || a.starts_with("--format=")
                || a == "--save"
                || a.starts_with("--save=")
        })
}

/// Parse a lattice given in Angstrom: one edge (cubic), three edges
/// (orthorhombic), nine row-major components, or three rows separated by
/// semicolons. Returns Bohr rows, matching the dedicated periodic parser.
fn driver_cell(value: &dft_rs::driver::Value) -> Result<[[f64; 3]; 3], String> {
    let numbers = |v: &dft_rs::driver::Value| -> Option<Vec<f64>> {
        match v {
            dft_rs::driver::Value::Array(items) => items.iter().map(|x| x.as_f64()).collect(),
            other => other.as_f64().map(|x| vec![x]),
        }
    };
    let a = dft_rs::constants::ANGSTROM_TO_BOHR;
    let rows: [[f64; 3]; 3] = match value {
        dft_rs::driver::Value::Array(items) if items.iter().all(|i| i.as_array().is_some()) => {
            if items.len() != 3 {
                return Err("--cell needs three lattice rows".into());
            }
            let mut rows = [[0.0; 3]; 3];
            for (r, item) in items.iter().enumerate() {
                let v = numbers(item)
                    .filter(|v| v.len() == 3)
                    .ok_or("--cell rows need three numbers")?;
                rows[r] = [v[0] * a, v[1] * a, v[2] * a];
            }
            rows
        }
        other => match numbers(other).as_deref() {
            Some([l]) => [[l * a, 0.0, 0.0], [0.0, l * a, 0.0], [0.0, 0.0, l * a]],
            Some([x, y, z]) => [[x * a, 0.0, 0.0], [0.0, y * a, 0.0], [0.0, 0.0, z * a]],
            Some([x0, x1, x2, y0, y1, y2, z0, z1, z2]) => [
                [x0 * a, x1 * a, x2 * a],
                [y0 * a, y1 * a, y2 * a],
                [z0 * a, z1 * a, z2 * a],
            ],
            _ => {
                return Err(
                    "--cell takes one edge, three edges, nine row-major components, or three \
                ';'-separated rows (Angstrom)"
                        .into(),
                )
            }
        },
    };
    if rows.iter().flatten().any(|x| !x.is_finite()) {
        return Err("--cell must be finite".into());
    }
    Ok(rows)
}

/// Lattice rows in Bohr and periodicity flags.
type ExtendedCell = ([[f64; 3]; 3], [bool; 3]);

/// Lattice rows (Bohr) and periodicity flags from the comment line of an
/// extended XYZ file: `Lattice="ax ay az bx by bz cx cy cz"` in Angstrom and
/// the optional `pbc="T T F"`.
fn extended_xyz_cell(path: &str) -> Result<Option<ExtendedCell>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let Some(comment) = text.lines().nth(1) else {
        return Ok(None);
    };
    let lower = comment.to_ascii_lowercase();
    let quoted = |key: &str| -> Option<&str> {
        let start = lower.find(&format!("{key}=\""))? + key.len() + 2;
        let end = comment[start..].find('"')? + start;
        Some(&comment[start..end])
    };
    let Some(lattice) = quoted("lattice") else {
        return Ok(None);
    };
    let values: Vec<f64> = lattice
        .split_whitespace()
        .map(|v| v.parse::<f64>())
        .collect::<Result<_, _>>()
        .map_err(|_| format!("{path}: Lattice needs nine numbers (Angstrom)"))?;
    if values.len() != 9 || values.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{path}: Lattice needs nine finite numbers (Angstrom)"
        ));
    }
    let a = dft_rs::constants::ANGSTROM_TO_BOHR;
    let rows: [[f64; 3]; 3] =
        std::array::from_fn(|r| std::array::from_fn(|c| values[3 * r + c] * a));
    let mut pbc = [true; 3];
    if let Some(flags) = quoted("pbc") {
        let parsed: Vec<bool> = flags
            .split_whitespace()
            .map(|f| match f.to_ascii_lowercase().as_str() {
                "t" | "true" | "1" => Ok(true),
                "f" | "false" | "0" => Ok(false),
                _ => Err(format!("{path}: pbc takes three T/F flags")),
            })
            .collect::<Result<_, _>>()?;
        if parsed.len() != 3 {
            return Err(format!("{path}: pbc takes three T/F flags"));
        }
        pbc = [parsed[0], parsed[1], parsed[2]];
    }
    Ok(Some((rows, pbc)))
}

/// `dft-rs <feature> [structure.xyz] [--control value ...]` through the
/// common driver. XYZ coordinates, `--cell` and an extended-XYZ `Lattice` are Angstrom; every reported
/// value is atomic units unless its key names another unit.
pub(crate) fn run_common_driver(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let feature = args[0].as_str();
    let spec = dft_rs::driver::find(feature).expect("routed feature is registered");
    let (path, rest) = match args.get(1) {
        Some(p) if !p.starts_with("--") => (Some(p.clone()), &args[2..]),
        _ => (None, &args[1..]),
    };
    let mut controls = dft_rs::driver::Controls::from_cli_args(rest)?;
    let format = controls.str("format", "text")?;
    let save = controls.opt_str("save")?;
    let log_mode = controls.str("log_mode", "console")?;
    let log_file = controls.opt_str("log_file")?;
    let mode = dft_rs::logger::LogMode::parse(&log_mode)
        .ok_or_else(|| format!("unknown log mode: {log_mode}"))?;
    if matches!(
        mode,
        dft_rs::logger::LogMode::File | dft_rs::logger::LogMode::Both
    ) && log_file.is_none()
    {
        return Err("--log-file is required with --log-mode file or both".into());
    }
    if !dft_rs::logger::is_active() {
        dft_rs::logger::init(mode, log_file.as_deref().map(std::path::Path::new))?;
    }
    if !matches!(format.as_str(), "text" | "json") {
        return Err(format!("--format must be text or json, got {format}").into());
    }
    let mut system = dft_rs::driver::System::default();
    let uses_geometry = spec.space != dft_rs::driver::registry::Space::Data
        || (feature == "no-pair-projector" && path.is_some());
    if uses_geometry {
        let path =
            path.ok_or_else(|| format!("{feature} needs a structure file (XYZ, Angstrom)"))?;
        let molecule = Molecule::from_xyz_file(&path)?;
        system.numbers = molecule.atoms.iter().map(|a| a.z).collect();
        system.positions = molecule
            .atoms
            .iter()
            .map(|a| a.position.to_array())
            .collect();
        if let Some(cell) = controls.take("cell").cloned() {
            system.cell = Some(driver_cell(&cell)?);
            system.pbc = [true; 3];
        } else if spec.space != dft_rs::driver::registry::Space::Molecule {
            // Without --cell an extended XYZ comment line supplies the lattice.
            if let Some((rows, pbc)) = extended_xyz_cell(&path)? {
                system.cell = Some(rows);
                system.pbc = pbc;
            }
        }
        if let Some(pbc) = controls.opt_usize_list("pbc")? {
            if pbc.len() != 3 || pbc.iter().any(|&p| p > 1) {
                return Err("--pbc takes three 0/1 flags".into());
            }
            system.pbc = [pbc[0] == 1, pbc[1] == 1, pbc[2] == 1];
        }
    } else if let Some(path) = path {
        return Err(format!("{feature} takes no structure file (got {path})").into());
    }
    // Rebuild once after geometry extraction. CLI output controls and geometry
    // already represented by System must not rely on stale consumption flags.
    let mut entries = Vec::new();
    if let dft_rs::driver::Value::Object(all) = controls.to_value() {
        for (key, value) in all {
            if !(["format", "save", "log_mode", "log_file"].contains(&key.as_str())
                || uses_geometry && matches!(key.as_str(), "cell" | "pbc"))
            {
                entries.push((key, value));
            }
        }
    }
    controls = dft_rs::driver::Controls::from_value(&dft_rs::driver::Value::Object(entries))?;
    let report = dft_rs::driver::run(feature, &system, &controls)?;
    if let Some(path) = save {
        std::fs::write(&path, report.to_json_pretty())?;
    }
    match format.as_str() {
        "json" => print!("{}", report.to_json_pretty()),
        _ => print!("{}", dft_rs::driver::report_text(&report)),
    }
    Ok(())
}

/// Extract logging settings while retaining existing command-specific defaults.
fn cli_log_options(args: Vec<String>) -> dft_rs::Result<(dft_rs::logger::LogOptions, Vec<String>)> {
    use dft_rs::logger::{LogLevel, LogMode, LogOptions};
    let mut logging = std::collections::BTreeMap::new();
    let mut retained = Vec::new();
    let mut arguments = args.into_iter();
    while let Some(argument) = arguments.next() {
        let (name, inline) = argument
            .split_once('=')
            .map_or((argument.as_str(), None), |(n, v)| (n, Some(v)));
        let name = name.replace('_', "-");
        if matches!(name.as_str(), "--log-file" | "--log-mode" | "--log-level") {
            let value = inline
                .map(str::to_string)
                .or_else(|| arguments.next())
                .filter(|value| !value.starts_with("--") && !value.is_empty())
                .ok_or_else(|| {
                    dft_rs::DftError::InvalidInput(format!("{name} requires a value"))
                })?;
            if logging.insert(name, value).is_some() {
                return Err(dft_rs::DftError::InvalidInput(
                    "duplicate logging option".into(),
                ));
            }
        } else {
            retained.push(argument);
        }
    }
    let explicit = !logging.is_empty();
    if !explicit && dft_rs::logger::is_active() {
        return Ok((LogOptions::default(), retained));
    }
    let path = logging.remove("--log-file");
    let mode = logging.remove("--log-mode");
    let level = logging.remove("--log-level");
    let requested_level = level
        .as_deref()
        .map(LogLevel::parse)
        .unwrap_or(Some(LogLevel::Info))
        .ok_or_else(|| {
            dft_rs::DftError::InvalidInput("log_level must be error, warn, info or debug".into())
        })?;
    let probe = dft_rs::resources::cli_limits(retained.clone())
        .ok()
        .map(|(_, args)| args);
    let common = probe.as_ref().is_some_and(|args| uses_common_driver(args));
    let legacy = probe
        .and_then(|args| parse_args_from(args).ok())
        .filter(|cli| !cli.list_functionals && !cli.list_solvents);
    let options = if path.is_some() {
        LogOptions::from_strings(path, mode, level)?
    } else if let Some(mode) = mode {
        let mode = LogMode::parse(&mode).ok_or_else(|| {
            dft_rs::DftError::InvalidInput("log_mode must be console, file, both or none".into())
        })?;
        let path = if matches!(mode, LogMode::File | LogMode::Both) {
            legacy
                .as_ref()
                .map(|cli| std::path::Path::new(&cli.path).with_extension("dft.log"))
        } else {
            None
        };
        LogOptions::new(mode, path, requested_level)?
    } else if common {
        LogOptions::new(LogMode::Console, None, requested_level)?
    } else if let Some(cli) = legacy {
        LogOptions::new(
            LogMode::Both,
            Some(std::path::Path::new(&cli.path).with_extension("dft.log")),
            requested_level,
        )?
    } else if explicit {
        LogOptions::new(LogMode::None, None, requested_level)?
    } else {
        LogOptions::default()
    };
    Ok((options, retained))
}

/// Run arguments without the program name and return the process exit status.
/// Both executables and Python cli_main share this entry point.
pub fn run_arguments(args: Vec<String>) -> i32 {
    let (logging, args) = match cli_log_options(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            dft_rs::logger::record_error(&error.to_string());
            eprintln!("dft-rs: {error}");
            return 1;
        }
    };
    match logging.install(move || run_arguments_with_resources(args)) {
        Ok(status) => status,
        Err(error) => {
            dft_rs::logger::record_error(&error.to_string());
            eprintln!("dft-rs: {error}");
            1
        }
    }
}

fn run_arguments_with_resources(args: Vec<String>) -> i32 {
    let (limits, args) = match dft_rs::resources::cli_limits(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            dft_rs::logger::record_error(&error.to_string());
            eprintln!("dft-rs: {error}");
            return 1;
        }
    };
    match limits.install(move || run_arguments_inner(args)) {
        Ok(status) => status,
        Err(error) => {
            dft_rs::logger::record_error(&error.to_string());
            eprintln!("dft-rs: {error}");
            1
        }
    }
}

fn run_arguments_inner(args: Vec<String>) -> i32 {
    if matches!(
        args.first().map(String::as_str),
        Some("--version") | Some("-V")
    ) {
        println!("dft-rs {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }
    if uses_common_driver(&args) {
        dft_rs::ensure_parallelism();
        return match run_common_driver(&args) {
            Ok(()) => 0,
            Err(err) => {
                dft_rs::logger::record_error(&err.to_string());
                eprintln!("dft-rs: {err}");
                1
            }
        };
    }
    match parse_args_from(args) {
        Ok(cli) => match run(cli) {
            Ok(()) => 0,
            Err(err) => {
                dft_rs::logger::record_error(&err.to_string());
                eprintln!("dft-rs: {err}");
                1
            }
        },
        Err(message) if message.is_empty() => {
            print!("{}", cli_usage());
            0
        }
        Err(message) => {
            dft_rs::logger::record_error(&message);
            eprintln!("dft-rs: {message}");
            1
        }
    }
}

fn print_orbital_gradients(v: &dft_rs::response::OrbitalEnergyGradients) {
    println!("coordinate_ids {:?}", v.coordinates);
    println!(
        "degeneracy_tolerance {:.12e} translation_residual {:?}",
        v.degeneracy_tolerance, v.translation_residual
    );
    for spin in 0..2 {
        for (i, energy) in v.orbital_energies[spin].iter().enumerate() {
            println!("orbital_energy {spin} {i} {energy:.12e}");
            if let Some(g) = &v.gradients[spin][i] {
                println!("orbital_gradient {spin} {i} {:?}", g);
            } else {
                println!("orbital_gradient {spin} {i} undefined");
            }
        }
    }
    for (i, cluster) in v.clusters.iter().enumerate() {
        println!(
            "orbital_cluster {i} spin {} orbitals {:?} energies {:?} energy_spread \
            {:.12e} commutator_defect {:.12e}",
            cluster.spin,
            cluster.orbitals,
            cluster.energies,
            cluster.energy_spread,
            cluster.commutator_defect
        );
        for (j, matrix) in cluster.derivative_matrices.iter().enumerate() {
            println!(
                "orbital_cluster_matrix {i} {} {:?}",
                v.coordinates[j],
                matrix.as_slice()
            );
        }
    }
    println!(
        "hermiticity_defects {:?} projector_defects {:?} stationarity_defects \
        {:?} metric_defects {:?} particle_number_coefficients {:?}",
        v.hermiticity_defects,
        v.projector_defects,
        v.stationarity_defects,
        v.metric_defects,
        v.particle_number_coefficients
    );
    println!(
        "reference_energy_errors {:?} reference_fock_errors {:?}",
        v.reference_energy_errors, v.reference_fock_errors
    );
}

#[cfg(test)]
mod cell_input_tests {
    use super::*;

    #[test]
    fn legacy_response_vv10_resolution_is_independent_and_validated() {
        let parse = |value: &str| {
            parse_args_from(vec![
                "dynamic-beta".into(),
                "molecule.xyz".into(),
                "--grid-level".into(),
                "1".into(),
                "--vv10-grid-level".into(),
                value.into(),
            ])
        };
        let options = parse("5").unwrap();
        assert_eq!(options.grid_level, 1);
        assert_eq!(options.vv10_grid_level, 5);
        for value in ["10", "-1", "1.5", "true", "18446744073709551616"] {
            assert!(parse(value).is_err(), "invalid VV10 grid level {value}");
        }
    }

    #[test]
    fn harmonic_text_requests_preserve_reference_and_correlation_controls() {
        for args in [
            vec!["frequencies", "molecule.xyz", "--xc", "b2plyp"],
            vec!["frequencies", "molecule.xyz", "--pt2-opposite-spin=0.5"],
            vec![
                "frequencies",
                "molecule.xyz",
                "--vibrational-projection=cartesian",
            ],
            vec![
                "frequencies",
                "molecule.xyz",
                "--electronic-degeneracy",
                "2",
            ],
            vec!["frequencies", "molecule.xyz", "--reference", "ghf"],
        ] {
            let args = args.into_iter().map(str::to_owned).collect::<Vec<_>>();
            assert!(uses_common_driver(&args), "{args:?}");
        }
    }

    #[test]
    fn common_cell_parser_preserves_flat_and_nested_triclinic_rows() {
        use dft_rs::driver::Value;
        let entries = [0.3, 3.1, 2.8, 2.9, 0.2, 3.2, 3.0, 2.7, 0.1];
        let flat = Value::Array(entries.iter().map(|&v| Value::Float(v)).collect());
        let nested = Value::Array(
            entries
                .chunks(3)
                .map(|r| Value::Array(r.iter().map(|&v| Value::Float(v)).collect()))
                .collect(),
        );
        assert_eq!(driver_cell(&flat).unwrap(), driver_cell(&nested).unwrap());
        let expected = entries.map(|v| v * dft_rs::constants::ANGSTROM_TO_BOHR);
        assert_eq!(
            driver_cell(&flat)
                .unwrap()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
            expected
        );
    }
}
