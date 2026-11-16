# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
"""Thin, explicit wrapper over the compiled `dft_rs_core` extension.

Everything here is in **atomic units** — Hartree, Bohr, Hartree/Bohr. That
is the whole computational core's convention, and this layer does not
convert. The one place units change is `dft_rs.DftRs`, the ASE calculator,
which converts to eV/Å at its boundary because ASE requires it.

Keeping the native surface in its own module means the ASE adapter can be
read as an adapter: it converts units and caches, and does nothing else.
"""

from __future__ import annotations

import functools
import inspect
import operator
import os
from typing import Any, Optional, Sequence, Union
from .result_units import attach_unit_labels

try:
    from . import dft_rs_core as _core
except ImportError as exc:  # pragma: no cover - build/install guidance
    raise ImportError(
        "the compiled dft_rs.dft_rs_core extension is not importable. "
        "Install a wheel, or run `maturin develop --release` in the source tree."
    ) from exc


__all__ = [
    "calculate",
    "hessian",
    "vibrations",
    "raman",
    "optimize",
    "saddle",
    "polarizability",
    "dynamic_polarizability",
    "stability",
    "stabilize",
    "dynamic_electric_tensor",
    "dynamic_electric_process",
    "dynamic_beta",
    "dynamic_gamma",
    "electric_field_taylor",
    "mixed_electric_field_response",
    "static_electric_tensor",
    "beta",
    "gamma",
    "nuclear_taylor",
    "orbital_energy_gradients",
    "frontier_gap_gradient",
    "mixed_nuclear_response",
    "molecular_taylor",
    "mixed_molecular_response",
    "static_spectroscopic_derivative",
    "analytic_force_constants",
    "anharmonic",
    "nmr",
    "spin_spin_zfs",
    "magnetic_response",
    "tddft",
    "sapt",
    "basis_sets",
    "periodic",
    "periodic_scf",
    "periodic_density_response",
    "periodic_field_response",
    "periodic_spin_response",
    "periodic_dielectric",
    "periodic_thermal_response",
    "periodic_capabilities",
    "periodic_electrostatics",
    "periodic_basis",
    "periodic_gaussian_integral",
    "periodic_coulomb_operators",
    "periodic_xc_operators",
    "dimensional_scf",
    "dimensional_field_response",
    "periodic_field_operators",
    "monkhorst_pack",
    "BOHR_PER_ANGSTROM",
    "HARTREE_IN_EV",
]

#: Unit bridges, exposed so callers converting at their own boundary do not
#: have to hard-code them. These are the SAME values the Rust core uses
#: (`dft_rs::constants`), which matters: a caller who converts a geometry
#: with a slightly different Bohr radius than the XYZ reader uses would get
#: two marginally different molecules from what looks like one input.
#:
#: The ASE adapter uses the same package CODATA 2018 factors, so geometry
#: and high-order response are independent of ASE's default CODATA edition.
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_IN_EV = 27.211386245988


def _atomic_numbers(values):
    try:
        numbers = tuple(operator.index(value) for value in values)
    except TypeError:
        raise ValueError("atomic numbers must be integers from 1 through 118") from None
    if any(number < 1 or number > 118 for number in numbers):
        raise ValueError("atomic numbers must be integers from 1 through 118")
    return numbers


def _periodicity(values):
    flags = tuple(values)
    if len(flags) != 3 or any(flag not in (False, True) for flag in flags):
        raise ValueError("periodicity requires three boolean or numeric 0/1 flags")
    return tuple(bool(flag) for flag in flags)


class Report(dict):
    """A driver report: a dictionary whose entries are also attributes.

    ``report["energy"]`` and ``report.energy`` are the same value. Reports
    are in atomic units unless a key names another unit; ``report.units``
    names the unit of every entry.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"this report has no entry {name!r}") from None

    def __dir__(self):
        return sorted(set(super().__dir__()) | {key for key in self if isinstance(key, str)})

    def unit(self, key):
        """The unit label of an entry, or ``None`` when it has none: from
        ``units`` (driver reports) or ``unit_labels`` (dedicated functions),
        by the name or the dotted path of the entry."""
        for name in ("units", "unit_labels"):
            labels = self.get(name)
            if isinstance(labels, dict) and key in labels:
                return labels[key]
        return None

    def require_converged(self):
        """Return the report itself, or raise when it is not converged."""
        if self.get("converged") is False:
            raise RuntimeError(
                f"the {self.get('feature', 'calculation')} report is not converged; see its warnings"
            )
        return self


def calculate(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    gradient: bool = False,
    wavefunction: bool = False,
    nbo: bool = False,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    field: Optional[Sequence[float]] = None,
    solvent: Optional[str] = None,
    eps: Optional[float] = None,
    pseudo_sub: Optional[str] = None,
    unrestricted: bool = False,
    stabilize: bool = False,
    grid_preset: Optional[str] = None,
    grid_angular_scheme: Optional[str] = None,
    grid_weight_cutoff: Optional[float] = None,
) -> dict[str, Any]:
    """Run one SCF and return its results.

    Parameters
    ----------
    numbers
        Atomic numbers, 1 through 118. A bare ``0`` is rejected because it
        does not identify which element's basis should be placed at a ghost
        centre.
    positions
        Cartesian coordinates in **Bohr**, shape ``(natoms, 3)``.
    basis
        Bundled family name (see :func:`basis_sets`), a path to a Gaussian94
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
    """
    return _core.calculate(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        gradient=gradient,
        wavefunction=wavefunction,
        nbo=nbo,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        field=None if field is None else list(field),
        solvent=solvent,
        eps=eps,
        pseudo_sub=pseudo_sub,
        unrestricted=unrestricted,
        stabilize=stabilize,
        grid_preset=grid_preset,
        grid_angular_scheme=grid_angular_scheme,
        grid_weight_cutoff=grid_weight_cutoff,
    )


def _require_analytic_hessian(report):
    """Reject contradictory derivative metadata before exposing an analytic tensor."""
    if report.get("provenance") != "analytic" or report.get("method", "analytic") != "analytic":
        raise RuntimeError("the analytic Hessian driver returned missing or contradictory analytic provenance")


def hessian(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    reference: Optional[str] = None,
    **controls: Any,
) -> dict[str, Any]:
    """Cartesian Hessian in Hartree/Bohr^2.

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
    """
    if reference is not None or controls:
        settings = dict(basis=basis, ecp=ecp, xc=xc, charge=charge,
                        multiplicity=multiplicity, grid_level=grid_level,
                        max_iter=max_iter, dispersion=dispersion, density_fit=density_fit,
                        conv_tol=conv_tol, conv_tol_grad=conv_tol_grad, conv_tol_dp=conv_tol_dp,
                        incore_mb=incore_mb, vv10_grid_level=vv10_grid_level)
        settings.update(controls)
        if reference is not None:
            settings["reference"] = reference
        report = run_feature("hessian", numbers, positions, **settings)
        _require_analytic_hessian(report)
        report["method"] = "analytic"
        return report
    return _core.hessian(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
    )


def vibrations(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    symmetry_number: int = 1,
) -> dict[str, Any]:
    """Harmonic frequencies, IR intensities, and ideal-gas RRHO thermochemistry.

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
    """
    return _core.vibrations(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        temperature=temperature,
        pressure=pressure,
        symmetry_number=symmetry_number,
    )


def raman(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    hessian_step: float = 5e-3,
    raman_step: float = 5e-3,
) -> dict[str, Any]:
    """Frequencies, IR intensities, and Raman activities.

    Normal modes use the analytic Hessian whenever the complete requested
    method supports it; ``hessian_method`` reports the selected path. Raman
    activities are in Angstrom^4/amu.

    Returns
    -------
    dict
        ``frequencies`` (cm^-1), ``ir_intensities`` (km/mol),
        ``raman_activities`` (Angstrom^4/amu), ``n_imaginary``,
        ``n_projected``, and ``hessian_method``.
    """
    return _core.raman(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        hessian_step=hessian_step,
        raman_step=raman_step,
    )


def optimize(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    max_steps: int = 200,
    gtol: float = 3e-4,
    step_max: float = 0.3,
    history: int = 8,
    trajectory: Optional[str] = None,
) -> dict[str, Any]:
    """Optimize a molecular geometry with analytic-gradient L-BFGS.

    Returns ``positions`` (Bohr), ``energy`` (Hartree), ``converged``,
    ``iterations``, and a ``trajectory`` list. Each trajectory item contains
    ``step``, ``positions``, ``energy``, and ``grad_max`` (Hartree/Bohr).
    When ``trajectory`` is a path, accepted frames are flushed as XYZ while
    the walk proceeds.
    """
    return _core.optimize(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        max_steps=max_steps,
        gtol=gtol,
        step_max=step_max,
        history=history,
        trajectory=trajectory,
    )


def saddle(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    method: str = "prfo",
    max_steps: int = 50,
    gtol: float = 3e-4,
    step_max: float = 0.2,
    hessian_step: float = 5e-3,
    trajectory: Optional[str] = None,
) -> dict[str, Any]:
    """Locate a first-order saddle with analytic-first Hessians.

    ``method="prfo"`` selects partitioned RFO. ``method="rsirfo"`` selects
    the coupled Householder image function with an intrinsic restricted step.
    ``step_max`` is the Euclidean trust radius for RS-I-RFO; its generalized
    RFO metric is solved so the step itself satisfies the restriction.

    Returns ``method``, final ``positions`` (Bohr), ``energy`` (Hartree),
    ``converged``, ``iterations``, ``n_negative``, and the same trajectory
    schema as :func:`optimize`.

    Raises
    ------
    ValueError
        For an unknown method or invalid numeric controls.
    NotImplementedError
        When a required analytic component has no numerical fallback.
    """
    return _core.saddle(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        method=method,
        max_steps=max_steps,
        gtol=gtol,
        step_max=step_max,
        hessian_step=hessian_step,
        trajectory=trajectory,
    )


def polarizability(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
) -> dict[str, Any]:
    """Static dipole polarizability (a.u., 3x3) from electric-field CPKS.

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
    """
    return _core.polarizability(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
    )


def dynamic_polarizability(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *,
    omega: float = 0.0, damping: float = 0.0, imaginary: bool = False,
    basis: str = "sto-3g", ecp: Optional[str] = None, xc: str = "hf",
    charge: int = 0, multiplicity: int = 1, grid_level: int = 3,
    max_iter: int = 150, dispersion: str = "auto", density_fit: Optional[str] = None,
    conv_tol: float = 1e-9, conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8, incore_mb: int = 2048, vv10_grid_level: int = 3,
    response_tol: float = 1e-10, response_max_iter: int = 500,
    solvent: Optional[str] = None, eps: Optional[float] = None,
    reference: Optional[str] = None, unrestricted: bool = False,
    scf_damping: Optional[float] = None, **options,
) -> dict[str, Any]:
    """Paired Casida dipole response with frequencies in Hartree.

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
    """
    resolved = []
    for legacy, value, default, aliases in (
        ("omega", omega, 0.0, ("frequency", "response_omega")),
        ("damping", damping, 0.0, ("frequency_damping", "response_damping")),
        ("imaginary", imaginary, False, ("imaginary_frequency",)),
    ):
        supplied = [alias for alias in aliases if alias in options]
        if supplied and (len(supplied) > 1 or value != default):
            raise ValueError("use either " + " or ".join((legacy, *aliases)))
        resolved.append(options.pop(supplied[0]) if supplied else value)
    omega, damping, imaginary = resolved
    if "response_tolerance" in options:
        if response_tol != 1e-10:
            raise ValueError("use either response_tol or response_tolerance")
        response_tol = options.pop("response_tolerance")
    if "response_max_iterations" in options:
        if response_max_iter != 500:
            raise ValueError("use either response_max_iter or response_max_iterations")
        response_max_iter = options.pop("response_max_iterations")
    controls = dict(basis=basis, ecp=ecp, xc=xc, charge=charge,
                    multiplicity=multiplicity, grid_level=grid_level,
                    max_iter=max_iter, dispersion=dispersion, density_fit=density_fit,
                    conv_tol=conv_tol, conv_tol_grad=conv_tol_grad, conv_tol_dp=conv_tol_dp,
                    incore_mb=incore_mb, vv10_grid_level=vv10_grid_level,
                    response_tol=response_tol, response_max_iter=response_max_iter,
                    solvent=solvent, eps=eps, unrestricted=unrestricted,
                    frequency=omega, frequency_damping=damping, imaginary=imaginary)
    if reference is not None:
        controls["reference"] = reference
    controls.update(options)
    if scf_damping is not None:
        controls["damping"] = scf_damping
    report = run_feature("dynamic-polarizability", numbers, positions, **controls)
    report["unit_labels"] = report["units"]
    report["units"] = "bohr^3"
    report["frequency_units"] = "hartree"
    report.update(omega=omega, damping=damping, imaginary=imaginary)
    report["unit_labels"].update(omega="hartree", damping="hartree", imaginary="dimensionless")
    return report


def stability(numbers, positions, *, n_roots=3, roots=None, channels=None,
              tolerance=1e-7, max_iterations=200, max_subspace=60, block_buffer=3,
              resolution=None, verify_lowest=True, verify_steps=24,
              follow_unstable=False, max_descents=3, step_angle=0.3,
              min_energy_drop=1e-8, unrestricted=False, **options):
    """Analytic collinear real/imaginary orbital stability in Hartree.

    Restricted references include singlet/triplet blocks; unrestricted
    references include independent spin variations. Numerical convergence
    and physical stability are separate report fields. Each mode includes
    its normalized generator, actual energy curvature and fresh residual.
    Accepts the dedicated stability binding's SCF/response options, without
    frequency or electric-process controls. The dynamic-only ``scf_damping``
    alias is not a keyword of this legacy binding.
    """
    if roots is not None:
        if n_roots != 3:
            raise ValueError("roots cannot be combined with n_roots")
        n_roots = roots
    if isinstance(channels, str):
        channels = [c.strip() for c in channels.split(",")]
    return _core.dynamic_polarizability(
        list(numbers), [list(p) for p in positions], stability_roots=n_roots,
        response_tol=options.pop("response_tol", tolerance),
        response_max_iter=options.pop("response_max_iter", max_iterations),
        stability_channels=None if channels is None else list(channels),
        stability_resolution=resolution, stability_max_subspace=max_subspace,
        stability_block_buffer=block_buffer, stability_verify_lowest=verify_lowest,
        stability_verify_steps=verify_steps, follow_unstable=follow_unstable,
        max_descents=max_descents, step_angle=step_angle, min_energy_drop=min_energy_drop,
        unrestricted=unrestricted, **options,
    )


def stabilize(numbers, positions, **options):
    """Follow internal instabilities and return the final reference and analysis."""
    return stability(numbers, positions, follow_unstable=True, **options)


def dynamic_electric_tensor(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *,
    input_frequencies: Optional[Sequence[Sequence[float]]] = None, imaginary: bool = False,
    process: Optional[str] = None, frequencies: Optional[Sequence[float]] = None,
    damping: float = 0.0,
    **options,
) -> dict[str, Any]:
    """Actual complex alpha/beta/gamma derivatives at one/two/three input legs.

    Each input is [omega, damping] in Hartree; imaginary-axis inputs instead
    use [positive frequency, 0]. Negative real omega supports frequency mixing.
    Output frequency is the complex sum. Flat tensor_real/tensor_imag have
    shape [3] repeated number_of_inputs+1, observable axis first. No output
    frequency or damping is independently substituted for a subset sum.
    Accepts the same SCF/response options as dynamic_polarizability.
    """
    return _core.dynamic_polarizability(
        list(numbers), [list(p) for p in positions],
        input_frequencies=None if input_frequencies is None else [list(f) for f in input_frequencies],
        imaginary=imaginary, process=process,
        frequencies=None if frequencies is None else list(frequencies), damping=damping, **options,
    )


def dynamic_electric_process(numbers, positions, *, process, frequencies, damping=0.0, **options):
    """Select a physical nonlinear optical frequency preset, Hartree inputs.

    Examples: process="shg", frequencies=[0.05]; process="optical-kerr",
    frequencies=[0.05]. The report preserves the ordered frequency legs.
    """
    return dynamic_electric_tensor(numbers, positions, process=process,
                                   frequencies=frequencies, damping=damping, **options)


def dynamic_beta(numbers, positions, *, input_frequencies=None, process=None,
                 frequencies=None, damping=0.0, **options):
    """Full 27-component beta, using explicit legs or an optical preset.

    Presets: shg, sfg, dfg, optical-rectification and electro-optic.
    Supply process and nonnegative frequencies (Hartree), with damping in
    Hartree, or two input_frequencies=[omega, damping] pairs. The output
    frequency is the sum of the incoming legs.
    """
    if input_frequencies is not None and len(input_frequencies) != 2:
        raise ValueError("dynamic_beta requires two input frequency pairs")
    return dynamic_electric_tensor(numbers, positions, input_frequencies=input_frequencies,
        process=process, frequencies=frequencies, damping=damping, tensor_order=2, **options)


def dynamic_gamma(numbers, positions, *, input_frequencies=None, process=None,
                  frequencies=None, damping=0.0, **options):
    """Full 81-component gamma, using explicit legs or an optical preset.

    Presets: thg, efish, field-induced-rectification,
    two-photon-absorption, optical-kerr, dc-kerr and four-wave-mixing.
    The first six take one nonnegative optical frequency; four-wave-mixing
    takes three. Frequencies and damping use Hartree. Alternatively supply
    three input_frequencies=[omega, damping] pairs. Optical legs retain
    positive damping, including negative frequencies; DC legs are static.
    The tensor is the ordered cubic dipole susceptibility, without optical
    intensity, refractive-index, polarization or macroscopic conversion.
    """
    if input_frequencies is not None and len(input_frequencies) != 3:
        raise ValueError("dynamic_gamma requires three input frequency pairs")
    return dynamic_electric_tensor(numbers, positions, input_frequencies=input_frequencies,
        process=process, frequencies=frequencies, damping=damping, tensor_order=3, **options)


def _electric_compat_report(report, summary):
    """Retain the historical summary and authoritative per-entry unit labels."""
    result = dict(report)
    result["unit_labels"] = result["units"]
    if summary is None:
        summary = result.get("response_units")
        if summary is None:
            summary = result["unit_labels"]["tensor"]
    result["units"] = summary
    return Report(result)


def _electric_driver(feature, numbers, positions, options, controls, summary):
    """Adapt legacy option names without building a second SCF reference."""
    for old, canonical, default in [
        ("response_tol", "response_tolerance", 1e-10),
        ("response_max_iter", "response_max_iterations", 500),
    ]:
        value = options.pop(old)
        if canonical in controls:
            if value != default:
                raise ValueError(f"use either {old} or {canonical}")
        else:
            options[canonical] = value
    options = {key: value for key, value in options.items() if value is not None}
    options.update(controls)
    report = run_feature(feature, numbers, positions, **options)
    return _electric_compat_report(report, summary)


def electric_field_taylor(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], direction: Sequence[float], *,
    order: int = 3, basis: str = "sto-3g", ecp: Optional[str] = None, xc: str = "hf",
    charge: int = 0, multiplicity: int = 1, grid_level: int = 3,
    max_iter: int = 150, dispersion: str = "auto", density_fit: Optional[str] = None,
    conv_tol: float = 1e-9, conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8, incore_mb: int = 2048, vv10_grid_level: int = 3,
    response_tol: float = 1e-10, response_max_iter: int = 500,
    solvent: Optional[str] = None, eps: Optional[float] = None,
    reference_field: Optional[Sequence[float]] = None,
    **controls,
) -> dict[str, Any]:
    """Analytic dipole series for E(t)=reference_field+t*direction.

    Positions, fields and dipoles use atomic units. Coefficient n is the
    nth directional derivative divided by n!, so beta and gamma along the
    supplied direction are 2*coefficient[2] and 6*coefficient[3].
    """
    options = locals().copy()
    for name in ("numbers", "positions", "controls"):
        options.pop(name)
    options["field_direction"] = options.pop("direction")
    options["response_order"] = options.pop("order")
    return _electric_driver("field-taylor", numbers, positions, options, controls,
                            "atomic units; coefficient n = derivative n / n!")


def mixed_electric_field_response(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    directions: Sequence[Sequence[float]], *,
    basis: str = "sto-3g", ecp: Optional[str] = None, xc: str = "hf",
    charge: int = 0, multiplicity: int = 1, grid_level: int = 3,
    max_iter: int = 150, dispersion: str = "auto", density_fit: Optional[str] = None,
    conv_tol: float = 1e-9, conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8, incore_mb: int = 2048, vv10_grid_level: int = 3,
    response_tol: float = 1e-10, response_max_iter: int = 500,
    solvent: Optional[str] = None, eps: Optional[float] = None,
    reference_field: Optional[Sequence[float]] = None,
    **controls,
) -> dict[str, Any]:
    """Analytic mixed dipole derivative, one field direction per derivative.

    Repeated directions are allowed. Two directions select beta, three
    select gamma, with no factorial normalization in the returned vector.
    All quantities and geometry use atomic units.
    """
    options = locals().copy()
    for name in ("numbers", "positions", "controls"):
        options.pop(name)
    options["field_directions"] = options.pop("directions")
    return _electric_driver("field-mixed", numbers, positions, options, controls,
                            "atomic units; unnormalized mixed derivative")


def static_electric_tensor(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *, order: int = 2,
    assembly: str = "projector",
    basis: str = "sto-3g", ecp: Optional[str] = None, xc: str = "hf",
    charge: int = 0, multiplicity: int = 1, grid_level: int = 3,
    max_iter: int = 150, dispersion: str = "auto", density_fit: Optional[str] = None,
    conv_tol: float = 1e-9, conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8, incore_mb: int = 2048, vv10_grid_level: int = 3,
    response_tol: float = 1e-10, response_max_iter: int = 500,
    solvent: Optional[str] = None, eps: Optional[float] = None,
    reference_field: Optional[Sequence[float]] = None,
    **controls,
) -> dict[str, Any]:
    """Complete analytic static beta (order=2) or gamma (order=3), atomic units.

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
    """
    options = locals().copy()
    for name in ("numbers", "positions", "controls"):
        options.pop(name)
    if order not in (2, 3):
        raise ValueError("static_electric_tensor order is 2 (beta) or 3 (gamma)")
    options.pop("order")
    options["tensor_assembly"] = options.pop("assembly")
    return _electric_driver("beta" if order == 2 else "gamma", numbers, positions,
                            options, controls,
                            f"atomic units; e^{order+1} Bohr^{order+1}/Hartree^{order}")


def beta(numbers: Sequence[int], positions: Sequence[Sequence[float]], **options) -> dict[str, Any]:
    """Full static first hyperpolarizability; options match static_electric_tensor.

    Integer RKS/UKS double hybrids use complete orbital/PT2 energy derivatives,
    including reference_field bias, with parent/target and separate SS/OS tensors.
    Integer GKS accepts a common PT2 weight and returns full-spin parent,
    target and pair tensors through the same driver.
    """
    return static_electric_tensor(numbers, positions, order=2, **options)


def gamma(numbers: Sequence[int], positions: Sequence[Sequence[float]], **options) -> dict[str, Any]:
    """Full static second hyperpolarizability; options match static_electric_tensor.

    Integer RKS/UKS double hybrids use complete fourth-order orbital/PT2 energy
    paths, including reference_field bias, and return all 81 Cartesian entries.
    Integer GKS accepts a common PT2 weight and returns full-spin parent,
    target and pair tensors through the same driver.
    """
    return static_electric_tensor(numbers, positions, order=3, **options)


def nmr(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    gauge: str = "giao",
    origin: Optional[Sequence[float]] = None,
    max_iter: int = 150,
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    response_tol: float = 1e-10,
    response_max_iter: int = 500,
    response_restart: int = 40,
    unrestricted: bool = False,
) -> dict[str, Any]:
    """Return closed/open collinear orbital GIAO shielding tensors in ppm.

    ``gauge="common_origin"`` is available for explicit origin studies;
    production calculations should use the default London-orbital path.

    The result contains ``gauge``, ``origin`` (Bohr), ``energy`` (Hartree)
    and ``shieldings``. Each nucleus has ``tensor``, ``diamagnetic`` and
    ``paramagnetic`` 3x3 arrays plus their isotropic values, all in ppm.
    ECP or nonlocal-functional combinations with missing magnetic derivatives raise
    ``NotImplementedError`` rather than returning an incomplete tensor.
    """
    return _core.nmr(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        gauge=gauge,
        origin=None if origin is None else list(origin),
        max_iter=max_iter,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        response_tol=response_tol, response_max_iter=response_max_iter,
        response_restart=response_restart, unrestricted=unrestricted,
    )


def magnetic_response(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    gauge: str = "giao",
    origin: Optional[Sequence[float]] = None,
    max_iter: int = 150,
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
    response_tol: float = 1e-10,
    response_max_iter: int = 500,
    response_restart: int = 40,
    unrestricted: bool = False,
) -> dict[str, Any]:
    """Return spin orbital dP/dB = i*density_hat with raw relative residuals.

    Arrays have axes [spin, external B, spherical AO, spherical AO].
    Fields use reduced atomic units in (p+A)^2/2; origin uses Bohr.
    The report includes overlap and fixed-density Fock derivatives, and
    the zero-field overlap/Fock/density/orbitals for independent verification.
    Spin Zeeman, hyperfine and thermal paramagnetic shifts are separate.
    """
    return _core.nmr(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        gauge=gauge,
        origin=None if origin is None else list(origin),
        max_iter=max_iter,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
        density_response=True,
        response_tol=response_tol, response_max_iter=response_max_iter,
        response_restart=response_restart, unrestricted=unrestricted,
    )


def tddft(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge: int = 0,
    multiplicity: int = 1,
    grid_level: int = 3,
    nstates: int = 5,
    method: str = "tda",
    spin: str = "auto",
    max_iter: int = 150,
    dispersion: str = "auto",
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
    vv10_grid_level: int = 3,
) -> dict[str, Any]:
    """Excitation energies, oscillator strengths, and rotatory strengths (ECD).

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
    """
    return _core.tddft(
        list(numbers),
        [list(p) for p in positions],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        grid_level=grid_level,
        nstates=nstates,
        method=method,
        spin=spin,
        max_iter=max_iter,
        dispersion=dispersion,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level,
    )


def sapt(
    numbers_a: Sequence[int],
    positions_a: Sequence[Sequence[float]],
    numbers_b: Sequence[int],
    positions_b: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "hf",
    charge_a: int = 0,
    multiplicity_a: int = 1,
    charge_b: int = 0,
    multiplicity_b: int = 1,
    grid_level: int = 3,
    max_iter: int = 150,
    density_fit: Optional[str] = None,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    incore_mb: int = 2048,
) -> dict[str, Any]:
    """Default molecular interaction-energy response decomposition for HF or KS
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
    """
    return _core.sapt(
        list(numbers_a),
        [list(p) for p in positions_a],
        list(numbers_b),
        [list(p) for p in positions_b],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge_a=charge_a,
        multiplicity_a=multiplicity_a,
        charge_b=charge_b,
        multiplicity_b=multiplicity_b,
        grid_level=grid_level,
        max_iter=max_iter,
        density_fit=density_fit,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        incore_mb=incore_mb,
    )


def nuclear_taylor(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    direction: Sequence[Sequence[float]], *, order: int = 4,
    basis: str = "sto-3g", ecp: Optional[str] = None, xc: str = "hf",
    charge: int = 0, multiplicity: int = 1, grid_level: int = 3,
    max_iter: int = 150, dispersion: str = "none", density_fit: Optional[str] = None,
    conv_tol: float = 1e-9, conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8, incore_mb: int = 2048, vv10_grid_level: int = 3,
    response_tol: float = 1e-10, response_max_iter: int = 500,
    reference_field: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Analytic energy coefficients along R(t) = R(0) + t direction.

    Positions and per-atom displacements use Bohr; energy coefficients use
    Hartree. Coefficient n is the directional derivative divided by n!.
    Order four includes cubic and quartic force constants. The report also
    contains component energies, density/overlap coefficients and number,
    metric, projector and stationarity diagnostics. The grid topology and
    reference AO rank are fixed; AO values and Becke weights move analytically.

    Current physical kernels cover all-electron exact-integral HF, semilocal,
    global hybrid and range-separated hybrid models without additional
    nonlocal, empirical or solvent terms. A uniform reference field is allowed.
    """
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [list(d) for d in direction],
        order=order, basis=basis, ecp=ecp, xc=xc, charge=charge,
        multiplicity=multiplicity, grid_level=grid_level, max_iter=max_iter,
        dispersion=dispersion, density_fit=density_fit, conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad, conv_tol_dp=conv_tol_dp, incore_mb=incore_mb,
        vv10_grid_level=vv10_grid_level, response_tol=response_tol,
        response_max_iter=response_max_iter,
        reference_field=reference_field,
    )


def orbital_energy_gradients(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *,
    coordinates: Optional[Sequence[int]] = None, degeneracy_tolerance: float = 1e-8,
    unrestricted: bool = False, **options: Any,
) -> dict[str, Any]:
    """Relaxed orbital gradients in Hartree/Bohr, including moving-basis terms.

    Cartesian IDs are 3*atom+axis; None requests all coordinates. Nuclear
    driver calculation keywords are accepted, including reference_field.
    Degenerate/near-degenerate groups return full projected derivative
    matrices and None for their individual gradients. Their actual energy
    spread is retained; separately sorted directional spectra must not be
    combined into multivariate orbital branches.
    """
    if any(k in options for k in ("order", "direction", "mixed_directions", "field_direction",
                                  "mixed_field_directions", "electric_order", "force_constant_order",
                                  "force_coordinates", "orbital_coordinates", "orbital_gradients", "frontier_gap")):
        raise TypeError("orbital gradients require their own Cartesian coordinates, not Taylor/mixed controls")
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [[0.0]*3 for _ in numbers],
        order=0, orbital_gradients=True,
        orbital_coordinates=None if coordinates is None else list(coordinates),
        degeneracy_tolerance=degeneracy_tolerance, unrestricted=unrestricted, **options,
    )


def frontier_gap_gradient(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *,
    coordinates: Optional[Sequence[int]] = None, degeneracy_tolerance: float = 1e-8,
    unrestricted: bool = False, **options: Any,
) -> dict[str, Any]:
    """Per-spin neutral-reference LUMO-HOMO gaps and relaxed gradients.

    This is an orbital gap, not a charged-state fundamental gap. Missing
    occupied/virtual spaces yield None; grouped frontier levels have no
    unique Cartesian gradient. The common orbital report retains their
    projected derivative matrices for explicit directional calculations.
    """
    if any(k in options for k in ("order", "direction", "mixed_directions", "field_direction",
                                  "mixed_field_directions", "electric_order", "force_constant_order",
                                  "force_coordinates", "orbital_coordinates", "orbital_gradients", "frontier_gap")):
        raise TypeError("frontier gap gradients require their own Cartesian coordinates")
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [[0.0]*3 for _ in numbers],
        order=0, orbital_gradients=True, frontier_gap=True,
        orbital_coordinates=None if coordinates is None else list(coordinates),
        degeneracy_tolerance=degeneracy_tolerance, unrestricted=unrestricted, **options,
    )


def mixed_nuclear_response(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    directions: Sequence[Sequence[Sequence[float]]], **options: Any,
) -> dict[str, Any]:
    """Actual mixed energy derivative along per-atom Bohr displacements.

    Three/four directions select cubic/quartic force constants. No factorial
    is applied to the returned derivative. Calculation keywords are those
    of nuclear_taylor except order; the direction count sets the order.
    """
    if "order" in options or "direction" in options or "mixed_directions" in options:
        raise TypeError("mixed nuclear response derives its order from the directions")
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [[0.0, 0.0, 0.0] for _ in numbers],
        order=0, mixed_directions=[[list(d) for d in path] for path in directions],
        **options,
    )


def molecular_taylor(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    nuclear_direction: Sequence[Sequence[float]],
    electric_direction: Sequence[float], *, order: int = 4, **options: Any,
) -> dict[str, Any]:
    """Energy and dipole jets along simultaneous nuclear and electric paths.

    Geometry and nuclear paths are in Bohr; electric fields are in atomic
    units. All coefficients are divided by n!. Calculation keywords are
    those of nuclear_taylor, including an optional reference_field.
    """
    if any(k in options for k in ("direction", "field_direction", "mixed_directions", "mixed_field_directions", "electric_order")):
        raise TypeError("molecular Taylor response requires one paired nuclear/electric path")
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [list(d) for d in nuclear_direction],
        order=order, field_direction=list(electric_direction), **options,
    )


def mixed_molecular_response(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    nuclear_directions: Sequence[Sequence[Sequence[float]]],
    electric_directions: Sequence[Sequence[float]], **options: Any,
) -> dict[str, Any]:
    """Actual mixed energy and dipole derivatives for paired nuclear/field paths.

    The two direction lists must have equal nonzero lengths. No factorial
    is applied. Roundoff estimates and physical response residuals accompany
    both observables. Zero nuclear or electric entries select pure legs.
    """
    if any(k in options for k in ("order", "direction", "mixed_directions", "field_direction", "mixed_field_directions", "electric_order")):
        raise TypeError("mixed molecular response derives its order from paired paths")
    return _core.nuclear_taylor(
        list(numbers), [list(p) for p in positions], [[0.0] * 3 for _ in numbers],
        order=0, mixed_directions=[[list(d) for d in path] for path in nuclear_directions],
        mixed_field_directions=[list(d) for d in electric_directions], **options,
    )


def static_spectroscopic_derivative(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    nuclear_direction: Sequence[Sequence[float]], *, electric_order: int = 2,
    **options: Any,
) -> dict[str, Any]:
    """Directional analytic APT (1), Raman (2), or hyper-Raman (3).

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
    """
    if any(k in options for k in ("order", "direction", "mixed_directions", "field_direction", "mixed_field_directions")):
        raise TypeError("spectroscopy requires one nuclear path and electric_order=1..3")
    canonical = {key: options.pop(key) for key in ("response_tolerance", "response_max_iterations")
                 if key in options}
    options.setdefault("response_tol", 1e-10)
    options.setdefault("response_max_iter", 500)
    options.update(nuclear_direction=[list(d) for d in nuclear_direction], electric_order=electric_order)
    return _electric_driver("static-spectroscopy", numbers, positions, options, canonical, None)


def analytic_force_constants(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *, order: int = 3,
    patterns: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    coordinates: Optional[Sequence[int]] = None, **options: Any,
) -> dict[str, Any]:
    """Analytic cubic/quartic pattern, Cartesian block, or full Cartesian tensor.

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
    """
    if any(k in options for k in ("direction", "field_direction", "mixed_directions", "mixed_field_directions", "electric_order", "force_constant_order", "force_coordinates", "response_order")):
        raise TypeError("force constants require patterns or coordinates and order=3 or 4")
    canonical = {key: options.pop(key) for key in ("response_tolerance", "response_max_iterations")
                 if key in options}
    options.setdefault("response_tol", 1e-10)
    options.setdefault("response_max_iter", 500)
    options.update(response_order=order, patterns=patterns, coordinates=coordinates)
    return _electric_driver("force-constants", numbers, positions, options, canonical, None)



def anharmonic(
    numbers: Sequence[int], positions: Sequence[Sequence[float]], *,
    mode_ids: Optional[Sequence[int]] = None, include_raman: bool = False,
    equilibrium_tol: float = 1e-4, resonance_tol: float = 1e-8,
    resonance_coupling_factor: float = 1.0, max_basis_states: int = 1024,
    max_polyad_states: int = 256, method: str = "retained-polyad", **options: Any,
) -> dict[str, Any]:
    """Analytic molecular rectilinear anharmonic IR and optional static Raman.

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
    """
    return _core.anharmonic(
        list(numbers), [list(p) for p in positions],
        mode_ids=None if mode_ids is None else list(mode_ids), include_raman=include_raman,
        equilibrium_tol=equilibrium_tol, resonance_tol=resonance_tol,
        resonance_coupling_factor=resonance_coupling_factor, max_basis_states=max_basis_states,
        max_polyad_states=max_polyad_states, method=method, **options,
    )


def periodic(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    cell: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    charge: int = 0,
    multiplicity: int = 1,
    k_points: Optional[Sequence[Sequence[float]]] = None,
    overlap_cutoff: float = 20.0,
    derivatives: bool = False,
    density: Optional[Sequence[Sequence[float]]] = None,
) -> dict[str, Any]:
    """Periodic one-electron matrices, optional Hartree, nuclear energy and stress.

    Parameters
    ----------
    cell
        3x3 lattice in **Bohr**, rows being the lattice vectors.
    positions
        Cartesian coordinates in **Bohr**; they need not lie inside the cell.
    k_points
        Cartesian reciprocal vectors (1/Bohr). Omit for the Gamma point;
        :func:`monkhorst_pack` builds a regular mesh.
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
    supplied density. Use :func:`periodic_scf` for finite-k exact exchange,
    XC-SCF, a periodic total energy, and analytic forces.

    Examples
    --------
    >>> from dft_rs.native import periodic, monkhorst_pack
    >>> cell = [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
    >>> r = periodic([1, 1], [[0, 0, 0], [0, 0, 1.4]], cell,
    ...              k_points=monkhorst_pack(cell, [2, 2, 2]))  # doctest: +SKIP
    >>> r["nuclear_stress"]  # doctest: +SKIP
    """
    return _core.periodic(
        list(numbers),
        [list(p) for p in positions],
        [list(row) for row in cell],
        basis=basis,
        ecp=ecp,
        charge=charge,
        multiplicity=multiplicity,
        k_points=None if k_points is None else [list(k) for k in k_points],
        overlap_cutoff=overlap_cutoff,
        derivatives=derivatives,
        density=None if density is None else [list(row) for row in density],
    )


def periodic_scf(
    numbers: Sequence[int],
    positions: Sequence[Sequence[float]],
    cell: Sequence[Sequence[float]],
    *,
    basis: str = "sto-3g",
    ecp: Optional[str] = None,
    xc: str = "pbe",
    charge: int = 0,
    multiplicity: int = 1,
    pbc: Sequence[bool] = (True, True, True),
    k_mesh: Sequence[int] = (1, 1, 1),
    overlap_cutoff: float = 20.0,
    exchange_cutoff: Optional[float] = None,
    grid_level: int = 3,
    grid_shell: int = 2,
    max_iter: int = 100,
    conv_tol: float = 1e-9,
    conv_tol_grad: Optional[float] = None,
    conv_tol_dp: float = 1e-8,
    forces: bool = False,
    wavefunction: bool = False,
    electronic_temperature: float = 0.0,
    level_shift: float = 0.0,
    damping: float = 0.25,
    diis_size: int = 8,
    use_adiis: bool = True,
    adiis_until_error: float = 0.1,
    guess: str = "core",
    relax: bool = False,
    opt_max_steps: int = 200,
    opt_gtol: float = 3e-4,
    opt_step_max: float = 0.3,
    opt_history: int = 8,
    trajectory: Optional[str] = None,
    response: Optional[dict[str, Any]] = None,
    grid_preset: Optional[str] = None,
    grid_angular_scheme: Optional[str] = None,
    grid_weight_cutoff: Optional[float] = None,
) -> dict[str, Any]:
    """Run finite-k periodic HF or generalized Kohn-Sham SCF.

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
    """
    return _core.periodic_scf(
        list(numbers),
        [list(p) for p in positions],
        [list(row) for row in cell],
        basis=basis,
        ecp=ecp,
        xc=xc,
        charge=charge,
        multiplicity=multiplicity,
        pbc=list(pbc),
        k_mesh=list(k_mesh),
        overlap_cutoff=overlap_cutoff,
        exchange_cutoff=exchange_cutoff,
        grid_level=grid_level,
        grid_shell=grid_shell,
        max_iter=max_iter,
        conv_tol=conv_tol,
        conv_tol_grad=conv_tol_grad,
        conv_tol_dp=conv_tol_dp,
        forces=forces,
        wavefunction=wavefunction,
        electronic_temperature=electronic_temperature,
        level_shift=level_shift,
        damping=damping,
        diis_size=diis_size,
        use_adiis=use_adiis,
        adiis_until_error=adiis_until_error,
        guess=guess,
        relax=relax,
        opt_max_steps=opt_max_steps,
        opt_gtol=opt_gtol,
        opt_step_max=opt_step_max,
        opt_history=opt_history,
        trajectory=trajectory,
        grid_preset=grid_preset,
        grid_angular_scheme=grid_angular_scheme,
        grid_weight_cutoff=grid_weight_cutoff,
        **({} if response is None else response),
    )


def periodic_field_response(
    numbers, positions, cell, *, q=(0,0,0), scalar_potential=0j,
    vector_potential=(0j,0j,0j), omega=0.0, damping=0.0, imaginary=False,
    fixed_number=True, reciprocal_cutoff=6.0, response_tol=1e-10,
    response_max_iter=500, response_restart=40, **scf_options,
):
    """Screened orbital density and current for periodic electric/magnetic fields.

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
    """
    vector=[complex(v) for v in vector_potential]
    if len(vector)!=3:
        raise ValueError("vector_potential requires three Cartesian components")
    scalar=complex(scalar_potential)
    return _core.periodic_scf(list(numbers), [list(p) for p in positions], [list(r) for r in cell],
        response_q=list(q), response_scalar_potential=[scalar.real,scalar.imag],
        response_vector_potential=[[v.real,v.imag] for v in vector], response_omega=omega,
        response_damping=damping, response_imaginary=imaginary, response_fixed_number=fixed_number,
        response_reciprocal_cutoff=reciprocal_cutoff,response_tol=response_tol,
        response_max_iter=response_max_iter,response_restart=response_restart,**scf_options)["response"]


def periodic_density_response(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    cell: Sequence[Sequence[float]], *, q: Sequence[float] = (0, 0, 0),
    mode: int = 0, omega: float = 0.0, damping: float = 0.0,
    imaginary: bool = False, fixed_number: bool = True,
    reciprocal_cutoff: float = 6.0, response_tol: float = 1e-10,
    response_max_iter: int = 500, response_restart: int = 40, **scf_options,
) -> dict[str, Any]:
    """Screened density response to a unit plane-wave scalar potential.

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
    """
    return _core.periodic_scf(
        list(numbers), [list(p) for p in positions], [list(r) for r in cell],
        response_q=list(q), response_density_mode=mode, response_omega=omega,
        response_damping=damping, response_imaginary=imaginary,
        response_fixed_number=fixed_number,
        response_reciprocal_cutoff=reciprocal_cutoff, response_tol=response_tol,
        response_max_iter=response_max_iter, response_restart=response_restart,
        **scf_options,
    )["response"]


def periodic_dielectric(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    cell: Sequence[Sequence[float]], *, q: Sequence[float] = (0, 0, 0),
    modes: Sequence[int] = (0,), omega: float = 0.0, damping: float = 0.0,
    imaginary: bool = False, fixed_number: bool = True,
    reciprocal_cutoff: float = 6.0, response_tol: float = 1e-10,
    response_max_iter: int = 500, response_restart: int = 40, **scf_options,
) -> dict[str, Any]:
    """Selected inverse dielectric entries, epsilon^-1 = I + v*screened_chi.

    Units, reference controls, and reciprocal mode ordering match
    periodic_density_response. ``modes`` selects both reported axes; all
    reciprocal modes within the cutoff still participate in screening.
    v=4*pi/|G+q|^2 multiplies the density row. chi is the fully screened density
    response to external potential, in bohr^-3/Hartree.
    """
    return _core.periodic_scf(
        list(numbers), [list(p) for p in positions], [list(r) for r in cell],
        response_q=list(q), response_modes=list(modes), response_omega=omega,
        response_damping=damping, response_imaginary=imaginary,
        response_fixed_number=fixed_number,
        response_reciprocal_cutoff=reciprocal_cutoff, response_tol=response_tol,
        response_max_iter=response_max_iter, response_restart=response_restart,
        **scf_options,
    )["response"]


def periodic_thermal_response(
    numbers: Sequence[int], positions: Sequence[Sequence[float]],
    cell: Sequence[Sequence[float]], *, delta_temperature: float = 1.0,
    delta_numbers: Sequence[float] = (0.0, 0.0),
    delta_mu: Sequence[float] = (0.0, 0.0), fixed_number: bool = True,
    reciprocal_cutoff: float = 6.0, response_tol: float = 1e-10,
    response_max_iter: int = 500, response_restart: int = 40, **scf_options,
) -> dict[str, Any]:
    """Uniform static self-consistent Mermin thermodynamic response.

    The positive reference electronic_temperature and delta_temperature use
    Kelvin; delta_mu uses Hartree and delta_numbers is per spin per cell.
    Fixed-number requests use delta_numbers; reservoir requests use delta_mu.
    Returned density/entropy variations are dimensionless/in k_B units;
    internal/free-energy and chemical-potential variations use Hartree.
    A unit temperature direction therefore returns derivatives per Kelvin.
    All geometry is in Bohr. No finite-temperature differences are used.
    """
    return _core.periodic_scf(
        list(numbers), [list(p) for p in positions], [list(r) for r in cell],
        response_q=[0.0, 0.0, 0.0], response_thermal=True,
        response_delta_temperature=delta_temperature,
        response_delta_numbers=list(delta_numbers), response_delta_mu=list(delta_mu),
        response_fixed_number=fixed_number, response_reciprocal_cutoff=reciprocal_cutoff,
        response_tol=response_tol, response_max_iter=response_max_iter,
        response_restart=response_restart, **scf_options,
    )["response"]


def periodic_basis(numbers, positions, cell, *, basis="sto-3g", pbc=(True, True, True),
                   k_mesh=(1, 1, 1), basis_cutoff=12.0, grid_level=None,
                   grid_cutoff=12.0, grid_derivatives=False):
    """Gaussian Bloch S/T and analytic atomic derivatives, in atomic units.

    Positions/cell/cutoffs use Bohr. Optional atom-centred quadrature covers
    one periodic cell and open directions; weights have the 3D Bohr^3 measure.
    This foundation is independent of electron occupations and functional.
    It is not a self-consistent energy calculation.
    """
    return _core.periodic_basis(numbers, positions, cell, basis=basis, pbc=pbc,
                                k_mesh=k_mesh, basis_cutoff=basis_cutoff,
                                grid_level=grid_level, grid_cutoff=grid_cutoff,
                                grid_derivatives=grid_derivatives)


def dimensional_scf(numbers, positions, cell, *, basis="sto-3g", xc="pbe", charge=0,
                    multiplicity=1, pbc=(True, True, True), k_mesh=(1, 1, 1), **options):
    """Self-consistent dimensional HF/DFT in a common compensated charge gauge.

    Geometry and explicit widths/cutoffs are Bohr, energy is Hartree per cell.
    A charged wire/slab requires Gaussian countercharge parameters. Raw fresh
    physical Fock, occupations, number and quadrature diagnostics are returned.
    This is the zero-field collinear driver; ECP/nonlocal tails remain pending.
    """
    return _core.dimensional_scf(list(numbers), [list(p) for p in positions],
        [list(v) for v in cell], basis=basis, xc=xc, charge=charge,
        multiplicity=multiplicity, pbc=list(pbc), k_mesh=list(k_mesh), **options)



def dimensional_field_response(numbers, positions, cell, *, transfer_fractional=(0.,0.,0.),
    scalar_potential=0j, vector_potential=(0j,0j,0j), omega=0., damping=0., imaginary=False,
    fixed_number=True, response_tol=1e-10, response_max_iter=500, response_restart=40, **scf_options):
    """Screened orbital fields in 1D/2D/3D; geometry Bohr, frequency Hartree.

    transfer_fractional labels reciprocal lattice vectors on periodic axes;
    open coordinates must be zero. Integer transfers retain the microscopic
    source wave but use the exact uniform Bloch sector and number constraint.
    V=-phi and reduced A use exp(i wave.r-i z.t); E=i wave V+i z A,
    B=i wave cross A. Density/current are complex per-cell quantities.
    All-electron collinear LDA/GGA density screening is implemented here;
    current XC, spin, SOC, Berry and magnetic flux reference extensions remain
    tracked. Raw continuity, quadrature and stationary-reference defects are
    returned independently of response convergence.
    """
    scalar=complex(scalar_potential);vector=[complex(v) for v in vector_potential]
    if len(vector)!=3:
        raise ValueError("vector_potential requires three Cartesian components")
    return dimensional_scf(numbers,positions,cell,field_transfer=list(transfer_fractional),
        field_scalar=[scalar.real,scalar.imag],field_vector=[[v.real,v.imag] for v in vector],
        response_omega=omega,response_damping=damping,response_imaginary=imaginary,
        response_fixed_number=fixed_number,response_tol=response_tol,
        response_max_iter=response_max_iter,response_restart=response_restart,**scf_options)["field_response"]


def periodic_field_operators(
    numbers, positions, cell, *, basis="sto-3g", charge=0, pbc=(True,True,True),
    k_mesh=(1,1,1), basis_cutoff=12., wave=(0.,0.,0.), k_offset=(0.,0.,0.),
    frequency=(0.,0.), scalar_potential=(0.,0.), vector_potential=None,
):
    """Analytic scalar/current/contact Fourier vertices for 1D/2D/3D.

    Coordinates/cutoff use Bohr and Cartesian wave/k offsets inverse Bohr.
    Complex potentials use [real,imag], and frequency [Hartree,damping].
    The first derivative connects (k+q,k), second order (k+2q,k).
    This operator report is independent of SCF occupations or functional.
    """
    return _core.periodic_field_operators(
        list(numbers),[list(p) for p in positions],[list(v) for v in cell],
        basis=basis,charge=charge,pbc=list(pbc),k_mesh=list(k_mesh),
        basis_cutoff=basis_cutoff,wave=list(wave),k_offset=list(k_offset),
        frequency=list(frequency),scalar_potential=list(scalar_potential),
        vector_potential=None if vector_potential is None else [list(v) for v in vector_potential],
    )


def periodic_xc_operators(numbers, positions, cell, *, density_real, density_imag=None,
                          basis="sto-3g", xc="pbe", charge=0, pbc=(True, True, True),
                          k_mesh=(1, 1, 1), basis_cutoff=12., grid_cutoff=12.,
                          grid_level=3, operator_tolerance=1e-11, grid_tolerance=1e-5):
    """Semilocal energy/components and analytic spin/Bloch XC potentials.

    Geometry/cutoffs use Bohr; densities use [spin=2,k,AO,AO] spherical AO
    ordering and uniform k weights. Exact-exchange coefficients and required
    nonlocal tails are reported separately; this is the semilocal component,
    without SCF stationarity or Coulomb compensation inferred from charge.
    """
    return _core.periodic_xc_operators(
        list(numbers), [list(p) for p in positions], [list(v) for v in cell], density_real,
        density_imag=density_imag, basis=basis, xc=xc, charge=charge, pbc=list(pbc),
        k_mesh=list(k_mesh), basis_cutoff=basis_cutoff, grid_cutoff=grid_cutoff,
        grid_level=grid_level, operator_tolerance=operator_tolerance, grid_tolerance=grid_tolerance)


def periodic_coulomb_operators(numbers, positions, cell, *, basis="sto-3g", charge=0,
                               pbc=(True, True, True), k_mesh=(1, 1, 1), basis_cutoff=12.,
                               density_real=None, density_imag=None, nuclear_widths=None,
                               compensation="neutral", countercharge_center=None,
                               countercharge_width=None, alpha=None, real_cutoff=None,
                               reciprocal_cutoff=None, quadrature_order=96, tolerance=1e-11):
    """Contracted spherical nuclear/Hartree operators and fixed-density gradients.

    All geometry and explicit widths/cutoffs use Bohr. Densities are uniformly
    weighted [k,AO,AO] Hermitian matrices; their imaginary parts are retained
    even without time reversal. No density requests nuclear operators only.
    Formal charge fixes the physical compensation and target electron count.
    Charged wires/slabs require a Gaussian countercharge; uniform is 3D only.
    The report includes every split component, raw defects, actual electron
    count, and fixed AO coefficient gradients. These are Coulomb components
    before SCF stationarity and Pulay/charge constraints, not SCF forces.
    """
    return _core.periodic_coulomb_operators(
        list(numbers), [list(p) for p in positions], [list(v) for v in cell],
        basis=basis, charge=charge, pbc=list(pbc), k_mesh=list(k_mesh),
        basis_cutoff=basis_cutoff, density_real=density_real, density_imag=density_imag,
        nuclear_widths=nuclear_widths, compensation=compensation,
        countercharge_center=countercharge_center, countercharge_width=countercharge_width,
        alpha=alpha, real_cutoff=real_cutoff, reciprocal_cutoff=reciprocal_cutoff,
        quadrature_order=quadrature_order, tolerance=tolerance,
    )


def periodic_gaussian_integral(positions, exponents, angular, cell, *,
                               pbc=(True, True, True), source_charge=None,
                               source_width=None, alpha=None, real_cutoff=None,
                               reciprocal_cutoff=None, quadrature_order=96,
                               tolerance=1e-11):
    """Normalized Cartesian primitive Coulomb integral and centre gradients.

    Three positions mean two Gaussian centres followed by a nuclear source;
    provide two exponents and angular triples. Four positions mean an ERI
    (ab|cd); provide four exponents and angular triples, without source knobs.
    Geometry/widths/cutoffs use Bohr, exponents Bohr^-2, source charge e.
    Nuclear defaults are charge 1 and point width 0. Values are Hartree and
    gradients Hartree/Bohr in input centre order. The 3D Green function has
    zero cell average; isolated wire/slab gauges match periodic_electrostatics.
    These are primitive Green integrals before Bloch or density contraction.
    A neutral electronic calculation must combine all charge contributions.
    """
    return _core.periodic_gaussian_integral(
        [list(p) for p in positions], list(exponents), [list(l) for l in angular],
        [list(v) for v in cell], pbc=list(pbc), source_charge=source_charge,
        source_width=source_width, alpha=alpha, real_cutoff=real_cutoff,
        reciprocal_cutoff=reciprocal_cutoff, quadrature_order=quadrature_order,
        tolerance=tolerance,
    )


def periodic_electrostatics(
    positions: Sequence[Sequence[float]], charges: Sequence[float],
    cell: Sequence[Sequence[float]], *, pbc: Sequence[bool] = (True, True, True),
    widths: Optional[Sequence[float]] = None, compensation: str = "neutral",
    countercharge_center: Optional[Sequence[float]] = None,
    countercharge_width: Optional[float] = None, alpha: Optional[float] = None,
    real_cutoff: Optional[float] = None, reciprocal_cutoff: Optional[float] = None,
    quadrature_order: int = 96, tolerance: float = 1e-11,
    k_mesh: Sequence[int] = (1, 1, 1),
) -> dict[str, Any]:
    """Dimensional Coulomb energy and analytic forces for prescribed charges.

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
    """
    return _core.periodic_electrostatics(
        [list(p) for p in positions], list(charges), [list(v) for v in cell],
        pbc=list(pbc), widths=None if widths is None else list(widths),
        compensation=compensation,
        countercharge_center=None if countercharge_center is None else list(countercharge_center),
        countercharge_width=countercharge_width, alpha=alpha, real_cutoff=real_cutoff,
        reciprocal_cutoff=reciprocal_cutoff, quadrature_order=quadrature_order,
        tolerance=tolerance, k_mesh=list(k_mesh),
    )


def periodic_capabilities() -> dict[str, bool]:
    """Report executable periodic kernels and staged feature boundaries."""
    return _core.periodic_capabilities()


def monkhorst_pack(
    cell: Sequence[Sequence[float]], mesh: Sequence[int], *,
    pbc: Sequence[bool] = (True, True, True),
) -> list[list[float]]:
    """Regular k-point mesh for a lattice, Cartesian in 1/Bohr.

    >>> from dft_rs.native import monkhorst_pack
    >>> len(monkhorst_pack([[5,0,0],[0,5,0],[0,0,5]], [2, 2, 2]))
    8
    """
    return _core.monkhorst_pack([list(row) for row in cell], list(mesh), pbc=list(pbc))


def basis_sets() -> list[str]:
    """Names of the bundled basis-set families.

    >>> from dft_rs.native import basis_sets
    >>> "def2-svp" in basis_sets()  # doctest: +SKIP
    True
    """
    return _core.basis_sets()


def functionals() -> list[dict]:
    """The functional registry, one dict per canonical name.

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
    """
    return _core.functionals()


# ---------------------------------------------------------------------------
# Common calculation driver
#
# Every calculation below is executed by the same Rust driver that serves the
# `dft-rs <feature>` command line and `dft_rs::driver::run`. Positions and
# lattice rows are Bohr; keyword arguments are the calculation controls
# (`xc`, `basis`, `charge`, `multiplicity`, `unrestricted`, `grid_level`,
# `dispersion`, `solvent`, `field`, SCF tolerances, ... and the controls of
# each calculation). A control that the calculation does not read is refused
# by name. Reports are dictionaries in atomic units unless a key names another
# unit; each carries `feature`, `model`, `units`, `converged` and `warnings`.
# ---------------------------------------------------------------------------


def run_feature(
    feature: str,
    numbers: Any = None,
    positions: Optional[Sequence[Sequence[float]]] = None,
    *,
    cell: Optional[Sequence[Sequence[float]]] = None,
    pbc: Optional[Sequence[bool]] = None,
    **controls: Any,
) -> Report:
    """Run any calculation registered with the common driver.

    The structure is given as ``numbers`` and ``positions`` (Bohr), with
    ``cell`` rows (Bohr) and ``pbc`` for a periodic one, or as a single
    object with those attributes (:class:`dft_rs.Molecule`,
    :class:`dft_rs.Crystal`) in place of ``numbers``.
    ``feature_registry()["features"]`` lists the registered IDs.
    """
    if positions is None and hasattr(numbers, "numbers") and hasattr(numbers, "positions"):
        structure = numbers
        numbers, positions = structure.numbers, structure.positions
        if cell is None:
            cell = getattr(structure, "cell", None)
        if pbc is None:
            pbc = getattr(structure, "pbc", None)
    return Report(
        _core.run_feature(
            feature,
            None if numbers is None else list(_atomic_numbers(numbers)),
            None if positions is None else [[float(x) for x in p] for p in positions],
            cell=None if cell is None else [[float(x) for x in row] for row in cell],
            pbc=None if pbc is None else list(_periodicity(pbc)),
            **controls,
        )
    )


def energy(numbers, positions, **controls) -> dict[str, Any]:
    """Molecular total energy (Hartree) with every energy component.

    Named double hybrids use this same entry: ``xc='b2plyp'``,
    ``'dsd-pbep86'`` or ``'dsd-blyp'`` (2013 D3(BJ) recipes), ``'pwpb95'``
    or ``'wb97m2'``. The latter uses a distinct target on omegaB97M-V
    orbitals and reports reference and target nonlocal terms separately. Any existing
    parent accepts ``pt2_same_spin`` and ``pt2_opposite_spin`` coefficients.
    Reports separate parent, dispersion, SS/OS PT2 and amplitude residuals.
    """
    return run_feature("energy", numbers, positions, **controls)


def mermin_free_energy(numbers, positions, **controls) -> dict[str, Any]:
    """Electronic Mermin free energy A=E-T_e*S_e in Hartree.

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
    """
    return run_feature("mermin-free-energy", numbers, positions, **controls)


def gibbs_free_energy(numbers, positions, **controls) -> dict[str, Any]:
    """Ideal-gas RRHO Gibbs free energy G=H-T*S in Hartree per molecule.

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
    """
    return run_feature("gibbs-free-energy", numbers, positions, **controls)


def free_energy(numbers, positions, **controls) -> dict[str, Any]:
    """Compatibility wrapper for mermin_free_energy, the electronic A=E-T_e*S_e.

    Prefer mermin_free_energy or gibbs_free_energy to state the thermodynamic
    potential explicitly. This wrapper retains the free-energy feature ID.
    """
    return run_feature("free-energy", numbers, positions, **controls)


def gradient(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic gradient (Hartree/Bohr); reference selects RKS/UKS/ROKS/GKS constraints within the documented model scope."""
    return run_feature("gradient", numbers, positions, **controls)


def charges(numbers, positions, **controls) -> dict[str, Any]:
    """Mulliken/Loewdin charges; reference selects auto/rks/uks/roks/gks and HF aliases."""
    return run_feature("charges", numbers, positions, **controls)


def natural_bond(numbers, positions, **controls) -> dict[str, Any]:
    """Natural bond orbital analysis of the converged reference.

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
    """
    return run_feature("natural-bond", numbers, positions, **controls)


def igmh(numbers, positions, **controls) -> dict[str, Any]:
    """Sample molecular IGMH, using the spin-summed SCF density.

    ``igmh_points`` are Cartesian triples in Bohr, or supply ``igmh_grid``
    as origin and three step vectors in Bohr with ``igmh_grid_shape``.
    ``fragments`` gives one integer label per atom. The default
    ``igmh_gradient="published"`` follows the authors' erratum.
    ``"hirshfeld"`` uses the spatial gradient of the Hirshfeld density.
    Results include total, inter- and intrafragment delta-g, signed density
    and raw partition residuals. ``igmh_atom_details=True`` adds atomic fields.
    """
    return run_feature("igmh", numbers, positions, **controls)


def periodic_natural_bond(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Natural bond orbital analysis of a periodic reference.

    The Bloch overlap, spin densities and Fock matrices of the k mesh are
    unfolded onto its Born-von Karman supercell; natural charges,
    configurations, Wiberg bonds (with cell offsets of the partners), the
    natural Lewis structure per cell and second-order donor-acceptor
    energies (Hartree) are reported for the atoms of the home cell. Open
    axes (``pbc``) use the dimensional SCF; use at least three k points
    along every periodic axis so that an atom and its image do not share
    supercell orbitals.
    """
    return run_feature("periodic-natural-bond", numbers, positions, cell=cell, **controls)


def natural_shielding(numbers, positions, **controls) -> dict[str, Any]:
    """Natural chemical shielding analysis: the NMR shielding of every
    nucleus (ppm) as a sum over the natural localized molecular orbitals of
    the natural Lewis structure, with diamagnetic and paramagnetic parts.
    ``gauge`` (``"giao"``), ``gauge_origin``, ``nuclei`` and the
    ``natural_bond`` controls apply."""
    return run_feature("natural-shielding", numbers, positions, **controls)


def natural_coupling(numbers, positions, **controls) -> dict[str, Any]:
    """Natural J-coupling analysis of a closed-shell molecule: reduced
    indirect spin-spin couplings (atomic units; Hz with ``nuclear_g``) of
    the ``nuclei`` pairs as sums over natural localized molecular orbitals,
    per Fermi-contact, spin-dipole, paramagnetic and diamagnetic term."""
    return run_feature("natural-coupling", numbers, positions, **controls)


def natural_energy_decomposition(numbers, positions, *, fragments, **controls) -> dict[str, Any]:
    """Natural energy decomposition of the interaction of molecular
    fragments (``fragments``: the fragment index 0, 1, ... of every atom):
    charge transfer from the deletion of inter-fragment NBO Fock couplings,
    electrostatics of the deformed fragments, exchange-correlation remainder
    and deformation against the isolated fragments in the complex basis
    (Hartree; their sum is the interaction energy)."""
    return run_feature("natural-energy-decomposition", numbers, positions,
                       fragments=fragments, **controls)


def sapt_analysis(numbers, positions, *, fragments, **controls) -> dict[str, Any]:
    """Intermolecular perturbation analysis of fragments 0 and 1.

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
    `casimir_polder_points`."""
    return run_feature("sapt-analysis", numbers, positions,
                       fragments=fragments, **controls)


def dipole(numbers, positions, **controls) -> dict[str, Any]:
    """Molecular dipole (e*Bohr and Debye) about ``origin`` (default
    ``"center_of_mass"``; also ``"center_of_nuclear_charge"``, ``"zero"`` or
    Bohr coordinates)."""
    return run_feature("dipole", numbers, positions, **controls)


def orbitals(numbers, positions, **controls) -> dict[str, Any]:
    """Orbital energies and occupations for reference=auto/rks/uks/roks/gks (HF aliases accepted).
    ``coefficients=True`` and ``density=True`` add AO matrices and overlap.
    ROKS returns spin semicanonical occupied/virtual blocks of its common
    spatial determinant, rather than unconstrained spin Fock eigenvectors."""
    return run_feature("orbitals", numbers, positions, **controls)


def write_molden(numbers, positions, *, path, **controls) -> dict[str, Any]:
    """Write a Molden file of the converged reference to ``path``."""
    return run_feature("molden", numbers, positions, path=str(path), **controls)


def save_orbitals(numbers, positions, *, path, **controls) -> dict[str, Any]:
    """Write the orbital report (coefficients, densities, occupations) as
    JSON to ``path``; refused when occupations do not reproduce the density."""
    return run_feature("orbitals-save", numbers, positions, path=str(path), **controls)


def molecular_multipole(numbers, positions, **controls) -> dict[str, Any]:
    """Raw Cartesian, Buckingham traceless and Racah spherical multipoles up
    to ``max_rank`` (default 4, at most 8) about ``origin``."""
    return run_feature("molecular-multipole", numbers, positions, **controls)


def molecular_medium_optics(numbers, positions, **controls) -> dict[str, Any]:
    """Molecular dielectric, normal indices and anisotropic slab scattering."""
    return run_feature("molecular-medium-optics", numbers, positions, **controls)


def quadrupole_polarizability(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic electric quadrupole response in five traceless field-gradient modes.

    The origin is in Bohr and the retarded frequency and damping are in Hartree.
    Supported collinear double hybrids include the complex parent, target
    and spin-resolved PT2 quasi-energy terms.
    """
    return run_feature("quadrupole-polarizability", numbers, positions, **controls)


def multipole_polarizability(numbers, positions, **controls) -> dict[str, Any]:
    """Racah electric multipole response for rank 1 to 5 (default 3).

    Rows are real spherical moments and columns their conjugate fields, in
    mode order 0, +1, -1, and so on. Origin is in Bohr, frequency and damping
    are in Hartree, and the response is in e² Bohr^(2 rank)/Hartree.
    Supported collinear double hybrids return complex parent, target
    and spin-resolved PT2 quasi-energy terms.
    The default center of mass moves with the molecule, making the result
    invariant to a rigid coordinate translation. An explicit different
    expansion center physically mixes lower multipole ranks.
    """
    return run_feature("multipole-polarizability", numbers, positions, **controls)


def conceptual_dft(numbers, positions, **controls) -> dict[str, Any]:
    """Conceptual DFT descriptors. ``route`` is ``"grand_canonical"``
    (analytic, needs ``electronic_temperature`` > 0 K and is then the
    default), ``"integer"`` (N, N-1, N+1 total energies; default at zero
    temperature) or ``"koopmans"``."""
    return run_feature("conceptual-dft", numbers, positions, **controls)


def spin_conceptual_dft(numbers, positions, **controls) -> dict[str, Any]:
    """Spin-channel conceptual DFT with M = N_alpha - N_beta."""
    return run_feature("spin-conceptual-dft", numbers, positions, **controls)


def hessian_report(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic Cartesian Hessian in Hartree/Bohr^2.

    With positive electronic_temperature, supported ROKS and GKS providers
    differentiate the stationary Mermin free energy and return thermal
    response and number diagnostics. Other report fields, including the
    atomic polar tensor and raw ordered asymmetry, depend on the provider.
    The complex London route accepts fixed laboratory electric and magnetic
    fields with integer occupations. See the API guide for provider scope.
    """
    return run_feature("hessian", numbers, positions, **controls)


def frequencies_report(numbers, positions, **controls) -> dict[str, Any]:
    """Harmonic frequencies (cm^-1), IR intensities (km/mol), Cartesian
    normal modes and RRHO thermochemistry (``temperature`` K, ``pressure``
    Pa, ``symmetry_number``). ROHF/ROKS and complex GHF/GKS use the same
    analytic Hessian provider as hessian_report, retaining its diagnostics.
    vibrational_projection is auto, eckart or cartesian: auto retains all
    Cartesian modes under a fixed laboratory field or extra AO operator.
    Missing atomic polar tensors give null IR intensities. Fixed-spinor
    electronic_degeneracy defaults to one. RRHO is withheld when its free
    molecule or positive-mode assumptions fail. See response.md for the
    correlated-reference and bounded acceptance scope."""
    return run_feature("frequencies", numbers, positions, **controls)


def vibrational_spectra(numbers, positions, **controls) -> dict[str, Any]:
    """Frequencies with analytic IR intensities and analytic Raman
    activities (Angstrom^4/amu) and depolarization ratios."""
    return run_feature("ir-raman", numbers, positions, **controls)


def polarizability_report(numbers, positions, **controls) -> dict[str, Any]:
    """Static dipole polarizability (Bohr^3 and Angstrom^3) from CPKS."""
    return run_feature("polarizability", numbers, positions, **controls)


def multipole_gradient(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic nuclear derivatives of Cartesian multipoles up to
    ``max_rank`` (default 2) about ``origin``."""
    return run_feature("multipole-gradient", numbers, positions, **controls)


def response_derivatives(numbers, positions, **controls) -> dict[str, Any]:
    """First-order response export: orbital rotations U^F and densities
    dP/dF for each field axis and nuclear density derivatives dP/dR;
    ``path`` also writes the report as JSON."""
    if "path" in controls and controls["path"] is not None:
        controls["path"] = str(controls["path"])
    return run_feature("response-save", numbers, positions, **controls)


def third_derivative_2n1(numbers, positions, **controls) -> dict[str, Any]:
    """Cartesian third derivatives of the energy (Hartree/Bohr^3) from
    first-order responses only, by the 2n+1 rule."""
    return run_feature("third-derivative-2n1", numbers, positions, **controls)


def cubic_fc_along(numbers, positions, **controls) -> dict[str, Any]:
    """Cubic force constants contracted with the displacement pattern
    ``probe`` (one value per Cartesian coordinate): the derivative of the
    Hessian along it. Optional ``bfield`` and ``field`` use the shared
    complex London stationary nuclear response about a biased reference;
    the matrix has units Hartree/Bohr^3 for dimensionless probe entries.
    Integer RKS/UKS correlated requests include full orbital/PT2 response,
    reference_field electric bias and separately reported parent/target/SS/OS
    and dispersion matrices. Magnetic PT2 and additional Hamiltonian providers
    remain separate."""
    return run_feature("cubic-fc-along", numbers, positions, **controls)


def quartic_fc_along(numbers, positions, **controls) -> dict[str, Any]:
    """Quartic force constants contracted with the displacement patterns
    ``probe`` and ``second_probe`` (default the same pattern). Optional
    ``bfield`` and ``field`` select complex London nuclear response with
    fixed laboratory fields; Hartree/Bohr^4 for dimensionless probe entries.
    Integer RKS/UKS correlated requests include full orbital/PT2 response,
    reference_field electric bias and separately reported parent/target/SS/OS
    and dispersion matrices. Magnetic PT2 and additional Hamiltonian providers
    remain separate."""
    return run_feature("quartic-fc-along", numbers, positions, **controls)


def streamed_gradient(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic gradient with integral-direct Coulomb and exchange builds:
    no integral tensor is stored."""
    return run_feature("streamed-gradient", numbers, positions, **controls)


def excitations(numbers, positions, **controls) -> dict[str, Any]:
    """Vertical excitation energies and transition strengths.
    Collinear references use their TDA/RPA models. ``reference="roks"`` or
    ``"rohf"`` selects reference-spin common-orbital flag TDA; its HF doublet
    and quartet routes have full derivative/public acceptance.
    ``reference="gks"``/``"ghf"`` selects full complex spinor TDA with transverse channels
    and no exact total-spin label. Full-spin RPA and correlated flag/spinor
    propagators remain unconnected. ``amplitudes=True`` includes amplitudes.
    ``td_nto=True`` returns particle–hole pairs for supported TDA states.
    ``td_difference_density=True`` returns unrelaxed attachment and detachment
    orbitals for RKS, UKS and GKS TDA."""
    return run_feature("excitations", numbers, positions, **controls)


def ecd_spectrum(numbers, positions, **controls) -> dict[str, Any]:
    """Gaussian-broadened absorption and circular-dichroism spectra
    (L mol^-1 cm^-1) of the TDA states; ``broadening_ev`` (standard
    deviation in eV) is required."""
    return run_feature("ecd-spectrum", numbers, positions, **controls)


def td_state_gradient(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic total excited-state gradient (Hartree/Bohr).
    ``root`` is one based. The collinear Z-vector route covers restricted
    singlet HF/pure LDA; ``td_gradient_route="path"`` selects compatible
    analytic nuclear paths. ROKS/ROHF uses the common-orbital reference-spin
    flag model; GKS/GHF uses full-spin TDA with a fixed additional AO source.
    HF flag doublet/quartet and isolated spinor subsets have complete Cartesian
    acceptance. Moving semilocal spinor kernels are implemented with molecular
    verification pending; correlated pole nuclear derivatives are unconnected."""
    return run_feature("td-state-gradient", numbers, positions, **controls)


def hyperfine(numbers, positions, **controls) -> dict[str, Any]:
    """Contact, spin-dipolar and total hyperfine tensors of an open shell.

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
    """
    return run_feature("hyperfine", numbers, positions, **controls)


def nmr_shielding(numbers, positions, **controls) -> dict[str, Any]:
    """Nuclear magnetic shielding tensors in ppm from the coupled-perturbed
    magnetic response; ``gauge`` is "giao" (default) or "common-origin" with
    ``gauge_origin``."""
    return run_feature("nmr-shielding", numbers, positions, **controls)


def nmr_spectrum(numbers, positions, **controls) -> dict[str, Any]:
    """Isotropic NMR spectrum: one integrated area per observed nucleus at
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
    and relaxation terms are excluded."""
    return run_feature("nmr-spectrum", numbers, positions, **controls)


def spin_spin_couplings(numbers, positions, **controls) -> dict[str, Any]:
    """Indirect nuclear spin-spin coupling tensors. UKS/GKS use full complex
    Pauli response; ``spin_response_constraint`` selects free (default) or fixed-spin-direction
    response. ROKS and correlated open-shell providers
    remain separate. Complete mixed spin/orbital terms are retained:
    reduced coupling tensors with their diamagnetic and paramagnetic
    spin-orbit, Fermi-contact and spin-dipole parts between the atoms
    ``nuclei`` (default all), and with ``nuclear_g`` the constants in Hz."""
    return run_feature("spin-spin-couplings", numbers, positions, **controls)


def fermi_contact_couplings(numbers, positions, **controls) -> dict[str, Any]:
    """Fermi-contact part of the indirect nuclear spin-spin couplings."""
    return run_feature("fermi-contact-couplings", numbers, positions, **controls)


def exafs(numbers, positions, **controls) -> dict[str, Any]:
    """Single-scattering EXAFS of the atom ``exafs_absorber`` from the
    tabulated backscattering amplitudes and phases of ``exafs_scattering``
    (rows of atomic number, k, amplitude, phase); no electronic-structure
    calculation is involved."""
    return run_feature("exafs", numbers, positions, **controls)


def xanes(numbers, positions, **controls) -> dict[str, Any]:
    """Core-level absorption from Tamm-Dancoff response in the
    core-valence-separated space of the K shell of ``xanes_element`` (or of
    ``core_orbitals``); the core hole is not relaxed."""
    return run_feature("xanes", numbers, positions, **controls)


def nexafs(numbers, positions, **controls) -> dict[str, Any]:
    """Polarized molecular near-edge absorption in a finite orbital basis."""
    return run_feature("nexafs", numbers, positions, **controls)


def constrained_dft(numbers, positions, **controls) -> dict[str, Any]:
    """Variational AO population or spin constraints with optional analytic forces,
    geometry optimization and static electric response. Supply fragment atoms
    and electrons in the entries of density_constraints."""
    return run_feature("constrained-dft", numbers, positions, **controls)


def dft_plus_u(numbers, positions, **controls) -> dict[str, Any]:
    """Fixed-geometry molecular DFT+U. Supply Hubbard site parameters in Hartree."""
    return run_feature("dft-plus-u", numbers, positions, **controls)


def pz_sic(numbers, positions, **controls) -> dict[str, Any]:
    """Perdew-Zunger SIC on real integer unrestricted orbitals at fixed geometry."""
    return run_feature("pz-sic", numbers, positions, **controls)


def flo_sic_fixed(numbers, positions, **controls) -> dict[str, Any]:
    """Fermi-Lowdin SIC on frozen UKS orbitals with fixed descriptors in Bohr."""
    return run_feature("flo-sic-fixed", numbers, positions, **controls)


def car_parrinello_md(numbers, positions, **controls) -> dict[str, Any]:
    """Microcanonical dynamics of nuclei and real or complex occupied orbitals."""
    return run_feature("car-parrinello-md", numbers, positions, **controls)


def magnetic_quadrupole(numbers, positions, **controls) -> dict[str, Any]:
    """Orbital and Pauli magnetic quadrupole at the selected field origin."""
    return run_feature("magnetic-quadrupole", numbers, positions, **controls)


def magnetizability(numbers, positions, **controls) -> dict[str, Any]:
    """Magnetizability tensor from the coupled-perturbed magnetic response
    with a common gauge origin (``gauge_origin``, default the centre of
    nuclear charge); the result depends on the origin in a finite basis.
    gauge="london" uses London orbitals. magnetic_directions optionally
    contracts a mixed energy derivative of order 1..4; matching optional
    electric_directions add electric parts to the same dimensionless
    parameters. The magnetic_mixed fields report this joint derivative.
    Spin Zeeman and current-dependent correlation are excluded on this path.
    """
    return run_feature("magnetizability", numbers, positions, **controls)


def franck_condon(numbers, positions, **controls) -> dict[str, Any]:
    """Franck-Condon progression of the singlet TDA state ``root`` in the
    vertical-gradient model (displaced ground-state harmonic modes, zero
    temperature): Huang-Rhys factors, 0-0 energy and line factors."""
    return run_feature("franck-condon", numbers, positions, **controls)


def excited_optimization(numbers, positions, **controls) -> dict[str, Any]:
    """Relax an excited-state geometry by L-BFGS on its analytic gradient.
    The default is the restricted singlet TDA Z-vector model. Explicit
    ``td_gradient_route="path"`` selects compatible nuclear-path models,
    including the tested HF triplet route. Reports identify the actual
    method, spin channel and gradient route. General spinor optimization
    is not connected; state following retains the squared-amplitude
    heuristic and does not certify crossings or degenerate surfaces."""
    return run_feature("excited-optimization", numbers, positions, **controls)


def optical_activity(numbers, positions, **controls) -> dict[str, Any]:
    """Optical rotation at the photon energies ``frequencies``: the electric
    dipole - magnetic dipole polarizability of the response in the length
    gauge about ``gauge_origin``, ``beta`` and the specific rotation."""
    return run_feature("optical-activity", numbers, positions, **controls)


def broken_symmetry_exchange(numbers, positions, **controls) -> dict[str, Any]:
    """Heisenberg exchange coupling J (H = -2 J S_A . S_B) of the sites
    ``site_a`` and ``site_b`` from the high-spin determinant (the
    ``multiplicity`` of the controls) and the broken-symmetry determinant."""
    return run_feature("broken-symmetry-exchange", numbers, positions, **controls)


def vcd(numbers, positions, **controls) -> dict[str, Any]:
    """Vibrational circular dichroism: atomic axial tensors with a common
    gauge origin and the rotational strengths of the harmonic modes."""
    return run_feature("vcd", numbers, positions, **controls)


def spin_susceptibility(numbers, positions, **controls) -> dict[str, Any]:
    """Static spin susceptibility chi = g_e^2 mu_B^2 / eta of an unrestricted
    reference with a positive ``electronic_temperature``, from the ensemble
    response to a spin transfer at fixed electron number."""
    return run_feature("spin-susceptibility", numbers, positions, **controls)


def spin_spin_zfs(numbers, positions, *, nuclear_order=0, **controls) -> dict[str, Any]:
    """Direct electron spin-spin D in H_D=S.D.S; SOC and spin projection excluded.

    Integer all-electron UHF/HF only, multiplicity >= 3. Omitted reference
    selects the dedicated UHF route; explicit reference must be ``"uhf"``.
    Strict integer ``nuclear_order=0/1/2`` selects D [3,3] in Hartree,
    dD/dR [3N,3,3] in Hartree/Bohr and d2D/dR2 [3N,3N,3,3] in
    Hartree/Bohr^2. Unrequested derivatives are None. Coordinates are
    atom-major xyz. Raw SCF/response diagnostics and managed workspace
    accounting are retained; nonconvergence raises an error. This is not
    total ZFS, a vibrational correction or experimental validation.
    """
    return run_feature("spin-spin-zfs", numbers, positions, nuclear_order=nuclear_order, **controls)


def g_tensor(numbers, positions, **controls) -> dict[str, Any]:
    """Electronic g tensor of an unrestricted open-shell reference:
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
    routing. Neither route is a complete many-electron relativistic model."""
    return run_feature("g-tensor", numbers, positions, **controls)


def roks(numbers, positions, **controls) -> dict[str, Any]:
    """Restricted open-shell Kohn-Sham energy of the state of
    ``multiplicity`` with one set of spatial orbitals, and its analytic
    gradient (``gradient=False`` skips it)."""
    return run_feature("roks", numbers, positions, **controls)


def roks_hessian(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic Hessian of the restricted open-shell Kohn-Sham energy of the
    state of ``multiplicity``, from the second-order Taylor coefficients of
    the energy along nuclear paths, with its gradient and the harmonic
    frequencies."""
    return run_feature("roks-hessian", numbers, positions, **controls)


def roks_polarizability(numbers, positions, **controls) -> dict[str, Any]:
    """Dipole moment and static polarizability of the restricted open-shell
    Kohn-Sham state of ``multiplicity``, from the Taylor series of the
    energy along field paths."""
    return run_feature("roks-polarizability", numbers, positions, **controls)


def divide_conquer(numbers, positions, **controls) -> dict[str, Any]:
    """Divide-and-conquer Kohn-Sham energy and gradient: subsystems of
    fragments (``dc_fragments``, one fragment index per atom; default one
    atom per fragment) with buffers of radius ``dc_buffer`` (Bohr), a common
    Fermi level with smearing ``dc_smearing`` (Hartree)."""
    return run_feature("dc", numbers, positions, **controls)


def mrsf(numbers, positions, **controls) -> dict[str, Any]:
    """Mixed-reference spin-flip TDA: ``nstates`` singlets and triplets on
    the restricted open-shell triplet of the molecule (the input names the
    singlet or the triplet); energies ``omega`` are relative to that
    triplet."""
    return run_feature("mrsf", numbers, positions, **controls)


def mrsf_state(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic gradient of the total energy of one mixed-reference
    spin-flip state (``mrsf_state="S0"``, ``"S1"``, ``"T0"``, ...) on the
    restricted open-shell triplet, from Taylor series along nuclear paths."""
    return run_feature("mrsf-state", numbers, positions, **controls)


def mrsf_hessian(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic Hessian, gradient and harmonic frequencies of the total
    energy of one mixed-reference spin-flip state (``mrsf_state``)."""
    return run_feature("mrsf-hessian", numbers, positions, **controls)


def spin_orbit(numbers, positions, **controls) -> dict[str, Any]:
    """One-shot spin-orbit coupling: the one-electron spin-orbit operator
    (nuclear or effective ``soc_charges``) diagonalized in the Kohn-Sham
    spin orbitals for the spin axis ``spin_axes`` (default z).

    Select reference="gks" for self-consistency with the full spin-covariant
    functional. Read the paired self_consistent_energies and
    self_consistent_orbital_residuals, both in Hartree.
    """
    return run_feature("spin-orbit", numbers, positions, **controls)


def spinor_polarizability(numbers, positions, *, frequencies, **controls) -> dict[str, Any]:
    """Dynamic polarizability of the self-consistent two-component state
    with the one-electron spin-orbit operator (``soc_charges``, ``soc_scale``)
    in its Fock matrix, at each of ``frequencies`` (Hartree, optional
    ``frequency_damping``): ``polarizability_real`` / ``polarizability_imag``
    as ``[frequency][a][b]`` and the axial vector of the antisymmetric
    imaginary part. ``spin_axis`` is the spin axis of the collinear start."""
    return run_feature("spinor-polarizability", numbers, positions, frequencies=frequencies, **controls)


def magnetic_anisotropy(numbers, positions, **controls) -> dict[str, Any]:
    """Magnetic anisotropy energies from the band energies of the spin axes
    ``spin_axes`` (default x, y, z) with the one-shot spin-orbit coupling
    (magnetic force theorem)."""
    return run_feature("magnetic-anisotropy", numbers, positions, **controls)


def magnon(numbers, positions, **controls) -> dict[str, Any]:
    """Transverse spin excitations: the lowest ``nstates`` spin-lowering and
    spin-raising spin-flip TDA energies of an unrestricted reference
    (collinear kernel, exact exchange only)."""
    return run_feature("magnon", numbers, positions, **controls)


def mcd(numbers, positions, **controls) -> dict[str, Any]:
    """Magnetic circular dichroism of a closed-shell molecule: Faraday B
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
    """
    return run_feature("mcd", numbers, positions, **controls)


def td_hessian(numbers, positions, **controls) -> dict[str, Any]:
    """Analytic Hessian (Hartree/Bohr squared) of one isolated state.
    The total surface is SCF energy plus the selected excitation pole. Use
    ``reference="roks"``/``"rohf"`` for reference-spin common-orbital flag TDA,
    or ``"gks"``/``"ghf"`` for full complex spinor TDA. HF doublet/quartet flag
    and isolated full-spin HF Cartesian subsets are accepted. These reports
    currently contain derivative tensors without the collinear harmonic
    postprocessing. Semilocal spinor molecular verification and correlated pole
    derivatives remain pending; degenerate single-root derivatives are refused."""
    return run_feature("td-hessian", numbers, positions, **controls)


def td_ecd_giao(numbers, positions, **controls) -> dict[str, Any]:
    """Length-gauge rotatory strengths with London orbitals of the TDA or
    RPA states (``td_method``, ``nstates``): ``rotatory_strength_london``
    per state next to the common-origin (``gauge_origin``) and velocity
    values, in atomic units and in 1e-40 erg cm^3."""
    return run_feature("td-ecd-giao", numbers, positions, **controls)


def roa(numbers, positions, **controls) -> dict[str, Any]:
    """Raman optical activity: normal-mode derivatives of the electric dipole
    polarizability, the optical rotation tensor and the dipole-quadrupole
    polarizability at the photon energies ``frequencies`` (Hartree), their
    invariants and the circular intensity differences; ``gauge`` is
    ``velocity`` (default, independent of the origin) or ``length``."""
    return run_feature("roa", numbers, positions, **controls)


def hypermagnetizability(numbers, positions, **controls) -> dict[str, Any]:
    """Static hypermagnetizability ``eta[a][b][c][d] = d2 alpha_ab/(dB_c dB_d)``
    with the static polarizability and the magnetizability, common gauge
    origin ``gauge_origin`` (Bohr; default the centre of nuclear charge)."""
    return run_feature("hypermagnetizability", numbers, positions, **controls)


def cotton_mouton(numbers, positions, **controls) -> dict[str, Any]:
    """Cotton-Mouton constant per molecule at ``temperature`` (K) from the
    hypermagnetizability anisotropy and the orientational term of the
    polarizability and magnetizability anisotropies."""
    return run_feature("cotton-mouton", numbers, positions, **controls)


def magneto_optical(numbers, positions, **controls) -> dict[str, Any]:
    """Magneto-optical tensor at the photon energy ``frequency`` (Hartree).

    An open-shell reference (``magneto_optical_source="spin-orbit"``, the
    automatic choice) gives the first-order change of the polarizability
    under one-electron spin-orbit coupling for the spins along each axis,
    ``magneto_optical_tensor[a][b][k]``. A closed-shell reference
    (``"field"``) gives the Faraday tensor ``Im d alpha_ab(w)/dB_c`` of an
    external field, with London orbitals (``gauge="london"``, default) or a
    common gauge origin."""
    return run_feature("magneto-optical", numbers, positions, **controls)


def magnetic_field_scf(numbers, positions, *, bfield, **controls) -> dict[str, Any]:
    """Self-consistent field in the finite uniform magnetic field ``bfield``
    (atomic units, 1 a.u. = 2.35052e5 T) with London orbitals; the spin
    Zeeman term is included unless ``spin_zeeman=False``. Optional field
    (Hartree/e/Bohr) includes the scalar potential and nuclear coupling, with
    the charged-system energy referenced to the original laboratory origin.
    """
    return run_feature("bfield", numbers, positions, bfield=bfield, **controls)


def magnetic_field_gradient(numbers, positions, *, bfield, **controls) -> dict[str, Any]:
    """Self-consistent field in the finite uniform magnetic field ``bfield``
    (atomic units) with London orbitals and the analytic gradient of its
    energy (``gradient``, Hartree/Bohr)."""
    return run_feature("bfield-gradient", numbers, positions, bfield=bfield, **controls)


def magnetoelectric(numbers, positions, **controls) -> dict[str, Any]:
    """Linear magnetoelectric response of an open-shell reference to first
    order in the one-electron spin-orbit coupling (``soc_charges``):
    ``magnetoelectric_tensor[a][b][k]`` is the electric dipole component
    ``a`` induced per unit magnetic field ``b`` with the spins along ``k``."""
    return run_feature("magnetoelectric", numbers, positions, **controls)


def periodic_energy(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Finite-k periodic energy per cell (Hartree; Bohr geometry and lattice
    rows), with analytic forces when ``forces=True``."""
    return run_feature("periodic-energy", numbers, positions, cell=cell, **controls)


def stress(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Analytic stress of the finite-k periodic energy with forces.

    The field-free spinor path uses the periodic length, area or volume as
    its normalization. In 3D the stress is also given in GPa. The Voigt
    order is (xx, yy, zz, yz, xz, xy). Commensurate scalar and vector
    Fourier controls use fixed reduced wave indices and fixed Cartesian
    amplitudes for the collinear strain derivative."""
    return run_feature("stress", numbers, positions, cell=cell, **controls)


def opt_fixed_cell(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Fixed-cell relaxation of the atoms with analytic finite-k forces
    (``max_steps``, ``gtol`` in Hartree/Bohr, ``step_max`` in Bohr)."""
    return run_feature("opt-fixed-cell", numbers, positions, cell=cell, **controls)


def cell_opt(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Variable-cell relaxation with analytic forces and stress at the
    hydrostatic ``pressure_gpa``; returns the relaxed ``cell`` rows and
    ``positions`` in Bohr."""
    return run_feature("cell-opt", numbers, positions, cell=cell, **controls)


def phonons(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Gamma-point harmonic phonons from analytic coupled-perturbed force
    constants (Hartree/Bohr^2; frequencies in cm^-1, negative = imaginary).
    Optional field (Hartree/e/Bohr) selects the stationary finite-bias Berry
    electric enthalpy on the Gamma supercell of the insulating k mesh.
    This path reports its reference and response residuals explicitly."""
    return run_feature("phonons", numbers, positions, cell=cell, **controls)


def phonon_bands(numbers, positions, cell, *, supercell, path, **controls) -> dict[str, Any]:
    """Phonon dispersion (cm^-1) along reduced q vertices ``path`` from
    analytic force constants of a ``supercell``, optional DOS.
    lattice_flexoelectric=True with wavevector_direction and response_order
    2..4 adds fixed-mass-center internal relaxation at Gamma. Born charges
    are computed on the primitive reference, or supplied as born_charge_rows
    (3N by 3, atom/polarization then displacement). born_lattice_flexoelectric
    is the Born-charge lattice contribution in e/Bohr, contracted twice with
    the normalized Cartesian direction; it excludes clamped-ion and dynamic
    multipole terms. Long-wave coefficient entries carry their order/units.
    """
    return run_feature("phonon-bands", numbers, positions, cell=cell, supercell=supercell, path=path, **controls)


def phonon_thermo(numbers, positions, cell, *, supercell, q_mesh, **controls) -> dict[str, Any]:
    """Harmonic lattice free energy, entropy and heat capacity per primitive
    cell from analytic supercell force constants on a uniform ``q_mesh``."""
    return run_feature("phonon-thermo", numbers, positions, cell=cell, supercell=supercell, q_mesh=q_mesh, **controls)


def electron_phonon(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Electron-phonon coupling lambda, the Eliashberg function,
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
    """
    return run_feature("electron-phonon", numbers, positions, cell=cell, **controls)


def quasi_harmonic(numbers, positions, cell, *, supercell, q_mesh, **controls) -> dict[str, Any]:
    """Quasi-harmonic V(T), isothermal bulk modulus, thermal expansion,
    heat capacities and Grueneisen parameter from analytic supercell phonons
    on isotropically scaled cells (``scales``)."""
    return run_feature("quasi-harmonic", numbers, positions, cell=cell, supercell=supercell, q_mesh=q_mesh, **controls)


def elastic(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Clamped- and relaxed-ion elastic constants (Voigt, GPa and
    Hartree/Bohr^3) from analytic strain, internal-strain and force-constant
    responses."""
    return run_feature("elastic", numbers, positions, cell=cell, **controls)


def elastic_fd(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Validation route: Lagrangian elastic constants (GPa) from central
    differences of the analytic stress, clamped or relaxed ions."""
    return run_feature("elastic-fd", numbers, positions, cell=cell, **controls)


def elastic_moduli(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Voigt, Reuss and Hill moduli (GPa), isotropic wave speeds (m/s) and
    the Debye temperature (K) from the analytic elastic tensor."""
    return run_feature("elastic-moduli", numbers, positions, cell=cell, **controls)


def christoffel(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Christoffel phase velocities (m/s) and polarizations along
    ``directions`` from the analytic elastic tensor."""
    return run_feature("christoffel", numbers, positions, cell=cell, **controls)


def material_card(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Continuum material card from the analytic elastic tensor: stiffness,
    compliance, engineering constants, Hill moduli, density and wave speeds
    (``card_path`` writes JSON or CSV)."""
    return run_feature("material-card", numbers, positions, cell=cell, **controls)


def equation_of_state(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Third-order Birch-Murnaghan fit of isotropically scaled cells
    (``scales``), with analytic pressures; V0 in Bohr^3, B0 in GPa."""
    return run_feature("equation-of-state", numbers, positions, cell=cell, **controls)


def dielectric_tensor(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Electronic (clamped-ion) dielectric tensor ``epsilon_infinity`` by
    periodic density-functional perturbation theory."""
    return run_feature("dielectric-tensor", numbers, positions, cell=cell, **controls)


def born_charges(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Born effective charges (elementary charges) and the electronic
    dielectric tensor by periodic density-functional perturbation theory."""
    return run_feature("born-charges", numbers, positions, cell=cell, **controls)


def piezoelectric_tensor(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Proper piezoelectric tensor of an insulating crystal: clamped-ion
    (strain response of the polarization) and relaxed-ion (Born charges with
    the internal relaxation), in e/bohr^2 and C/m^2."""
    return run_feature("piezoelectric-tensor", numbers, positions, cell=cell, **controls)


def berry_polarization(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Berry-phase polarization of an insulating cell (e/Bohr^2 and C/m^2),
    defined modulo the polarization quanta; ``berry_mesh`` defaults to the
    SCF k mesh."""
    return run_feature("berry-polarization", numbers, positions, cell=cell, **controls)


def crystal_ir(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Gamma transverse modes, mode effective charges, infrared intensities
    (km/mol) and LO frequencies along ``directions``."""
    return run_feature("crystal-ir", numbers, positions, cell=cell, **controls)


def static_dielectric(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Static (relaxed-ion) dielectric tensor ``epsilon_static`` with the
    electronic tensor, Born charges and Gamma modes."""
    return run_feature("static-dielectric", numbers, positions, cell=cell, **controls)


def electronic_structure(numbers, positions, cell, **controls) -> dict[str, Any]:
    """SCF bands and populations; ``coefficients=True`` adds complex Bloch
    orbitals and F/S/P matrices with raw residuals, in the SCF parent model."""
    return run_feature("electronic-structure", numbers, positions, cell=cell, **controls)


def effective_mass(numbers, positions, cell, *, k_point, band, **controls) -> dict[str, Any]:
    """Analytic band velocity and inverse effective-mass tensor of ``band``
    (zero-based) at the reduced ``k_point``; for a degenerate band give a
    Cartesian ``direction`` for the branch masses."""
    return run_feature("effective-mass", numbers, positions, cell=cell, k_point=k_point, band=band, **controls)


def boltzmann_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls) -> dict[str, Any]:
    """Scalar-relaxation-time, rigid-band Boltzmann transport: sigma/tau,
    Seebeck and electronic thermal conductivity per temperature and chemical
    potential (``transport_mu_shifts`` in Hartree or
    ``carrier_concentrations_cm3``). Optional ``scattering_table`` rows
    [absolute band energy in Hartree, positive lifetime in fs] include the
    energy-dependent lifetime and return conductivity in S/m and electronic
    thermal conductivity in W/(m K), without division by tau. Interpolation
    is log-linear with constant endpoint tails. Frequencies may accompany
    the table; do not also give ``relaxation_time_fs``."""
    return run_feature(
        "boltzmann-transport",
        numbers,
        positions,
        cell=cell,
        transport_k_mesh=transport_k_mesh,
        transport_temperatures_k=transport_temperatures_k,
        **controls,
    )


def hall_transport(numbers, positions, cell, *, transport_k_mesh, transport_temperatures_k, **controls) -> dict[str, Any]:
    """Ordinary weak-field Hall transport of isolated rigid bands, with
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
    references use the same band operator and scattering provider."""
    return run_feature(
        "hall-transport", numbers, positions, cell=cell,
        transport_k_mesh=transport_k_mesh,
        transport_temperatures_k=transport_temperatures_k, **controls,
    )


def coherent_transport(numbers, positions, cell, *, transport_k_mesh, transport_frequencies_hartree, **controls) -> dict[str, Any]:
    """Covariant independent-particle intra/interband conductivity, including
    exact degenerate blocks. SI S/m tensors have last axis [real, imaginary].
    Defaults: transport_temperature_k=300, scattering_hartree=0.001,
    transport_mu_shift=0 relative to the reference level. Includes physical
    AO position connections; exp(-i omega t), signed angular frequencies."""
    return run_feature("coherent-transport", numbers, positions, cell=cell,
                       transport_k_mesh=transport_k_mesh,
                       transport_frequencies_hartree=transport_frequencies_hartree, **controls)


def quantum_hall(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Non-Abelian Chern invariant and integer Hall conductance (S) of a
    positive-gap occupied Bloch slice. topology_mesh=[8,8],
    topology_plane_axes=[0,1], topology_fixed_k=0 by default. Zero-temperature
    insulating reference; refine the mesh to verify the gap and invariant.
    The zero-field collinear reference has no orbital TR-breaking field/SOC."""
    return run_feature("quantum-hall", numbers, positions, cell=cell, **controls)


def wannier_flow(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Hybrid Wannier charge centres of the occupied bands along the loop
    axis as a function of the transverse wave vector along the flow axis
    (``topology_plane_axes=[loop, flow]``, ``wannier_loop_divisions``,
    ``wannier_flow_points``); the winding of their sum is the Chern number."""
    return run_feature("wannier-flow", numbers, positions, cell=cell, **controls)


def periodic_cubic_fc_along(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Cubic force constants of a cell (Hartree/Bohr^3 per primitive cell)
    contracted with the Gamma-point displacement pattern ``probe`` (one
    value per Cartesian coordinate of the cell): the derivative of the
    Gamma-point force constants along it, from first-order responses of the
    Gamma-point supercell of the k mesh. Optional field (Hartree/e/Bohr)
    differentiates the stationary finite-bias Berry electric enthalpy at
    fixed Cartesian electric bias."""
    return run_feature("periodic-cubic-fc-along", numbers, positions, cell=cell, **controls)


def periodic_quartic_fc_along(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Quartic force constants of a cell (Hartree/Bohr^4 per primitive cell)
    contracted with the Gamma-point displacement patterns ``probe`` and
    ``second_probe`` (default the same pattern). Optional field
    (Hartree/e/Bohr) selects the stationary finite-bias Berry enthalpy."""
    return run_feature("periodic-quartic-fc-along", numbers, positions, cell=cell, **controls)


def periodic_cubic_fc(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Complete cubic force constants of a cell (Hartree/Bohr^3 per primitive
    cell) for Gamma-point displacements, from first-order responses of the
    Gamma-point supercell of the k mesh. Optional field (Hartree/e/Bohr)
    selects the stationary finite-bias Berry electric enthalpy."""
    return run_feature("periodic-cubic-fc", numbers, positions, cell=cell, **controls)


def periodic_quartic_fc(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Complete quartic force constants of a cell (Hartree/Bohr^4 per
    primitive cell) for Gamma-point displacements. Optional field
    (Hartree/e/Bohr) selects the stationary finite-bias Berry enthalpy."""
    return run_feature("periodic-quartic-fc", numbers, positions, cell=cell, **controls)


def toec(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Third-order Lagrangian (Brugger) elastic constants at clamped and
    relaxed ions (``high_order_clamped``, ``high_order_relaxed`` in
    Hartree/Bohr^3, also in GPa), Voigt order xx, yy, zz, yz, xz, xy."""
    return run_feature("toec", numbers, positions, cell=cell, **controls)


def foec(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Fourth-order Lagrangian (Brugger) elastic constants at clamped and
    relaxed ions, with the second-order relaxation of the atoms."""
    return run_feature("foec", numbers, positions, cell=cell, **controls)


def mode_gruneisen(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Mode Grueneisen parameters of the Gamma-point phonons from the strain
    derivative of the force constants (``gruneisen_shear=True`` adds the
    three shear strains). ``gruneisen_order=2`` selects analytic isotropic
    log-volume first/second derivatives, with ``relaxed_ion=False`` by default.
    Second order returns ``second_log_frequency = -d gamma/d ln(V)`` and
    undefined acoustic/unstable coefficients as null. Relaxed ions require
    ``gruneisen_force_tolerance`` internal stationarity (Hartree/Bohr in atomic
    units). The second-order route does not include shear derivatives."""
    return run_feature("mode-gruneisen", numbers, positions, cell=cell, **controls)


def periodic_beta(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Clamped-ion electric susceptibility of a cell and its field derivative
    ``second_susceptibility`` = d2P/dE2 (chi2 is half of it), from the
    single-point Berry phase of the Gamma-point supercell of the k mesh."""
    return run_feature("periodic-beta", numbers, positions, cell=cell, **controls)


def periodic_gamma(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Static clamped-ion d3P/dE3 from the Berry-phase electric enthalpy.
    ``tensor``/``third_susceptibility`` are flat [3,3,3,3] actual derivatives;
    ``chi3`` divides them by six. The Gaussian susceptibility derivative
    unit is e^2 Bohr^2/Hartree^2. ``use_symmetry=True`` evaluates 15 independent
    static components and expands their permutations. Set it to False to
    evaluate all 81 components and measure the raw permutation residual;
    that residual is None when symmetry expansion is used. Path residuals
    are always freshly checked with the response operator.
    Second-order internal relaxation and dynamic response remain separate.
    """
    return run_feature("periodic-gamma", numbers, positions, cell=cell, **controls)


def periodic_raman(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Born charges and the displacement derivative of the susceptibility
    (``raman_tensor``, Bohr^-1) of a periodic cell."""
    return run_feature("periodic-raman", numbers, positions, cell=cell, **controls)


def electrostriction(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Strain derivative of the electric susceptibility of a periodic cell
    at clamped ions and fixed reduced field."""
    return run_feature("electrostriction", numbers, positions, cell=cell, **controls)


def piezo_higher(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Proper clamped-ion piezoelectric tensor of a periodic cell and its
    strain derivative."""
    return run_feature("piezo-higher", numbers, positions, cell=cell, **controls)


def periodic_efield(numbers, positions, cell, *, field, **controls) -> dict[str, Any]:
    """Self-consistent field of a periodic cell in the finite uniform
    electric field ``field`` (atomic units), with the Berry-phase dipole and,
    with ``forces=True``, analytic forces."""
    return run_feature("periodic-efield", numbers, positions, cell=cell, field=field, **controls)


def dielectric_q(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]:
    """Self-consistent density response ``chi(q, q)`` and inverse dielectric
    function at the reduced wave vectors ``q_points`` (flat list, three
    numbers each), which must be commensurate with the k mesh."""
    return run_feature("dielectric-q", numbers, positions, cell=cell, q_points=q_points, **controls)


def orbital_susceptibility_q(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]:
    """Orbital magnetic susceptibility ``chi(q)`` of a periodic cell at the
    reduced wave vectors ``q_points`` commensurate with the k mesh, for two
    transverse polarizations of the vector potential. GKS and GHF also return
    the phase-averaged ``chi_transverse`` tensor in the two magnetic-field axes.
    See the API guide for the reference and functional limits.
    """
    return run_feature("orbital-susceptibility-q", numbers, positions, cell=cell, q_points=q_points, **controls)


def periodic_dc(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Divide-and-conquer Kohn-Sham free energy of a periodic cell at Gamma
    (``dc_fragments``, ``dc_buffer`` in Bohr by the minimum image,
    ``dc_smearing`` in Hartree), with ``forces=True`` and ``stress=True``
    for the derivatives including the response of the partition."""
    return run_feature("periodic-dc", numbers, positions, cell=cell, **controls)


def dc_stress(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Stress of the divide-and-conquer free energy of a periodic cell at
    Gamma (Hartree/Bohr^3 and GPa)."""
    return run_feature("dc-stress", numbers, positions, cell=cell, **controls)


def periodic_nmr_shielding(numbers, positions, cell, *, q_points, **controls) -> dict[str, Any]:
    """Nuclear magnetic shielding in a periodic cell at the reduced wave
    vectors ``q_points`` commensurate with the k mesh (``nuclei`` selects
    atoms of the cell); wave vectors along two axes give the full tensor.
    The bulk magnetization (G = 0) term is not included."""
    return run_feature("periodic-nmr-shielding", numbers, positions, cell=cell, q_points=q_points, **controls)


def berry_dipole(numbers, positions, cell, *, transport_k_mesh, **controls) -> dict[str, Any]:
    """Berry curvature dipole of the Fermi surface on ``transport_k_mesh`` at
    ``transport_temperature_k``, with the chemical potential shifted by
    ``transport_mu_shift`` (Hartree) or fixed by ``berry_dipole_doping``
    (electrons per cell)."""
    return run_feature("berry-dipole", numbers, positions, cell=cell, transport_k_mesh=transport_k_mesh, **controls)


def lab_frame_spin_spiral(numbers, positions, cell, *, spiral_q, **controls) -> dict[str, Any]:
    """Energy of a flat spin spiral of reduced wave vector ``spiral_q`` by a
    two-component SCF in the smallest supercell that holds it (``k_mesh``
    must be divisible by it). The spins turn about ``spiral_axis``; the state
    is seeded by a Zeeman field of strength ``spiral_seed_field`` and then
    converged again without it. ``nuclear_soc_scale`` (default 0) adds the
    one-electron nuclear spin-orbit term. Positions/cell are Bohr, energies
    Hartree per cell of the structure; pbc selects physical periodic axes."""
    return run_feature("lab-frame-spin-spiral", numbers, positions, cell=cell, spiral_q=spiral_q, **controls)


def dmi(numbers, positions, cell, *, spiral_q, **controls) -> dict[str, Any]:
    """Chirality-odd spin-orbit energy of the flat spin spiral of reduced wave
    vector ``spiral_q``: half the difference of the nuclear spin-orbit energy
    of the two chiralities, to first order in ``nuclear_soc_scale`` on the
    states converged without it, and with ``dmi_self_consistent=True`` also
    from SCF runs with the spin-orbit term. Hartree per cell of the structure."""
    return run_feature("dmi", numbers, positions, cell=cell, spiral_q=spiral_q, **controls)


def bands(numbers, positions, cell, *, path, **controls) -> dict[str, Any]:
    """Band structure along ``path`` (reduced k vertices)."""
    return run_feature("bands", numbers, positions, cell=cell, path=path, **controls)

def periodic_spin_response(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Longitudinal spin-only Zeeman response, H'_alpha/beta=+/-g B_z N(q)/4.

    response_q is Cartesian inverse Bohr; frequency and frequency_damping
    use Hartree, with imaginary=True for the imaginary axis. spin_probe_field
    is [real, imaginary] magnetic atomic units (default [1,0]); spin_g_factor
    defaults to 2. Results are per cell; complex values end in [real,imag].
    Orbital fields, transverse spin, SOC and spin relaxation are separate.
    The uniform static equilibrium Pauli limit uses pauli_spin_susceptibility
    with exchange_enhanced=True; it is refused here instead of independently
    fixing both spin populations. Geometry and lattice rows are Bohr."""
    return run_feature("periodic-spin-response", numbers, positions, cell=cell, **controls)


def periodic_field_scf(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Complex self-consistent collinear 1D/2D/3D reference in a real static
    periodic field. field_wave_indices=[1,0,0], field_vector_real/imag=[0,0,0],
    field_scalar=[0,0], field_amplitude=1. A(r)=lambda*(Aq*exp(iqr)+conjugate).
    Includes both paramagnetic and A^2 terms and complex variational exchange.
    Energy/free energy and analytic amplitude derivative are Hartree per cell.
    Zero net magnetic flux; spin Zeeman, SOC and uniform flux are separate models.
    Native geometry is Bohr; use pbc to select physical periodic axes.
    Optional forces and stress differentiate the commensurate Fourier field
    at fixed reduced wave indices and fixed Cartesian amplitudes."""
    return run_feature("periodic-field-scf", numbers, positions, cell=cell, **controls)


def periodic_spinor_scf(numbers, positions, cell, **controls) -> dict[str, Any]:
    """General complex two-component 1D/2D/3D SCF with one chemical potential.
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
    Positions/cell are Bohr; pbc selects physical periodic axes."""
    return run_feature("periodic-spinor-scf", numbers, positions, cell=cell, **controls)


def periodic_magnetic_scf(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Uniform-flux London spinor SCF on a physical 2D/3D magnetic cell.
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
    units, raw residuals, acceptance scope and remaining providers."""
    return run_feature("periodic-magnetic-scf", numbers, positions, cell=cell, **controls)


def dos(numbers, positions, cell, *, sigma, **controls) -> dict[str, Any]:
    """Gaussian-broadened electronic DOS on the physical periodic axes.

    The native geometry is in Bohr and ``sigma`` is in Hartree.
    ``projected=True`` adds Mulliken weights by atom and angular momentum.
    ROKS uses separate spin spectra within its common-orbital subspaces.
    """
    return run_feature("dos", numbers, positions, cell=cell, sigma=sigma, **controls)


def pauli_spin_susceptibility(numbers, positions, cell, **controls) -> dict[str, Any]:
    """Bare equal/unequal-channel rigid-band thermodynamics reoccupies the
    frozen bands with a common chemical potential and zero reservoir spin
    bias, at fixed total particle number. It includes the common-potential
    magnetic derivative, heat capacity and thermally averaged spin DOS.
    ``exchange_enhanced=True`` selects self-consistent static spin response.
    The enhanced semilocal mode holds the baseline reference spin bias and
    converges a separate thermal reference for each ``temperatures_k`` and
    uses ``fermi_k_mesh`` as its SCF/response mesh. Spin populations exchange
    particles at fixed total number to first order. ``spin_g_factor=2``;
    no orbital or SOC contribution. Reports units and raw residuals."""
    return run_feature("pauli-spin-susceptibility", numbers, positions, cell=cell, **controls)


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``dft-rs`` command line and return its process exit status.

    Explicit ``argv`` omits the program name. The installed console script
    calls this without arguments, using ``sys.argv[1:]``. The compiled core
    supplies the parser and calculations; no external executable is needed.
    """
    if argv is None:
        import sys
        argv = sys.argv[1:]
    return _core.cli_main([str(a) for a in argv])


def piezoelectric_strain(*, piezoelectric_stress, piezoelectric_compliance,
                        ion_response) -> dict[str, Any]:
    """Convert supplied proper e and fixed-field S to d=eS without SCF.

    Inputs are 3x6 e/Bohr^2 and 6x6 Bohr^3/Hartree, in engineering Voigt
    order xx,yy,zz,yz,xz,xy with identical axes and ion_response='clamped'
    or 'relaxed'. Returns d in e Bohr/Hartree, C/N and pm/V. Neither the
    physical reference nor mechanical stability is validated by this conversion.
    """
    return run_feature("piezoelectric-strain", piezoelectric_stress=piezoelectric_stress,
                       piezoelectric_compliance=piezoelectric_compliance,
                       ion_response=ion_response)


def aggregate_moduli(*, phases) -> dict[str, Any]:
    """Voigt, Reuss, Hill, Hashin-Shtrikman and self-consistent moduli (GPa)
    of an isotropic aggregate; ``phases`` rows are [fraction, K, G]."""
    return run_feature("aggregate-moduli", phases=[[float(v) for v in row] for row in phases])


def scdft_pairing_kernel(**inputs) -> dict[str, Any]:
    """Static LM2005 singlet pairing and phonon mass on supplied normal
    bands. Energies and chemical potential are Hartree, squared weighted
    electron-phonon vertices are Hartree squared, temperature is Kelvin.
    Supplying band_weights also generates dimensionless diagonal Z; K
    excludes these final-band weights and Z includes them once. Optional
    energy, chemical-potential, temperature, phonon, vertex, screened-Coulomb
    and weight directions give Taylor coefficients through order four.
    Reports separate K/Z quadrature errors; no gap or density backreaction
    is inferred. The mass uses the particle-hole-symmetric approximation.
    """
    return run_feature("scdft-pairing-kernel", **inputs)


def superconducting_state(**inputs) -> dict[str, Any]:
    """Static superconducting gap or full Nambu density response, orders 0..4.

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
    """
    return run_feature("superconducting-state", **inputs)


def dirac_coulomb_scf(numbers, positions, **controls) -> dict[str, Any]:
    """Finite-nucleus no-pair four-component Hartree-Fock in atomic units.

    Optional dirac_interaction selects coulomb, gaunt or breit. The positive
    sector is fixed by the nuclear core and the current energies are reported
    as separate direct and signed exchange terms.
    """
    return run_feature("dirac-coulomb-scf", numbers, positions, **controls)


def two_component_relativistic(numbers, positions, **controls) -> dict[str, Any]:
    """Finite-nucleus X2C, DKH2 or matrix ZORA with scalar or spin-orbit SCF.

    Give positions and nuclear_rms_radii in Bohr. The common picture
    transforms one-electron observables and the interaction density.
    response_order=1 returns the uniform electric polarizability, with
    optional frequencies and positive damping in Hartree.
    """
    return run_feature("two-component-relativistic", numbers, positions, **controls)


def dirac_dks(numbers, positions, **controls) -> dict[str, Any]:
    """Fixed-sector four-component Dirac–Coulomb KS in atomic units."""
    return run_feature("dirac-dks", numbers, positions, **controls)


def no_pair_projector(numbers=None, positions=None, **inputs) -> dict[str, Any]:
    """Electronic-sector projection of balanced four-component H/S.
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
    """
    return run_feature("no-pair-projector", numbers, positions, **inputs)


def rock_physics(**inputs) -> dict[str, Any]:
    """Dry frame, Gassmann fluid substitution and Biot parameters from
    supplied mineral and fluid data (no defaults for physical inputs).

    ``frame`` is ``critical-porosity``, ``hertz-mindlin``,
    ``hashin-shtrikman``, ``given``, or one of the spheroidal-pore models
    ``dem``, ``kuster-toksoz`` and ``self-consistent`` (these need
    ``pore_aspect_ratio``). ``mineral_conductivity_w_mk`` together with
    ``fluid_conductivity_w_mk`` adds ``thermal_conductivity_w_mk``."""
    return run_feature("rock-physics", **inputs)


def continuous_species(numbers, positions, *, fixed_electron_count, charge_directions,
                       charge_coordinates, charge_bounds, **controls) -> dict[str, Any]:
    """Fixed-template real nuclear-charge collinear model, in atomic units.

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
    """
    controls.update(fixed_electron_count=fixed_electron_count,
                    charge_directions=charge_directions,
                    charge_coordinates=charge_coordinates, charge_bounds=charge_bounds)
    return run_feature("continuous-species", numbers, positions, **controls)


def generate_methyl_fragment(numbers, positions, *, fragment_spec, **controls) -> dict[str, Any]:
    """Generate a fixed one-center methyl candidate through the common driver.

    Positions and optional construction ESP probes use Bohr. ``fragment_spec``
    declares explicit/retained atom indices, the three port anchors, bonds and
    one electronic center. The returned ``model_json`` is opaque native-unit
    data; its presence does not establish chemical accuracy. Frozen and active
    electron counts, projection residuals and candidate status are reported.
    """
    return run_feature("continuous-species", numbers, positions, fragment_mode="generate",
                       fragment_spec=fragment_spec, **controls)


def evaluate_reduced_fragment(numbers, positions, *, model, fragment_mode="evaluate", **controls) -> dict[str, Any]:
    """Evaluate a saved reduced candidate in atomic units.

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
    """
    if fragment_mode not in ("evaluate", "few-evaluate", "rigid-evaluate", "scalar-evaluate", "scalar-optimize",
                             "scalar-gradient", "scalar-geometry-optimize", "scalar-hessian"):
        raise ValueError("unsupported saved fragment evaluation mode")
    return run_feature("continuous-species", numbers, positions, fragment_mode=fragment_mode,
                       fragment_model_json=model, **controls)


def multicomponent(numbers, positions, *, quantum_atoms=None, **controls) -> dict[str, Any]:
    """Experimental simultaneous electronic/quantum-nuclear SCF, in atomic units.

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
    """
    if quantum_atoms is not None:
        controls["quantum_atoms"] = quantum_atoms
    return run_feature("multicomponent", numbers, positions, **controls)


def casscf(numbers, positions, *, cas, **controls) -> dict[str, Any]:
    """Complete-active-space SCF through the common driver, in atomic units.

    ``cas=[active electrons, active orbitals]``; the remaining electrons fill
    doubly occupied core orbitals. The reference SCF (``xc``, normally
    ``hf``) supplies starting orbitals; ``active_orbitals`` names them
    explicitly. ``cas_roots`` lowest states of the reference multiplicity are
    averaged with ``state_weights`` (equal by default); ``casci=True`` keeps
    the starting orbitals. Energies are Hartree.
    """
    return run_feature("casscf", numbers, positions, cas=cas, **controls)


def mcpdft(numbers, positions, *, cas, **controls) -> dict[str, Any]:
    """Multiconfiguration pair-density functional theory, in atomic units.

    Takes the ``casscf`` controls plus ``on_top_functional`` (``t<name>`` or
    ``ft<name>`` for a registered local or gradient-corrected functional,
    ``tpbe0``/``ftpbe0``; default ``tpbe``), ``pdft_lambda`` (hybrid fraction
    of the active-space energy) and ``pdft_method`` (``state-specific``,
    ``l-pdft``, ``cms-pdft`` or ``xms-pdft``). Energies are Hartree.
    """
    return run_feature("mcpdft", numbers, positions, cas=cas, **controls)


def feature_registry() -> dict[str, Any]:
    """The registry of public calculations as data."""
    return run_feature("feature-registry")


__all__ += [
    "continuous_species",
    "generate_methyl_fragment",
    "evaluate_reduced_fragment",
    "multicomponent",
    "casscf",
    "mcpdft",
    "superconducting_state",
    "no_pair_projector",
    "dirac_coulomb_scf",
    "two_component_relativistic",
    "dirac_dks",
    "scdft_pairing_kernel",
    "Report",
    "run_feature",
    "energy",
    "mermin_free_energy",
    "gibbs_free_energy",
    "free_energy",
    "gradient",
    "charges",
    "dipole",
    "orbitals",
    "molecular_multipole",
    "quadrupole_polarizability",
    "multipole_polarizability",
    "molecular_medium_optics",
    "write_molden",
    "save_orbitals",
    "conceptual_dft",
    "spin_conceptual_dft",
    "hessian_report",
    "frequencies_report",
    "vibrational_spectra",
    "polarizability_report",
    "multipole_gradient",
    "response_derivatives",
    "cli_main",
    "periodic_energy",
    "electronic_structure",
    "bands",
    "dos",
    "feature_registry",
]


def _reporting(function):
    """Return dictionary results of ``function`` as :class:`Report`."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        for name in ("grid_level", "vv10_grid_level"):
            if name in kwargs:
                value = kwargs[name]
                try:
                    integer = operator.index(value)
                except TypeError:
                    raise ValueError(f"{name} must be an integer between 0 and 9") from None
                if isinstance(value, bool) or not 0 <= integer <= 9:
                    raise ValueError(f"{name} must be an integer between 0 and 9")
                kwargs[name] = integer
        logging = {name: kwargs.pop(name) for name in ("log_file", "log_mode", "log_level") if name in kwargs}
        logging = {name: value for name, value in logging.items() if value is not None}
        if "log_file" in logging:
            try:
                path = os.fspath(logging["log_file"])
            except TypeError:
                raise ValueError("log_file must be a nonempty text path") from None
            if not isinstance(path, str) or not path or "\x00" in path:
                raise ValueError("log_file must be a nonempty text path without null characters")
            logging["log_file"] = path
        for name, allowed in (("log_mode", {"console", "stdout", "file", "both", "none", "quiet"}),
                              ("log_level", {"error", "warn", "warning", "info", "debug"})):
            if name in logging:
                value = logging[name]
                if not isinstance(value, str) or value.lower() not in allowed:
                    raise ValueError(f"invalid {name}; expected one of {sorted(allowed)}")
                logging[name] = value.lower()
        limits = {name: kwargs.pop(name) for name in ("cpu_threads", "memory_mb") if name in kwargs}
        limits = {name: value for name, value in limits.items() if value is not None}
        for name, value in limits.items():
            try:
                integer = operator.index(value)
            except TypeError:
                raise ValueError(f"{name} must be a positive integer") from None
            if isinstance(value, bool) or integer <= 0 or integer > (1 << 63) - 1:
                raise ValueError(f"{name} must be a positive representable integer")
            limits[name] = integer
        if logging and not hasattr(_core, "_logging_active"):
            raise NotImplementedError("this compiled core does not implement scoped logging; rebuild the extension")
        if limits or logging:
            executor = getattr(_core, "_with_resources", None)
            if executor is None:
                raise NotImplementedError("this compiled core does not implement cpu_threads/memory_mb; rebuild the extension")
            result = executor(function, args, kwargs, **limits, **logging)
        else:
            result = function(*args, **kwargs)
        if type(result) is not dict:
            return result
        if not isinstance(result.get("units"), dict):
            # A dedicated binding: label its entries like a driver report.
            attach_unit_labels(result)
        return Report(result)

    signature = inspect.signature(function)
    parameters = list(signature.parameters.values())
    for name in ("cpu_threads", "memory_mb", "log_file", "log_mode", "log_level"):
        if name not in signature.parameters:
            annotation = Optional[int]
            if name == "log_file":
                annotation = Optional[Union[str, os.PathLike]]
            elif name.startswith("log_"):
                annotation = Optional[str]
            position = next((i for i, parameter in enumerate(parameters)
                             if parameter.kind == inspect.Parameter.VAR_KEYWORD), len(parameters))
            parameters.insert(position, inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY,
                                                         default=None, annotation=annotation))
    wrapper.__signature__ = signature.replace(parameters=parameters)
    return wrapper


# Every public function returns its dictionary result as a `Report`, so that
# the dedicated functions and the driver functions read the same way.
for _name, _item in list(globals().items()):
    if not _name.startswith("_") and inspect.isfunction(_item) and _item.__module__ == __name__:
        globals()[_name] = _reporting(_item)
del _name, _item
