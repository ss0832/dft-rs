# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# GIAO NMR oracle for the asymmetric water geometry used by tests/p15_nmr.rs.
# The output contains raw atomic-unit total/diamagnetic/paramagnetic tensors;
# PySCF's public NMR kernel converts these values to ppm, so this generator
# deliberately calls the lower-level dia() and para() methods instead.
#
# Meta-GGA is intentionally absent.  pyscf-properties currently raises
# NotImplementedError("meta-GGA") in its GIAO XC matrix implementation.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_nmr.py

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


# Exact Bohr literals copied from tests/p15_nmr.rs::skew_water.
GEOMETRY = (
    ("O", 8, 0.16, -0.24, 0.31),
    ("H", 1, 1.41, 0.38, -0.62),
    ("H", 1, -0.72, 1.22, 0.47),
)

# (dft-rs name, PySCF/libxc expression); None is the independent HF control.
CASES = (
    ("hf", None),
    ("svwn5", "slater,vwn"),
    ("pbe", "pbe,pbe"),
    ("blyp", "b88,lyp"),
    ("b3lyp", "b3lypg"),
    ("cam-b3lyp", "camb3lyp"),
)

GRID_LEVELS = (8, 9)
GRID_CONVERGENCE_PPM = 5e-3
AXES = ("x", "y", "z")
OUTPUT_NAME = "skew_water_sto3g_nmr.csv"
PYSCF_PROPERTIES_COMMIT = "4eee5a430fb47eca5962f36fdcaf75c2b87e7ede"
PYSCF_PROPERTIES_RHF_SHA256 = "5d11ef7f2b18edaf2b1d9dc577d7d81ea455c25255f0c928c6d1153897ec7308"
PYSCF_PROPERTIES_RKS_SHA256 = "4e5d6e88c95d01284e000070a0f3738b71cf40a6fc6ba977f5234b3e712ab122"


def dependency_error(message: str, exc: BaseException | None = None) -> None:
    detail = f" ({exc})" if exc is not None else ""
    raise SystemExit(
        "dependency error: "
        + message
        + detail
        + "; install compatible 'pyscf' and 'pyscf-properties' packages"
    )


def load_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_error("NumPy, required by PySCF, is unavailable", exc)

    try:
        import pyscf
        from pyscf import dft, gto, scf
        from pyscf.data import nist
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_error("PySCF is unavailable", exc)

    try:
        from pyscf.prop import nmr
        from pyscf.prop.nmr import rhf as nmr_rhf
        from pyscf.prop.nmr import rks as nmr_rks
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_error("the pyscf-properties NMR extension is unavailable", exc)
    if not hasattr(nmr, "RHF") or not hasattr(nmr, "RKS"):
        dependency_error(
            "the pyscf-properties installation does not expose RHF and RKS NMR drivers"
        )

    # Import only after the dependency checks: gen_integrals itself imports
    # PySCF and NumPy at module load time.
    from gen_integrals import DATA_DIR, OUT_DIR, parse_g94

    return {
        "np": np,
        "pyscf": pyscf,
        "dft": dft,
        "gto": gto,
        "scf": scf,
        "nist": nist,
        "nmr": nmr,
        "nmr_rhf": nmr_rhf,
        "nmr_rks": nmr_rks,
        "data_dir": Path(DATA_DIR),
        "out_dir": Path(OUT_DIR),
        "parse_g94": parse_g94,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_canonical() -> str:
    return "\n".join(
        f"{symbol},{z},{x:.17g},{y:.17g},{zcoord:.17g}"
        for symbol, z, x, y, zcoord in GEOMETRY
    )


def module_sha256(module: Any) -> str:
    path = getattr(module, "__file__", None)
    if not path:
        return "unavailable"
    try:
        return sha256_file(Path(path))
    except OSError:
        return "unavailable"


def properties_version() -> str:
    try:
        return importlib.metadata.version("pyscf-properties")
    except importlib.metadata.PackageNotFoundError:
        # Editable source trees may have no distribution metadata.  The
        # rhf.py/rks.py hashes written beside this value still pin the code.
        return "unregistered-source-tree"


def libxc_version(libxc: Any) -> str:
    for name in ("__version__", "libxc_version"):
        value = getattr(libxc, name, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        return str(value)
    return "unavailable"


def callable_name(value: Any) -> str:
    if value is None:
        return "none"
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    return f"{module}.{name}"


def build_molecule(deps: dict[str, Any], basis: dict[str, Any]) -> Any:
    atom = [(symbol, (x, y, zcoord)) for symbol, _z, x, y, zcoord in GEOMETRY]
    return deps["gto"].M(
        atom=atom,
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=0,
        cart=False,
        symmetry=False,
        verbose=0,
    )


def raw_nmr_parts(deps: dict[str, Any], mf: Any, is_hf: bool) -> dict[str, Any]:
    np = deps["np"]
    nmr_driver = deps["nmr"].RHF(mf) if is_hf else deps["nmr"].RKS(mf)
    nmr_driver.gauge_orig = None  # None is GIAO in pyscf-properties.
    nmr_driver.cphf = True
    nmr_driver.conv_tol = 1e-10
    nmr_driver.max_cycle_cphf = 100

    # kernel() prepares the three GIAO field components and stores mo10.  A
    # bare solve_mo1() is not a supported entry point here: without kernel's
    # setup pyscf-properties can hand CPHF a one-component RHS.
    kernel_ppm = np.asarray(nmr_driver.kernel(), dtype=float)
    if nmr_driver.mo10 is None:
        raise RuntimeError("pyscf-properties NMR kernel did not retain mo10")
    dia_source = np.asarray(nmr_driver.dia(gauge_orig=None), dtype=float)
    para_result = nmr_driver.para(mo10=nmr_driver.mo10)
    if not isinstance(para_result, tuple) or len(para_result) < 1:
        raise RuntimeError("unexpected pyscf-properties para() return value")
    para_source = np.asarray(para_result[0], dtype=float)
    total_source = dia_source + para_source

    expected = (len(GEOMETRY), 3, 3)
    for name, tensor in (
        ("total", total_source),
        ("dia", dia_source),
        ("para", para_source),
    ):
        if tensor.shape != expected:
            raise RuntimeError(f"{name} NMR tensor shape {tensor.shape}, expected {expected}")
        if not np.isfinite(tensor).all():
            raise RuntimeError(f"{name} NMR tensor contains a non-finite value")
    scale = deps["nist"].ALPHA**2 * 1e6
    kernel_gap = float(np.max(np.abs(kernel_ppm - total_source * scale)))
    if not np.isfinite(kernel_gap) or kernel_gap >= 1e-9:
        raise RuntimeError(
            f"raw dia+para does not reproduce NMR kernel: {kernel_gap:.6e} ppm"
        )

    # pyscf-properties stores/prints rows as external-field B_x/B_y/B_z.
    # dft-rs stores sigma[m_N_axis][B_axis], so transpose the last two axes.
    return {
        "total": total_source.transpose(0, 2, 1).copy(),
        "dia": dia_source.transpose(0, 2, 1).copy(),
        "para": para_source.transpose(0, 2, 1).copy(),
    }


def run_case(
    deps: dict[str, Any],
    basis: dict[str, Any],
    name: str,
    xc: str | None,
    grid_level: int | None,
) -> dict[str, Any]:
    mol = build_molecule(deps, basis)
    if xc is None:
        mf = deps["scf"].RHF(mol)
    else:
        mf = deps["dft"].RKS(mol)
        mf.xc = xc
        if grid_level is None:
            raise ValueError("DFT NMR case requires an explicit grid level")
        mf.grids.level = grid_level
    mf.chkfile = None
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        level = "none" if grid_level is None else str(grid_level)
        raise RuntimeError(f"{name}/grid-{level} SCF did not converge")

    result = {
        "functional": name,
        "pyscf_xc": "HF" if xc is None else xc,
        "grid_level": grid_level,
        "grid_points": 0 if xc is None else len(mf.grids.weights),
        "grid_prune": "not-applicable" if xc is None else callable_name(mf.grids.prune),
        "energy": float(mf.e_tot),
        "parts": raw_nmr_parts(deps, mf, xc is None),
    }
    level = "none" if grid_level is None else str(grid_level)
    print(f"{name:10s} grid {level:>4s}: E = {mf.e_tot:.12f}")
    return result


def provenance(deps: dict[str, Any], basis_path: Path) -> dict[str, str]:
    geom = geometry_canonical()
    script_path = Path(__file__).resolve()
    rhf_hash = module_sha256(deps["nmr_rhf"])
    rks_hash = module_sha256(deps["nmr_rks"])
    if rhf_hash != PYSCF_PROPERTIES_RHF_SHA256 or rks_hash != PYSCF_PROPERTIES_RKS_SHA256:
        raise RuntimeError(
            "installed pyscf-properties sources do not match pinned commit "
            f"{PYSCF_PROPERTIES_COMMIT}: RHF={rhf_hash}, RKS={rks_hash}"
        )
    return {
        "generator": "tools/oracle/gen_nmr.py",
        "generator_sha256": sha256_file(script_path),
        "pyscf_version": str(deps["pyscf"].__version__),
        "pyscf_properties_version": properties_version(),
        "pyscf_properties_pinned_commit": PYSCF_PROPERTIES_COMMIT,
        "pyscf_properties_rhf_sha256": rhf_hash,
        "pyscf_properties_rks_sha256": rks_hash,
        "libxc_version": libxc_version(deps["dft"].libxc),
        "basis": "sto-3g",
        "basis_source": "src/basis/data/sto-3g.g94",
        "basis_sha256": sha256_file(basis_path),
        "basis_elements": "H,O",
        "geometry_source": "tests/p15_nmr.rs::skew_water",
        "geometry_unit": "Bohr",
        "geometry_canonical": geom.replace("\n", ";"),
        "geometry_sha256": hashlib.sha256(geom.encode("ascii")).hexdigest(),
        "charge": "0",
        "multiplicity": "1",
        "cart": "False",
        "symmetry": "False",
        "gauge": "GIAO (pyscf-properties gauge_orig=None)",
        "grid_levels": json.dumps(GRID_LEVELS, separators=(",", ":")),
        "grid_convergence_limit_ppm": f"{GRID_CONVERGENCE_PPM:.17g}",
        "scf_conv_tol": "1e-12",
        "nmr_cphf_conv_tol": "1e-10",
        "nmr_cphf_max_cycle": "100",
        "value_units": "raw atomic units (no alpha^2*1e6 scaling)",
        "raw_au_to_ppm": f"{deps['nist'].ALPHA**2 * 1e6:.17g}",
        "pyscf_source_tensor_axes": "[atom,external_B_axis,nuclear_magnetic_dipole_axis]",
        "stored_tensor_axes": "[atom,nuclear_magnetic_dipole_axis,external_B_axis]",
        "tensor_transform": "transpose the final two PySCF axes",
        "meta_gga": "excluded: pyscf-properties GIAO XC raises NotImplementedError",
    }


def write_output(
    deps: dict[str, Any],
    path: Path,
    metadata: dict[str, str],
    results: dict[tuple[str, int | None], dict[str, Any]],
) -> None:
    np = deps["np"]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "record",
        "functional",
        "pyscf_xc",
        "grid_level",
        "grid_points",
        "grid_prune",
        "scf_energy_hartree",
        "atom_index",
        "atomic_number",
        "part",
        "row_m_axis",
        "column_b_axis",
        "value_au",
        "reference_grid_level",
        "reference_value_au",
        "delta_au",
        "abs_delta_au",
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# PROVENANCE: independent PySCF+pyscf-properties GIAO NMR oracle\n")
        for key, value in metadata.items():
            fh.write(f"# {key}: {value}\n")
        for atom_index, (symbol, z, x, y, zcoord) in enumerate(GEOMETRY):
            fh.write(
                f"# atom_{atom_index}: {symbol},{z},{x:.17g},{y:.17g},{zcoord:.17g} Bohr\n"
            )
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(columns)

        for name, xc in CASES:
            levels = (None,) if xc is None else GRID_LEVELS
            for level in levels:
                result = results[(name, level)]
                for part in ("total", "dia", "para"):
                    tensor = result["parts"][part]
                    for atom_index, (_symbol, z, _x, _y, _zcoord) in enumerate(GEOMETRY):
                        for row in range(3):
                            for col in range(3):
                                writer.writerow(
                                    (
                                        "tensor",
                                        name,
                                        result["pyscf_xc"],
                                        "none" if level is None else level,
                                        result["grid_points"],
                                        result["grid_prune"],
                                        f"{result['energy']:.16e}",
                                        atom_index,
                                        z,
                                        part,
                                        AXES[row],
                                        AXES[col],
                                        f"{tensor[atom_index, row, col]:.16e}",
                                        "",
                                        "",
                                        "",
                                        "",
                                    )
                                )

            if xc is None:
                continue
            low = results[(name, GRID_LEVELS[0])]
            high = results[(name, GRID_LEVELS[1])]
            for part in ("total", "dia", "para"):
                low_tensor = low["parts"][part]
                high_tensor = high["parts"][part]
                delta = high_tensor - low_tensor
                for atom_index, (_symbol, z, _x, _y, _zcoord) in enumerate(GEOMETRY):
                    for row in range(3):
                        for col in range(3):
                            writer.writerow(
                                (
                                    "grid_delta_9_minus_8",
                                    name,
                                    high["pyscf_xc"],
                                    GRID_LEVELS[1],
                                    high["grid_points"],
                                    high["grid_prune"],
                                    f"{high['energy']:.16e}",
                                    atom_index,
                                    z,
                                    part,
                                    AXES[row],
                                    AXES[col],
                                    f"{high_tensor[atom_index, row, col]:.16e}",
                                    GRID_LEVELS[0],
                                    f"{low_tensor[atom_index, row, col]:.16e}",
                                    f"{delta[atom_index, row, col]:.16e}",
                                    f"{abs(delta[atom_index, row, col]):.16e}",
                                )
                            )
                writer.writerow(
                    (
                        "grid_delta_max_abs",
                        name,
                        high["pyscf_xc"],
                        GRID_LEVELS[1],
                        high["grid_points"],
                        high["grid_prune"],
                        f"{high['energy']:.16e}",
                        "",
                        "",
                        part,
                        "",
                        "",
                        "",
                        GRID_LEVELS[0],
                        "",
                        "",
                        f"{np.max(np.abs(delta)):.16e}",
                    )
                )


def main() -> None:
    deps = load_dependencies()
    basis_path = deps["data_dir"] / "sto-3g.g94"
    if not basis_path.is_file():
        raise SystemExit(f"basis error: bundled basis file is missing: {basis_path}")
    basis = deps["parse_g94"](str(basis_path), {"H", "O"})
    if set(basis) != {"H", "O"}:
        raise RuntimeError(f"parsed STO-3G basis has unexpected elements: {sorted(basis)}")

    results: dict[tuple[str, int | None], dict[str, Any]] = {}
    for name, xc in CASES:
        levels = (None,) if xc is None else GRID_LEVELS
        for level in levels:
            results[(name, level)] = run_case(deps, basis, name, xc, level)

    for name, xc in CASES:
        if xc is None:
            continue
        scale = deps["nist"].ALPHA**2 * 1e6
        for part in ("total", "dia", "para"):
            low = results[(name, GRID_LEVELS[0])]["parts"][part]
            high = results[(name, GRID_LEVELS[1])]["parts"][part]
            worst_au = float(deps["np"].max(deps["np"].abs(high - low)))
            worst_ppm = worst_au * scale
            print(
                f"{name:10s} max |grid9-grid8| {part:5s} = "
                f"{worst_au:.3e} au = {worst_ppm:.3e} ppm"
            )
            if not deps["np"].isfinite(worst_ppm) or worst_ppm >= GRID_CONVERGENCE_PPM:
                raise RuntimeError(
                    f"{name} {part} grid-8/grid-9 NMR gap {worst_ppm:.6e} ppm "
                    f"does not satisfy {GRID_CONVERGENCE_PPM:.6e} ppm"
                )

    # Only replace the frozen oracle after every convergence condition has
    # passed. A failed regeneration must not leave a plausible-looking CSV.
    output = deps["out_dir"] / OUTPUT_NAME
    write_output(deps, output, provenance(deps, basis_path), results)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
