# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Oracle generator for the registry entries that gen_dft.py never froze.
# Same CSV layout, same molecule set, same grid level and SCF thresholds —
# the output drops straight into tests/oracle_data/ and is picked up by the
# existing p3 loader, which then removes the entry from REFERENCE_FREE in
# tests/p14_functional_coverage.rs.
#
# Two of these cases (wb97x_d, wb97x_v) failed on the PySCF build that
# produced the frozen set because their SCF path was unsupported; two
# others (lc-blyp, lc-pbe) need the range-separation tokens in the xc
# string. Nothing here is expected to work everywhere, so each case is run
# independently and a failure is reported rather than fatal.
#
# The ω a functional is frozen at MUST be the ω the registry instantiates,
# or the CSV silently pins the wrong functional. Every range-separated case
# therefore declares its expected ω and the script refuses to write a CSV
# whose resolved ω disagrees.
#
# Run (WSL): ~/.venvs/dft-rs-oracle/bin/python gen_dft_missing.py [name ...]

import os
import traceback

import pyscf
from pyscf import gto, dft

from gen_integrals import parse_g94, DATA_DIR, OUT_DIR
from gen_scf import MOLECULES, Z

# (case tag, molecule, basis, dft-rs spec, pyscf xc string, expected omega,
#  post-mf hook)
#
# The dft-rs spec is either a combined name or an ("x", "c") pair; the pair
# is written as `x,<name>` / `c,<name>` keys, which the p3 loader resolves
# as a split specification.
CASES = [
    # HSE03: the 2003 parameters, ω_HF = 0.15/√2 and ω_PBE = 0.15·2^(1/3).
    # libxc carries both inside HYB_GGA_XC_HSE03, so do NOT override
    # mf.omega here — a single ω would collapse them onto each other and
    # turn the case into HSE06 with a shifted screening length.
    ("h2o", "def2-svp", "hse03", "hse03", 0.15 / 2 ** 0.5, None),
    ("ch3_doublet", "sto-3g", "hse03", "hse03", 0.15 / 2 ** 0.5, None),
    # LC-BLYP: ITYH-attenuated B88 + LYP with full long-range HF. libxc's
    # HYB_GGA_XC_LC_BLYP is the same construction but ships its own ω, so
    # the components are spelled out and ω is pinned to the registry value.
    (
        "h2o",
        "def2-svp",
        "lc-blyp",
        "1.00*LR_HF(0.33) + GGA_X_ITYH, GGA_C_LYP",
        0.33,
        lambda mf: setattr(mf, "omega", 0.33),
    ),
    (
        "ch3_doublet",
        "sto-3g",
        "lc-blyp",
        "1.00*LR_HF(0.33) + GGA_X_ITYH, GGA_C_LYP",
        0.33,
        lambda mf: setattr(mf, "omega", 0.33),
    ),
    # LC-PBE: the ITYH attenuation of PBE exchange (NOT the HJS/wPBEh hole
    # of LC-ωPBE — those are separate entries with their own oracles).
    # Needs a libxc with GGA_X_ITYH_PBE.
    (
        "h2o",
        "def2-svp",
        "lc-pbe",
        "1.00*LR_HF(0.47) + GGA_X_ITYH_PBE, GGA_C_PBE",
        0.47,
        lambda mf: setattr(mf, "omega", 0.47),
    ),
    # ωB97X-D semilocal part. The CHG dispersion tail is dft-rs `--disp chg`
    # and is gated separately, so the frozen number must exclude it.
    #
    # The spelling is load-bearing: PySCF's `scf.dispersion.parse_dft` sees
    # "wb97x_d" (and "wb97x-d") as a functional carrying a dispersion tail it
    # cannot construct, and raises NotImplementedError before the SCF starts.
    # "wb97xd" resolves to the SAME libxc entry — HYB_GGA_XC_WB97X_D, id 471,
    # ω = 0.2, cx = 0.222036 — but is not matched by that lookup, so the
    # semilocal functional runs with no tail added (`mf.disp is None`). That
    # is precisely the number wanted here; the ω check below confirms the
    # spelling did not quietly land on a different functional.
    ("h2o", "def2-svp", "wb97x-d", "wb97xd", 0.2, None),
    ("ch3_doublet", "sto-3g", "wb97x-d", "wb97xd", 0.2, None),
    # ωB97X-V semilocal part only (nlc off); the VV10 total is gated in
    # p9_functionals against its own oracle, exactly as for ωB97M-V.
    ("h2o", "def2-svp", "wb97x-v", "wb97x_v", 0.3, lambda mf: setattr(mf, "nlc", 0)),
    # Split components that no combined oracle reaches at their registry
    # parameters: OPTX with its own a1 (O3LYP forces a1 = 0), and the bare
    # HJS ωPBE hole at the registry default ω = 0.11 (LC-ωPBE/HJS pins it
    # only at ω = 0.4).
    ("h2o", "def2-svp", ("optx", "lyp"), "gga_x_optx,gga_c_lyp", None, None),
    # Catalogue LDA correlations, each on Slater exchange.
    ("h2o", "sto-3g", ("slater", "wigner"), "lda_x,lda_c_wigner", None, None),
    ("ch3_doublet", "sto-3g", ("slater", "wigner"), "lda_x,lda_c_wigner", None, None),
    ("h2o", "sto-3g", ("slater", "hl"), "lda_x,lda_c_hl", None, None),
    ("ch3_doublet", "sto-3g", ("slater", "hl"), "lda_x,lda_c_hl", None, None),
    ("h2o", "sto-3g", ("slater", "gl"), "lda_x,lda_c_gl", None, None),
    ("ch3_doublet", "sto-3g", ("slater", "gl"), "lda_x,lda_c_gl", None, None),
    # Catalogue GGA exchanges without a combined entry, on PBE correlation.
    ("h2o", "sto-3g", ("pw86", "pbe"), "gga_x_pw86,gga_c_pbe", None, None),
    ("h2o", "sto-3g", ("lg93", "pbe"), "gga_x_lg93,gga_c_pbe", None, None),
    ("h2o", "sto-3g", ("ft97", "pbe"), "gga_x_ft97_b,gga_c_pbe", None, None),
    ("h2o", "sto-3g", ("pbe", "vpbe"), "gga_x_pbe,gga_c_pbe_vwn", None, None),
    (
        "h2o",
        "def2-svp",
        ("hjs-pbe-x", "none"),
        "gga_x_hjs_pbe,",
        0.11,
        lambda mf: setattr(mf, "omega", 0.11),
    ),
]


def resolved_omega(xc):
    """PySCF's range-separation parameter for an xc string, or None."""
    try:
        omega, _alpha, _beta = dft.libxc.rsh_coeff(xc)
    except Exception:
        return None
    return omega


def run_case(mol_name, basis_name, spec, pyscf_xc, expect_omega, hook):
    charge, mult, atoms = MOLECULES[mol_name]
    elements = sorted({sym for sym, *_ in atoms})
    basis = parse_g94(os.path.join(DATA_DIR, f"{basis_name}.g94"), set(elements))
    geom = "; ".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    mol = gto.M(
        atom=geom,
        basis=basis,
        unit="Bohr",
        charge=charge,
        spin=mult - 1,
        cart=False,
    )
    mf = dft.RKS(mol) if mult == 1 else dft.UKS(mol)
    mf.xc = pyscf_xc
    if hook is not None:
        hook(mf)
    mf.grids.level = 9
    mf.conv_tol = 1e-12
    mf.max_cycle = 200

    # Freeze nothing whose screening length is not the registry's. mf.omega
    # (when set) wins over the string, so check it first.
    if expect_omega is not None:
        omega = getattr(mf, "omega", None) or resolved_omega(pyscf_xc)
        if omega is None:
            raise RuntimeError(
                f"could not resolve ω for `{pyscf_xc}`; refusing to freeze a "
                f"range-separated functional with an unknown screening length"
            )
        if abs(omega - expect_omega) > 1e-12:
            raise RuntimeError(
                f"ω mismatch: PySCF resolves {omega} but the dft-rs registry "
                f"instantiates {expect_omega}. Freezing this would pin a "
                f"different functional."
            )

    mf.kernel()
    assert mf.converged, f"{mol_name}/{basis_name}/{spec} did not converge"

    if isinstance(spec, tuple):
        x_name, c_name = spec
        tag = x_name if c_name == "none" else f"{x_name}-{c_name}"
        keys = [f"x,{x_name}", f"c,{c_name}"]
    else:
        tag = spec
        keys = [f"functional,{spec}"]
    case = f"{mol_name}_{basis_name.replace('-', '')}_{tag}"
    path = os.path.join(OUT_DIR, f"{case}_dft.csv")
    with open(path, "w", newline="\n") as fh:
        fh.write(
            f"# PROVENANCE: PySCF {pyscf.__version__} DFT oracle "
            f"(libxc-backed, values only)\n"
        )
        fh.write("# geometry unit: Bohr (exact literals); pyscf grid level 9\n")
        fh.write(f"# pyscf xc string: {pyscf_xc}\n")
        fh.write(f"basis,{basis_name}\n")
        for key in keys:
            fh.write(f"{key}\n")
        fh.write(f"charge,{charge}\n")
        fh.write(f"mult,{mult}\n")
        for sym, x, y, z in atoms:
            fh.write(f"atom,{Z[sym]},{x},{y},{z}\n")
        fh.write(f"e_tot,{mf.e_tot:.14e}\n")
    return case, mf.e_tot


def main():
    import sys

    only = set(sys.argv[1:])
    os.makedirs(OUT_DIR, exist_ok=True)
    done, failed = [], []
    for mol_name, basis_name, spec, pyscf_xc, expect_omega, hook in CASES:
        label = spec if isinstance(spec, str) else "/".join(spec)
        if only and label not in only:
            continue
        try:
            case, e_tot = run_case(
                mol_name, basis_name, spec, pyscf_xc, expect_omega, hook
            )
            done.append(case)
            print(f"{case}: E = {e_tot:.10f}")
        except Exception as exc:  # one unsupported path must not stop the rest
            failed.append((f"{mol_name}/{basis_name}/{label}", exc))
            print(f"SKIPPED {mol_name}/{basis_name}/{label}: {exc}")
            traceback.print_exc()

    print(f"\n{len(done)} frozen, {len(failed)} skipped")
    for label, exc in failed:
        print(f"  {label}: {type(exc).__name__}: {exc}")
    if done:
        print(
            "\nReview the new CSVs like code, then drop the corresponding "
            "names from REFERENCE_FREE / COMPONENT_REFERENCE_FREE in "
            "tests/p14_functional_coverage.rs — the census test fails while "
            "a name is listed there AND has an oracle."
        )


if __name__ == "__main__":
    main()
