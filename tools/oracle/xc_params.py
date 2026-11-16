# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ss0832
# This file is part of the dft-rs library.
#
# Print the default external parameters, mixing coefficients of the
# components and range-separation data of libxc functionals, through the
# libxc C API of the library shipped with PySCF (numerical reference only).
#
# Run (WSL): <venv>/bin/python xc_params.py GGA_X_PW86 HYB_GGA_XC_X3LYP ...

import ctypes
import glob
import os
import sys

import pyscf
from pyscf.dft import libxc


def load():
    base = os.path.dirname(pyscf.__file__)
    path = glob.glob(os.path.join(base, "lib", "deps", "lib", "libxc.so*"))[0]
    lib = ctypes.CDLL(path)
    lib.xc_func_alloc.restype = ctypes.c_void_p
    lib.xc_func_init.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.xc_func_get_info.argtypes = [ctypes.c_void_p]
    lib.xc_func_get_info.restype = ctypes.c_void_p
    lib.xc_func_info_get_n_ext_params.argtypes = [ctypes.c_void_p]
    lib.xc_func_info_get_ext_params_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xc_func_info_get_ext_params_name.restype = ctypes.c_char_p
    lib.xc_func_info_get_ext_params_description.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xc_func_info_get_ext_params_description.restype = ctypes.c_char_p
    lib.xc_func_info_get_ext_params_default_value.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.xc_func_info_get_ext_params_default_value.restype = ctypes.c_double
    lib.xc_func_info_get_name.argtypes = [ctypes.c_void_p]
    lib.xc_func_info_get_name.restype = ctypes.c_char_p
    lib.xc_func_info_get_family.argtypes = [ctypes.c_void_p]
    lib.xc_func_info_get_flags.argtypes = [ctypes.c_void_p]
    lib.xc_num_aux_funcs.argtypes = [ctypes.c_void_p]
    lib.xc_aux_func_ids.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.xc_aux_func_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    return lib


def params_of(lib, name):
    """Default external parameters of a libxc functional, by name."""
    func = lib.xc_func_alloc()
    lib.xc_func_init(func, libxc.XC_CODES[name.upper()], 1)
    info = lib.xc_func_get_info(func)
    return {
        lib.xc_func_info_get_ext_params_name(info, i).decode():
            lib.xc_func_info_get_ext_params_default_value(info, i)
        for i in range(lib.xc_func_info_get_n_ext_params(info))
    }


def main():
    lib = load()
    for name in sys.argv[1:]:
        code = libxc.XC_CODES[name.upper()]
        func = lib.xc_func_alloc()
        lib.xc_func_init(func, code, 1)
        info = lib.xc_func_get_info(func)
        print(f"== {name} ({code}): {lib.xc_func_info_get_name(info).decode()}")
        print("   family", lib.xc_func_info_get_family(info))
        print("   flags", lib.xc_func_info_get_flags(info))
        for i in range(lib.xc_func_info_get_n_ext_params(info)):
            print(
                "  ",
                lib.xc_func_info_get_ext_params_name(info, i).decode(),
                repr(lib.xc_func_info_get_ext_params_default_value(info, i)),
                "-",
                lib.xc_func_info_get_ext_params_description(info, i).decode(),
            )
        n = lib.xc_num_aux_funcs(func)
        if n:
            ids = (ctypes.c_int * n)()
            weights = (ctypes.c_double * n)()
            lib.xc_aux_func_ids(func, ids)
            lib.xc_aux_func_weights(func, weights)
            names = {v: k for k, v in libxc.XC_CODES.items()}
            print("   aux", [(names.get(i, i), w) for i, w in zip(ids, weights)])
        print("   components", libxc.parse_xc(name))
        if libxc.is_hybrid_xc(name):
            print("   hybrid", libxc.hybrid_coeff(name), "rsh", libxc.rsh_coeff(name))
        print("   reference", libxc.xc_reference(name))


if __name__ == "__main__":
    main()
