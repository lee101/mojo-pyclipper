"""ctypes loader for the Mojo polygon kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "capi.mojo")
LIB = os.environ.get("MOJO_PYCLIPPER_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyclipper.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpc_area": ([I, I], F),
    "mpc_area_f64": ([I, I], F),
    "mpc_point_in_polygon": ([I, I, I, I], I),
    "mpc_points_in_polygon": ([I, I, I, I, I], None),
    "mpc_is_convex": ([I, I], I),
    "mpc_clip_halfplane": ([I, I, F, F, F, F, F, I, I, I], I),
    "mpc_intersection": ([I, I, I, I, F, I, I, I], I),
    "mpc_offset_polygon": ([I, I, I, F, I, F, F, I, I], I),
}


class BuildError(RuntimeError):
    pass


def _build() -> str:
    if os.path.exists(LIB) and (
        not os.path.exists(SRC) or os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    if os.environ.get("MOJO_PYCLIPPER_LIB"):
        raise BuildError(f"MOJO_PYCLIPPER_LIB does not exist: {LIB}")
    pixi = shutil.which("pixi")
    if pixi:
        command = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    else:
        command = ["bash", os.path.join(ROOT, "build", "build.sh")]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(_build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_LIBRARY, name)
            function.argtypes = argtypes
            function.restype = restype
    return _LIBRARY


def addr(
    array: np.ndarray,
    dtype: np.dtype,
    *,
    writable: bool = False,
) -> int:
    """Return an ABI-safe address after validating the complete buffer contract."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.dtype != expected:
        raise TypeError(f"FFI buffer must have dtype {expected}")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffer must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffer must be writable")
    if array.size == 0:
        raise ValueError("empty buffers must not cross the FFI boundary")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffer has a null address")
    return address
