"""Minimal CPU stand-in for CuPy, used only to dry-run the notebooks
on a machine without an NVIDIA GPU.

- Arrays are plain NumPy arrays; unknown attributes fall back to NumPy.
- RawKernel compiles the CUDA C source with g++ (OpenMP over blocks) and
  launches it serially per thread, so kernel *logic* is exercised for real.
- Timings produced under this shim are meaningless and must never be reported.

Limitations: no __shared__ memory or __syncthreads (asserted at compile time).
"""
import numpy as _np

from . import cuda  # noqa: F401
from . import fft  # noqa: F401
from . import random  # noqa: F401
from ._kernel import RawKernel  # noqa: F401

__version__ = "0.0-fakecupy"
IS_FAKE = True
ndarray = _np.ndarray


def asarray(a, dtype=None, order="C"):
    return _np.array(a, dtype=dtype, order=order or "C", copy=True)


def asnumpy(a, stream=None, out=None):
    if out is not None:
        out[...] = a
        return out
    return _np.array(a, copy=True)


def get_array_module(*args):
    return _np


class _MemoryPool:
    def free_all_blocks(self):
        pass

    def used_bytes(self):
        return 0

    def total_bytes(self):
        return 0


_pool = _MemoryPool()


def get_default_memory_pool():
    return _pool


def get_default_pinned_memory_pool():
    return _pool


def __getattr__(name):
    return getattr(_np, name)
