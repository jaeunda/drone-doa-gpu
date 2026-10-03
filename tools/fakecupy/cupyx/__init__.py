"""cupyx stand-in for fakecupy dry runs: pinned buffers are ordinary NumPy arrays."""
import numpy as _np


def empty_pinned(shape, dtype=float, order="C"):
    return _np.empty(shape, dtype=dtype, order=order)


def zeros_pinned(shape, dtype=float, order="C"):
    return _np.zeros(shape, dtype=dtype, order=order)
