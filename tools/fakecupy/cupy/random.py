import numpy as _np


def seed(s=None):
    _np.random.seed(s)


def __getattr__(name):
    return getattr(_np.random, name)
