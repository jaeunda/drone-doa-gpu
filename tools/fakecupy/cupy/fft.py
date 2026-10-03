"""cupy.fft stand-in: keeps single precision like cuFFT does."""
import numpy as _np


def _c(x, out):
    return out.astype(_np.complex64) if _np.asarray(x).dtype in (_np.float32, _np.complex64) else out


def rfft(a, n=None, axis=-1, norm=None):
    return _c(a, _np.fft.rfft(a, n=n, axis=axis, norm=norm))


def irfft(a, n=None, axis=-1, norm=None):
    out = _np.fft.irfft(a, n=n, axis=axis, norm=norm)
    return out.astype(_np.float32) if _np.asarray(a).dtype == _np.complex64 else out


def fft(a, n=None, axis=-1, norm=None):
    return _c(a, _np.fft.fft(a, n=n, axis=axis, norm=norm))


def ifft(a, n=None, axis=-1, norm=None):
    return _c(a, _np.fft.ifft(a, n=n, axis=axis, norm=norm))


class _PlanCache:
    def set_size(self, n):
        pass

    def clear(self):
        pass

    def get_size(self):
        return 0


class config:
    @staticmethod
    def get_plan_cache():
        return _PlanCache()


def __getattr__(name):
    return getattr(_np.fft, name)
