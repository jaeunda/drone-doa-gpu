"""cupy.cuda stand-in: events are host timestamps, streams are no-ops."""
import time as _time

from . import runtime  # noqa: F401


class Device:
    def __init__(self, device=0):
        self.id = device

    def synchronize(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Stream:
    null = None

    def __init__(self, non_blocking=False, null=False, ptds=False):
        pass

    def synchronize(self):
        pass

    def use(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


Stream.null = Stream()


def get_current_stream():
    return Stream.null


class Event:
    def __init__(self, block=False, disable_timing=False, interprocess=False):
        self._t = None

    def record(self, stream=None):
        self._t = _time.perf_counter()

    def synchronize(self):
        pass

    @property
    def done(self):
        return True


def get_elapsed_time(start_event, end_event):
    return (end_event._t - start_event._t) * 1e3


def alloc_pinned_memory(nbytes):
    return bytearray(nbytes)
