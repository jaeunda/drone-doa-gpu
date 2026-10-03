import os as _os


def getDeviceCount():
    return 1


def getDeviceProperties(device_id=0):
    return {
        "name": b"FAKE-CPU-EMULATION (not a GPU)",
        "major": 7,
        "minor": 5,
        "totalGlobalMem": 16 * 2**30,
        "multiProcessorCount": 40,
        "maxThreadsPerBlock": 1024,
        "maxThreadsPerMultiProcessor": 1024,
        "warpSize": 32,
        "regsPerBlock": 65536,
        "sharedMemPerBlock": 49152,
        "clockRate": 1590000,
        "memoryClockRate": 5001000,
        "memoryBusWidth": 256,
        "l2CacheSize": 4 * 2**20,
    }


def runtimeGetVersion():
    return 12090


def driverGetVersion():
    return 13000


def deviceSynchronize():
    pass


def memGetInfo():
    return (15 * 2**30, 16 * 2**30)


def getDevice():
    return 0


_ = _os
