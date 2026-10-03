"""RawKernel emulation: compile CUDA C with g++ and run every (block, thread)."""
import ctypes
import hashlib
import os
import re
import subprocess
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_INCLUDE = os.path.join(os.path.dirname(_HERE), "include")
_CACHE = os.path.join(tempfile.gettempdir(), "fakecupy_cache")

_SCALAR = {
    "int": ctypes.c_int, "unsigned int": ctypes.c_uint, "unsigned": ctypes.c_uint,
    "long long": ctypes.c_longlong, "long long int": ctypes.c_longlong,
    "size_t": ctypes.c_size_t, "float": ctypes.c_float, "double": ctypes.c_double,
    "bool": ctypes.c_bool,
}
_POINTER_DTYPE = {
    "float": np.float32, "double": np.float64, "int": np.int32, "unsigned int": np.uint32,
    "long long": np.int64, "unsigned char": np.uint8, "char": np.int8, "float2": np.complex64,
    "unsigned long long": np.uint64,
}


def _parse_params(code, name):
    m = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\(([^)]*)\)", code)
    if not m:
        raise ValueError(f"kernel {name!r} not found in source")
    params = []
    for raw in m.group(1).split(","):
        decl = " ".join(raw.replace("__restrict__", " ").split())
        ident = re.findall(r"[A-Za-z_]\w*", decl)[-1]
        base = decl[: decl.rfind(ident)].replace("const", " ").replace("*", " ")
        base = " ".join(base.split())
        params.append({"decl": decl, "name": ident, "pointer": "*" in decl, "base": base})
    return params


class RawKernel:
    def __init__(self, code, name, options=(), backend="nvrtc", translate_cucomplex=False, **kw):
        if "__shared__" in code or "__syncthreads" in code:  # noqa: E501
            raise NotImplementedError("fakecupy cannot emulate shared memory / __syncthreads")
        self.code, self.name = code, name
        self.params = _parse_params(code, name)
        self._fn = None

    def _compile(self):
        os.makedirs(_CACHE, exist_ok=True)
        decls = ", ".join(p["decl"] for p in self.params)
        names = ", ".join(p["name"] for p in self.params)
        launcher = f"""
extern "C" void fakecupy_launch(unsigned gx, unsigned gy, unsigned gz,
                            unsigned bx, unsigned by, unsigned bz, {decls}) {{
    long long nblocks = (long long)gx * gy * gz;
    #pragma omp parallel for schedule(dynamic, 16)
    for (long long b = 0; b < nblocks; ++b) {{
        gridDim.x = gx; gridDim.y = gy; gridDim.z = gz;
        blockDim.x = bx; blockDim.y = by; blockDim.z = bz;
        blockIdx.x = (unsigned)(b % gx); blockIdx.y = (unsigned)((b / gx) % gy); blockIdx.z = (unsigned)(b / ((long long)gx * gy));
        for (unsigned tz = 0; tz < bz; ++tz)
            for (unsigned ty = 0; ty < by; ++ty)
                for (unsigned tx = 0; tx < bx; ++tx) {{
                    threadIdx.x = tx; threadIdx.y = ty; threadIdx.z = tz;
                    {self.name}({names});
                }}
    }}
}}
"""
        src = '#include "fakecuda.h"\n' + self.code + "\n" + launcher
        key = hashlib.sha1(src.encode()).hexdigest()[:16]
        so = os.path.join(_CACHE, f"{self.name}_{key}.so")
        if not os.path.exists(so):
            cpp = so[:-3] + ".cpp"
            with open(cpp, "w") as f:
                f.write(src)
            cmd = ["g++", "-O2", "-std=c++14", "-fopenmp", "-shared", "-fPIC", "-Wno-unknown-pragmas",
                   "-I", _INCLUDE, cpp, "-o", so]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"fakecupy compile failed for {self.name}:\n{res.stderr[-4000:]}")
        lib = ctypes.CDLL(so)
        fn = lib.fakecupy_launch
        fn.restype = None
        argtypes = [ctypes.c_uint] * 6
        for p in self.params:
            if p["pointer"]:
                argtypes.append(ctypes.c_void_p)
            elif p["base"] in _SCALAR:
                argtypes.append(_SCALAR[p["base"]])
            else:
                raise TypeError(f"unsupported scalar parameter type {p['base']!r} in {self.name}")
        fn.argtypes = argtypes
        self._fn = fn

    def __call__(self, grid, block, args, shared_mem=0, stream=None):
        if self._fn is None:
            self._compile()
        grid = tuple(grid) if isinstance(grid, (tuple, list)) else (grid,)
        block = tuple(block) if isinstance(block, (tuple, list)) else (block,)
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        block = tuple(int(b) for b in block) + (1,) * (3 - len(block))
        if block[0] * block[1] * block[2] > 1024:
            raise ValueError("block has more than 1024 threads (invalid on real CUDA)")
        if len(args) != len(self.params):
            raise TypeError(f"{self.name}: expected {len(self.params)} args, got {len(args)}")
        cargs = []
        keep = []
        for p, a in zip(self.params, args):
            if p["pointer"]:
                if not isinstance(a, np.ndarray):
                    raise TypeError(f"{self.name}.{p['name']}: expected array, got {type(a)}")
                if not a.flags["C_CONTIGUOUS"]:
                    raise ValueError(f"{self.name}.{p['name']}: array must be C-contiguous")
                want = _POINTER_DTYPE.get(p["base"])
                if want is not None and a.dtype != want:
                    raise TypeError(f"{self.name}.{p['name']}: dtype {a.dtype} does not match {p['base']}*")
                keep.append(a)
                cargs.append(ctypes.c_void_p(a.ctypes.data))
            else:
                if isinstance(a, np.ndarray) and a.ndim > 0:
                    raise TypeError(f"{self.name}.{p['name']}: expected scalar")
                cargs.append(a.item() if isinstance(a, np.generic) else a)
        self._fn(*grid, *block, *cargs)
