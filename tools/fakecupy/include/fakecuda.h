// Host-side emulation header for CUDA C kernels (fakecupy dry runs only).
#pragma once
#include <math.h>
#include <stdint.h>
#include <float.h>

struct dim3 { unsigned x = 1, y = 1, z = 1; };
static thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;
static const int warpSize = 32;

#define __global__
#define __device__ static inline
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __restrict__ __restrict
#define __launch_bounds__(...)
#define __shared__ FAKECUPY_SHARED_MEMORY_IS_NOT_SUPPORTED
#define __syncthreads() FAKECUPY_SYNCTHREADS_IS_NOT_SUPPORTED
#define __ldg(p) (*(p))

struct float2 { float x, y; };
static inline float2 make_float2(float a, float b) { float2 r; r.x = a; r.y = b; return r; }
static inline float __fdividef(float a, float b) { return a / b; }
static inline float __int_as_float(int v) { union { int i; float f; } u; u.i = v; return u.f; }
static inline int min(int a, int b) { return a < b ? a : b; }
static inline int max(int a, int b) { return a > b ? a : b; }
static inline long long min(long long a, long long b) { return a < b ? a : b; }
static inline long long max(long long a, long long b) { return a > b ? a : b; }
#include <chrono>
static inline long long clock64() {   // emulation: nanoseconds stand in for SM cycles
    return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
