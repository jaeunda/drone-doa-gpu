# %% [markdown]
# # Phase 1 — GPU Parallelization of Acoustic Drone Direction Finding
#
# **In one sentence.** Given a drone sound captured by an 8-microphone array, find its direction by scoring every candidate azimuth. This notebook measures what happens when that computation moves to the GPU: from what problem size the GPU wins, why moving only the search gives a small overall gain (Amdahl), where the bottleneck goes once the preprocessing also runs on the GPU, and therefore how many microphone arrays one GPU can serve in real time.
#
# **Motivating scenario.** Drone intrusion monitoring around airports and power plants. Each monitoring site has a microphone array, and one server processes many arrays concurrently. If a frame (42.7 ms of audio) takes longer than 42.7 ms to process, the system falls behind real time.
#
# **Scope.** Assumes a drone sound is present and estimates its direction only. Drone detection (Log-Mel classification) is a later phase. Design notes: `design.md` in this folder; project roadmap: `docs/roadmap.md`.
#
# ## Design summary
#
# | Item | Design |
# |---|---|
# | Input | 8-channel float32 audio `[F, 8, 2048]` (48 kHz, 42.7 ms frames). The source is a real drone propeller recording with synthetic array delays applied |
# | Output | Direction scores `[F, A]`. The real-time path transfers only the per-frame argmax index `[F]` back to the host |
# | Parallel unit | Direction scan: 1 thread = (frame, angle); the 28 microphone pairs are a loop inside the thread. PHAT: 1 thread = (frame, pair, freq bin) |
# | Transfers & dependencies | Frames are independent. Stages are sequential (FFT → PHAT → IFFT → lag extraction → scan → argmax). The delay table is uploaded once per configuration and reused |
# | Expected bottleneck | Scan-only GPU: CPU GCC-PHAT. Full GPU: audio H2D (pageable), or host dispatch for small batches |
# | Validation | Score allclose, argmax agreement, circular angle error vs. ground truth, negative controls must be caught as FAIL |
#
# ## Hypotheses (fixed before measurement; rationale in `design.md`)
#
# | ID | Hypothesis | Criterion |
# |---|---|---|
# | H1 | Small workloads favour the CPU because of GPU fixed costs. The win/lose boundary in the (F, A) plane is predicted by a two-term fixed-cost model (fixed latency + bytes/bandwidth + work/throughput) | Measured crossovers (A* at F=1, F* at A=1440) within 2× of the prediction |
# | H2 | Moving only the scan to the GPU caps the overall gain at the Amdahl bound 1/(1−p), and p grows with A | Hybrid speedup ≤ bound at every A, and the bound increases with A |
# | H3 | With everything on the GPU, the largest segment is audio H2D or FFT | Largest segment of the breakdown (host wait included) |
# | H4 | The `[A][P]` delay table is uncoalesced per warp, but each thread reuses the same cache line for the next p, so it differs from `[P][A]` by less than 1.2× | Kernel time ratio < 1.2 |
# | H5 | Under a p95 deadline (42.7 ms), full-GPU real-time capacity is at least 10× that of the best CPU pipeline | Capacity ratio |
# | H6a | The real drone source is harder than white noise **even at the same in-band SNR** | The in-band SNR at which the failure rate reaches 20% is higher for the drone |
# | H6b | Band-limited PHAT fails less often than full-band PHAT on the drone source (after 16→48 kHz upsampling the 8–24 kHz bins carry no signal) | Failure rate averaged over SNR. Unweighted GCC and PHAT-β (0.7) are reported as well |
#
# The numeric predictions come from a model calibrated on an earlier baseline run (2026-10-02, T4). They are **out-of-sample predictions** with respect to the calibration points, not a-priori predictions.

# %% [markdown]
# ## Cell 1 — Execution environment
# Records the GPU, CPU model and core count, driver/CUDA/library versions, and GPU clocks and temperature. Every "N× faster" in the conclusions holds only for this environment.
# `CUPY_CACHE_DIR` points to a fresh temporary directory so that the first (cold) call really includes kernel compilation.

# %%
import os, sys, re, json, time, math, platform, subprocess, importlib, warnings, tempfile
from pathlib import Path

os.environ.setdefault("CUPY_CACHE_DIR", tempfile.mkdtemp(prefix="cupy_cold_cache_"))
QUICK = os.environ.get("CUDA_REPORT_QUICK") == "1"  # local dry-run mode only: smaller sweeps


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable ({exc.__class__.__name__})"


try:
    import cupy as cp
except ImportError:
    print("CuPy not found; installing cupy-cuda12x ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"])
    importlib.invalidate_caches()
    import cupy as cp
import cupyx
import numpy as np
import numba
import scipy

IS_FAKE_GPU = bool(getattr(cp, "IS_FAKE", False))
if cp.cuda.runtime.getDeviceCount() < 1:
    raise RuntimeError("No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU, then Run all.")

props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])


def cpu_model_name():
    m = re.search(r"Model name:\s*(.+)", command_output(["lscpu"]))
    return m.group(1).strip() if m else platform.processor()


def gpu_state():
    return command_output(["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
                           "--format=csv,noheader"])


ENV = {
    "gpu_name": GPU_NAME,
    "gpu_sm_count": int(props.get("multiProcessorCount", -1)),
    "compute_capability": f"{props['major']}.{props['minor']}",
    "gpu_memory_gib": round(props["totalGlobalMem"] / 2**30, 2),
    "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    "gpu_state_start": gpu_state(),
    "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
    "cuda_driver": cp.cuda.runtime.driverGetVersion(),
    "cpu_model": cpu_model_name(),
    "cpu_logical_cores": os.cpu_count(),
    "numba_threads": numba.get_num_threads(),
    "python": platform.python_version(),
    "cupy": cp.__version__, "numpy": np.__version__, "numba": numba.__version__, "scipy": scipy.__version__,
    "dtype": "float32 audio / complex64 spectra on both CPU (NumPy>=2 keeps single precision) and GPU",
    "host_memory_for_E2E": "pageable unless marked pinned",
    "quick_mode": QUICK, "fake_gpu": IS_FAKE_GPU,
}
RESULTS_DIR = Path("/content/uav_results") if Path("/content").exists() else Path.cwd() / "uav_results"
FIG_DIR = RESULTS_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
for k, v in ENV.items():
    print(f"{k:20s}: {v}")
if IS_FAKE_GPU:
    print("\n*** FAKE GPU (CPU emulation). Logic check only: every timing below is meaningless. ***")

# %% [markdown]
# ## Cell 2 — Configuration
# 8 microphones on an 8 cm-radius circle → 28 pairs. A 2048-sample frame = 42.7 ms is the real-time deadline. Lag range ±25 samples (0.16 m / 343 m/s × 48 kHz = 22.4, plus a margin of 2).
# Large A values (3600, 7200) are meaningless as azimuth resolution (Cell 11 shows accuracy saturating around A≈360–720). They serve as a workload knob standing in for an **azimuth 360 × elevation 10–20 grid** (the 2D search on the roadmap).

# %%
from itertools import combinations
import random

SEED = 20261002
random.seed(SEED); np.random.seed(SEED); cp.random.seed(SEED)

SAMPLE_RATE = 48_000
SOUND_SPEED = 343.0
MIC_COUNT = 8
ARRAY_RADIUS_M = 0.08
FRAME_SIZE = 2048
FRAME_MS = 1000.0 * FRAME_SIZE / SAMPLE_RATE          # real-time deadline per frame
NFFT = 1 << int(np.ceil(np.log2(2 * FRAME_SIZE - 1)))  # 4096 -> linear (not circular) correlation
PHAT_BAND_HZ = (300.0, 7_500.0)
PHAT_EPS = 1e-7
PHAT_BETA = 0.7
WEIGHTING = {0: "GCC (no weighting)", 1: "PHAT full band", 2: "PHAT 300-7500 Hz", 3: "PHAT-beta 0.7, 300-7500 Hz"}
MODES = (0, 1, 2, 3)
DEFAULT_MODE = 2
SNR_DB = 18.0
DEFAULT_A = 720
TRAJECTORY_FRAMES = 96
STATIC_AZIMUTHS_DEG = np.array([30.0, 73.0, 145.0, 250.0], dtype=np.float32)

A_SWEEP = [36, 90, 360, 1440, 3600, 7200]          # used at F = FIXED_F and at F = 1
F_SWEEP = [1, 4, 16, 64, 256, 1024]
FIXED_F, FIXED_A = 64, 1440
GRID_A, GRID_F = [36, 360, 3600], [1, 16, 256]
CALIBRATION = [(1, 36), (16, 1440)]          # model calibration points (in-sample)
BREAK_CASES = [(64, 360), (64, 1440), (64, 7200), (1024, 1440)]
CAPACITY_S = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ACC_SNRS = [-20, -15, -10, -5, 0, 10]
ACC_TRIALS = 108                              # random azimuths per (source, SNR)
ACC_A_SWEEP = [36, 90, 180, 360, 720, 1440, 3600, 7200]
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
CUDA_BLOCK = 256

WARMUP, MIN_REPS, TARGET_REPS, BUDGET_S, MAX_INNER = 3, 10, 30, 2.0, 200
SCORE_ATOL, SCORE_RTOL = 2e-5, 2e-4
GCC_REL_TOL = 1e-4                            # max|gpu-cpu| / max|cpu| for correlation tensors
STATIC_MAX_ERR_DEG = 3.0
TRAJ_MEDIAN_ERR_DEG = 5.0
FAIL_ERR_DEG = 5.0                            # accuracy trial counts as a failure above this

if QUICK:
    A_SWEEP, F_SWEEP = [36, 360, 1440], [1, 16, 64]
    GRID_A, GRID_F = [36, 360], [1, 16]
    BREAK_CASES = [(16, 360), (16, 1440)]
    CAPACITY_S = [1, 4, 16, 64]
    ACC_SNRS, ACC_TRIALS, ACC_A_SWEEP = [-10, 0], 12, [36, 360, 1440]
    BLOCK_SIZES = [64, 256]
    TRAJECTORY_FRAMES = 24
    WARMUP, MIN_REPS, TARGET_REPS, BUDGET_S, MAX_INNER = 1, 3, 3, 0.2, 2

mic_angles = np.linspace(0.0, 2.0 * np.pi, MIC_COUNT, endpoint=False)
MIC_POS = np.column_stack([ARRAY_RADIUS_M * np.cos(mic_angles), ARRAY_RADIUS_M * np.sin(mic_angles)]).astype(np.float32)
PAIRS = np.array(list(combinations(range(MIC_COUNT), 2)), dtype=np.int32)
PAIR_I, PAIR_J = PAIRS[:, 0].copy(), PAIRS[:, 1].copy()
P = len(PAIRS)
MAX_LAG = int(np.ceil(np.max(np.linalg.norm(MIC_POS[PAIR_I] - MIC_POS[PAIR_J], axis=1)) * SAMPLE_RATE / SOUND_SPEED)) + 2
L = 2 * MAX_LAG + 1
K_BINS = NFFT // 2 + 1
BIN_HZ = SAMPLE_RATE / NFFT
K_LO, K_HI = int(np.ceil(PHAT_BAND_HZ[0] / BIN_HZ)), int(np.floor(PHAT_BAND_HZ[1] / BIN_HZ))
assert P == 28
print(f"pairs={P}, frame={FRAME_SIZE} samples = {FRAME_MS:.2f} ms deadline, NFFT={NFFT}, lags=+/-{MAX_LAG} ({L} bins)")
print(f"PHAT band {PHAT_BAND_HZ} Hz -> bins [{K_LO}, {K_HI}] of {K_BINS}")
print("quick mode (dry run)" if QUICK else "full mode")

# %% [markdown]
# ## Cell 3 — Measurement harness
# Four benchmark questions are fixed in code:
# - **What is measured:** one function per measurement scope. CPU, GPU kernel, and GPU E2E (H2D + kernel + D2H) are separate functions.
# - **When does it end:** for the GPU, the clock stops after `deviceSynchronize` or a CUDA Event synchronization.
# - **Distribution:** the first call (`first_ms`) is recorded separately; after warm-up, each condition is repeated at least 10 and up to 30 times (2 s budget per condition), reporting median and p95. Only conditions taking over 1 s per call drop to a minimum of 5 repetitions, flagged by `reduced_reps`. A true cold start (NVRTC compilation, context, cuFFT plan creation) happens only once per kernel per notebook, so Cell 8 measures it separately (because `CUPY_CACHE_DIR` is fresh, compilation really happens).
# - **Tolerance:** performance cells run only after the correctness gate (Cell 8) passes.
#
# **Pitfall in kernel timing.** On an idle GPU, `Event → kernel → Event` includes the gap (several to tens of µs) between the GPU processing the start event and Python/CuPy issuing the kernel. Issuing R kernels back-to-back still measures launch throughput if each kernel is shorter than the launch interval.
# Fix: first enqueue a **spin kernel** (`clock64` loop) that keeps the GPU busy for a few ms, and while it runs enqueue `Event → R kernels → Event`. When the spin ends, the queued work runs without gaps, so the interval between the two events is pure GPU execution time. The difference from a separately measured "one kernel + synchronize" wall time is the **host dispatch overhead**.
#
# **Pitfall in D2H timing.** If `cp.asnumpy()` allocates a new NumPy array each time, first-touch page faults leak into the D2H time. The receiving buffer is therefore preallocated and passed via `out=`. The CPU baselines also receive preallocated output buffers so that conditions match. GPU memory is measured with a warm CuPy memory pool.
#
# **H2D timing.** `cp.asarray(numpy)` may route through an internal pinned staging buffer, which blurs the pageable vs. pinned comparison. Instead, the GPU array is preallocated and filled with `d.set(host)`: from a pageable source the driver does a staging copy + DMA, from a pinned source DMA only.

# %%
import pandas as pd


def gpu_sync():
    cp.cuda.runtime.deviceSynchronize()


def _summarize(samples, first_ms, reduced):
    x = np.asarray(samples, dtype=np.float64)
    return {"median_ms": float(np.median(x)), "p95_ms": float(np.percentile(x, 95)),
            "min_ms": float(x.min()), "n": int(x.size), "first_ms": float(first_ms), "reduced_reps": bool(reduced)}


def measure_host(fn, gpu=False):
    """Wall-clock timing of fn(). gpu=True synchronizes before the start and after the end."""
    def once():
        if gpu:
            gpu_sync()
        t0 = time.perf_counter()
        fn()
        if gpu:
            gpu_sync()
        return (time.perf_counter() - t0) * 1e3

    cold = once()
    slow = cold > 1000.0
    for _ in range(1 if slow else WARMUP):
        once()
    min_reps = min(5, MIN_REPS) if slow else MIN_REPS
    samples, start = [], time.perf_counter()
    while True:
        samples.append(once())
        elapsed = time.perf_counter() - start
        if len(samples) >= TARGET_REPS or (len(samples) >= min_reps and elapsed > BUDGET_S):
            break
    return _summarize(samples, cold, slow)


SPIN_SRC = r'''
extern "C" __global__ void spin(const long long cycles)
{   /* keeps one SM busy so the host can enqueue work behind it without gaps */
    const long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}
'''
k_spin = cp.RawKernel(SPIN_SRC, "spin")
CLOCK_HZ = float(props.get("clockRate", 1_590_000)) * 1e3


def spin(ms):
    k_spin((1,), (1,), (np.int64(ms * 1e-3 * CLOCK_HZ),))


def h2d_into(dst, src):
    """Host -> preallocated device array (cudaMemcpy semantics: pageable = staged, pinned = direct DMA)."""
    if hasattr(dst, "set"):
        dst.set(src)
    else:              # fakecupy dry run: device arrays are NumPy arrays
        dst[...] = src


def _event_ms(launch, reps, queued=True):
    if queued:
        spin(max(2.0, 0.03 * reps))        # host enqueues the timed work while the GPU is still spinning
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(reps):
        launch()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop))


def measure_kernel(launch):
    """Pure GPU time per launch (spin-queued, R launches / R) and single-launch wall time; difference = dispatch cost."""
    gpu_sync()
    first = _event_ms(launch, 1, queued=False)
    for _ in range(WARMUP):
        _event_ms(launch, 1)
    single = _event_ms(launch, 1)
    R = int(min(MAX_INNER, max(1, math.ceil(1.0 / max(single, 1e-3)))))   # aim for >= ~1 ms per sample
    samples, start = [], time.perf_counter()
    while True:
        samples.append(_event_ms(launch, R) / R)
        if len(samples) >= TARGET_REPS or (len(samples) >= MIN_REPS and time.perf_counter() - start > BUDGET_S):
            break
    st = _summarize(samples, first, False)
    wall = measure_host(launch, gpu=True)
    st.update(inner_R=R, single_wall_ms=wall["median_ms"], dispatch_overhead_ms=wall["median_ms"] - st["median_ms"])
    return st


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


print("harness ready: measure_host(fn, gpu=...), measure_kernel(launch), spin(ms), h2d_into(dst, src)")

# %% [markdown]
# ## Cell 4 — Real drone source
# Downloads indoor propeller recordings (Parrot Bebop, 16 kHz, eight ~1 s clips) from `Binary_Drone_Audio/yes_drone` of DroneAudioDataset (Al-Emadi et al., IWCMC 2019) using a fixed file list, and upsamples them to 48 kHz.
# To keep clip seams out of frames, **frames are cut only within a single clip** (about 23 frames per 1 s clip; the last clip may be shorter).
# Upsampling loses no information but adds none either: 8–24 kHz contains only independent per-microphone noise. This is why PHAT needs band limiting, which H6 tests directly.
# If the download fails, a synthetic source imitating propeller harmonics is used and labelled `SYNTHETIC FALLBACK` (H6 is then not judged).

# %%
import requests
from scipy.io import wavfile
from scipy.signal import resample_poly
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})

DRONE_BASE = "https://raw.githubusercontent.com/saraalemadi/DroneAudioDataset/master/Binary_Drone_Audio/yes_drone/"
# Files are numbered per recording (000-004); these 8 exist (checked against the GitHub listing).
DRONE_FILES = [f"B_S2_D1_067-bebop_{i:03d}_.wav" for i in range(5)] + [f"B_S2_D1_068-bebop_{i:03d}_.wav" for i in range(3)]
AUDIO_CACHE = RESULTS_DIR / "drone_audio"
AUDIO_CACHE.mkdir(exist_ok=True)


def load_drone_clips():
    clips = []
    for name in DRONE_FILES:
        path = AUDIO_CACHE / name
        if not path.exists():
            r = requests.get(DRONE_BASE + name, timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
        fs, x = wavfile.read(path)
        if fs != 16_000:
            raise ValueError(f"{name}: expected 16 kHz, got {fs}")
        x = x.astype(np.float32) / 32768.0 if x.dtype == np.int16 else x.astype(np.float32)
        x = x.mean(axis=1) if x.ndim > 1 else x
        x48 = resample_poly(x, 3, 1).astype(np.float32)
        clips.append(x48 / (np.std(x48) + 1e-12))
    return clips


def synthetic_propeller_clips(rng, n_clips=8, n=49_152):
    t = np.arange(n) / SAMPLE_RATE
    out = []
    for _ in range(n_clips):
        x = sum((0.8 ** h) * np.sin(2 * np.pi * h * 395.0 * t + rng.uniform(0, 2 * np.pi)) for h in range(1, 12))
        x = x + 2.0 * np.sin(2 * np.pi * 45.0 * t) + 0.3 * rng.standard_normal(n)
        out.append((x / np.std(x)).astype(np.float32))
    return out


try:
    DRONE_CLIPS = load_drone_clips()
    SOURCE_KIND, SOURCE_LABEL = "real_recording", "DroneAudioDataset yes_drone (Parrot Bebop, indoor), 8 x 1 s clips"
except Exception as exc:
    warnings.warn(f"Drone audio download failed ({exc}); using SYNTHETIC FALLBACK")
    DRONE_CLIPS = synthetic_propeller_clips(np.random.default_rng(SEED))
    SOURCE_KIND, SOURCE_LABEL = "synthetic_fallback", "SYNTHETIC FALLBACK propeller-like harmonics"

src_all = np.concatenate(DRONE_CLIPS)
spec = np.abs(np.fft.rfft(src_all[:SAMPLE_RATE] * np.hanning(SAMPLE_RATE))) ** 2
freqs = np.fft.rfftfreq(SAMPLE_RATE, 1 / SAMPLE_RATE)
LOW_SHARE = float(spec[freqs < PHAT_BAND_HZ[0]].sum() / spec.sum())
HIGH_SHARE = float(spec[freqs > 8000].sum() / spec.sum())
print(f"source: {SOURCE_LABEL}; {len(DRONE_CLIPS)} clips x {DRONE_CLIPS[0].size / SAMPLE_RATE:.2f} s at 48 kHz")
print(f"energy share below {PHAT_BAND_HZ[0]:.0f} Hz: {LOW_SHARE:.1%}; above 8 kHz: {HIGH_SHARE:.2%}")

fig, axes = plt.subplots(2, 1, figsize=(11, 5.5))
seg = src_all[: SAMPLE_RATE // 2]
axes[0].plot(np.arange(seg.size) / SAMPLE_RATE * 1e3, seg, lw=0.5, color="tab:blue")
axes[0].set(xlabel="time (ms)", ylabel="amplitude", title=f"Figure 0a - drone source waveform ({SOURCE_KIND})")
axes[1].specgram(src_all[: 2 * SAMPLE_RATE], NFFT=2048, Fs=SAMPLE_RATE, noverlap=1024, cmap="magma")
axes[1].axhspan(*PHAT_BAND_HZ, color="cyan", alpha=0.08)
axes[1].set(ylim=(0, 12000), xlabel="time (s)", ylabel="frequency (Hz)",
            title=f"Figure 0b - spectrogram ({LOW_SHARE:.0%} of energy below 300 Hz; nothing above 8 kHz after upsampling)")
plt.tight_layout(); save_fig(fig, "fig0_source"); plt.show()

# %% [markdown]
# ## Cell 5 — Controlled multichannel synthesis
# Far-field plane-wave model. For a source unit vector u, the delay at microphone m is −(r_m · u)/c. Fractional-sample delays are applied by linear interpolation (the interpolation error is small because the 16 kHz source was upsampled 3×). Independent noise is added per microphone, with SNR defined on full-band power.
# Because the true angle is known, correctness can be verified. Room reverberation, multipath, and Doppler are not modelled (a limitation).

# %%
MARGIN = int(np.ceil(ARRAY_RADIUS_M * SAMPLE_RATE / SOUND_SPEED)) + 8
SEG_LEN = FRAME_SIZE + 2 * MARGIN
# every whole segment that fits inside one clip, in order: (clip, start)
SEGMENTS_IN_ORDER = [(c, k * FRAME_SIZE) for c, clip in enumerate(DRONE_CLIPS)
                     for k in range((clip.size - SEG_LEN) // FRAME_SIZE + 1)]


def circular_error_deg(est, truth):
    est, truth = np.asarray(est, np.float64), np.asarray(truth, np.float64)
    return np.abs((est - truth + 180.0) % 360.0 - 180.0)


def white_segment(rng):
    x = rng.standard_normal(SEG_LEN)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(SEG_LEN, 1.0 / SAMPLE_RATE)
    X *= (f >= 300.0) & (f <= 8000.0)
    x = np.fft.irfft(X, n=SEG_LEN)
    return (x / (np.std(x) + 1e-12)).astype(np.float32)


def drone_segment(f, rng=None):
    if rng is None:   # consecutive frames inside one clip (trajectory, gate)
        clip, start = SEGMENTS_IN_ORDER[f % len(SEGMENTS_IN_ORDER)]
    else:             # random clip, random offset (accuracy trials)
        clip = int(rng.integers(len(DRONE_CLIPS)))
        start = int(rng.integers(0, DRONE_CLIPS[clip].size - SEG_LEN))
    return DRONE_CLIPS[clip][start: start + SEG_LEN]


def synthesize(azimuths_deg, source="drone", seed=SEED, snr_db=SNR_DB, random_segments=False):
    """Return [F, M, N] float32 multichannel frames for per-frame azimuths."""
    az = np.atleast_1d(azimuths_deg).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = np.arange(FRAME_SIZE, dtype=np.float64)
    grid = np.arange(SEG_LEN)
    out = np.empty((az.size, MIC_COUNT, FRAME_SIZE), np.float32)
    for f, a in enumerate(az):
        src = drone_segment(f, rng if random_segments else None) if source == "drone" else white_segment(rng)
        u = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
        delays = -(MIC_POS @ u) * SAMPLE_RATE / SOUND_SPEED
        clean = np.stack([np.interp(idx + MARGIN - d, grid, src) for d in delays]).astype(np.float32)
        sigma = np.sqrt(np.mean(clean ** 2)) * 10.0 ** (-snr_db / 20.0)
        out[f] = clean + rng.normal(0.0, sigma, clean.shape).astype(np.float32)
    return np.ascontiguousarray(out)


static_drone = synthesize(STATIC_AZIMUTHS_DEG, "drone")
static_white = synthesize(STATIC_AZIMUTHS_DEG, "white", seed=SEED + 7)
print("static frames:", static_drone.shape, static_drone.dtype, "| whole frames available in the clips:", len(SEGMENTS_IN_ORDER))

# %% [markdown]
# ## Cell 6 — CPU reference: GCC weightings and the direction scan
# - **GCC:** per frame, Hann window → FFT (NFFT=4096) → per pair `C = X_i·conj(X_j)` → weighting → IFFT → keep only lags −25..+25.
#   - mode 0 `GCC`: no weighting
#   - mode 1 `PHAT full band`: `C / |C|` (unit magnitude in every bin)
#   - mode 2 `PHAT 300–7500 Hz`: mode 1 + zero outside the band (default)
#   - mode 3 `PHAT-β 0.7, 300–7500 Hz`: `C / |C|^0.7` + zero outside the band; a standard compromise that keeps some magnitude information
# - **Direction scan:** for each candidate angle, linearly interpolate each of the 28 pairs' correlation at its expected (fractional-sample) delay, sum, and divide by 28. A pair whose expected delay falls outside the lag range contributes 0.
# - **Two CPU baselines.** `1T` = NumPy FFT + single-threaded Numba. `MT` = `scipy.fft(workers=cores)` + Numba `prange` (over the flattened (f, a) index, so it is parallel even at F=1). Colab's 2 vCPUs are usually one physical core plus its hyper-thread, so the MT gain may be only about 1.0–1.6×.

# %%
import scipy.fft as sfft
from numba import njit, prange

BAND_MASK = np.zeros(K_BINS, np.float32)
BAND_MASK[K_LO: K_HI + 1] = 1.0
HANN = np.hanning(FRAME_SIZE).astype(np.float32)


def gcc_cpu(frames, mode=DEFAULT_MODE, workers=None):
    """frames [F, M, N] -> correlations [F, P, L] float32. workers=None: NumPy FFT (single thread)."""
    frames = np.asarray(frames, np.float32)
    fwd = (lambda x: np.fft.rfft(x, n=NFFT, axis=-1)) if workers is None else \
        (lambda x: sfft.rfft(x, n=NFFT, axis=-1, workers=workers))
    inv = (lambda x: np.fft.irfft(x, n=NFFT, axis=-1)) if workers is None else \
        (lambda x: sfft.irfft(x, n=NFFT, axis=-1, workers=workers))
    spec = fwd(frames * HANN).astype(np.complex64)
    out = np.empty((frames.shape[0], P, L), np.float32)
    for p, (i, j) in enumerate(PAIRS):
        cross = spec[:, i] * np.conj(spec[:, j])
        if mode in (1, 2):
            cross = cross / np.maximum(np.abs(cross), np.float32(PHAT_EPS))
        elif mode == 3:
            cross = cross / np.maximum(np.abs(cross), np.float32(PHAT_EPS)) ** np.float32(PHAT_BETA)
        if mode >= 2:
            cross = cross * BAND_MASK
        cc = inv(cross).astype(np.float32)
        out[:, p] = np.concatenate([cc[:, -MAX_LAG:], cc[:, :MAX_LAG + 1]], axis=-1)
    return out


def candidate_angles(A):
    return np.linspace(0.0, 360.0, int(A), endpoint=False).astype(np.float32)


def delay_table_ap(angles_deg, mic_pos=MIC_POS):
    """Expected peak lag (samples) per [angle, pair]."""
    th = np.deg2rad(np.asarray(angles_deg, np.float64))
    u = np.column_stack([np.cos(th), np.sin(th)])
    base = mic_pos[PAIR_I] - mic_pos[PAIR_J]
    return np.ascontiguousarray((-(u @ base.T) * SAMPLE_RATE / SOUND_SPEED).astype(np.float32))


@njit(cache=False, inline="always")
def _score_one(corr, delays, max_lag, f, a):
    P_, L_ = corr.shape[1], corr.shape[2]
    acc = np.float32(0.0)
    for p in range(P_):
        pos = delays[a, p] + np.float32(max_lag)
        if pos < 0.0 or pos > L_ - 1:
            continue
        i0 = int(math.floor(pos))
        i1 = min(i0 + 1, L_ - 1)
        frac = np.float32(pos - i0)
        v0 = corr[f, p, i0]
        acc += v0 + frac * (corr[f, p, i1] - v0)
    return acc / np.float32(P_)


@njit(cache=False)
def cpu_scan_1t_into(corr, delays, max_lag, scores):
    for f in range(corr.shape[0]):
        for a in range(delays.shape[0]):
            scores[f, a] = _score_one(corr, delays, max_lag, f, a)
    return scores


@njit(parallel=True, cache=False)
def cpu_scan_mt_into(corr, delays, max_lag, scores):
    A_ = delays.shape[0]
    for idx in prange(corr.shape[0] * A_):          # flattened (f, a): parallel even when F == 1
        f = idx // A_
        a = idx - f * A_
        scores[f, a] = _score_one(corr, delays, max_lag, f, a)
    return scores


def cpu_scan_1t(corr, delays, max_lag):
    return cpu_scan_1t_into(corr, delays, max_lag, np.empty((corr.shape[0], delays.shape[0]), np.float32))


def cpu_scan_mt(corr, delays, max_lag):
    return cpu_scan_mt_into(corr, delays, max_lag, np.empty((corr.shape[0], delays.shape[0]), np.float32))


DELAYS_DEFAULT = delay_table_ap(candidate_angles(DEFAULT_A))
corr_static_drone = gcc_cpu(static_drone)
_ = cpu_scan_1t(corr_static_drone[:1], DELAYS_DEFAULT[:4], MAX_LAG)   # JIT compile outside timing
_ = cpu_scan_mt(corr_static_drone[:1], DELAYS_DEFAULT[:4], MAX_LAG)
print("corr:", corr_static_drone.shape, "| delay table [A, P]:", DELAYS_DEFAULT.shape)

# %% [markdown]
# ## Cell 7 — CUDA kernels (CuPy RawKernel)
# | Kernel | Mapping | Notes |
# |---|---|---|
# | `scan_ap` | thread = (f, a), delay table `[A][P]` | Reference kernel |
# | `scan_pa` | thread = (f, a), delay table `[P][A]` | Neighbouring threads read consecutive addresses (H4) |
# | `cross_phat` | thread = (f, p, k) | Complex product + weighting (mode 0/1/2/3) in one kernel, computed directly on float2 |
# | `extract_lags` | thread = (f, p, l) | Gathers lags −L..+L in order from the circular indices of the IFFT output |
# | negative control `scan_ap_offby1` | `scan_ap` with the interpolation start index shifted by one (clamped, memory-safe) | The gate must catch it as FAIL |
#
# FFT/IFFT use cuFFT (`cupy.fft`) and argmax uses a CuPy reduction. There is no reason to hand-write an FFT to beat cuFFT, so only the operations libraries lack (pairwise cross-spectrum, lag extraction, direction scan) are hand-written.

# %%
SCAN_SRC = r'''
#define SCAN_BODY(DELAY_EXPR, I0_EXPR)                                              \
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;         \
    if (idx >= (long long)F * A) return;                                            \
    const int f = (int)(idx / A);                                                   \
    const int a = (int)(idx - (long long)f * A);                                    \
    const float* c = corr + (long long)f * P * L;                                   \
    float acc = 0.0f;                                                               \
    for (int p = 0; p < P; ++p) {                                                   \
        const float pos = (DELAY_EXPR) + (float)max_lag;                            \
        if (pos < 0.0f || pos > (float)(L - 1)) continue;                           \
        const int i0 = (I0_EXPR);                                                   \
        const int i1 = (i0 + 1 < L) ? i0 + 1 : L - 1;                               \
        const float frac = pos - floorf(pos);                                       \
        const float v0 = c[p * L + i0];                                             \
        acc += v0 + frac * (c[p * L + i1] - v0);                                    \
    }                                                                               \
    scores[idx] = acc / (float)P;

extern "C" __global__
void scan_ap(const float* __restrict__ corr, const float* __restrict__ delay, float* __restrict__ scores,
             const int F, const int A, const int P, const int L, const int max_lag)
{   /* [A][P]: within a warp, consecutive a -> addresses 112 B apart */
    SCAN_BODY(delay[(long long)a * P + p], (int)floorf(pos))
}

extern "C" __global__
void scan_pa(const float* __restrict__ corr, const float* __restrict__ delay_t, float* __restrict__ scores,
             const int F, const int A, const int P, const int L, const int max_lag)
{   /* [P][A]: within a warp, consecutive a -> consecutive addresses (coalesced) */
    SCAN_BODY(delay_t[(long long)p * A + a], (int)floorf(pos))
}

extern "C" __global__
void scan_ap_offby1(const float* __restrict__ corr, const float* __restrict__ delay, float* __restrict__ scores,
                    const int F, const int A, const int P, const int L, const int max_lag)
{   /* NEGATIVE CONTROL: interpolation starts one lag too late (clamped, memory-safe) */
    SCAN_BODY(delay[(long long)a * P + p], min((int)floorf(pos) + 1, L - 1))
}
'''

PHAT_SRC = r'''
extern "C" __global__
void cross_phat(const float2* __restrict__ spec, const int* __restrict__ pi, const int* __restrict__ pj,
                float2* __restrict__ out, const int F, const int M, const int P, const int K,
                const int mode, const int k_lo, const int k_hi, const float eps, const float beta)
{
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)F * P * K) return;
    const int k = (int)(idx % K);
    const long long fp = idx / K;
    const int p = (int)(fp % P);
    const int f = (int)(fp / P);
    float2 r; r.x = 0.0f; r.y = 0.0f;
    if (mode < 2 || (k >= k_lo && k <= k_hi)) {
        const float2 a = spec[((long long)f * M + pi[p]) * K + k];
        const float2 b = spec[((long long)f * M + pj[p]) * K + k];
        float cx = a.x * b.x + a.y * b.y;          /* a * conj(b) */
        float cy = a.y * b.x - a.x * b.y;
        if (mode >= 1) {
            float mag = fmaxf(sqrtf(cx * cx + cy * cy), eps);
            if (mode == 3) mag = powf(mag, beta);  /* PHAT-beta */
            cx = cx / mag; cy = cy / mag;
        }
        r.x = cx; r.y = cy;
    }
    out[idx] = r;
}

extern "C" __global__
void extract_lags(const float* __restrict__ cc, float* __restrict__ out, const int FP, const int nfft, const int max_lag)
{
    const int L_ = 2 * max_lag + 1;
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)FP * L_) return;
    const int l = (int)(idx % L_);
    const long long fp = idx / L_;
    const int lag = l - max_lag;
    out[idx] = cc[fp * nfft + (lag < 0 ? nfft + lag : lag)];   /* circular index of the IFFT output */
}
'''

k_scan_ap = cp.RawKernel(SCAN_SRC, "scan_ap")
k_scan_pa = cp.RawKernel(SCAN_SRC, "scan_pa")
k_scan_bug = cp.RawKernel(SCAN_SRC, "scan_ap_offby1")
k_cross = cp.RawKernel(PHAT_SRC, "cross_phat")
k_extract = cp.RawKernel(PHAT_SRC, "extract_lags")
D_PAIR_I, D_PAIR_J, D_HANN = cp.asarray(PAIR_I), cp.asarray(PAIR_J), cp.asarray(HANN)


def grid_for(n, block=CUDA_BLOCK):
    return ((int(n) + block - 1) // block,)


def launch_scan(kernel, d_corr, d_delay, d_out, block=CUDA_BLOCK):
    F_, A_ = d_out.shape
    kernel(grid_for(F_ * A_, block), (block,),
           (d_corr, d_delay, d_out, np.int32(F_), np.int32(A_), np.int32(P), np.int32(L), np.int32(MAX_LAG)))


def gpu_fft(d_frames):
    return cp.fft.rfft(d_frames * D_HANN, n=NFFT, axis=-1)                    # [F, M, K] complex64


def gpu_cross(d_spec, mode=DEFAULT_MODE):
    F_ = d_spec.shape[0]
    d_cross = cp.empty((F_, P, K_BINS), dtype=cp.complex64)
    k_cross(grid_for(F_ * P * K_BINS), (CUDA_BLOCK,),
            (d_spec, D_PAIR_I, D_PAIR_J, d_cross, np.int32(F_), np.int32(MIC_COUNT), np.int32(P), np.int32(K_BINS),
             np.int32(mode), np.int32(K_LO), np.int32(K_HI), np.float32(PHAT_EPS), np.float32(PHAT_BETA)))
    return d_cross


def gpu_extract(d_cc):
    F_ = d_cc.shape[0]
    d_corr = cp.empty((F_, P, L), dtype=cp.float32)
    k_extract(grid_for(F_ * P * L), (CUDA_BLOCK,), (d_cc, d_corr, np.int32(F_ * P), np.int32(NFFT), np.int32(MAX_LAG)))
    return d_corr


def gpu_gcc(d_frames, mode=DEFAULT_MODE):
    """Device frames [F, M, N] -> device correlations [F, P, L]."""
    return gpu_extract(cp.fft.irfft(gpu_cross(gpu_fft(d_frames), mode), n=NFFT, axis=-1))


def gpu_scan_host(corr, delays_ap, kernel=k_scan_ap):
    d_corr = cp.asarray(corr)
    d_delay = cp.asarray(np.ascontiguousarray(delays_ap.T) if kernel is k_scan_pa else delays_ap)
    d_out = cp.empty((corr.shape[0], delays_ap.shape[0]), dtype=cp.float32)
    launch_scan(kernel, d_corr, d_delay, d_out)
    return cp.asnumpy(d_out)


def gpu_angles(frames, mode=DEFAULT_MODE, A=DEFAULT_A):
    angles = candidate_angles(A)
    d_tab = cp.asarray(delay_table_ap(angles))
    d_corr = gpu_gcc(cp.asarray(frames), mode)
    d_out = cp.empty((frames.shape[0], A), dtype=cp.float32)
    launch_scan(k_scan_ap, d_corr, d_tab, d_out)
    return angles[cp.asnumpy(cp.argmax(d_out, axis=1))]


print("kernels defined: scan_ap, scan_pa, scan_ap_offby1, cross_phat, extract_lags")

# %% [markdown]
# ## Cell 8 — Correctness gate (required before any timing) + negative controls
# Scan gate: ① CPU/GPU scores allclose ② argmax agreement ③ error vs. ground truth ≤ 3°.
# GCC gate: relative error of the CPU/GPU correlation tensors `max|gpu−cpu| / max|cpu| ≤ 1e-4` (all four weightings). Bit-exact agreement is not expected because the FFT implementations differ.
# First, the **true cold start** of each GPU path (first call, including NVRTC compilation and cuFFT plan creation) is compared with the second call.
#
# **Negative controls.** Evidence that the gate does not pass just anything: two deliberately wrong cases must FAIL.
# 1. Delay table with flipped sign: delays are linear in u, so it points exactly at θ+180° (a data bug).
# 2. Kernel with the interpolation index shifted by one: the angle may stay within 3°, but the element-wise tolerance catches it (a code bug). **This is why a check that compares only argmax is not enough.**

# %%
ANGLES_DEFAULT = candidate_angles(DEFAULT_A)

# True cold start: the first call of each GPU path compiles its kernels (fresh CUPY_CACHE_DIR) and builds cuFFT plans.
COLD = {}
for label, fn in [("scan path (scan_ap compile + launch)", lambda: gpu_scan_host(corr_static_drone, DELAYS_DEFAULT)),
                  ("GCC path (cross_phat/extract compile + cuFFT plans)", lambda: cp.asnumpy(gpu_gcc(cp.asarray(static_drone))))]:
    times = []
    for _ in range(2):
        gpu_sync(); t0 = time.perf_counter(); fn(); gpu_sync()
        times.append((time.perf_counter() - t0) * 1e3)
    COLD[label] = {"first_call_ms": times[0], "second_call_ms": times[1]}
display(pd.DataFrame(COLD).T.round(3))


def scan_gate(cpu_scores, gpu_scores, truth_deg):
    ci, gi = cpu_scores.argmax(1), gpu_scores.argmax(1)
    ang_err = circular_error_deg(ANGLES_DEFAULT[gi], truth_deg)
    res = {"max_abs_err": float(np.max(np.abs(cpu_scores - gpu_scores))),
           "allclose": bool(np.allclose(cpu_scores, gpu_scores, atol=SCORE_ATOL, rtol=SCORE_RTOL)),
           "argmax_same": bool(np.array_equal(ci, gi)), "max_angle_err_deg": float(ang_err.max())}
    res["PASS"] = res["allclose"] and res["argmax_same"] and res["max_angle_err_deg"] <= STATIC_MAX_ERR_DEG
    return res


gate_rows = []
cpu_ref = cpu_scan_1t(corr_static_drone, DELAYS_DEFAULT, MAX_LAG)
for label, kernel, table in [("scan_ap (baseline kernel)", k_scan_ap, DELAYS_DEFAULT),
                             ("scan_pa ([P][A] layout)", k_scan_pa, DELAYS_DEFAULT),
                             ("NEG: sign-flipped delay table", k_scan_ap, -DELAYS_DEFAULT),
                             ("NEG: off-by-one interpolation kernel", k_scan_bug, DELAYS_DEFAULT)]:
    g = scan_gate(cpu_ref, gpu_scan_host(corr_static_drone, table, kernel), STATIC_AZIMUTHS_DEG)
    gate_rows.append({"case": label, "expected": "FAIL" if label.startswith("NEG") else "PASS", **g})

corr_static_white = gcc_cpu(static_white)
g = scan_gate(cpu_scan_1t(corr_static_white, DELAYS_DEFAULT, MAX_LAG),
              gpu_scan_host(corr_static_white, DELAYS_DEFAULT), STATIC_AZIMUTHS_DEG)
gate_rows.append({"case": "scan_ap, white-noise source", "expected": "PASS", **g})

GCC_REL_ERR = {}
for mode in MODES:
    ref = gcc_cpu(static_drone, mode)
    got = cp.asnumpy(gpu_gcc(cp.asarray(static_drone), mode))
    rel = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    GCC_REL_ERR[mode] = rel
    row = {"case": f"GPU GCC vs CPU GCC, mode {mode} ({WEIGHTING[mode]})", "expected": "PASS",
           "max_abs_err": rel, "allclose": rel <= GCC_REL_TOL, "argmax_same": True, "max_angle_err_deg": np.nan}
    if mode == DEFAULT_MODE:
        g = scan_gate(cpu_ref, gpu_scan_host(got, DELAYS_DEFAULT), STATIC_AZIMUTHS_DEG)
        row.update(case=row["case"] + " -> full GPU scan", argmax_same=g["argmax_same"], max_angle_err_deg=g["max_angle_err_deg"])
    row["PASS"] = row["allclose"] and row["argmax_same"] and not (row["max_angle_err_deg"] > STATIC_MAX_ERR_DEG)
    gate_rows.append(row)

gate_df = pd.DataFrame(gate_rows)[["case", "expected", "PASS", "max_abs_err", "allclose", "argmax_same", "max_angle_err_deg"]]
gate_df["as_expected"] = gate_df.PASS == (gate_df.expected == "PASS")
display(gate_df)
gate_df.to_csv(RESULTS_DIR / "correctness_gate.csv", index=False)
print("(GCC rows report relative error max|gpu-cpu|/max|cpu| in max_abs_err)")
CORRECTNESS_PASS = bool(gate_df.as_expected.all())
if not CORRECTNESS_PASS:
    raise AssertionError("Correctness gate FAILED (or a negative control was not caught). Benchmarks stopped.")
print("CORRECTNESS GATE: PASS (all real cases pass, both negative controls caught)")

# %% [markdown]
# ## Cell 9 — Figure 1: array geometry and direction scores for one frame
# The whole score curve is computed on the GPU. Ground truth (green dashed) and argmax (red dashed) are overlaid.

# %%
full_gpu_static = gpu_scan_host(cp.asnumpy(gpu_gcc(cp.asarray(static_drone))), DELAYS_DEFAULT)
fig = plt.figure(figsize=(12, 5.2))
ax0 = fig.add_subplot(1, 2, 1)
ax0.scatter(MIC_POS[:, 0], MIC_POS[:, 1], s=90, color="tab:blue")
for m, (x, y) in enumerate(MIC_POS):
    ax0.annotate(f"M{m}", (x, y), xytext=(6, 6), textcoords="offset points")
truth1 = float(STATIC_AZIMUTHS_DEG[1])
u1 = np.array([np.cos(np.deg2rad(truth1)), np.sin(np.deg2rad(truth1))])
ax0.arrow(0, 0, 0.14 * u1[0], 0.14 * u1[1], width=0.002, head_width=0.012, length_includes_head=True, color="tab:red")
ax0.set(aspect="equal", xlim=(-0.18, 0.18), ylim=(-0.18, 0.18), xlabel="x (m)", ylabel="y (m)",
        title=f"Figure 1a - 8-mic circular array (r = 8 cm), source at {truth1:.0f} deg")
ax1 = fig.add_subplot(1, 2, 2, projection="polar")
sc = full_gpu_static[1]
pred1 = float(ANGLES_DEFAULT[sc.argmax()])
ax1.plot(np.deg2rad(ANGLES_DEFAULT), sc, color="tab:blue", lw=1.4, label="GPU score (720 directions)")
ax1.axvline(np.deg2rad(truth1), color="tab:green", ls="--", lw=2, label=f"truth {truth1:.1f}")
ax1.axvline(np.deg2rad(pred1), color="tab:red", ls=":", lw=2.5, label=f"argmax {pred1:.1f}")
ax1.set_title("Figure 1b - exhaustive azimuth score, drone recording", pad=18)
ax1.legend(loc="upper right", bbox_to_anchor=(1.4, 1.12))
plt.tight_layout(); save_fig(fig, "fig1_array_polar"); plt.show()

# %% [markdown]
# ## Cell 10 — Figure 2: time × azimuth map of a moving drone
# The drone moves from 120° to 30° (4.1 s). The drone recording and the white-noise control source go through the same full-GPU pipeline.
# Colour is the per-frame normalized score (display only); argmax is taken on the raw scores. The result is also compared with the CPU path's argmax.

# %%
traj_truth = np.linspace(120.0, 30.0, TRAJECTORY_FRAMES).astype(np.float32)
traj_time = np.arange(TRAJECTORY_FRAMES) * FRAME_MS / 1000.0
traj = {}
for src in ["drone", "white"]:
    frames = synthesize(traj_truth, src, seed=SEED + (1 if src == "drone" else 2))
    s_cpu = cpu_scan_1t(gcc_cpu(frames), DELAYS_DEFAULT, MAX_LAG)
    s_gpu = gpu_scan_host(cp.asnumpy(gpu_gcc(cp.asarray(frames))), DELAYS_DEFAULT)
    pred_cpu, pred_gpu = ANGLES_DEFAULT[s_cpu.argmax(1)], ANGLES_DEFAULT[s_gpu.argmax(1)]
    traj[src] = {"frames": frames, "scores": s_gpu, "pred_gpu": pred_gpu,
                 "err_gpu": circular_error_deg(pred_gpu, traj_truth),
                 "cpu_gpu_diff_deg": circular_error_deg(pred_cpu, pred_gpu)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, src in zip(axes, ["drone", "white"]):
    s = traj[src]["scores"]
    norm = (s - s.min(1, keepdims=True)) / np.maximum(np.ptp(s, axis=1, keepdims=True), 1e-8)
    mesh = ax.pcolormesh(traj_time, ANGLES_DEFAULT, norm.T, shading="auto", cmap="magma", vmin=0, vmax=1)
    ax.plot(traj_time, traj_truth, color="cyan", lw=2.0, label="ground truth")
    ax.plot(traj_time, traj[src]["pred_gpu"], color="lime", lw=1.1, ls="--", label="GPU argmax")
    ax.set(xlabel="time (s)", ylim=(0, 360),
           title=f"Figure 2{'a' if src == 'drone' else 'b'} - {'drone recording' if src == 'drone' else 'white-noise control'}"
                 f" (median error {np.median(traj[src]['err_gpu']):.2f} deg)")
    ax.legend(loc="upper right")
axes[0].set_ylabel("candidate azimuth (deg)")
fig.colorbar(mesh, ax=axes, label="per-frame normalized score", fraction=0.025)
save_fig(fig, "fig2_trajectory"); plt.show()

TRAJ_OK = bool(np.median(traj["drone"]["err_gpu"]) <= TRAJ_MEDIAN_ERR_DEG)
print("drone trajectory median error <= 5 deg:", TRAJ_OK,
      "| max CPU/GPU argmax difference (deg):", {k: float(v["cpu_gpu_diff_deg"].max()) for k, v in traj.items()})

# %% [markdown]
# ## Cell 11 — Accuracy: 4 weightings × 2 sources × SNR, and resolution A (H6, failure region)
# Four static directions are too small a sample. For each SNR, 108 random azimuths (with random source segments) are generated and processed by the GPU pipeline, reporting the median error and the **failure rate (error > 5°)**. This cell also shows that repeated trials that would be slow on the CPU are cheap on the GPU.
#
# **SNR pitfall.** The synthetic SNR is defined on full-band power. Most of the white-noise source's power lies inside the PHAT band (300–7500 Hz), but only about a third of the drone's does (the rest is below 300 Hz). At the same nominal SNR the drone's **in-band SNR** is therefore several dB lower, so comparisons use in-band SNR on the x-axis (H6a).
# **Model limitation.** The synthesis has no reverberation. PHAT's main advantage in real rooms is robustness to reverberation, which is not tested here. The weighting comparison holds only for an anechoic far-field model.
# Finally, the resolution A is varied to see where the error saturates. This supports treating A beyond saturation as a workload knob rather than an accuracy setting.

# %%
# in-band power share of each source (PSD from whole frames inside single clips) and of the white noise we add
f_bins = np.fft.rfftfreq(NFFT, 1 / SAMPLE_RATE)
band = (f_bins >= PHAT_BAND_HZ[0]) & (f_bins <= PHAT_BAND_HZ[1])
segs = np.stack([DRONE_CLIPS[c][st: st + FRAME_SIZE] for c, st in SEGMENTS_IN_ORDER[:64]]) * HANN
psd_drone = np.mean(np.abs(np.fft.rfft(segs, n=NFFT, axis=-1)) ** 2, axis=0)
segs_w = np.stack([white_segment(np.random.default_rng(i))[:FRAME_SIZE] for i in range(64)]) * HANN
psd_white = np.mean(np.abs(np.fft.rfft(segs_w, n=NFFT, axis=-1)) ** 2, axis=0)
NOISE_BAND_SHARE = float(band.mean())                      # flat noise: share of bins in the band
conc_rows = []
for name, psd in [("drone", psd_drone), ("white", psd_white)]:
    inb = np.sort(psd[band])[::-1]
    conc_rows.append({"source": name,
                      "share_below_300Hz": float(psd[f_bins < PHAT_BAND_HZ[0]].sum() / psd.sum()),
                      "share_in_band": float(psd[band].sum() / psd.sum()),
                      "share_above_8kHz": float(psd[f_bins > 8000].sum() / psd.sum()),
                      "top10pct_bins_share_of_band_power": float(inb[: max(1, inb.size // 10)].sum() / inb.sum())})
conc_df = pd.DataFrame(conc_rows).set_index("source")
IN_BAND_OFFSET_DB = {k: float(10 * np.log10(conc_df.loc[k, "share_in_band"] / NOISE_BAND_SHARE)) for k in conc_df.index}
conc_df["in_band_snr_minus_nominal_db"] = pd.Series(IN_BAND_OFFSET_DB)
conc_df.to_csv(RESULTS_DIR / "spectral_shares.csv")
print(f"noise: {NOISE_BAND_SHARE:.1%} of its power falls in the PHAT band; 8-24 kHz bins = {np.mean(f_bins > 8000):.0%} of all bins")
display(conc_df.round(3))

acc_rows = []
rng_acc = np.random.default_rng(SEED + 99)
for src in ["drone", "white"]:
    for snr in ACC_SNRS:
        truth = rng_acc.uniform(0, 360, ACC_TRIALS).astype(np.float32)
        frames = synthesize(truth, src, seed=int(rng_acc.integers(1 << 30)), snr_db=snr, random_segments=True)
        for mode in MODES:
            err = circular_error_deg(gpu_angles(frames, mode), truth)
            acc_rows.append({"source": src, "snr_db": snr, "in_band_snr_db": snr + IN_BAND_OFFSET_DB[src], "mode": mode,
                             "weighting": WEIGHTING[mode], "median_err_deg": float(np.median(err)),
                             "p95_err_deg": float(np.percentile(err, 95)), "fail_rate": float(np.mean(err > FAIL_ERR_DEG)),
                             "trials": ACC_TRIALS})
acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv(RESULTS_DIR / "accuracy_weighting_snr.csv", index=False)
display(acc_df.pivot_table(index=["source", "snr_db"], columns="weighting", values="fail_rate").round(3))


def snr_at_fail(g, level=0.2):
    """In-band SNR where the failure rate crosses `level` (linear interpolation; NaN if never crossed)."""
    g = g.sort_values("in_band_snr_db")
    x, y = g.in_band_snr_db.to_numpy(), g.fail_rate.to_numpy()
    for i in range(len(x) - 1):
        if y[i] >= level > y[i + 1]:
            return float(x[i] + (y[i] - level) * (x[i + 1] - x[i]) / (y[i] - y[i + 1]))
    return float("nan")


SNR20 = {(src, m): snr_at_fail(g) for (src, m), g in acc_df.groupby(["source", "mode"])}
print("in-band SNR (dB) where the failure rate falls below 20%:", {f"{k[0]}/{WEIGHTING[k[1]]}": round(v, 1) for k, v in SNR20.items()})

res_rows = []
truth = rng_acc.uniform(0, 360, ACC_TRIALS).astype(np.float32)
frames_res = synthesize(truth, "drone", seed=SEED + 5, random_segments=True)
for A_ in ACC_A_SWEEP:
    err = circular_error_deg(gpu_angles(frames_res, DEFAULT_MODE, A_), truth)
    res_rows.append({"A": A_, "grid_step_deg": 360 / A_, "median_err_deg": float(np.median(err)),
                     "p95_err_deg": float(np.percentile(err, 95))})
res_df = pd.DataFrame(res_rows)
res_df.to_csv(RESULTS_DIR / "accuracy_vs_A.csv", index=False)
display(res_df.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
styles = {0: ":", 1: "--", 2: "-", 3: "-."}
for (src, mode), g in acc_df.groupby(["source", "mode"]):
    g = g.sort_values("in_band_snr_db")
    axes[0].plot(g.in_band_snr_db, 100 * g.fail_rate, styles[mode], marker="o",
                 color="tab:red" if src == "drone" else "tab:blue", label=f"{src}, {WEIGHTING[mode]}")
axes[0].axhline(20, color="gray", lw=0.8)
axes[0].set(xlabel="in-band SNR, 300-7500 Hz (dB)", ylabel=f"failure rate (% with error > {FAIL_ERR_DEG:.0f} deg)",
            title=f"Figure 2c - where direction finding FAILS ({ACC_TRIALS} random azimuths per point)")
axes[0].legend(fontsize=6.5)
axes[1].plot(res_df.A, res_df.median_err_deg, "o-", label="median error")
axes[1].plot(res_df.A, res_df.p95_err_deg, "s--", label="p95 error")
axes[1].plot(res_df.A, res_df.grid_step_deg / 4, "k:", label="grid quantization (step / 4)")
axes[1].set(xscale="log", yscale="log", xlabel="candidate directions A", ylabel="azimuth error (deg)",
            title=f"Figure 2d - accuracy saturates; larger A is a workload knob (drone, {SNR_DB:.0f} dB)")
axes[1].legend()
plt.tight_layout(); save_fig(fig, "fig2cd_accuracy"); plt.show()

# %% [markdown]
# ## Cell 12 — Hypothesis H1: build the fixed-cost model first
# GPU E2E time and CPU time are each split into a fixed term and size-dependent terms.
#
# `T_gpu(F, A) = T0 + B_h2d/BW_h2d + B_d2h/BW_d2h + W/R_gpu`,  `T_cpu(F, A) = c0 + W/R_cpu`,  `W = F·A·28`
#
# - `B_h2d = F·28·51·4` bytes (correlation tensor), `B_d2h = F·A·4` bytes (scores). The delay table is uploaded once per configuration and excluded from timing.
# - `BW`: pageable bandwidth measured with a 32 MB round trip. `R_gpu`: kernel throughput at calibration point (16, 1440). `T0`: E2E time at calibration point (1, 36) minus its transfer and compute shares. `c0`, `R_cpu`: a line solved from the CPU times at the two calibration points.
# - The model treats F and A separately instead of a single W. Transfer scales with F, output with F·A, and compute with F·A·28, so **the winner can differ at the same W.** The crossover is therefore predicted as a boundary in the (A, F) plane rather than a single point.

# %%
def scan_bytes(F_, A_):
    return F_ * P * L * 4, F_ * A_ * 4


corr_bank_src = gcc_cpu(traj["drone"]["frames"])        # real correlations from the drone trajectory


def corr_bank(F_):
    reps = int(np.ceil(F_ / corr_bank_src.shape[0]))
    return np.ascontiguousarray(np.tile(corr_bank_src, (reps, 1, 1))[:F_])


def gpu_scan_e2e_fn(corr, d_tab):
    """Host correlations -> H2D -> scan kernel -> D2H into a preallocated host buffer. Delay table resident."""
    F_, A_ = corr.shape[0], d_tab.shape[0]
    d_corr = cp.empty(corr.shape, cp.float32)
    d_out = cp.empty((F_, A_), cp.float32)
    out_host = np.empty((F_, A_), np.float32)

    def run():
        h2d_into(d_corr, corr)
        launch_scan(k_scan_ap, d_corr, d_tab, d_out)
        cp.asnumpy(d_out, out=out_host)
    return run


probe = np.ones(8 * 2**20, np.float32)          # 32 MB, pageable
probe_back = np.empty_like(probe)
d_probe = cp.empty(probe.shape, cp.float32)
BW_H2D = probe.nbytes / (measure_host(lambda: h2d_into(d_probe, probe), gpu=True)["median_ms"] / 1e3)
BW_D2H = probe.nbytes / (measure_host(lambda: cp.asnumpy(d_probe, out=probe_back), gpu=True)["median_ms"] / 1e3)
del d_probe

calib = {}
for F_, A_ in CALIBRATION:
    corr, tab = corr_bank(F_), delay_table_ap(candidate_angles(A_))
    d_c, d_t, d_o = cp.asarray(corr), cp.asarray(tab), cp.empty((F_, A_), cp.float32)
    out_c = np.empty((F_, A_), np.float32)
    calib[(F_, A_)] = {"cpu1t": measure_host(lambda: cpu_scan_1t_into(corr, tab, MAX_LAG, out_c))["median_ms"],
                       "cpumt": measure_host(lambda: cpu_scan_mt_into(corr, tab, MAX_LAG, out_c))["median_ms"],
                       "kernel": measure_kernel(lambda: launch_scan(k_scan_ap, d_c, d_t, d_o))["median_ms"],
                       "e2e": measure_host(gpu_scan_e2e_fn(corr, d_t), gpu=True)["median_ms"]}
(F_s, A_s), (F_m, A_m) = CALIBRATION
W_s, W_m = F_s * A_s * P, F_m * A_m * P
R_GPU = W_m / calib[(F_m, A_m)]["kernel"]


def _transfer_ms(F_, A_):
    h, d = scan_bytes(F_, A_)
    return (h / BW_H2D + d / BW_D2H) * 1e3


T0_MS = max(calib[(F_s, A_s)]["e2e"] - _transfer_ms(F_s, A_s) - W_s / R_GPU, 0.0)
CPU_LINE = {}
for base in ["cpu1t", "cpumt"]:
    dt = calib[(F_m, A_m)][base] - calib[(F_s, A_s)][base]
    if dt > 0:
        r = (W_m - W_s) / dt
        CPU_LINE[base] = (max(calib[(F_s, A_s)][base] - W_s / r, 0.0), r)    # (c0_ms, R lookups/ms)
    else:                                   # non-physical fit (noise): fall back to a pure-throughput line
        CPU_LINE[base] = (0.0, W_m / calib[(F_m, A_m)][base])


def predict_gpu_e2e(F_, A_):
    return T0_MS + _transfer_ms(F_, A_) + F_ * A_ * P / R_GPU


def predict_cpu(F_, A_, base="cpu1t"):
    c0, r = CPU_LINE[base]
    return c0 + F_ * A_ * P / r


def predicted_boundary(base="cpu1t", F_grid=np.unique(np.geomspace(1, 4096, 60).astype(int)),
                       A_grid=np.geomspace(4, 200_000, 600)):
    """For each F, the smallest A where the model says GPU E2E < CPU (NaN if never within the grid)."""
    out = []
    for F_ in F_grid:
        wins = [A_ for A_ in A_grid if predict_gpu_e2e(F_, A_) < predict_cpu(F_, A_, base)]
        out.append(wins[0] if wins else np.nan)
    return F_grid, np.array(out)


def predicted_cross_along(vary, base="cpu1t"):
    """Model crossover along F=1 (vary='A') or along A=FIXED_A (vary='F'); log-interpolated, NaN if none."""
    xs = np.geomspace(4, 200_000, 2000) if vary == "A" else np.geomspace(1, 65_536, 2000)
    for x in xs:
        F_, A_ = (1, x) if vary == "A" else (x, FIXED_A)
        if predict_gpu_e2e(F_, A_) < predict_cpu(F_, A_, base):
            return float(x)
    return float("nan")


PRED_CROSS = {"A_at_F1": predicted_cross_along("A"), "F_at_A1440": predicted_cross_along("F")}
MODEL = {"T0_ms": T0_MS, "BW_H2D_GBps": BW_H2D / 1e9, "BW_D2H_GBps": BW_D2H / 1e9,
         "pred_crossover_A_at_F1": PRED_CROSS["A_at_F1"], "pred_crossover_F_at_A1440": PRED_CROSS["F_at_A1440"],
         "R_gpu_lookups_per_us": R_GPU / 1e3,
         "cpu1t_c0_ms": CPU_LINE["cpu1t"][0], "cpu1t_R_lookups_per_us": CPU_LINE["cpu1t"][1] / 1e3,
         "cpumt_c0_ms": CPU_LINE["cpumt"][0], "cpumt_R_lookups_per_us": CPU_LINE["cpumt"][1] / 1e3,
         "calibration_points": [list(c) for c in CALIBRATION]}
(RESULTS_DIR / "fixed_cost_model.json").write_text(json.dumps(MODEL, indent=2))
for k, v in MODEL.items():
    print(f"{k:26s}: {v:.4g}" if isinstance(v, float) else f"{k:26s}: {v}")
print("\nModel fixed BEFORE the sweep; Cell 13 tests it on configurations it has not seen.")

# %% [markdown]
# ## Cell 13 — Direction scan alone: CPU 1T / CPU MT / GPU kernel / GPU E2E
# - The input is the correlation tensor of a real drone trajectory, repeated to F frames (real values, only the size is enlarged).
# - kernel: input already on the GPU, output preallocated; R spin-queued launches / R (Cell 3). `dispatch_overhead` = single-launch wall time − kernel time.
# - E2E: host correlation tensor → H2D (`set`, pageable) → kernel → score D2H (preallocated buffer) → synchronize. The CPU is also measured with a preallocated output buffer under the same conditions.
# - The A axis is measured on two lines, F=64 and F=1. The F=1 line gives the measured crossover A*.
# - Conditions are measured in shuffled order so that GPU clock/temperature drift does not concentrate on one size. GPU state is recorded before and after the sweep.

# %%
cases = sorted({(FIXED_F, a) for a in A_SWEEP} | {(1, a) for a in A_SWEEP} | {(f, FIXED_A) for f in F_SWEEP} |
               {(f, a) for f in GRID_F for a in GRID_A})
order = np.random.default_rng(SEED).permutation(len(cases))
ENV["gpu_state_before_sweep"] = gpu_state()
rows = []
for n, ci in enumerate(order, 1):
    F_, A_ = cases[ci]
    corr, tab = corr_bank(F_), delay_table_ap(candidate_angles(A_))
    d_c, d_t, d_o = cp.asarray(corr), cp.asarray(tab), cp.empty((F_, A_), cp.float32)
    out_c = np.empty((F_, A_), np.float32)
    cpu1 = measure_host(lambda: cpu_scan_1t_into(corr, tab, MAX_LAG, out_c))
    cpum = measure_host(lambda: cpu_scan_mt_into(corr, tab, MAX_LAG, out_c))
    ker = measure_kernel(lambda: launch_scan(k_scan_ap, d_c, d_t, d_o))
    e2e = measure_host(gpu_scan_e2e_fn(corr, d_t), gpu=True)
    rows.append({"F": F_, "A": A_, "W_lookups": F_ * A_ * P, "calibration_point": (F_, A_) in CALIBRATION,
                 **{f"cpu1t_{k}": v for k, v in cpu1.items()}, **{f"cpumt_{k}": v for k, v in cpum.items()},
                 **{f"kernel_{k}": v for k, v in ker.items()}, **{f"e2e_{k}": v for k, v in e2e.items()},
                 "pred_e2e_ms": predict_gpu_e2e(F_, A_), "pred_cpu1t_ms": predict_cpu(F_, A_, "cpu1t"),
                 "pred_cpumt_ms": predict_cpu(F_, A_, "cpumt")})
    print(f"[{n:02d}/{len(cases)}] F={F_:5d} A={A_:5d}  CPU1T {cpu1['median_ms']:9.3f}  CPUMT {cpum['median_ms']:9.3f}  "
          f"kernel {ker['median_ms']:8.4f} (dispatch {ker['dispatch_overhead_ms']:.4f})  E2E {e2e['median_ms']:8.3f} ms")
    del d_c, d_t, d_o
ENV["gpu_state_after_sweep"] = gpu_state()
sweep = pd.DataFrame(rows).sort_values(["F", "A"]).reset_index(drop=True)
for base in ["cpu1t", "cpumt"]:
    sweep[f"speedup_e2e_vs_{base}"] = sweep[f"{base}_median_ms"] / sweep["e2e_median_ms"]
    sweep[f"speedup_kernel_vs_{base}"] = sweep[f"{base}_median_ms"] / sweep["kernel_median_ms"]
    sweep[f"gpu_wins_{base}"] = sweep[f"{base}_median_ms"] > sweep["e2e_median_ms"]
    sweep[f"pred_gpu_wins_{base}"] = sweep[f"pred_{base}_ms"] > sweep["pred_e2e_ms"]
sweep["pred_e2e_err_pct"] = 100 * (sweep.pred_e2e_ms - sweep.e2e_median_ms) / sweep.e2e_median_ms
sweep.to_csv(RESULTS_DIR / "scan_sweep.csv", index=False)
(RESULTS_DIR / "env.json").write_text(json.dumps(ENV, indent=2))
display(sweep[["F", "A", "W_lookups", "cpu1t_median_ms", "cpumt_median_ms", "kernel_median_ms", "kernel_dispatch_overhead_ms",
               "e2e_median_ms", "e2e_p95_ms", "e2e_n", "speedup_e2e_vs_cpu1t", "speedup_e2e_vs_cpumt", "pred_e2e_err_pct"]].round(4))

# %% [markdown]
# ## Cell 14 — Figure 3: win/lose boundary, predicted vs. measured
# (a) Increasing A at F=1, (b) increasing F at A=1440. On each line, the **measured crossover** where CPU 1T and GPU E2E meet (log interpolation) and the model-predicted crossover are drawn as vertical lines. (c) The (A, F) plane: model boundary and measured winners (filled = GPU E2E wins, hollow = CPU 1T wins).
# H1 is judged not by the percentage of correctly predicted winners but by **whether the crossover location is within 2× of the prediction**. Conditions far from the boundary have obvious winners, which inflates a winner-accuracy percentage.

# %%
def measured_cross(sub, xcol, base="cpu1t"):
    """Where log(CPU/GPU E2E) crosses 0 along one line (log-log interpolation); NaN if not crossed."""
    sub = sub.sort_values(xcol)
    r = np.log(sub[f"{base}_median_ms"].to_numpy() / sub.e2e_median_ms.to_numpy())
    x = np.log(sub[xcol].to_numpy(dtype=float))
    for i in range(len(r) - 1):
        if r[i] <= 0 < r[i + 1]:
            return float(np.exp(x[i] - r[i] * (x[i + 1] - x[i]) / (r[i + 1] - r[i])))
    return float("nan")


MEAS_CROSS = {"A_at_F1": measured_cross(sweep[sweep.F == 1], "A"), "F_at_A1440": measured_cross(sweep[sweep.A == FIXED_A], "F")}
CROSS_RATIO = {k: MEAS_CROSS[k] / PRED_CROSS[k] for k in MEAS_CROSS}
oos = sweep[~sweep.calibration_point]
WINNER_ACC = {b: float((oos[f"gpu_wins_{b}"] == oos[f"pred_gpu_wins_{b}"]).mean()) for b in ["cpu1t", "cpumt"]}
cross_df = pd.DataFrame({"measured": MEAS_CROSS, "predicted": PRED_CROSS, "measured/predicted": CROSS_RATIO})
display(cross_df.round(3))
print("secondary: out-of-sample winner accuracy", WINNER_ACC,
      f"| median |E2E prediction error| {oos.pred_e2e_err_pct.abs().median():.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, sub, xcol, key) in zip(axes[:2], [("A sweep at F=1", sweep[sweep.F == 1], "A", "A_at_F1"),
                                                   (f"F sweep at A={FIXED_A}", sweep[sweep.A == FIXED_A], "F", "F_at_A1440")]):
    sub = sub.sort_values(xcol)
    for col, lab, st in [("cpu1t_median_ms", "CPU 1 thread", "o-"),
                         ("cpumt_median_ms", f"CPU {ENV['numba_threads']} threads", "o--"),
                         ("kernel_median_ms", "GPU kernel", "s-"), ("e2e_median_ms", "GPU E2E (H2D+kernel+D2H)", "s-"),
                         ("pred_e2e_ms", "GPU E2E model", "k:")]:
        ax.plot(sub[xcol], sub[col], st, label=lab)
    for val, ls_, lab in [(MEAS_CROSS[key], "-", "measured crossover"), (PRED_CROSS[key], "--", "predicted crossover")]:
        if np.isfinite(val):
            ax.axvline(val, color="gray", ls=ls_, lw=1.2, label=f"{lab} {val:.0f}")
    ax.set(xscale="log", yscale="log", xlabel=xcol, ylabel="median latency (ms)", title=f"Figure 3{'ab'[xcol == 'F']} - {title}")
    ax.legend(fontsize=7)
ax = axes[2]
for base, color in [("cpu1t", "tab:blue"), ("cpumt", "tab:purple")]:
    Fg, Ab = predicted_boundary(base)
    ax.plot(Ab, Fg, "-", color=color, lw=2, label=f"model boundary vs {'CPU 1T' if base == 'cpu1t' else 'CPU MT'}")
win = sweep.gpu_wins_cpu1t
ax.scatter(sweep.A[win], sweep.F[win], s=70, color="tab:blue", label="measured: GPU E2E beats CPU 1T")
ax.scatter(sweep.A[~win], sweep.F[~win], s=70, facecolors="none", edgecolors="tab:blue", label="measured: CPU 1T wins")
ax.scatter(*np.array(CALIBRATION)[:, ::-1].T, marker="x", s=90, color="k", label="calibration points")
ax.set(xscale="log", yscale="log", xlabel="candidate directions A", ylabel="frames per batch F",
       title="Figure 3c - who wins where (model boundary vs measured)")
ax.legend(fontsize=7)
plt.tight_layout(); save_fig(fig, "fig3_crossover"); plt.show()

# %% [markdown]
# ## Cell 15 — Hypotheses H2 and H3: pipeline breakdown (CPU / H2D / kernel / D2H / host wait)
# The same input (F frames of real drone trajectory audio) is processed by five pipelines. All of them output a per-frame direction.
#
# | Pipeline | Stages |
# |---|---|
# | CPU 1T | NumPy GCC-PHAT → Numba 1T scan → argmax |
# | CPU MT | scipy.fft(workers) GCC-PHAT → Numba MT scan → argmax |
# | Hybrid | NumPy GCC-PHAT → correlation tensor H2D → GPU scan → argmax → D2H |
# | Full GPU | audio H2D → window + cuFFT → cross_phat → inverse cuFFT → extract → scan → argmax → D2H `[F]` |
# | Full GPU (pinned) | Same as above, with the audio in a pinned host buffer (assumes the audio driver writes into a pinned ring buffer) |
#
# GPU segments are separated by Events at every stage boundary. Each pipeline is run twice:
# - **Segment timing:** a spin kernel after H2D lets the remaining GPU stages queue without gaps → Event intervals = pure GPU execution time. D2H is timed only after the last Event has completed (otherwise unfinished GPU work would be attributed to D2H).
# - **Total time:** plain wall time without the spin.
# - `host wait` = total wall time − (CPU segments + pure GPU segments + D2H): the Python/CuPy time spent issuing kernels plus synchronization wait. For small batches this can be the largest term.
# - H2D copies into a preallocated GPU array with `set` (pageable source = staging + DMA, pinned source = DMA only).
#
# Amdahl bound: if the scan takes a fraction p of the CPU 1T time, the overall speedup cannot exceed 1/(1−p) even with an infinitely fast scan. p depends on A, so it is measured at three values of A.

# %%
SEGMENTS = {
    "cpu1t": ["gcc", "scan"], "cpumt": ["gcc", "scan"],
    "hybrid": ["gcc", "h2d", "scan", "argmax", "d2h", "host_gap"],
    "gpu": ["h2d", "fft", "phat", "ifft", "extract", "scan", "argmax", "d2h", "host_gap"],
    "gpu_pinned": ["h2d", "fft", "phat", "ifft", "extract", "scan", "argmax", "d2h", "host_gap"],
}


class Pipelines:
    """All five pipelines for one (F, A), with preallocated host/device buffers.

    run(kind, queued=True): GPU stages are enqueued behind a spin kernel, so each Event interval is pure GPU time.
    run(kind, queued=False): plain run; only its wall-clock total is used (host_gap = plain total - sum of segments).
    """

    def __init__(self, frames, A_):
        self.F, self.A = frames.shape[0], A_
        self.frames = frames
        self.frames_pinned = cupyx.empty_pinned(frames.shape, np.float32)
        self.frames_pinned[...] = frames
        self.angles = candidate_angles(A_)
        self.tab = delay_table_ap(self.angles)
        self.d_tab = cp.asarray(self.tab)
        self.d_frames = cp.empty(frames.shape, cp.float32)
        self.d_corr = cp.empty((self.F, P, L), cp.float32)
        self.d_out = cp.empty((self.F, A_), cp.float32)
        self.scores_cpu = np.empty((self.F, A_), np.float32)
        self.idx_host = np.empty(self.F, np.int64)
        self.idx_pinned = cupyx.empty_pinned(self.F, np.int64)

    def run(self, kind, queued=True):
        if kind in ("cpu1t", "cpumt"):
            mt = kind == "cpumt"
            t0 = time.perf_counter()
            corr = gcc_cpu(self.frames, workers=ENV["numba_threads"] if mt else None)
            t1 = time.perf_counter()
            (cpu_scan_mt_into if mt else cpu_scan_1t_into)(corr, self.tab, MAX_LAG, self.scores_cpu)
            pred = self.angles[self.scores_cpu.argmax(1)]
            t2 = time.perf_counter()
            return {"gcc": (t1 - t0) * 1e3, "scan": (t2 - t1) * 1e3, "total": (t2 - t0) * 1e3}, pred
        gpu_sync()
        t0 = time.perf_counter()
        seg = {}
        if kind == "hybrid":
            corr = gcc_cpu(self.frames)
            seg["gcc"] = (time.perf_counter() - t0) * 1e3
            ev = [cp.cuda.Event() for _ in range(5)]
            ev[0].record(); h2d_into(self.d_corr, corr); ev[1].record()
            if queued:
                spin(3.0)
            ev[2].record()
            launch_scan(k_scan_ap, self.d_corr, self.d_tab, self.d_out); ev[3].record()
            d_idx = cp.argmax(self.d_out, axis=1); ev[4].record()
            ev[4].synchronize()                          # everything queued is done before the D2H clock starts
            t2 = time.perf_counter(); cp.asnumpy(d_idx, out=self.idx_host); gpu_sync(); t3 = time.perf_counter()
            seg.update(h2d=cp.cuda.get_elapsed_time(ev[0], ev[1]), scan=cp.cuda.get_elapsed_time(ev[2], ev[3]),
                       argmax=cp.cuda.get_elapsed_time(ev[3], ev[4]), d2h=(t3 - t2) * 1e3)
            idx = self.idx_host
        else:
            src = self.frames_pinned if kind == "gpu_pinned" else self.frames
            idx = self.idx_pinned if kind == "gpu_pinned" else self.idx_host
            ev = [cp.cuda.Event() for _ in range(9)]
            ev[0].record(); h2d_into(self.d_frames, src); ev[1].record()
            if queued:
                spin(3.0)
            ev[2].record()
            d_spec = gpu_fft(self.d_frames); ev[3].record()
            d_cross = gpu_cross(d_spec); ev[4].record()
            d_cc = cp.fft.irfft(d_cross, n=NFFT, axis=-1); ev[5].record()
            d_corr = gpu_extract(d_cc); ev[6].record()
            launch_scan(k_scan_ap, d_corr, self.d_tab, self.d_out); ev[7].record()
            d_idx = cp.argmax(self.d_out, axis=1); ev[8].record()
            ev[8].synchronize()
            t2 = time.perf_counter(); cp.asnumpy(d_idx, out=idx); gpu_sync(); t3 = time.perf_counter()
            seg["h2d"] = cp.cuda.get_elapsed_time(ev[0], ev[1])
            for i, name in enumerate(["fft", "phat", "ifft", "extract", "scan", "argmax"]):
                seg[name] = cp.cuda.get_elapsed_time(ev[i + 2], ev[i + 3])
            seg["d2h"] = (t3 - t2) * 1e3
            del d_spec, d_cross, d_cc, d_corr
        seg["total"] = (time.perf_counter() - t0) * 1e3
        return seg, self.angles[idx]


def tiled_frames(F_):
    src = traj["drone"]["frames"]
    return np.ascontiguousarray(np.tile(src, (int(np.ceil(F_ / src.shape[0])), 1, 1))[:F_])


def free_gpu_memory():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cp.fft.config.get_plan_cache().clear()


def pipeline_breakdown(F_, A_):
    pipes = Pipelines(tiled_frames(F_), A_)
    reps = 3 if QUICK else (15 if F_ <= 64 else 7)
    rec = {k: [] for k in SEGMENTS}
    agree = {}
    for r in range(reps + 2):                         # first two rounds = warm-up (cuFFT plans, numba threads, pool)
        preds = {}
        for kind in SEGMENTS:
            seg, preds[kind] = pipes.run(kind, queued=True)
            if kind not in ("cpu1t", "cpumt"):
                plain, _ = pipes.run(kind, queued=False)
                seg["total"] = plain["total"]
                seg["host_gap"] = max(plain["total"] - sum(v for k_, v in seg.items() if k_ != "total"), 0.0)
            if r >= 2:
                rec[kind].append(seg)
        if r == 0:
            agree = {k: float(circular_error_deg(preds[k], preds["cpu1t"]).max()) for k in ("hybrid", "gpu", "gpu_pinned")}
    summary = {k: {"median": pd.DataFrame(v).median().to_dict(), "p95": pd.DataFrame(v).quantile(0.95).to_dict()}
               for k, v in rec.items()}
    del pipes
    free_gpu_memory()
    return summary, agree


breakdowns = {}
for F_, A_ in BREAK_CASES:
    s_, agree = pipeline_breakdown(F_, A_)
    breakdowns[(F_, A_)] = s_
    m = {k: v["median"]["total"] for k, v in s_.items()}
    p = s_["cpu1t"]["median"]["scan"] / m["cpu1t"]
    print(f"F={F_:5d} A={A_:5d}: CPU1T {m['cpu1t']:.2f} | CPUMT {m['cpumt']:.2f} | hybrid {m['hybrid']:.2f} | "
          f"full GPU {m['gpu']:.2f} | pinned {m['gpu_pinned']:.2f} ms; scan share p={p:.1%}, Amdahl bound {1 / (1 - p):.2f}x, "
          f"hybrid {m['cpu1t'] / m['hybrid']:.2f}x; max argmax difference vs CPU (deg) {agree}")

bd_rows = [{"F": F_, "A": A_, "pipeline": k, "segment": seg, "median_ms": v["median"][seg], "p95_ms": v["p95"][seg]}
           for (F_, A_), s_ in breakdowns.items() for k, v in s_.items() for seg in v["median"]]
pd.DataFrame(bd_rows).to_csv(RESULTS_DIR / "pipeline_breakdown.csv", index=False)

# %% [markdown]
# ### Figure 5 — Stacked bars (per-segment medians) and the Amdahl bound vs. A

# %%
COLORS = {"gcc": "#4C72B0", "scan": "#C44E52", "h2d": "#DD8452", "fft": "#55A868", "ifft": "#55A868",
          "phat": "#8CC78C", "extract": "#8CC78C", "argmax": "#C44E52", "d2h": "#937860", "host_gap": "#BBBBBB"}
LABELS = {"gcc": "GCC-PHAT on CPU", "scan": "direction scan (+argmax)", "h2d": "H2D", "fft": "FFT + IFFT (cuFFT)",
          "phat": "PHAT + lag-extract kernels", "d2h": "D2H", "host_gap": "host dispatch / sync wait"}
GROUP = {"ifft": "fft", "extract": "phat", "argmax": "scan"}
show_cases = [c for c in [(64, 1440), (1024, 1440)] if c in breakdowns] or list(breakdowns)[:2]
pipes = ["cpu1t", "cpumt", "hybrid", "gpu", "gpu_pinned"]
pipe_labels = ["CPU 1T", f"CPU {ENV['numba_threads']}T", "Hybrid", "Full GPU", "Full GPU\n(pinned)"]

fig, axes = plt.subplots(1, len(show_cases) + 1, figsize=(6.5 * (len(show_cases) + 1), 5.5))
for ax, case in zip(axes, show_cases):
    s = breakdowns[case]
    bottom = np.zeros(len(pipes))
    for key in ["gcc", "h2d", "fft", "phat", "scan", "d2h", "host_gap"]:
        vals = np.array([sum(s[p]["median"].get(seg, 0.0) for seg in SEGMENTS[p] if GROUP.get(seg, seg) == key)
                         for p in pipes])
        ax.bar(pipe_labels, vals, bottom=bottom, color=COLORS[key], label=LABELS[key])
        bottom += vals
    totals = [s[p]["median"]["total"] for p in pipes]
    for i, tot in enumerate(totals):
        ax.text(i, max(bottom[i], tot) * 1.02, f"{tot:.2f} ms\n{totals[0] / tot:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set(ylabel="median latency (ms, log)", title=f"Figure 5{'ab'[show_cases.index(case)]} - pipeline, F={case[0]}, A={case[1]}")
    ax.set_ylim(min(totals) * 0.3, max(totals) * 3)
axes[0].legend(fontsize=7, loc="upper right")
ax = axes[-1]
amd = []
for (F_, A_), s in breakdowns.items():
    p = s["cpu1t"]["median"]["scan"] / s["cpu1t"]["median"]["total"]
    amd.append({"F": F_, "A": A_, "p_scan": p, "bound": 1 / (1 - p),
                "hybrid_speedup": s["cpu1t"]["median"]["total"] / s["hybrid"]["median"]["total"],
                "full_gpu_speedup": s["cpu1t"]["median"]["total"] / s["gpu"]["median"]["total"]})
amd_df = pd.DataFrame(amd)
amd_df.to_csv(RESULTS_DIR / "amdahl.csv", index=False)
sub = amd_df[amd_df.F == amd_df.F.min()].sort_values("A")
ax.plot(sub.A, sub.bound, "k--o", label="Amdahl bound 1/(1-p) for 'scan only on GPU'")
ax.plot(sub.A, sub.hybrid_speedup, "s-", color="#C44E52", label="measured: hybrid (scan on GPU)")
ax.plot(sub.A, sub.full_gpu_speedup, "^-", color="#55A868", label="measured: full GPU pipeline")
for _, r in sub.iterrows():
    ax.annotate(f"p={r.p_scan:.0%}", (r.A, r.bound), xytext=(4, 6), textcoords="offset points", fontsize=8)
ax.set(xscale="log", yscale="log", xlabel="candidate directions A", ylabel="speedup vs CPU 1T pipeline",
       title=f"Figure 5c - Amdahl: moving only the scan is capped (F={int(sub.F.iloc[0])})")
ax.legend(fontsize=7)
plt.tight_layout(); save_fig(fig, "fig5_pipeline_amdahl"); plt.show()
display(amd_df.round(3))

# %% [markdown]
# ## Cell 16 — Hypothesis H5: real-time capacity (how many arrays one GPU can serve)
# If S arrays each send one frame for the same instant, the batch to process at once is F = S, and it must finish before the next frame arrives (42.7 ms).
# The verdict uses **p95**, not the median, and compares against the best CPU pipeline. Pipelines whose median exceeds 3× the deadline are not measured at larger S.
# Throughput (frames/s) is not the same as real-time capacity. This definition holds because the system processes the same-instant frames of S arrays together, without waiting to accumulate larger batches.

# %%
cap_rows, stopped = [], set()
KINDS = ["cpu1t", "cpumt", "hybrid", "gpu", "gpu_pinned"]
for S in CAPACITY_S:
    pipes = Pipelines(tiled_frames(S), FIXED_A)
    for kind in KINDS:
        if kind in stopped:
            continue
        st = measure_host(lambda: pipes.run(kind, queued=False), gpu=kind not in ("cpu1t", "cpumt"))
        cap_rows.append({"S_arrays": S, "pipeline": kind, **st, "meets_deadline_p95": st["p95_ms"] <= FRAME_MS})
        if st["median_ms"] > 3 * FRAME_MS:
            stopped.add(kind)
    del pipes
    free_gpu_memory()
    print(f"S={S:5d}: " + "  ".join(f"{r['pipeline']}={r['p95_ms']:.2f}" for r in cap_rows if r["S_arrays"] == S) + " ms (p95)")
cap_df = pd.DataFrame(cap_rows)
cap_df.to_csv(RESULTS_DIR / "capacity.csv", index=False)


def capacity_of(g):
    ok = g[g.meets_deadline_p95].S_arrays
    return int(ok.max()) if len(ok) else 0


CAPACITY = {k: capacity_of(g) for k, g in cap_df.groupby("pipeline")}
CAPACITY_LOWER_BOUND = {k: bool(g.sort_values("S_arrays").meets_deadline_p95.iloc[-1]) for k, g in cap_df.groupby("pipeline")}
BEST_CPU = max(["cpu1t", "cpumt"], key=lambda k: CAPACITY[k])
print(f"real-time capacity (max arrays with p95 <= {FRAME_MS:.1f} ms):", CAPACITY)
print("'+' = still meeting the deadline at the largest tested S (true capacity is higher):", CAPACITY_LOWER_BOUND)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
names = {"cpu1t": "CPU 1 thread", "cpumt": f"CPU {ENV['numba_threads']} threads", "hybrid": "Hybrid",
         "gpu": "Full GPU", "gpu_pinned": "Full GPU, pinned"}
for kind in KINDS:
    g = cap_df[cap_df.pipeline == kind].sort_values("S_arrays")
    ax.plot(g.S_arrays, g.p95_ms, "o-", label=f"{names[kind]}: {CAPACITY[kind]}{'+' if CAPACITY_LOWER_BOUND[kind] else ''} arrays")
ax.axhline(FRAME_MS, color="red", ls="--", label=f"real-time deadline {FRAME_MS:.1f} ms")
ax.set(xscale="log", yscale="log", xlabel="microphone arrays processed together (S)", ylabel="p95 latency per batch (ms)",
       title=f"Figure 6 - real-time capacity at A={FIXED_A} directions")
ax.legend(fontsize=8)
plt.tight_layout(); save_fig(fig, "fig6_capacity"); plt.show()

# %% [markdown]
# ## Cell 17 — Hypothesis H4: delay-table layout and block size (appendix)
# The 32 threads of a warp handle neighbouring angles a.
# - `[A][P]`: at a given p, threads are 112 B apart → one warp read splits into 32 sectors (fetching 8× the needed bytes). But the same thread re-reads the same sector for p+1..p+7, so it is reused if it stays in L1. The whole table (403 KB at A=3600) also fits in the T4's L2 (4 MB).
# - `[P][A]`: at a given p, addresses are contiguous → a single 128 B chunk.
# - Hence the **difference is predicted to be under 1.2×**. The kernel's main memory pattern is actually the correlation-tensor gather (a different position within 51 values for each thread).

# %%
F_l, A_l = (256, 3600) if not QUICK else (16, 1440)
corr_l, tab_l = corr_bank(F_l), delay_table_ap(candidate_angles(A_l))
d_c, d_ap, d_pa = cp.asarray(corr_l), cp.asarray(tab_l), cp.asarray(np.ascontiguousarray(tab_l.T))
d_o = cp.empty((F_l, A_l), cp.float32)
layout_rows = []
for bs in BLOCK_SIZES:
    for name, kern, tab in [("[A][P] scan_ap", k_scan_ap, d_ap), ("[P][A] scan_pa", k_scan_pa, d_pa)]:
        st = measure_kernel(lambda: launch_scan(kern, d_c, tab, d_o, block=bs))
        layout_rows.append({"layout": name, "block": bs, **st})
layout_df = pd.DataFrame(layout_rows)
layout_df.to_csv(RESULTS_DIR / "layout_blocksize.csv", index=False)
piv = layout_df.pivot(index="block", columns="layout", values="median_ms")
display(piv.round(5))
H4_RATIO = float(piv["[A][P] scan_ap"].min() / piv["[P][A] scan_pa"].min())
print(f"F={F_l}, A={A_l}: best [A][P] / best [P][A] = {H4_RATIO:.3f}x")

fig, ax = plt.subplots(figsize=(8, 4.5))
for name in piv.columns:
    ax.plot(piv.index, piv[name], "o-", label=name)
ax.set(xscale="log", xlabel="threads per block", ylabel="kernel time per launch (ms)",
       title=f"Figure 7 - delay-table layout x block size (F={F_l}, A={A_l}); ratio {H4_RATIO:.2f}x")
ax.set_xticks(BLOCK_SIZES); ax.set_xticklabels(BLOCK_SIZES); ax.legend()
plt.tight_layout(); save_fig(fig, "fig7_layout_blocksize"); plt.show()
del d_c, d_ap, d_pa, d_o

# %% [markdown]
# ## Cell 18 — Hypothesis verdicts and automatic summary
# Every verdict is built only from objects produced in this run. Required criteria (is the experiment valid?) are kept separate from hypothesis verdicts (was the prediction right?). A rejected hypothesis does not invalidate the experiment.

# %%
def verdict(ok, inconclusive=False):
    return "INCONCLUSIVE" if inconclusive else ("SUPPORTED" if ok else "NOT SUPPORTED")


main_case = (64, 1440) if (64, 1440) in breakdowns else list(breakdowns)[0]
gseg = {k: v for k, v in breakdowns[main_case]["gpu"]["median"].items() if k != "total"}
merged = {"H2D": gseg["h2d"], "FFT+IFFT": gseg["fft"] + gseg["ifft"], "PHAT+extract": gseg["phat"] + gseg["extract"],
          "scan+argmax": gseg["scan"] + gseg["argmax"], "D2H": gseg["d2h"], "host dispatch/sync": gseg["host_gap"]}
largest = max(merged, key=merged.get)
cap_ratio = CAPACITY.get("gpu", 0) / max(CAPACITY.get(BEST_CPU, 0), 1)
ratios_ok = [0.5 <= r <= 2.0 for r in CROSS_RATIO.values() if np.isfinite(r)]
fail_mean = acc_df.groupby(["source", "mode"]).fail_rate.mean()
s20_drone, s20_white = SNR20.get(("drone", DEFAULT_MODE), np.nan), SNR20.get(("white", DEFAULT_MODE), np.nan)

H = [
    ("H1 crossover location predicted within 2x", verdict(len(ratios_ok) > 0 and all(ratios_ok), inconclusive=len(ratios_ok) == 0),
     "measured/predicted: " + ", ".join(f"{k} {MEAS_CROSS[k]:.0f}/{PRED_CROSS[k]:.0f} = {CROSS_RATIO[k]:.2f}" for k in MEAS_CROSS)
     + f"; (secondary) winner accuracy {WINNER_ACC['cpu1t']:.0%}"),
    ("H2 scan-only speedup capped by Amdahl; cap grows with A",
     verdict(bool((amd_df.hybrid_speedup <= amd_df.bound * 1.05).all()) and sub.bound.is_monotonic_increasing),
     "; ".join(f"A={r.A}: p={r.p_scan:.0%} bound {r.bound:.2f}x hybrid {r.hybrid_speedup:.2f}x" for r in sub.itertuples())),
    ("H3 full-GPU largest segment is H2D or FFT", verdict(largest in ("H2D", "FFT+IFFT")),
     f"F={main_case[0]}, A={main_case[1]}: " + ", ".join(f"{k} {v:.3f}" for k, v in merged.items()) + " ms"),
    ("H4 [A][P] vs [P][A] differ by < 1.2x (L1 reuse)", verdict(H4_RATIO < 1.2), f"ratio {H4_RATIO:.3f}x"),
    ("H5 full-GPU capacity >= 10x best CPU", verdict(cap_ratio >= 10),
     f"capacity {CAPACITY} (best CPU = {BEST_CPU}, Colab {ENV['cpu_logical_cores']} vCPU); '+' lower bounds {CAPACITY_LOWER_BOUND}"),
    ("H6a drone harder than white at the same in-band SNR",
     verdict(bool(s20_drone > s20_white), inconclusive=SOURCE_KIND != "real_recording" or not (np.isfinite(s20_drone) and np.isfinite(s20_white))),
     f"in-band SNR for <20% failures ({WEIGHTING[DEFAULT_MODE]}): drone {s20_drone:.1f} dB, white {s20_white:.1f} dB"),
    ("H6b band-limited PHAT fails less than full-band PHAT (drone)",
     verdict(bool(fail_mean[("drone", 2)] < fail_mean[("drone", 1)]), inconclusive=SOURCE_KIND != "real_recording"),
     "drone mean failure rate: " + ", ".join(f"{WEIGHTING[m]} {fail_mean[('drone', m)]:.1%}" for m in MODES)),
]
hyp_df = pd.DataFrame(H, columns=["hypothesis", "verdict", "evidence"])
with pd.option_context("display.max_colwidth", 200):
    display(hyp_df)
hyp_df.to_csv(RESULTS_DIR / "hypotheses.csv", index=False)

required = {
    "real GPU run (not emulated)": not IS_FAKE_GPU,
    "correctness gate passed and both negative controls caught": CORRECTNESS_PASS,
    "static angle error <= 3 deg": bool(gate_df[gate_df.expected == "PASS"].max_angle_err_deg.max() <= STATIC_MAX_ERR_DEG),
    "drone trajectory median error <= 5 deg": TRAJ_OK,
    ">= 10 repetitions per timed condition (or flagged)": bool(((sweep.e2e_n >= MIN_REPS) | sweep.e2e_reduced_reps).all()),
    "kernel / E2E / dispatch overhead separated": True,
    "pipeline segments separated (CPU / H2D / kernel / D2H / host)": len(breakdowns) > 0,
}
print("\n=== REQUIRED (experiment validity) ===")
for k, v in required.items():
    print(f"[{'PASS' if v else 'FAIL'}] {k}")
summary = {"env": ENV, "source": SOURCE_LABEL, "model": MODEL, "winner_accuracy": WINNER_ACC, "capacity": CAPACITY,
           "capacity_lower_bound": CAPACITY_LOWER_BOUND, "gcc_rel_err": GCC_REL_ERR, "required": required,
           "hypotheses": hyp_df.to_dict("records"), "cold_start": COLD, "measured_crossover": MEAS_CROSS,
           "predicted_crossover": PRED_CROSS}
(RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\nCSV / JSON / PNG saved under {RESULTS_DIR}")

# %% [markdown]
# ## What this experiment shows / does not show
#
# **Shows.** On this session's GPU and CPU: from what size the exhaustive (frame, angle) search favours the GPU and how well a fixed-cost model predicts that boundary; why moving only the search yields a small overall gain (Amdahl, as a function of A); where the time goes once preprocessing also moves to the GPU (transfers, FFT, host dispatch); how many arrays can therefore be processed in real time under a p95 deadline; and which weighting real drone sound needs, and why.
#
# **Does not show.** Accuracy on real multichannel recordings (the array delays are synthetic); reverberation, multipath, and Doppler; elevation; drone detection; speedups on other GPUs/CPUs; session-to-session variance (this is a single session).
#
# **Next experiments.**
# 1. If the remaining time is H2D and host dispatch: overlap per-array transfers and compute with a pinned ring buffer + CUDA Streams, and cut dispatch cost with CUDA Graphs.
# 2. Only 51 of the 4096 IFFT lags (1.2%) are used. Frequency-domain SRP-PHAT (`Σ_p Σ_k Re(G_p[k]·e^{jωτ_p(a)})`) needs no IFFT and is compute-bound, which makes it a good roofline comparison.
# 3. Add a Log-Mel branch on the same STFT to detect drones vs. non-drones. Classifiers typically use 16 kHz and 25 ms windows while direction finding uses 48 kHz and 2048 points, so the STFT parameters need a compromise.
# 4. Direction-scan kernel: stage the per-frame correlation tile (28×51 floats = 5.7 KB) in shared memory. Putting the delay table in constant memory would backfire with this mapping (each thread reads a different a, so accesses serialize).
# 5. Validate accuracy on real multichannel recordings (UaVirBASE) and extend to a 2D azimuth × elevation search.
