# Phase 1 design — GPU direction finding

Design record for `phase1_doa_scan.ipynb` (source: `phase1_doa_scan.py`).
Results of the reference run: `results/colab-t4_2026-10-02/`. Report: `report/phase1_interim_report.pdf`. Results write-up: `docs/phase1.md`.

## 1. Problem and scope

Given a drone sound captured by an 8-microphone circular array, estimate its azimuth by scoring every
candidate direction (GCC-PHAT + steered-response-power scan) and taking the argmax. Phase 1 asks:

- From what problem size does the GPU beat the CPU, and can a simple fixed-cost model predict it?
- Why does moving only the scan to the GPU give a small overall gain (Amdahl)?
- Where does the bottleneck move once the GCC-PHAT preprocessing also runs on the GPU?
- How many arrays can one GPU serve in real time (p95 latency within the 42.7 ms frame deadline)?

Scenario: drone-intrusion monitoring around airports or power plants, where one server processes many
microphone arrays at once. Phase 1 assumes a drone is present and estimates direction only; detection is a
later phase (`../../docs/roadmap.md`).

## 2. Starting point (baseline run)

An earlier baseline notebook (2026-10-02, Colab T4) established the (frame, angle) thread mapping and CPU/GPU
agreement (max error 7.45e-8, identical argmax), but left these gaps, which shaped Phase 1:

| Baseline observation | Phase 1 response |
|---|---|
| CPU GCC-PHAT 71.6 ms vs. GPU scan 0.63 ms → only 1.21× overall | Amdahl bound as a function of A; GCC-PHAT moved to the GPU |
| Even the smallest case favoured the GPU (1.12×) → crossover outside the measured range | Go down to A=36, F=1; predict a boundary in the (A, F) plane |
| `Event → kernel → Event` timings of 0.017–0.025 ms ≈ launch latency | Spin-queued R launches / R, plus single-launch wall time → dispatch overhead |
| 7 repetitions (3 for large cases) | Adaptive repetitions (min 10, target 30) |
| Band-limited white noise as the source | Real drone recording + white-noise control |
| CPU alone already handles one array in real time | Real-time capacity (arrays per GPU under a p95 deadline) as the headline metric |

## 3. Design summary

| Item | Design |
|---|---|
| Input | 8-channel float32 audio `[F, 8, 2048]` (48 kHz, 42.7 ms frames) |
| Output | Scores `[F, A]`; the real-time path returns only the argmax index `[F]` |
| Parallel unit | Scan: 1 thread = (frame, angle), 28 pairs looped in-thread. PHAT: 1 thread = (frame, pair, bin). Lag extraction: 1 thread = (frame, pair, lag) |
| Transfers & dependencies | Frames independent, stages sequential. Delay table uploaded once per configuration and excluded from timing |
| Expected bottleneck | Scan-only GPU: CPU GCC-PHAT. Full GPU: pageable audio H2D; host dispatch for small batches |
| Validation | Score allclose (atol 2e-5, rtol 2e-4), argmax agreement, error vs. truth ≤ 3°, GCC relative error ≤ 1e-4, negative controls must FAIL |

## 4. Hypotheses

Fixed before measurement. Numeric predictions come from a model calibrated on the baseline run and are
out-of-sample with respect to its calibration points.

| ID | Hypothesis | Criterion | Rationale |
|---|---|---|---|
| H1 | Small workloads favour the CPU because of GPU fixed costs; a two-term model predicts the boundary in the (F, A) plane | Measured crossovers (A* at F=1, F* at A=1440) within 2× of prediction. Winner accuracy is secondary (conditions far from the boundary are trivially predicted) | Transfer ∝ F, output ∝ F·A, compute ∝ F·A·28, so the winner can differ at equal work |
| H2 | Scan-only GPU speedup is capped by Amdahl 1/(1−p), and p grows with A | Hybrid ≤ bound (5% slack) at every A; bound increases with A | Baseline: p=18% at A=1440 → 1.22× cap; larger A raises p |
| H3 | With everything on the GPU, the largest segment is audio H2D or FFT | Largest segment of the breakdown, host wait included | 4 MB pageable H2D at F=64 ≈ ≥1 ms vs. FFT+PHAT ≈ 0.2–0.4 ms |
| H4 | `[A][P]` and `[P][A]` delay-table layouts differ by < 1.2× | Kernel time ratio | 8× sector overfetch per warp, but L1 reuse across p and the table fits in L2 |
| H5 | Full-GPU p95 real-time capacity ≥ 10× best CPU pipeline | Capacity ratio | Capacity = max S with p95(T(S arrays × 1 frame)) ≤ 42.7 ms |
| H6a | The drone source is harder than white noise at equal in-band SNR | In-band SNR at 20% failure is higher for the drone | Nominal SNR is full-band; the drone has ~1/3 of its power in band vs. ~94% for white noise, so in-band SNR must be used |
| H6b | Band-limited PHAT fails less than full-band PHAT on the drone | SNR-averaged failure rate; GCC and PHAT-β (0.7) also reported | Full-band PHAT gives weight 1 to the signal-free 8–24 kHz bins (2/3 of all bins) |

## 5. Experiment design

### 5.1 Input data
- **Source:** DroneAudioDataset (Al-Emadi et al., IWCMC 2019, `saraalemadi/DroneAudioDataset`),
  `Binary_Drone_Audio/yes_drone`, files `B_S2_D1_067-bebop_000..004_.wav` and `B_S2_D1_068-bebop_000..002_.wav`
  (Parrot Bebop, indoor, 16 kHz, ~1 s each). Downloaded at run time; not redistributed in this repository.
- **Spectrum:** ~80% of the energy below 300 Hz, harmonic at ~395 Hz; nothing above 8 kHz after upsampling to 48 kHz.
- **Framing:** frames are cut within a single clip only, so clip seams never fall inside a frame.
- **Array:** 8 mics on an 8 cm-radius circle; far-field plane wave; fractional delays by linear interpolation;
  independent per-mic noise with full-band SNR.
- **Control source:** 300 Hz–8 kHz band-limited white noise.
- **Fallback:** synthetic propeller harmonics labelled `SYNTHETIC FALLBACK`; H6 is not judged in that case.

### 5.2 Kernels

| Kernel | Mapping | Purpose |
|---|---|---|
| `scan_ap` | (f, a), table `[A][P]` | Reference scan |
| `scan_pa` | (f, a), table `[P][A]` | H4 |
| `scan_ap_offby1` | interpolation start shifted by one (clamped) | Negative control |
| `cross_phat` | (f, p, k), weighting modes 0–3 | Cross-spectrum + weighting |
| `extract_lags` | (f, p, l) | Circular IFFT index → lags −L..+L |
| FFT/IFFT, argmax | cuFFT, CuPy reduction | Library calls |

### 5.3 Workloads
- A ∈ {36, 90, 360, 1440, 3600, 7200} at F=64 and F=1; F ∈ {1, 4, 16, 64, 256, 1024} at A=1440;
  grid {36, 360, 3600} × {1, 16, 256}.
- Accuracy saturates around A≈360–720, so A=3600/7200 act as a workload knob standing in for an
  azimuth 360 × elevation 10–20 grid.
- Model calibration points: (F=1, A=36) and (F=16, A=1440).

### 5.4 Measurement protocol
- **Kernel time:** a `clock64` spin kernel keeps the GPU busy while `Event → R launches → Event` is queued, so
  the interval is pure GPU time ÷ R (R chosen for ≥ 1 ms, max 200). Single-launch wall time minus this is the
  host dispatch overhead.
- **H2D:** `set` into a preallocated device array (pageable = driver staging + DMA, pinned = DMA only);
  `cp.asarray` is avoided because its internal staging blurs the comparison.
- **D2H and CPU outputs:** preallocated host buffers on both sides.
- **E2E:** host input → H2D → kernels → D2H into a preallocated buffer → synchronize; pageable by default,
  plus a pinned variant for the full-GPU path.
- **Cold start:** fresh `CUPY_CACHE_DIR`, so the first call includes NVRTC compilation; measured once per path.
- **Repetitions:** warm-up 3; min 10, target 30, 2 s budget per condition; conditions over 1 s/call drop to 5 and are flagged.
- **Ordering:** conditions shuffled; GPU clocks and temperature recorded before and after sweeps.
- **Pipeline breakdown:** Events at stage boundaries in a spin-queued run; total time from a separate plain run;
  host wait = wall − Σ segments. Memory pools and cuFFT plan caches are cleared between capacity conditions.

### 5.5 Correctness gate and negative controls
- Scan: allclose, argmax agreement, static-direction error ≤ 3°, for the reference kernel, `[P][A]`, and the white-noise source.
- GCC: relative error ≤ 1e-4 for all weightings; full-GPU argmax agreement.
- Negative controls: a sign-flipped table (must point to θ+180°) and the off-by-one kernel (angle may stay
  within 3°, but allclose must fail — evidence that argmax-only checks are insufficient).
- Any failure raises `AssertionError` and stops the notebook.

### 5.6 Accuracy experiment
- 108 random azimuths × SNR {−20, −15, −10, −5, 0, 10} dB × 2 sources × 4 weightings (GCC, PHAT full band,
  PHAT 300–7500 Hz, PHAT-β 0.7 band-limited), processed on the GPU path.
- x-axis is in-band SNR = nominal SNR + 10·log10(source in-band share / noise in-band share).
- Metrics: median and p95 error, failure rate (> 5°), in-band SNR at 20% failure; per-source spectral shares.
- Anechoic model: PHAT's robustness to reverberation is not tested.

### 5.7 Fairness
- Same float32/complex64 precision, interpolation, and boundary handling on CPU and GPU.
- Library vs. library for FFTs: NumPy pocketfft (1T) / `scipy.fft` (workers) vs. cuFFT.
- Audio synthesis and downloads are excluded from timing.

## 6. Notebook map

| Cell | Content | Output |
|---|---|---|
| 1 | Environment | `env.json` |
| 2–3 | Configuration, measurement harness | |
| 4 | Drone source | Figure 0 |
| 5–7 | Multichannel synthesis, CPU reference, CUDA kernels | |
| 8 | Correctness gate + negative controls | `correctness_gate.csv` |
| 9–10 | Figure 1 (array, polar scores), Figure 2 (trajectory map) | |
| 11 | Accuracy (H6), Figures 2c/2d | `accuracy_*.csv`, `spectral_shares.csv` |
| 12 | Fixed-cost model (H1) | `fixed_cost_model.json` |
| 13–14 | Scan-only sweep, Figure 3 crossover | `scan_sweep.csv` |
| 15 | Pipeline breakdown, Figure 5 (H2, H3) | `pipeline_breakdown.csv`, `amdahl.csv` |
| 16 | Real-time capacity, Figure 6 (H5) | `capacity.csv` |
| 17 | Table layout × block size, Figure 7 (H4) | `layout_blocksize.csv` |
| 18 | Verdicts and summary | `hypotheses.csv`, `summary.json` |

## 7. Validity criteria

Required for the run to count: real GPU; gate passed and both negative controls caught; static error ≤ 3°;
drone trajectory median error ≤ 5°; ≥ 10 repetitions per condition (or flagged); kernel / E2E / dispatch
separated; pipeline segments separated. Hypothesis verdicts are not validity criteria — a rejected hypothesis
is a result.

## 8. Limitations

- Anechoic far-field plane-wave model: no reverberation, multipath, or Doppler.
- The drone sound is real, but the array delays are synthetic (a controlled experiment built from real audio).
- 2D circular array: no elevation estimate.
- Colab's 2 vCPUs are typically one physical core plus its hyper-thread, which limits the multithreaded CPU baseline.
- Single T4 session; session-to-session variance is not measured.
- Large F is obtained by repeating a 96-frame correlation tensor (real values, limited diversity).

## 9. Local dry-run notes

`tools/run_notebook.py` with `tools/fakecupy` executes the notebook without a GPU (RawKernel CUDA C compiled
with g++ and run per thread). It validates logic, gates, and figures; timings and verdicts are meaningless.
Findings that changed the notebook:

- The original SNR range showed 0% failures even at 0 dB, so the range was moved to {−20 … 10} dB.
- The dataset numbers files 000–004 per recording; requesting 005–007 returned 404 and silently triggered the
  synthetic fallback. The file list was fixed to the eight files that exist.
- An apparent "unweighted GCC beats PHAT" result on the synthetic fallback did not reproduce on the real
  recording (CPU probe, 80 trials per SNR: drone failure at −15 dB was GCC 57% / full-band PHAT 66% /
  band-limited PHAT 49%).
