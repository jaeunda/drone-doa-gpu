# Phase 1 — GPU direction finding

How far must the GPU-DOA computation move to the GPU for end-to-end real-time performance to improve?
Results write-up: [`docs/phase1.md`](../../docs/phase1.md).

| File | Content |
|---|---|
| [`phase1_doa_scan.py`](phase1_doa_scan.py) | Notebook source in percent format. Edit this file, then regenerate the `.ipynb` |
| [`phase1_doa_scan.ipynb`](phase1_doa_scan.ipynb) | Colab notebook (18 cells: environment → correctness gate → accuracy → H1–H6 → verdicts) |
| [`design.md`](design.md) | Design, hypotheses with rationale, measurement protocol, limitations |
| [`results/colab-t4_2026-10-02/`](results/colab-t4_2026-10-02/) | Reference run on Colab Tesla T4: CSV/JSON outputs and figures |
| [`report/phase1_interim_report.pdf`](report/phase1_interim_report.pdf) | Interim report (Korean) |

## Run on Colab

1. Upload `phase1_doa_scan.ipynb` to Google Colab.
2. *Runtime → Change runtime type → T4 GPU*, then *Run all* (about 10 minutes).
3. Outputs are written to `/content/uav_results/`. Download them before the session ends and store them as
   `results/<environment>_<date>/` (leave out `drone_audio/`; the dataset is not redistributed).

## Regenerate the notebook

From the repository root:

```bash
python3 tools/py2nb.py experiments/phase1_gpu_doa/phase1_doa_scan.py experiments/phase1_gpu_doa/phase1_doa_scan.ipynb
```

## Local dry run (no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/run_notebook.py experiments/phase1_gpu_doa/phase1_doa_scan.ipynb /tmp/dry/phase1
```

`tools/fakecupy` compiles each RawKernel's CUDA C with g++ (needs `g++` on PATH) and runs it once per
(block, thread). Kernel logic, gates, and figures are exercised, but there are no warps or memory hierarchy,
so **dry-run timings and hypothesis verdicts are meaningless.**

## Result files

| File | Produced by | Content |
|---|---|---|
| `env.json` | Cell 1 | GPU/CPU, driver and library versions, GPU clocks before/after sweeps |
| `correctness_gate.csv` | Cell 8 | Gate checks and negative controls |
| `accuracy_weighting_snr.csv`, `accuracy_vs_A.csv`, `spectral_shares.csv` | Cell 11 | Accuracy by weighting × source × SNR, by resolution A |
| `fixed_cost_model.json` | Cell 12 | Calibrated two-term model (H1) |
| `scan_sweep.csv` | Cell 13 | Scan-only CPU / GPU kernel / GPU E2E sweep |
| `pipeline_breakdown.csv`, `amdahl.csv` | Cell 15 | Per-segment pipeline timings, Amdahl bound (H2, H3) |
| `capacity.csv` | Cell 16 | p95 latency vs. number of arrays (H5) |
| `layout_blocksize.csv` | Cell 17 | Delay-table layout × block size (H4) |
| `hypotheses.csv`, `summary.json` | Cell 18 | Verdicts and run summary |
| `figs/` | Cells 4–17 | Figures 0–7 |
