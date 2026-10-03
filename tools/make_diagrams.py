"""Render the README/docs diagrams in the style of the Phase 1 report (Pretendard, light-gray cards).

Usage: python tools/make_diagrams.py <pretendard_static_otf_dir> [out_dir=docs/assets/diagrams]
Pretendard (SIL OFL): https://github.com/orioncactus/pretendard/releases (public/static/*.otf).
Needs matplotlib. Phase 2 bar values come from the Phase 1 reference run (pipeline_breakdown.csv).
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow, Polygon  # noqa: E402

INK, INK2, MUTED = "#111111", "#444444", "#7A7A7A"
CARD, CARD_HI, RAIL = "#F0F0F0", "#E2E2E2", "#D0D0D0"
NODE, NODE_OFF = "#4A4A4A", "#B5B5B5"
ACCENT, NEUTRAL = "#FF6A13", "#BDBDBD"
DPI = 200
ROOT = Path(__file__).resolve().parents[1]
BREAKDOWN = ROOT / "experiments/phase1_gpu_doa/results/colab-t4_2026-10-02/pipeline_breakdown.csv"


def load_fonts(font_dir):
    for f in Path(font_dir).glob("Pretendard-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Pretendard"


def canvas(w_px, h_px):
    fig = plt.figure(figsize=(w_px / DPI, h_px / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)          # y grows downward, like a slide
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def card(ax, x, y, w, h, color=CARD):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                fc=color, ec="none"))


def text(ax, x, y, s, size, weight="regular", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va)


def save(fig, out_dir, name):
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(path)


def pipeline(out_dir):
    """Four-step computation, as on report page 2."""
    steps = [("01", "8-channel input", "real drone audio\n+ per-mic arrival delays"),
             ("02", "Pair time\ndifferences", "28 mic pairs\nFFT / GCC-PHAT"),
             ("03", "Score every\ndirection", "A candidate angles,\neach scored on the GPU"),
             ("04", "Pick the best", "argmax(score)\n→ estimated azimuth")]
    W, H, pad, gap = 1800, 360, 30, 28
    cw = (W - 2 * pad - 3 * gap) / 4
    fig, ax = canvas(W, H)
    for i, (num, title, sub) in enumerate(steps):
        x = pad + i * (cw + gap)
        card(ax, x, 40, cw, 280, CARD_HI if num == "03" else CARD)
        text(ax, x + cw / 2, 88, num, 11, "bold")
        text(ax, x + cw / 2, 148, title, 12, "bold" if num == "03" else "semibold")
        text(ax, x + cw / 2, 248, sub, 9, "semibold" if num == "03" else "regular", INK2)
        if i < 3:
            ax.add_patch(Polygon([[x + cw + 8, 168], [x + cw + gap - 8, 180], [x + cw + 8, 192]],
                                 closed=True, fc=NODE_OFF, ec="none"))
    save(fig, out_dir, "pipeline")


def roadmap(out_dir):
    """Phase timeline, as the rail-and-node timeline on report page 1 (kept compact for the README)."""
    phases = [("Phase 1", "GPU direction finding", "Crossover · Amdahl · capacity", "Done", True),
              ("Phase 2", "Kernel optimization", "Remove the IFFT · cut H2D · tune the scan", "Next", True),
              ("Phase 3", "Detection + direction", "Shared STFT · Log-Mel · multi-array", "Later", False)]
    W, H = 1800, 230
    fig, ax = canvas(W, H)
    x0, x1, yr = 120, 1700, 50
    ax.add_patch(FancyBboxPatch((x0 - 40, yr - 6), x1 - x0 + 40, 12,
                                boxstyle="round,pad=0,rounding_size=3", fc=RAIL, ec="none"))
    ax.add_patch(Polygon([[x1, yr - 18], [x1 + 28, yr], [x1, yr + 18]], closed=True, fc=RAIL, ec="none"))
    xs = [340, 900, 1460]
    for x, (tag, title, sub, status, active) in zip(xs, phases):
        done = status == "Done"
        ax.add_patch(Circle((x, yr), 18, fc=NODE if done else ("white" if active else NODE_OFF),
                            ec=NODE if active else NODE_OFF, lw=3))
        if done:
            ax.plot([x - 7, x - 2, x + 8], [yr + 1, yr + 6, yr - 6], color="white", lw=2.5,
                    solid_capstyle="round", solid_joinstyle="round")
        c = INK if active else MUTED
        text(ax, x, 100, f"{tag}  ·  {status}", 9, "semibold", INK2 if active else MUTED)
        text(ax, x, 143, title, 11.5, "bold", c)
        text(ax, x, 188, sub, 8.5, "regular", INK2 if active else MUTED)
    save(fig, out_dir, "roadmap")


def phase1_flow(out_dir):
    """Phase 1 question chain with the headline answer of each step (report pages 4-7)."""
    steps = [("H1", "When does the GPU\nbeat the CPU?", "≈ 340", "candidate directions", "model predicted ≈ 369"),
             ("H2", "Is moving only the\nscan enough?", "1.11×", "whole-program speedup", "Amdahl bound 1.16×"),
             ("H3", "Move preprocessing\ntoo?", "2.58 ms", "from 99.4 ms (CPU)", "1.89 ms with pinned input"),
             ("H5", "How many arrays\nin real time?", "1,024", "arrays (CPU: 16)", "p95 ≤ 42.7 ms deadline")]
    W, H, pad, gap = 1800, 520, 30, 44
    cw = (W - 2 * pad - 3 * gap) / 4
    fig, ax = canvas(W, H)
    for i, (hid, q, big, unit, note) in enumerate(steps):
        x = pad + i * (cw + gap)
        card(ax, x, 30, cw, 460)
        text(ax, x + cw / 2, 78, hid, 13, "bold")
        text(ax, x + cw / 2, 150, q, 13, "semibold", INK2)
        ax.plot([x + 40, x + cw - 40], [212, 212], color=RAIL, lw=1.5)
        text(ax, x + cw / 2, 292, big, 34, "heavy")
        text(ax, x + cw / 2, 370, unit, 12, "semibold")
        text(ax, x + cw / 2, 425, note, 10.5, "regular", MUTED)
        if i < 3:
            xa = x + cw + 10
            ax.add_patch(Polygon([[xa, 244], [xa + gap - 20, 260], [xa, 276]], closed=True, fc=NODE_OFF, ec="none"))
    save(fig, out_dir, "phase1_flow")


def phase2_targets(out_dir):
    """Where the 1.89 ms of the pinned full-GPU pipeline goes (F=64, A=1440), with the Phase 2 action per segment."""
    med = {}
    for r in csv.DictReader(open(BREAKDOWN)):
        if r["F"] == "64" and r["A"] == "1440" and r["pipeline"] == "gpu_pinned":
            med[r["segment"]] = float(r["median_ms"])
    rows = [("IFFT + lag extraction", med["ifft"] + med["extract"], "Replace with a 51-lag DFT (GEMM)"),
            ("H2D (pinned audio)", med["h2d"], "int16 transfer, overlap with streams"),
            ("FFT", med["fft"], "NFFT 4096 → 2048"),
            ("PHAT weighting", med["phat"], "Fuse into the lag DFT"),
            ("Scan + argmax", med["scan"] + med["argmax"], "Shared-memory tile, on-the-fly delays"),
            ("D2H + host dispatch", med["d2h"] + med["host_gap"], "CUDA Graph")]
    total = sum(v for _, v, _ in rows)
    W, H = 1800, 640
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.235, 0.05, 0.40, 0.76])
    ys = range(len(rows))[::-1]
    vmax = max(v for _, v, _ in rows)
    ax.set_xlim(0, vmax * 1.45)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    for y, (name, v, action) in zip(ys, rows):
        hi = v == vmax
        ax.add_patch(FancyBboxPatch((0, y - 0.31), v, 0.62, boxstyle="round,pad=0,rounding_size=0.012",
                                    fc=ACCENT if hi else NEUTRAL, ec="none", mutation_aspect=8))
        ax.text(v + 0.02, y, f"{v:.2f} ms  ({v / total:.0%})", va="center", fontsize=11,
                fontweight="bold" if hi else "semibold", color=INK)
        ax.text(-0.03, y, name, va="center", ha="right", fontsize=11.5,
                fontweight="bold" if hi else "medium", color=INK)
        fig.text(0.70, ax.transData.transform((0, y))[1] / H, action, va="center", fontsize=11,
                 fontweight="semibold" if hi else "regular", color=INK if hi else INK2)
    ax.axis("off")
    fig.text(0.04, 0.93, f"Pinned full-GPU pipeline, F=64, A=1440 — {total:.2f} ms", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.70, 0.86, "Phase 2 action", fontsize=11, fontweight="bold", color=MUTED)
    fig.text(0.04, 0.86, "Phase 1 median per segment (Colab T4)", fontsize=11, color=MUTED)
    save(fig, out_dir, "phase2_targets")


if __name__ == "__main__":
    load_fonts(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "docs/assets/diagrams")
    os.makedirs(out, exist_ok=True)
    for fn in (pipeline, roadmap, phase1_flow, phase2_targets):
        fn(out)
