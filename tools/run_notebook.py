"""Dry-run a notebook locally (no GPU) with the fakecupy shim.

Usage: .venv/bin/python tools/run_notebook.py <notebook.ipynb> <out_dir>
(e.g. experiments/phase1_gpu_doa/phase1_doa_scan.ipynb /tmp/dry/phase1)

Executes code cells in order in one namespace, saves every figure that
plt.show() would display, and stops at the first failing cell. Timings under
the shim are meaningless; this only checks logic, shapes, gates, and figures.
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fakecupy"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_REPORT_QUICK", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(nb_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    nb = json.load(open(nb_path, encoding="utf-8"))
    counter = {"n": 0, "cell": -1}

    def show(*a, **k):
        for num in plt.get_fignums():
            counter["n"] += 1
            path = os.path.join(out_dir, f"fig{counter['n']:02d}_cell{counter['cell']:02d}.png")
            plt.figure(num).savefig(path, dpi=80)
            print(f"[saved figure] {path}")
        plt.close("all")

    plt.show = show

    def display(*objs, **k):
        for o in objs:
            try:
                import pandas as pd
                if isinstance(o, pd.DataFrame):
                    with pd.option_context("display.width", 200, "display.max_columns", 30):
                        print(o.to_string(max_rows=40))
                    continue
            except ImportError:
                pass
            print(o)

    ns = {"__name__": "__main__", "display": display}
    os.chdir(out_dir)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        counter["cell"] = i
        src = "".join(cell["source"])
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        code = "\n".join(lines)
        print(f"\n===== cell {i} =====")
        t0 = time.perf_counter()
        try:
            exec(compile(code, f"<cell {i}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"\n!!!!! FAILED at cell {i}")
            sys.exit(1)
        show()
        print(f"[cell {i} done in {time.perf_counter() - t0:.1f}s]")
    print("\nALL CELLS PASSED")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
