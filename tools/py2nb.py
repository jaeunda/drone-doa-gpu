"""Convert a percent-format script (# %% / # %% [markdown]) into an .ipynb.

Usage: python tools/py2nb.py <source.py> <out.ipynb>
Markdown cells: every line must start with '# ' (or be '#'), which is stripped.
"""
import json
import sys


def convert(src_path, out_path):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        text = "\n".join(lines)
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)

    for line in open(src_path, encoding="utf-8").read().splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{out_path}: {len(cells)} cells")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
