#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import pickle


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("autorl-dfl-mm", "RTDFL"),
    ("dfl-mm", "MMDFL"),
    ("dfl", "DFL"),
    ("dfedpgp", "DFedPGP"),
    ("ld-sgd", "LD-SGD"),
)
COLORS = ("#243b64", "#c95652", "#4f8a5b", "#8060a8", "#d18b32")


def _load_accuracy(path):
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        value = value["acc"]
    return [float(item) for item in value]


def _method_for_run(run_name):
    for suffix, label in METHODS:
        if run_name.endswith(f"-{suffix}"):
            return label
    raise ValueError(f"unknown quick run name: {run_name}")


def _plot_accuracy(run_dir):
    curves = {}
    for path in run_dir.rglob("test_acc.pkl"):
        curves[_method_for_run(path.parent.name)] = _load_accuracy(path)
    expected = {label for _, label in METHODS}
    if set(curves) != expected:
        raise SystemExit(f"quick accuracy curves incomplete: {sorted(curves)}")

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for (_, label), color in zip(METHODS, COLORS):
        values = curves[label]
        axis.plot(range(1, len(values) + 1), values, label=label, color=color, linewidth=1.8)
    axis.set_xlabel("Global Round")
    axis.set_ylabel("Test Accuracy (%)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "quick_accuracy.pdf", bbox_inches="tight")
    fig.savefig(run_dir / "quick_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_resources(run_dir):
    with (run_dir / "metrics_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {_method_for_run(row["run"]): row for row in rows}
    labels = [label for _, label in METHODS]
    if set(by_method) != set(labels):
        raise SystemExit(f"quick resource summary incomplete: {sorted(by_method)}")

    panels = (
        ("time_s", "Time (s)"),
        ("comm_mb", "Communication (MB)"),
        ("comm_time_s", "Communication Time (s)"),
        ("energy_j", "Energy (J)"),
    )
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6))
    for axis, (field, title) in zip(axes.flat, panels):
        values = [float(by_method[label][field]) for label in labels]
        bars = axis.bar(x, values, color=COLORS)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    fig.tight_layout()
    fig.savefig(run_dir / "quick_resources.pdf", bbox_inches="tight")
    fig.savefig(run_dir / "quick_resources.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a completed RTDFL quick experiment.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    _plot_accuracy(run_dir)
    _plot_resources(run_dir)
    print(f"quick plots: {run_dir}")


if __name__ == "__main__":
    main()
