#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import pickle
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = ROOT / "expected_results" / "source"
OUTPUT = ROOT / "results" / "reproduced"
PAPER_ALPHAS = ("0.5", "0.1", "0.05", "0.01")
PAPER_SEEDS = (1, 42, 1024)
METHOD_LABELS = {
    "AutoRL_DFL_MM": "RTDFL",
    "DFL_MM": "MMDFL",
    "DFL": "DFL",
    "DFedPGP": "DFedPGP",
    "LD_SGD": "LD-SGD",
}
PROVENANCE_SOURCES = {
    "CIFAR-10": ("cifar10_baseline_runs.csv", "cifar10_rtdfl_mmdfl_runs.csv"),
    "CIFAR-100": ("cifar100_detail.csv",),
    "Tiny-ImageNet": ("tinyimagenet_runs.csv",),
}
TABLE2_TARGETS = {
    "0.5": (18, 28),
    "0.1": (10, 20),
    "0.05": (8, 16),
    "0.01": (6, 8),
}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_table1(output_csv, output_tex):
    provenance_root = ROOT / "expected_results" / "provenance"
    samples = {}
    for dataset, names in PROVENANCE_SOURCES.items():
        for name in names:
            for row in _read_csv(provenance_root / name):
                algorithm = row.get("algorithm")
                alpha = row.get("beta")
                if algorithm not in METHOD_LABELS or alpha not in PAPER_ALPHAS:
                    continue
                key = (dataset, algorithm, alpha)
                samples.setdefault(key, {})[int(row["seed"])] = float(row["best_acc"])

    rows = []
    for dataset in PROVENANCE_SOURCES:
        for algorithm, method in METHOD_LABELS.items():
            for alpha in PAPER_ALPHAS:
                values_by_seed = samples.get((dataset, algorithm, alpha), {})
                missing = set(PAPER_SEEDS) - set(values_by_seed)
                if missing:
                    raise ValueError(
                        f"missing Table I provenance for {dataset}/{algorithm}/alpha={alpha}: {sorted(missing)}"
                    )
                values = [values_by_seed[seed] for seed in PAPER_SEEDS]
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "alpha": alpha,
                    "mean_best_acc": f"{statistics.mean(values):.2f}",
                    "std_best_acc": f"{statistics.pstdev(values):.2f}",
                    "seeds": ",".join(str(seed) for seed in PAPER_SEEDS),
                })

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(row["dataset"], row["method"], row["alpha"]): row for row in rows}
    tex = [
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"Dataset & Method & $\alpha=0.5$ & $\alpha=0.1$ & $\alpha=0.05$ & $\alpha=0.01$ \\",
        r"\hline",
    ]
    for dataset in PROVENANCE_SOURCES:
        for method in METHOD_LABELS.values():
            cells = []
            for alpha in PAPER_ALPHAS:
                row = lookup[(dataset, method, alpha)]
                cells.append(f"${row['mean_best_acc']} \\pm {row['std_best_acc']}$")
            tex.append(f"{dataset} & {method} & " + " & ".join(cells) + r" \\")
        tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    output_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _metric_series(path, metric):
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, dict):
        if metric == "test_acc" and "acc" in value:
            value = value["acc"]
        else:
            value = value[metric]
    return [float(item) for item in value]


def _write_table2(output_csv, output_tex):
    raw_root = ROOT / "expected_results" / "raw" / "tinyimagenet_round_metrics"
    rows = []
    for alpha in PAPER_ALPHAS:
        for target in TABLE2_TARGETS[alpha]:
            for algorithm, method in METHOD_LABELS.items():
                hits = []
                for seed in PAPER_SEEDS:
                    run_dir = raw_root / f"alpha{alpha}" / f"seed{seed}" / algorithm
                    accuracy = _metric_series(run_dir / "test_acc.pkl", "test_acc")
                    modeled_time = _metric_series(run_dir / "time.pkl", "time")
                    communication = _metric_series(run_dir / "comm.pkl", "comm")
                    if not (len(accuracy) == len(modeled_time) == len(communication) == 200):
                        raise ValueError(f"Table II raw series must contain 200 rounds: {run_dir}")
                    first_hit = next((index for index, value in enumerate(accuracy) if value >= target), None)
                    if first_hit is not None:
                        hits.append((modeled_time[first_hit], communication[first_hit]))
                if len(hits) == len(PAPER_SEEDS):
                    time_s = str(round(statistics.mean(item[0] for item in hits)))
                    comm_mib = str(round(statistics.mean(item[1] for item in hits)))
                else:
                    time_s = "N/A"
                    comm_mib = "N/A"
                rows.append({
                    "alpha": alpha,
                    "target_acc": target,
                    "method": method,
                    "seeds_reached": len(hits),
                    "time_s": time_s,
                    "comm_mib": comm_mib,
                })

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tex = [
        r"\begin{tabular}{cclrrr}",
        r"\hline",
        r"$\alpha$ & Target (\%) & Method & Seeds reached & Modeled time (s) & Comm. (MiB) \\",
        r"\hline",
    ]
    for row in rows:
        tex.append(
            f"{row['alpha']} & {row['target_acc']} & {row['method']} & "
            f"{row['seeds_reached']} & {row['time_s']} & {row['comm_mib']} " + r"\\"
        )
    tex.extend((r"\hline", r"\end{tabular}"))
    output_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")


def _plot_fig9(source, output_pdf, output_png):
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    client_values = (10, 30, 60)
    methods = ("AutoRL_DFL_MM", "DFL_MM", "DFL", "DFedPGP", "LD_SGD")
    method_labels = ("RTDFL\n(ours)", "MMDFL", "DFL", "DFedPGP", "LD-SGD")
    datasets = ("cifar10", "cifar100", "TinyImagenet")
    dataset_labels = ("CIFAR-10", "CIFAR-100", "Tiny-ImageNet")
    colors = ("#243b64", "#c95652", "#8060a8")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
    width = 0.22
    x = np.arange(len(methods))
    for axis, clients in zip(axes, client_values):
        panel_bars = []
        for offset, (dataset, label, color) in enumerate(zip(datasets, dataset_labels, colors)):
            values = []
            for method in methods:
                row = next(
                    item for item in rows
                    if int(item["clients"]) == clients and item["dataset"] == dataset and item["method"] == method
                )
                values.append(float(row["best_acc"]))
            bars = axis.bar(x + (offset - 1) * width, values, width, label=label, color=color)
            panel_bars.append((bars, values))
        for method_index in range(len(methods)):
            labels_to_place = []
            for bars, values in panel_bars:
                labels_to_place.append([bars[method_index], values[method_index], values[method_index] + 0.5])
            labels_to_place.sort(key=lambda item: item[2])
            for label_index in range(1, len(labels_to_place)):
                labels_to_place[label_index][2] = max(
                    labels_to_place[label_index][2],
                    labels_to_place[label_index - 1][2] + 1.7,
                )
            for bar, value, label_y in labels_to_place:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        axis.set_title(f"{clients} Clients", fontsize=12, fontweight="bold")
        axis.set_xticks(x, method_labels, fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 50)
    axes[0].set_ylabel("Best Test Accuracy (%)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Recreate currently available paper result artifacts.")
    parser.add_argument("--only", choices=("all", "table1", "table2", "fig9"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "table1"):
        _write_table1(OUTPUT / "table1.csv", OUTPUT / "table1.tex")
    if args.only in ("all", "table2"):
        _write_table2(OUTPUT / "table2.csv", OUTPUT / "table2.tex")
    if args.only in ("all", "fig9"):
        _plot_fig9(SOURCE / "fig9_scalability.csv", OUTPUT / "fig9_scalability.pdf", OUTPUT / "fig9_scalability.png")
    print(f"reproduced artifacts: {OUTPUT}")


if __name__ == "__main__":
    main()
