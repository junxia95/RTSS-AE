#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_METRICS = ("test_acc", "time", "comm", "comm_time", "energy")
PAPER_ALPHAS = ("0.5", "0.1", "0.05", "0.01")
PAPER_SEEDS = (1, 42, 1024)
METHOD_LABELS = {
    "AutoRL_DFL_MM": "RTDFL",
    "DFL_MM": "DFL-MM",
    "DFL": "DFL",
    "DFedPGP": "DFedPGP",
    "LD_SGD": "LD-SGD",
}
PROVENANCE_SOURCES = {
    "CIFAR-10": ("cifar10_baseline_runs.csv", "cifar10_rtdfl_mmdfl_runs.csv"),
    "CIFAR-100": ("cifar100_detail.csv",),
    "Tiny-ImageNet": ("tinyimagenet_runs.csv",),
}


def _series(value, metric):
    if isinstance(value, dict):
        if metric == "test_acc" and "acc" in value:
            value = value["acc"]
        elif metric in value:
            value = value[metric]
        else:
            raise ValueError(f"cannot locate {metric} series in dict")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{metric} must be a non-empty sequence")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{metric} contains a non-finite value")
    return values


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _table1_values(path):
    dataset = None
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\\multirow\{5\}\{\*\}\{([^}]+)\}", raw)
        if match:
            dataset = match.group(1)
            continue
        if dataset is None or "\\pm" not in raw or not raw.lstrip().startswith("&"):
            continue
        parts = [part.strip() for part in raw.split("&")]
        method = parts[1].replace("(ours)", "").strip()
        if method == "AutoRL":
            method = "RTDFL"
        for alpha, cell in zip(PAPER_ALPHAS, parts[2:6]):
            match = re.search(r"([0-9.]+)\\pm([0-9.]+)", cell)
            if match:
                values[(dataset, method, alpha)] = tuple(float(item) for item in match.groups())
    return values


def _validate_table1_provenance(errors):
    provenance_root = ROOT / "expected_results" / "provenance"
    observed = {}
    for dataset, names in PROVENANCE_SOURCES.items():
        for name in names:
            for row in _read_csv(provenance_root / name):
                algorithm = row.get("algorithm")
                alpha = row.get("beta")
                if algorithm not in METHOD_LABELS or alpha not in PAPER_ALPHAS:
                    continue
                key = (dataset, METHOD_LABELS[algorithm], alpha, int(row["seed"]))
                if key in observed:
                    errors.append(f"duplicate Table I provenance run: {key}")
                    continue
                if int(row["rounds"]) != 200:
                    errors.append(f"Table I run does not contain 200 rounds: {key}")
                observed[key] = float(row["best_acc"])

    paper_values = _table1_values(ROOT / "expected_results" / "source" / "table1.tex")
    expected_group_count = len(PROVENANCE_SOURCES) * len(METHOD_LABELS) * len(PAPER_ALPHAS)
    if len(paper_values) != expected_group_count:
        errors.append(f"Table I expected {expected_group_count} cells, found {len(paper_values)}")

    for group, expected in paper_values.items():
        samples = []
        for seed in PAPER_SEEDS:
            key = (*group, seed)
            if key not in observed:
                errors.append(f"missing Table I provenance run: {key}")
            else:
                samples.append(observed[key])
        if len(samples) != len(PAPER_SEEDS):
            continue
        actual = (statistics.mean(samples), statistics.pstdev(samples))
        if any(abs(left - right) > 0.0051 for left, right in zip(actual, expected)):
            errors.append(
                f"Table I mismatch for {group}: raw={actual[0]:.4f}+/-{actual[1]:.4f}, "
                f"paper={expected[0]:.2f}+/-{expected[1]:.2f}"
            )


def _validate_full_config(errors):
    path = ROOT / "configs" / "paper_full.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    matrix = config["matrix"]
    dataset_names = [item["name"] for item in matrix["datasets"]]
    checks = {
        "datasets": (dataset_names, ["cifar10", "cifar100", "TinyImagenet"]),
        "alphas": ([str(item) for item in matrix["alphas"]], list(PAPER_ALPHAS)),
        "seeds": (matrix["seeds"], list(PAPER_SEEDS)),
        "algorithms": (matrix["algorithms"], list(METHOD_LABELS)),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"full config {label} mismatch: {actual} != {expected}")
    defaults = config["defaults"]
    expected_defaults = {
        "model": "resnet8",
        "num_users": 10,
        "frac": 0.2,
        "epochs": 200,
        "local_ep": 5,
        "local_bs": 128,
        "lr": 0.01,
        "lr_decay": 0.998,
        "noniid_case": 5,
    }
    for key, expected in expected_defaults.items():
        if defaults.get(key) != expected:
            errors.append(f"full config {key} mismatch: {defaults.get(key)} != {expected}")
    job_count = len(dataset_names) * len(matrix["alphas"]) * len(matrix["seeds"]) * len(matrix["algorithms"])
    if job_count != 180:
        errors.append(f"full config expected 180 jobs, found {job_count}")


def _validate_table2(errors):
    source = ROOT / "expected_results" / "source" / "table2.tex"
    methods = ("RTDFL", "DFL-MM", "DFL", "DFedPGP", "LD-SGD")
    algorithms = {label: algorithm for algorithm, label in METHOD_LABELS.items()}
    paper_values = {}
    paper_targets = []
    alpha = None
    for raw in source.read_text(encoding="utf-8").splitlines():
        match = re.search(r"\\alpha=([0-9.]+)", raw)
        if match:
            alpha = match.group(1)
            continue
        if alpha in PAPER_ALPHAS and raw.lstrip().startswith("&"):
            parts = [part.strip() for part in raw.split("&")]
            if len(parts) >= 12 and re.search(r"\d", parts[1]):
                target = re.sub(r"[^0-9.]", "", parts[1])
                paper_targets.append((alpha, target))
                cells = parts[2:12]
                for method, time_cell, comm_cell in zip(methods, cells[::2], cells[1::2]):
                    time_match = re.search(r"([0-9]+)s", time_cell)
                    comm_match = re.search(r"([0-9]+)M", comm_cell)
                    paper_values[(alpha, int(float(target)), method)] = (
                        int(time_match.group(1)) if time_match else None,
                        int(comm_match.group(1)) if comm_match else None,
                    )
    targets = {
        "0.5": ("18", "28"),
        "0.1": ("10", "20"),
        "0.05": ("8", "16"),
        "0.01": ("6", "8"),
    }
    for alpha_value, expected in targets.items():
        actual = tuple(target for item_alpha, target in paper_targets if item_alpha == alpha_value)
        if actual != expected:
            errors.append(f"Table II targets mismatch for alpha={alpha_value}: {actual} != {expected}")

    raw_root = ROOT / "expected_results" / "raw" / "tinyimagenet_round_metrics"
    manifest_path = raw_root / "manifest.csv"
    manifest = _read_csv(manifest_path)
    if len(manifest) != 180:
        errors.append(f"Table II expected 180 raw metric files, found {len(manifest)}")
    for row in manifest:
        path = raw_root / f"alpha{row['beta']}" / f"seed{row['seed']}" / row["algorithm"] / f"{row['metric']}.pkl"
        if not path.exists():
            errors.append(f"missing Table II raw metric: {path}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != row["sha256"]:
            errors.append(f"Table II raw metric checksum mismatch: {path}")

    for (alpha_value, target, method), expected in paper_values.items():
        hits = []
        for seed in PAPER_SEEDS:
            run_dir = raw_root / f"alpha{alpha_value}" / f"seed{seed}" / algorithms[method]
            metrics = {}
            for metric in ("test_acc", "time", "comm"):
                path = run_dir / f"{metric}.pkl"
                if not path.exists():
                    continue
                with path.open("rb") as handle:
                    metrics[metric] = _series(pickle.load(handle), metric)
            if len(metrics) != 3:
                continue
            if any(len(values) != 200 for values in metrics.values()):
                errors.append(f"Table II raw run does not contain 200 rounds: {run_dir}")
                continue
            hit = next((index for index, value in enumerate(metrics["test_acc"]) if value >= target), None)
            if hit is not None:
                hits.append((metrics["time"][hit], metrics["comm"][hit]))
        actual = (None, None)
        if len(hits) == len(PAPER_SEEDS):
            actual = (
                round(statistics.mean(value[0] for value in hits)),
                round(statistics.mean(value[1] for value in hits)),
            )
        if actual != expected:
            errors.append(
                f"Table II mismatch for alpha={alpha_value}, target={target}, method={method}: "
                f"raw={actual}, paper={expected}"
            )


def _validate_fig9(errors):
    rows = _read_csv(ROOT / "expected_results" / "source" / "fig9_scalability.csv")
    table1 = _table1_values(ROOT / "expected_results" / "source" / "table1.tex")
    dataset_names = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "TinyImagenet": "Tiny-ImageNet"}
    for row in rows:
        clients = int(row["clients"])
        method = METHOD_LABELS[row["method"]]
        value = float(row["best_acc"])
        if clients == 10:
            expected = table1[(dataset_names[row["dataset"]], method, "0.1")][0]
            if abs(value - expected) > 0.0051:
                errors.append(f"Fig. 9 10-client value mismatch: {row}")

    scaling_rows = _read_csv(ROOT / "expected_results" / "provenance" / "scalability_seed1_detail.csv")
    scaling = {
        (int(row["num_users"]), row["dataset"], row["method"]): float(row["best_acc"])
        for row in scaling_rows
    }
    for row in rows:
        clients = int(row["clients"])
        if clients == 10:
            continue
        key = (clients, row["dataset"], row["method"])
        if key not in scaling or abs(float(row["best_acc"]) - scaling[key]) > 1e-9:
            errors.append(f"Fig. 9 scalability provenance mismatch: {key}")


def _validate_expected():
    source = ROOT / "expected_results" / "source"
    required = {
        "table1.tex": 100,
        "table2.tex": 100,
        "fig9_scalability.csv": 40,
    }
    errors = []
    for name, min_size in required.items():
        path = source / name
        if not path.exists() or path.stat().st_size < min_size:
            errors.append(f"missing or incomplete expected source: {path}")
    provenance_path = ROOT / "expected_results" / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for relative_path, metadata in provenance.get("files", {}).items():
            path = ROOT / "expected_results" / relative_path
            expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
            if path.exists() and expected_hash:
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    errors.append(f"checksum mismatch: {path}")
    fig9 = source / "fig9_scalability.csv"
    if fig9.exists():
        with fig9.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 45:
            errors.append(f"Fig. 9 expected 45 rows, found {len(rows)}")
        for clients in (10, 30, 60):
            for dataset in ("cifar10", "cifar100", "TinyImagenet"):
                subset = [row for row in rows if int(row["clients"]) == clients and row["dataset"] == dataset]
                if len(subset) != 5:
                    errors.append(f"Fig. 9 incomplete group: clients={clients}, dataset={dataset}")
                    continue
                best = max(subset, key=lambda row: float(row["best_acc"]))
                if best["method"] != "AutoRL_DFL_MM":
                    errors.append(f"Fig. 9 trend mismatch: clients={clients}, dataset={dataset}")
    _validate_table1_provenance(errors)
    _validate_table2(errors)
    _validate_fig9(errors)
    _validate_full_config(errors)
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({
        "expected_results": "valid",
        "source": str(source),
        "table1_run_records": 180,
        "table2_raw_metric_files": 180,
        "fig9_source_rows": 45,
        "full_jobs": 180,
    }, indent=2))


def _validate_run(run_dir):
    run_dir = run_dir.resolve()
    run_dirs = sorted(path.parent for path in run_dir.rglob("test_acc.pkl"))
    if not run_dirs:
        raise SystemExit(f"no completed result directories found under {run_dir}")
    rows = []
    series_by_run = {}
    for result_dir in run_dirs:
        metrics = {}
        for metric in REQUIRED_METRICS:
            path = result_dir / f"{metric}.pkl"
            if not path.exists():
                raise SystemExit(f"missing metric: {path}")
            with path.open("rb") as handle:
                metrics[metric] = _series(pickle.load(handle), metric)
        lengths = {metric: len(values) for metric, values in metrics.items()}
        if len(set(lengths.values())) != 1:
            raise SystemExit(f"metric length mismatch in {result_dir}: {lengths}")
        for metric in ("time", "comm", "comm_time", "energy"):
            values = metrics[metric]
            if any(current + 1e-9 < previous for previous, current in zip(values, values[1:])):
                raise SystemExit(f"{metric} is not cumulative in {result_dir}")
        series_by_run[result_dir.name] = metrics
        rows.append({
            "run": result_dir.name,
            "rounds": len(metrics["test_acc"]),
            "best_acc": max(metrics["test_acc"]),
            "last_acc": metrics["test_acc"][-1],
            "time_s": metrics["time"][-1],
            "comm_mb": metrics["comm"][-1],
            "comm_time_s": metrics["comm_time"][-1],
            "energy_j": metrics["energy"][-1],
        })
    output = run_dir / "metrics_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"run_results": "valid", "runs": rows, "summary": str(output)}, indent=2))
    return rows, series_by_run


def _validate_quick_profile(run_dir, rows, series_by_run):
    config_path = ROOT / "configs" / "quick_cifar10.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_rounds = int(config["defaults"]["epochs"])
    expected_methods = {
        "AutoRL_DFL_MM": "cifar10-alpha0-1-seed1-autorl-dfl-mm",
        "DFL_MM": "cifar10-alpha0-1-seed1-dfl-mm",
        "DFL": "cifar10-alpha0-1-seed1-dfl",
        "DFedPGP": "cifar10-alpha0-1-seed1-dfedpgp",
        "LD_SGD": "cifar10-alpha0-1-seed1-ld-sgd",
    }
    checks = []

    def record(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    observed_runs = set(series_by_run)
    expected_runs = set(expected_methods.values())
    record(
        "all_methods_completed",
        observed_runs == expected_runs,
        {"expected": sorted(expected_runs), "observed": sorted(observed_runs)},
    )

    lengths = {name: len(metrics["test_acc"]) for name, metrics in series_by_run.items()}
    record(
        "round_count",
        observed_runs == expected_runs and all(value == expected_rounds for value in lengths.values()),
        {"expected": expected_rounds, "observed": lengths},
    )

    partition_tag = f"{config['partition_prefix']}1"
    partition_path = ROOT / "data" / f"cifar10_10_noniidCase5_beta0.1_{partition_tag}.json"
    partition_ok = False
    partition_detail = {"path": str(partition_path), "assigned": 0, "unique": 0}
    if partition_path.exists():
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        assignments = [
            int(index)
            for values in partition.get("train_data", {}).values()
            for index in values
        ]
        partition_detail.update({"assigned": len(assignments), "unique": len(set(assignments))})
        partition_ok = (
            len(assignments) == 50000
            and len(set(assignments)) == 50000
            and set(assignments) == set(range(50000))
        )
    record("full_cifar10_partition", partition_ok, partition_detail)

    if observed_runs == expected_runs:
        metrics_by_method = {
            method: series_by_run[run_name]
            for method, run_name in expected_methods.items()
        }
        best_acc = {
            method: max(metrics["test_acc"])
            for method, metrics in metrics_by_method.items()
        }
        rtdfl_best = best_acc["AutoRL_DFL_MM"]
        best_overall = max(best_acc.values())
        rtdfl_comm = metrics_by_method["AutoRL_DFL_MM"]["comm"][-1]
        mmdfl_comm = metrics_by_method["DFL_MM"]["comm"][-1]
        comm_ratio = rtdfl_comm / mmdfl_comm if mmdfl_comm > 0 else math.inf
        record("rtdfl_learns", rtdfl_best >= 20.0, {"best_acc": rtdfl_best, "minimum": 20.0})
        record(
            "rtdfl_accuracy_competitive",
            rtdfl_best + 2.0 >= best_overall,
            {"rtdfl_best_acc": rtdfl_best, "best_overall": best_overall, "tolerance_pp": 2.0},
        )
        record(
            "rtdfl_communication_reduction",
            comm_ratio <= 0.70,
            {"rtdfl_comm_mb": rtdfl_comm, "mmdfl_comm_mb": mmdfl_comm, "ratio": comm_ratio},
        )
    else:
        record("rtdfl_learns", False, "cannot evaluate with missing methods")
        record("rtdfl_accuracy_competitive", False, "cannot evaluate with missing methods")
        record("rtdfl_communication_reduction", False, "cannot evaluate with missing methods")

    report = {
        "profile": "quick-cifar10-alpha0.1-seed1",
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "config": str(config_path),
        "run_dir": str(run_dir),
        "checks": checks,
        "metrics": rows,
    }
    output = run_dir / "quick_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise SystemExit(
            "quick trend validation failed:\n"
            + "\n".join(f"- {item['name']}: {item['detail']}" for item in failed)
        )
    print(json.dumps({"quick_profile": "valid", "report": str(output)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Validate expected or newly generated RTDFL results.")
    parser.add_argument("--expected-only", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--quick-profile", action="store_true")
    args = parser.parse_args()
    if args.expected_only:
        _validate_expected()
    if args.run_dir:
        rows, series_by_run = _validate_run(args.run_dir)
        if args.quick_profile:
            _validate_quick_profile(args.run_dir.resolve(), rows, series_by_run)
    elif args.quick_profile:
        parser.error("--quick-profile requires --run-dir")
    if not args.expected_only and args.run_dir is None:
        parser.error("specify --expected-only or --run-dir")


if __name__ == "__main__":
    main()
