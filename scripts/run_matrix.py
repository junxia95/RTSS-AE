#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run_text(command):
    try:
        return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_metadata():
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run_text(["git", "status", "--porcelain"]) not in ("", "unavailable")),
    }


def _gpu_metadata():
    if shutil.which("nvidia-smi") is None:
        return "unavailable"
    return _run_text([
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader",
    ])


def _safe_slug(value):
    text = str(value).strip().lower().replace("_", "-")
    return "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in text).strip("-")


def _partition_path(dataset, num_users, alpha, tag):
    return ROOT / "data" / f"{dataset}_{num_users}_noniidCase5_beta{alpha}_{tag}.json"


def _command_args(values):
    result = []
    for key, value in values.items():
        if value is None:
            continue
        result.extend([f"--{key}", str(value)])
    return result


def _load_config(path):
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("defaults", "matrix"):
        if key not in config:
            raise ValueError(f"missing config key: {key}")
    return config


def _expand(config):
    matrix = config["matrix"]
    for dataset in matrix["datasets"]:
        for alpha in matrix["alphas"]:
            for seed in matrix["seeds"]:
                for algorithm in matrix["algorithms"]:
                    yield dataset, alpha, seed, algorithm


def _parse_csv(value, cast=str):
    if value is None:
        return None
    parsed = {cast(item.strip()) for item in value.split(",") if item.strip()}
    if not parsed:
        raise ValueError("matrix filter cannot be empty")
    return parsed


def _filter_jobs(jobs, datasets=None, alphas=None, seeds=None, algorithms=None):
    filtered = []
    for dataset, alpha, seed, algorithm in jobs:
        if datasets is not None and dataset["name"] not in datasets:
            continue
        if alphas is not None and float(alpha) not in alphas:
            continue
        if seeds is not None and int(seed) not in seeds:
            continue
        if algorithms is not None and algorithm not in algorithms:
            continue
        filtered.append((dataset, alpha, seed, algorithm))
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Run an RTDFL experiment matrix.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--gpu", default="0", help="Physical GPU index, or cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--datasets", help="Comma-separated dataset names to run")
    parser.add_argument("--alphas", help="Comma-separated Dirichlet alpha values to run")
    parser.add_argument("--seeds", help="Comma-separated seeds to run")
    parser.add_argument("--algorithms", help="Comma-separated algorithm names to run")
    parser.add_argument("--run-label", help="Label appended to this matrix run")
    args = parser.parse_args()

    config = _load_config(args.config.resolve())
    timestamp = dt.datetime.now().strftime("%H%M%S")
    date = dt.date.today().isoformat()
    slug = _safe_slug(config.get("name", args.config.stem))
    if args.run_label:
        slug = f"{slug}-{_safe_slug(args.run_label)}"
    run_id = f"{timestamp}__{slug}"
    log_dir = ROOT / "logs" / date / run_id
    result_root = ROOT / "results" / args.mode / f"{date}__{run_id}"
    log_dir.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)

    base_manifest = {
        "name": config.get("name"),
        "mode": args.mode,
        "started_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "config": str(args.config.resolve()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": _run_text([args.python, "--version"]),
        "torch": _run_text([args.python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]),
        "gpu": _gpu_metadata(),
        **_git_metadata(),
        "log_dir": str(log_dir),
        "result_root": str(result_root),
    }
    (log_dir / "manifest.json").write_text(json.dumps(base_manifest, indent=2) + "\n", encoding="utf-8")
    index_path = ROOT / "logs" / date / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as index_file:
        index_file.write(json.dumps(base_manifest, sort_keys=True) + "\n")

    data_root = os.environ.get("RTDFL_DATA_ROOT", str(ROOT / "data" / "raw"))
    env = os.environ.copy()
    physical_gpu = str(args.gpu)
    if physical_gpu.lower() == "cpu":
        logical_gpu = -1
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        logical_gpu = 0
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu

    jobs = _filter_jobs(
        list(_expand(config)),
        datasets=_parse_csv(args.datasets),
        alphas=_parse_csv(args.alphas, float),
        seeds=_parse_csv(args.seeds, int),
        algorithms=_parse_csv(args.algorithms),
    )
    if not jobs:
        raise ValueError("matrix filters selected no jobs")
    commands_path = log_dir / "commands.jsonl"
    failures = []
    for job_index, (dataset, alpha, seed, algorithm) in enumerate(jobs, start=1):
        dataset_name = dataset["name"]
        tag_prefix = str(config.get("partition_prefix", "seed"))
        partition_tag = f"{tag_prefix}{seed}"
        run_name = _safe_slug(f"{dataset_name}-alpha{alpha}-seed{seed}-{algorithm}")
        result_dir = result_root / run_name
        result_dir.mkdir(parents=True, exist_ok=False)

        values = dict(config["defaults"])
        values.update(dataset.get("args", {}))
        values.update(config.get("algorithm_args", {}).get(algorithm, {}))
        if algorithm == "AutoRL_DFL_MM":
            values.update(config.get("autorl_args", {}))
        values.update({
            "algorithm": algorithm,
            "dataset": dataset_name,
            "num_classes": dataset["num_classes"],
            "data_beta": alpha,
            "seed": seed,
            "partition_tag": partition_tag,
            "experiment_tag": run_name,
            "result_dir": str(result_dir),
            "data_root": data_root,
            "gpu": logical_gpu,
        })
        partition = _partition_path(dataset_name, int(values["num_users"]), alpha, partition_tag)
        values["generate_data"] = 0 if partition.exists() else 1

        command = [args.python, str(ROOT / "main_fed.py"), *_command_args(values)]
        record = {
            "job": job_index,
            "total_jobs": len(jobs),
            "run_name": run_name,
            "command": command,
            "command_shell": shlex.join(command),
            "result_dir": str(result_dir),
            "partition": str(partition),
        }
        with commands_path.open("a", encoding="utf-8") as commands_file:
            commands_file.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"[{job_index}/{len(jobs)}] {record['command_shell']}", flush=True)
        if args.dry_run:
            continue

        (result_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        with (log_dir / f"{run_name}.log").open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if process.returncode != 0:
            failures.append({"run_name": run_name, "returncode": process.returncode})
            if not args.continue_on_error:
                break

    summary = {
        "completed_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "jobs_planned": len(jobs),
        "failures": failures,
        "dry_run": args.dry_run,
    }
    (log_dir / "status.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
