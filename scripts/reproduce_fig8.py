#!/usr/bin/env python3
"""Reproduce paper Figure 8 from the bundled physical-testbed traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = ROOT_DIR / "expected_results" / "source" / "fig8"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "reproduced"

METHOD_ORDER = ["dfl", "mmdfl", "rtdfl"]
STYLE = {
    "dfl": {"color": "#27AE60", "label": "DFL-Real", "marker": "D"},
    "mmdfl": {"color": "#2980B9", "label": "MMDFL-Real", "marker": "s"},
    "rtdfl": {"color": "#E67E22", "label": "RTDFL-Real (Ours)", "marker": "P"},
}

START_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*ROUND=(\d+) start"
)
END_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*"
    r"ROUND=(\d+) end avg_acc=([0-9.]+) best_acc=([0-9.]+) "
    r"duration=([0-9.]+)s comm_mb=([0-9.]+)"
)
RECV_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*"
    r"RECV round=(\d+).*cid=(\d+).*battery=([0-9.]+)J"
)

TIME_COLUMN = "\u63a5\u6536\u65f6\u95f4"
ACTIVE_POWER_COLUMN = "\u6709\u529f\u529f\u7387"


@dataclass
class RoundRecord:
    round_idx: int
    start_ts: datetime
    end_ts: datetime
    avg_acc: float
    best_acc: float
    duration_s: float
    comm_mb: float
    measured_round_energy_j: float = 0.0
    covered_power_streams: int = 0
    total_battery_remaining_j: float = 0.0
    min_battery_remaining_j: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="parse and validate all records without rendering figures",
    )
    return parser.parse_args()


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")


def load_metadata(source_dir: Path) -> dict:
    path = source_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 8 metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    methods = metadata.get("methods", {})
    if set(methods) != set(METHOD_ORDER):
        raise ValueError(f"metadata methods must be exactly {METHOD_ORDER}")
    return metadata


def verify_manifest(source_dir: Path) -> None:
    manifest_path = source_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Figure 8 manifest: {manifest_path}")
    checked = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_path = Path(row["path"])
            path = (source_dir / relative_path).resolve()
            if source_dir.resolve() not in path.parents:
                raise RuntimeError(f"Manifest path escapes source directory: {relative_path}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing Figure 8 source file: {path}")
            if path.stat().st_size != int(row["size_bytes"]):
                raise RuntimeError(f"Size mismatch for Figure 8 source file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for Figure 8 source file: {path}")
            checked += 1
    if checked != 15:
        raise RuntimeError(f"Figure 8 manifest must contain 15 source files; got {checked}")


def parse_log(path: Path, expected_rounds: int, initial_energy_j: float) -> list[RoundRecord]:
    starts_by_round: dict[int, list[datetime]] = {}
    ends_by_round: dict[int, RoundRecord] = {}
    recv_by_round: dict[int, list[tuple[int, float]]] = {}

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        start_match = START_RE.search(line)
        if start_match:
            starts_by_round.setdefault(int(start_match.group(2)), []).append(
                parse_dt(start_match.group(1))
            )
            continue

        recv_match = RECV_RE.search(line)
        if recv_match:
            recv_by_round.setdefault(int(recv_match.group(2)), []).append(
                (int(recv_match.group(3)), float(recv_match.group(4)))
            )
            continue

        end_match = END_RE.search(line)
        if not end_match:
            continue
        round_idx = int(end_match.group(2))
        end_ts = parse_dt(end_match.group(1))
        candidate_starts = [ts for ts in starts_by_round.get(round_idx, []) if ts <= end_ts]
        if not candidate_starts:
            raise RuntimeError(f"Missing round start: {path}, round={round_idx}")
        if round_idx not in ends_by_round:
            ends_by_round[round_idx] = RoundRecord(
                round_idx=round_idx,
                start_ts=max(candidate_starts),
                end_ts=end_ts,
                avg_acc=float(end_match.group(3)),
                best_acc=float(end_match.group(4)),
                duration_s=float(end_match.group(5)),
                comm_mb=float(end_match.group(6)),
            )

    records = [ends_by_round[idx] for idx in sorted(ends_by_round)]
    expected_ids = list(range(expected_rounds))
    actual_ids = [record.round_idx for record in records]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"{path} must contain complete rounds 0..{expected_rounds - 1}; got {actual_ids[:3]}...{actual_ids[-3:]}"
        )

    battery_state = {cid: initial_energy_j for cid in range(10)}
    for record in records:
        for cid, battery_j in recv_by_round.get(record.round_idx, []):
            battery_state[cid] = battery_j
        record.total_battery_remaining_j = sum(battery_state.values())
        record.min_battery_remaining_j = min(battery_state.values())
    return records


def read_power_csv(path: Path, start_date: date) -> list[tuple[datetime, float]]:
    rows: list[tuple[datetime, float]] = []
    current_date = start_date
    previous_time = None
    with path.open("r", encoding="gbk", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"Power CSV has no header: {path}")
        for row in reader:
            time_text = row.get(TIME_COLUMN) or row.get("\ufeff" + TIME_COLUMN)
            if not time_text:
                continue
            try:
                current_time = datetime.strptime(time_text.strip(), "%H:%M:%S").time()
                active_power_w = float(row[ACTIVE_POWER_COLUMN])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_time is not None and current_time < previous_time:
                current_date += timedelta(days=1)
            previous_time = current_time
            rows.append((datetime.combine(current_date, current_time), active_power_w))
    if len(rows) < 2:
        raise RuntimeError(f"Power CSV has fewer than two usable samples: {path}")
    return rows


def load_power_streams(source_dir: Path, method_config: dict) -> list[list[tuple[datetime, float]]]:
    streams: list[list[tuple[datetime, float]]] = []
    for session in method_config["power_sessions"]:
        session_dir = source_dir / session["directory"]
        paths = sorted(session_dir.glob("meter_*.csv"))
        expected_streams = int(method_config["expected_power_streams"])
        if len(paths) != expected_streams:
            raise RuntimeError(
                f"{session_dir} must contain {expected_streams} meter CSV files; got {len(paths)}"
            )
        start_date = date.fromisoformat(session["start_date"])
        streams.extend(read_power_csv(path, start_date) for path in paths)
    return streams


def integrate_power(
    rows: list[tuple[datetime, float]], start_ts: datetime, end_ts: datetime
) -> tuple[float, float]:
    energy_j = 0.0
    covered_s = 0.0
    for (left_ts, power_w), (right_ts, _) in zip(rows, rows[1:]):
        segment_start = max(start_ts, left_ts)
        segment_end = min(end_ts, right_ts)
        if segment_end <= segment_start:
            continue
        seconds = (segment_end - segment_start).total_seconds()
        energy_j += power_w * seconds
        covered_s += seconds
    return energy_j, covered_s


def attach_power(
    records: list[RoundRecord],
    streams: list[list[tuple[datetime, float]]],
    expected_streams: int,
    minimum_coverage: float,
) -> None:
    for record in records:
        expected_duration_s = max((record.end_ts - record.start_ts).total_seconds(), 0.0)
        for stream in streams:
            stream_energy_j, covered_s = integrate_power(stream, record.start_ts, record.end_ts)
            if covered_s >= expected_duration_s * minimum_coverage:
                record.covered_power_streams += 1
                record.measured_round_energy_j += stream_energy_j
        if record.covered_power_streams != expected_streams:
            raise RuntimeError(
                f"round={record.round_idx} has {record.covered_power_streams}/{expected_streams} "
                "covered power streams"
            )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        smoothed[idx] = float(np.mean(values[max(0, idx - window + 1) : idx + 1]))
    return smoothed


def write_metrics(
    output_path: Path, all_records: dict[str, list[RoundRecord]], initial_total_energy_kj: float
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "label",
        "round",
        "avg_acc",
        "best_acc",
        "round_duration_s",
        "cumulative_time_s",
        "round_energy_j",
        "cumulative_energy_j",
        "remaining_energy_kj_measured",
        "battery_remaining_kj_model",
        "min_battery_remaining_kj_model",
        "comm_mb",
        "covered_power_streams",
        "round_start",
        "round_end",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in METHOD_ORDER:
            cumulative_time_s = 0.0
            cumulative_energy_j = 0.0
            for record in all_records[method]:
                cumulative_time_s += record.duration_s
                cumulative_energy_j += record.measured_round_energy_j
                writer.writerow(
                    {
                        "method": method,
                        "label": STYLE[method]["label"],
                        "round": record.round_idx,
                        "avg_acc": record.avg_acc,
                        "best_acc": record.best_acc,
                        "round_duration_s": record.duration_s,
                        "cumulative_time_s": cumulative_time_s,
                        "round_energy_j": record.measured_round_energy_j,
                        "cumulative_energy_j": cumulative_energy_j,
                        "remaining_energy_kj_measured": initial_total_energy_kj
                        - cumulative_energy_j / 1000.0,
                        "battery_remaining_kj_model": record.total_battery_remaining_j / 1000.0,
                        "min_battery_remaining_kj_model": record.min_battery_remaining_j / 1000.0,
                        "comm_mb": record.comm_mb,
                        "covered_power_streams": record.covered_power_streams,
                        "round_start": record.start_ts.isoformat(sep=" "),
                        "round_end": record.end_ts.isoformat(sep=" "),
                    }
                )


def build_time_curve(
    records: list[RoundRecord],
    max_time_h: float,
    initial_accuracy: float,
    window: int,
    points: int,
) -> dict:
    elapsed_h = np.array([0.0] + list(np.cumsum([r.duration_s for r in records]) / 3600.0))
    accuracy = rolling_mean(np.array([initial_accuracy] + [r.avg_acc for r in records]), window)
    grid = np.linspace(0.0, max_time_h, points)
    values = np.interp(grid, elapsed_h, accuracy, left=np.nan, right=np.nan)
    valid = np.isfinite(values)
    return {"x": grid[valid], "y": values[valid]}


def build_energy_curve(
    records: list[RoundRecord],
    initial_energy_kj: float,
    minimum_energy_kj: float,
    initial_accuracy: float,
    window: int,
    points: int,
) -> dict:
    cumulative_j = np.cumsum([r.measured_round_energy_j for r in records])
    remaining = np.array([initial_energy_kj] + list(initial_energy_kj - cumulative_j / 1000.0))
    accuracy = rolling_mean(np.array([initial_accuracy] + [r.avg_acc for r in records]), window)
    order = np.argsort(remaining)
    unique_energy, unique_idx = np.unique(remaining[order], return_index=True)
    unique_accuracy = accuracy[order][unique_idx]
    grid = np.linspace(initial_energy_kj, minimum_energy_kj, points)
    values = np.interp(grid[::-1], unique_energy, unique_accuracy, left=np.nan, right=np.nan)[::-1]
    valid = np.isfinite(values)
    return {"x": grid[valid], "y": values[valid]}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.edgecolor": "black",
            "axes.linewidth": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )


def configure_axes(ax, xlabel: str, xlim: tuple[float, float], y_top: float) -> None:
    ax.set_xlabel(xlabel, fontsize=42, fontweight="bold", labelpad=15)
    ax.set_ylabel("Test Accuracy (%)", fontsize=42, fontweight="bold", labelpad=15)
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, y_top)
    ax.tick_params(axis="both", labelsize=38, width=2.5, length=10)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    ax.grid(True, linestyle="--", alpha=0.85, color="#4F4F4F", linewidth=1.25)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.5)


def add_legend(ax) -> None:
    legend = ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.55,
        edgecolor="black",
        fancybox=False,
        prop={"size": 30, "weight": "bold"},
    )
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_facecolor("white")


def render_curve(all_records: dict[str, list[RoundRecord]], metadata: dict, output_dir: Path, kind: str) -> None:
    plot = metadata["plot"]
    window = int(plot["smooth_window"])
    points = int(plot["interpolation_points"])
    y_max = max(record.avg_acc for records in all_records.values() for record in records)
    y_top = float(np.ceil((y_max + 5.0) / 5.0) * 5.0)

    fig, ax = plt.subplots(figsize=(10, 10))
    for method in METHOD_ORDER:
        style = STYLE[method]
        if kind == "energy":
            curve = build_energy_curve(
                all_records[method],
                float(metadata["initial_total_energy_kj"]),
                float(plot["minimum_remaining_energy_kj"]),
                float(plot["initial_accuracy_percent"]),
                window,
                points,
            )
        else:
            curve = build_time_curve(
                all_records[method],
                float(plot["maximum_time_hours"]),
                float(plot["initial_accuracy_percent"]),
                window,
                points,
            )
        ax.plot(
            curve["x"],
            curve["y"],
            label=style["label"],
            color=style["color"],
            linewidth=6,
            marker=style["marker"],
            markersize=14,
            markevery=max(len(curve["x"]) // 14, 1),
        )

    if kind == "energy":
        stem = "fig8a_accuracy_energy"
        xlim = (
            float(metadata["initial_total_energy_kj"]),
            float(plot["minimum_remaining_energy_kj"]),
        )
        xlabel = "Remaining Nominal Energy\nBudget (kJ)"
    else:
        stem = "fig8b_accuracy_time"
        xlim = (0.0, float(plot["maximum_time_hours"]))
        xlabel = "Cumulative Active\nRuntime (hours)"
    configure_axes(ax, xlabel, xlim, y_top)
    add_legend(ax)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.98)
    fig.savefig(
        output_dir / f"{stem}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    verify_manifest(source_dir)
    metadata = load_metadata(source_dir)
    expected_rounds = int(metadata["expected_rounds_per_method"])
    initial_device_energy_j = float(metadata["initial_device_energy_j"])
    minimum_coverage = float(metadata["minimum_power_coverage_fraction"])

    all_records: dict[str, list[RoundRecord]] = {}
    for method in METHOD_ORDER:
        config = metadata["methods"][method]
        records = parse_log(
            source_dir / config["server_log"], expected_rounds, initial_device_energy_j
        )
        attach_power(
            records,
            load_power_streams(source_dir, config),
            int(config["expected_power_streams"]),
            minimum_coverage,
        )
        all_records[method] = records

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "fig8_round_metrics.csv"
    write_metrics(metrics_path, all_records, float(metadata["initial_total_energy_kj"]))

    if not args.validate_only:
        configure_plot_style()
        render_curve(all_records, metadata, output_dir, "energy")
        render_curve(all_records, metadata, output_dir, "time")

    summary = {
        method: {
            "rounds": len(records),
            "cumulative_time_s": sum(record.duration_s for record in records),
            "cumulative_energy_j": sum(record.measured_round_energy_j for record in records),
            "covered_power_streams_per_round": sorted(
                {record.covered_power_streams for record in records}
            ),
        }
        for method, records in all_records.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"metrics={metrics_path}")
    if args.validate_only:
        print("Figure 8 source validation passed")
    else:
        print(f"figure8a={output_dir / 'fig8a_accuracy_energy.pdf'}")
        print(f"figure8b={output_dir / 'fig8b_accuracy_time.pdf'}")


if __name__ == "__main__":
    main()
