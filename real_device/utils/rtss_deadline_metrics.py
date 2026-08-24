import math


DEFAULT_SENSITIVITY_SECS = (120.0, 150.0, 180.0, 210.0)


def parse_deadline_sensitivity(value, primary_deadline_sec=0.0):
    if value is None:
        values = list(DEFAULT_SENSITIVITY_SECS)
    elif isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        values = [float(item.strip()) for item in str(value).split(",") if item.strip()]

    primary = float(primary_deadline_sec)
    if primary > 0.0:
        values.append(primary)
    return sorted(set(item for item in values if item > 0.0))


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(quantile), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rate(misses, total):
    return float(misses) / float(total) if total else 0.0


def _latency_distribution(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean_sec": sum(values) / len(values) if values else None,
        "p50_sec": percentile(values, 0.50),
        "p90_sec": percentile(values, 0.90),
        "p95_sec": percentile(values, 0.95),
        "max_sec": max(values) if values else None,
    }


def summarize_deadlines(records, primary_deadline_sec, sensitivity_secs):
    primary = float(primary_deadline_sec)
    thresholds = parse_deadline_sensitivity(sensitivity_secs, primary)
    task_latencies = []
    round_latencies = []
    controller_latencies = []
    dispatched_tasks = 0
    failed_tasks = 0

    for record in records:
        dispatched_tasks += len(record.get("send_records", []))
        failed_tasks += len(record.get("status_records", []))
        for train_record in record.get("train_records", []):
            latency = train_record.get("task_latency_sec")
            if latency is not None:
                task_latencies.append(float(latency))
        round_latency = record.get("round_critical_path_sec")
        if round_latency is not None:
            round_latencies.append(float(round_latency))
        for controller_record in record.get("controller_records", []):
            latency = controller_record.get("controller_duration_sec")
            if latency is not None:
                controller_latencies.append(float(latency))

    sensitivity = {}
    for threshold in thresholds:
        task_misses = sum(latency > threshold for latency in task_latencies)
        round_misses = sum(latency > threshold for latency in round_latencies)
        sensitivity[str(int(threshold) if threshold.is_integer() else threshold)] = {
            "deadline_sec": threshold,
            "task_misses": task_misses,
            "task_total": len(task_latencies),
            "task_miss_rate": _rate(task_misses, len(task_latencies)),
            "round_misses": round_misses,
            "round_total": len(round_latencies),
            "round_miss_rate": _rate(round_misses, len(round_latencies)),
        }

    primary_key = str(int(primary) if primary.is_integer() else primary) if primary > 0.0 else None
    primary_result = sensitivity.get(primary_key) if primary_key is not None else None
    return {
        "enabled": primary > 0.0,
        "mode": "observe_only",
        "primary_deadline_sec": primary,
        "primary": primary_result,
        "sensitivity": sensitivity,
        "dispatched_tasks": dispatched_tasks,
        "completed_tasks": len(task_latencies),
        "failed_tasks": failed_tasks,
        "task_latency": _latency_distribution(task_latencies),
        "round_critical_path": _latency_distribution(round_latencies),
        "controller_latency": _latency_distribution(controller_latencies),
        "controller_total_sec": sum(controller_latencies),
    }
