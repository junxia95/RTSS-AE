import pickle
from pathlib import Path


REAL_DEFAULTS = {
    "dataset": "cifar10",
    "model": "resnet8",
    "num_classes": 10,
    "num_channels": 3,
    "num_users": 10,
    "frac": 0.2,
    "local_ep": 5,
    "local_bs": 32,
    "bs": 128,
    "iid": 0,
    "noniid_case": 5,
    "data_beta": 0.1,
    "generate_data": 1,
}


def apply_real_defaults(args, algorithm):
    for key, value in REAL_DEFAULTS.items():
        setattr(args, key, value)
    args.algorithm = str(algorithm)
    return args


def run_tag(args, algorithm):
    tag = str(getattr(args, "log_tag", "") or "").strip()
    if not tag:
        tag = f"seed{int(args.seed)}"
    safe_tag = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in tag)
    safe_algorithm = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(algorithm)
    ).lower()
    return f"{safe_algorithm}_{safe_tag}"


def checkpoint_path(root, algorithm, args, filename="checkpoint.pt"):
    return Path(root) / "checkpoints" / run_tag(args, algorithm) / filename


def pickle_size_mb(payload):
    return len(pickle.dumps(payload)) / (1024.0 * 1024.0)
