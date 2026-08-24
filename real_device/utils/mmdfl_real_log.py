import json
import time
from pathlib import Path


TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def now_str():
    return time.strftime(TIME_FORMAT)


def make_log_dir(root, prefix):
    log_dir = Path(root) / prefix
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
