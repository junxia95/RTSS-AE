#!/usr/bin/env python3
import json
import platform
import shutil
import subprocess

import matplotlib
import numpy
from PIL import Image
import torch
import torchvision


def main():
    gpu = "unavailable"
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            pass
    print(json.dumps({
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": Image.__version__,
        "gpu": gpu,
    }, indent=2))


if __name__ == "__main__":
    main()
