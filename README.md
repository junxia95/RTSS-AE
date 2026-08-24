# RTDFL

Code and environment release for **Real-Time Energy-Efficient Decentralized
Federated Learning via Self-Improving RL-based Dynamic Quantization**.

Repository: <https://github.com/junxia95/RTSS-AE>

## Scope

This public repository contains:

- the RTDFL implementation and four comparison methods;
- quick and full experiment configurations;
- environment, Docker, validation, and plotting code;
- synthetic unit tests; and
- the optional heterogeneous Jetson deployment code.

It intentionally does **not** contain the paper, datasets, raw physical traces,
submitted result files, generated figures, logs, or AE documents. Those
materials are distributed separately through the RTSS artifact submission.

## System Requirements

The reference platform is Ubuntu 20.04 on x86-64 with Bash, GNU Make, GNU
findutils, Python 3.9.19, PyTorch 2.1.0, torchvision 0.16.0, and CUDA 12.1.
Exact Python packages are pinned in `environment.yml`.

- Environment checks and unit tests run on CPU after the dependencies are
  installed.
- Training requires a CUDA-capable NVIDIA GPU with at least 8 GB memory.
- We recommend 16 GB host RAM and 20 GB free disk space.
- GPU Docker runs require an NVIDIA driver and NVIDIA Container Toolkit.

## Installation

### Conda

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate rtdfl-ae
make smoke
```

`make smoke` checks the environment and runs five synthetic unit tests. If an
RTSS evidence bundle is placed in `expected_results/`, it additionally
validates that bundle; otherwise the evidence check is skipped.

### Docker

```bash
docker build -t rtdfl-ae .
docker run --rm rtdfl-ae make smoke
```

For GPU training, mount the mutable data and output directories:

```bash
mkdir -p data results logs tmp topo
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/RTDFL/data" \
  -v "$PWD/results:/workspace/RTDFL/results" \
  -v "$PWD/logs:/workspace/RTDFL/logs" \
  -v "$PWD/tmp:/workspace/RTDFL/tmp" \
  -v "$PWD/topo:/workspace/RTDFL/topo" \
  rtdfl-ae make quick DATA_ROOT=/workspace/RTDFL/data/raw GPU=0
```

## Data

CIFAR-10 and CIFAR-100 are downloaded automatically by torchvision. To reuse
an existing data directory, pass `DATA_ROOT=/absolute/path/to/data`.

Tiny-ImageNet-200 is not redistributed. Place it below the selected data root:

```text
tiny-imagenet-200/
  train/
  val/images/
  val/val_annotations.txt
  wnids.txt
  words.txt
```

## Running the Code

### Reduced End-to-End Profile

```bash
make quick DATA_ROOT=/absolute/path/to/data GPU=0
```

This profile trains RTDFL, MMDFL, DFL, DFedPGP, and LD-SGD for 20 rounds on
CIFAR-10 with 10 clients, Dirichlet alpha 0.1, and seed 1. Outputs are written
under `results/quick/` and execution records under `logs/`.

For direct script execution:

```bash
RTDFL_DATA_ROOT=/absolute/path/to/data GPU=0 \
  bash scripts/reproduce_quick.sh
```

### Full Experiment Matrix

Inspect the 180-job matrix without training:

```bash
make full-dry-run DATA_ROOT=/absolute/path/to/data GPU=0
```

Run it with:

```bash
make full DATA_ROOT=/absolute/path/to/data GPU=0
```

The executable specification is `configs/paper_full.json`: three datasets,
four Dirichlet alpha values, three seeds, five algorithms, and 200 rounds per
job. A sequential run is a multi-day GPU workload.

Topology and network cache names are not seed-qualified. For consistent
sharding, use one fresh workspace and share its `topo/` directory across every
job.

### Paper-Result Processing Code

The following targets are included as code but require the separately
distributed `expected_results/` evidence directory:

```bash
make table1
make table2
make fig8
make fig9
make validate
```

## Repository Layout

- `Algorithm/`: RTDFL and baseline training implementations.
- `models/`: model definitions.
- `utils/`: datasets, sampling, quantization, energy, and support utilities.
- `configs/`: quick and full experiment specifications.
- `scripts/`: execution, validation, aggregation, and plotting tools.
- `tests/`: synthetic core tests.
- `real_device/`: optional heterogeneous Jetson deployment workflow.

See `real_device/README.md` for the physical-device environment and local
configuration procedure. Credentials and testbed addresses must be stored only
in the ignored `real_device/deploy_sh/devices_local.py`.

## License

Original RTDFL code is released under the MIT License in `LICENSE`. Adapted
third-party portions and their BSD notices are documented in
`THIRD_PARTY_NOTICES.md`.
