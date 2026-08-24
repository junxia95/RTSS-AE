# RTDFL Physical-Testbed Workflow

This directory contains the physical-device workflow for RTDFL-Real, DFL-Real,
and MMDFL-Real.

The physical testbed consists of one Ubuntu controller/server, 7 Jetson Nano
boards, 2 Jetson AGX Xavier boards, 1 Jetson Orin Nano Super board, and three
HP 9800 power meters. All devices communicate over the same local network.

## Method Entry Points

| Method | Server | Client |
|---|---|---|
| RTDFL-Real | `server_autorl_real.py` | `client_autorl_real.py` |
| DFL-Real | `server_dfl_real.py` | `client_dfl_real.py` |
| MMDFL-Real | `server_mmdfl_real.py` | `client_mmdfl_real.py` |

RTDFL routing, UCB exploration, and selective inbound aggregation are
implemented in `utils/autorl_real_policy.py`. Shared model training, topology,
logging, and deadline accounting are isolated under
`utils/mmdfl_real_*.py`, `utils/mmdfl_topology.py`, and
`utils/rtss_deadline_metrics.py`.

The submitted physical runtime sends complete full-precision PyTorch state
dictionaries. Quantization-aware training and quantized network payloads are
not enabled in this directory. Deadline thresholds are logged for post-run
analysis; they do not stop or reschedule a run.

## Formal Configuration

The deployment controller uses the following formal settings:

- dataset: CIFAR-10;
- clients: 10;
- traveler models per round: 2;
- rounds: 50;
- seed: 1;
- Dirichlet alpha: 0.1;
- local epochs: 5;
- local batch size: 32;
- primary deadline: 150 seconds;
- deadline sensitivity thresholds: 120, 150, 180, and 210 seconds.

The RTDFL server uses the online policy. Archived development schedules are
not consumed by the runtime and are not included in this code-only release.

## Environment

The controller requires Python 3 and `paramiko`. Each Jetson requires the
PyTorch and torchvision builds supplied for its installed JetPack release,
plus NumPy, tqdm, loguru, and ujson. CUDA and the selected `nvpmodel` mode must
be verified on every device before a formal run.

Do not install generic x86 CUDA wheels on Jetson boards. Use the NVIDIA wheels
matching the board's JetPack version.

After the board-specific PyTorch installation, install the common packages:

```bash
python3 -m pip install -r requirements.txt
```

## Device Configuration

Copy the credential template and edit only the local copy:

```bash
cp deploy_sh/devices_example.py deploy_sh/devices_local.py
```

`deploy_sh/devices_local.py` is ignored by Git. It defines SSH credentials for
the server and ten clients. The addresses in the example are RFC 5737
documentation addresses and must be replaced. List the clients in CID order;
the first item in `ALL_DEVICES` is CID 0.

Runtime paths and optional board-specific settings are supplied through local
environment variables rather than committed source code. The default Python
executable is `/usr/bin/python3`. Override only the devices that need it:

```bash
export RTDFL_SERVER_PYTHON=/path/to/server/python
export RTDFL_CLIENT_0_PYTHON=/path/to/client/python
export RTDFL_CLIENT_0_POWER_MODE=MAXN
export RTDFL_CLIENT_0_SHELL_PREFIX=OPENBLAS_CORETYPE=ARMV8
export RTDFL_CLIENT_0_LOGIN_PTY=1
```

Repeat the `RTDFL_CLIENT_<CID>_*` variables for other clients as needed. The
deployment controller takes the server endpoint from the first
`SERVER_DEVICES` entry and passes it to every client with `--HOST`. For a
direct client invocation, use `--HOST <server-address>` or set
`RTDFL_SERVER_HOST`.

Before starting, verify that CIFAR-10 and the Python environments are already
available on every device, TCP port 8080 is free on the server, and no previous
RTDFL/DFL/MMDFL process remains active.

## Deployment and Execution

Run all commands below from this directory on the controller.

Deploy the runtime files:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py deploy
```

Check SSH access, Python, CUDA, power modes, port availability, and processes:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py preflight --method rtdfl
```

Run a two-round physical smoke test:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py smoke --method rtdfl
```

Start a formal run:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py start --method rtdfl
```

Use `dfl` or `mmdfl` in place of `rtdfl` for the baselines. The three methods
share the same server port and testbed and therefore run sequentially.

Remote processes are launched with `nohup`. Monitor or stop them with:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py status
python3 deploy_sh/deploy_rtss_deadline_real.py stop --method rtdfl
```

## Result Collection

After a run reaches `FINAL_DEADLINE_SUMMARY`, collect its server-side records:

```bash
python3 deploy_sh/deploy_rtss_deadline_real.py collect --method rtdfl
```

The controller stores collected output under
`data_process/rtss/deadline_real/<run-tag>/`. The main records are:

- `server.log`: complete server output;
- `rounds_*.jsonl`: per-round accuracy, latency, communication, and deadline data;
- `actions_*.jsonl`: RTDFL controller decisions and decision overhead;
- `summary_*.json`: aggregate deadline statistics;
- `run.json`: method, run tag, and server metadata.

External power-meter exports are aligned with the timestamps in the server and
client records. Keep the original meter files, device-to-meter mapping, sample
rate, and run start/end markers together with each collected run.

## Paper-Trace Replay

The code for parsing and plotting the paper's physical traces is included under
`../scripts/`. The raw server logs and power-meter exports are distributed
separately with the RTSS artifact evidence bundle and are not stored in this
code-only repository.
