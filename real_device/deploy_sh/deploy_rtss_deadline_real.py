import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

import paramiko

from device_config import ALL_DEVICES, SERVER_DEVICES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT_NAME = "server_fedavg"
DEADLINE_SEC = 150.0
SENSITIVITY_SECS = "120,150,180,210"
FORMAL_EPOCHS = 50
SEED = 1


def environment_setting(name, default=""):
    return os.environ.get(name, default).strip()


if not SERVER_DEVICES:
    raise RuntimeError(
        "SERVER_DEVICES is empty; create deploy_sh/devices_local.py from the example"
    )

# Network addresses and credentials come from the ignored devices_local.py file.
# Per-machine runtime differences stay outside the public source tree as
# RTDFL_SERVER_PYTHON and RTDFL_CLIENT_<CID>_* environment variables.
SERVER_IP = SERVER_DEVICES[0][0]
SERVER_PYTHON = environment_setting("RTDFL_SERVER_PYTHON", "/usr/bin/python3")
CLIENTS = []
CLIENT_ENVIRONMENTS = {}
EXPECTED_POWER_MODE = {}
LOGIN_PTY_CLIENTS = set()

for cid, device in enumerate(ALL_DEVICES):
    ip = device[0]
    prefix = "RTDFL_CLIENT_{}_".format(cid)
    python_path = environment_setting(prefix + "PYTHON", "/usr/bin/python3")
    CLIENTS.append((cid, ip, python_path))

    shell_prefix = environment_setting(prefix + "SHELL_PREFIX")
    if shell_prefix:
        CLIENT_ENVIRONMENTS[ip] = shell_prefix

    expected_mode = environment_setting(prefix + "POWER_MODE")
    if expected_mode:
        EXPECTED_POWER_MODE[ip] = expected_mode

    if environment_setting(prefix + "LOGIN_PTY").lower() in ("1", "true", "yes"):
        LOGIN_PTY_CLIENTS.add(ip)

METHODS = {
    "rtdfl": {
        "label": "RTDFL",
        "server_script": "server_autorl_real.py",
        "client_script": "client_autorl_real.py",
        "server_log_dir": "logs_real/autorl_real_server",
    },
    "dfl": {
        "label": "DFL-Real",
        "server_script": "server_dfl_real.py",
        "client_script": "client_dfl_real.py",
        "server_log_dir": "logs_real/dfl_real_server",
    },
    "mmdfl": {
        "label": "MMDFL-Real",
        "server_script": "server_mmdfl_real.py",
        "client_script": "client_mmdfl_real.py",
        "server_log_dir": "logs_real/mmdfl_real_server",
    },
}

RUNTIME_FILES = [
    "client_autorl_real.py",
    "server_autorl_real.py",
    "client_dfl_real.py",
    "server_dfl_real.py",
    "client_mmdfl_real.py",
    "server_mmdfl_real.py",
    "models/resnet8_mmdfl.py",
    "utils/ConnectHandler_client.py",
    "utils/ConnectHandler_server.py",
    "utils/FEMNIST.py",
    "utils/FL_utils.py",
    "utils/ShakeSpare.py",
    "utils/autorl_real_policy.py",
    "utils/dataset_utils.py",
    "utils/get_dataset.py",
    "utils/language_utils.py",
    "utils/main_real_profiles.py",
    "utils/mmdfl_real_client_runtime.py",
    "utils/mmdfl_real_common.py",
    "utils/mmdfl_real_log.py",
    "utils/mmdfl_real_policy.py",
    "utils/mmdfl_real_train.py",
    "utils/mmdfl_topology.py",
    "utils/options.py",
    "utils/power_manager_real.py",
    "utils/rtss_deadline_metrics.py",
    "utils/sampling.py",
    "utils/set_seed.py",
]


def device_by_ip(ip):
    for device in ALL_DEVICES + SERVER_DEVICES:
        if device[0] == ip:
            return device
    raise RuntimeError("Device {} is missing from deploy_sh/devices_local.py".format(ip))


def connect(device):
    ip, user, password = device
    if user == "CHANGE_ME" or password == "CHANGE_ME":
        raise RuntimeError(
            "placeholder credentials detected; copy devices_example.py to "
            "devices_local.py and fill in the target testbed settings"
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        ip,
        username=user,
        password=password,
        timeout=8,
        banner_timeout=8,
        auth_timeout=8,
    )
    return client


def remote_root(device):
    return "/home/{}/{}".format(device[1], REMOTE_ROOT_NAME)


def run_remote(ssh, command, timeout=30, check=True, get_pty=False):
    _, stdout, stderr = ssh.exec_command(command, timeout=timeout, get_pty=get_pty)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    status = stdout.channel.recv_exit_status()
    if check and status != 0:
        raise RuntimeError("remote command failed ({}): {}\n{}".format(status, command, err))
    return status, out, err


def ensure_remote_dir(ssh, path):
    run_remote(ssh, "mkdir -p {}".format(shlex.quote(str(path))))


def upload_files(device):
    ssh = connect(device)
    try:
        root = PurePosixPath(remote_root(device))
        sftp = ssh.open_sftp()
        for relative in RUNTIME_FILES:
            local_path = PROJECT_ROOT / relative
            if not local_path.is_file():
                raise RuntimeError("Missing local runtime file: {}".format(local_path))
            remote_path = root / relative
            ensure_remote_dir(ssh, remote_path.parent)
            sftp.put(str(local_path), str(remote_path))
        sftp.close()
    finally:
        ssh.close()


def deploy(_args):
    targets = [device_by_ip(SERVER_IP)] + [device_by_ip(ip) for _, ip, _ in CLIENTS]
    for device in targets:
        print("deploy -> {}@{}".format(device[1], device[0]), flush=True)
        upload_files(device)
    print("deployed {} files to {} devices".format(len(RUNTIME_FILES), len(targets)))


def method_spec(method):
    return METHODS[str(method)]


def matching_process_command(role, method=None):
    key = "server_script" if role == "server" else "client_script"
    specs = [method_spec(method)] if method else list(METHODS.values())
    patterns = ["[{}]{}".format(spec[key][0], spec[key][1:]) for spec in specs]
    return "pgrep -af {} || true".format(shlex.quote("|".join(patterns)))


def target_environment(device):
    return CLIENT_ENVIRONMENTS.get(device[0], "")


def requires_login_pty(device):
    return device[0] in LOGIN_PTY_CLIENTS


def inspect_target(device, python_path, role, method):
    ssh = connect(device)
    try:
        root = remote_root(device)
        spec = method_spec(method)
        script = spec["server_script"] if role == "server" else spec["client_script"]
        environment = target_environment(device)
        python_command = "{} {}".format(environment, shlex.quote(python_path)).strip()
        runtime_module = Path(script).stem
        checks = [
            "test -x {}".format(shlex.quote(python_path)),
            "test -f {}".format(shlex.quote(root + "/" + script)),
            "cd {} && {} -c {}".format(
                shlex.quote(root),
                python_command,
                shlex.quote(
                    "import torch, torchvision, numpy, loguru; "
                    "import {}; "
                    "assert torch.cuda.is_available(); "
                    "print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
                    .format(runtime_module)
                ),
            ),
        ]
        if role == "server":
            checks.append("! ss -ltn | grep -q ':8080 '")
        command = " && ".join(checks)
        use_login_pty = role == "client" and requires_login_pty(device)
        if use_login_pty:
            command = "bash -lic {}".format(shlex.quote(command))
        status, out, err = run_remote(
            ssh,
            command,
            timeout=60,
            check=False,
            get_pty=use_login_pty,
        )
        _, processes, _ = run_remote(ssh, matching_process_command(role), check=False)
        if processes.strip():
            status = 1
            err += "matching process is already running:\n{}".format(processes)
        meta_cmd = (
            "printf 'model='; tr -d '\\0' </proc/device-tree/model 2>/dev/null || hostname; "
            "echo; nvpmodel -q 2>/dev/null | head -4 || true; uptime"
        )
        _, meta, _ = run_remote(ssh, meta_cmd, check=False)
        expected_mode = EXPECTED_POWER_MODE.get(device[0])
        if role == "client" and expected_mode and "NV Power Mode: {}".format(expected_mode) not in meta:
            status = 1
            err += "unexpected power mode; expected {}\n".format(expected_mode)
        return status, out + meta, err
    finally:
        ssh.close()


def preflight(args):
    method = str(getattr(args, "method", "rtdfl"))
    targets = [(device_by_ip(SERVER_IP), SERVER_PYTHON, "server")]
    targets.extend((device_by_ip(ip), python_path, "client") for _, ip, python_path in CLIENTS)
    failed = []
    for device, python_path, role in targets:
        label = "{} {}".format(role, device[0])
        try:
            status, out, err = inspect_target(device, python_path, role, method)
        except Exception as exc:
            failed.append(label)
            print("[FAIL] {}\nconnection/check error: {}".format(label, exc), flush=True)
            continue
        if status == 0:
            print("[OK] {}\n{}".format(label, out.strip()), flush=True)
        else:
            failed.append(label)
            print("[FAIL] {}\n{}\n{}".format(label, out.strip(), err.strip()), flush=True)
    if failed:
        raise RuntimeError("preflight failed: {}".format(", ".join(failed)))


def make_tag(kind, method="rtdfl"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "rtss_{}_ddl150_{}_seed1_{}".format(method, kind, timestamp)


def server_command(device, tag, epochs, fresh_start, method="rtdfl", append_log=False):
    root = remote_root(device)
    redirect = ">>" if append_log else ">"
    spec = method_spec(method)
    args = [
        SERVER_PYTHON,
        "./{}".format(spec["server_script"]),
        "--epochs", str(int(epochs)),
        "--seed", str(SEED),
        "--fresh_start", str(int(fresh_start)),
        "--deadline_sec", str(DEADLINE_SEC),
        "--deadline_sensitivity_secs", SENSITIVITY_SECS,
        "--log_tag", tag,
    ]
    command = " ".join(shlex.quote(item) for item in args)
    return (
        "cd {root} && printf '%s\\n' {tag} > .rtss_deadline_run_tag && "
        "(nohup env PYTHONUNBUFFERED=1 {command} {redirect} server.log 2>&1 < /dev/null & "
        "echo $! > server.pid)"
    ).format(
        root=shlex.quote(root),
        tag=shlex.quote(tag),
        command=command,
        redirect=redirect,
    )


def client_command(device, python_path, cid, tag, epochs, method="rtdfl"):
    root = remote_root(device)
    spec = method_spec(method)
    args = [
        python_path,
        "./{}".format(spec["client_script"]),
        "--HOST", SERVER_IP,
        "--CID", str(int(cid)),
        "--epochs", str(int(epochs)),
        "--seed", str(SEED),
        "--deadline_sec", str(DEADLINE_SEC),
        "--deadline_sensitivity_secs", SENSITIVITY_SECS,
        "--log_tag", tag,
    ]
    command = " ".join(shlex.quote(item) for item in args)
    environment = target_environment(device)
    if environment:
        command = "{} {}".format(environment, command)
    return (
        "cd {root} && printf '%s\\n' {tag} > .rtss_deadline_run_tag && "
        "(nohup env PYTHONUNBUFFERED=1 {command} > client.log 2>&1 < /dev/null & "
        "echo $! > client.pid)"
    ).format(root=shlex.quote(root), tag=shlex.quote(tag), command=command)


def start_run(tag, epochs, fresh_start=1, wait_for_round=False, method="rtdfl"):
    server = device_by_ip(SERVER_IP)
    ssh = connect(server)
    try:
        run_remote(
            ssh,
            server_command(server, tag, epochs, fresh_start, method=method),
            timeout=20,
        )
        time.sleep(2)
        status, out, _ = run_remote(
            ssh,
            "cd {} && kill -0 $(cat server.pid) 2>/dev/null; tail -30 server.log".format(
                shlex.quote(remote_root(server))
            ),
            check=False,
        )
        if status != 0:
            raise RuntimeError("server failed immediately:\n{}".format(out))
    finally:
        ssh.close()

    for cid, ip, python_path in CLIENTS:
        device = device_by_ip(ip)
        print("start client CID={} {}".format(cid, ip), flush=True)
        ssh = connect(device)
        try:
            command = client_command(device, python_path, cid, tag, epochs, method=method)
            use_login_pty = requires_login_pty(device)
            if use_login_pty:
                command = "bash -lic {}".format(shlex.quote(command))
            run_remote(ssh, command, timeout=20, get_pty=use_login_pty)
            time.sleep(0.3)
            status, _, err = run_remote(
                ssh,
                "cd {} && kill -0 $(cat client.pid)".format(shlex.quote(remote_root(device))),
                check=False,
            )
            if status != 0:
                _, log, _ = run_remote(
                    ssh,
                    "cd {} && tail -60 client.log".format(shlex.quote(remote_root(device))),
                    check=False,
                )
                raise RuntimeError("client CID={} failed immediately: {}\n{}".format(cid, err, log))
        finally:
            ssh.close()

    if wait_for_round:
        wait_for_log("ROUND=0 end", timeout_sec=900)
    else:
        wait_for_log("all clients are ready", timeout_sec=180)
    print("run started method={} tag={} epochs={}".format(method, tag, epochs))


def read_server_log():
    server = device_by_ip(SERVER_IP)
    ssh = connect(server)
    try:
        _, out, _ = run_remote(
            ssh,
            "cd {} && tail -200 server.log 2>/dev/null || true".format(shlex.quote(remote_root(server))),
            check=False,
        )
        return out
    finally:
        ssh.close()


def wait_for_log(pattern, timeout_sec):
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        log = read_server_log()
        if pattern in log:
            return log
        if "Traceback (most recent call last)" in log:
            raise RuntimeError("server traceback:\n{}".format(log))
        time.sleep(10)
    raise RuntimeError("timed out waiting for {!r}; latest log:\n{}".format(pattern, read_server_log()))


def wait_for_completion(timeout_sec):
    deadline = time.time() + float(timeout_sec)
    server = device_by_ip(SERVER_IP)
    while time.time() < deadline:
        ssh = connect(server)
        try:
            status, _, _ = run_remote(
                ssh,
                "cd {} && kill -0 $(cat server.pid) 2>/dev/null".format(shlex.quote(remote_root(server))),
                check=False,
            )
        finally:
            ssh.close()
        log = read_server_log()
        if status != 0:
            if "FINAL_DEADLINE_SUMMARY" not in log:
                raise RuntimeError("server stopped without final summary:\n{}".format(log))
            return log
        time.sleep(15)
    raise RuntimeError("run did not complete before timeout")


def smoke(args):
    method = str(args.method)
    preflight(args)
    tag = make_tag("smoke2r", method=method)
    start_run(tag, epochs=2, fresh_start=1, wait_for_round=True, method=method)
    log = wait_for_completion(timeout_sec=1200)
    if "ROUND=1 end" not in log:
        raise RuntimeError("smoke run did not finish both rounds:\n{}".format(log))
    print(log)
    print("smoke passed tag={}".format(tag))


def start(args):
    method = str(args.method)
    preflight(args)
    tag = args.tag or make_tag("50r", method=method)
    start_run(
        tag,
        epochs=FORMAL_EPOCHS,
        fresh_start=1,
        wait_for_round=True,
        method=method,
    )
    server = device_by_ip(SERVER_IP)
    print(
        "monitor: ssh {}@{} 'cd ~/server_fedavg && tail -f server.log'".format(
            server[1], SERVER_IP
        )
    )


def status(_args):
    server = device_by_ip(SERVER_IP)
    ssh = connect(server)
    try:
        _, out, _ = run_remote(
            ssh,
            "cd {root} && echo tag=$(cat .rtss_deadline_run_tag 2>/dev/null); "
            "if test -f server.pid && kill -0 $(cat server.pid) 2>/dev/null; then echo state=running pid=$(cat server.pid); "
            "else echo state=stopped; fi; tail -80 server.log 2>/dev/null || true".format(
                root=shlex.quote(remote_root(server))
            ),
            check=False,
        )
        print(out)
    finally:
        ssh.close()


def stop_role(device, role, method="rtdfl"):
    pid_file = "server.pid" if role == "server" else "client.pid"
    spec = method_spec(method)
    script = spec["server_script"] if role == "server" else spec["client_script"]
    ssh = connect(device)
    try:
        command = (
            "cd {root} && if test -f {pid_file}; then pid=$(cat {pid_file}); "
            "if kill -0 $pid 2>/dev/null && ps -p $pid -o args= | grep -q {script}; "
            "then kill $pid; fi; fi"
        ).format(
            root=shlex.quote(remote_root(device)),
            pid_file=pid_file,
            script=shlex.quote(script),
        )
        run_remote(ssh, command, check=False)
    finally:
        ssh.close()


def stop(args):
    method = str(args.method)
    stop_role(device_by_ip(SERVER_IP), "server", method=method)
    for _, ip, _ in CLIENTS:
        stop_role(device_by_ip(ip), "client", method=method)
    print("requested stop for {} deadline processes".format(method_spec(method)["label"]))


def collect(args):
    method = str(args.method)
    spec = method_spec(method)
    server = device_by_ip(SERVER_IP)
    ssh = connect(server)
    try:
        root = remote_root(server)
        _, tag, _ = run_remote(ssh, "cat {}/.rtss_deadline_run_tag".format(shlex.quote(root)))
        tag = tag.strip()
        output_dir = Path(args.output_dir or PROJECT_ROOT / "data_process" / "rtss" / "deadline_real" / tag)
        output_dir.mkdir(parents=True, exist_ok=True)
        sftp = ssh.open_sftp()
        sftp.get(root + "/server.log", str(output_dir / "server.log"))
        _, paths, _ = run_remote(
            ssh,
            "cd {root} && find {log_dir} -type f -name '*{tag}*' -print".format(
                root=shlex.quote(root),
                log_dir=shlex.quote(spec["server_log_dir"]),
                tag=tag,
            ),
        )
        for relative in paths.splitlines():
            if not relative.strip():
                continue
            destination = output_dir / Path(relative).name
            sftp.get(root + "/" + relative, str(destination))
        sftp.close()
        (output_dir / "run.json").write_text(
            json.dumps({"method": method, "tag": tag, "server": SERVER_IP}, indent=2),
            encoding="utf-8",
        )
        print("collected tag={} -> {}".format(tag, output_dir))
    finally:
        ssh.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Deploy and run RTDFL deadline measurements")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    subparsers.add_parser("deploy").set_defaults(func=deploy)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--method", choices=sorted(METHODS), default="rtdfl")
    preflight_parser.set_defaults(func=preflight)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--method", choices=sorted(METHODS), default="rtdfl")
    smoke_parser.set_defaults(func=smoke)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--tag", default="")
    start_parser.add_argument("--method", choices=sorted(METHODS), default="rtdfl")
    start_parser.set_defaults(func=start)
    subparsers.add_parser("status").set_defaults(func=status)
    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--method", choices=sorted(METHODS), default="rtdfl")
    stop_parser.set_defaults(func=stop)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--output-dir", default="")
    collect_parser.add_argument("--method", choices=sorted(METHODS), default="rtdfl")
    collect_parser.set_defaults(func=collect)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
