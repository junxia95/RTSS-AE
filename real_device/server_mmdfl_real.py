import copy
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

from utils.ConnectHandler_server import ConnectHandler
from utils.get_dataset import get_dataset
from utils.main_real_profiles import get_device_type
from utils.mmdfl_real_common import checkpoint_path as method_checkpoint_path
from utils.mmdfl_real_log import JsonlWriter, make_log_dir, now_str
from utils.mmdfl_real_policy import RealMMDFLPolicy
from utils.mmdfl_real_train import (
    average_state_dicts,
    build_model,
    evaluate_model,
    label_distribution,
    state_dict_to_cpu,
)
from utils.options import args_parser
from utils.power_manager_real import LOW_BATTERY_THRESHOLD_J, get_device_capacity
from utils.rtss_deadline_metrics import parse_deadline_sensitivity, summarize_deadlines
from utils.set_seed import set_random_seed


REAL_DEFAULTS = {
    "algorithm": "DFL_MM",
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


LEGACY_CHECKPOINT_PATH = Path("logs_real/mmdfl_real_server/checkpoints/mmdfl_real_checkpoint.pt")


def apply_real_defaults(args):
    cli_algorithm = str(args.algorithm)
    for key, value in REAL_DEFAULTS.items():
        setattr(args, key, value)
    if cli_algorithm not in {"", "heterofl"}:
        args.algorithm = cli_algorithm
    return args


def pickle_size_mb(payload):
    return len(pickle.dumps(payload)) / (1024.0 * 1024.0)


def client_capacity_map(num_clients):
    return {cid: get_device_capacity(get_device_type(cid)) for cid in range(num_clients)}


def relocate_if_inactive(token_locations, active_clients, rng):
    active_list = sorted(active_clients)
    if not active_list:
        return token_locations
    return [
        int(cid) if int(cid) in active_clients else int(rng.choice(active_list))
        for cid in token_locations
    ]


def enforce_unique_train_locations(token_locations, active_clients, rng):
    active_list = sorted(int(cid) for cid in active_clients)
    if not active_list:
        return list(token_locations), []

    used = set()
    fixed_locations = []
    relocations = []
    for token_id, cid in enumerate(token_locations):
        cid = int(cid)
        if cid in active_clients and cid not in used:
            fixed_locations.append(cid)
            used.add(cid)
            continue

        available = [candidate for candidate in active_list if candidate not in used]
        if not available:
            fixed_locations.append(cid)
            relocations.append(
                {
                    "token_id": int(token_id),
                    "src": cid,
                    "dst": cid,
                    "reason": "no_unique_active_client",
                }
            )
            continue

        dst = int(rng.choice(available))
        fixed_locations.append(dst)
        used.add(dst)
        relocations.append(
            {
                "token_id": int(token_id),
                "src": cid,
                "dst": dst,
                "reason": "duplicate_or_inactive_location",
            }
        )
    return fixed_locations, relocations


def aggregate_collided_tokens(model_tokens, token_locations, model_distributions, label_distributions):
    tokens_by_cid = defaultdict(list)
    for token_id, cid in enumerate(token_locations):
        tokens_by_cid[int(cid)].append(token_id)

    for cid, token_ids in tokens_by_cid.items():
        if len(token_ids) <= 1:
            continue
        averaged_state = average_state_dicts([model_tokens[token_id].state_dict() for token_id in token_ids])
        for token_id in token_ids:
            model_tokens[token_id].load_state_dict(averaged_state, strict=True)
            model_distributions[token_id] = label_distributions[cid].copy()
        logger.info(f"{now_str()} AGG_COLLISION cid={cid} tokens={token_ids}")


def build_train_payload(args, token_id, round_idx, model, idxs_list, battery_state_joules):
    lr = float(args.lr * (args.lr_decay ** round_idx))
    return {
        "type": "train_round",
        "round": int(round_idx),
        "token_id": int(token_id),
        "algorithm": str(args.algorithm),
        "net": state_dict_to_cpu(model.state_dict()),
        "idxs_list": list(idxs_list),
        "local_ep": int(args.local_ep),
        "local_bs": int(args.local_bs),
        "lr": lr,
        "server_known_battery_joules": float(battery_state_joules),
    }


def policy_state_dict(policy):
    return {
        "q_table": [(src, dst, value) for (src, dst), value in policy.q_table.items()],
        "epsilon": float(policy.epsilon),
        "rng_state": policy.rng.getstate(),
    }


def load_policy_state(policy, state):
    if not state:
        return
    policy.q_table = {
        (int(src), int(dst)): float(value)
        for src, dst, value in state.get("q_table", [])
    }
    policy.epsilon = float(state.get("epsilon", policy.epsilon))
    rng_state = state.get("rng_state")
    if rng_state is not None:
        policy.rng.setstate(rng_state)


def save_checkpoint(
    checkpoint_path,
    args,
    round_idx,
    model_tokens,
    token_locations,
    model_distributions,
    last_visit_round,
    active_clients,
    battery_state_joules,
    best_acc,
    summary_records,
    train_time_estimates,
    rng,
    policy,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "saved_at": now_str(),
        "completed_round": int(round_idx),
        "next_round": int(round_idx) + 1,
        "algorithm": str(args.algorithm),
        "dataset": str(args.dataset),
        "model": str(args.model),
        "num_users": int(args.num_users),
        "frac": float(args.frac),
        "token_count": len(model_tokens),
        "local_ep": int(args.local_ep),
        "seed": int(args.seed),
        "model_token_state_dicts": [
            state_dict_to_cpu(model.state_dict()) for model in model_tokens
        ],
        "token_locations": list(token_locations),
        "model_distributions": copy.deepcopy(model_distributions),
        "last_visit_round": list(last_visit_round),
        "active_clients": sorted(active_clients),
        "battery_state_joules": dict(battery_state_joules),
        "best_acc": float(best_acc),
        "summary_records": copy.deepcopy(summary_records),
        "train_time_estimates": dict(train_time_estimates),
        "rng_state": rng.getstate(),
        "policy_state": policy_state_dict(policy),
    }
    torch.save(state, checkpoint_path)


def load_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def restore_checkpoint(
    checkpoint,
    args,
    model_tokens,
    rng,
    policy,
    token_count,
    num_clients,
):
    if int(checkpoint.get("token_count", -1)) != int(token_count):
        raise ValueError("checkpoint token_count does not match current settings")
    if int(checkpoint.get("num_users", -1)) != int(num_clients):
        raise ValueError("checkpoint num_users does not match current settings")

    token_states = checkpoint.get("model_token_state_dicts", [])
    if len(token_states) != token_count:
        raise ValueError("checkpoint model token count is invalid")
    for token_id, token_state in enumerate(token_states):
        model_tokens[token_id].load_state_dict(token_state, strict=True)

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        rng.setstate(rng_state)
    load_policy_state(policy, checkpoint.get("policy_state"))

    return {
        "start_round": int(checkpoint.get("next_round", int(checkpoint.get("completed_round", -1)) + 1)),
        "token_locations": [int(cid) for cid in checkpoint["token_locations"]],
        "model_distributions": checkpoint["model_distributions"],
        "last_visit_round": [int(value) for value in checkpoint["last_visit_round"]],
        "active_clients": set(int(cid) for cid in checkpoint["active_clients"]),
        "battery_state_joules": {
            int(cid): float(value)
            for cid, value in checkpoint["battery_state_joules"].items()
        },
        "best_acc": float(checkpoint.get("best_acc", 0.0)),
        "summary_records": list(checkpoint.get("summary_records", [])),
        "train_time_estimates": {
            int(cid): float(value)
            for cid, value in checkpoint.get("train_time_estimates", {}).items()
        },
    }


if __name__ == "__main__":
    args = apply_real_defaults(args_parser())
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(args.seed)
    rng = random.Random(int(args.seed))
    deadline_sec = max(float(args.deadline_sec), 0.0)
    deadline_sensitivity_secs = parse_deadline_sensitivity(
        args.deadline_sensitivity_secs,
        deadline_sec,
    )

    log_dir = make_log_dir("logs_real", "mmdfl_real_server")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_label = args.log_tag or f"seed{args.seed}"
    checkpoint_path = method_checkpoint_path(log_dir, args.algorithm, args)
    logger.add(log_dir / f"server_mmdfl_real_{run_label}_{timestamp}.log")
    round_writer = JsonlWriter(log_dir / f"rounds_{run_label}_{timestamp}.jsonl")

    num_clients = int(args.num_users)
    token_count = max(int(num_clients * float(args.frac)), 1)
    active_clients = set(range(num_clients))
    battery_state_joules = client_capacity_map(num_clients)
    best_acc = 0.0
    summary_records = []
    train_time_estimates = {
        cid: 1.0 if get_device_type(cid) == "orinnanosuper" else (1.5 if get_device_type(cid) == "agx_xavier" else 2.5)
        for cid in range(num_clients)
    }

    logger.info(
        f"Starting MMDFL real server algorithm={args.algorithm} device={args.device} "
        f"dataset={args.dataset} users={num_clients} token_count={token_count} seed={args.seed} "
        f"deadline_sec={deadline_sec} deadline_mode=observe_only "
        f"deadline_sensitivity_secs={deadline_sensitivity_secs}"
    )

    dataset_train, dataset_test, dict_users = get_dataset(args)
    label_distributions = {
        cid: label_distribution(dataset_train, dict_users[cid], args.num_classes)
        for cid in range(num_clients)
    }

    base_model = build_model(args)
    model_tokens = [copy.deepcopy(base_model) for _ in range(token_count)]
    token_locations = rng.sample(list(range(num_clients)), token_count)
    model_distributions = [label_distributions[cid].copy() for cid in token_locations]
    last_visit_round = [0 for _ in range(num_clients)]
    policy = RealMMDFLPolicy(args=args, algorithm=args.algorithm)
    start_round = 0

    restore_path = checkpoint_path
    if not restore_path.exists() and LEGACY_CHECKPOINT_PATH.exists() and not int(getattr(args, "fresh_start", 0)):
        logger.warning(
            f"Legacy checkpoint exists at {LEGACY_CHECKPOINT_PATH}; not auto-restoring it into "
            f"isolated run path {checkpoint_path}. Re-run with matching log_tag or move it manually if needed."
        )

    if restore_path.exists() and not int(getattr(args, "fresh_start", 0)):
        checkpoint = load_checkpoint(restore_path)
        restored = restore_checkpoint(
            checkpoint=checkpoint,
            args=args,
            model_tokens=model_tokens,
            rng=rng,
            policy=policy,
            token_count=token_count,
            num_clients=num_clients,
        )
        start_round = restored["start_round"]
        token_locations = restored["token_locations"]
        model_distributions = restored["model_distributions"]
        last_visit_round = restored["last_visit_round"]
        active_clients = restored["active_clients"]
        battery_state_joules = restored["battery_state_joules"]
        best_acc = restored["best_acc"]
        summary_records = restored["summary_records"]
        restored_train_time_estimates = restored["train_time_estimates"]
        for cid, default_value in train_time_estimates.items():
            restored_train_time_estimates.setdefault(cid, default_value)
        train_time_estimates = restored_train_time_estimates
        logger.info(
            f"Resumed MMDFL server from checkpoint={restore_path} "
            f"completed_round={checkpoint.get('completed_round')} start_round={start_round}"
        )
    else:
        logger.info(f"No checkpoint restored at {checkpoint_path}, starting from round 0 fresh_start={args.fresh_start}")

    logger.info(
        f"Token locations at start_round={start_round}: {token_locations} "
        f"waiting_clients={num_clients}"
    )
    connect_handler = ConnectHandler(num_clients, args.HOST, args.POST)

    for round_idx in range(start_round, int(args.epochs)):
        if not active_clients:
            logger.warning("No active clients left, stopping.")
            break

        round_start_ts = time.time()
        round_start_perf = time.perf_counter()
        round_start_time = now_str()
        token_locations = relocate_if_inactive(token_locations, active_clients, rng)
        aggregate_collided_tokens(model_tokens, token_locations, model_distributions, label_distributions)
        token_locations, pre_send_relocations = enforce_unique_train_locations(
            token_locations, active_clients, rng
        )
        locations_before = list(token_locations)
        for relocation in pre_send_relocations:
            logger.info(f"{now_str()} PRE_SEND_RELOCATE round={round_idx} {relocation}")

        logger.info(
            f"{round_start_time} ROUND={round_idx} start token_locations={locations_before} "
            f"active_clients={sorted(active_clients)}"
        )

        expected_tokens = set()
        send_records = []
        send_record_by_token = {}
        task_start_perf_by_token = {}
        comm_mb = 0.0
        sent_clients = set()
        for token_id, cid in enumerate(locations_before):
            if cid not in active_clients:
                continue
            if cid in sent_clients:
                logger.warning(
                    f"{now_str()} SKIP_DUPLICATE_SEND round={round_idx} token={token_id} cid={cid}"
                )
                continue
            sent_clients.add(cid)
            payload = build_train_payload(
                args=args,
                token_id=token_id,
                round_idx=round_idx,
                model=model_tokens[token_id],
                idxs_list=dict_users[cid],
                battery_state_joules=battery_state_joules[cid],
            )
            payload_mb = pickle_size_mb(payload)
            send_start = time.time()
            send_start_perf = time.perf_counter()
            ok = connect_handler.sendData(cid, payload)
            send_duration = time.perf_counter() - send_start_perf
            if not ok:
                logger.warning(f"Failed sending round={round_idx} token={token_id} to cid={cid}")
                active_clients.discard(cid)
                continue
            expected_tokens.add(token_id)
            comm_mb += payload_mb
            send_record = {
                "token_id": token_id,
                "cid": cid,
                "server_send_start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(send_start),
                ),
                "send_duration_sec": send_duration,
                "payload_mb": payload_mb,
            }
            send_records.append(send_record)
            send_record_by_token[token_id] = send_record
            task_start_perf_by_token[token_id] = send_start_perf
            logger.info(
                f"{now_str()} SEND round={round_idx} token={token_id} cid={cid} "
                f"payload_mb={payload_mb:.3f} duration={send_duration:.3f}s"
            )

        token_updates = {}
        status_records = []
        while len(token_updates) + len(status_records) < len(expected_tokens):
            msg, cid = connect_handler.receiveData()
            msg_type = msg.get("type")
            token_id = int(msg.get("token_id", -1))
            if msg_type == "client_update" and int(msg.get("round", -1)) == round_idx and token_id in expected_tokens:
                receive_time = now_str()
                receive_ts = time.time()
                receive_perf = time.perf_counter()
                task_latency_sec = receive_perf - task_start_perf_by_token[token_id]
                comm_mb += pickle_size_mb(msg)
                token_updates[token_id] = (
                    cid,
                    msg,
                    receive_ts,
                    receive_time,
                    task_latency_sec,
                )
                battery_state_joules[cid] = float(msg.get("battery_joules", battery_state_joules.get(cid, 0.0)))
                train_time_estimates[cid] = 0.7 * train_time_estimates.get(cid, 1.0) + 0.3 * float(
                    msg.get("train_duration_sec", train_time_estimates.get(cid, 1.0))
                )
                logger.info(
                    f"{receive_time} RECV round={round_idx} token={token_id} cid={cid} "
                    f"train_duration={msg.get('train_duration_sec')} "
                    f"task_latency={task_latency_sec:.3f}s "
                    f"deadline_miss={bool(deadline_sec > 0.0 and task_latency_sec > deadline_sec)} "
                    f"battery={battery_state_joules[cid]:.2f}J"
                )
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": token_id})
            elif msg_type == "status" and msg.get("status") == "low_battery":
                active_clients.discard(cid)
                battery_state_joules[cid] = float(msg.get("battery_joules", 0.0))
                status_records.append({
                    "cid": cid,
                    "token_id": token_id,
                    "status": "low_battery",
                    "task_latency_sec": (
                        time.perf_counter() - task_start_perf_by_token[token_id]
                        if token_id in task_start_perf_by_token else None
                    ),
                })
                logger.warning(f"{now_str()} LOW_BATTERY cid={cid} token={token_id}")
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": token_id})
            elif msg_type == "client_error":
                active_clients.discard(cid)
                status_records.append({
                    "cid": cid,
                    "token_id": token_id,
                    "status": "client_error",
                    "reason": msg.get("reason"),
                    "task_latency_sec": (
                        time.perf_counter() - task_start_perf_by_token[token_id]
                        if token_id in task_start_perf_by_token else None
                    ),
                })
                logger.warning(f"{now_str()} CLIENT_ERROR cid={cid} token={token_id} reason={msg.get('reason')}")
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": token_id})
            else:
                logger.warning(f"Unexpected message from cid={cid}: {msg}")

        train_records = []
        for token_id, (cid, msg, _, receive_time, task_latency_sec) in token_updates.items():
            send_record = send_record_by_token[token_id]
            model_tokens[token_id].load_state_dict(msg["net"], strict=True)
            train_records.append(
                {
                    "token_id": token_id,
                    "cid": cid,
                    "num_samples": int(msg.get("num_samples", 0)),
                    "loss": float(msg.get("loss", 0.0)),
                    "train_start_time": msg.get("train_start_time"),
                    "train_end_time": msg.get("train_end_time"),
                    "train_duration_sec": float(msg.get("train_duration_sec", 0.0)),
                    "server_receive_time": receive_time,
                    "server_send_duration_sec": send_record["send_duration_sec"],
                    "client_recv_duration_sec": float(msg.get("recv_duration_sec", 0.0)),
                    "task_latency_sec": task_latency_sec,
                    "deadline_sec": deadline_sec,
                    "deadline_miss": bool(
                        deadline_sec > 0.0 and task_latency_sec > deadline_sec
                    ),
                    "battery_before": msg.get("battery_before"),
                    "battery_after": msg.get("battery_joules"),
                }
            )

        token_moves = []
        for token_id, current_cid in enumerate(locations_before):
            if token_id not in token_updates:
                if current_cid not in active_clients:
                    token_locations[token_id] = relocate_if_inactive([current_cid], active_clients, rng)[0]
                continue
            next_cid, policy_meta = policy.choose_next(
                token_id=token_id,
                current_cid=current_cid,
                round_idx=round_idx,
                model_distribution=model_distributions,
                label_distributions=label_distributions,
                last_visit_round=last_visit_round,
                train_time_estimates=train_time_estimates,
                active_clients=active_clients,
            )
            if next_cid != current_cid:
                model_distributions[token_id] = model_distributions[token_id] + label_distributions[next_cid]
            last_visit_round[next_cid] = round_idx + 1
            token_locations[token_id] = next_cid
            token_moves.append(
                {
                    "token_id": token_id,
                    "src": current_cid,
                    "dst": next_cid,
                    "policy": policy_meta,
                }
            )

        round_train_end_ts = time.time()
        round_train_end_perf = time.perf_counter()
        round_train_end_time = now_str()
        token_accs = [evaluate_model(model_tokens[token_id], dataset_test, args) for token_id in range(token_count)]
        avg_acc = sum(token_accs) / max(len(token_accs), 1)
        best_acc = max(best_acc, avg_acc)
        round_end_ts = time.time()
        round_end_time = now_str()

        record = {
            "round": round_idx,
            "round_start_time": round_start_time,
            "round_train_end_time": round_train_end_time,
            "round_end_time": round_end_time,
            "round_train_wall_time": round_train_end_ts - round_start_ts,
            "round_critical_path_sec": round_train_end_perf - round_start_perf,
            "deadline_sec": deadline_sec,
            "round_deadline_miss": bool(
                deadline_sec > 0.0 and round_train_end_perf - round_start_perf > deadline_sec
            ),
            "round_wall_time": round_end_ts - round_start_ts,
            "eval_duration_sec": round_end_ts - round_train_end_ts,
            "token_locations_before": locations_before,
            "token_locations_after": list(token_locations),
            "active_clients": sorted(active_clients),
            "send_records": send_records,
            "train_records": train_records,
            "status_records": status_records,
            "token_moves": token_moves,
            "comm_mb": comm_mb,
            "token_accs": token_accs,
            "avg_acc": avg_acc,
            "best_acc_so_far": best_acc,
            "battery_remaining_by_client": dict(battery_state_joules),
        }
        summary_records.append(record)
        deadline_summary = summarize_deadlines(
            summary_records,
            deadline_sec,
            deadline_sensitivity_secs,
        )
        round_writer.write(record)
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            round_idx=round_idx,
            model_tokens=model_tokens,
            token_locations=token_locations,
            model_distributions=model_distributions,
            last_visit_round=last_visit_round,
            active_clients=active_clients,
            battery_state_joules=battery_state_joules,
            best_acc=best_acc,
            summary_records=summary_records,
            train_time_estimates=train_time_estimates,
            rng=rng,
            policy=policy,
        )
        primary_deadline = deadline_summary.get("primary") or {}
        logger.info(
            f"{round_end_time} ROUND={round_idx} end avg_acc={avg_acc:.4f} "
            f"best_acc={best_acc:.4f} duration={record['round_wall_time']:.2f}s "
            f"critical_path={record['round_critical_path_sec']:.2f}s "
            f"task_miss={sum(item['deadline_miss'] for item in train_records)}/{len(train_records)} "
            f"cumulative_task_dmr={primary_deadline.get('task_miss_rate', 0.0):.4%} "
            f"round_miss={record['round_deadline_miss']} "
            f"cumulative_round_dmr={primary_deadline.get('round_miss_rate', 0.0):.4%} "
            f"comm_mb={comm_mb:.3f} locations_after={token_locations} checkpoint={checkpoint_path}"
        )

    for cid in range(num_clients):
        try:
            connect_handler.sendData(cid, {"type": "stop"})
        except Exception:
            pass

    summary_path = log_dir / f"summary_{run_label}_{timestamp}.json"
    deadline_summary = summarize_deadlines(
        summary_records,
        deadline_sec,
        deadline_sensitivity_secs,
    )
    summary = {
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "frac": args.frac,
        "token_count": token_count,
        "local_ep": args.local_ep,
        "seed": args.seed,
        "iid": args.iid,
        "noniid_case": args.noniid_case,
        "data_beta": args.data_beta,
        "device_map": {cid: get_device_type(cid) for cid in range(num_clients)},
        "low_battery_threshold_j": LOW_BATTERY_THRESHOLD_J,
        "checkpoint_path": str(checkpoint_path),
        "start_round": start_round,
        "records": summary_records,
        "best_acc": best_acc,
        "last_acc": summary_records[-1]["avg_acc"] if summary_records else 0.0,
        "deadline": deadline_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"FINAL_DEADLINE_SUMMARY {json.dumps(deadline_summary, ensure_ascii=False)}")
    logger.info(f"Saved summary to {summary_path}")
