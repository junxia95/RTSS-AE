import copy
import json
import random
import time

import torch
from loguru import logger

from utils.ConnectHandler_server import ConnectHandler
from utils.get_dataset import get_dataset
from utils.main_real_profiles import get_device_type
from utils.mmdfl_real_common import apply_real_defaults, checkpoint_path, pickle_size_mb
from utils.mmdfl_real_log import JsonlWriter, make_log_dir, now_str
from utils.mmdfl_real_train import (
    build_model,
    evaluate_model,
    state_dict_to_cpu,
    weighted_average_state_dicts,
)
from utils.mmdfl_topology import get_neighbors
from utils.options import args_parser
from utils.power_manager_real import LOW_BATTERY_THRESHOLD_J, get_device_capacity
from utils.rtss_deadline_metrics import parse_deadline_sensitivity, summarize_deadlines
from utils.set_seed import set_random_seed


ALGORITHM = "DFL"


def client_capacity_map(num_clients):
    return {cid: get_device_capacity(get_device_type(cid)) for cid in range(num_clients)}


def select_round_robin(active_clients, cursor, count, num_clients):
    selected = []
    active = set(int(cid) for cid in active_clients)
    if not active:
        return selected, cursor
    idx = int(cursor) % int(num_clients)
    visited = 0
    while len(selected) < int(count) and visited < int(num_clients) * 2:
        cid = idx % int(num_clients)
        if cid in active and cid not in selected:
            selected.append(cid)
        idx += 1
        visited += 1
    return selected, idx % int(num_clients)


def build_train_payload(args, cid, round_idx, model, idxs_list, battery_state_joules):
    return {
        "type": "train_round",
        "round": int(round_idx),
        "token_id": int(cid),
        "algorithm": ALGORITHM,
        "net": state_dict_to_cpu(model.state_dict()),
        "idxs_list": list(idxs_list),
        "local_ep": int(args.local_ep),
        "local_bs": int(args.local_bs),
        "lr": float(args.lr * (args.lr_decay ** round_idx)),
        "server_known_battery_joules": float(battery_state_joules),
    }


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_checkpoint(
    path,
    args,
    round_idx,
    client_models,
    active_clients,
    cursor,
    battery_state_joules,
    best_acc,
    summary_records,
    train_time_estimates,
    rng,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "algorithm": ALGORITHM,
            "saved_at": now_str(),
            "completed_round": int(round_idx),
            "next_round": int(round_idx) + 1,
            "num_users": int(args.num_users),
            "seed": int(args.seed),
            "client_model_state_dicts": [
                state_dict_to_cpu(model.state_dict()) for model in client_models
            ],
            "active_clients": sorted(active_clients),
            "cursor": int(cursor),
            "battery_state_joules": dict(battery_state_joules),
            "best_acc": float(best_acc),
            "summary_records": copy.deepcopy(summary_records),
            "train_time_estimates": dict(train_time_estimates),
            "rng_state": rng.getstate(),
        },
        path,
    )


if __name__ == "__main__":
    args = apply_real_defaults(args_parser(), ALGORITHM)
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

    log_dir = make_log_dir("logs_real", "dfl_real_server")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.add(log_dir / f"server_dfl_real_{args.log_tag or 'seed' + str(args.seed)}_{timestamp}.log")
    round_writer = JsonlWriter(log_dir / f"rounds_{args.log_tag or 'seed' + str(args.seed)}_{timestamp}.jsonl")
    ckpt_path = checkpoint_path(log_dir, ALGORITHM, args)

    num_clients = int(args.num_users)
    train_count = max(int(num_clients * float(args.frac)), 1)
    active_clients = set(range(num_clients))
    battery_state_joules = client_capacity_map(num_clients)
    train_time_estimates = {
        cid: 1.0 if get_device_type(cid) == "orinnanosuper" else (1.5 if get_device_type(cid) == "agx_xavier" else 2.5)
        for cid in range(num_clients)
    }
    best_acc = 0.0
    summary_records = []
    cursor = 0
    start_round = 0

    logger.info(
        f"Starting fair sampled DFL real server device={args.device} dataset={args.dataset} "
        f"model={args.model} beta={args.data_beta} users={num_clients} train_count={train_count} seed={args.seed} "
        f"deadline_sec={deadline_sec} deadline_mode=observe_only "
        f"deadline_sensitivity_secs={deadline_sensitivity_secs}"
    )

    dataset_train, dataset_test, dict_users = get_dataset(args)
    base_model = build_model(args)
    client_models = [copy.deepcopy(base_model) for _ in range(num_clients)]

    if ckpt_path.exists() and not int(getattr(args, "fresh_start", 0)):
        checkpoint = load_checkpoint(ckpt_path)
        if int(checkpoint.get("num_users", -1)) != num_clients:
            raise ValueError("checkpoint num_users does not match current settings")
        for cid, state in enumerate(checkpoint["client_model_state_dicts"]):
            client_models[cid].load_state_dict(state, strict=True)
        active_clients = set(int(cid) for cid in checkpoint["active_clients"])
        cursor = int(checkpoint.get("cursor", 0))
        battery_state_joules = {
            int(cid): float(value) for cid, value in checkpoint["battery_state_joules"].items()
        }
        best_acc = float(checkpoint.get("best_acc", 0.0))
        summary_records = list(checkpoint.get("summary_records", []))
        train_time_estimates.update({
            int(cid): float(value) for cid, value in checkpoint.get("train_time_estimates", {}).items()
        })
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            rng.setstate(rng_state)
        start_round = int(checkpoint.get("next_round", int(checkpoint.get("completed_round", -1)) + 1))
        logger.info(f"Resumed DFL server from checkpoint={ckpt_path} start_round={start_round}")
    else:
        logger.info(f"No checkpoint restored for DFL at {ckpt_path}; fresh_start={args.fresh_start}")

    connect_handler = ConnectHandler(num_clients, args.HOST, args.POST)

    for round_idx in range(start_round, int(args.epochs)):
        if not active_clients:
            logger.warning("No active clients left, stopping.")
            break

        round_start_ts = time.time()
        round_start_perf = time.perf_counter()
        round_start_time = now_str()
        selected_clients, cursor = select_round_robin(active_clients, cursor, train_count, num_clients)
        logger.info(
            f"{round_start_time} ROUND={round_idx} start selected={selected_clients} "
            f"active_clients={sorted(active_clients)} cursor={cursor}"
        )

        expected = set()
        send_records = []
        send_record_by_cid = {}
        task_start_perf_by_cid = {}
        comm_mb = 0.0
        for cid in selected_clients:
            payload = build_train_payload(
                args=args,
                cid=cid,
                round_idx=round_idx,
                model=client_models[cid],
                idxs_list=dict_users[cid],
                battery_state_joules=battery_state_joules[cid],
            )
            payload_mb = pickle_size_mb(payload)
            send_start = time.time()
            send_start_perf = time.perf_counter()
            ok = connect_handler.sendData(cid, payload)
            send_duration = time.perf_counter() - send_start_perf
            if not ok:
                logger.warning(f"Failed sending round={round_idx} to cid={cid}")
                active_clients.discard(cid)
                continue
            expected.add(cid)
            comm_mb += payload_mb
            send_record = {
                "cid": cid,
                "server_send_start_time": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(send_start),
                ),
                "send_duration_sec": send_duration,
                "payload_mb": payload_mb,
            }
            send_records.append(send_record)
            send_record_by_cid[cid] = send_record
            task_start_perf_by_cid[cid] = send_start_perf
            logger.info(
                f"{now_str()} SEND round={round_idx} cid={cid} "
                f"payload_mb={payload_mb:.3f} duration={send_duration:.3f}s"
            )

        updates = {}
        status_records = []
        while len(updates) + len(status_records) < len(expected):
            msg, cid = connect_handler.receiveData()
            msg_type = msg.get("type")
            if msg_type == "client_update" and int(msg.get("round", -1)) == round_idx and cid in expected:
                receive_time = now_str()
                receive_ts = time.time()
                receive_perf = time.perf_counter()
                task_latency_sec = receive_perf - task_start_perf_by_cid[cid]
                comm_mb += pickle_size_mb(msg)
                updates[cid] = (msg, receive_ts, receive_time, task_latency_sec)
                battery_state_joules[cid] = float(msg.get("battery_joules", battery_state_joules.get(cid, 0.0)))
                train_time_estimates[cid] = 0.7 * train_time_estimates.get(cid, 1.0) + 0.3 * float(
                    msg.get("train_duration_sec", train_time_estimates.get(cid, 1.0))
                )
                logger.info(
                    f"{receive_time} RECV round={round_idx} cid={cid} "
                    f"train_duration={msg.get('train_duration_sec')} "
                    f"task_latency={task_latency_sec:.3f}s "
                    f"deadline_miss={bool(deadline_sec > 0.0 and task_latency_sec > deadline_sec)} "
                    f"battery={battery_state_joules[cid]:.2f}J"
                )
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": cid})
            elif msg_type == "status" and msg.get("status") == "low_battery":
                active_clients.discard(cid)
                battery_state_joules[cid] = float(msg.get("battery_joules", 0.0))
                status_records.append({
                    "cid": cid,
                    "status": "low_battery",
                    "task_latency_sec": (
                        time.perf_counter() - task_start_perf_by_cid[cid]
                        if cid in task_start_perf_by_cid else None
                    ),
                })
                logger.warning(f"{now_str()} LOW_BATTERY cid={cid}")
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": cid})
            elif msg_type == "client_error":
                active_clients.discard(cid)
                status_records.append({
                    "cid": cid,
                    "status": "client_error",
                    "reason": msg.get("reason"),
                    "task_latency_sec": (
                        time.perf_counter() - task_start_perf_by_cid[cid]
                        if cid in task_start_perf_by_cid else None
                    ),
                })
                logger.warning(f"{now_str()} CLIENT_ERROR cid={cid} reason={msg.get('reason')}")
                connect_handler.sendData(cid, {"type": "upload_ack", "round": round_idx, "token_id": cid})
            else:
                logger.warning(f"Unexpected message from cid={cid}: {msg}")

        for cid, (msg, _, _, _) in updates.items():
            client_models[cid].load_state_dict(msg["net"], strict=True)

        snapshot_states = [state_dict_to_cpu(model.state_dict()) for model in client_models]
        aggregate_records = []
        for cid in sorted(updates):
            neighbors = get_neighbors(cid, active_clients=active_clients, num_users=num_clients)
            participants = [cid] + [nb for nb in neighbors if nb in active_clients]
            states = [snapshot_states[idx] for idx in participants]
            weights = [len(dict_users[idx]) for idx in participants]
            aggregated = weighted_average_state_dicts(states, weights)
            client_models[cid].load_state_dict(aggregated, strict=True)
            aggregate_records.append({"cid": cid, "participants": participants, "weights": weights})

        round_train_end_ts = time.time()
        round_train_end_perf = time.perf_counter()
        round_train_end_time = now_str()
        accs = [evaluate_model(client_models[cid], dataset_test, args) for cid in range(num_clients)]
        avg_acc = sum(accs) / max(len(accs), 1)
        best_acc = max(best_acc, avg_acc)
        round_end_ts = time.time()
        round_end_time = now_str()

        train_records = []
        for cid, (msg, _, receive_time, task_latency_sec) in updates.items():
            send_record = send_record_by_cid[cid]
            train_records.append(
                {
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
            "selected_clients": selected_clients,
            "active_clients": sorted(active_clients),
            "send_records": send_records,
            "train_records": train_records,
            "status_records": status_records,
            "aggregate_records": aggregate_records,
            "comm_mb": comm_mb,
            "client_accs": accs,
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
            path=ckpt_path,
            args=args,
            round_idx=round_idx,
            client_models=client_models,
            active_clients=active_clients,
            cursor=cursor,
            battery_state_joules=battery_state_joules,
            best_acc=best_acc,
            summary_records=summary_records,
            train_time_estimates=train_time_estimates,
            rng=rng,
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
            f"comm_mb={comm_mb:.3f} selected={selected_clients} checkpoint={ckpt_path}"
        )

    for cid in range(num_clients):
        try:
            connect_handler.sendData(cid, {"type": "stop"})
        except Exception:
            pass

    summary_path = log_dir / f"summary_{args.log_tag or 'seed' + str(args.seed)}_{timestamp}.json"
    deadline_summary = summarize_deadlines(
        summary_records,
        deadline_sec,
        deadline_sensitivity_secs,
    )
    summary = {
        "algorithm": ALGORITHM,
        "dataset": args.dataset,
        "model": args.model,
        "num_users": args.num_users,
        "frac": args.frac,
        "train_count": train_count,
        "local_ep": args.local_ep,
        "seed": args.seed,
        "iid": args.iid,
        "noniid_case": args.noniid_case,
        "data_beta": args.data_beta,
        "device_map": {cid: get_device_type(cid) for cid in range(num_clients)},
        "low_battery_threshold_j": LOW_BATTERY_THRESHOLD_J,
        "checkpoint_path": str(ckpt_path),
        "start_round": start_round,
        "records": summary_records,
        "best_acc": best_acc,
        "last_acc": summary_records[-1]["avg_acc"] if summary_records else 0.0,
        "deadline": deadline_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"FINAL_DEADLINE_SUMMARY {json.dumps(deadline_summary, ensure_ascii=False)}")
    logger.info(f"Saved summary to {summary_path}")
