import subprocess
import time
from pathlib import Path

import torch
from loguru import logger

from utils.ConnectHandler_client import ConnectHandler
from utils.get_dataset import get_dataset
from utils.main_real_profiles import get_device_type
from utils.mmdfl_real_log import JsonlWriter, make_log_dir, now_str
from utils.mmdfl_real_train import build_model, train_local_model
from utils.options import args_parser
from utils.power_manager_real import BatteryManagerReal, LOW_BATTERY_THRESHOLD_J
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


def apply_real_defaults(args):
    cli_algorithm = str(args.algorithm)
    for key, value in REAL_DEFAULTS.items():
        setattr(args, key, value)
    if cli_algorithm not in {"", "heterofl"}:
        args.algorithm = cli_algorithm
    return args


def upload_status(connect_handler, payload, battery_manager, mode_label="low"):
    upload_start_ts = time.time()
    connect_handler.uploadToServer(payload)
    upload_duration = time.time() - upload_start_ts
    battery_manager.consume("communication", upload_duration, mode_label=mode_label)
    return upload_duration


if __name__ == "__main__":
    args = apply_real_defaults(args_parser())
    args.device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu"
    )
    set_random_seed(int(args.seed) + int(args.CID))

    device_type = get_device_type(args.CID)
    mode_label = "low"
    battery_manager = BatteryManagerReal(device_type=device_type, initial_mode_label=mode_label)

    log_dir = make_log_dir("logs_real", "mmdfl_real_client")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger.add(log_dir / f"client_mmdfl_real_{args.CID}_{timestamp}.log")
    event_writer = JsonlWriter(log_dir / f"client_{args.CID}_{timestamp}.jsonl")

    logger.info(
        f"Starting MMDFL real client cid={args.CID} device_type={device_type} "
        f"algorithm={args.algorithm} dataset={args.dataset} gpu={args.gpu}"
    )

    dataset_train, _, _ = get_dataset(args)
    connect_handler = ConnectHandler(args.HOST, args.POST, args.CID)
    last_activity_ts = time.time()

    while True:
        idle_duration = time.time() - last_activity_ts
        if idle_duration > 0:
            battery_manager.consume("idle", idle_duration, mode_label=mode_label)

        recv_start_ts = time.time()
        recv = connect_handler.receiveFromServer()
        recv_duration = time.time() - recv_start_ts
        battery_manager.consume("communication", recv_duration, mode_label=mode_label)

        if not recv:
            logger.warning("Received empty payload, continue.")
            last_activity_ts = time.time()
            continue

        if "server_known_battery_joules" in recv:
            battery_manager.set_charge(recv["server_known_battery_joules"])
        elif "battery_joules" in recv:
            battery_manager.set_charge(recv["battery_joules"])
        elif "battery_level" in recv:
            battery_manager.set_charge(float(recv["battery_level"]) * battery_manager.total_capacity)

        msg_type = recv.get("type")
        if msg_type == "stop":
            logger.info("Received stop signal from server.")
            break
        if msg_type != "train_round":
            logger.warning(f"Unknown message type: {msg_type}")
            last_activity_ts = time.time()
            continue

        round_idx = int(recv["round"])
        token_id = int(recv["token_id"])
        idxs_list = recv["idxs_list"]
        lr = float(recv.get("lr", args.lr))
        battery_before = battery_manager.get_charge()

        if not battery_manager.check_energy(LOW_BATTERY_THRESHOLD_J):
            logger.warning(f"cid={args.CID} battery too low before training, shutting down.")
            upload_status(
                connect_handler,
                {
                    "type": "status",
                    "status": "low_battery",
                    "cid": args.CID,
                    "round": round_idx,
                    "token_id": token_id,
                    "device_type": device_type,
                    "battery_joules": battery_manager.get_charge(),
                    "battery_level": battery_manager.get_ratio(),
                },
                battery_manager,
                mode_label=mode_label,
            )
            ack = connect_handler.receiveFromServer()
            logger.info(f"Shutdown ack={ack}")
            subprocess.run(["sudo", "poweroff"], check=False)
            break

        try:
            local_model = build_model(args)
            local_model.load_state_dict(recv["net"], strict=True)
            train_start_time = now_str()
            train_result = train_local_model(
                model=local_model,
                dataset_train=dataset_train,
                idxs=idxs_list,
                args=args,
                round_idx=round_idx,
                lr=lr,
            )
            train_end_time = now_str()
            battery_manager.consume("train", train_result["duration_sec"], mode_label=mode_label)
        except Exception as exc:
            logger.exception(f"Training failed cid={args.CID} round={round_idx} token={token_id}: {exc}")
            upload_status(
                connect_handler,
                {
                    "type": "client_error",
                    "cid": args.CID,
                    "round": round_idx,
                    "token_id": token_id,
                    "reason": repr(exc),
                    "battery_joules": battery_manager.get_charge(),
                    "battery_level": battery_manager.get_ratio(),
                },
                battery_manager,
                mode_label=mode_label,
            )
            connect_handler.receiveFromServer()
            last_activity_ts = time.time()
            continue

        upload_start_time = now_str()
        upload_start_ts = time.time()
        update_payload = {
            "type": "client_update",
            "cid": args.CID,
            "round": round_idx,
            "token_id": token_id,
            "net": train_result["state_dict"],
            "num_samples": train_result["num_samples"],
            "num_batches": train_result["num_batches"],
            "loss": train_result["loss"],
            "device_type": device_type,
            "battery_before": battery_before,
            "battery_joules": battery_manager.get_charge(),
            "battery_level": battery_manager.get_ratio(),
            "recv_duration_sec": recv_duration,
            "train_start_time": train_start_time,
            "train_end_time": train_end_time,
            "train_duration_sec": train_result["duration_sec"],
            "upload_start_time": upload_start_time,
        }
        connect_handler.uploadToServer(update_payload)
        upload_duration = time.time() - upload_start_ts
        upload_end_time = now_str()
        battery_manager.consume("communication", upload_duration, mode_label=mode_label)
        ack_recv_start = time.time()
        ack = connect_handler.receiveFromServer()
        ack_recv_duration = time.time() - ack_recv_start
        battery_manager.consume("communication", ack_recv_duration, mode_label=mode_label)

        event = {
            "cid": int(args.CID),
            "round": round_idx,
            "token_id": token_id,
            "train_start_time": train_start_time,
            "train_end_time": train_end_time,
            "upload_start_time": upload_start_time,
            "upload_end_time": upload_end_time,
            "recv_duration_sec": recv_duration,
            "train_duration_sec": train_result["duration_sec"],
            "upload_duration_sec": upload_duration,
            "ack_recv_duration_sec": ack_recv_duration,
            "num_samples": train_result["num_samples"],
            "loss": train_result["loss"],
            "battery_before": battery_before,
            "battery_after": battery_manager.get_charge(),
            "ack": ack,
        }
        event_writer.write(event)
        logger.info(
            f"{now_str()} ROUND={round_idx} cid={args.CID} token={token_id} "
            f"samples={train_result['num_samples']} loss={train_result['loss']:.4f} "
            f"train={train_result['duration_sec']:.2f}s upload={upload_duration:.2f}s "
            f"battery={battery_manager.get_charge():.2f}J ack={ack}"
        )
        last_activity_ts = time.time()
