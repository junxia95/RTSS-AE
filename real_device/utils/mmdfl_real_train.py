import copy
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet8_mmdfl import build_resnet8
from utils.FL_utils import Accumulator, accuracy
from utils.get_dataset import DatasetSplit


def build_model(args):
    return build_resnet8(num_classes=args.num_classes, in_channels=args.num_channels)


def state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def model_logits(output):
    if isinstance(output, dict):
        return output["output"]
    return output


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    if len(state_dicts) == 1:
        return copy.deepcopy(state_dicts[0])

    averaged = {}
    keys = state_dicts[0].keys()
    for key in keys:
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            averaged[key] = first.clone()
            continue
        stacked = torch.stack([state[key].detach().float() for state in state_dicts], dim=0)
        averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
    return averaged


def weighted_average_state_dicts(state_dicts, weights):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights must have the same length")
    if len(state_dicts) == 1:
        return copy.deepcopy(state_dicts[0])

    float_weights = [max(float(weight), 0.0) for weight in weights]
    total_weight = sum(float_weights)
    if total_weight <= 0.0:
        return average_state_dicts(state_dicts)

    averaged = {}
    keys = state_dicts[0].keys()
    for key in keys:
        first = state_dicts[0][key]
        if not torch.is_floating_point(first):
            averaged[key] = first.clone()
            continue
        acc = torch.zeros_like(first, dtype=torch.float32)
        for state, weight in zip(state_dicts, float_weights):
            if weight <= 0.0:
                continue
            acc += state[key].detach().float() * (weight / total_weight)
        averaged[key] = acc.to(dtype=first.dtype)
    return averaged


def train_local_model(model, dataset_train, idxs, args, round_idx, lr=None):
    model = model.to(args.device)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    loader = DataLoader(
        DatasetSplit(dataset_train, idxs),
        batch_size=args.local_bs,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr if lr is None else lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    total_loss = 0.0
    batch_count = 0
    start_ts = time.time()
    total_epochs = int(args.local_ep)
    for epoch in range(total_epochs):
        progress = tqdm(
            loader,
            desc=f"CID {args.CID} Round {round_idx} Epoch {epoch + 1}/{total_epochs}",
            unit="batch",
            file=sys.stdout,
        )
        for images, labels in progress:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            loss = loss_func(model_logits(model(images)), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batch_count += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
    duration = time.time() - start_ts
    return {
        "state_dict": state_dict_to_cpu(model.state_dict()),
        "loss": total_loss / max(batch_count, 1),
        "duration_sec": duration,
        "num_batches": batch_count,
        "num_samples": len(idxs),
    }


def evaluate_model(model, dataset_test, args):
    eval_model = copy.deepcopy(model).to(args.device)
    eval_model.eval()
    loader = DataLoader(dataset_test, batch_size=args.bs, shuffle=False, num_workers=0)
    metric = Accumulator(2)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            logits = model_logits(eval_model(images))
            metric.add(accuracy(logits, labels), labels.numel())
    return 100.0 * metric[0] / max(metric[1], 1)


def label_distribution(dataset, idxs, num_classes):
    targets = np.asarray(dataset.targets)
    selected_targets = targets[list(idxs)]
    counts = np.zeros(int(num_classes), dtype=float)
    for label in selected_targets:
        counts[int(label)] += 1.0
    return counts
