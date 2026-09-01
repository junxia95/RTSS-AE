#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import os
import random
from collections import Counter, defaultdict, deque

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn

from config import *
from utils.energy import consume_communication_energy, consume_energy, consume_sleep_energy, initialize_battery_state, snapshot_battery_state
from utils.FL_utils import *
from utils.FL_utils import DataLoader
from utils.quantization import (
    apply_conv_linear_qat_forward,
    effective_quant_bits,
    get_quant_comm_ratio,
    maybe_quantize_batch,
    project_model_to_quantization,
    quantized_state_dict,
    restore_conv_linear_qat_forward,
    state_dict_payload_nbytes,
    state_dict_weighted_mean_bits,
    transmit_state_dict,
)
from utils.utils import save_result

MIN_QUANT_BITS = 8
RUNTIME_CHECKPOINT_NAME = 'autorl_runtime_checkpoint.pt'


def _runtime_checkpoint_signature(args):
    keys = (
        'algorithm', 'dataset', 'model', 'epochs', 'num_users', 'frac',
        'local_ep', 'local_bs', 'lr', 'lr_decay', 'seed', 'iid',
        'noniid_case', 'data_beta', 'partition_tag', 'experiment_tag',
    )
    return {key: getattr(args, key, None) for key in keys}


def _model_state_list(models):
    return [
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        for model in models
    ]


def _load_model_state_list(models, state_list):
    if len(models) != len(state_list):
        raise RuntimeError('runtime checkpoint model count does not match this run')
    for model, state in zip(models, state_list):
        model.load_state_dict(state)


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_torch_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = '{}.tmp.{}'.format(path, os.getpid())
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _autorl_battery_energy_scale(args_):
    return max(float(getattr(args_, 'autorl_battery_energy_scale', 1.0)), 0.0)


def _apply_autorl_battery_capacity_scale(battery_state, args_):
    s = float(getattr(args_, 'autorl_battery_capacity_scale', 1.0))
    if abs(s - 1.0) < 1e-15:
        return
    for b in battery_state:
        c = float(b['capacity_j']) * s
        b['capacity_j'] = c
        b['remaining_j'] = c


def _safe_normalize(vector):
    vector = np.asarray(vector, dtype=float)
    total = vector.sum()
    if total <= 0:
        return np.zeros_like(vector)
    return vector / total


def _coverage_gap(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.sum() <= 0:
        return 1.0
    normalized = _safe_normalize(vector)
    uniform = np.ones_like(normalized) / max(len(normalized), 1)
    return float(np.linalg.norm(normalized - uniform))


def _entropy_score(vector):
    normalized = _safe_normalize(vector)
    if normalized.sum() <= 0:
        return 0.0
    entropy = -np.sum(np.where(normalized > 0, normalized * np.log(normalized + 1e-12), 0.0))
    return float(entropy / max(np.log(max(len(normalized), 2)), 1e-12))


def _deficit_vector(memory):
    memory = np.asarray(memory, dtype=float)
    if len(memory) == 0:
        return memory
    normalized = _safe_normalize(memory)
    if normalized.sum() <= 0:
        return np.ones_like(memory) / max(len(memory), 1)
    target = np.ones_like(normalized) / max(len(normalized), 1)
    deficit = np.maximum(target - normalized, 0.0)
    if deficit.sum() <= 0:
        return np.zeros_like(deficit)
    return deficit / deficit.sum()


def _novelty_score(memory, candidate_signature):
    """Ratio of new categories in candidate that are absent in memory."""
    memory = np.asarray(memory, dtype=float)
    candidate_signature = np.asarray(candidate_signature, dtype=float)
    if candidate_signature.sum() <= 0:
        return 0.0
    absent = (memory <= 0)
    present_in_candidate = (candidate_signature > 0)
    new_categories = np.sum(absent & present_in_candidate)
    total_candidate = np.sum(present_in_candidate)
    return float(new_categories / max(total_candidate, 1))


def _decay_compensation(memory, candidate_signature, threshold=1e-6):
    """How much the candidate signature compensates for categories that have decayed in memory."""
    memory = np.asarray(memory, dtype=float)
    candidate_signature = np.asarray(candidate_signature, dtype=float)
    if candidate_signature.sum() <= 0:
        return 0.0
    decayed = (memory > 0) & (memory <= threshold * memory.max())
    compensated = np.sum(candidate_signature * decayed)
    total_candidate = candidate_signature.sum()
    return float(compensated / max(total_candidate, 1))


def _parse_quant_bits(raw_bits, default_bits, min_bits=MIN_QUANT_BITS):
    if isinstance(raw_bits, str):
        values = [item.strip() for item in raw_bits.split(',') if item.strip()]
        bits = [int(value) for value in values]
    else:
        bits = [int(value) for value in raw_bits]
    bits = sorted(set(bit for bit in bits if bit >= int(min_bits)))
    return bits or [max(int(default_bits), int(min_bits))]


def _bit_counts(values, options):
    counts = Counter(int(value) for value in values)
    total = float(sum(counts.values()))
    return {
        'counts': {int(bit): int(counts.get(int(bit), 0)) for bit in options},
        'ratio': {int(bit): float(counts.get(int(bit), 0) / total) if total > 0 else 0.0 for bit in options},
    }


def _mean_weight_records(records):
    if not records:
        return {'accuracy': 0.0, 'energy': 0.0, 'latency': 0.0}
    return {
        key: float(np.mean([record.get(key, 0.0) for record in records]))
        for key in ['accuracy', 'energy', 'latency']
    }


def _mean_train_policy(records, options):
    if not records:
        return {
            'probabilities': {int(bit): 0.0 for bit in options},
            'accuracy_pressure': 0.0,
            'energy_pressure': 0.0,
        }
    return {
        'probabilities': {
            int(bit): float(np.mean([
                record.get('probabilities', {}).get(int(bit), 0.0)
                for record in records
            ]))
            for bit in options
        },
        'accuracy_pressure': float(np.mean([record.get('accuracy_pressure', 0.0) for record in records])),
        'energy_pressure': float(np.mean([record.get('energy_pressure', 0.0) for record in records])),
    }


def _finite_mean(values, default=0.0):
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return float(default)
    return float(np.mean(finite))


def _round_proxy_score(controller, train_signal_set, round_energy, round_time, round_comm, battery_state, round_idx):
    """Deployable round-quality score. It uses no test/validation data."""
    signals = [signal for signal in train_signal_set if signal is not None]
    mean_loss = _finite_mean([signal.get('loss', 0.0) for signal in signals], default=0.0)
    mean_confidence = _finite_mean([signal.get('confidence', 0.0) for signal in signals], default=0.0)
    mean_entropy = _finite_mean([signal.get('entropy', 0.0) for signal in signals], default=1.0)
    loss_score = 1.0 / (1.0 + max(mean_loss, 0.0))
    confidence_score = float(np.clip(mean_confidence, 0.0, 1.0))
    entropy_score = float(np.clip(1.0 - mean_entropy, 0.0, 1.0))
    global_coverage = float(controller._global_node_coverage_ratio())
    model_coverage = (
        float(np.mean([controller._node_coverage_ratio(mid) for mid in range(controller.model_cnt)]))
        if controller.model_cnt
        else 0.0
    )
    forgetting_pressure = (
        float(np.mean([controller._forgetting_pressure(mid, round_idx) for mid in range(controller.model_cnt)]))
        if controller.model_cnt
        else 0.0
    )
    recent_rewards = controller.reward_history[-max(len(signals), 1):] if controller.reward_history else []
    reward_score = _finite_mean([record.get('reward', 0.0) for record in recent_rewards], default=0.0)
    reward_score = float(np.tanh(max(reward_score, 0.0)))
    total_capacity = sum(float(item.get('capacity_j', 0.0)) for item in battery_state)
    energy_cost = float(round_energy / max(total_capacity, 1e-12))
    latency_cost = float(round_time / max(round_time + 100.0, 1e-12))
    comm_cost = float(round_comm / max(round_comm + 100.0, 1e-12))
    score = (
        0.20 * global_coverage
        + 0.20 * model_coverage
        + 0.18 * loss_score
        + 0.14 * confidence_score
        + 0.10 * entropy_score
        + 0.18 * reward_score
        - 0.05 * forgetting_pressure
        - 0.03 * energy_cost
        - 0.015 * latency_cost
        - 0.005 * comm_cost
    )
    return float(score), {
        'score': float(score),
        'loss': float(mean_loss),
        'loss_score': float(loss_score),
        'confidence': float(mean_confidence),
        'confidence_score': float(confidence_score),
        'entropy': float(mean_entropy),
        'entropy_score': float(entropy_score),
        'global_node_coverage_ratio': float(global_coverage),
        'model_node_coverage_mean': float(model_coverage),
        'forgetting_pressure_mean': float(forgetting_pressure),
        'reward_score': float(reward_score),
        'energy_cost': float(energy_cost),
        'latency_cost': float(latency_cost),
        'comm_cost': float(comm_cost),
    }


def _parse_weighted_bits(raw_weights, options, default_floor=0.0):
    if isinstance(raw_weights, str):
        tokens = [item.strip() for item in raw_weights.split(',') if item.strip()]
        parsed = {}
        has_mapping = any(':' in token for token in tokens)
        if has_mapping:
            for token in tokens:
                if ':' not in token:
                    continue
                key_str, value_str = token.split(':', 1)
                parsed[int(key_str.strip())] = float(value_str.strip())
            weights = [float(parsed.get(int(bit), default_floor)) for bit in options]
        else:
            values = [float(token) for token in tokens]
            if len(values) == 1 and len(options) > 1:
                weights = values * len(options)
            else:
                weights = values[:len(options)]
    else:
        values = [float(value) for value in raw_weights]
        if len(values) == 1 and len(options) > 1:
            weights = values * len(options)
        else:
            weights = values[:len(options)]

    if len(weights) < len(options):
        weights = list(weights) + [float(default_floor)] * (len(options) - len(weights))
    weights = [max(float(weight), 0.0) for weight in weights]
    total = sum(weights)
    if total <= 0:
        weights = [1.0 / max(len(options), 1) for _ in options]
    else:
        weights = [weight / total for weight in weights]
    return {int(bit): float(weight) for bit, weight in zip(options, weights)}


def _parse_bit_float_map(raw_values, options, default_value=1.0, minimum=1e-12):
    values = {int(bit): float(default_value) for bit in options}
    if isinstance(raw_values, str):
        tokens = [item.strip() for item in raw_values.split(',') if item.strip()]
        if tokens and any(':' in token for token in tokens):
            for token in tokens:
                if ':' not in token:
                    continue
                key_str, value_str = token.split(':', 1)
                values[int(key_str.strip())] = float(value_str.strip())
        elif tokens:
            parsed = [float(token) for token in tokens]
            for bit, value in zip(options, parsed):
                values[int(bit)] = float(value)
    else:
        parsed = [float(value) for value in raw_values]
        for bit, value in zip(options, parsed):
            values[int(bit)] = float(value)
    return {int(bit): max(float(values.get(int(bit), default_value)), float(minimum)) for bit in options}


def _parse_layer_quant_rules(raw):
    """substr:delta 规则；同一参数名取最长匹配子串。空串表示使用 ResNet8 系默认分层。"""
    text = '' if raw is None else str(raw).strip()
    default = [
        ('outlayer', 8),
        ('layer1', 0),
        ('layer2', 0),
        ('blk2', -8),
        ('blk3', -8),
        ('blk4', -8),
    ]
    if not text:
        return list(default)
    rules = []
    for item in text.split(','):
        item = item.strip()
        if not item or ':' not in item:
            continue
        key, val = item.split(':', 1)
        rules.append((key.strip(), int(val.strip())))
    return rules if rules else list(default)


class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.args = args
        self.device_profile = get_client_device_profile(id)
        self.quant_bits = max(int(get_client_quant_bits(id)), MIN_QUANT_BITS)
        self.train_quant_bits = max(int(getattr(args, 'autorl_train_quant_bits', MIN_QUANT_BITS)), MIN_QUANT_BITS)
        self.feature_signature = np.zeros(args.num_classes)

    def build_feature_signature(self, dataset):
        signature = np.zeros(self.args.num_classes)
        loader = DataLoader(DatasetSplit(dataset, self.data_idx), batch_size=self.args.local_bs, shuffle=False)
        for _, labels in loader:
            for label in labels:
                label_idx = int(label.item()) if hasattr(label, 'item') else int(label)
                if 0 <= label_idx < self.args.num_classes:
                    signature[label_idx] += 1
        self.feature_signature = signature


class LocalUpdate_AutoRL(object):
    def __init__(self, args, dataset=None, quant_enabled=True):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.quant_enabled = quant_enabled

    def train(self, client, round, local_ep_scale=1.0, train_quant_bits=None, bits_policy=None):
        train_quant_bits = max(int(train_quant_bits or client.train_quant_bits), MIN_QUANT_BITS)
        q_policy = bits_policy if bits_policy is not None else train_quant_bits
        net = copy.deepcopy(client.local_net)
        net.train()
        net = net.to(self.args.device)
        prox_mu = max(float(getattr(self.args, 'autorl_prox_mu', 0.0)), 0.0)
        prox_start = float(getattr(self.args, 'autorl_prox_start_frac', 0.35))
        round_frac = float(round) / max(float(getattr(self.args, 'epochs', 1) - 1), 1.0)
        use_prox = prox_mu > 0.0 and round_frac >= prox_start
        prox_reference = {}
        if use_prox:
            prox_reference = {
                name: param.detach().clone()
                for name, param in net.named_parameters()
                if param.requires_grad
            }
        qat_forward = bool(int(getattr(self.args, 'autorl_qat_forward', 1))) and self.quant_enabled
        act_ch = bool(int(getattr(self.args, 'autorl_qat_act_channelwise', 0))) and self.quant_enabled
        comm_8bit_format = getattr(self.args, 'quant_comm_8bit_format', 'int8')
        restored = []
        if qat_forward:
            restored = apply_conv_linear_qat_forward(net, q_policy, True)
        else:
            project_model_to_quantization(net, q_policy, self.quant_enabled, comm_8bit_format=comm_8bit_format)
        loader = DataLoader(DatasetSplit(self.dataset, client.data_idx), batch_size=self.args.local_bs, shuffle=True)
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.args.lr * (self.args.lr_decay ** round),
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        loss_sum = 0.0
        confidence_sum = 0.0
        entropy_sum = 0.0
        batch_count = 0
        effective_local_ep = max(1, int(np.rint(self.args.local_ep * float(np.clip(local_ep_scale, 0.25, 3.0)))))
        for _ in range(effective_local_ep):
            for images, labels in loader:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                images = maybe_quantize_batch(images, train_quant_bits, self.quant_enabled, channelwise_act=act_ch)
                net.zero_grad()
                logits = net(images)['output']
                loss = self.loss_func(logits, labels)
                if use_prox:
                    prox_term = None
                    for name, param in net.named_parameters():
                        if not param.requires_grad:
                            continue
                        ref = prox_reference.get(name)
                        if ref is None:
                            continue
                        term = torch.sum((param - ref) ** 2)
                        prox_term = term if prox_term is None else prox_term + term
                    if prox_term is not None:
                        loss = loss + 0.5 * prox_mu * prox_term
                loss.backward()
                optimizer.step()
                if not qat_forward:
                    project_model_to_quantization(net, q_policy, self.quant_enabled, comm_8bit_format=comm_8bit_format)

                with torch.no_grad():
                    probabilities = torch.softmax(logits, dim=1)
                    confidence_sum += probabilities.max(dim=1)[0].mean().item()
                    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12), dim=1)
                    entropy_sum += (entropy / max(np.log(max(self.args.num_classes, 2)), 1e-12)).mean().item()
                    loss_sum += loss.item()
                    batch_count += 1

        restore_conv_linear_qat_forward(restored)
        if self.quant_enabled:
            project_model_to_quantization(net, q_policy, self.quant_enabled, comm_8bit_format=comm_8bit_format)

        eff_train_bits = float(state_dict_weighted_mean_bits(net, q_policy, self.quant_enabled))
        net = net.to('cpu')
        signal = {
            'loss': loss_sum / max(batch_count, 1),
            'confidence': confidence_sum / max(batch_count, 1),
            'entropy': entropy_sum / max(batch_count, 1),
            'local_ep_scale': float(local_ep_scale),
            'effective_local_ep': int(effective_local_ep),
            'train_quant_bits': int(train_quant_bits),
            'effective_train_bits_mean': eff_train_bits,
            'prox_mu': float(prox_mu if use_prox else 0.0),
            'prox_enabled': bool(use_prox),
            'layer_mixed_precision': bool(bits_policy is not None and callable(bits_policy)),
            'qat_forward': bool(qat_forward),
            'qat_act_channelwise': bool(act_ch),
        }
        return copy.deepcopy(net), signal


class SelfEvolvingRLController(object):
    def __init__(self, args, clients, model_cnt, quant_enabled=True, ref_model=None):
        self.args = args
        self.clients = clients
        self.model_cnt = model_cnt
        self.quant_enabled = quant_enabled
        self.autorl_battery_energy_scale = max(float(getattr(args, 'autorl_battery_energy_scale', 1.0)), 0.0)
        self.min_quant_bits = MIN_QUANT_BITS
        self.quant_options = _parse_quant_bits(
            getattr(args, 'autorl_quant_bits', '8,16,32'),
            args.quant_comm_base_bits,
            min_bits=self.min_quant_bits,
        )
        self.train_quant_options = _parse_quant_bits(
            getattr(args, 'autorl_train_quant_options', getattr(args, 'autorl_quant_bits', '8,16,32')),
            args.quant_comm_base_bits,
            min_bits=self.min_quant_bits,
        )
        self.train_bit_base_weights = _parse_weighted_bits(
            getattr(args, 'autorl_train_bit_weights', '8:0.05,16:0.15,32:0.80'),
            self.train_quant_options,
        )
        self.quant_quality_gamma = max(float(getattr(args, 'autorl_quant_quality_gamma', 0.12)), 0.0)
        self.train_bit_speedups = _parse_bit_float_map(
            getattr(args, 'autorl_train_bit_speedups', '8:1.8,16:1.35,32:1.0'),
            self.train_quant_options,
            default_value=1.0,
            minimum=1e-6,
        )
        self.train_bit_energy_scales = _parse_bit_float_map(
            getattr(args, 'autorl_train_bit_energy_scales', '8:0.55,16:0.75,32:1.0'),
            self.train_quant_options,
            default_value=1.0,
            minimum=1e-6,
        )
        self.low_bit_bonus_weight = max(float(getattr(args, 'autorl_low_bit_bonus_weight', 0.08)), 0.0)
        self.train_scale_max = max(float(getattr(args, 'autorl_train_scale_max', 1.0)), 1.0)
        self.train_precision_mode = str(getattr(args, 'autorl_train_precision_mode', 'adaptive'))
        self.train_high_bit_floor = float(getattr(args, 'autorl_train_high_bit_floor', 0.70))
        self.train_precision_boost = float(getattr(args, 'autorl_train_precision_boost', 0.25))
        self.train_sampling = bool(int(getattr(args, 'autorl_train_bit_sampling', 1)))
        self.acc_drop_high_pct = float(getattr(args, 'autorl_acc_drop_high_pct', 1.5))
        self.acc_drop_low_pct = float(getattr(args, 'autorl_acc_drop_low_pct', 0.35))
        self.comm_bits_drop_schedule = bool(int(getattr(args, 'autorl_comm_bits_drop_schedule', 0)))
        self.comm_stable_bits = max(int(getattr(args, 'autorl_comm_stable_bits', 16)), self.min_quant_bits)
        self.late_comm_min_bits = max(int(getattr(args, 'autorl_late_comm_min_bits', 16)), self.min_quant_bits)
        self.late_train_min_bits = max(int(getattr(args, 'autorl_late_train_min_bits', 32)), self.min_quant_bits)
        self.layer_mixed_precision = bool(int(getattr(args, 'autorl_layer_mixed_precision', 0)))
        self.layer_quant_rules = _parse_layer_quant_rules(getattr(args, 'autorl_layer_quant_deltas', ''))
        self._param_numel = {}
        if ref_model is not None:
            with torch.no_grad():
                for k, t in ref_model.state_dict().items():
                    self._param_numel[str(k)] = int(t.numel())
        self.prev_mean_acc = None
        self.last_mean_acc_drop = None
        self.latest_acc_abs_delta = None
        self.acc_min_bits_until_accuracy = float(getattr(args, 'autorl_acc_min_bits_until_accuracy', 0.0))
        self.acc_drop_signal = str(getattr(args, 'autorl_acc_drop_signal', 'regression')).strip().lower()
        if self.acc_drop_signal not in ('regression', 'abs_delta'):
            self.acc_drop_signal = 'regression'
        self.respect_device_quant_cap = bool(int(getattr(args, 'autorl_respect_device_quant_cap', 0)))
        self.lr = float(getattr(args, 'rl_lr', 0.2))
        self.discount = float(getattr(args, 'rl_discount', 0.9))
        self.epsilon = float(getattr(args, 'autorl_epsilon', getattr(args, 'rl_epsilon', 0.2)))
        self.min_epsilon = float(getattr(args, 'autorl_min_epsilon', 0.02))
        self.epsilon_decay = float(getattr(args, 'autorl_epsilon_decay', 0.995))
        self.stability_start_frac = float(np.clip(getattr(args, 'autorl_stability_start_frac', 0.55), 0.0, 1.0))
        self.late_min_epsilon = max(float(getattr(args, 'autorl_late_min_epsilon', 0.005)), 0.0)
        self.late_comprehensive_mix = max(float(getattr(args, 'autorl_late_comprehensive_mix', 0.20)), 0.0)
        self.late_ucb_c_scale = float(np.clip(getattr(args, 'autorl_late_ucb_c_scale', 0.20), 0.0, 1.0))
        self.late_stay_bonus = max(float(getattr(args, 'autorl_late_stay_bonus', 0.25)), 0.0)
        self.curiosity_temperature = max(float(getattr(args, 'autorl_curiosity_temperature', 0.75)), 1e-6)
        self.curiosity_topk = max(int(getattr(args, 'autorl_curiosity_topk', 4)), 1)
        self.memory_decay = float(getattr(args, 'autorl_memory_decay', 1.0))
        self.route_mode = str(getattr(args, 'autorl_route_mode', 'rl')).strip().lower()
        self.inbound_mode = str(getattr(args, 'autorl_inbound_mode', 'selective')).strip().lower()
        self.reward_accuracy_scale = max(float(getattr(args, 'autorl_reward_accuracy_scale', 1.0)), 0.0)
        self.reward_energy_scale = max(float(getattr(args, 'autorl_reward_energy_scale', 1.0)), 0.0)
        self.reward_latency_scale = max(float(getattr(args, 'autorl_reward_latency_scale', 1.0)), 0.0)
        if self.route_mode not in ('rl', 'comprehensive'):
            raise ValueError('unsupported autorl_route_mode: {}'.format(self.route_mode))
        if self.inbound_mode not in ('selective', 'none', 'all'):
            raise ValueError('unsupported autorl_inbound_mode: {}'.format(self.inbound_mode))
        if self.reward_accuracy_scale + self.reward_energy_scale + self.reward_latency_scale <= 0:
            raise ValueError('at least one AutoRL reward scale must be positive')
        self.min_inbound_models = max(int(getattr(args, 'autorl_min_inbound_models', 0)), 0)
        self.max_inbound_models = int(getattr(args, 'autorl_max_inbound_models', 0))
        self.inbound_score_threshold = float(getattr(args, 'autorl_inbound_score_threshold', 0.0))
        self.late_inbound_threshold_boost = max(float(getattr(args, 'autorl_late_inbound_threshold_boost', 0.20)), 0.0)
        self.late_inbound_gap_threshold = max(float(getattr(args, 'autorl_late_inbound_gap_threshold', 0.03)), 0.0)
        self.inbound_sample_prob = float(getattr(args, 'autorl_inbound_sample_prob', 0.5))
        self.prior_weight = float(getattr(args, 'autorl_prior_weight', 1.0))
        self.coverage_gain_weight = float(getattr(args, 'autorl_coverage_gain_weight', 1.0))
        self.coverage_gap_weight = max(float(getattr(args, 'autorl_coverage_gap_weight', 0.5)), 0.0)
        self.unseen_class_weight = max(float(getattr(args, 'autorl_unseen_class_weight', 0.3)), 0.0)
        self.diversity_weight = float(getattr(args, 'autorl_diversity_weight', 0.15))
        self.diversity_fallback = bool(int(getattr(args, 'autorl_diversity_fallback', 1)))
        self.energy_penalty_weight = float(getattr(args, 'autorl_energy_penalty_weight', 0.1))
        self.latency_penalty_weight = float(getattr(args, 'autorl_latency_penalty_weight', 0.02))
        self.meta_lr = float(getattr(args, 'autorl_meta_lr', 0.15))
        self.meta_prior_enabled = bool(int(getattr(args, 'autorl_meta_prior', 1)))
        self.coarse_state = bool(int(getattr(args, 'autorl_tabular_coarse_state', 1)))
        self.ucb_c = float(getattr(args, 'autorl_tabular_ucb_c', 0.35))
        self.replay_capacity = max(0, int(getattr(args, 'autorl_replay_capacity', 384)))
        self.replay_steps = max(0, int(getattr(args, 'autorl_replay_steps', 6)))
        self.replay_buffer = deque(maxlen=self.replay_capacity) if self.replay_capacity > 0 else None
        self.visit_sa = defaultdict(int)
        self.comprehensive_mix = float(getattr(args, 'autorl_comprehensive_mix', 0.0))
        self.device_utility = np.zeros((model_cnt, args.num_users), dtype=float)
        self.device_visit_count = np.zeros((model_cnt, args.num_users), dtype=np.int32)
        self.device_td_accum = np.zeros((model_cnt, args.num_users), dtype=float)
        self.node_visit_count = np.zeros((model_cnt, args.num_users), dtype=np.int32)
        self.node_last_visit_round = np.full((model_cnt, args.num_users), -1, dtype=np.int32)
        self.global_visit_count = np.zeros(args.num_users, dtype=np.int32)
        self.class_last_seen_round = np.full((model_cnt, args.num_classes), -1, dtype=np.int32)
        self.state_bins = max(int(getattr(args, 'rl_state_bins', 4)), 2)
        self.min_train_pressure = 0.5
        self.max_train_pressure = 3.0
        self.pressure_decay = 0.995
        self.node_coverage_weight = float(getattr(args, 'autorl_node_coverage_weight', 0.35))
        self.global_coverage_weight = float(getattr(args, 'autorl_global_coverage_weight', 0.20))
        self.retention_gain_weight = float(getattr(args, 'autorl_retention_gain_weight', 0.35))
        self.forgetting_penalty_weight = float(getattr(args, 'autorl_forgetting_penalty_weight', 0.45))
        self.visit_staleness_weight = float(getattr(args, 'autorl_visit_staleness_weight', 0.15))
        self.forgetting_horizon = max(int(getattr(args, 'autorl_forgetting_horizon', 50)), 1)
        self.q_table = defaultdict(dict)
        self.model_memory = [np.zeros(args.num_classes) for _ in range(model_cnt)]
        self.model_quality = [{'loss': 1.0, 'confidence': 0.0, 'entropy': 1.0} for _ in range(model_cnt)]
        self.train_pressure = np.ones(model_cnt)
        self.pressure_last_update_round = np.full(model_cnt, -1, dtype=np.int32)
        self.action_history = []
        self.reward_history = []
        self.state_history = []
        self.train_precision_history = []
        self.coverage_history = []

    def _round_frac(self, round_idx):
        return float(round_idx) / max(float(self.args.epochs - 1), 1.0)

    def _stability_progress(self, round_idx):
        round_frac = self._round_frac(round_idx)
        start = float(np.clip(self.stability_start_frac, 0.0, 1.0))
        if round_frac <= start:
            return 0.0
        return float(np.clip((round_frac - start) / max(1.0 - start, 1e-12), 0.0, 1.0))

    def _epsilon_floor(self, round_idx):
        p = self._stability_progress(round_idx)
        return float((1.0 - p) * self.min_epsilon + p * self.late_min_epsilon)

    def _effective_ucb_c(self, round_idx):
        p = self._stability_progress(round_idx)
        scale = (1.0 - p) + p * self.late_ucb_c_scale
        return float(self.ucb_c * scale)

    @staticmethod
    def _bit_floor_option(options, floor_bits):
        opts = sorted(set(int(b) for b in options))
        if not opts:
            return MIN_QUANT_BITS
        floor_bits = int(floor_bits)
        for bit in opts:
            if bit >= floor_bits:
                return int(bit)
        return int(max(opts))

    def _stability_action_bonus(self, current_client, action, round_idx):
        if int(action[0]) != int(current_client):
            return 0.0
        return float(self.late_stay_bonus * self._stability_progress(round_idx))

    def _late_inbound_threshold(self, round_idx):
        return float(self.inbound_score_threshold + self.late_inbound_threshold_boost * self._stability_progress(round_idx))

    def layerwise_bits_policy(self, base_bits):
        base_bits = max(int(base_bits), self.min_quant_bits)
        max_bits = max(int(self.args.quant_comm_base_bits), self.min_quant_bits)
        if not self.layer_mixed_precision:
            return base_bits

        def policy(name):
            name = str(name)
            b = base_bits
            best = ''
            best_delta = 0
            for substr, delta in self.layer_quant_rules:
                if substr in name and len(substr) > len(best):
                    best = substr
                    best_delta = delta
            if best:
                b = base_bits + best_delta
            return max(self.min_quant_bits, min(max_bits, int(b)))

        return policy

    def effective_comm_bits_estimate(self, base_bits):
        base_bits = max(int(base_bits), self.min_quant_bits)
        if not self.layer_mixed_precision or not self._param_numel:
            return float(effective_quant_bits(base_bits))
        policy = self.layerwise_bits_policy(base_bits)
        total = float(sum(self._param_numel.values()))
        w = 0.0
        for k, n in self._param_numel.items():
            bb = policy(k)
            eff = float(effective_quant_bits(bb)) if bb < 32 else 32.0
            w += eff * float(n)
        return w / max(total, 1.0)

    def _quant_quality_score(self, bits, reference_bits=None):
        if not self.quant_enabled:
            return 1.0
        ref = float(reference_bits if reference_bits is not None else max(max(self.quant_options), self.min_quant_bits))
        ref = max(ref, float(self.min_quant_bits))
        bit_value = max(float(bits), float(self.min_quant_bits))
        ratio = float(np.clip(bit_value / ref, 0.0, 1.0))
        if self.quant_quality_gamma <= 0:
            return 1.0
        return float(np.clip(ratio ** self.quant_quality_gamma, 0.0, 1.0))

    def _compression_bonus(self, bits, reference_bits=None):
        if not self.quant_enabled:
            return 0.0
        ref = float(reference_bits if reference_bits is not None else max(max(self.quant_options), self.min_quant_bits))
        ref = max(ref, float(self.min_quant_bits))
        eff = max(float(self.effective_comm_bits_estimate(bits)), 1.0)
        return float(np.clip(1.0 - eff / ref, 0.0, 1.0))

    def training_time_multiplier(self, bits):
        return 1.0 / max(float(self.train_bit_speedups.get(int(bits), 1.0)), 1e-6)

    def training_energy_multiplier(self, bits):
        return max(float(self.train_bit_energy_scales.get(int(bits), 1.0)), 1e-6)

    def _bucket(self, value):
        value = max(min(float(value), 0.999999), 0.0)
        return int(value * self.state_bins)

    def _neighbors(self, client_idx, battery_state=None):
        nbs = [idx for idx in range(self.args.num_users) if Adjacency_matrix[client_idx][idx] == 1]
        if battery_state is None:
            return nbs
        return [j for j in nbs if not battery_state[j]['depleted']]

    def _class_staleness_vector(self, model_id, round_idx):
        round_idx = max(int(round_idx), 0)
        last_seen = self.class_last_seen_round[model_id].astype(float)
        if last_seen.size == 0:
            return np.zeros(0, dtype=float)
        seen = last_seen >= 0
        ages = np.where(seen, np.maximum(float(round_idx) - last_seen, 0.0), float(round_idx + 1))
        return np.clip(ages / max(float(self.forgetting_horizon), 1.0), 0.0, 1.0)

    def _node_coverage_ratio(self, model_id):
        if self.args.num_users <= 0:
            return 0.0
        return float(np.mean(self.node_visit_count[model_id] > 0))

    def _global_node_coverage_ratio(self):
        if self.args.num_users <= 0:
            return 0.0
        return float(np.mean(self.global_visit_count > 0))

    def _node_coverage_reward(self, model_id, candidate_idx, round_idx):
        model_visit = float(self.node_visit_count[model_id, candidate_idx])
        global_visit = float(self.global_visit_count[candidate_idx])
        stale = float(self.node_last_visit_round[model_id, candidate_idx] < 0)
        if self.node_last_visit_round[model_id, candidate_idx] >= 0:
            gap = max(int(round_idx) - int(self.node_last_visit_round[model_id, candidate_idx]), 0)
            stale = min(1.0, gap / max(float(self.forgetting_horizon), 1.0))
        model_bonus = 1.0 / math.sqrt(1.0 + model_visit)
        global_bonus = 1.0 / math.sqrt(1.0 + global_visit)
        global_w = max(float(self.global_coverage_weight), 0.0)
        stale_w = max(float(self.visit_staleness_weight), 0.0)
        model_w = max(1.0 - global_w - stale_w, 0.0)
        total = max(model_w + global_w + stale_w, 1e-12)
        return (model_w * model_bonus + global_w * global_bonus + stale_w * stale) / total

    def _retention_gain(self, model_id, candidate_signature, round_idx):
        memory = np.asarray(self.model_memory[model_id], dtype=float)
        candidate_signature = np.asarray(candidate_signature, dtype=float)
        if memory.sum() <= 0 or candidate_signature.sum() <= 0:
            return 0.0
        normalized = _safe_normalize(memory)
        staleness = self._class_staleness_vector(model_id, round_idx)
        present = (candidate_signature > 0).astype(float)
        return float(np.sum(normalized * present * (0.5 + 0.5 * staleness)))

    def _forgetting_risk(self, model_id, candidate_signature, round_idx):
        memory = np.asarray(self.model_memory[model_id], dtype=float)
        candidate_signature = np.asarray(candidate_signature, dtype=float)
        if memory.sum() <= 0:
            return 0.0
        normalized = _safe_normalize(memory)
        staleness = self._class_staleness_vector(model_id, round_idx)
        absent = (candidate_signature <= 0).astype(float)
        return float(np.sum(normalized * absent * (0.5 + 0.5 * staleness)))

    def _forgetting_pressure(self, model_id, round_idx):
        memory = np.asarray(self.model_memory[model_id], dtype=float)
        if memory.sum() <= 0:
            return 0.0
        normalized = _safe_normalize(memory)
        staleness = self._class_staleness_vector(model_id, round_idx)
        return float(np.sum(normalized * staleness))

    def _route_signature(self, model_id, candidate_idx, selected_inbound=None):
        signature = np.asarray(self.clients[candidate_idx].feature_signature, dtype=float).copy()
        if selected_inbound:
            for inbound in selected_inbound:
                signature = signature + np.asarray(self.model_memory[inbound['model_id']], dtype=float)
                signature = signature + np.asarray(self.clients[inbound['client_id']].feature_signature, dtype=float)
        return signature

    def record_visit(self, model_id, client_idx, round_idx, update_class_memory=True):
        self.node_visit_count[model_id, client_idx] += 1
        self.global_visit_count[client_idx] += 1
        self.node_last_visit_round[model_id, client_idx] = int(round_idx)
        if not update_class_memory:
            return
        signature = np.asarray(self.clients[client_idx].feature_signature, dtype=float)
        active_classes = np.where(signature > 0)[0]
        for class_idx in active_classes:
            self.class_last_seen_round[model_id, class_idx] = int(round_idx)

    @staticmethod
    def _uniform_loss_distribution(vec):
        vec = np.asarray(vec, dtype=float)
        if vec.sum() <= 0:
            return 0.0
        u = vec / vec.sum()
        unif = np.ones_like(u) / max(len(u), 1)
        return float(np.linalg.norm(u - unif))

    def _comprehensive_best_dst(self, model_id, current_idx, last_visit_round, iter_round, battery_state=None):
        """DFL-MM comprehensive fallback with dynamic data/forget tradeoff."""
        idx = int(current_idx)
        model_dist = np.asarray(self.model_memory[model_id], dtype=float)
        comm_ratio = get_quant_comm_ratio(self.clients[idx].quant_bits, False, self.args.quant_comm_base_bits)
        nb_list = []
        data_raw = []
        for nb_idx in range(self.args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1:
                if battery_state is not None and battery_state[nb_idx]['depleted']:
                    continue
                nb_list.append(nb_idx)
                nxt = model_dist + self.clients[nb_idx].feature_signature
                data_raw.append(self._uniform_loss_distribution(nxt))
        if not nb_list:
            return idx
        s = sum(data_raw)
        data_score = [100.0 / len(nb_list)] * len(nb_list) if s <= 0 else [x / s * 100.0 for x in data_raw]
        speed_raw = []
        for nb_idx in nb_list:
            t = get_client_communication_time(idx, nb_idx, multiplier=comm_ratio) + get_client_training_time(nb_idx)
            speed_raw.append(t)
        s = sum(speed_raw)
        speed_score = [100.0 / len(nb_list)] * len(nb_list) if s <= 0 else [x / s * 100.0 for x in speed_raw]
        forget_raw = []
        for nb_idx in nb_list:
            forget_raw.append(float(int(iter_round) - int(last_visit_round[nb_idx]) + 2) ** (-0.5))
        s = sum(forget_raw)
        forget_score = [100.0 / len(nb_list)] * len(nb_list) if s <= 0 else [x / s * 100.0 for x in forget_raw]

        round_frac = float(iter_round) / max(float(self.args.epochs - 1), 1.0)
        coverage_gap = _coverage_gap(model_dist)
        wd = min(max(0.20 + 0.50 * (1.0 - round_frac) + 0.20 * coverage_gap, 0.20), 0.80)
        ws = 0.10
        wf = max(1.0 - wd - ws, 0.10)
        wt = max(wd + ws + wf, 1e-12)
        wd, ws, wf = wd / wt, ws / wt, wf / wt

        comp = [
            wd * data_score[i] + ws * speed_score[i] + wf * forget_score[i]
            for i in range(len(nb_list))
        ]
        best = min(comp)
        ties = [nb_list[i] for i in range(len(nb_list)) if abs(comp[i] - best) < 1e-12]
        if len(ties) > 1:
            best_visit = min(int(last_visit_round[nb_idx]) for nb_idx in ties)
            ties = [nb_idx for nb_idx in ties if int(last_visit_round[nb_idx]) == best_visit]
        return self._curiosity_tie_break_dst(model_id, idx, ties, battery_state, iter_round)

    def _comprehensive_mix_probability(self, model_id, round_idx):
        if self.comprehensive_mix <= 0:
            return 0.0
        round_frac = self._round_frac(round_idx)
        coverage_gap = _coverage_gap(self.model_memory[model_id])
        progress = self._stability_progress(round_idx)
        base_mix = (1.0 - progress) * float(self.comprehensive_mix) + progress * float(self.late_comprehensive_mix)
        prob = base_mix + 0.20 * coverage_gap + 0.15 * (1.0 - round_frac) * (1.0 - progress)
        if self.last_mean_acc_drop is not None and self.last_mean_acc_drop > 0:
            prob += min(float(self.last_mean_acc_drop) / 8.0, 0.10)
        return float(np.clip(prob, 0.0, 0.85))

    def _inbound_candidates(self, model_id, current_client, model_locations, battery_state=None):
        candidates = []
        if battery_state is not None and battery_state[current_client]['depleted']:
            return candidates
        neighbor_set = set(self._neighbors(current_client, battery_state))
        for sender_model_id, sender_client in enumerate(model_locations):
            if sender_model_id == model_id:
                continue
            if battery_state is not None and battery_state[sender_client]['depleted']:
                continue
            if sender_client in neighbor_set:
                candidates.append({
                    'model_id': int(sender_model_id),
                    'client_id': int(sender_client),
                })
        return candidates

    def _quant_options_for_sender(self, sender_idx):
        if not self.respect_device_quant_cap:
            return sorted(set(bit for bit in self.quant_options if bit >= self.min_quant_bits))
        max_bits = max(int(self.clients[sender_idx].quant_bits), self.min_quant_bits)
        options = [bit for bit in self.quant_options if self.min_quant_bits <= bit <= max_bits]
        return sorted(set(options or [self.min_quant_bits]))

    def _state_key(self, model_id, current_client, battery_state, round_idx):
        memory = self.model_memory[model_id]
        normalized = _safe_normalize(memory)
        missing_ratio = float(np.mean(normalized <= 0)) if len(normalized) else 0.0
        node_coverage_ratio = self._node_coverage_ratio(model_id)
        forgetting_pressure = self._forgetting_pressure(model_id, round_idx)
        mid = int(model_id)
        cid = int(current_client)
        if self.coarse_state:
            return (
                mid,
                cid,
                self._bucket(_coverage_gap(memory)),
                self._bucket(missing_ratio),
                self._bucket(node_coverage_ratio),
                self._bucket(forgetting_pressure),
            )
        current_battery = battery_state[current_client]['remaining_j'] / max(battery_state[current_client]['capacity_j'], 1e-12)
        return (
            mid,
            cid,
            self._bucket(_coverage_gap(memory)),
            self._bucket(missing_ratio),
            self._bucket(node_coverage_ratio),
            self._bucket(forgetting_pressure),
            self._bucket(_entropy_score(memory)),
            self._bucket(current_battery),
        )

    def _adaptive_weights(self, model_id, battery_state):
        memory = self.model_memory[model_id]
        normalized = _safe_normalize(memory)
        missing_ratio = float(np.mean(normalized <= 0)) if len(normalized) else 1.0
        avg_remaining = np.mean([
            item['remaining_j'] / max(item['capacity_j'], 1e-12)
            for item in battery_state
        ])
        energy_pressure = 1.0 - avg_remaining
        training_pressure = (self.train_pressure[model_id] - 1.0) / max(self.max_train_pressure - 1.0, 1e-12)
        accuracy_weight = 0.60 + 0.45 * missing_ratio + 0.20 * max(training_pressure, 0.0)
        energy_weight = 0.08 + 0.18 * energy_pressure
        latency_weight = 0.05 + 0.05 * (1.0 - missing_ratio)
        accuracy_weight *= self.reward_accuracy_scale
        energy_weight *= self.reward_energy_scale
        latency_weight *= self.reward_latency_scale
        total = accuracy_weight + energy_weight + latency_weight
        return accuracy_weight / total, energy_weight / total, latency_weight / total

    def _bootstrap_q(self, model_id, src_idx, dst_idx, bits, battery_state, round_idx=0):
        if not self.meta_prior_enabled:
            return self.prior_weight * self.predict_reward(model_id, src_idx, dst_idx, bits, battery_state, round_idx=round_idx)
        return self._meta_prior_q(model_id, src_idx, dst_idx, bits, battery_state, round_idx=round_idx)

    def _next_state_best_q(self, model_id, next_client, next_state_key, battery_state, round_idx=0):
        next_neighbors = self._neighbors(next_client, battery_state)
        if next_client not in next_neighbors:
            next_neighbors.append(next_client)
        if not next_neighbors:
            next_neighbors = [next_client]
        next_actions = [
            (neighbor_idx, bits)
            for neighbor_idx in next_neighbors
            for bits in self._quant_options_for_sender(next_client)
        ]
        next_q_values = self.q_table.setdefault(next_state_key, {})
        next_best = 0.0
        for next_action in next_actions:
            if next_action not in next_q_values:
                next_q_values[next_action] = self._bootstrap_q(
                    model_id, next_client, next_action[0], next_action[1], battery_state, round_idx=round_idx
                )
            next_best = max(next_best, next_q_values[next_action])
        return next_best

    def _action_ucb_score(self, state_key, action, q_val, round_idx):
        ucb_c = self._effective_ucb_c(round_idx)
        if ucb_c <= 0:
            return q_val
        n = self.visit_sa.get((state_key, action), 0)
        t = max(2, int(round_idx) + 2)
        return q_val + ucb_c * math.sqrt(math.log(t) / (n + 1))

    def _replay_td(self, tr):
        model_id = tr['model_id']
        state_key = tr['state_key']
        next_client = int(tr['next_client'])
        pair_action = tuple(tr['action'])
        reward = float(tr['reward'])
        next_state_key = tr['next_state_key']
        battery_state = tr['battery']
        round_idx = int(tr.get('round_idx', 0))
        next_best = self._next_state_best_q(model_id, next_client, next_state_key, battery_state, round_idx=round_idx)
        q_values = self.q_table.setdefault(state_key, {})
        old_value = q_values.get(pair_action, 0.0)
        td_target = reward + self.discount * next_best
        q_values[pair_action] = (1.0 - self.lr) * old_value + self.lr * td_target

    def record_round_performance(self, performance_value):
        """记录上一轮的单调性能信号；可来自 test_acc，也可来自在线 proxy。"""
        if self.prev_mean_acc is not None:
            delta = float(self.prev_mean_acc) - float(performance_value)
            self.last_mean_acc_drop = max(0.0, delta)
            self.latest_acc_abs_delta = abs(delta)
        else:
            self.last_mean_acc_drop = None
            self.latest_acc_abs_delta = None
        self.prev_mean_acc = float(performance_value)

    def record_round_test_accuracy(self, mean_acc_percent):
        self.record_round_performance(mean_acc_percent)

    def _training_pressure_signal(self, model_id):
        memory = np.asarray(self.model_memory[model_id], dtype=float)
        normalized = _safe_normalize(memory)
        coverage_gap = _coverage_gap(memory)
        missing_ratio = float(np.mean(normalized <= 0)) if len(normalized) else 1.0
        node_gap = 1.0 - self._node_coverage_ratio(model_id)
        forgetting_pressure = self._forgetting_pressure(model_id, getattr(self, 'current_round', 0))
        quality = self.model_quality[model_id]
        confidence_gap = 1.0 - float(quality.get('confidence', 0.0))
        entropy_gap = float(quality.get('entropy', 1.0))
        loss_val = max(float(quality.get('loss', 1.0)), 0.0)
        loss_gap = min(loss_val / max(loss_val + 1.0, 1e-12), 1.0)
        global_drop = 0.0
        if self.last_mean_acc_drop is not None:
            global_drop = min(max(float(self.last_mean_acc_drop) / 10.0, 0.0), 1.0)
        signal = (
            1.0
            + 1.05 * coverage_gap
            + 0.45 * missing_ratio
            + 0.25 * node_gap
            + 0.35 * forgetting_pressure
            + 0.30 * confidence_gap
            + 0.20 * entropy_gap
            + 0.10 * loss_gap
            + 0.50 * global_drop
        )
        return float(np.clip(signal, self.min_train_pressure, self.max_train_pressure))

    def _refresh_train_pressure(self, model_id, force=False):
        current_round = int(getattr(self, 'current_round', -1))
        if not force and self.pressure_last_update_round[model_id] == current_round:
            return float(self.train_pressure[model_id])
        target = self._training_pressure_signal(model_id)
        current = float(self.train_pressure[model_id])
        refreshed = 0.45 * current + 0.55 * target
        self.train_pressure[model_id] = float(np.clip(refreshed, self.min_train_pressure, self.max_train_pressure))
        self.pressure_last_update_round[model_id] = current_round
        return float(self.train_pressure[model_id])

    def _acc_gate_blocks_low_bits(self):
        t = float(getattr(self, 'acc_min_bits_until_accuracy', 0.0))
        if t <= 0.0:
            return False
        ref = self.prev_mean_acc
        return ref is not None and ref < t

    def _drop_adaptive_metric(self):
        if self.acc_drop_signal == 'abs_delta':
            return self.latest_acc_abs_delta
        return self.last_mean_acc_drop

    def _pick_bits_drop_adaptive(self, options):
        """
        Map schedule metric to a bit width in ``options`` (ascending).
        regression: metric = max(0, prev_acc - current_acc). metric==0 (变好或持平) 不再压到最低比特。
        abs_delta: metric = |prev_acc - current_acc|；变化大用高位宽，变化小允许低位宽（仍受 acc_gate 约束）。
        """
        opts = sorted(set(int(b) for b in options if b >= self.min_quant_bits))
        if not opts:
            return self.min_quant_bits, {}
        metric = self._drop_adaptive_metric()
        if metric is None:
            return max(opts), {
                'acc_drop': None,
                'acc_abs_delta': None,
                'scheduled_train_bits': max(opts),
                'schedule_reason': 'warmup',
            }

        hi, lo = self.acc_drop_high_pct, self.acc_drop_low_pct
        if hi <= lo:
            hi = lo + 1e-6

        m = float(metric)

        if self.acc_drop_signal == 'regression' and self.last_mean_acc_drop is not None and float(self.last_mean_acc_drop) == 0.0:
            return max(opts), {
                'acc_drop': 0.0,
                'acc_abs_delta': float(self.latest_acc_abs_delta) if self.latest_acc_abs_delta is not None else 0.0,
                'scheduled_train_bits': max(opts),
                'schedule_reason': 'no_regression',
            }

        if self.acc_drop_signal == 'abs_delta' and m == 0.0:
            return max(opts), {
                'acc_drop': float(self.last_mean_acc_drop) if self.last_mean_acc_drop is not None else 0.0,
                'acc_abs_delta': 0.0,
                'scheduled_train_bits': max(opts),
                'schedule_reason': 'zero_delta',
            }

        if m >= hi:
            b = max(opts)
            reason = 'high_metric'
        elif m <= lo:
            if self._acc_gate_blocks_low_bits():
                b = max(opts)
                reason = 'acc_gate'
            else:
                b = min(opts)
                reason = 'low_metric'
        else:
            t = (m - lo) / (hi - lo)
            idx = int(round(t * (len(opts) - 1)))
            b = int(opts[idx])
            reason = 'interpolate'

        meta = {
            'acc_drop': float(self.last_mean_acc_drop) if self.last_mean_acc_drop is not None else None,
            'acc_abs_delta': float(self.latest_acc_abs_delta) if self.latest_acc_abs_delta is not None else None,
            'scheduled_train_bits': int(b),
            'schedule_reason': reason,
            'schedule_metric': m,
        }
        return int(b), meta

    def _train_bits_from_acc_drop(self):
        """跌幅大→高位宽；跌幅小→低位宽；无回落(prev<=current)不再压到 8bit。无历史时高位宽热身。"""
        options = sorted(set(int(b) for b in self.train_quant_options if b >= self.min_quant_bits))
        b, meta = self._pick_bits_drop_adaptive(options)
        return int(b), meta

    def _comm_bit_floor_from_acc_drop(self, sender_idx):
        if not self.comm_bits_drop_schedule:
            return min(self._quant_options_for_sender(sender_idx))
        options = sorted(self._quant_options_for_sender(sender_idx))
        metric = self._drop_adaptive_metric()
        if metric is None:
            b = max(options)
        elif self._acc_gate_blocks_low_bits():
            b = max(options)
        elif self.acc_drop_signal == 'regression' and self.last_mean_acc_drop is not None and float(self.last_mean_acc_drop) == 0.0:
            b = self._bit_floor_option(options, self.comm_stable_bits)
        else:
            b, _ = self._pick_bits_drop_adaptive(options)
        if self._stability_progress(getattr(self, 'current_round', 0)) > 0.0:
            b = max(int(b), self._bit_floor_option(options, self.late_comm_min_bits))
        return int(b)

    def update_model_memory(self, model_id, client_idx, signal, round_idx=None):
        self.model_memory[model_id] = self.model_memory[model_id] * self.memory_decay + self.clients[client_idx].feature_signature
        self.model_quality[model_id] = dict(signal)
        if round_idx is not None:
            self.current_round = int(round_idx)
        self._refresh_train_pressure(model_id, force=True)

    def _update_device_utility(self, model_id, dst_idx, td_target):
        dst_idx = int(dst_idx)
        td = float(td_target)
        old_u = float(self.device_utility[model_id, dst_idx])
        n = int(self.device_visit_count[model_id, dst_idx])
        if n == 0:
            self.device_utility[model_id, dst_idx] = td
        else:
            alpha = self.meta_lr
            self.device_utility[model_id, dst_idx] = (1.0 - alpha) * old_u + alpha * td
        self.device_visit_count[model_id, dst_idx] = n + 1
        self.device_td_accum[model_id, dst_idx] += td

    def _meta_prior_q(self, model_id, src_idx, dst_idx, bits, battery_state, round_idx=0):
        sender_bits = max(self.clients[src_idx].quant_bits, self.min_quant_bits)
        quant_quality = self._quant_quality_score(bits, sender_bits)
        visits = int(self.device_visit_count[model_id, dst_idx])
        proxy = self.predict_reward(model_id, src_idx, dst_idx, bits, battery_state, round_idx=round_idx)
        if visits < 1:
            return self.prior_weight * proxy
        u = float(self.device_utility[model_id, dst_idx])
        return self.prior_weight * u * quant_quality

    def _best_comm_bits(self, model_id, current_client, sender_model_id, sender_client, battery_state, round_idx=0, inbound=False):
        options = self._quant_options_for_sender(sender_client)
        if not options:
            return self.min_quant_bits
        floor_bits = self._comm_bit_floor_from_acc_drop(sender_client)
        options = [bit for bit in options if int(bit) >= int(floor_bits)]
        if not options:
            options = [max(self._quant_options_for_sender(sender_client))]
        scored = []
        for bits in options:
            if inbound:
                score = self.predict_inbound_reward(
                    model_id,
                    current_client,
                    sender_model_id,
                    sender_client,
                    bits,
                    battery_state,
                    round_idx=round_idx,
                )
            else:
                score = self.predict_reward(
                    model_id,
                    sender_client,
                    current_client,
                    bits,
                    battery_state,
                    round_idx=round_idx,
                )
            score += self.low_bit_bonus_weight * self._compression_bonus(bits, max(self.clients[sender_client].quant_bits, self.min_quant_bits))
            scored.append((float(score), int(bits)))
        best = max(score for score, _ in scored)
        ties = [bits for score, bits in scored if abs(score - best) < 1e-12]
        if len(ties) > 1:
            return int(min(ties))
        return int(max(scored, key=lambda item: (item[0], -item[1]))[1])

    def get_training_scale(self, model_id):
        pressure = self._refresh_train_pressure(model_id)
        scale = 1.0 + 0.85 * max(pressure - 1.0, 0.0)
        if self.last_mean_acc_drop is not None and self.last_mean_acc_drop > 0:
            scale += min(float(self.last_mean_acc_drop) / 8.0, 0.5)
        return float(np.clip(scale, 1.0, self.train_scale_max))

    def select_train_quant_bits(self, model_id, client_idx, battery_state):
        options = list(self.train_quant_options)
        if len(options) == 0:
            return self.min_quant_bits, {'options': [self.min_quant_bits], 'probabilities': {self.min_quant_bits: 1.0}}

        pressure = self._refresh_train_pressure(model_id)

        if self.train_precision_mode == 'fixed':
            fixed_bits = max(int(getattr(self.args, 'autorl_train_quant_bits', options[-1])), self.min_quant_bits)
            if fixed_bits not in options:
                options = sorted(set(options + [fixed_bits]))
            return fixed_bits, {
                'options': [int(bit) for bit in options],
                'probabilities': {int(bit): float(1.0 if bit == fixed_bits else 0.0) for bit in options},
            }

        if self.train_precision_mode == 'drop_adaptive':
            selected_bit, meta = self._train_bits_from_acc_drop()
            opts = sorted(set(int(b) for b in options if b >= self.min_quant_bits))
            if selected_bit not in opts and opts:
                selected_bit = min(opts, key=lambda x: abs(x - selected_bit))
            if opts and self._stability_progress(getattr(self, 'current_round', 0)) > 0.0:
                selected_bit = max(int(selected_bit), self._bit_floor_option(opts, self.late_train_min_bits))
            probs = {int(b): float(1.0 if b == selected_bit else 0.0) for b in opts}
            out = {
                'options': opts,
                'probabilities': probs,
                'precision_mode': 'drop_adaptive',
            }
            out.update(meta)
            return selected_bit, out

        base_probs = np.asarray([self.train_bit_base_weights.get(int(bit), 0.0) for bit in options], dtype=float)
        if base_probs.sum() <= 0:
            base_probs = np.ones(len(options), dtype=float)
        base_probs = base_probs / base_probs.sum()

        quality = self.model_quality[model_id]
        coverage_gap = _coverage_gap(self.model_memory[model_id])
        confidence_gap = 1.0 - float(quality.get('confidence', 0.0))
        entropy_gap = float(quality.get('entropy', 1.0))
        pressure_gap = (pressure - 1.0) / max(self.max_train_pressure - 1.0, 1e-12)
        battery_ratio = battery_state[client_idx]['remaining_j'] / max(battery_state[client_idx]['capacity_j'], 1e-12)
        accuracy_pressure = np.clip(0.55 * coverage_gap + 0.20 * confidence_gap + 0.15 * entropy_gap + 0.10 * max(pressure_gap, 0.0), 0.0, 1.0)
        if self.last_mean_acc_drop is not None and self.last_mean_acc_drop > 0:
            accuracy_pressure = min(1.0, accuracy_pressure + min(float(self.last_mean_acc_drop) / 10.0, 0.25))
        energy_pressure = np.clip(1.0 - battery_ratio, 0.0, 1.0)

        strong_accuracy_focus = (
            accuracy_pressure >= 0.88
            or pressure >= 2.35
            or (self.last_mean_acc_drop is not None and self.last_mean_acc_drop > 0)
        )

        max_bits = max(max(options), self.min_quant_bits)
        bit_scores = []
        for bit, prior in zip(options, base_probs):
            quality = self._quant_quality_score(bit, max_bits)
            speedup = max(float(self.train_bit_speedups.get(int(bit), 1.0)), 1e-6)
            time_saving = float(np.clip(1.0 - 1.0 / speedup, 0.0, 1.0))
            energy_factor = max(float(self.train_bit_energy_scales.get(int(bit), 1.0)), 1e-6)
            energy_saving = float(np.clip(1.0 - energy_factor / speedup, 0.0, 1.0))
            efficiency_pressure = (1.0 - accuracy_pressure) * (0.55 + 0.45 * energy_pressure)
            score = (
                (0.72 + 0.28 * accuracy_pressure) * quality
                + 0.20 * efficiency_pressure * time_saving
                + 0.22 * efficiency_pressure * energy_saving
                + 0.04 * float(prior)
            )
            bit_scores.append(float(score))

        scores = np.asarray(bit_scores, dtype=float)
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            adjusted_probs = np.ones(len(options), dtype=float) / float(len(options))
        else:
            temp = 0.10 if strong_accuracy_focus else 0.35
            shifted = (scores - scores.max()) / max(temp, 1e-6)
            adjusted_probs = np.exp(shifted)
            adjusted_probs = adjusted_probs / max(adjusted_probs.sum(), 1e-12)

        if strong_accuracy_focus:
            selected_bit = int(max(options))
        elif self.train_sampling:
            selected_bit = int(random.choices(options, weights=adjusted_probs.tolist(), k=1)[0])
        else:
            selected_bit = int(options[int(np.argmax(adjusted_probs))])
        if self._stability_progress(getattr(self, 'current_round', 0)) > 0.0:
            selected_bit = max(int(selected_bit), self._bit_floor_option(options, self.late_train_min_bits))

        return selected_bit, {
            'options': [int(bit) for bit in options],
            'probabilities': {int(bit): float(prob) for bit, prob in zip(options, adjusted_probs)},
            'accuracy_pressure': float(accuracy_pressure),
            'energy_pressure': float(energy_pressure),
            'train_pressure': float(pressure),
            'bit_scores': {int(bit): float(score) for bit, score in zip(options, bit_scores)},
            'speedups': {int(bit): float(self.train_bit_speedups.get(int(bit), 1.0)) for bit in options},
            'energy_scales': {int(bit): float(self.train_bit_energy_scales.get(int(bit), 1.0)) for bit in options},
        }

    def _expected_training_time(self, client_idx):
        profile = self.clients[client_idx].device_profile
        client_type = profile['type']
        return TRAINING_TIME_MEAN[client_type] * DEVICE_TYPE_PROFILES[client_type]['frequency_ghz'] / max(profile['frequency_ghz'], 1e-12)

    def _expected_comm_time(self, src_idx, dst_idx, bits):
        eff_bits = self.effective_comm_bits_estimate(bits)
        ratio = get_quant_comm_ratio(eff_bits, self.quant_enabled, self.args.quant_comm_base_bits)
        return get_client_communication_time(src_idx, dst_idx, multiplier=ratio)

    def _model_diversity_score(self, model_id, candidate_idx):
        """鼓励各 model 分头探索不同类别，避免与其它模型重复覆盖。"""
        other_models = [m for m in range(self.model_cnt) if m != model_id]
        if not other_models:
            return 0.0
        candidate_sig = np.asarray(self.clients[candidate_idx].feature_signature, dtype=float)
        mem = np.asarray(self.model_memory[model_id], dtype=float)
        diversity = 0.0
        for other_id in other_models:
            other_mem = np.asarray(self.model_memory[other_id], dtype=float)
            unique_mask = (other_mem > 0) & (mem <= 0)
            diversity += float(np.sum(candidate_sig * unique_mask))
        return diversity / float(len(other_models))

    def _pick_action_diversity_fallback(self, model_id, actions):
        """保底：在候选动作中优先选「分头探索」得分最高的目的地，再在等量比特中随机。"""
        if not actions:
            raise ValueError('actions must be non-empty')
        by_dst = {}
        for dst_idx, bits in actions:
            by_dst.setdefault(dst_idx, []).append((dst_idx, bits))
        div_scores = {dst: self._model_diversity_score(model_id, dst) for dst in by_dst}
        best_div = max(div_scores.values())
        best_dsts = [d for d, s in div_scores.items() if s >= best_div - 1e-15]
        dst = random.choice(best_dsts)
        return random.choice(by_dst[dst])

    def _curiosity_action_score(self, model_id, current_client, action, battery_state, round_idx):
        """Prefer actions that cover missing/stale features and under-visited nodes."""
        dst_idx, bits = action
        if battery_state is not None and battery_state[dst_idx]['depleted']:
            return -1e12
        memory = np.asarray(self.model_memory[model_id], dtype=float)
        candidate_signature = np.asarray(self.clients[dst_idx].feature_signature, dtype=float)
        node_coverage = self._node_coverage_reward(model_id, dst_idx, round_idx)
        novelty = _novelty_score(memory, candidate_signature)
        last_seen = int(self.node_last_visit_round[model_id, dst_idx])
        if last_seen < 0:
            staleness = 1.0
        else:
            staleness = min(
                max(int(round_idx) - last_seen, 0) / max(float(self.forgetting_horizon), 1.0),
                1.0,
            )
        global_unvisited = 1.0 / math.sqrt(1.0 + float(self.global_visit_count[dst_idx]))
        max_bits = max(max(self._quant_options_for_sender(current_client)), self.min_quant_bits)
        bit_quality = self._quant_quality_score(bits, max_bits)
        bit_efficiency = self._compression_bonus(bits, max_bits)
        if battery_state is None:
            battery_ratio = 1.0
        else:
            battery_ratio = battery_state[current_client]['remaining_j'] / max(
                battery_state[current_client]['capacity_j'],
                1e-12,
            )
        return (
            0.42 * node_coverage
            + 0.22 * novelty
            + 0.16 * staleness
            + 0.10 * global_unvisited
            + 0.03 * bit_quality
            + 0.07 * bit_efficiency
            + 0.05 * battery_ratio
        )

    def _sample_curiosity_action(self, model_id, current_client, actions, battery_state, round_idx):
        """Random exploration, biased toward high-curiosity actions."""
        if not actions:
            raise ValueError('actions must be non-empty')
        scored = [
            (action, self._curiosity_action_score(model_id, current_client, action, battery_state, round_idx))
            for action in actions
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        scored = scored[:min(self.curiosity_topk, len(scored))]
        max_score = max(score for _, score in scored)
        weights = [
            math.exp((score - max_score) / self.curiosity_temperature)
            for _, score in scored
        ]
        if sum(weights) <= 0:
            return random.choice([action for action, _ in scored])
        return random.choices([action for action, _ in scored], weights=weights, k=1)[0]

    def _curiosity_tie_break_action(self, model_id, current_client, actions, battery_state, round_idx):
        if len(actions) <= 1:
            return actions[0]
        scored = [
            (action, self._curiosity_action_score(model_id, current_client, action, battery_state, round_idx))
            for action in actions
        ]
        best_score = max(score for _, score in scored)
        tied = [action for action, score in scored if score >= best_score - 1e-12]
        if self.diversity_fallback and len(tied) > 1:
            best_div = max(self._model_diversity_score(model_id, action[0]) for action in tied)
            tied = [
                action for action in tied
                if self._model_diversity_score(model_id, action[0]) >= best_div - 1e-15
            ]
        return random.choice(tied)

    def _curiosity_tie_break_dst(self, model_id, current_client, dst_candidates, battery_state, round_idx):
        if len(dst_candidates) <= 1:
            return int(dst_candidates[0])
        max_bits = max(max(self._quant_options_for_sender(current_client)), self.min_quant_bits)
        actions = [(int(dst_idx), int(max_bits)) for dst_idx in dst_candidates]
        return int(self._curiosity_tie_break_action(model_id, current_client, actions, battery_state, round_idx)[0])

    def _coverage_reward(self, model_id, candidate_idx, bits, current_idx, candidate_signature=None, round_idx=0):
        memory = self.model_memory[model_id]
        if candidate_signature is None:
            candidate_signature = self.clients[candidate_idx].feature_signature
        candidate_signature = np.asarray(candidate_signature, dtype=float)
        current_gap = _coverage_gap(memory)
        next_gap = _coverage_gap(memory + candidate_signature)

        gap_reduction = max(current_gap - next_gap, 0.0)
        novelty = _novelty_score(memory, candidate_signature)
        decay_comp = _decay_compensation(memory, candidate_signature)
        retention_gain = self._retention_gain(model_id, candidate_signature, round_idx)
        forgetting_risk = self._forgetting_risk(model_id, candidate_signature, round_idx)
        node_coverage = self._node_coverage_reward(model_id, candidate_idx, round_idx)

        sender_bits = max(self.clients[current_idx].quant_bits, self.min_quant_bits)
        quant_quality = self._quant_quality_score(bits, sender_bits)
        diversity_norm = self._model_diversity_score(model_id, candidate_idx) / max(float(self.args.num_classes), 1.0)
        coverage_gain = (
            self.coverage_gap_weight * gap_reduction + self.unseen_class_weight * novelty + 0.2 * decay_comp
            + self.diversity_weight * diversity_norm
            + self.node_coverage_weight * node_coverage
            + self.retention_gain_weight * retention_gain
            - self.forgetting_penalty_weight * forgetting_risk
        )
        return coverage_gain * quant_quality

    def _initial_action_value(self, model_id, src_idx, dst_idx, bits, battery_state, round_idx=0):
        return self._meta_prior_q(model_id, src_idx, dst_idx, bits, battery_state, round_idx=round_idx)

    def _inbound_coverage_reward(self, model_id, sender_model_id, sender_client, bits, round_idx=0):
        memory = self.model_memory[model_id]
        sender_memory = self.model_memory[sender_model_id]
        sender_signature = self.clients[sender_client].feature_signature
        inbound_feature = sender_memory + sender_signature
        current_gap = _coverage_gap(memory)
        next_gap = _coverage_gap(memory + inbound_feature)
        gap_reduction = max(current_gap - next_gap, 0.0)
        novelty = _novelty_score(memory, inbound_feature)
        decay_comp = _decay_compensation(memory, inbound_feature)
        retention_gain = self._retention_gain(model_id, inbound_feature, round_idx)
        forgetting_risk = self._forgetting_risk(model_id, inbound_feature, round_idx)
        sender_bits = max(self.clients[sender_client].quant_bits, self.min_quant_bits)
        quant_quality = self._quant_quality_score(bits, sender_bits)
        coverage_gain = self.coverage_gap_weight * gap_reduction + self.unseen_class_weight * novelty + 0.2 * decay_comp
        coverage_gain = coverage_gain + self.retention_gain_weight * retention_gain - self.forgetting_penalty_weight * forgetting_risk
        return coverage_gain * quant_quality

    def _energy_reward(self, src_idx, dst_idx, bits, battery_state):
        comm_time = self._expected_comm_time(src_idx, dst_idx, bits)
        src_energy, dst_energy = get_communication_energy_breakdown(src_idx, dst_idx, comm_time)
        es = self.autorl_battery_energy_scale
        src_energy = src_energy * es
        dst_energy = dst_energy * es
        src_remaining = battery_state[src_idx]['remaining_j'] / max(battery_state[src_idx]['capacity_j'], 1e-12)
        dst_remaining = battery_state[dst_idx]['remaining_j'] / max(battery_state[dst_idx]['capacity_j'], 1e-12)
        src_cost = src_energy / max(battery_state[src_idx]['capacity_j'], 1e-12)
        dst_cost = dst_energy / max(battery_state[dst_idx]['capacity_j'], 1e-12)
        return max((src_remaining + dst_remaining) / 2.0 - (src_cost + dst_cost), 0.0)

    def _latency_reward(self, src_idx, dst_idx, bits):
        comm_time = self._expected_comm_time(src_idx, dst_idx, bits)
        sender_bits = max(self.clients[src_idx].quant_bits, self.min_quant_bits)
        quant_loss_proxy = max(sender_bits - bits, 0) / float(sender_bits)
        return 1.0 / (1.0 + comm_time * (1.0 + quant_loss_proxy))

    def predict_reward(self, model_id, src_idx, dst_idx, bits, battery_state, round_idx=0, candidate_signature=None):
        accuracy_weight, energy_weight, latency_weight = self._adaptive_weights(model_id, battery_state)
        if candidate_signature is None:
            candidate_signature = self.clients[dst_idx].feature_signature
        compression_bonus = self._compression_bonus(bits, max(self.clients[src_idx].quant_bits, self.min_quant_bits))
        return (
            accuracy_weight * self._coverage_reward(model_id, dst_idx, bits, src_idx, candidate_signature=candidate_signature, round_idx=round_idx)
            + energy_weight * self._energy_reward(src_idx, dst_idx, bits, battery_state)
            + latency_weight * self._latency_reward(src_idx, dst_idx, bits)
            + self.low_bit_bonus_weight * (energy_weight + latency_weight) * compression_bonus
        )

    def predict_inbound_reward(self, model_id, current_client, sender_model_id, sender_client, bits, battery_state, round_idx=0):
        accuracy_weight, energy_weight, latency_weight = self._adaptive_weights(model_id, battery_state)
        compression_bonus = self._compression_bonus(bits, max(self.clients[sender_client].quant_bits, self.min_quant_bits))
        return (
            accuracy_weight * self._inbound_coverage_reward(model_id, sender_model_id, sender_client, bits, round_idx=round_idx)
            + energy_weight * self._energy_reward(sender_client, current_client, bits, battery_state)
            + latency_weight * self._latency_reward(sender_client, current_client, bits)
            + self.low_bit_bonus_weight * (energy_weight + latency_weight) * compression_bonus
        )

    def _select_inbound_models(self, model_id, current_client, model_locations, battery_state, state_key, round_idx):
        candidates = self._inbound_candidates(model_id, current_client, model_locations, battery_state)
        if self.inbound_mode == 'none':
            return [], candidates
        if self.inbound_mode == 'all':
            selected = []
            for candidate in candidates:
                bits = self._best_comm_bits(
                    model_id,
                    current_client,
                    candidate['model_id'],
                    candidate['client_id'],
                    battery_state,
                    round_idx=round_idx,
                    inbound=True,
                )
                selected.append({
                    'model_id': int(candidate['model_id']),
                    'client_id': int(candidate['client_id']),
                    'quant_bits': int(bits),
                })
            return selected, candidates
        if not candidates or self.max_inbound_models <= 0:
            return [], candidates
        stability_progress = self._stability_progress(round_idx)
        if (
            stability_progress > 0.0
            and self.min_inbound_models <= 0
            and _coverage_gap(self.model_memory[model_id]) <= self.late_inbound_gap_threshold
        ):
            return [], candidates

        max_count = min(self.max_inbound_models, len(candidates))
        min_count = min(self.min_inbound_models, max_count)
        threshold = self._late_inbound_threshold(round_idx)

        scored = []
        for candidate in candidates:
            sender_memory = self.model_memory[candidate['model_id']]
            memory = self.model_memory[model_id]
            sender_signature = self.clients[candidate['client_id']].feature_signature
            inbound_feature = sender_memory + sender_signature
            current_gap = _coverage_gap(memory)
            next_gap = _coverage_gap(memory + inbound_feature)
            gap_reduction = max(current_gap - next_gap, 0.0)
            novelty = _novelty_score(memory, inbound_feature)
            retention_gain = self._retention_gain(model_id, inbound_feature, round_idx)
            forgetting_risk = self._forgetting_risk(model_id, inbound_feature, round_idx)
            node_coverage = self._node_coverage_reward(candidate['model_id'], candidate['client_id'], round_idx)
            bits = self._best_comm_bits(
                model_id,
                current_client,
                candidate['model_id'],
                candidate['client_id'],
                battery_state,
                round_idx=round_idx,
                inbound=True,
            )
            score = self.predict_inbound_reward(
                model_id,
                current_client,
                candidate['model_id'],
                candidate['client_id'],
                bits,
                battery_state,
                round_idx=round_idx,
            )
            score += 0.20 * gap_reduction + 0.12 * novelty + 0.12 * node_coverage
            score += self.retention_gain_weight * retention_gain - self.forgetting_penalty_weight * forgetting_risk
            if score < threshold and len(scored) >= min_count:
                continue
            if len(scored) >= min_count and random.random() > self.inbound_sample_prob:
                continue
            scored.append((score, {
                'model_id': int(candidate['model_id']),
                'client_id': int(candidate['client_id']),
                'quant_bits': int(bits),
            }))

        scored.sort(key=lambda item: item[0], reverse=True)

        selected = []
        selected_model_ids = set()
        for score, candidate in scored:
            if len(selected) >= max_count:
                break
            if candidate['model_id'] not in selected_model_ids:
                selected.append({
                    'model_id': candidate['model_id'],
                    'client_id': candidate['client_id'],
                    'quant_bits': int(candidate['quant_bits']),
                })
                selected_model_ids.add(candidate['model_id'])

        while len(selected) < min_count:
            for score, candidate in scored:
                if candidate['model_id'] not in selected_model_ids:
                    selected.append({
                        'model_id': candidate['model_id'],
                        'client_id': candidate['client_id'],
                        'quant_bits': int(candidate['quant_bits']),
                    })
                    selected_model_ids.add(candidate['model_id'])
                    break

        return selected[:max_count], candidates

    def select_action(self, model_id, current_client, battery_state, round_idx, model_locations, last_visit_round):
        self.current_round = int(round_idx)
        state_key = self._state_key(model_id, current_client, battery_state, round_idx)
        if battery_state[current_client]['depleted']:
            neighbors = self._neighbors(current_client, battery_state)
            bits_sleep = int(max(self._quant_options_for_sender(current_client)))
            weights = self._adaptive_weights(model_id, battery_state)
            action_record = {
                'model_id': int(model_id),
                'current_client': int(current_client),
                'candidate_neighbors': [int(item) for item in neighbors],
                'candidate_moves': [int(current_client)],
                'next_client': int(current_client),
                'move_decision': False,
                'quant_bits': bits_sleep,
                'candidate_inbound': [],
                'selected_inbound': [],
                'epsilon': float(self.epsilon),
                'curiosity_temperature': float(self.curiosity_temperature),
                'weights': {'accuracy': weights[0], 'energy': weights[1], 'latency': weights[2]},
                'state': state_key,
                'comprehensive_mix': False,
                'comprehensive_prob': 0.0,
                'stability_progress': float(self._stability_progress(round_idx)),
                'effective_ucb_c': float(self._effective_ucb_c(round_idx)),
                'route_mode': str(self.route_mode),
                'inbound_mode': str(self.inbound_mode),
                'round_idx': int(round_idx),
            }
            self.action_history.append(action_record)
            return action_record

        neighbors = self._neighbors(current_client, battery_state)
        move_targets = list(neighbors)
        if current_client not in move_targets:
            move_targets.append(current_client)
        if not move_targets:
            move_targets = [current_client]
        actions = [
            (neighbor_idx, bits)
            for neighbor_idx in move_targets
            for bits in self._quant_options_for_sender(current_client)
        ]
        comm_floor = self._comm_bit_floor_from_acc_drop(current_client)
        actions = [(d, b) for d, b in actions if b >= comm_floor]
        if not actions:
            max_bits = max(self._quant_options_for_sender(current_client))
            actions = [(n, max_bits) for n in move_targets]
        q_values = self.q_table.setdefault(state_key, {})
        for action in actions:
            if action not in q_values:
                q_values[action] = self._initial_action_value(model_id, current_client, action[0], action[1], battery_state, round_idx)

        used_comprehensive = False
        comprehensive_prob = self._comprehensive_mix_probability(model_id, round_idx)
        if self.route_mode == 'comprehensive':
            rule_dst = self._comprehensive_best_dst(
                model_id, current_client, last_visit_round, round_idx, battery_state
            )
            cand = [a for a in actions if a[0] == rule_dst] or list(actions)
            scored = [
                (
                    self._action_ucb_score(state_key, action, q_values[action], round_idx)
                    + self._stability_action_bonus(current_client, action, round_idx),
                    action,
                )
                for action in cand
            ]
            best_value = max(s for s, _ in scored)
            best_actions = [a for s, a in scored if abs(s - best_value) < 1e-12]
            next_client, bits = self._curiosity_tie_break_action(
                model_id, current_client, best_actions, battery_state, round_idx
            )
            used_comprehensive = True
            comprehensive_prob = 1.0
        elif (
            comprehensive_prob > 0
            and last_visit_round is not None
            and random.random() < comprehensive_prob
        ):
            rule_dst = self._comprehensive_best_dst(model_id, current_client, last_visit_round, round_idx, battery_state)
            cand = [a for a in actions if a[0] == rule_dst]
            if not cand:
                cand = list(actions)
            scored = [
                (
                    self._action_ucb_score(state_key, action, q_values[action], round_idx)
                    + self._stability_action_bonus(current_client, action, round_idx),
                    action,
                )
                for action in cand
            ]
            best_value = max(s for s, _ in scored)
            best_actions = [a for s, a in scored if abs(s - best_value) < 1e-12]
            if len(best_actions) > 1:
                best_cov = max(self._node_coverage_reward(model_id, a[0], round_idx) for a in best_actions)
                best_actions = [
                    a for a in best_actions
                    if self._node_coverage_reward(model_id, a[0], round_idx) >= best_cov - 1e-12
                ]
            next_client, bits = self._curiosity_tie_break_action(model_id, current_client, best_actions, battery_state, round_idx)
            used_comprehensive = True
        elif random.random() < self.epsilon:
            next_client, bits = self._sample_curiosity_action(model_id, current_client, actions, battery_state, round_idx)
        else:
            scored = [
                (
                    self._action_ucb_score(state_key, action, q_values[action], round_idx)
                    + self._stability_action_bonus(current_client, action, round_idx),
                    action,
                )
                for action in actions
            ]
            best_value = max(s for s, _ in scored)
            best_actions = [a for s, a in scored if abs(s - best_value) < 1e-12]
            if len(best_actions) > 1:
                best_cov = max(self._node_coverage_reward(model_id, a[0], round_idx) for a in best_actions)
                best_actions = [
                    a for a in best_actions
                    if self._node_coverage_reward(model_id, a[0], round_idx) >= best_cov - 1e-12
                ]
            next_client, bits = self._curiosity_tie_break_action(model_id, current_client, best_actions, battery_state, round_idx)

        weights = self._adaptive_weights(model_id, battery_state)
        selected_inbound, inbound_candidates = self._select_inbound_models(
            model_id, current_client, model_locations, battery_state, state_key, round_idx
        )
        action_record = {
            'model_id': int(model_id),
            'current_client': int(current_client),
            'candidate_neighbors': [int(item) for item in neighbors],
            'candidate_moves': [int(item) for item in move_targets],
            'next_client': int(next_client),
            'move_decision': bool(next_client != current_client),
            'quant_bits': int(bits),
            'candidate_inbound': [dict(item) for item in inbound_candidates],
            'selected_inbound': [dict(item) for item in selected_inbound],
            'epsilon': float(self.epsilon),
            'curiosity_temperature': float(self.curiosity_temperature),
            'weights': {'accuracy': weights[0], 'energy': weights[1], 'latency': weights[2]},
            'state': state_key,
            'comprehensive_mix': bool(used_comprehensive),
            'comprehensive_prob': float(comprehensive_prob),
            'stability_progress': float(self._stability_progress(round_idx)),
            'effective_ucb_c': float(self._effective_ucb_c(round_idx)),
            'route_mode': str(self.route_mode),
            'inbound_mode': str(self.inbound_mode),
            'round_idx': int(round_idx),
        }
        self.action_history.append(action_record)
        return action_record

    def observed_reward(self, model_id, src_idx, action, battery_state, train_signal,
                        training_time, comm_time, train_energy, comm_energy,
                        inbound_comm_time=0.0, inbound_energy=0.0):
        dst_idx = action['next_client']
        bits = action['quant_bits']
        round_idx = int(action.get('round_idx', 0))
        route_signature = self._route_signature(model_id, dst_idx, action.get('selected_inbound', []))

        coverage_gain = self._coverage_reward(model_id, dst_idx, bits, src_idx, candidate_signature=route_signature, round_idx=round_idx)
        if action['selected_inbound']:
            inbound_feature = np.zeros(self.args.num_classes)
            inbound_quality = []
            for inbound in action['selected_inbound']:
                inbound_feature = inbound_feature + self.model_memory[inbound['model_id']]
                inbound_feature = inbound_feature + self.clients[inbound['client_id']].feature_signature
                sender_bits = max(self.clients[inbound['client_id']].quant_bits, self.min_quant_bits)
                inbound_quality.append(self._quant_quality_score(inbound['quant_bits'], sender_bits))
            current_gap = _coverage_gap(self.model_memory[model_id])
            next_gap = _coverage_gap(self.model_memory[model_id] + inbound_feature)
            gap_reduction = max(current_gap - next_gap, 0.0)
            novelty = _novelty_score(self.model_memory[model_id], inbound_feature)
            decay_comp = _decay_compensation(self.model_memory[model_id], inbound_feature)
            retention_gain = self._retention_gain(model_id, inbound_feature, round_idx)
            forgetting_risk = self._forgetting_risk(model_id, inbound_feature, round_idx)
            inbound_gain = (
                self.coverage_gap_weight * gap_reduction + self.unseen_class_weight * novelty + 0.2 * decay_comp
                + self.retention_gain_weight * retention_gain
                - self.forgetting_penalty_weight * forgetting_risk
            ) * float(np.mean(inbound_quality))
            coverage_gain += inbound_gain

        coverage_gain = coverage_gain * self.coverage_gain_weight
        node_coverage_gain = self._node_coverage_reward(model_id, dst_idx, round_idx)
        retention_gain = self._retention_gain(model_id, route_signature, round_idx)
        forgetting_risk = self._forgetting_risk(model_id, route_signature, round_idx)
        accuracy_signal = coverage_gain + self.node_coverage_weight * node_coverage_gain + self.retention_gain_weight * retention_gain - self.forgetting_penalty_weight * forgetting_risk

        total_energy = train_energy + comm_energy + inbound_energy
        energy_penalty = self.energy_penalty_weight * (total_energy / max(battery_state[src_idx]['capacity_j'], 1e-12))

        total_latency = training_time + inbound_comm_time + comm_time
        latency_penalty = self.latency_penalty_weight * (total_latency / max(total_latency + 100.0, 1e-12))

        accuracy_weight, energy_weight, latency_weight = self._adaptive_weights(model_id, battery_state)
        reward = (
            accuracy_weight * accuracy_signal
            + energy_weight * self._energy_reward(src_idx, dst_idx, bits, battery_state)
            + latency_weight * self._latency_reward(src_idx, dst_idx, bits)
            - energy_penalty
            - latency_penalty
            + self.low_bit_bonus_weight
            * (energy_weight + latency_weight)
            * self._compression_bonus(bits, max(self.clients[src_idx].quant_bits, self.min_quant_bits))
        )

        self.reward_history.append({
            'model_id': int(model_id),
            'src': int(src_idx),
            'dst': int(dst_idx),
            'quant_bits': int(bits),
            'train_quant_bits': int(train_signal.get('train_quant_bits', self.min_quant_bits)),
            'selected_inbound': [dict(item) for item in action['selected_inbound']],
            'reward': float(reward),
            'coverage_gain': float(coverage_gain),
            'node_coverage_gain': float(node_coverage_gain),
            'retention_gain': float(retention_gain),
            'forgetting_risk': float(forgetting_risk),
            'energy_penalty': float(energy_penalty),
            'latency_penalty': float(latency_penalty),
            'loss': float(train_signal.get('loss', 0.0)),
            'confidence': float(train_signal.get('confidence', 0.0)),
            'entropy': float(train_signal.get('entropy', 0.0)),
        })
        return reward

    def observe(self, model_id, src_idx, action, reward, battery_state, round_idx):
        state_key = action['state']
        next_state_key = self._state_key(model_id, action['next_client'], battery_state, round_idx)
        next_client = int(action['next_client'])
        next_best = self._next_state_best_q(model_id, next_client, next_state_key, battery_state, round_idx)

        q_values = self.q_table.setdefault(state_key, {})
        pair_action = (next_client, int(action['quant_bits']))
        self.visit_sa[(state_key, pair_action)] += 1
        old_value = q_values.get(pair_action, 0.0)
        td_target = reward + self.discount * next_best
        q_values[pair_action] = (1.0 - self.lr) * old_value + self.lr * td_target

        if self.meta_prior_enabled:
            self._update_device_utility(model_id, next_client, td_target)

        if self.replay_buffer is not None:
            self.replay_buffer.append({
                'model_id': int(model_id),
                'state_key': state_key,
                'next_client': next_client,
                'action': pair_action,
                'reward': float(reward),
                'next_state_key': next_state_key,
                'battery': copy.deepcopy(battery_state),
                'round_idx': int(round_idx),
            })
            for _ in range(self.replay_steps):
                if not self.replay_buffer:
                    break
                self._replay_td(random.choice(list(self.replay_buffer)))

        self.epsilon = max(self._epsilon_floor(round_idx), self.epsilon * self.epsilon_decay)

    def absorb_inbound_memory(self, model_id, selected_inbound, round_idx=None):
        inbound_feature = np.zeros(self.args.num_classes)
        for inbound in selected_inbound:
            self.model_memory[model_id] = self.model_memory[model_id] + self.model_memory[inbound['model_id']]
            self.model_memory[model_id] = self.model_memory[model_id] + self.clients[inbound['client_id']].feature_signature
            inbound_feature = inbound_feature + self.model_memory[inbound['model_id']]
            inbound_feature = inbound_feature + self.clients[inbound['client_id']].feature_signature
        if round_idx is not None:
            active_classes = np.where(inbound_feature > 0)[0]
            for class_idx in active_classes:
                self.class_last_seen_round[model_id, class_idx] = int(round_idx)

    def snapshot_state(self, round_idx):
        snapshot = []
        for model_id, memory in enumerate(self.model_memory):
            normalized = _safe_normalize(memory)
            snapshot.append({
                'round': int(round_idx),
                'model_id': int(model_id),
                'coverage_gap': float(_coverage_gap(memory)),
                'missing_ratio': float(np.mean(normalized <= 0)) if len(normalized) else 0.0,
                'entropy': float(_entropy_score(memory)),
                'node_coverage_ratio': float(self._node_coverage_ratio(model_id)),
                'global_node_coverage_ratio': float(self._global_node_coverage_ratio()),
                'forgetting_pressure': float(self._forgetting_pressure(model_id, round_idx)),
                'quality': dict(self.model_quality[model_id]),
                'train_pressure': float(self.train_pressure[model_id]),
                'epsilon': float(self.epsilon),
                'curiosity_temperature': float(self.curiosity_temperature),
                'curiosity_topk': int(self.curiosity_topk),
            })
        self.state_history.extend(snapshot)
        return snapshot


def AutoRL_DFL(args, net_glob, dataset_train, dataset_test, dict_users):
    args.client_selection = 'autorl'
    quant_enabled = bool(getattr(args, 'autorl_quant_aware', 1))
    validation_rollback = bool(int(getattr(args, 'autorl_validation_rollback', 0)))
    checkpoint_metric = str(getattr(args, 'autorl_checkpoint_metric', 'proxy')).strip().lower()
    eval_test_every_round = bool(int(getattr(args, 'autorl_eval_test_every_round', 1))) and dataset_test is not None
    final_test_eval = bool(int(getattr(args, 'autorl_final_test_eval', 0))) and dataset_test is not None
    runtime_checkpoint_enabled = bool(int(getattr(args, 'autorl_runtime_checkpoint', 0)))
    runtime_checkpoint_every = max(int(getattr(args, 'autorl_checkpoint_every', 1)), 1)
    runtime_resume = bool(int(getattr(args, 'autorl_resume', 1)))
    result_dir = str(getattr(args, 'result_dir', '') or '').strip()
    if runtime_checkpoint_enabled and not result_dir:
        raise ValueError('--autorl_runtime_checkpoint requires --result_dir')
    runtime_checkpoint_path = (
        os.path.join(os.path.abspath(result_dir), RUNTIME_CHECKPOINT_NAME)
        if runtime_checkpoint_enabled else None
    )
    if checkpoint_metric not in ('proxy', 'test_acc'):
        checkpoint_metric = 'proxy'
    if checkpoint_metric == 'test_acc' and not eval_test_every_round:
        print('[AutoRL] checkpoint_metric=test_acc requested but no round-wise test evaluation is enabled; falling back to proxy.')
        checkpoint_metric = 'proxy'
    comm_8bit_format = getattr(args, 'quant_comm_8bit_format', 'int8')
    net_size = sum([param.nelement() for param in net_glob.parameters()])
    base_payload_bytes = max(state_dict_payload_nbytes(net_glob, args.quant_comm_base_bits, enabled=False), 1)
    payload_codec = getattr(args, 'payload_codec', 'none')
    payload_compression_level = getattr(args, 'payload_compression_level', 6)
    battery_sleep_scale = getattr(args, 'autorl_sleep_energy_scale', None)
    if battery_sleep_scale is None:
        battery_sleep_scale = getattr(args, 'autorl_idle_energy_scale', 1.0)
    if battery_sleep_scale is None:
        battery_sleep_scale = 1.0
    battery_sleep_scale = max(float(battery_sleep_scale), 0.0)
    target_time = dict()
    target_comm = dict()
    target_energy = dict()
    target_acc1 = 55
    target_acc2 = 60
    acc = dict()
    model_cnt = max(1, int(args.num_users * args.frac))

    model_set = []
    clients = []
    for model_id in range(model_cnt):
        model_set.append(copy.deepcopy(net_glob))
        acc["model" + str(model_id)] = []
    for idx in range(args.num_users):
        client = Client(idx, dict_users[idx], net_glob, args)
        client.build_feature_signature(dataset_train)
        clients.append(client)

    battery_state = initialize_battery_state(clients)
    _apply_autorl_battery_capacity_scale(battery_state, args)
    controller = SelfEvolvingRLController(args, clients, model_cnt, quant_enabled=quant_enabled, ref_model=net_glob)
    training_client = random.sample(list(range(args.num_users)), model_cnt)
    last_visit_round = [0 for _ in range(args.num_users)]

    acc["acc"] = []
    time_consume = []
    comm_consume = []
    comm_time_consume = []
    energy_consume = []
    battery_history = []
    state_history = []
    precision_history = []
    coverage_history = []
    proxy_history = []
    current_time = 0
    current_comm = 0
    current_comm_time = 0
    current_energy = 0
    best_score = float('-inf')
    best_eval_acc = float('-inf')
    best_round_score = -1
    best_round_eval = -1
    best_round = -1
    best_model_set = None
    best_training_client = None
    best_last_visit_round = None
    best_battery_state = None
    final_test_result = None
    start_round = 0

    if runtime_checkpoint_path and runtime_resume and os.path.exists(runtime_checkpoint_path):
        checkpoint = torch.load(runtime_checkpoint_path, map_location='cpu', weights_only=False)
        expected_signature = _runtime_checkpoint_signature(args)
        if checkpoint.get('signature') != expected_signature:
            raise RuntimeError(
                'runtime checkpoint configuration mismatch: {} != {}'.format(
                    checkpoint.get('signature'), expected_signature,
                )
            )
        _load_model_state_list(model_set, checkpoint['model_set'])
        for key, value in checkpoint['controller_state'].items():
            setattr(controller, key, value)
        training_client = list(checkpoint['training_client'])
        last_visit_round = list(checkpoint['last_visit_round'])
        battery_state = checkpoint['battery_state']
        acc = checkpoint['acc']
        time_consume = checkpoint['time_consume']
        comm_consume = checkpoint['comm_consume']
        comm_time_consume = checkpoint['comm_time_consume']
        energy_consume = checkpoint['energy_consume']
        battery_history = checkpoint['battery_history']
        state_history = checkpoint['state_history']
        precision_history = checkpoint['precision_history']
        coverage_history = checkpoint['coverage_history']
        proxy_history = checkpoint['proxy_history']
        current_time = checkpoint['current_time']
        current_comm = checkpoint['current_comm']
        current_comm_time = checkpoint['current_comm_time']
        current_energy = checkpoint['current_energy']
        best_score = checkpoint['best_score']
        best_eval_acc = checkpoint['best_eval_acc']
        best_round_score = checkpoint['best_round_score']
        best_round_eval = checkpoint['best_round_eval']
        best_round = checkpoint['best_round']
        if checkpoint.get('best_model_set') is not None:
            best_model_set = [copy.deepcopy(net_glob) for _ in checkpoint['best_model_set']]
            _load_model_state_list(best_model_set, checkpoint['best_model_set'])
        best_training_client = checkpoint['best_training_client']
        best_last_visit_round = checkpoint['best_last_visit_round']
        best_battery_state = checkpoint['best_battery_state']
        target_time = checkpoint['target_time']
        target_comm = checkpoint['target_comm']
        target_energy = checkpoint['target_energy']
        start_round = int(checkpoint['next_round'])
        _restore_rng_state(checkpoint['rng_state'])
        print('[AutoRL] resumed runtime checkpoint:', runtime_checkpoint_path, 'next round:', start_round)

    def save_runtime_checkpoint(next_round):
        if not runtime_checkpoint_path:
            return
        controller_state = {
            key: value for key, value in vars(controller).items()
            if key not in ('args', 'clients')
        }
        payload = {
            'version': 1,
            'signature': _runtime_checkpoint_signature(args),
            'next_round': int(next_round),
            'model_set': _model_state_list(model_set),
            'controller_state': controller_state,
            'training_client': list(training_client),
            'last_visit_round': list(last_visit_round),
            'battery_state': battery_state,
            'acc': acc,
            'time_consume': time_consume,
            'comm_consume': comm_consume,
            'comm_time_consume': comm_time_consume,
            'energy_consume': energy_consume,
            'battery_history': battery_history,
            'state_history': state_history,
            'precision_history': precision_history,
            'coverage_history': coverage_history,
            'proxy_history': proxy_history,
            'current_time': current_time,
            'current_comm': current_comm,
            'current_comm_time': current_comm_time,
            'current_energy': current_energy,
            'best_score': best_score,
            'best_eval_acc': best_eval_acc,
            'best_round_score': best_round_score,
            'best_round_eval': best_round_eval,
            'best_round': best_round,
            'best_model_set': _model_state_list(best_model_set) if best_model_set is not None else None,
            'best_training_client': best_training_client,
            'best_last_visit_round': best_last_visit_round,
            'best_battery_state': best_battery_state,
            'target_time': target_time,
            'target_comm': target_comm,
            'target_energy': target_energy,
            'rng_state': _capture_rng_state(),
        }
        _atomic_torch_save(payload, runtime_checkpoint_path)
        print('[AutoRL] runtime checkpoint saved:', runtime_checkpoint_path, 'next round:', next_round)

    for iter in range(start_round, args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        controller.current_round = int(iter)
        round_locations = list(training_client)
        print("autorl client:", round_locations)
        round_time = 0.0
        round_comm = 0.0
        round_comm_time = 0.0
        round_energy = 0.0
        busy_time = {idx: 0.0 for idx in range(args.num_users)}
        round_train_bits = []
        round_comm_bits = []
        round_comm_payload_bytes = []
        round_train_policies = []
        round_reward_weights = []
        trained_model_set = [None for _ in range(model_cnt)]
        train_signal_set = [None for _ in range(model_cnt)]
        train_time_set = [0.0 for _ in range(model_cnt)]
        train_energy_set = [0.0 for _ in range(model_cnt)]
        pre_train_battery_state = snapshot_battery_state(battery_state)
        autorl_e_s = _autorl_battery_energy_scale(args)

        if args.aggregation:
            grouped_models = defaultdict(list)
            for model_id, idx in enumerate(round_locations):
                grouped_models[idx].append(model_id)
            for idx, model_ids in grouped_models.items():
                if len(model_ids) > 1:
                    aggregation_bits = max(max(controller.train_quant_options), args.quant_comm_base_bits)
                    agg_policy = (
                        controller.layerwise_bits_policy(aggregation_bits)
                        if controller.layer_mixed_precision
                        else aggregation_bits
                    )
                    states = [
                        quantized_state_dict(model_set[model_id], agg_policy, quant_enabled, comm_8bit_format=comm_8bit_format)
                        for model_id in model_ids
                    ]
                    agg_state = Aggregation(states, [max(clients[idx].data_cnt, 1) for _ in model_ids])
                    for model_id in model_ids:
                        model_set[model_id].load_state_dict(agg_state)

        for model_id, idx in enumerate(round_locations):
            clients[idx].local_net = copy.deepcopy(model_set[model_id])
            if battery_state[idx]['depleted']:
                train_quant_bits, train_policy = controller.select_train_quant_bits(model_id, idx, pre_train_battery_state)
                trained_model_set[model_id] = copy.deepcopy(model_set[model_id])
                train_signal_set[model_id] = {
                    'loss': 0.0,
                    'confidence': 0.0,
                    'entropy': 0.0,
                    'train_quant_bits': int(train_quant_bits),
                }
                train_time_set[model_id] = 0.0
                train_energy_set[model_id] = 0.0
                round_train_bits.append(int(train_quant_bits))
                round_train_policies.append(dict(train_policy))
                controller.record_visit(model_id, idx, iter, update_class_memory=False)
                continue
            local = LocalUpdate_AutoRL(args=args, dataset=dataset_train, quant_enabled=quant_enabled)
            local_scale = controller.get_training_scale(model_id)
            train_quant_bits, train_policy = controller.select_train_quant_bits(model_id, idx, pre_train_battery_state)
            train_bits_policy = controller.layerwise_bits_policy(train_quant_bits) if controller.layer_mixed_precision else None
            trained_net, train_signal = local.train(
                client=clients[idx],
                round=iter,
                local_ep_scale=local_scale,
                train_quant_bits=train_quant_bits,
                bits_policy=train_bits_policy,
            )
            train_signal['train_quant_bits'] = int(train_quant_bits)
            trained_model_set[model_id] = copy.deepcopy(trained_net)
            train_signal_set[model_id] = train_signal
            effective_local_ep = int(train_signal.get('effective_local_ep', args.local_ep))
            ep_scale = effective_local_ep / max(int(args.local_ep), 1)
            train_time = get_client_training_time(idx) * ep_scale * controller.training_time_multiplier(train_quant_bits)
            train_energy = (
                get_training_energy(idx, train_time)
                * controller.training_energy_multiplier(train_quant_bits)
                * autorl_e_s
            )
            train_time_set[model_id] = train_time
            train_energy_set[model_id] = train_energy
            consume_energy(battery_state, idx, train_energy)
            round_time = max(round_time, train_time)
            round_energy += train_energy
            busy_time[idx] += train_time
            controller.update_model_memory(model_id, idx, train_signal, iter)
            controller.record_visit(model_id, idx, iter, update_class_memory=True)
            round_train_bits.append(int(train_quant_bits))
            round_train_policies.append(dict(train_policy))

        snapshot_model_set = [copy.deepcopy(trained_model_set[model_id]) for model_id in range(model_cnt)]
        model_set = [copy.deepcopy(snapshot_model_set[model_id]) for model_id in range(model_cnt)]
        decision_battery_state = snapshot_battery_state(battery_state)

        for model_id, idx in enumerate(round_locations):
            action = controller.select_action(model_id, idx, decision_battery_state, iter, round_locations, last_visit_round)
            action['train_quant_bits'] = int(train_signal_set[model_id].get('train_quant_bits', controller.min_quant_bits))
            action['train_precision_policy'] = dict(round_train_policies[model_id]) if model_id < len(round_train_policies) else {}
            next_client = action['next_client']
            quant_bits = action['quant_bits']
            if battery_state[idx]['depleted']:
                model_set[model_id] = copy.deepcopy(snapshot_model_set[model_id])
                comm_time = 0.0
                src_energy, dst_energy = 0.0, 0.0
                inbound_comm_time = 0.0
                inbound_energy = 0.0
                training_client[model_id] = idx
                reward = controller.observed_reward(
                    model_id,
                    idx,
                    action,
                    decision_battery_state,
                    train_signal_set[model_id],
                    train_time_set[model_id],
                    comm_time,
                    train_energy_set[model_id],
                    0.0,
                    inbound_comm_time=inbound_comm_time,
                    inbound_energy=inbound_energy,
                )
                controller.reward_history[-1]['reward'] = float(reward)
                controller.observe(model_id, idx, action, reward, battery_state, iter + 1)
                last_visit_round[idx] = iter + 1
                round_time = max(round_time, train_time_set[model_id] + inbound_comm_time + comm_time)
                continue

            effective_bits = quant_bits if quant_enabled else args.quant_comm_base_bits

            train_tb = train_signal_set[model_id].get('train_quant_bits', controller.min_quant_bits)
            own_bits_arg = controller.layerwise_bits_policy(train_tb) if controller.layer_mixed_precision else train_tb
            own_state_dict, own_payload_bytes, own_payload_meta = transmit_state_dict(
                snapshot_model_set[model_id],
                own_bits_arg,
                quant_enabled,
                codec=payload_codec,
                compression_level=payload_compression_level,
                comm_8bit_format=comm_8bit_format,
            )
            inbound_states = [own_state_dict]
            inbound_weights = [max(clients[idx].data_cnt, 1)]
            inbound_comm_time = 0.0
            inbound_energy = 0.0

            for inbound in action['selected_inbound']:
                sender_model_id = inbound['model_id']
                sender_client = inbound['client_id']
                sender_bits = inbound['quant_bits']
                sender_effective_bits = sender_bits if quant_enabled else args.quant_comm_base_bits
                sender_bits_arg = (
                    controller.layerwise_bits_policy(sender_bits) if controller.layer_mixed_precision else sender_bits
                )
                sender_state_dict, sender_payload_bytes, sender_payload_meta = transmit_state_dict(
                    snapshot_model_set[sender_model_id],
                    sender_bits_arg,
                    quant_enabled,
                    codec=payload_codec,
                    compression_level=payload_compression_level,
                    comm_8bit_format=comm_8bit_format,
                )
                sender_comm_ratio = max(sender_payload_bytes / float(base_payload_bytes), 1e-12)
                sender_comm_time = get_client_communication_time(sender_client, idx, multiplier=sender_comm_ratio)
                sender_src_energy, sender_dst_energy = get_communication_energy_breakdown(sender_client, idx, sender_comm_time)
                sender_src_energy *= autorl_e_s
                sender_dst_energy *= autorl_e_s
                inbound_states.append(sender_state_dict)
                inbound_weights.append(max(clients[sender_client].data_cnt, 1))
                inbound_comm_time += sender_comm_time
                round_comm_time += sender_comm_time
                inbound_energy += sender_src_energy + sender_dst_energy
                round_comm += sender_payload_bytes / (1024 * 1024)
                round_energy += sender_src_energy + sender_dst_energy
                busy_time[sender_client] += sender_comm_time
                busy_time[idx] += sender_comm_time
                consume_communication_energy(battery_state, sender_client, idx, sender_src_energy, sender_dst_energy)
                round_comm_bits.append(int(sender_bits))
                round_comm_payload_bytes.append(int(sender_payload_bytes))

            if len(inbound_states) > 1:
                agg_state = Aggregation(inbound_states, inbound_weights)
                model_set[model_id].load_state_dict(agg_state)

            if next_client != idx:
                move_bits_arg = controller.layerwise_bits_policy(quant_bits) if controller.layer_mixed_precision else quant_bits
                next_state_dict, move_payload_bytes, move_payload_meta = transmit_state_dict(
                    model_set[model_id],
                    move_bits_arg,
                    quant_enabled,
                    codec=payload_codec,
                    compression_level=payload_compression_level,
                    comm_8bit_format=comm_8bit_format,
                )
                model_set[model_id].load_state_dict(next_state_dict)
                comm_ratio = max(move_payload_bytes / float(base_payload_bytes), 1e-12)
                comm_time = get_client_communication_time(idx, next_client, multiplier=comm_ratio)
                src_energy, dst_energy = get_communication_energy_breakdown(idx, next_client, comm_time)
                src_energy *= autorl_e_s
                dst_energy *= autorl_e_s
                busy_time[idx] += comm_time
                busy_time[next_client] += comm_time
                consume_communication_energy(battery_state, idx, next_client, src_energy, dst_energy)
                round_comm_time += comm_time
            else:
                comm_time = 0.0
                src_energy, dst_energy = 0.0, 0.0
                move_payload_bytes = 0
                move_payload_meta = {'payload_bytes': 0, 'data_bytes': 0, 'metadata_bytes': 0, 'tensor_count': 0}
            round_comm += move_payload_bytes / (1024 * 1024) if next_client != idx else 0.0
            round_energy += src_energy + dst_energy
            if next_client != idx:
                round_comm_bits.append(int(quant_bits))
                round_comm_payload_bytes.append(int(move_payload_bytes))

            round_reward_weights.append(dict(action.get('weights', {})))

            reward = controller.observed_reward(
                model_id,
                idx,
                action,
                decision_battery_state,
                train_signal_set[model_id],
                train_time_set[model_id],
                comm_time,
                train_energy_set[model_id],
                src_energy + dst_energy,
                inbound_comm_time=inbound_comm_time,
                inbound_energy=inbound_energy,
            )
            controller.reward_history[-1]['reward'] = float(reward)
            if action['selected_inbound']:
                controller.absorb_inbound_memory(model_id, action['selected_inbound'], iter)
            controller.observe(model_id, idx, action, reward, battery_state, iter + 1)

            training_client[model_id] = next_client
            last_visit_round[next_client] = iter + 1
            round_time = max(round_time, train_time_set[model_id] + inbound_comm_time + comm_time)

        sleep_energy = consume_sleep_energy(battery_state, round_time, busy_time_by_client=busy_time, scale=battery_sleep_scale)
        round_energy += sleep_energy
        current_time += round_time
        current_comm += round_comm
        current_comm_time += round_comm_time
        comm_time_consume.append(current_comm_time)
        current_energy = sum([node_state['used_j'] for node_state in battery_state])
        proxy_score, proxy_detail = _round_proxy_score(
            controller,
            train_signal_set,
            round_energy,
            round_time,
            round_comm,
            battery_state,
            iter,
        )
        proxy_history.append({
            'round': int(iter),
            **proxy_detail,
        })
        precision_history.append({
            'round': int(iter),
            'train_bits': [int(bit) for bit in round_train_bits],
            'train_bit_summary': _bit_counts(round_train_bits, controller.train_quant_options),
            'comm_bits': [int(bit) for bit in round_comm_bits],
            'comm_bit_summary': _bit_counts(round_comm_bits, controller.quant_options),
            'comm_payload_bytes': [int(value) for value in round_comm_payload_bytes],
            'comm_payload_summary': {
                'total_bytes': int(sum(round_comm_payload_bytes)),
                'mean_bytes': float(np.mean(round_comm_payload_bytes)) if round_comm_payload_bytes else 0.0,
                'ratio_to_base': float(sum(round_comm_payload_bytes) / max(base_payload_bytes, 1)) if round_comm_payload_bytes else 0.0,
            },
            'train_policy': _mean_train_policy(round_train_policies, controller.train_quant_options),
            'reward_weights': _mean_weight_records(round_reward_weights),
            'current_epsilon': float(controller.epsilon),
            'epsilon_floor': float(controller._epsilon_floor(iter)),
            'stability_progress': float(controller._stability_progress(iter)),
            'effective_ucb_c': float(controller._effective_ucb_c(iter)),
            'curiosity_temperature': float(controller.curiosity_temperature),
            'curiosity_topk': int(controller.curiosity_topk),
            'train_pressure_mean': float(np.mean(controller.train_pressure)) if len(controller.train_pressure) else 0.0,
            'global_node_coverage_ratio': float(controller._global_node_coverage_ratio()),
            'model_node_coverage_mean': float(np.mean([controller._node_coverage_ratio(mid) for mid in range(model_cnt)])) if model_cnt else 0.0,
            'forgetting_pressure_mean': float(np.mean([controller._forgetting_pressure(mid, iter) for mid in range(model_cnt)])) if model_cnt else 0.0,
            'proxy_score': float(proxy_score),
        })
        controller.train_precision_history.append(precision_history[-1])
        coverage_history.append({
            'round': int(iter),
            'global_node_coverage_ratio': float(controller._global_node_coverage_ratio()),
            'model_node_coverage_mean': float(np.mean([controller._node_coverage_ratio(mid) for mid in range(model_cnt)])) if model_cnt else 0.0,
            'forgetting_pressure_mean': float(np.mean([controller._forgetting_pressure(mid, iter) for mid in range(model_cnt)])) if model_cnt else 0.0,
            'model_node_coverage_ratio': [float(controller._node_coverage_ratio(mid)) for mid in range(model_cnt)],
        })
        controller.coverage_history.append(coverage_history[-1])
        eval_acc = None
        if eval_test_every_round:
            avg_acc = 0.0
            for model_id in range(model_cnt):
                acc["model" + str(model_id)].append(test(model_set[model_id], dataset_test, args))
                avg_acc += acc["model" + str(model_id)][-1]
                print("model ", model_id, "acc: ", acc["model" + str(model_id)][-1])
            eval_acc = float(avg_acc / model_cnt)
            acc["acc"].append(eval_acc)
            if eval_acc > best_eval_acc + 1e-12:
                best_eval_acc = float(eval_acc)
                best_round_eval = int(iter)
        else:
            for model_id in range(model_cnt):
                acc["model" + str(model_id)].append(float(proxy_score))
                print("model ", model_id, "proxy metric: ", float(proxy_score))
            acc["acc"].append(float(proxy_score))

        feedback_value = float(eval_acc) if (checkpoint_metric == 'test_acc' and eval_acc is not None) else float(proxy_score)
        controller.record_round_performance(feedback_value)
        if feedback_value > best_score + 1e-12:
            best_score = float(feedback_value)
            best_round_score = int(iter)
            best_round = int(iter)
            best_model_set = [copy.deepcopy(model) for model in model_set]
            best_training_client = list(training_client)
            best_last_visit_round = list(last_visit_round)
            best_battery_state = snapshot_battery_state(battery_state)
        time_consume.append(current_time)
        comm_consume.append(current_comm)
        energy_consume.append(current_energy)
        battery_history.append(snapshot_battery_state(battery_state))
        state_history.append(controller.snapshot_state(iter))
        print("acc acc: ", acc["acc"][-1])

        if eval_acc is not None and eval_acc >= target_acc1:
            if target_acc1 not in target_time:
                target_time[target_acc1] = time_consume[-1]
            if target_acc1 not in target_comm:
                target_comm[target_acc1] = comm_consume[-1]
            if target_acc1 not in target_energy:
                target_energy[target_acc1] = energy_consume[-1]
        if eval_acc is not None and eval_acc >= target_acc2:
            if target_acc2 not in target_time:
                target_time[target_acc2] = time_consume[-1]
            if target_acc2 not in target_comm:
                target_comm[target_acc2] = comm_consume[-1]
            if target_acc2 not in target_energy:
                target_energy[target_acc2] = energy_consume[-1]

        if runtime_checkpoint_path and (
            (iter + 1) % runtime_checkpoint_every == 0 or iter + 1 == args.epochs
        ):
            save_runtime_checkpoint(iter + 1)

    best_checkpoint = {
        'enabled': bool(validation_rollback),
        'metric_name': str(checkpoint_metric if checkpoint_metric else 'proxy'),
        'best_round': int(best_round),
        'best_score': float(best_score) if best_score > float('-inf') else None,
        'best_eval_acc': float(best_eval_acc) if best_eval_acc > float('-inf') else None,
        'best_round_score': int(best_round_score),
        'best_round_eval': int(best_round_eval),
        'best_training_client': [int(x) for x in best_training_client] if best_training_client is not None else None,
        'best_last_visit_round': [int(x) for x in best_last_visit_round] if best_last_visit_round is not None else None,
        'best_battery': best_battery_state,
    }
    if validation_rollback and best_model_set is not None:
        model_set = [copy.deepcopy(model) for model in best_model_set]
        if eval_test_every_round:
            restored_avg_acc = 0.0
            for model_id in range(model_cnt):
                acc["model" + str(model_id)][-1] = test(model_set[model_id], dataset_test, args)
                restored_avg_acc += acc["model" + str(model_id)][-1]
            acc["acc"][-1] = restored_avg_acc / model_cnt
            print("rollback to best checkpoint round:", best_round, "best acc:", acc["acc"][-1])
        else:
            acc["acc"][-1] = float(best_score)
            print("rollback to best checkpoint round:", best_round, "best proxy score:", acc["acc"][-1])
        controller.record_round_performance(acc["acc"][-1])

    if final_test_eval:
        final_model_acc = []
        final_avg_acc = 0.0
        for model_id in range(model_cnt):
            final_acc = test(model_set[model_id], dataset_test, args)
            final_model_acc.append(float(final_acc))
            final_avg_acc += float(final_acc)
        final_test_result = {
            'mean_acc': float(final_avg_acc / max(model_cnt, 1)),
            'model_acc': final_model_acc,
            'metric_source': 'final_only_test',
        }
        print('final_test_acc:', final_test_result['mean_acc'])

    save_result(acc, 'test_acc', args)
    save_result(time_consume, 'time', args)
    save_result(comm_consume, 'comm', args)
    save_result(comm_time_consume, 'comm_time', args)
    save_result(energy_consume, 'energy', args)
    save_result(battery_history, 'battery', args)
    save_result(controller.action_history, 'autorl_action', args)
    save_result(controller.reward_history, 'autorl_reward', args)
    save_result(controller.state_history, 'autorl_state', args)
    save_result(precision_history, 'autorl_precision', args)
    save_result(coverage_history, 'autorl_coverage', args)
    save_result(proxy_history, 'autorl_proxy_metric', args)
    save_result(best_checkpoint, 'autorl_best_checkpoint', args)
    if final_test_result is not None:
        save_result(final_test_result, 'autorl_final_test_acc', args)
    save_result({
        'target_acc': [target_acc1, target_acc2],
        'time': target_time,
        'comm': target_comm,
        'energy': target_energy,
    }, 'target_metrics', args)
    save_result({
        'train_quant_options': [int(bit) for bit in controller.train_quant_options],
        'comm_quant_options': [int(bit) for bit in controller.quant_options],
        'train_bit_base_weights': dict(controller.train_bit_base_weights),
        'train_precision_mode': controller.train_precision_mode,
        'quant_quality_gamma': float(controller.quant_quality_gamma),
        'low_bit_bonus_weight': float(controller.low_bit_bonus_weight),
        'train_high_bit_floor': float(controller.train_high_bit_floor),
        'train_precision_boost': float(controller.train_precision_boost),
        'coverage_gain_weight': float(controller.coverage_gain_weight),
        'coverage_gap_weight': float(controller.coverage_gap_weight),
        'unseen_class_weight': float(controller.unseen_class_weight),
        'diversity_weight': float(controller.diversity_weight),
        'diversity_fallback': bool(controller.diversity_fallback),
        'energy_penalty_weight': float(controller.energy_penalty_weight),
        'latency_penalty_weight': float(controller.latency_penalty_weight),
        'meta_lr': float(controller.meta_lr),
        'meta_prior_enabled': bool(controller.meta_prior_enabled),
        'tabular_coarse_state': bool(controller.coarse_state),
        'tabular_ucb_c': float(controller.ucb_c),
        'rl_lr': float(controller.lr),
        'rl_discount': float(controller.discount),
        'replay_capacity': int(controller.replay_capacity),
        'replay_steps': int(controller.replay_steps),
        'comprehensive_mix': float(controller.comprehensive_mix),
        'acc_drop_high_pct': float(controller.acc_drop_high_pct),
        'acc_drop_low_pct': float(controller.acc_drop_low_pct),
        'acc_drop_signal': str(controller.acc_drop_signal),
        'acc_min_bits_until_accuracy': float(controller.acc_min_bits_until_accuracy),
        'comm_bits_drop_schedule': bool(controller.comm_bits_drop_schedule),
        'comm_stable_bits': int(controller.comm_stable_bits),
        'late_comm_min_bits': int(controller.late_comm_min_bits),
        'late_train_min_bits': int(controller.late_train_min_bits),
        'stability_start_frac': float(controller.stability_start_frac),
        'late_min_epsilon': float(controller.late_min_epsilon),
        'late_comprehensive_mix': float(controller.late_comprehensive_mix),
        'late_ucb_c_scale': float(controller.late_ucb_c_scale),
        'late_stay_bonus': float(controller.late_stay_bonus),
        'late_inbound_threshold_boost': float(controller.late_inbound_threshold_boost),
        'late_inbound_gap_threshold': float(controller.late_inbound_gap_threshold),
        'inbound_score_threshold': float(controller.inbound_score_threshold),
        'battery_capacity_scale': float(getattr(args, 'autorl_battery_capacity_scale', 1.0)),
        'prox_mu': float(getattr(args, 'autorl_prox_mu', 0.0)),
        'prox_start_frac': float(getattr(args, 'autorl_prox_start_frac', 0.35)),
        'curiosity_temperature': float(controller.curiosity_temperature),
        'curiosity_topk': int(controller.curiosity_topk),
        'train_scale_max': float(controller.train_scale_max),
        'validation_rollback': bool(validation_rollback),
        'checkpoint_metric': str(checkpoint_metric),
        'eval_test_every_round': bool(eval_test_every_round),
        'final_test_eval': bool(final_test_eval),
        'sleep_energy_scale': float(battery_sleep_scale),
        'device_utility': controller.device_utility.tolist(),
        'device_visit_count': controller.device_visit_count.tolist(),
        'device_td_accum': controller.device_td_accum.tolist(),
        'node_coverage_weight': float(controller.node_coverage_weight),
        'global_coverage_weight': float(controller.global_coverage_weight),
        'retention_gain_weight': float(controller.retention_gain_weight),
        'forgetting_penalty_weight': float(controller.forgetting_penalty_weight),
        'visit_staleness_weight': float(controller.visit_staleness_weight),
        'forgetting_horizon': int(controller.forgetting_horizon),
        'quant_enabled': bool(controller.quant_enabled),
        'experiment_tag': str(getattr(args, 'experiment_tag', '')),
        'route_mode': str(controller.route_mode),
        'inbound_mode': str(controller.inbound_mode),
        'reward_accuracy_scale': float(controller.reward_accuracy_scale),
        'reward_energy_scale': float(controller.reward_energy_scale),
        'reward_latency_scale': float(controller.reward_latency_scale),
    }, 'autorl_hparams', args)
    print("target_time:", target_time)
    print("target_comm:", target_comm)
    print("target_energy:", target_energy)
    for key in acc.keys():
        print(key)
        avg_acc_and_var(acc[key])
