import random
from collections import defaultdict

import numpy as np

from utils.FL_utils import uniform_distribution_loss
from utils.quantization import get_quant_comm_ratio
from config import DEVICE_TYPE_PROFILES, TRAINING_TIME_MEAN, get_communication_energy_breakdown

MIN_QUANT_BITS = 8


def _mean_comm_time(edge_type):
    if edge_type == 'weak':
        return 10.0
    if edge_type == 'middle':
        return 5.0
    if edge_type == 'strong':
        return 2.0
    return 0.0


class DeviceWalkRLController(object):
    def __init__(self, args, clients, edge_types):
        self.args = args
        self.clients = clients
        self.edge_types = edge_types
        self.alpha = getattr(args, 'rl_accuracy_weight', 0.65)
        self.beta = getattr(args, 'rl_energy_weight', 0.25)
        self.gamma = getattr(args, 'rl_latency_weight', 0.10)
        self.lr = getattr(args, 'rl_lr', 0.2)
        self.discount = getattr(args, 'rl_discount', 0.9)
        self.epsilon = getattr(args, 'rl_epsilon', 0.1)
        self.state_bins = max(int(getattr(args, 'rl_state_bins', 4)), 2)
        self.quant_base_bits = max(int(getattr(args, 'quant_comm_base_bits', 32)), MIN_QUANT_BITS)
        self.q_table = defaultdict(dict)
        self.pending = {}
        self.reward_history = []

    def _bucket(self, value):
        value = max(min(float(value), 0.999999), 0.0)
        return int(value * self.state_bins)

    def _coverage_features(self, model_distribution):
        dist = np.asarray(model_distribution, dtype=float)
        total = dist.sum()
        if total <= 0:
            normalized = np.zeros_like(dist, dtype=float)
        else:
            normalized = dist / total
        coverage_gap = float(uniform_distribution_loss(dist))
        missing_ratio = float(np.mean(normalized == 0)) if len(normalized) else 0.0
        entropy = 0.0
        if total > 0:
            entropy = float(-np.sum(np.where(normalized > 0, normalized * np.log(normalized + 1e-12), 0.0)))
            entropy = entropy / max(np.log(max(len(normalized), 2)), 1e-12)
        return coverage_gap, missing_ratio, entropy, normalized

    def _state_key(self, agent_id, current_client, model_distribution, battery_state, round_idx):
        coverage_gap, missing_ratio, entropy, _ = self._coverage_features(model_distribution)
        battery_ratio = battery_state[current_client]['remaining_j'] / max(battery_state[current_client]['capacity_j'], 1e-12)
        progress_ratio = round_idx / max(self.args.epochs - 1, 1)
        return (
            int(agent_id),
            int(current_client),
            self._bucket(coverage_gap / max(len(model_distribution), 1)),
            self._bucket(missing_ratio),
            self._bucket(entropy),
            self._bucket(battery_ratio),
            self._bucket(progress_ratio),
        )

    def _neighbors(self, current_client, adjacency_matrix):
        return [nb_idx for nb_idx in range(self.args.num_users) if adjacency_matrix[current_client][nb_idx] == 1]

    def _expected_training_time(self, client_idx):
        profile = self.clients[client_idx].device_profile
        client_type = profile['type']
        default_freq = DEVICE_TYPE_PROFILES[client_type]['frequency_ghz']
        actual_freq = profile['frequency_ghz']
        return max(TRAINING_TIME_MEAN[client_type] * default_freq / max(actual_freq, 1e-12), 0.0)

    def _expected_comm_time(self, src_idx, dst_idx):
        edge_type = self.edge_types[src_idx][dst_idx]
        base_time = _mean_comm_time(edge_type)
        sender_bits = self.clients[src_idx].quant_bits
        comm_ratio = get_quant_comm_ratio(sender_bits, bool(self.args.quant_aware), self.quant_base_bits)
        return max(base_time * comm_ratio, 0.0)

    def _accuracy_reward(self, model_distribution, candidate_idx):
        current_gap, _, _, normalized = self._coverage_features(model_distribution)
        next_distribution = np.asarray(model_distribution, dtype=float) + np.asarray(self.clients[candidate_idx].label_distribution, dtype=float)
        next_gap, _, _, next_normalized = self._coverage_features(next_distribution)
        novelty = float(np.mean((normalized <= 0) & (np.asarray(self.clients[candidate_idx].label_distribution) > 0)))
        gain = max(current_gap - next_gap, 0.0)
        return gain + novelty

    def _energy_reward(self, current_client, candidate_idx, battery_state, train_energy, comm_energy_total):
        current_capacity = max(battery_state[current_client]['capacity_j'], 1e-12)
        candidate_capacity = max(battery_state[candidate_idx]['capacity_j'], 1e-12)
        current_battery_ratio = battery_state[current_client]['remaining_j'] / current_capacity
        candidate_battery_ratio = battery_state[candidate_idx]['remaining_j'] / candidate_capacity
        consumed_ratio = (train_energy + comm_energy_total) / current_capacity
        return max(candidate_battery_ratio, 0.0) + max(current_battery_ratio - consumed_ratio, 0.0)

    def _latency_reward(self, current_client, candidate_idx):
        train_time = self._expected_training_time(current_client)
        comm_time = self._expected_comm_time(current_client, candidate_idx)
        quant_bits = self.clients[current_client].quant_bits
        quant_penalty = 0.0
        if bool(self.args.quant_aware):
            quant_penalty = abs(self.quant_base_bits - quant_bits) / float(self.quant_base_bits)
        latency_cost = (train_time + comm_time) * (1.0 + quant_penalty)
        return 1.0 / (1.0 + latency_cost)

    def predict_reward(self, agent_id, current_client, candidate_idx, model_distribution, battery_state):
        accuracy_score = self._accuracy_reward(model_distribution, candidate_idx)
        train_time = self._expected_training_time(current_client)
        comm_time = self._expected_comm_time(current_client, candidate_idx)
        train_energy = self.clients[current_client].device_profile['compute_power_w'] * train_time
        comm_src, comm_dst = get_communication_energy_breakdown(current_client, candidate_idx, comm_time)
        energy_score = self._energy_reward(current_client, candidate_idx, battery_state, train_energy, comm_src + comm_dst)
        latency_score = self._latency_reward(current_client, candidate_idx)
        return self.alpha * accuracy_score + self.beta * energy_score + self.gamma * latency_score

    def observed_reward(self, current_client, candidate_idx, model_distribution, battery_state,
                        train_time, comm_time, train_energy, comm_energy_total):
        accuracy_score = self._accuracy_reward(model_distribution, candidate_idx)
        energy_score = self._energy_reward(current_client, candidate_idx, battery_state, train_energy, comm_energy_total)
        quant_bits = self.clients[current_client].quant_bits
        quant_penalty = 0.0
        if bool(self.args.quant_aware):
            quant_penalty = abs(self.quant_base_bits - quant_bits) / float(self.quant_base_bits)
        latency_cost = (train_time + comm_time) * (1.0 + quant_penalty)
        latency_score = 1.0 / (1.0 + latency_cost)
        return self.alpha * accuracy_score + self.beta * energy_score + self.gamma * latency_score

    def select_next(self, agent_id, current_client, model_distribution, battery_state, adjacency_matrix, round_idx):
        state_key = self._state_key(agent_id, current_client, model_distribution, battery_state, round_idx)
        candidates = self._neighbors(current_client, adjacency_matrix)
        if not candidates:
            self.pending[agent_id] = {
                'state_key': state_key,
                'current_client': current_client,
                'candidates': [],
                'choice': current_client,
                'model_distribution': np.asarray(model_distribution, dtype=float).copy(),
            }
            return current_client

        action_values = self.q_table.setdefault(state_key, {})
        scored_candidates = []
        for candidate_idx in candidates:
            if candidate_idx not in action_values:
                action_values[candidate_idx] = self.predict_reward(
                    agent_id, current_client, candidate_idx, model_distribution, battery_state
                )
            scored_candidates.append((candidate_idx, action_values[candidate_idx]))

        if random.random() < self.epsilon:
            choice = random.choice(candidates)
        else:
            choice = max(scored_candidates, key=lambda item: item[1])[0]

        self.pending[agent_id] = {
            'state_key': state_key,
            'current_client': current_client,
            'candidates': candidates,
            'choice': choice,
            'model_distribution': np.asarray(model_distribution, dtype=float).copy(),
        }
        return choice

    def observe(self, agent_id, reward, next_client, next_model_distribution, battery_state, adjacency_matrix, round_idx):
        transition = self.pending.get(agent_id)
        if transition is None:
            return

        state_key = transition['state_key']
        action = transition['choice']
        next_state_key = self._state_key(agent_id, next_client, next_model_distribution, battery_state, round_idx)
        next_candidates = self._neighbors(next_client, adjacency_matrix)
        next_action_values = self.q_table.setdefault(next_state_key, {})
        next_best = 0.0
        for candidate_idx in next_candidates:
            if candidate_idx not in next_action_values:
                next_action_values[candidate_idx] = self.predict_reward(
                    agent_id, next_client, candidate_idx, next_model_distribution, battery_state
                )
            next_best = max(next_best, next_action_values[candidate_idx])

        old_value = self.q_table[state_key][action]
        target = reward + self.discount * next_best
        self.q_table[state_key][action] = (1.0 - self.lr) * old_value + self.lr * target
        self.reward_history.append({
            'agent_id': int(agent_id),
            'state': state_key,
            'action': int(action),
            'reward': float(reward),
            'next_state': next_state_key,
            'updated_q': float(self.q_table[state_key][action]),
        })


class DeviceAggregationRLController(object):
    def __init__(self, args, clients, adjacency_matrix, edge_types):
        self.args = args
        self.clients = clients
        self.adjacency_matrix = adjacency_matrix
        self.edge_types = edge_types
        self.alpha = getattr(args, 'rl_accuracy_weight', 0.65)
        self.beta = getattr(args, 'rl_energy_weight', 0.25)
        self.gamma = getattr(args, 'rl_latency_weight', 0.10)
        self.lr = getattr(args, 'rl_lr', 0.2)
        self.discount = getattr(args, 'rl_discount', 0.9)
        self.epsilon = getattr(args, 'rl_epsilon', 0.1)
        self.state_bins = max(int(getattr(args, 'rl_state_bins', 4)), 2)
        self.quant_base_bits = max(int(getattr(args, 'quant_comm_base_bits', 32)), MIN_QUANT_BITS)
        self.quant_options = self._parse_quant_options(getattr(args, 'rl_quant_bits', '8,16,32'))
        self.min_neighbors = max(int(getattr(args, 'rl_min_agg_neighbors', 1)), 0)
        self.max_neighbors = int(getattr(args, 'rl_max_agg_neighbors', 0))
        self.score_threshold = float(getattr(args, 'rl_neighbor_score_threshold', 0.0))
        self.random_neighbor_prob = float(getattr(args, 'rl_neighbor_sample_prob', 0.5))
        self.q_table = defaultdict(dict)
        self.action_history = []
        self.q_history = []

    def _parse_quant_options(self, raw_bits):
        if isinstance(raw_bits, str):
            values = [item.strip() for item in raw_bits.split(',') if item.strip()]
            options = [int(value) for value in values]
        else:
            options = [int(value) for value in raw_bits]
        options = sorted(set(bit for bit in options if bit >= MIN_QUANT_BITS))
        return options or [MIN_QUANT_BITS]

    def _bucket(self, value):
        value = max(min(float(value), 0.999999), 0.0)
        return int(value * self.state_bins)

    def _neighbors(self, receiver_idx):
        return [nb_idx for nb_idx in range(self.args.num_users) if self.adjacency_matrix[receiver_idx][nb_idx] == 1]

    def _label_distribution(self, client_idx):
        distribution = getattr(self.clients[client_idx], 'label_distribution', None)
        if distribution is None:
            return np.zeros(self.args.num_classes)
        return np.asarray(distribution, dtype=float)

    def _coverage_features(self, distribution):
        distribution = np.asarray(distribution, dtype=float)
        total = distribution.sum()
        normalized = distribution / total if total > 0 else np.zeros_like(distribution)
        coverage_gap = float(uniform_distribution_loss(distribution))
        missing_ratio = float(np.mean(normalized == 0)) if len(normalized) else 0.0
        entropy = 0.0
        if total > 0:
            entropy = float(-np.sum(np.where(normalized > 0, normalized * np.log(normalized + 1e-12), 0.0)))
            entropy = entropy / max(np.log(max(len(normalized), 2)), 1e-12)
        return coverage_gap, missing_ratio, entropy, normalized

    def _state_key(self, receiver_idx, battery_state, round_idx):
        distribution = self._label_distribution(receiver_idx)
        coverage_gap, missing_ratio, entropy, _ = self._coverage_features(distribution)
        degree_ratio = len(self._neighbors(receiver_idx)) / max(self.args.num_users - 1, 1)
        battery_ratio = battery_state[receiver_idx]['remaining_j'] / max(battery_state[receiver_idx]['capacity_j'], 1e-12)
        progress_ratio = round_idx / max(self.args.epochs - 1, 1)
        return (
            int(receiver_idx),
            self._bucket(coverage_gap / max(len(distribution), 1)),
            self._bucket(missing_ratio),
            self._bucket(entropy),
            self._bucket(degree_ratio),
            self._bucket(battery_ratio),
            self._bucket(progress_ratio),
        )

    def _quant_options_for_sender(self, sender_idx):
        sender_max_bits = max(int(getattr(self.clients[sender_idx], 'quant_bits', self.quant_base_bits)), MIN_QUANT_BITS)
        options = [bit for bit in self.quant_options if MIN_QUANT_BITS <= bit <= sender_max_bits]
        return sorted(set(options or [MIN_QUANT_BITS]))

    def _expected_comm_time(self, sender_idx, receiver_idx, bits):
        edge_type = self.edge_types[sender_idx][receiver_idx]
        base_time = _mean_comm_time(edge_type)
        comm_ratio = get_quant_comm_ratio(bits, bool(self.args.quant_aware), self.quant_base_bits)
        return max(base_time * comm_ratio, 0.0)

    def _pair_accuracy_score(self, receiver_idx, sender_idx, bits):
        receiver_dist = self._label_distribution(receiver_idx)
        sender_dist = self._label_distribution(sender_idx)
        current_gap, _, _, normalized = self._coverage_features(receiver_dist)
        next_gap, _, _, _ = self._coverage_features(receiver_dist + sender_dist)
        novelty = float(np.mean((normalized <= 0) & (sender_dist > 0)))
        quant_quality = bits / float(max(getattr(self.clients[sender_idx], 'quant_bits', bits), MIN_QUANT_BITS))
        return (max(current_gap - next_gap, 0.0) + novelty) * quant_quality

    def _pair_energy_score(self, sender_idx, receiver_idx, bits, battery_state):
        comm_time = self._expected_comm_time(sender_idx, receiver_idx, bits)
        sender_energy, receiver_energy = get_communication_energy_breakdown(sender_idx, receiver_idx, comm_time)
        sender_capacity = max(battery_state[sender_idx]['capacity_j'], 1e-12)
        receiver_capacity = max(battery_state[receiver_idx]['capacity_j'], 1e-12)
        sender_remaining = battery_state[sender_idx]['remaining_j'] / sender_capacity
        receiver_remaining = battery_state[receiver_idx]['remaining_j'] / receiver_capacity
        cost_ratio = (sender_energy / sender_capacity + receiver_energy / receiver_capacity) / 2.0
        return max((sender_remaining + receiver_remaining) / 2.0 - cost_ratio, 0.0)

    def _pair_latency_score(self, sender_idx, receiver_idx, bits):
        comm_time = self._expected_comm_time(sender_idx, receiver_idx, bits)
        quant_loss_proxy = 0.0
        sender_bits = max(getattr(self.clients[sender_idx], 'quant_bits', bits), MIN_QUANT_BITS)
        if bool(self.args.quant_aware):
            quant_loss_proxy = max(sender_bits - bits, 0) / float(sender_bits)
        return 1.0 / (1.0 + comm_time * (1.0 + quant_loss_proxy))

    def predict_pair_reward(self, receiver_idx, sender_idx, bits, battery_state):
        accuracy_score = self._pair_accuracy_score(receiver_idx, sender_idx, bits)
        energy_score = self._pair_energy_score(sender_idx, receiver_idx, bits, battery_state)
        latency_score = self._pair_latency_score(sender_idx, receiver_idx, bits)
        return self.alpha * accuracy_score + self.beta * energy_score + self.gamma * latency_score

    def select_action(self, receiver_idx, battery_state, round_idx):
        state_key = self._state_key(receiver_idx, battery_state, round_idx)
        candidate_neighbors = self._neighbors(receiver_idx)
        if not candidate_neighbors:
            action = {
                'receiver': int(receiver_idx),
                'candidate_neighbors': [],
                'selected_neighbors': [],
                'quant_bits': {},
                'selected_pairs': [],
                'state_key': state_key,
            }
            self.action_history.append(dict(action))
            return action

        max_neighbors = self.max_neighbors if self.max_neighbors > 0 else len(candidate_neighbors)
        max_neighbors = max(min(max_neighbors, len(candidate_neighbors)), 1)
        min_neighbors = min(self.min_neighbors, max_neighbors)
        state_values = self.q_table.setdefault(state_key, {})

        if random.random() < self.epsilon:
            if min_neighbors == max_neighbors:
                sample_size = min_neighbors
            else:
                sample_size = random.randint(min_neighbors, max_neighbors)
            if sample_size == 0:
                selected_neighbors = [
                    nb_idx for nb_idx in candidate_neighbors if random.random() < self.random_neighbor_prob
                ]
                if not selected_neighbors:
                    selected_neighbors = [random.choice(candidate_neighbors)]
            else:
                selected_neighbors = random.sample(candidate_neighbors, sample_size)
            selected_pairs = [
                (nb_idx, random.choice(self._quant_options_for_sender(nb_idx))) for nb_idx in selected_neighbors
            ]
        else:
            best_pairs = []
            for nb_idx in candidate_neighbors:
                neighbor_pairs = []
                for bits in self._quant_options_for_sender(nb_idx):
                    pair_key = (int(nb_idx), int(bits))
                    if pair_key not in state_values:
                        state_values[pair_key] = self.predict_pair_reward(receiver_idx, nb_idx, bits, battery_state)
                    neighbor_pairs.append((pair_key, state_values[pair_key]))
                best_pairs.append(max(neighbor_pairs, key=lambda item: item[1]))
            best_pairs.sort(key=lambda item: item[1], reverse=True)
            selected_pairs = [
                pair_key for pair_key, score in best_pairs
                if score >= self.score_threshold
            ][:max_neighbors]
            if len(selected_pairs) < min_neighbors:
                selected_pairs = [pair_key for pair_key, _ in best_pairs[:min_neighbors]]
            if not selected_pairs:
                selected_pairs = [best_pairs[0][0]]

        selected_pairs = [(int(nb_idx), int(bits)) for nb_idx, bits in selected_pairs]
        quant_bits = {int(nb_idx): int(bits) for nb_idx, bits in selected_pairs}
        action = {
            'receiver': int(receiver_idx),
            'candidate_neighbors': [int(nb_idx) for nb_idx in candidate_neighbors],
            'selected_neighbors': [int(nb_idx) for nb_idx, _ in selected_pairs],
            'quant_bits': quant_bits,
            'selected_pairs': selected_pairs,
            'state_key': state_key,
        }
        self.action_history.append({
            'receiver': action['receiver'],
            'candidate_neighbors': action['candidate_neighbors'],
            'selected_neighbors': action['selected_neighbors'],
            'quant_bits': dict(action['quant_bits']),
            'selected_pairs': list(action['selected_pairs']),
            'state_key': action['state_key'],
        })
        return action

    def observed_reward(self, receiver_idx, action, battery_state, comm_time_total, comm_energy_total):
        if not action['selected_neighbors']:
            return 0.0

        receiver_dist = self._label_distribution(receiver_idx)
        selected_dist = np.zeros_like(receiver_dist)
        quant_quality_values = []
        for sender_idx, bits in action['selected_pairs']:
            selected_dist = selected_dist + self._label_distribution(sender_idx)
            sender_bits = max(getattr(self.clients[sender_idx], 'quant_bits', bits), MIN_QUANT_BITS)
            quant_quality_values.append(bits / float(sender_bits))

        current_gap, _, _, normalized = self._coverage_features(receiver_dist)
        next_gap, _, _, _ = self._coverage_features(receiver_dist + selected_dist)
        novelty = float(np.mean((normalized <= 0) & (selected_dist > 0)))
        quant_quality = float(np.mean(quant_quality_values)) if quant_quality_values else 1.0
        accuracy_score = (max(current_gap - next_gap, 0.0) + novelty) * quant_quality

        involved_nodes = [receiver_idx] + list(action['selected_neighbors'])
        remaining_score = np.mean([
            battery_state[node_idx]['remaining_j'] / max(battery_state[node_idx]['capacity_j'], 1e-12)
            for node_idx in involved_nodes
        ])
        mean_capacity = np.mean([
            max(battery_state[node_idx]['capacity_j'], 1e-12)
            for node_idx in involved_nodes
        ])
        energy_score = max(remaining_score - comm_energy_total / max(mean_capacity, 1e-12), 0.0)
        latency_score = 1.0 / (1.0 + comm_time_total)
        return self.alpha * accuracy_score + self.beta * energy_score + self.gamma * latency_score

    def observe(self, receiver_idx, action, reward, battery_state, round_idx):
        state_key = action['state_key']
        next_state_key = self._state_key(receiver_idx, battery_state, round_idx)
        next_values = self.q_table.setdefault(next_state_key, {})
        next_best = max(next_values.values()) if next_values else 0.0
        state_values = self.q_table.setdefault(state_key, {})

        for pair_key in action['selected_pairs']:
            pair_key = (int(pair_key[0]), int(pair_key[1]))
            old_value = state_values.get(pair_key, 0.0)
            target = reward + self.discount * next_best
            state_values[pair_key] = (1.0 - self.lr) * old_value + self.lr * target
            self.q_history.append({
                'receiver': int(receiver_idx),
                'state': state_key,
                'pair_action': pair_key,
                'reward': float(reward),
                'next_state': next_state_key,
                'updated_q': float(state_values[pair_key]),
            })
