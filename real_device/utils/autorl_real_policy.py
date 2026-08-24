import math
import random
from collections import defaultdict, deque

import numpy as np

from utils.mmdfl_real_policy import normalize_scores, uniform_distribution_loss
from utils.mmdfl_topology import get_neighbors
from utils.power_manager_real import get_device_capacity
from utils.main_real_profiles import get_device_type


def _safe_normalize(values):
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return arr / total


def _coverage_gap(values):
    return uniform_distribution_loss(values)


def _novelty(memory, candidate):
    mem = _safe_normalize(memory)
    cand = np.asarray(candidate, dtype=float)
    if cand.sum() <= 0.0:
        return 0.0
    return float(np.mean((mem <= 0.0) & (cand > 0.0)))


class RealAutoRLPolicy:
    """Server-side tabular AutoRL scheduler for real-device MMDFL.

    This mirrors the deployable parts of MMDFL's AutoRL controller: tabular
    Q-learning, DFL-MM comprehensive prior, UCB exploration, late-stage
    consolidation, stay actions, and optional inbound model absorption.
    """

    def __init__(self, args, label_distributions, token_count):
        self.args = args
        self.label_distributions = label_distributions
        self.token_count = int(token_count)
        self.rng = random.Random(int(args.seed) + 9917)
        self.q_table = defaultdict(dict)
        self.visit_sa = defaultdict(int)
        self.replay_buffer = deque(maxlen=int(getattr(args, "autorl_replay_capacity", 384)))
        self.epsilon = float(getattr(args, "autorl_epsilon", 0.12))
        self.min_epsilon = float(getattr(args, "autorl_min_epsilon", 0.02))
        self.epsilon_decay = float(getattr(args, "autorl_epsilon_decay", 0.994))
        self.discount = float(getattr(args, "rl_discount", 0.9))
        self.lr = float(getattr(args, "rl_lr", 0.2))
        self.ucb_c = float(getattr(args, "autorl_tabular_ucb_c", 0.35))
        self.comprehensive_mix = float(getattr(args, "autorl_comprehensive_mix", 0.70))
        self.late_comprehensive_mix = float(getattr(args, "autorl_late_comprehensive_mix", 0.20))
        self.stability_start_frac = float(getattr(args, "autorl_stability_start_frac", 0.55))
        self.late_min_epsilon = float(getattr(args, "autorl_late_min_epsilon", 0.005))
        self.late_ucb_c_scale = float(getattr(args, "autorl_late_ucb_c_scale", 0.20))
        self.late_stay_bonus = float(getattr(args, "autorl_late_stay_bonus", 0.25))
        self.coverage_gain_weight = float(getattr(args, "autorl_coverage_gain_weight", 5.0))
        self.node_coverage_weight = float(getattr(args, "autorl_node_coverage_weight", 0.35))
        self.retention_gain_weight = float(getattr(args, "autorl_retention_gain_weight", 0.35))
        self.forgetting_penalty_weight = float(getattr(args, "autorl_forgetting_penalty_weight", 0.45))
        self.energy_penalty_weight = float(getattr(args, "autorl_energy_penalty_weight", 0.008))
        self.latency_penalty_weight = float(getattr(args, "autorl_latency_penalty_weight", 0.001))
        self.prior_weight = float(getattr(args, "autorl_prior_weight", 1.0))
        self.inbound_max = int(getattr(args, "autorl_max_inbound_models", 1))
        self.inbound_threshold = float(getattr(args, "autorl_inbound_score_threshold", 0.08))
        self.inbound_late_boost = float(getattr(args, "autorl_late_inbound_threshold_boost", 0.20))
        self.inbound_late_gap = float(getattr(args, "autorl_late_inbound_gap_threshold", 0.03))
        self.rollback_margin = float(getattr(args, "autorl_fallback_margin", 0.08))
        self.state_bins = max(int(getattr(args, "rl_state_bins", 4)), 2)
        self.forgetting_horizon = max(int(getattr(args, "autorl_forgetting_horizon", 50)), 1)

        self.node_visit_count = np.zeros((self.token_count, int(args.num_users)), dtype=np.int32)
        self.global_visit_count = np.zeros(int(args.num_users), dtype=np.int32)
        self.node_last_visit_round = np.full((self.token_count, int(args.num_users)), -1, dtype=np.int32)
        self.class_last_seen_round = np.full((self.token_count, int(args.num_classes)), -1, dtype=np.int32)
        self.action_history = []
        self.reward_history = []

    def _round_frac(self, round_idx):
        return float(round_idx) / max(float(int(self.args.epochs) - 1), 1.0)

    def _stability_progress(self, round_idx):
        frac = self._round_frac(round_idx)
        start = min(max(self.stability_start_frac, 0.0), 1.0)
        if frac <= start:
            return 0.0
        return min(max((frac - start) / max(1.0 - start, 1e-12), 0.0), 1.0)

    def _epsilon_floor(self, round_idx):
        p = self._stability_progress(round_idx)
        return (1.0 - p) * self.min_epsilon + p * self.late_min_epsilon

    def _effective_ucb_c(self, round_idx):
        p = self._stability_progress(round_idx)
        return self.ucb_c * ((1.0 - p) + p * self.late_ucb_c_scale)

    def _bucket(self, value):
        value = min(max(float(value), 0.0), 0.999999)
        return int(value * self.state_bins)

    def _battery_ratio(self, cid, battery_state_joules):
        device_type = get_device_type(cid)
        return min(
            max(float(battery_state_joules.get(int(cid), get_device_capacity(device_type))), 0.0),
            get_device_capacity(device_type),
        ) / max(get_device_capacity(device_type), 1e-12)

    def _class_staleness_vector(self, token_id, round_idx):
        last_seen = self.class_last_seen_round[token_id].astype(float)
        seen = last_seen >= 0
        ages = np.where(seen, np.maximum(float(round_idx) - last_seen, 0.0), float(round_idx + 1))
        return np.clip(ages / max(float(self.forgetting_horizon), 1.0), 0.0, 1.0)

    def _retention_gain(self, token_id, memory, candidate, round_idx):
        normalized = _safe_normalize(memory)
        candidate = np.asarray(candidate, dtype=float)
        if normalized.sum() <= 0.0 or candidate.sum() <= 0.0:
            return 0.0
        present = (candidate > 0.0).astype(float)
        staleness = self._class_staleness_vector(token_id, round_idx)
        return float(np.sum(normalized * present * (0.5 + 0.5 * staleness)))

    def _forgetting_risk(self, token_id, memory, candidate, round_idx):
        normalized = _safe_normalize(memory)
        if normalized.sum() <= 0.0:
            return 0.0
        absent = (np.asarray(candidate, dtype=float) <= 0.0).astype(float)
        staleness = self._class_staleness_vector(token_id, round_idx)
        return float(np.sum(normalized * absent * (0.5 + 0.5 * staleness)))

    def _node_coverage_ratio(self, token_id):
        return float(np.mean(self.node_visit_count[token_id] > 0))

    def _node_coverage_reward(self, token_id, cid, round_idx):
        model_visit = float(self.node_visit_count[token_id, cid])
        global_visit = float(self.global_visit_count[cid])
        if self.node_last_visit_round[token_id, cid] < 0:
            stale = 1.0
        else:
            gap = max(int(round_idx) - int(self.node_last_visit_round[token_id, cid]), 0)
            stale = min(1.0, gap / max(float(self.forgetting_horizon), 1.0))
        return 0.65 / math.sqrt(1.0 + model_visit) + 0.20 / math.sqrt(1.0 + global_visit) + 0.15 * stale

    def _state_key(self, token_id, current_cid, model_distribution, battery_state_joules, round_idx):
        memory = np.asarray(model_distribution[token_id], dtype=float)
        normalized = _safe_normalize(memory)
        missing_ratio = float(np.mean(normalized <= 0.0)) if normalized.size else 1.0
        forgetting_pressure = float(np.sum(normalized * self._class_staleness_vector(token_id, round_idx)))
        return (
            int(token_id),
            int(current_cid),
            self._bucket(_coverage_gap(memory)),
            self._bucket(missing_ratio),
            self._bucket(self._node_coverage_ratio(token_id)),
            self._bucket(forgetting_pressure),
            self._bucket(self._battery_ratio(current_cid, battery_state_joules)),
        )

    def record_visit(self, token_id, cid, round_idx):
        token_id = int(token_id)
        cid = int(cid)
        self.node_visit_count[token_id, cid] += 1
        self.global_visit_count[cid] += 1
        self.node_last_visit_round[token_id, cid] = int(round_idx)
        active_classes = np.where(np.asarray(self.label_distributions[cid], dtype=float) > 0.0)[0]
        for class_idx in active_classes:
            self.class_last_seen_round[token_id, class_idx] = int(round_idx)

    def _candidate_prior_scores(
        self,
        token_id,
        current_cid,
        round_idx,
        model_distribution,
        train_time_estimates,
        active_clients,
    ):
        neighbors = get_neighbors(current_cid, active_clients=active_clients, num_users=self.args.num_users)
        candidates = list(neighbors)
        if int(current_cid) in active_clients and int(current_cid) not in candidates:
            candidates.append(int(current_cid))
        if not candidates:
            return {}

        data_raw = []
        speed_raw = []
        forget_raw = []
        memory = np.asarray(model_distribution[token_id], dtype=float)
        for cid in candidates:
            candidate_memory = memory + self.label_distributions[cid]
            data_raw.append(_coverage_gap(candidate_memory))
            speed_raw.append(float(train_time_estimates.get(cid, 1.0)))
            if cid == int(current_cid):
                forget_raw.append(1.0)
            else:
                last = self.node_last_visit_round[token_id, cid]
                forget_raw.append(float(int(round_idx) - int(last) + 2) ** (-0.5))

        data_score = normalize_scores(data_raw)
        speed_score = normalize_scores(speed_raw)
        forget_score = normalize_scores(forget_raw)
        coverage_gap = _coverage_gap(memory)
        round_frac = self._round_frac(round_idx)
        weight_data = min(max(0.20 + 0.50 * (1.0 - round_frac) + 0.20 * coverage_gap, 0.20), 0.80)
        weight_speed = 0.10
        weight_forget = max(1.0 - weight_data - weight_speed, 0.10)
        total = max(weight_data + weight_speed + weight_forget, 1e-12)
        weight_data /= total
        weight_speed /= total
        weight_forget /= total

        prior = {}
        for idx, cid in enumerate(candidates):
            cost = weight_data * data_score[idx] + weight_speed * speed_score[idx] + weight_forget * forget_score[idx]
            value = 1.0 - cost / 100.0
            if cid == int(current_cid):
                value += self.late_stay_bonus * self._stability_progress(round_idx)
            prior[cid] = {
                "value": float(value),
                "cost": float(cost),
                "data_score": float(data_score[idx]),
                "speed_score": float(speed_score[idx]),
                "forget_score": float(forget_score[idx]),
                "weights": {
                    "data": float(weight_data),
                    "speed": float(weight_speed),
                    "forget": float(weight_forget),
                },
            }
        return prior

    def _predict_reward(
        self,
        token_id,
        src_cid,
        dst_cid,
        round_idx,
        model_distribution,
        train_time_estimates,
        battery_state_joules,
        train_record=None,
        inbound_feature=None,
    ):
        memory = np.asarray(model_distribution[token_id], dtype=float)
        candidate = np.asarray(self.label_distributions[dst_cid], dtype=float)
        if inbound_feature is not None:
            candidate = candidate + np.asarray(inbound_feature, dtype=float)
        current_gap = _coverage_gap(memory)
        next_gap = _coverage_gap(memory + candidate)
        coverage_gain = max(current_gap - next_gap, 0.0)
        novelty = _novelty(memory, candidate)
        retention_gain = self._retention_gain(token_id, memory, candidate, round_idx)
        forgetting_risk = self._forgetting_risk(token_id, memory, candidate, round_idx)
        node_gain = self._node_coverage_reward(token_id, dst_cid, round_idx)
        battery_pressure = 1.0 - self._battery_ratio(dst_cid, battery_state_joules)
        latency = float(train_time_estimates.get(dst_cid, 1.0))
        latency_norm = latency / max(max(train_time_estimates.values()) if train_time_estimates else latency, 1e-12)
        loss_bonus = 0.0
        if train_record is not None:
            loss = max(float(train_record.get("loss", 0.0)), 0.0)
            loss_bonus = 0.05 / (1.0 + loss)
        return float(
            self.coverage_gain_weight * coverage_gain
            + 0.25 * novelty
            + self.node_coverage_weight * node_gain
            + self.retention_gain_weight * retention_gain
            - self.forgetting_penalty_weight * forgetting_risk
            - self.energy_penalty_weight * battery_pressure
            - self.latency_penalty_weight * latency_norm
            + loss_bonus
        )

    def _select_inbound(self, token_id, current_cid, round_idx, token_locations, model_distribution, active_clients):
        if self.inbound_max <= 0:
            return [], []
        candidates = []
        neighbor_set = set(get_neighbors(current_cid, active_clients=active_clients, num_users=self.args.num_users))
        memory = np.asarray(model_distribution[token_id], dtype=float)
        for other_token, other_cid in enumerate(token_locations):
            if other_token == token_id or int(other_cid) not in neighbor_set:
                continue
            inbound_feature = np.asarray(model_distribution[other_token], dtype=float)
            current_gap = _coverage_gap(memory)
            next_gap = _coverage_gap(memory + inbound_feature)
            score = max(current_gap - next_gap, 0.0) + 0.2 * _novelty(memory, inbound_feature)
            score += self.retention_gain_weight * self._retention_gain(token_id, memory, inbound_feature, round_idx)
            score -= self.forgetting_penalty_weight * self._forgetting_risk(token_id, memory, inbound_feature, round_idx)
            candidates.append(
                {
                    "model_id": int(other_token),
                    "client_id": int(other_cid),
                    "score": float(score),
                }
            )
        candidates.sort(key=lambda item: item["score"], reverse=True)
        threshold = self.inbound_threshold + self.inbound_late_boost * self._stability_progress(round_idx)
        if self._stability_progress(round_idx) > 0.0 and _coverage_gap(memory) <= self.inbound_late_gap:
            selected = []
        else:
            selected = [item for item in candidates if item["score"] >= threshold][: self.inbound_max]
        return selected, candidates

    def choose_next(
        self,
        token_id,
        current_cid,
        round_idx,
        model_distribution,
        train_time_estimates,
        active_clients,
        battery_state_joules,
        token_locations,
        last_visit_round,
        train_record=None,
    ):
        token_id = int(token_id)
        current_cid = int(current_cid)
        state_key = self._state_key(token_id, current_cid, model_distribution, battery_state_joules, round_idx)
        prior_scores = self._candidate_prior_scores(
            token_id,
            current_cid,
            round_idx,
            model_distribution,
            train_time_estimates,
            active_clients,
        )
        if not prior_scores:
            return current_cid, {"reason": "no_active_candidate", "state": state_key, "candidates": []}

        selected_inbound, inbound_candidates = self._select_inbound(
            token_id,
            current_cid,
            round_idx,
            token_locations,
            model_distribution,
            active_clients,
        )
        inbound_feature = np.zeros(int(self.args.num_classes), dtype=float)
        for item in selected_inbound:
            inbound_feature += np.asarray(model_distribution[item["model_id"]], dtype=float)

        q_values = self.q_table[state_key]
        candidates = []
        for cid, prior in prior_scores.items():
            if cid not in q_values:
                q_values[cid] = self.prior_weight * prior["value"] + self._predict_reward(
                    token_id,
                    current_cid,
                    cid,
                    round_idx,
                    model_distribution,
                    train_time_estimates,
                    battery_state_joules,
                    train_record=train_record,
                    inbound_feature=inbound_feature if selected_inbound else None,
                )
            ucb = self._effective_ucb_c(round_idx) * math.sqrt(
                math.log(max(int(round_idx) + 2, 2)) / (self.visit_sa[(state_key, cid)] + 1)
            )
            predicted_reward = self._predict_reward(
                token_id,
                current_cid,
                cid,
                round_idx,
                model_distribution,
                train_time_estimates,
                battery_state_joules,
                train_record=train_record,
                inbound_feature=inbound_feature if selected_inbound else None,
            )
            candidates.append(
                {
                    "cid": int(cid),
                    "q": float(q_values[cid]),
                    "ucb": float(ucb),
                    "prior": prior,
                    "predicted_reward": float(predicted_reward),
                    "score": float(q_values[cid] + ucb),
                }
            )

        comprehensive_prob = (
            (1.0 - self._stability_progress(round_idx)) * self.comprehensive_mix
            + self._stability_progress(round_idx) * self.late_comprehensive_mix
        )
        used_comprehensive = False
        if self.rng.random() < comprehensive_prob:
            chosen = max(candidates, key=lambda item: item["prior"]["value"])
            used_comprehensive = True
        elif self.rng.random() < self.epsilon:
            topk = sorted(candidates, key=lambda item: item["predicted_reward"], reverse=True)[: min(4, len(candidates))]
            chosen = self.rng.choice(topk)
        else:
            chosen = max(candidates, key=lambda item: item["score"])

        best_prior = max(candidates, key=lambda item: item["prior"]["value"])
        if chosen["score"] + self.rollback_margin < best_prior["score"]:
            chosen = best_prior
            reason = "fallback_to_dfl_mm_prior"
        elif used_comprehensive:
            reason = "comprehensive_mix"
        else:
            reason = "epsilon" if chosen not in sorted(candidates, key=lambda item: item["score"], reverse=True)[:1] else "q_ucb"

        reward = self._predict_reward(
            token_id,
            current_cid,
            chosen["cid"],
            round_idx,
            model_distribution,
            train_time_estimates,
            battery_state_joules,
            train_record=train_record,
            inbound_feature=inbound_feature if selected_inbound else None,
        )
        next_state_key = self._state_key(token_id, chosen["cid"], model_distribution, battery_state_joules, round_idx + 1)
        next_candidates = self._candidate_prior_scores(
            token_id,
            chosen["cid"],
            round_idx + 1,
            model_distribution,
            train_time_estimates,
            active_clients,
        )
        next_q = self.q_table[next_state_key]
        next_best = 0.0
        for cid, prior in next_candidates.items():
            if cid not in next_q:
                next_q[cid] = self.prior_weight * prior["value"]
            next_best = max(next_best, float(next_q[cid]))
        old_q = float(q_values.get(chosen["cid"], 0.0))
        td_target = reward + self.discount * next_best
        q_values[chosen["cid"]] = (1.0 - self.lr) * old_q + self.lr * td_target
        self.visit_sa[(state_key, chosen["cid"])] += 1
        self.epsilon = max(self._epsilon_floor(round_idx), self.epsilon * self.epsilon_decay)

        meta = {
            "reason": reason,
            "state": state_key,
            "chosen": chosen,
            "candidates": candidates,
            "reward": float(reward),
            "old_q": old_q,
            "updated_q": float(q_values[chosen["cid"]]),
            "epsilon": float(self.epsilon),
            "selected_inbound": selected_inbound,
            "candidate_inbound": inbound_candidates,
            "comprehensive_prob": float(comprehensive_prob),
            "stability_progress": float(self._stability_progress(round_idx)),
        }
        self.action_history.append(meta)
        self.reward_history.append(
            {
                "round": int(round_idx),
                "token_id": int(token_id),
                "src": int(current_cid),
                "dst": int(chosen["cid"]),
                "reward": float(reward),
                "epsilon": float(self.epsilon),
                "reason": reason,
            }
        )
        self.record_visit(token_id, chosen["cid"], round_idx + 1)
        return int(chosen["cid"]), meta

    def state_dict(self):
        return {
            "q_table": [
                (state_key, int(action), float(value))
                for state_key, actions in self.q_table.items()
                for action, value in actions.items()
            ],
            "visit_sa": [(state_key, int(action), int(count)) for (state_key, action), count in self.visit_sa.items()],
            "epsilon": float(self.epsilon),
            "rng_state": self.rng.getstate(),
            "node_visit_count": self.node_visit_count,
            "global_visit_count": self.global_visit_count,
            "node_last_visit_round": self.node_last_visit_round,
            "class_last_seen_round": self.class_last_seen_round,
            "action_history": list(self.action_history),
            "reward_history": list(self.reward_history),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.q_table = defaultdict(dict)
        for state_key, action, value in state.get("q_table", []):
            self.q_table[tuple(state_key)][int(action)] = float(value)
        self.visit_sa = defaultdict(int)
        for state_key, action, count in state.get("visit_sa", []):
            self.visit_sa[(tuple(state_key), int(action))] = int(count)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)
        for name in (
            "node_visit_count",
            "global_visit_count",
            "node_last_visit_round",
            "class_last_seen_round",
        ):
            if name in state:
                setattr(self, name, state[name])
        self.action_history = list(state.get("action_history", []))
        self.reward_history = list(state.get("reward_history", []))
