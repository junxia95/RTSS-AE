import random

import numpy as np

from utils.mmdfl_topology import get_neighbors


def uniform_distribution_loss(label_counts):
    label_counts = np.asarray(label_counts, dtype=float)
    total = float(label_counts.sum())
    if total <= 0:
        return 0.0
    distribution = label_counts / total
    uniform = np.ones_like(distribution) / max(len(distribution), 1)
    return float(np.linalg.norm(distribution - uniform))


def normalize_scores(values):
    values = [float(v) for v in values]
    total = sum(values)
    if total <= 0:
        return [100.0 / max(len(values), 1) for _ in values]
    return [value / total * 100.0 for value in values]


class RealMMDFLPolicy:
    def __init__(self, args, algorithm="DFL_MM"):
        self.args = args
        self.algorithm = str(algorithm)
        self.rng = random.Random(int(args.seed) + 7301)
        self.q_table = {}
        self.epsilon = 0.25
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995

    def _q(self, src, dst):
        return float(self.q_table.get((int(src), int(dst)), 0.0))

    def _update_q(self, src, dst, reward):
        key = (int(src), int(dst))
        old_value = self.q_table.get(key, 0.0)
        self.q_table[key] = old_value + 0.2 * (float(reward) - old_value)

    def _candidate_scores(
        self,
        token_id,
        current_cid,
        round_idx,
        model_distribution,
        label_distributions,
        last_visit_round,
        train_time_estimates,
        active_clients,
    ):
        neighbors = get_neighbors(current_cid, active_clients=active_clients, num_users=self.args.num_users)
        if not neighbors:
            return []

        data_raw = []
        speed_raw = []
        forget_raw = []
        token_distribution = np.asarray(model_distribution[token_id], dtype=float)
        for nb_idx in neighbors:
            data_raw.append(uniform_distribution_loss(token_distribution + label_distributions[nb_idx]))
            speed_raw.append(float(train_time_estimates.get(nb_idx, 1.0)))
            forget_raw.append(float(int(round_idx) - int(last_visit_round[nb_idx]) + 2) ** (-0.5))

        data_score = normalize_scores(data_raw)
        speed_score = normalize_scores(speed_raw)
        forget_score = normalize_scores(forget_raw)

        round_frac = float(round_idx) / max(float(self.args.epochs - 1), 1.0)
        coverage_gap = uniform_distribution_loss(token_distribution)
        weight_data = min(max(0.20 + 0.50 * (1.0 - round_frac) + 0.20 * coverage_gap, 0.20), 0.80)
        weight_speed = 0.10
        weight_forget = max(1.0 - weight_data - weight_speed, 0.10)
        total_weight = max(weight_data + weight_speed + weight_forget, 1e-12)
        weight_data /= total_weight
        weight_speed /= total_weight
        weight_forget /= total_weight

        scored = []
        for idx, nb_idx in enumerate(neighbors):
            score = (
                weight_data * data_score[idx]
                + weight_speed * speed_score[idx]
                + weight_forget * forget_score[idx]
            )
            scored.append(
                {
                    "cid": int(nb_idx),
                    "score": float(score),
                    "data_score": float(data_score[idx]),
                    "speed_score": float(speed_score[idx]),
                    "forget_score": float(forget_score[idx]),
                    "weights": {
                        "data": float(weight_data),
                        "speed": float(weight_speed),
                        "forget": float(weight_forget),
                    },
                }
            )
        return scored

    def choose_next(
        self,
        token_id,
        current_cid,
        round_idx,
        model_distribution,
        label_distributions,
        last_visit_round,
        train_time_estimates,
        active_clients,
    ):
        scored = self._candidate_scores(
            token_id=token_id,
            current_cid=current_cid,
            round_idx=round_idx,
            model_distribution=model_distribution,
            label_distributions=label_distributions,
            last_visit_round=last_visit_round,
            train_time_estimates=train_time_estimates,
            active_clients=active_clients,
        )
        if not scored:
            return int(current_cid), {"reason": "no_active_neighbor", "candidates": []}

        if self.algorithm.lower() in {"autorl_dfl_mm", "autorl", "autorl_dfl"}:
            if self.rng.random() < self.epsilon:
                chosen = self.rng.choice(scored)
                mode = "epsilon"
            else:
                chosen = min(scored, key=lambda item: item["score"] - 5.0 * self._q(current_cid, item["cid"]))
                mode = "q_score"
            reward = -chosen["score"] / 100.0
            self._update_q(current_cid, chosen["cid"], reward)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            return chosen["cid"], {
                "reason": mode,
                "score": chosen,
                "epsilon": float(self.epsilon),
                "q_value": self._q(current_cid, chosen["cid"]),
                "candidates": scored,
            }

        best_score = min(item["score"] for item in scored)
        ties = [item for item in scored if abs(item["score"] - best_score) < 1e-12]
        chosen = self.rng.choice(ties)
        return chosen["cid"], {"reason": "dfl_mm_comprehensive", "score": chosen, "candidates": scored}
