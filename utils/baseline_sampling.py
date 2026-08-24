import random


def baseline_active_sampling_enabled(args):
    return bool(int(getattr(args, "baseline_active_sampling", 0)))


def baseline_active_client_count(args):
    num_users = int(getattr(args, "num_users", 0))
    if num_users <= 0:
        return 0
    frac = float(getattr(args, "frac", 1.0))
    return max(1, min(num_users, int(num_users * frac)))


def select_baseline_active_clients(args):
    num_users = int(getattr(args, "num_users", 0))
    if num_users <= 0:
        return []
    if not baseline_active_sampling_enabled(args):
        return list(range(num_users))
    count = baseline_active_client_count(args)
    return sorted(random.sample(list(range(num_users)), count))
