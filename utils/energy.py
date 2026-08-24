def initialize_battery_state(clients):
    battery_state = []
    for client in clients:
        capacity = float(client.device_profile.get('battery_capacity_j', 0.0))
        battery_state.append({
            'client_id': int(client.id),
            'type': client.device_profile.get('type', ''),
            'quant_bits': int(getattr(client, 'quant_bits', client.device_profile.get('quant_bits', 0))),
            'capacity_j': capacity,
            'remaining_j': capacity,
            'used_j': 0.0,
            'sleep_used_j': 0.0,
            'idle_used_j': 0.0,
            'depleted': False,
        })
    return battery_state


def consume_energy(battery_state, client_idx, amount_j):
    if battery_state[client_idx]['depleted']:
        return 0.0
    amount_j = max(float(amount_j), 0.0)
    before = float(battery_state[client_idx]['remaining_j'])
    consumed = min(amount_j, before)
    battery_state[client_idx]['used_j'] += consumed
    battery_state[client_idx]['remaining_j'] = max(
        0.0,
        battery_state[client_idx]['remaining_j'] - amount_j,
    )
    battery_state[client_idx]['depleted'] = battery_state[client_idx]['remaining_j'] <= 0.0
    return consumed


def consume_communication_energy(battery_state, src_idx, dst_idx, src_energy_j, dst_energy_j):
    # 任一端已耗尽则整条链路不计通信耗电（与「没电不参与通信」一致）
    if battery_state[src_idx]['depleted'] or battery_state[dst_idx]['depleted']:
        return 0.0
    return (
        consume_energy(battery_state, src_idx, src_energy_j)
        + consume_energy(battery_state, dst_idx, dst_energy_j)
    )


def consume_sleep_energy(battery_state, round_duration, busy_time_by_client=None, scale=1.0):
    """Drain sleep energy for non-depleted devices during the unused part of a round."""
    from config import get_sleep_energy

    busy_time_by_client = busy_time_by_client or {}
    round_duration = max(float(round_duration), 0.0)
    scale = max(float(scale), 0.0)
    total = 0.0
    for client_idx, node_state in enumerate(battery_state):
        if node_state['depleted']:
            continue
        busy_time = max(float(busy_time_by_client.get(client_idx, 0.0)), 0.0)
        sleep_time = max(round_duration - busy_time, 0.0)
        sleep_energy = get_sleep_energy(client_idx, sleep_time) * scale
        consumed = consume_energy(battery_state, client_idx, sleep_energy)
        node_state['sleep_used_j'] = node_state.get('sleep_used_j', node_state.get('idle_used_j', 0.0)) + consumed
        node_state['idle_used_j'] = node_state['sleep_used_j']
        total += consumed
    return total


def consume_idle_energy(battery_state, round_duration, busy_time_by_client=None, scale=1.0):
    return consume_sleep_energy(battery_state, round_duration, busy_time_by_client=busy_time_by_client, scale=scale)


def snapshot_battery_state(battery_state):
    return [dict(item) for item in battery_state]
