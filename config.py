import os
import pickle
import random

import numpy as np
from utils.options import args_parser


args = args_parser()
"+++++++++++++++++++++client config+++++++++++++++++++++++++"
# weak/middle/strong keeps the original heterogeneous device abstraction.
# The device profile adds explicit frequency and power parameters per client.
DEFAULT_CLIENT_TYPE_LIST = [
    'strong', 'strong', 'weak', 'strong', 'middle', 'weak', 'middle', 'strong', 'strong', 'weak',
    'weak', 'middle', 'weak', 'weak', 'weak', 'strong', 'middle', 'strong', 'strong', 'middle',
]

DEVICE_TYPE_PROFILES = {
    'weak': {
        'frequency_ghz': 1.20,
        'compute_power_w': 8.0,
        'communication_power_w': 1.8,
        'sleep_power_w': 0.08,
        'quant_bits': 8,
        'battery_capacity_j': 50000.0,
    },
    'middle': {
        'frequency_ghz': 1.80,
        'compute_power_w': 15.0,
        'communication_power_w': 2.8,
        'sleep_power_w': 0.12,
        'quant_bits': 16,
        'battery_capacity_j': 100000.0,
    },
    'strong': {
        'frequency_ghz': 2.40,
        'compute_power_w': 28.0,
        'communication_power_w': 4.5,
        'sleep_power_w': 0.20,
        'quant_bits': 32,
        'battery_capacity_j': 150000.0,
    },
}

TRAINING_TIME_MEAN = {
    'weak': 50,
    'middle': 20,
    'strong': 10,
}

TRAINING_TIME_STD = {
    'weak': 2,
    'middle': 2,
    'strong': 2,
}


def _build_client_type_list(num_users):
    client_types = list(DEFAULT_CLIENT_TYPE_LIST[:num_users])
    if len(client_types) < num_users:
        rng = random.Random(args.seed)
        all_types = list(DEVICE_TYPE_PROFILES.keys())
        client_types.extend(rng.choice(all_types) for _ in range(num_users - len(client_types)))
    return client_types


client_type_list = _build_client_type_list(args.num_users)


def _clip_positive(value, minimum=0.01):
    return max(float(value), minimum)


def _build_client_device_profiles():
    rng = np.random.RandomState(args.seed + 2024)
    profiles = []
    for idx, client_type in enumerate(client_type_list):
        base_profile = DEVICE_TYPE_PROFILES[client_type]
        freq_scale = rng.normal(1.0, args.device_freq_jitter)
        power_scale = rng.normal(1.0, args.device_power_jitter)
        comm_scale = rng.normal(1.0, args.device_power_jitter)
        battery_scale = rng.normal(1.0, args.device_battery_jitter)
        sleep_power = round(_clip_positive(base_profile['sleep_power_w'] * power_scale), 4)
        profiles.append({
            'client_id': idx,
            'type': client_type,
            'frequency_ghz': round(_clip_positive(base_profile['frequency_ghz'] * freq_scale), 4),
            'compute_power_w': round(_clip_positive(base_profile['compute_power_w'] * power_scale), 4),
            'communication_power_w': round(_clip_positive(base_profile['communication_power_w'] * comm_scale), 4),
            'sleep_power_w': sleep_power,
            'idle_power_w': sleep_power,
            'quant_bits': int(base_profile['quant_bits']),
            'battery_capacity_j': round(_clip_positive(base_profile['battery_capacity_j'] * battery_scale), 4),
        })
    return profiles


client_device_profile_list = _build_client_device_profiles()


def get_client_device_profile(client_idx):
    return client_device_profile_list[client_idx]


def get_client_quant_bits(client_idx):
    return max(int(client_device_profile_list[client_idx]['quant_bits']), 8)


def get_training_time(client_type, client_idx=None):
    client_training_time = np.random.normal(TRAINING_TIME_MEAN[client_type], TRAINING_TIME_STD[client_type])
    if client_idx is not None:
        default_freq = DEVICE_TYPE_PROFILES[client_type]['frequency_ghz']
        actual_freq = client_device_profile_list[client_idx]['frequency_ghz']
        client_training_time = client_training_time * default_freq / actual_freq
    return max(client_training_time, 0)


def get_client_training_time(client_idx, multiplier=1.0):
    client_type = client_type_list[client_idx]
    return get_training_time(client_type, client_idx) * multiplier


def get_training_energy(client_idx, training_time):
    return client_device_profile_list[client_idx]['compute_power_w'] * training_time


def get_sleep_energy(client_idx, sleep_time):
    return client_device_profile_list[client_idx]['sleep_power_w'] * max(float(sleep_time), 0.0)


def get_idle_energy(client_idx, idle_time):
    return get_sleep_energy(client_idx, idle_time)
"+++++++++++++++++++++++topo config+++++++++++++++++++++++++"
TOPO_ROOT = os.path.abspath(os.environ.get('RTDFL_TOPO_ROOT', './topo'))
os.makedirs(TOPO_ROOT, exist_ok=True)


def _add_undirected_edge(matrix, src_idx, dst_idx):
    if src_idx == dst_idx:
        return
    matrix[src_idx][dst_idx] = 1
    matrix[dst_idx][src_idx] = 1


def _build_ring_topology(num_users):
    matrix = np.zeros((num_users, num_users))
    for idx in range(num_users):
        left = (idx - 1 + num_users) % num_users
        right = (idx + 1) % num_users
        _add_undirected_edge(matrix, idx, left)
        _add_undirected_edge(matrix, idx, right)
    return matrix


def _build_m_ring_topology(num_users):
    rng = random.Random(args.seed + num_users)
    matrix = _build_ring_topology(num_users)
    for idx in range(num_users):
        left = (idx - 1 + num_users) % num_users
        right = (idx + 1) % num_users
        nb_list = list(range(num_users))
        nb_list.remove(idx)
        if left in nb_list:
            nb_list.remove(left)
        if right in nb_list:
            nb_list.remove(right)
        new_edge_cnt = min(rng.randint(0, 2), len(nb_list))
        for nb_idx in rng.sample(nb_list, new_edge_cnt):
            _add_undirected_edge(matrix, idx, nb_idx)
    return matrix


def _build_full_topology(num_users):
    matrix = np.ones((num_users, num_users))
    np.fill_diagonal(matrix, 0)
    return matrix


def _build_star_topology(num_users):
    matrix = np.zeros((num_users, num_users))
    center_idx = 0
    for idx in range(1, num_users):
        _add_undirected_edge(matrix, center_idx, idx)
    return matrix


def _build_random_topology(num_users):
    rng = random.Random(args.seed + num_users * 17)
    matrix = _build_ring_topology(num_users)
    edge_prob = min(0.35, max(0.08, 4.0 / max(num_users - 1, 1)))
    for idx in range(num_users):
        for jdx in range(idx + 1, num_users):
            if matrix[idx][jdx] == 0 and rng.random() < edge_prob:
                _add_undirected_edge(matrix, idx, jdx)
    return matrix


def _build_clustered_topology(num_users):
    rng = random.Random(args.seed + num_users * 31)
    matrix = np.zeros((num_users, num_users))
    cluster_count = min(4, num_users)
    clusters = [list(cluster) for cluster in np.array_split(np.arange(num_users), cluster_count)]

    for cluster in clusters:
        if len(cluster) == 1:
            continue
        for pos, idx in enumerate(cluster):
            _add_undirected_edge(matrix, idx, cluster[(pos + 1) % len(cluster)])
        for pos, idx in enumerate(cluster):
            for jdx in cluster[pos + 1:]:
                if matrix[idx][jdx] == 0 and rng.random() < 0.6:
                    _add_undirected_edge(matrix, idx, jdx)

    for cluster_idx in range(cluster_count):
        src_idx = clusters[cluster_idx][0]
        dst_idx = clusters[(cluster_idx + 1) % cluster_count][0]
        _add_undirected_edge(matrix, src_idx, dst_idx)

    for cluster_idx, cluster in enumerate(clusters):
        for next_cluster in clusters[cluster_idx + 1:]:
            for idx in cluster:
                for jdx in next_cluster:
                    if matrix[idx][jdx] == 0 and rng.random() < 0.03:
                        _add_undirected_edge(matrix, idx, jdx)
    return matrix


TOPOLOGY_BUILDERS = {
    'ring': _build_ring_topology,
    'M-ring': _build_m_ring_topology,
    'full': _build_full_topology,
    'star': _build_star_topology,
    'random': _build_random_topology,
    'cluster': _build_clustered_topology,
    'clustered': _build_clustered_topology,
}

topo_file = os.path.join(TOPO_ROOT, '{}-{}.pkl'.format(args.topo, args.num_users))
if os.path.exists(topo_file):
    print("loaded exist topo")
    with open(topo_file, "rb") as file:
        Adjacency_matrix = pickle.load(file)
else:
    if args.topo not in TOPOLOGY_BUILDERS:
        raise ValueError("{} topo has not been implemented".format(args.topo))
    print("generate a new topo")
    Adjacency_matrix = TOPOLOGY_BUILDERS[args.topo](args.num_users)
    with open(topo_file, "wb") as file:
        pickle.dump(Adjacency_matrix, file)

"+++++++++++++++++++++++bandwith config+++++++++++++++++++++++++"
network_file = os.path.join(TOPO_ROOT, '{}-{}-network.pkl'.format(args.topo, args.num_users))
if os.path.exists(network_file):
    print("loaded exist network")
    with open(network_file, "rb") as file:
        NetWork_type = pickle.load(file)
else:
    print("generate a new network_type")
    network_rng = np.random.RandomState(args.seed + args.num_users * 97)
    NetWork_type = [["" for _ in range(args.num_users)] for _ in range(args.num_users)]
    for idx in range(args.num_users):
        for jdx in range(idx+1, args.num_users):
            if Adjacency_matrix[idx][jdx] == 1:
                client_type = network_rng.choice(['weak', 'middle', 'strong'], p=[0.2, 0.3, 0.5])
                NetWork_type[idx][jdx] = client_type
                NetWork_type[jdx][idx] = client_type
    with open(network_file, "wb") as file:
        pickle.dump(NetWork_type, file)

def get_communication_time(net_type):
    if net_type == 'weak':
        communication_time = np.random.normal(10, 0.5)
    elif net_type == 'middle':
        communication_time = np.random.normal(5, 0.5)
    elif net_type == 'strong':
        communication_time = np.random.normal(2, 0.5)
    else:
        communication_time = 0
    return max(communication_time, 0)


def get_client_communication_time(src_idx, dst_idx, multiplier=1.0):
    return get_communication_time(NetWork_type[src_idx][dst_idx]) * multiplier


def get_communication_energy(src_idx, dst_idx, communication_time):
    src_energy, dst_energy = get_communication_energy_breakdown(src_idx, dst_idx, communication_time)
    return src_energy + dst_energy


def get_communication_energy_breakdown(src_idx, dst_idx, communication_time):
    src_power = client_device_profile_list[src_idx]['communication_power_w']
    dst_power = client_device_profile_list[dst_idx]['communication_power_w']
    return src_power * communication_time, dst_power * communication_time
