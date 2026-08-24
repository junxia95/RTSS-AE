M_RING_10_ADJACENCY = {
    0: [1, 9],
    1: [0, 2, 3, 7, 9],
    2: [1, 3, 5, 7],
    3: [1, 2, 4, 8],
    4: [3, 5, 7, 9],
    5: [2, 4, 6, 7, 8],
    6: [5, 7],
    7: [1, 2, 4, 5, 6, 8],
    8: [3, 5, 7, 9],
    9: [0, 1, 4, 8],
}


def get_adjacency(num_users=10):
    if int(num_users) != 10:
        raise ValueError("The real-device MMDFL script currently supports the fixed 10-client M-ring only.")
    return {cid: list(neighbors) for cid, neighbors in M_RING_10_ADJACENCY.items()}


def get_neighbors(cid, active_clients=None, num_users=10):
    neighbors = get_adjacency(num_users)[int(cid)]
    if active_clients is None:
        return neighbors
    active_set = set(int(x) for x in active_clients)
    return [idx for idx in neighbors if idx in active_set]


def is_neighbor(src, dst, num_users=10):
    return int(dst) in get_adjacency(num_users)[int(src)]
