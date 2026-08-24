#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import datetime
import pickle
import random

import matplotlib
try:
    from thop import profile
except ModuleNotFoundError:
    profile = None
matplotlib.use('Agg')
from utils.utils import save_result
from config import *
import torch
from torch import nn, autograd
import numpy as np
from utils.FL_utils import *
from utils.FL_utils import DataLoader
from utils.energy import consume_communication_energy, consume_energy, consume_sleep_energy, initialize_battery_state, snapshot_battery_state

Global_Client_set = []
Global_Model_set = []
training_client = None
model_distribution = None
last_visit_round = None

def calculate_uniform_loss(a):
    if np.sum(a) == 0:
        return 0
    a = a / np.sum(a)
    uniform_vec = np.array([1 / len(a) for _ in range(len(a))])
    return np.linalg.norm(a - uniform_vec)

class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.args = args
        self.device_profile = get_client_device_profile(id)

    def calculate_label_distribuion(self, dataset):
        self.label_distribution = np.zeros(self.args.num_classes)
        ldr_train = DataLoader(DatasetSplit(dataset, self.data_idx), batch_size=self.args.local_bs, shuffle=True)
        for batch_idx, (images, labels) in enumerate(ldr_train):
            for label in labels:
                self.label_distribution[label] = self.label_distribution[label] + 1;

class LocalUpdate_DFL(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
    def train(self, client, round):
        net = copy.deepcopy(client.local_net)
        net.train()
        net = net.to(self.args.device)
        ldr_train = DataLoader(DatasetSplit(self.dataset, client.data_idx), batch_size=self.args.local_bs, shuffle=True)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
        net = net.to('cpu')
        client.local_net = copy.deepcopy(net)
        return

def _node_active(battery_state, idx):
    return battery_state is None or not battery_state[idx]['depleted']

def _active_neighbors(idx, args, battery_state):
    return [
        nb_idx for nb_idx in range(args.num_users)
        if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx)
    ]

def _deplete_if_insufficient(battery_state, idx, required_energy):
    if battery_state is None:
        return False
    if battery_state[idx]['depleted']:
        return True
    required_energy = max(float(required_energy), 0.0)
    if battery_state[idx]['remaining_j'] >= required_energy:
        return False
    consume_energy(battery_state, idx, battery_state[idx]['remaining_j'])
    return True

def choose_next_neighbor(k, iter, args, battery_state=None):
    global training_client, model_distribution, last_visit_round
    idx = training_client[k]
    if not _node_active(battery_state, idx):
        return idx

    if args.client_selection=='data_aware':
        model_dist = model_distribution[k]
        next_idx = -1
        min_uniform_loss = 1e9
        random_num = random.uniform(0, 1)
        if random_num >= args.curiosity:
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                    next_model_dist = model_dist + Global_Client_set[nb_idx].label_distribution
                    uniform_loss = calculate_uniform_loss(next_model_dist)
                    if uniform_loss < min_uniform_loss:
                        next_idx = nb_idx
                        min_uniform_loss = uniform_loss
            if next_idx < 0:
                return idx
        else:
            nb_list = _active_neighbors(idx, args, battery_state)
            if not nb_list:
                return idx
            next_idx = random.choice(nb_list)
        model_distribution[k] = model_distribution[k] + Global_Client_set[next_idx].label_distribution

    elif args.client_selection=="speed_aware":
        next_idx = -1
        min_epoch_time = 1e9
        random_num = random.uniform(0,1)
        if random_num >= args.curiosity:
            for nb_idx in range(args.num_users):
                if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                    epoch_time = get_communication_time(NetWork_type[idx][nb_idx]) + get_training_time(client_type_list[nb_idx])
                    if epoch_time < min_epoch_time:
                        next_idx = nb_idx
                        min_epoch_time = epoch_time
            if next_idx < 0:
                return idx
        else:
            nb_list = _active_neighbors(idx, args, battery_state)
            if not nb_list:
                return idx
            next_idx = random.choice(nb_list)

    elif args.client_selection=="forget_aware":
        min_visit_epoch = 1e9
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                visit_epoch = last_visit_round[nb_idx]
                min_visit_epoch = min(min_visit_epoch, visit_epoch)
        nb_list = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                if last_visit_round[nb_idx] == min_visit_epoch:
                    nb_list.append(nb_idx)
        if not nb_list:
            return idx
        next_idx = random.choice(nb_list)

    elif args.client_selection == 'comprehensive':
        nb_list = []
        '''==========data_aware score=========='''
        model_dist = model_distribution[k]
        data_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                nb_list.append(nb_idx)
                next_model_dist = model_dist + Global_Client_set[nb_idx].label_distribution
                uniform_loss = calculate_uniform_loss(next_model_dist)
                data_aware_score.append(uniform_loss)
        if not nb_list:
            return idx
        sum_score = sum(data_aware_score)
        if sum_score == 0:
            data_aware_score = [1.0 for _ in data_aware_score]
        else:
            data_aware_score = [x / sum_score * 100 for x in data_aware_score]
        '''==========speed_aware score=========='''
        speed_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                epoch_time = get_communication_time(NetWork_type[idx][nb_idx]) + get_training_time(client_type_list[nb_idx])
                speed_aware_score.append(epoch_time)
        sum_score = sum(speed_aware_score)
        if sum_score == 0:
            speed_aware_score = [1.0 for _ in speed_aware_score]
        else:
            speed_aware_score = [x / sum_score * 100 for x in speed_aware_score]
        '''==========forget_aware score=========='''
        forget_aware_score = []
        for nb_idx in range(args.num_users):
            if Adjacency_matrix[idx][nb_idx] == 1 and _node_active(battery_state, nb_idx):
                visit_epoch = last_visit_round[nb_idx]
                forget_aware_score.append(visit_epoch)
        forget_aware_score = [(iter - x + 2)**(-0.5) for x in forget_aware_score]
        sum_score = sum(forget_aware_score)
        if sum_score == 0:
            forget_aware_score = [1.0 for _ in forget_aware_score]
        else:
            forget_aware_score = [x / sum_score * 100 for x in forget_aware_score]

        '''==========comprehensive score=========='''
        round_frac = float(iter) / max(float(args.epochs - 1), 1.0)
        coverage_gap = calculate_uniform_loss(model_dist)
        weight_data = min(max(0.20 + 0.50 * (1.0 - round_frac) + 0.20 * coverage_gap, 0.20), 0.80)
        weight_speed = 0.10
        weight_forget = max(1.0 - weight_data - weight_speed, 0.10)
        total_w = weight_data + weight_speed + weight_forget
        weight_data /= total_w
        weight_speed /= total_w
        weight_forget /= total_w
        comprehensive_score = []
        for i in range(len(nb_list)):
            score = (data_aware_score[i] * weight_data +
                     speed_aware_score[i] * weight_speed +
                     forget_aware_score[i] * weight_forget)
            comprehensive_score.append(score)

        best_score = min(comprehensive_score)
        ties = [nb_list[i] for i in range(len(nb_list)) if abs(comprehensive_score[i] - best_score) < 1e-12]
        if len(ties) > 1:
            best_visit = min(last_visit_round[nb_idx] for nb_idx in ties)
            ties = [nb_idx for nb_idx in ties if last_visit_round[nb_idx] == best_visit]
        next_idx = random.choice(ties)
        model_distribution[k] = model_distribution[k] + Global_Client_set[next_idx].label_distribution
    elif args.client_selection=='random':
        nb_list = _active_neighbors(idx, args, battery_state)
        if not nb_list:
            return idx
        next_idx = random.choice(nb_list)

    return next_idx
def DFL_MM(args, net_glob, dataset_train, dataset_test, dict_users):
    global training_client, model_distribution, last_visit_round
    net_size = sum([param.nelement() for param in net_glob.parameters()])
    target_time = dict()
    target_comm = dict()
    target_energy = dict()
    target_acc1 = 55
    target_acc2 = 60
    acc = dict()
    model_cnt = int(args.num_users * args.frac)
    for idx in range(model_cnt):
        Global_Model_set.append(copy.deepcopy(net_glob))
        acc["model" + str(idx)] = []
    for idx in range(args.num_users):
        client = Client(idx, dict_users[idx], net_glob, args)
        client.calculate_label_distribuion(dataset_train)
        Global_Client_set.append(client)
    acc["acc"] = []
    time_consume = []
    comm_consume = []
    comm_time_consume = []
    energy_consume = []
    active_node_consume = []
    battery_history = []
    current_time = 0
    current_comm = 0
    current_comm_time = 0
    current_energy = 0
    battery_state = initialize_battery_state(Global_Client_set)
    battery_capacity_scale = getattr(args, 'dfl_battery_capacity_scale', 1.0)
    battery_energy_scale = getattr(args, 'dfl_battery_energy_scale', 1.0)
    battery_sleep_scale = getattr(args, 'dfl_sleep_energy_scale', None)
    if battery_sleep_scale is None:
        battery_sleep_scale = getattr(args, 'dfl_idle_energy_scale', 1.0)
    if battery_sleep_scale is None:
        battery_sleep_scale = 1.0
    battery_sleep_scale = max(float(battery_sleep_scale), 0.0)
    if battery_capacity_scale != 1.0:
        for node_state in battery_state:
            node_state['capacity_j'] = max(node_state['capacity_j'] * battery_capacity_scale, 0.0)
            node_state['remaining_j'] = node_state['capacity_j']
            node_state['used_j'] = 0.0
            node_state['sleep_used_j'] = 0.0
            node_state['idle_used_j'] = 0.0
            node_state['depleted'] = node_state['remaining_j'] <= 0.0
    training_client = random.sample(list(range(args.num_users)), model_cnt)
    model_distribution = [np.zeros(args.num_classes) for _ in range(model_cnt)]
    last_visit_round = [0 for _ in range(args.num_users)]
    for k, idx in enumerate(training_client):
        model_distribution[k] = model_distribution[k] + Global_Client_set[idx].label_distribution
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        if args.client_selection in ['data_aware', 'comprehensive']:
            print("+++++++model_distribution++++++")
            for k, m_d in enumerate(model_distribution):
                print(k,"th model distribution:", m_d)
        if args.client_selection in ['forget_aware', 'comprehensive']:
            print("+++++++visit_round+++++++")
            print(last_visit_round)
        print("choose client:", training_client)
        round_time = 0
        round_comm = 0
        round_comm_time = 0
        round_energy = 0
        busy_time = {idx: 0.0 for idx in range(args.num_users)}
        active_node_count = sum([0 if node_state['depleted'] else 1 for node_state in battery_state])
        print("active nodes:", active_node_count, "/", args.num_users)

        if args.aggregation:
            agg_dict = dict()
            for k, idx in enumerate(training_client):
                if not _node_active(battery_state, idx):
                    continue
                if idx in agg_dict:
                    agg_dict[idx].append(copy.deepcopy(Global_Model_set[k]).state_dict())
                else:
                    agg_dict[idx] = []
                    agg_dict[idx].append(copy.deepcopy(Global_Model_set[k]).state_dict())

            for key in agg_dict.keys():
                if len(agg_dict[key]) > 1:
                    agg_model = Aggregation(agg_dict[key], [1 for _ in range(len(agg_dict[key]))])
                    for k, idx in enumerate(training_client):
                        if idx == key:
                            Global_Model_set[k].load_state_dict(agg_model)
                            model_distribution[k] = Global_Client_set[idx].label_distribution

        for k, (idx, model) in enumerate(zip(training_client, Global_Model_set)):
            if not _node_active(battery_state, idx):
                print("model ", k, "skip depleted client:", idx)
                continue
            Global_Client_set[idx].local_net = copy.deepcopy(model)
            local = LocalUpdate_DFL(args=args, dataset=dataset_train)
            training_time = get_client_training_time(idx)
            training_energy = get_training_energy(idx, training_time) * battery_energy_scale
            if _deplete_if_insufficient(battery_state, idx, training_energy):
                print("client ", idx, "depleted before local training")
                continue
            local.train(client=Global_Client_set[idx], round=iter)
            consume_energy(battery_state, idx, training_energy)
            round_energy = round_energy + training_energy
            round_time = max(round_time, training_time)
            busy_time[idx] += training_time
            Global_Model_set[k] = copy.deepcopy(Global_Client_set[idx].local_net)
            prev_model_distribution = copy.deepcopy(model_distribution[k])
            next_client = choose_next_neighbor(k, iter, args, battery_state=battery_state)
            if next_client == idx:
                model_distribution[k] = prev_model_distribution
                last_visit_round[idx] = iter + 1
                continue
            communication_time = get_client_communication_time(idx, next_client)
            round_comm_time = round_comm_time + communication_time
            src_energy, dst_energy = get_communication_energy_breakdown(idx, next_client, communication_time)
            src_energy = src_energy * battery_energy_scale
            dst_energy = dst_energy * battery_energy_scale
            src_insufficient = _deplete_if_insufficient(battery_state, idx, src_energy)
            dst_insufficient = _deplete_if_insufficient(battery_state, next_client, dst_energy)
            if src_insufficient or dst_insufficient:
                model_distribution[k] = prev_model_distribution
                print("communication skipped because battery is insufficient:",
                      idx, "->", next_client)
                continue
            consume_communication_energy(battery_state, idx, next_client, src_energy, dst_energy)
            round_energy = round_energy + src_energy + dst_energy
            busy_time[idx] += communication_time
            busy_time[next_client] += communication_time
            training_client[k] = next_client
            last_visit_round[next_client] = iter + 1
            round_time = max(round_time, training_time + communication_time)
            round_comm = round_comm + (net_size * 8) / (1024 * 1024)  # MB
        sleep_energy = consume_sleep_energy(battery_state, round_time, busy_time_by_client=busy_time, scale=battery_sleep_scale)
        round_energy = round_energy + sleep_energy
        current_time = current_time + round_time
        current_comm = current_comm + round_comm
        current_comm_time = current_comm_time + round_comm_time
        comm_time_consume.append(current_comm_time)
        current_energy = sum([node_state['used_j'] for node_state in battery_state])
        avg_acc = 0
        for idx in range(model_cnt):
            acc["model" + str(idx)].append(test(Global_Model_set[idx], dataset_test, args))
            avg_acc = avg_acc + acc["model" + str(idx)][-1]
            print("model ", idx, "acc: ", acc["model" + str(idx)][-1])
        acc["acc"].append(avg_acc / model_cnt)
        time_consume.append(current_time)
        comm_consume.append(current_comm)
        energy_consume.append(current_energy)
        active_node_consume.append(sum([0 if node_state['depleted'] else 1 for node_state in battery_state]))
        battery_history.append(snapshot_battery_state(battery_state))
        print("acc acc: ", acc["acc"][-1])
        print("energy consume: ", current_energy)
        print("active nodes after round:", active_node_consume[-1], "/", args.num_users)
        if acc["acc"][-1] >= target_acc1:
            if target_acc1 not in target_time:
                target_time[target_acc1] = time_consume[-1]
            if target_acc1 not in target_comm:
                target_comm[target_acc1] = comm_consume[-1]
            if target_acc1 not in target_energy:
                target_energy[target_acc1] = energy_consume[-1]
        if acc["acc"][-1] >= target_acc2:
            if target_acc2 not in target_time:
                target_time[target_acc2] = time_consume[-1]
            if target_acc2 not in target_comm:
                target_comm[target_acc2] = comm_consume[-1]
            if target_acc2 not in target_energy:
                target_energy[target_acc2] = energy_consume[-1]

    save_result(acc, 'test_acc', args)
    save_result(time_consume, 'time', args)
    save_result(comm_consume, 'comm', args)
    save_result(comm_time_consume, 'comm_time', args)
    save_result(energy_consume, 'energy', args)
    save_result(active_node_consume, 'active_nodes', args)
    save_result(battery_history, 'battery', args)
    save_result({
        'target_acc': [target_acc1, target_acc2],
        'time': target_time,
        'comm': target_comm,
        'energy': target_energy,
    }, 'target_metrics', args)
    print("target_time:", target_time)
    print("target_comm:", target_comm)
    print("target_energy:", target_energy)
    for key in acc.keys():
        print(key)
        avg_acc_and_var(acc[key])
    print(args.weight_data, " ", args.weight_speed, " ", args.weight_forget)
