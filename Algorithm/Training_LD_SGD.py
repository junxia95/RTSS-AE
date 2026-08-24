
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
from utils.energy import consume_communication_energy, consume_energy, initialize_battery_state, snapshot_battery_state
from utils.quantization import (
    get_quant_comm_ratio,
    maybe_quantize_batch,
    project_model_to_quantization,
    quantized_state_dict,
    state_dict_payload_nbytes,
    transmit_state_dict,
)
from utils.baseline_sampling import select_baseline_active_clients
from utils.FL_utils import *
from utils.FL_utils import DataLoader

Global_Client_set = []

class Client(object):
    def __init__(self, id, data_idx, net, args):
        self.id = id
        self.data_idx = data_idx
        self.data_cnt = len(self.data_idx)
        self.local_net = copy.deepcopy(net)
        self.trans_net = copy.deepcopy(net)
        self.device_profile = get_client_device_profile(id)
        self.quant_bits = get_client_quant_bits(id)


class LocalUpdate_LD_SGD(object):
    def __init__(self, args, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
    def train(self, client, round):
        net = copy.deepcopy(client.local_net)
        net.train()
        net = net.to(self.args.device)
        project_model_to_quantization(net, client.quant_bits, False)
        ldr_train = DataLoader(DatasetSplit(self.dataset, client.data_idx), batch_size=self.args.local_bs, shuffle=True)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr*(self.args.lr_decay**round),
                                        momentum=self.args.momentum,weight_decay=self.args.weight_decay)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                images = maybe_quantize_batch(images, client.quant_bits, False)
                net.zero_grad()
                log_probs = net(images)['output']
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                project_model_to_quantization(net, client.quant_bits, False)
        net = net.to('cpu')
        client.local_net = copy.deepcopy(net)
        client.trans_net = copy.deepcopy(net)
        return


def LD_SGD(args, net_glob, dataset_train, dataset_test, dict_users):
    args.quant_aware = 0
    args.payload_codec = 'none'
    args.payload_compression_level = 0
    net_size = sum([param.nelement() for param in net_glob.parameters()])
    base_payload_bytes = max(state_dict_payload_nbytes(net_glob, args.quant_comm_base_bits, enabled=False), 1)
    payload_codec = 'none'
    payload_compression_level = 0
    comm_8bit_format = getattr(args, 'quant_comm_8bit_format', 'int8')
    target_time = dict()
    target_comm = dict()
    target_acc1 = 55
    target_acc2 = 60
    acc = dict()
    for idx in range(args.num_users):
        client = Client(idx, dict_users[idx], net_glob, args)
        Global_Client_set.append(client)
        acc["client" + str(idx)] = []
    acc["acc"] = []
    time_consume = []
    comm_consume = []
    comm_time_consume = []
    energy_consume = []
    battery_history = []
    battery_state = initialize_battery_state(Global_Client_set)
    current_time = 0
    current_comm = 0
    current_energy = 0
    for iter in range(args.epochs):
        print('*' * 80)
        print('Round {:3d}'.format(iter), '  current time: ', current_time)
        active_clients = select_baseline_active_clients(args)
        active_client_set = set(active_clients)
        print("active clients:", active_clients)
        avg_acc = 0
        round_time = 0
        round_comm = 0
        round_comm_time = 0
        round_energy = 0
        # training of each client
        if iter % 5 in [0, 1, 2]:
            for idx in active_clients:
                local = LocalUpdate_LD_SGD(args=args, dataset=dataset_train)
                local.train(client=Global_Client_set[idx],round=iter)
                training_time = get_client_training_time(idx)
                train_energy = get_training_energy(idx, training_time)
                round_time = max(round_time, training_time)
                round_energy = round_energy + train_energy
                consume_energy(battery_state, idx, train_energy)
        elif iter % 5 in [3, 4]:
            for idx in active_clients:
                local = LocalUpdate_LD_SGD(args=args, dataset=dataset_train)
                local.train(client=Global_Client_set[idx], round=iter)
            # communicate with neighboring clients to update local model
            for idx in active_clients:
                w_locals = []
                lens = []
                training_time = get_client_training_time(idx)
                t = training_time
                train_energy = get_training_energy(idx, training_time)
                round_energy = round_energy + train_energy
                consume_energy(battery_state, idx, train_energy)
                for nb_idx in range(args.num_users):
                    if Adjacency_matrix[idx][nb_idx] == 1 and nb_idx in active_client_set:
                        sender_state, sender_payload_bytes, _ = transmit_state_dict(
                            Global_Client_set[nb_idx].trans_net,
                            args.quant_comm_base_bits,
                            False,
                            codec=payload_codec,
                            compression_level=payload_compression_level,
                            comm_8bit_format=comm_8bit_format,
                        )
                        w_locals.append(sender_state)
                        lens.append(Global_Client_set[nb_idx].data_cnt)
                        comm_ratio = max(sender_payload_bytes / float(base_payload_bytes), 1e-12)
                        comm_time = get_client_communication_time(nb_idx, idx, multiplier=comm_ratio)
                        t = t + comm_time
                        src_energy, dst_energy = get_communication_energy_breakdown(nb_idx, idx, comm_time)
                        round_energy = round_energy + src_energy + dst_energy
                        consume_communication_energy(battery_state, nb_idx, idx, src_energy, dst_energy)
                        round_comm = round_comm + sender_payload_bytes / (1024 * 1024)
                        round_comm_time = round_comm_time + comm_time
                round_time = max(round_time, t)
                own_state, own_payload_bytes, _ = transmit_state_dict(
                    Global_Client_set[idx].trans_net,
                    args.quant_comm_base_bits,
                    False,
                    codec=payload_codec,
                    compression_level=payload_compression_level,
                    comm_8bit_format=comm_8bit_format,
                )
                w_locals.append(own_state)
                lens.append(Global_Client_set[idx].data_cnt)

                w_agg = Aggregation(w_locals, lens)
                Global_Client_set[idx].local_net.load_state_dict(w_agg)
        for idx in range(args.num_users):
            acc["client" + str(idx)].append(test(Global_Client_set[idx].local_net, dataset_test, args))
            avg_acc = avg_acc + acc["client" + str(idx)][-1]
            print("client ", idx, "acc: ", acc["client" + str(idx)][-1])
        current_time = current_time + round_time
        current_comm = current_comm + round_comm
        current_comm_time = (comm_time_consume[-1] if comm_time_consume else 0) + round_comm_time
        current_energy = current_energy + round_energy
        acc["acc"].append(avg_acc / args.num_users)
        time_consume.append(current_time)
        comm_consume.append(current_comm)
        comm_time_consume.append(current_comm_time)
        energy_consume.append(current_energy)
        battery_history.append(snapshot_battery_state(battery_state))
        print("acc acc: ", acc["acc"][-1])
        if acc["acc"][-1] >= target_acc1:
            if target_acc1 not in target_time:
                target_time[target_acc1] = time_consume[-1]
            if target_acc1 not in target_comm:
                target_comm[target_acc1] = comm_consume[-1]
        if acc["acc"][-1] >= target_acc2:
            if target_acc2 not in target_time:
                target_time[target_acc2] = time_consume[-1]
            if target_acc2 not in target_comm:
                target_comm[target_acc2] = comm_consume[-1]

    save_result(acc, 'test_acc', args)
    save_result(time_consume, 'time', args)
    save_result(comm_consume, 'comm', args)
    save_result(comm_time_consume, 'comm_time', args)
    save_result(energy_consume, 'energy', args)
    save_result(battery_history, 'battery', args)
    print("target_time:", target_time)
    print("target_comm:", target_comm)
    for key in acc.keys():
        print(key)
        avg_acc_and_var(acc[key])
