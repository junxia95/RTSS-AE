#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
from torchvision import models
try:
    from vit_pytorch import SimpleViT
except ModuleNotFoundError:
    SimpleViT = None

from utils.options import args_parser
from utils.get_dataset import get_dataset
from utils.set_seed import set_random_seed
from utils.utils import save_result
from config import client_device_profile_list
import torch
from models.SplitModel import *
matplotlib.use('Agg')

if __name__ == '__main__':
    # args initialize
    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    set_random_seed(args.seed)
    save_result(client_device_profile_list, 'device_profile', args)

    # dataset initialize
    dataset_train, dataset_test, dict_users = get_dataset(args)

    # model initialize
    client_net_list = []
    if 'resnet8' in args.model:
        net_glob = ResNet8_entire(in_channels=args.num_channels, num_classes=args.num_classes)
        share_net_glob = ResNet8_share(in_channels=args.num_channels)
        private_net_glob = ResNet8_private(num_classes=args.num_classes)
    if 'vgg' in args.model:
        net_glob = VGG16_entire(in_channels=args.num_channels, num_classes=args.num_classes)
    if 'mobilenet' in args.model:
        net_glob = mobilenet_entire(in_channels=args.num_channels, num_classes=args.num_classes)

    net_glob.apply(init_weights)
    share_net_glob.apply(init_weights)
    private_net_glob.apply(init_weights)

    if args.algorithm == 'DFL':
        from Algorithm.Training_DFL import DFL
        DFL(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFL_MM':
        from Algorithm.Training_DFL_MM import DFL_MM
        DFL_MM(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm in ['AutoRL_DFL', 'AutoRL_DFL_MM', 'SelfEvolvingRL_DFL']:
        from Algorithm.Training_AutoRL_DFL import AutoRL_DFL
        AutoRL_DFL(args, net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'DFedPGP':
        from Algorithm.Training_DFedPGP import DFedPGP
        DFedPGP(args, share_net_glob, private_net_glob, dataset_train, dataset_test, dict_users)
    elif args.algorithm == 'LD_SGD':
        from Algorithm.Training_LD_SGD import LD_SGD
        LD_SGD(args, net_glob, dataset_train, dataset_test, dict_users)
    else:
        raise "%s algorithm has not achieved".format(args.algorithm)
