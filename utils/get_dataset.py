#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchvision import datasets, transforms
from utils import mydata
from utils.sampling import *
from utils.dataset_utils import separate_data,read_record
from utils.FEMNIST import FEMNIST
import os
import json
import torch
import torch.nn.functional as F
try:
    import torchaudio
except ModuleNotFoundError:
    torchaudio = None
from torch.utils.data import DataLoader, Dataset, Subset
from utils.tinyimagenet import TinyImageNet


SPEECHCOMMANDS_LABELS = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin',
    'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six',
    'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero',
]


class WaveformToImage(object):
    def __init__(self, sample_rate=16000, n_mels=32, image_size=32):
        if torchaudio is None:
            raise RuntimeError('torchaudio is required for audio datasets')
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        self.image_size = image_size

    def __call__(self, waveform):
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        spec = self.mel(waveform)
        spec = torch.log1p(spec)
        spec = F.interpolate(spec.unsqueeze(0), size=(self.image_size, self.image_size),
                             mode='bilinear', align_corners=False).squeeze(0)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        return spec.repeat(3, 1, 1)


class SpeechCommandsImageDataset(Dataset):
    def __init__(self, root, subset, download=True):
        os.makedirs(root, exist_ok=True)
        self.dataset = torchaudio.datasets.SPEECHCOMMANDS(root, download=download, subset=subset)
        self.label_to_idx = {label: idx for idx, label in enumerate(SPEECHCOMMANDS_LABELS)}
        self.transform = WaveformToImage()
        self.filtered_indices = [
            i for i in range(len(self.dataset))
            if self.dataset[i][2] in self.label_to_idx
        ]
        self.targets = [self.label_to_idx[self.dataset[i][2]] for i in self.filtered_indices]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, index):
        waveform, sample_rate, label, _, _ = self.dataset[self.filtered_indices[index]]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return self.transform(waveform), self.label_to_idx[label]


class YesNoImageDataset(Dataset):
    def __init__(self, root, download=True):
        os.makedirs(root, exist_ok=True)
        self.dataset = torchaudio.datasets.YESNO(root, download=download)
        self.transform = WaveformToImage()
        self.targets = [sum(self.dataset[i][2]) for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sample_rate, labels = self.dataset[index]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return self.transform(waveform), sum(labels)


def _limit_dataset(dataset, max_samples, seed):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def _raw_data_root(args):
    return os.path.abspath(getattr(args, 'data_root', './data') or './data')


def _cifar10_root(data_root):
    if (os.path.exists(os.path.join(data_root, 'cifar-10-batches-py')) or
            os.path.exists(os.path.join(data_root, 'cifar-10-python.tar.gz'))):
        return data_root
    return os.path.join(data_root, 'cifar10')


def _cifar100_root(data_root):
    if (os.path.exists(os.path.join(data_root, 'cifar-100-python')) or
            os.path.exists(os.path.join(data_root, 'cifar-100-python.tar.gz'))):
        return data_root
    return os.path.join(data_root, 'cifar100')


def _tinyimagenet_root(data_root):
    nested = os.path.join(data_root, 'tiny-imagenet-200')
    if os.path.exists(nested):
        return nested
    return data_root


def get_dataset(args):
    data_root = _raw_data_root(args)

    file = os.path.join("data", args.dataset + "_" + str(args.num_users))
    if args.iid:
        file += "_iid"
    else:
        file += "_noniidCase" + str(args.noniid_case)

    if args.noniid_case > 4:
        file += "_beta" + str(args.data_beta)

    partition_tag = str(getattr(args, 'partition_tag', '') or '').strip()
    if partition_tag:
        safe_tag = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in partition_tag)
        file += "_" + safe_tag

    file += ".json"
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
        ])
        dataset_train = datasets.MNIST(os.path.join(data_root, 'mnist'), train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(os.path.join(data_root, 'mnist'), train=False, download=True, transform=trans_mnist)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar10':
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        cifar10_root = _cifar10_root(data_root)
        dataset_train = datasets.CIFAR10(cifar10_root, train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10(cifar10_root, train=False, download=True, transform=trans)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train,args.num_users,args.noniid_case)
            else:
                dict_users = separate_data(dataset_train,args.num_users,args.num_classes,args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'cifar100':
        args.num_channels = 3
        trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                        std=[0.267, 0.256, 0.276])])
        trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                      std=[0.267, 0.256, 0.276])])
        cifar100_root = _cifar100_root(data_root)
        if args.cifar100_coarse == 0:
            args.num_classes = 100
            dataset_train = datasets.CIFAR100(cifar100_root, train=True, download=True, transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100(cifar100_root, train=False, download=True, transform=trans_cifar100_val)
        else:
            args.num_classes = 20
            dataset_train = mydata.CIFAR100_coarse(cifar100_root, train=True, download=True, transform=trans_cifar100_train)
            dataset_test = mydata.CIFAR100_coarse(cifar100_root, train=False, download=True, transform=trans_cifar100_val)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'fashion-mnist':
        trans = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
        dataset_train = datasets.FashionMNIST(os.path.join(data_root, 'fashion-mnist'), train=True, download=True, transform=trans)
        dataset_test = datasets.FashionMNIST(os.path.join(data_root, 'fashion-mnist'), train=False, download=True, transform=trans)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            if args.iid:
                dict_users = fashion_mnist_iid(dataset_train, args.num_users)
            elif args.noniid_case < 5:
                dict_users = fashion_mnist_noniid(dataset_train, args.num_users, case=args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'speechcommands':
        args.num_channels = 3
        args.num_classes = len(SPEECHCOMMANDS_LABELS)
        dataset_train = SpeechCommandsImageDataset(os.path.join(data_root, 'speechcommands'), subset='training', download=True)
        dataset_test = SpeechCommandsImageDataset(os.path.join(data_root, 'speechcommands'), subset='testing', download=True)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'yesno':
        args.num_channels = 3
        args.num_classes = 9
        dataset_all = YesNoImageDataset(os.path.join(data_root, 'yesno'), download=True)
        split_index = int(len(dataset_all) * 0.8)
        train_indices = list(range(split_index))
        test_indices = list(range(split_index, len(dataset_all)))
        dataset_train = Subset(dataset_all, train_indices)
        dataset_test = Subset(dataset_all, test_indices)
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)
        if args.generate_data:
            dict_users = iid(dataset_train, args.num_users)
        else:
            dict_users = read_record(file)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(True)
        dataset_test = FEMNIST(False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        args.num_channels = 1
        args.num_classes = 62
    elif args.dataset == 'TinyImagenet':

        trans_imagenet_train = transforms.Compose([transforms.RandomCrop(64),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                                                        std=[0.2770, 0.2691, 0.2821])])
        trans_imagenet_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                 std=[0.2770, 0.2691, 0.2821])])
        data_dir = _tinyimagenet_root(data_root)
        dataset_train = TinyImageNet(data_dir, train=True, transform=trans_imagenet_train)
        dataset_test = TinyImageNet(data_dir, train=False, transform=trans_imagenet_val)
        args.num_channels = 3
        args.num_classes = 200
        dataset_train = _limit_dataset(dataset_train, args.max_train_samples, args.seed)
        dataset_test = _limit_dataset(dataset_test, args.max_test_samples, args.seed + 1)

        if args.generate_data:
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            elif args.noniid_case < 4:
                dict_users = cifar_noniid(dataset_train, args.num_users, args.noniid_case)
            else:
                dict_users = separate_data(dataset_train, args.num_users, args.num_classes, args.data_beta)
        else:
            dict_users = read_record(file)
    else:
        exit('Error: unrecognized dataset')

    if args.generate_data:
        with open(file,'w') as f:
            dataJson = {"dataset":args.dataset,"num_users":args.num_users,"iid":args.iid,"noniid_case":args.noniid_case,"data_beta":args.data_beta,"partition_tag":partition_tag,"train_data":dict_users}
            json.dump(dataJson,f)

    return dataset_train, dataset_test, dict_users


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
