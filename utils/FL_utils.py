import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from utils.int8_accel import build_int8_eval_model, extract_logits

def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def uniform_distribution_loss(a, b=None):
    if b is None:
        b = np.zeros_like(a)
    sum_vec = a + b
    if np.sum(sum_vec) == 0:
        return 0
    sum_vec = sum_vec / np.sum(sum_vec)
    uniform_vec = np.array([1 / len(sum_vec) for _ in range(len(sum_vec))])
    return np.linalg.norm(sum_vec - uniform_vec)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg

def Aggregation_push_sum_w(push_sum_locals, lens):
    sum = 0
    for i in range(len(push_sum_locals)):
        sum = sum + push_sum_locals[i] * lens[i]
    return sum


def Aggregation_AM(AM, lens):
    AM_agg = None
    total_count = sum(lens)

    for i in range(0, len(AM)):
        if i == 0:
            AM_agg = copy.deepcopy(AM[i])
            for k in AM_agg.keys():
                AM_agg[k] = AM[i][k] * lens[i]
        else:
            for k in AM_agg.keys():
                AM_agg[k] = AM_agg[k] + AM[i][k] * lens[i]
    for k in AM_agg.keys():
        AM_agg[k] = AM_agg[k] / total_count

    return AM_agg


def test(net_glob, dataset_test, args):
    # testing
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    #print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()

def test_split(share_net, private_net, dataset_test, args):
    # testing
    acc_test, loss_test = test_img_split(share_net, private_net, dataset_test, args)

    # print("Testing accuracy: {:.2f}".format(acc_test))

    return acc_test.item()


def test_img(net_g, datatest, args):
    data_loader = DataLoader(datatest, batch_size=args.bs)
    sample_batch = None
    try:
        sample_batch = next(iter(data_loader))[0]
    except StopIteration:
        return 0.0, 0.0

    eval_backend = getattr(args, 'int8_kernel_backend', 'none')
    calib_batches = max(int(getattr(args, 'int8_calib_batches', 8)), 1)
    eval_net, eval_device, used_backend = build_int8_eval_model(
        copy.deepcopy(net_g),
        sample_batch,
        calibration_loader=data_loader,
        backend=eval_backend,
        calib_batches=calib_batches,
    )

    eval_net.eval()
    eval_net.to(eval_device)
    # testing
    test_loss = 0
    correct = 0
    l = len(data_loader)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(eval_device), target.to(eval_device)
            log_probs = extract_logits(eval_net(data))
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    if used_backend == 'none':
        net_g.to('cpu')
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def test_img_split(share_net, private_net, datatest, args):
    data_loader = DataLoader(datatest, batch_size=args.bs)
    sample_batch = None
    try:
        sample_batch = next(iter(data_loader))[0]
    except StopIteration:
        return 0.0, 0.0

    eval_backend = getattr(args, 'int8_kernel_backend', 'none')
    calib_batches = max(int(getattr(args, 'int8_calib_batches', 8)), 1)
    eval_share_net, eval_device, used_backend = build_int8_eval_model(
        copy.deepcopy(share_net),
        sample_batch,
        calibration_loader=data_loader,
        backend=eval_backend,
        calib_batches=calib_batches,
    )
    eval_private_net, _, _ = build_int8_eval_model(
        copy.deepcopy(private_net),
        sample_batch,
        calibration_loader=data_loader,
        backend=eval_backend,
        calib_batches=calib_batches,
    )

    eval_share_net.eval()
    eval_private_net.eval()
    # testing
    test_loss = 0
    correct = 0
    eval_share_net.to(eval_device)
    eval_private_net.to(eval_device)
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(eval_device), target.to(eval_device)
            log_probs = extract_logits(eval_private_net(eval_share_net(data)))
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    if used_backend == 'none':
        share_net.to('cpu')
        private_net.to('cpu')
    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

def avg_acc_and_var(acc_list):
    max_acc = -1
    cor_var = 0
    for i in range(len(acc_list)):
        j = i + 50
        if j < len(acc_list):
            mean = np.mean(acc_list[i:j])
            var = np.std(acc_list[i:j])
            if mean > max_acc:
                max_acc = mean
                cor_var = var
    print(max_acc,"(",cor_var,")")
