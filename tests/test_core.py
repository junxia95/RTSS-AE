import importlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.SplitModel import ResNet8_entire
from utils.dataset_utils import separate_data
from utils.FL_utils import Aggregation
from utils.int8_accel import extract_logits
from utils.quantization import pack_tensor_payload, unpack_tensor_payload


class DummyDataset:
    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.zeros(3, 32, 32), self.targets[index]


class CoreSmokeTests(unittest.TestCase):
    def test_partition_is_deterministic_and_disjoint(self):
        dataset = DummyDataset([index % 4 for index in range(200)])
        np.random.seed(1)
        first = separate_data(dataset, num_clients=4, num_classes=4, beta=0.5)
        np.random.seed(1)
        second = separate_data(dataset, num_clients=4, num_classes=4, beta=0.5)
        self.assertEqual(first, second)
        assigned = [index for values in first.values() for index in values]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), set(range(len(dataset))))

    def test_quantized_payload_sizes_and_roundtrip(self):
        tensor = torch.linspace(-1.0, 1.0, steps=4096).reshape(64, 64)
        payloads = {bits: pack_tensor_payload("weight", tensor, bits, enabled=True) for bits in (8, 16, 32)}
        self.assertLess(payloads[8]["payload_bytes"], payloads[16]["payload_bytes"])
        self.assertLess(payloads[16]["payload_bytes"], payloads[32]["payload_bytes"])
        for bits, payload in payloads.items():
            restored = unpack_tensor_payload(payload)
            self.assertEqual(restored.shape, tensor.shape)
            self.assertTrue(torch.isfinite(restored).all(), msg=f"invalid {bits}-bit roundtrip")

    def test_weighted_aggregation(self):
        models = [{"w": torch.tensor([1.0])}, {"w": torch.tensor([3.0])}]
        averaged = Aggregation(models, [1, 3])
        self.assertTrue(torch.allclose(averaged["w"], torch.tensor([2.5])))

    def test_route_selection_stays_on_graph(self):
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            module = importlib.import_module("utils.device_rl")
        finally:
            sys.argv = old_argv
        profile = {
            "type": "strong",
            "frequency_ghz": 2.4,
            "compute_power_w": 28.0,
            "communication_power_w": 4.5,
        }
        clients = [
            SimpleNamespace(device_profile=profile, quant_bits=32, label_distribution=np.array([20, 0])),
            SimpleNamespace(device_profile=profile, quant_bits=32, label_distribution=np.array([0, 20])),
        ]
        args = SimpleNamespace(
            epochs=2,
            num_users=2,
            quant_aware=1,
            quant_comm_base_bits=32,
            rl_accuracy_weight=0.65,
            rl_energy_weight=0.25,
            rl_latency_weight=0.10,
            rl_lr=0.2,
            rl_discount=0.9,
            rl_epsilon=0.0,
            rl_state_bins=4,
        )
        adjacency = np.array([[0, 1], [1, 0]])
        edge_types = [[None, "strong"], ["strong", None]]
        battery = [
            {"remaining_j": 1000.0, "capacity_j": 1000.0},
            {"remaining_j": 1000.0, "capacity_j": 1000.0},
        ]
        random.seed(1)
        controller = module.DeviceWalkRLController(args, clients, edge_types)
        selected = controller.select_next(0, 0, np.array([20, 0]), battery, adjacency, 0)
        self.assertEqual(selected, 1)

    def test_resnet8_training_step(self):
        torch.manual_seed(1)
        model = ResNet8_entire(in_channels=3, num_classes=10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.randn(4, 3, 32, 32)
        targets = torch.tensor([0, 1, 2, 3])
        loss = torch.nn.functional.cross_entropy(extract_logits(model(inputs)), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
