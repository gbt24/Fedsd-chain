# -*- coding: UTF-8 -*-
import sys
import unittest
from unittest.mock import mock_open, patch

import torch
from torch.utils.data import TensorDataset

from utils.utils import load_args
from utils.datasets import apply_deterministic_subset


class DatasetSubsetArgsTest(unittest.TestCase):
    def test_load_args_accepts_dataset_subset_limits(self):
        argv = [
            "prog",
            "--max_train_samples",
            "20000",
            "--max_test_samples",
            "5000",
        ]
        with patch.object(sys, "argv", argv):
            args = load_args()

        self.assertEqual(args.max_train_samples, 20000)
        self.assertEqual(args.max_test_samples, 5000)


class DeterministicSubsetTest(unittest.TestCase):
    def test_apply_deterministic_subset_returns_reproducible_indices(self):
        dataset = TensorDataset(torch.arange(10))

        first = apply_deterministic_subset(dataset, max_samples=4, seed=7)
        second = apply_deterministic_subset(dataset, max_samples=4, seed=7)

        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first), 4)

    def test_apply_deterministic_subset_keeps_full_dataset_when_limit_is_none(self):
        dataset = TensorDataset(torch.arange(10))
        result = apply_deterministic_subset(dataset, max_samples=None, seed=7)

        self.assertIs(result, dataset)


class GetFullDatasetSubsetTest(unittest.TestCase):
    def test_get_full_dataset_applies_train_and_test_limits(self):
        import utils.datasets as dataset_module

        fake_train = TensorDataset(torch.arange(10))
        fake_test = TensorDataset(torch.arange(8))

        with patch.object(dataset_module, "CIFAR10", side_effect=[fake_train, fake_test]):
            train_dataset, test_dataset = dataset_module.get_full_dataset(
                "cifar10",
                img_size=(32, 32),
                max_train_samples=4,
                max_test_samples=3,
                seed=5,
            )

        self.assertEqual(len(train_dataset), 4)
        self.assertEqual(len(test_dataset), 3)

    def test_get_full_dataset_celeba64_returns_constant_class_labels(self):
        import utils.datasets as dataset_module

        partition_lines = "000001.jpg 0\n000002.jpg 0\n000003.jpg 1\n000004.jpg 1\n"

        class FakeDataset:
            def __init__(self, *args, **kwargs):
                pass
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                return torch.zeros(3, 64, 64), 0

        m_open = mock_open(read_data=partition_lines)
        def fake_isfile(path):
            return True

        with patch("builtins.open", m_open), patch("os.path.isfile", fake_isfile), \
             patch.object(dataset_module, "FlatImageDataset", FakeDataset):
            train_dataset, test_dataset = dataset_module.get_full_dataset(
                "celeba64",
                img_size=(64, 64),
            )

        train_label = train_dataset[0][1]
        test_label = test_dataset[0][1]
        self.assertEqual(train_label, 0)
        self.assertEqual(test_label, 0)


if __name__ == "__main__":
    unittest.main()
