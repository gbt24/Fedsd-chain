# -*- coding: UTF-8 -*-
import sys
import unittest
from unittest.mock import patch

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

        class FakeCelebA:
            def __init__(self, *args, **kwargs):
                self.images = torch.arange(12).view(6, 2)
                self.targets = torch.ones(6, 40)

            def __len__(self):
                return len(self.images)

            def __getitem__(self, index):
                return self.images[index], self.targets[index]

        with patch.object(dataset_module, "CelebA", side_effect=[FakeCelebA(), FakeCelebA()]):
            train_dataset, test_dataset = dataset_module.get_full_dataset(
                "celeba64",
                img_size=(64, 64),
                max_train_samples=4,
                max_test_samples=3,
                seed=5,
            )

        _, train_label = train_dataset[0]
        _, test_label = test_dataset[0]
        self.assertEqual(len(train_dataset), 4)
        self.assertEqual(len(test_dataset), 3)
        self.assertEqual(train_label, 0)
        self.assertEqual(test_label, 0)


if __name__ == "__main__":
    unittest.main()
