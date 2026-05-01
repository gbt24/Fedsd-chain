# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, CIFAR100, LSUN
import numpy as np
import os
import sys
from PIL import Image


class LSUNBedroom(Dataset):
    def __init__(
        self, root, split="train", download=True, transform=None, max_samples=None
    ):
        self.root = root
        self.split = split
        self.transform = transform

        if download:
            self._download()

        data_dir = os.path.join(root, "bedroom")
        if split == "train":
            data_dir = os.path.join(data_dir, "train")
        elif split == "val":
            data_dir = os.path.join(data_dir, "val")
        else:
            data_dir = os.path.join(data_dir, "test")

        self.image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            self.image_paths.extend(
                sorted(
                    [
                        os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.endswith(ext.replace("*", ""))
                    ]
                )
                if os.path.exists(data_dir)
                else []
            )

        if max_samples is not None and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]

        self.targets = [0] * len(self.image_paths)

    def _download(self):
        if os.path.exists(os.path.join(self.root, "bedroom")):
            return
        try:
            dataset = LSUN(self.root, classes=["bedroom_train"], download=True)
            del dataset
        except Exception as e:
            print(f"LSUN download note: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == "mnist":
        train_dataset = MNIST(
            "./data/mnist/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
        test_dataset = MNIST(
            "./data/mnist/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        )
    elif dataset_name == "cifar10":
        train_dataset = CIFAR10(
            "./data/cifar10/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_dataset = CIFAR10(
            "./data/cifar10/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif dataset_name == "cifar100":
        train_dataset = CIFAR100(
            "./data/cifar100/",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Pad(4, padding_mode="reflect"),
                    transforms.RandomCrop(img_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_dataset = CIFAR100(
            "./data/cifar100/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize(img_size),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif dataset_name == "lsun_bedroom" or dataset_name == "lsun":
        train_dataset = LSUNBedroom(
            "./data/lsun/",
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        test_dataset = LSUNBedroom(
            "./data/lsun/",
            split="val",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset


def iid_split(dataset, num_clients):
    """
    Split I.I.D client data
    :param dataset:
    :param num_clients:
    :return: dict of image indexes
    """

    dataset_len = len(dataset)
    num_items = dataset_len // num_clients
    dict_clients = dict()
    all_idxs = [i for i in range(dataset_len)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def dniid_split(dataset, num_clients, param=0.8):
    """
    Using Dirichlet distribution to sample non I.I.D client data
    :param dataset:
    :param num_clients:
    :param param: parameter used in Dirichlet distribution
    :return: dict of image indexes
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)
    labels = set(dataset_y)
    sorted_idxs = dict()
    for label in labels:
        sorted_idxs[label] = []

    # sort indexes by labels
    for i in range(dataset_len):
        sorted_idxs[dataset_y[i]].append(i)

    for label in labels:
        sorted_idxs[label] = np.array(sorted_idxs[label])

    # initialize the clients' dataset dict
    dict_clients = dict()
    for i in range(num_clients):
        dict_clients[i] = np.array([], dtype=np.int64)
    # split the dataset separately
    for label in labels:
        idxs = sorted_idxs[label]
        sample_split = np.random.dirichlet(np.array(num_clients * [param]))
        accum = 0.0
        num_of_current_class = idxs.shape[0]
        for i in range(num_clients):
            client_idxs = idxs[
                int(accum * num_of_current_class) : min(
                    dataset_len, int((accum + sample_split[i]) * num_of_current_class)
                )
            ]
            dict_clients[i] = np.concatenate((dict_clients[i], client_idxs))
            accum += sample_split[i]
    return dict_clients


def pniid_split(dataset, num_clients, num_of_shards_each_clients=2):
    """
    Simulate pathological non I.I.D distribution
    :param dataset:
    :param num_clients:
    :param num_of_shards_each_clients:
    :return:
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)

    sorted_idxs = np.argsort(dataset_y)

    size_of_each_shards = dataset_len // (num_clients * num_of_shards_each_clients)
    per = np.random.permutation(num_clients * num_of_shards_each_clients)
    dict_clients = dict()
    for i in range(num_clients):
        idxs = np.array([])
        for j in range(num_of_shards_each_clients):
            idxs = np.concatenate(
                (
                    idxs,
                    sorted_idxs[
                        per[num_of_shards_each_clients * i + j]
                        * size_of_each_shards : min(
                            dataset_len,
                            (per[num_of_shards_each_clients * i + j] + 1)
                            * size_of_each_shards,
                        )
                    ],
                )
            )
        dict_clients[i] = idxs
    return dict_clients
