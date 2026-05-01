# -*- coding: UTF-8 -*-
import copy
import numpy as np
from torch import nn
import torch
import random
from geneal.genetic_algorithms import BinaryGenAlgSolver


def dec2bin(num, length):
    mid = []
    while True:
        if num == 0:
            break
        num, rem = divmod(num, 2)
        if int(rem) == 0:
            mid.append(-1)
        else:
            mid.append(1)
    while len(mid) < length:
        mid.insert(0, -1)
    return mid


def generate_fingerprints(num_clients, length):
    np.random.seed(0)
    random.seed(0)
    fingerprints_int = set()
    while len(fingerprints_int) < num_clients:
        fingerprints_int.add(random.getrandbits(length))
    solver = BinaryGenAlgSolver(
        n_genes=num_clients,
        fitness_function=get_minimum_hamming_distance_func(num_clients, length),
        n_bits=length,
        pop_size=10,
        max_gen=50,
        mutation_rate=0.05,
        selection_rate=0.5,
    )
    solver.solve()
    fingerprints = []
    count = 0
    for i in range(num_clients):
        fingerprint = np.array(solver.best_individual_[count : count + length])
        fingerprint[fingerprint == 0.0] = -1.0
        fingerprints.append(fingerprint)
        count += length
    return fingerprints


def hamming_distance(a, b):
    return bin(int(a) ^ int(b)).count("1")


def get_minimum_hamming_distance_func(num_clients, length):
    def minimum_hamming_distance(fingerprints):
        x = fingerprints.reshape(num_clients, length)
        n = num_clients
        min_hamming = 100000
        for i in range(n):
            if sum(x[i] == np.ones(x[i].shape)) == 0:
                return -100000
            for j in range(i + 1, n):
                distance = np.sum(x[i] != x[j])
                if distance == 0:
                    return -100000
                if distance < min_hamming:
                    min_hamming = distance
        return min_hamming

    return minimum_hamming_distance


def generate_extracting_matrices(weight_size, total_length, num_clients):
    np.random.seed(0)
    extracting_matrices = []
    for i in range(num_clients):
        extracting_matrices.append(
            np.random.standard_normal((total_length, weight_size)).astype(np.float32)
        )
    return extracting_matrices


class HingeLikeLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, results, labels):
        loss = torch.mul(results, labels)
        loss = torch.mul(loss, -1)
        loss = torch.add(loss, self.epsilon)
        loss = torch.sum(torch.clamp(loss, min=0))
        return loss


def calculate_local_grad(layers, local_fingerprint, extracting_metrix, epsilon=0.5):
    for layer in layers:
        layer.zero_grad()
    weight = layers[0].weight.detach().cpu().numpy().flatten()
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().cpu().numpy().flatten())
    weight = nn.Parameter(torch.from_numpy(weight))
    loss_func = HingeLikeLoss(epsilon=epsilon)
    matrix = torch.from_numpy(extracting_metrix).float()
    fingerprint = torch.from_numpy(local_fingerprint).float()
    result = torch.matmul(matrix, weight)
    loss = loss_func(result, fingerprint)
    loss.backward()
    return copy.deepcopy(weight.grad)


def extracting_fingerprints(
    layers, local_fingerprints, extracting_matrices, epsilon=0.5, hd=False
):
    min_ber = 100000
    min_idx = 0
    max_score = -100000
    max_idx = 0
    bit_length = local_fingerprints[0].shape[0]
    weight = layers[0].weight.detach().cpu().numpy().flatten()
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().cpu().numpy().flatten())
    for idx in range(len(local_fingerprints)):
        matrix = extracting_matrices[idx]
        result = np.dot(matrix, weight)
        if hd:
            result[result >= 0] = 1
            result[result < 0] = -1
            ber = np.sum(result != local_fingerprints[idx]) / bit_length
            if ber < min_ber:
                min_ber = ber
                min_idx = idx
        else:
            result = np.multiply(result, local_fingerprints[idx])
            result[result > epsilon] = epsilon
            score = np.sum(result) / bit_length / epsilon
            if score > max_score:
                max_score = score
                max_idx = idx
    if hd:
        return min_ber, min_idx
    else:
        return max_score, max_idx


def get_diffusion_embed_layers(model, embed_layer_names):
    embed_layers = []
    embed_layer_names_list = embed_layer_names.split(";")

    for embed_layer_name in embed_layer_names_list:
        parts = embed_layer_name.split(".")
        embed_layer = model

        if hasattr(model, "unet"):
            embed_layer = model.unet
            for part in parts:
                if part.isdigit():
                    embed_layer = embed_layer[int(part)]
                else:
                    embed_layer = getattr(embed_layer, part)
        else:
            for part in parts:
                if part.isdigit():
                    embed_layer = embed_layer[int(part)]
                else:
                    if hasattr(embed_layer, part):
                        embed_layer = getattr(embed_layer, part)
                    elif part == "attentions" and hasattr(embed_layer, "attention"):
                        embed_layer = getattr(embed_layer, "attention")
                    else:
                        embed_layer = getattr(embed_layer, part)

        embed_layers.append(embed_layer)

    return embed_layers


def get_diffusion_embed_layers_length(model, embed_layer_names):
    weight_size = 0
    embed_layers = get_diffusion_embed_layers(model, embed_layer_names)
    for embed_layer in embed_layers:
        weight_size += embed_layer.weight.numel()
    return weight_size


def set_diffusion_embed_layers_weights(
    model, embed_layer_names, grad_update, lambda_factor=-0.01
):
    embed_layers = get_diffusion_embed_layers(model, embed_layer_names)
    weight_count = 0

    for embed_layer in embed_layers:
        weight_length = embed_layer.weight.numel()
        embed_layer.weight = nn.Parameter(
            embed_layer.weight
            + lambda_factor
            * grad_update[weight_count : weight_count + weight_length].view_as(
                embed_layer.weight
            )
        )
        weight_count += weight_length

    return model
