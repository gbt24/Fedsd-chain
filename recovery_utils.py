# -*- coding: UTF-8 -*-
import copy
import json
import os
import re

import numpy as np
import torch
from torch import nn


def find_latest_checkpoint(file_names):
    pattern = re.compile(r"checkpoint_epoch_(\d+)\.pth$")
    best_name = None
    best_epoch = -1
    for file_name in file_names:
        match = pattern.search(file_name)
        if not match:
            continue
        epoch = int(match.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_name = file_name
    return best_name


def trace_metadata_matches_args(metadata, args, weight_size):
    return (
        metadata.get("num_clients") == args.get("num_clients")
        and metadata.get("lfp_length") == args.get("lfp_length")
        and metadata.get("embed_layer_names") == args.get("embed_layer_names")
        and metadata.get("extract_matrices_shape", [None, None, None])[2] == weight_size
    )


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_diffusion_embed_layers(model, embed_layer_names):
    embed_layers = []
    for embed_layer_name in embed_layer_names.split(";"):
        embed_layer = model
        for part in embed_layer_name.split("."):
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


class HingeLikeLoss(nn.Module):
    def __init__(self, epsilon=1.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, results, labels):
        loss = torch.mul(results, labels)
        loss = torch.mul(loss, -1)
        loss = torch.add(loss, self.epsilon)
        return torch.sum(torch.clamp(loss, min=0))


def calculate_local_grad(layers, local_fingerprint, extracting_matrix, epsilon=0.5):
    for layer in layers:
        layer.zero_grad()
    weight = layers[0].weight.detach().cpu().numpy().flatten()
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().cpu().numpy().flatten())
    weight = nn.Parameter(torch.from_numpy(weight))
    loss_func = HingeLikeLoss(epsilon=epsilon)
    matrix = torch.from_numpy(extracting_matrix).float()
    fingerprint = torch.from_numpy(local_fingerprint).float()
    result = torch.matmul(matrix, weight)
    loss = loss_func(result, fingerprint)
    loss.backward()
    return copy.deepcopy(weight.grad)


def extracting_fingerprints(layers, local_fingerprints, extracting_matrices, epsilon=0.5):
    max_score = -100000
    max_idx = 0
    bit_length = local_fingerprints[0].shape[0]
    weight = layers[0].weight.detach().cpu().numpy().flatten()
    for i in range(1, len(layers)):
        weight = np.append(weight, layers[i].weight.detach().cpu().numpy().flatten())
    for idx in range(len(local_fingerprints)):
        matrix = extracting_matrices[idx]
        result = np.dot(matrix, weight)
        result = np.multiply(result, local_fingerprints[idx])
        result[result > epsilon] = epsilon
        score = np.sum(result) / bit_length / epsilon
        if score > max_score:
            max_score = score
            max_idx = idx
    return max_score, max_idx


def get_embed_layer_weight_size(model, embed_layer_names):
    return sum(
        layer.weight.numel()
        for layer in get_diffusion_embed_layers(model, embed_layer_names)
    )


def apply_fingerprint_to_client_model(
    global_model,
    client_idx,
    local_fingerprints,
    extracting_matrices,
    embed_layer_names,
    lambda2,
    fingerprint_max_iters,
):
    client_model = copy.deepcopy(global_model)
    embed_layers = get_diffusion_embed_layers(client_model, embed_layer_names)
    fss, extract_idx = extracting_fingerprints(
        embed_layers, local_fingerprints, extracting_matrices
    )
    count = 0

    while (
        extract_idx != client_idx or (client_idx == extract_idx and fss < 0.85)
    ) and count <= fingerprint_max_iters:
        client_grad = calculate_local_grad(
            embed_layers,
            local_fingerprints[client_idx],
            extracting_matrices[client_idx],
        )
        client_grad = torch.mul(client_grad, -lambda2)

        weight_count = 0
        for embed_layer in embed_layers:
            weight_length = embed_layer.weight.numel()
            embed_layer.weight = torch.nn.Parameter(
                embed_layer.weight
                + client_grad[weight_count : weight_count + weight_length].view_as(
                    embed_layer.weight
                )
            )
            weight_count += weight_length
        count += 1
        fss, extract_idx = extracting_fingerprints(
            embed_layers, local_fingerprints, extracting_matrices
        )

    return client_model, fss, extract_idx
