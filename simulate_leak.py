# -*- coding: UTF-8 -*-
"""
Simulate client model leak for testing owner identification.

This script creates a "leaked" model by embedding a specific client's fingerprint
into a global model, simulating the scenario where a client's model is leaked.

Usage:
    python simulate_leak.py \
        --checkpoint ./result/simpleunet_cifar10_stage2/model_final.pth \
        --trace_dir ./result/simpleunet_cifar10_stage2/trace_data \
        --client_idx 3 \
        --output ./leaked_model_client_3.pth
"""

import argparse
import copy
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from save_trace_data import load_trace_data
from watermark.fingerprint_diffusion import (
    calculate_local_grad,
    extracting_fingerprints,
    get_diffusion_embed_layers,
)


def simulate_leak(
    model,
    client_idx,
    local_fingerprints,
    extracting_matrices,
    embed_layer_names,
    max_iters=10,
    epsilon=0.5,
    lambda_factor=0.01,
    device="cpu",
):
    """
    Embed client fingerprint into model to simulate a leaked model.

    Args:
        model: Global model to embed fingerprint into
        client_idx: Index of client to simulate leak for
        local_fingerprints: List of fingerprint vectors
        extracting_matrices: List of extracting matrices
        embed_layer_names: Names of layers to embed fingerprint
        max_iters: Maximum iterations for embedding
        epsilon: Epsilon for hinge-like loss
        lambda_factor: Learning rate for gradient update
        device: Device to use

    Returns:
        model: Model with embedded fingerprint
        fss: Final FSS score
        iterations: Number of iterations used
    """
    model = copy.deepcopy(model)
    model.to(device)

    client_fingerprint = local_fingerprints[client_idx]
    embed_layers = get_diffusion_embed_layers(model, embed_layer_names)

    fss, extract_idx = extracting_fingerprints(
        embed_layers, local_fingerprints, extracting_matrices, epsilon=epsilon
    )

    iterations = 0
    while (extract_idx != client_idx or fss < 0.85) and iterations < max_iters:
        grad_update = calculate_local_grad(
            embed_layers, client_fingerprint, extracting_matrices[client_idx], epsilon
        )
        grad_update = torch.mul(grad_update, -lambda_factor)
        grad_update = grad_update.to(device)

        weight_count = 0
        for embed_layer in embed_layers:
            weight_length = embed_layer.weight.numel()
            embed_layer.weight = torch.nn.Parameter(
                embed_layer.weight
                + grad_update[weight_count : weight_count + weight_length].view_as(
                    embed_layer.weight
                )
            )
            weight_count += weight_length

        fss, extract_idx = extracting_fingerprints(
            embed_layers, local_fingerprints, extracting_matrices, epsilon=epsilon
        )
        iterations += 1

    model.cpu()
    return model, fss, iterations


def main():
    parser = argparse.ArgumentParser(
        description="Simulate client model leak for testing owner identification"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to global model checkpoint",
    )
    parser.add_argument(
        "--trace_dir",
        type=str,
        required=True,
        help="Path to trace data directory",
    )
    parser.add_argument(
        "--client_idx",
        type=int,
        required=True,
        help="Client index to simulate leak for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for leaked model (default: ./leaked_model_client_{idx}.pth)",
    )
    parser.add_argument(
        "--args_file",
        type=str,
        default=None,
        help="Path to args.json (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID (default: 0)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=10,
        help="Maximum iterations for fingerprint embedding",
    )
    parser.add_argument(
        "--lambda_factor",
        type=float,
        default=0.01,
        help="Learning rate for gradient update",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Simulating Client Model Leak")
    print("=" * 60)
    print(f"Global model: {args.checkpoint}")
    print(f"Trace data: {args.trace_dir}")
    print(f"Client index: {args.client_idx}")

    if args.args_file is None:
        args_file = os.path.join(os.path.dirname(args.checkpoint), "args.txt")
    else:
        args_file = args.args_file

    print(f"\nLoading model arguments from {args_file}...")
    with open(args_file, "r") as f:
        args_dict = json.load(f)

    class TrainArgs:
        pass

    train_args = TrainArgs()
    for k, v in args_dict.items():
        setattr(train_args, k, v)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    train_args.device = device

    print("\nLoading trace data...")
    local_fingerprints, extracting_matrices, metadata = load_trace_data(args.trace_dir)
    embed_layer_names = metadata["embed_layer_names"]

    print(f"  - Number of clients: {metadata['num_clients']}")
    print(f"  - Fingerprint length: {metadata['lfp_length']}")
    print(f"  - Embed layers: {embed_layer_names}")

    if args.client_idx >= len(local_fingerprints):
        print(
            f"Error: client_idx ({args.client_idx}) >= num_clients ({len(local_fingerprints)})"
        )
        sys.exit(1)

    print("\nLoading model...")
    from utils.models import get_model

    model = get_model(train_args)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print("Embedding fingerprint...")
    leaked_model, fss, iterations = simulate_leak(
        model,
        args.client_idx,
        local_fingerprints,
        extracting_matrices,
        embed_layer_names,
        max_iters=args.max_iters,
        lambda_factor=args.lambda_factor,
        device=str(device),
    )

    print(f"  - Iterations used: {iterations}")
    print(f"  - Final FSS score: {fss:.4f}")

    if args.output is None:
        args.output = f"./leaked_model_client_{args.client_idx}.pth"

    print(f"\nSaving leaked model to {args.output}...")
    torch.save(leaked_model.state_dict(), args.output)

    # Copy args file to leak model directory for identify_owner.py
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args_output_path = os.path.join(output_dir if output_dir else ".", "args.txt")
    if os.path.exists(args_file):
        shutil.copy(args_file, args_output_path)
        print(f"Copied args file to {args_output_path}")

    print("\n" + "=" * 60)
    print("Leak Simulation Complete")
    print("=" * 60)
    print(f"Client index: {args.client_idx}")
    print(f"FSS score: {fss:.4f}")
    print(f"Iterations: {iterations}")
    print(f"Saved to: {args.output}")
    print("\nYou can now use identify_owner.py to identify the owner.")


if __name__ == "__main__":
    main()
