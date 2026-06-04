# -*- coding: UTF-8 -*-
"""
Save traceability data for fingerprint-based owner identification.

This script saves the fingerprints and extracting matrices after training,
which are required for simulating leaks and identifying model owners.

Usage:
    python save_trace_data.py --save_dir ./result/simpleunet_cifar10_stage2/
"""

import argparse
import json
import os

import numpy as np

from blockchain.evidence import EvidenceLogger


def save_trace_data(
    save_dir,
    local_fingerprints,
    extracting_matrices,
    embed_layer_names,
    num_clients,
    lfp_length,
    enable_blockchain=False,
    run_id=None,
    chain_path=None,
    anchor_client=None,
    anchor_every_n_rounds=1,
):
    """
    Save traceability data for owner identification.

    Args:
        save_dir: Directory to save trace data
        local_fingerprints: List of fingerprint vectors for each client
        extracting_matrices: List of extracting matrices for each client
        embed_layer_names: Names of layers where fingerprints are embedded
        num_clients: Number of clients
        lfp_length: Length of fingerprint vector
    """
    trace_dir = os.path.join(save_dir, "trace_data")
    os.makedirs(trace_dir, exist_ok=True)

    fingerprints_array = np.array(local_fingerprints)
    np.save(os.path.join(trace_dir, "fingerprints.npy"), fingerprints_array)

    extract_matrices_array = np.array(extracting_matrices)
    np.save(os.path.join(trace_dir, "extract_matrices.npy"), extract_matrices_array)

    metadata = {
        "num_clients": num_clients,
        "lfp_length": lfp_length,
        "embed_layer_names": embed_layer_names,
        "fingerprints_shape": fingerprints_array.shape,
        "extract_matrices_shape": extract_matrices_array.shape,
    }
    with open(os.path.join(trace_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Trace data saved to {trace_dir}")
    print(f"  - fingerprints.npy: {fingerprints_array.shape}")
    print(f"  - extract_matrices.npy: {extract_matrices_array.shape}")
    print(f"  - metadata.json")

    if enable_blockchain:
        evidence_logger = EvidenceLogger(
            save_dir=save_dir,
            run_id=run_id or "trace_data_run",
            chain_path=chain_path,
            anchor_client=anchor_client,
            anchor_every_n_rounds=anchor_every_n_rounds,
        )
        commitment = evidence_logger.commit_trace_data(trace_dir)
        print(f"  - trace_commitment.json")
        print(f"Blockchain commitment saved: {commitment['block_hash']}")


def load_trace_data(trace_dir):
    """
    Load traceability data for owner identification.

    Args:
        trace_dir: Directory containing trace data

    Returns:
        local_fingerprints: List of fingerprint vectors
        extracting_matrices: List of extracting matrices
        metadata: Dictionary of metadata
    """
    fingerprints_path = os.path.join(trace_dir, "fingerprints.npy")
    extract_matrices_path = os.path.join(trace_dir, "extract_matrices.npy")
    metadata_path = os.path.join(trace_dir, "metadata.json")

    if not os.path.exists(fingerprints_path):
        raise FileNotFoundError(f"fingerprints.npy not found in {trace_dir}")
    if not os.path.exists(extract_matrices_path):
        raise FileNotFoundError(f"extract_matrices.npy not found in {trace_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {trace_dir}")

    local_fingerprints = np.load(fingerprints_path)
    extracting_matrices = np.load(extract_matrices_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    local_fingerprints = [local_fingerprints[i] for i in range(len(local_fingerprints))]
    extracting_matrices = [
        extracting_matrices[i] for i in range(len(extracting_matrices))
    ]

    return local_fingerprints, extracting_matrices, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Save traceability data for owner identification"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Training save directory containing fingerprints data",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Number of clients (read from args.txt if not provided)",
    )
    parser.add_argument(
        "--lfp_length",
        type=int,
        default=128,
        help="Fingerprint length (default: 128)",
    )
    parser.add_argument(
        "--embed_layer_names",
        type=str,
        default="mid_block.attention.proj",
        help="Names of embedding layers",
    )
    parser.add_argument(
        "--enable_blockchain",
        action="store_true",
        help="Write trace_data commitment into the local blockchain",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run identifier for blockchain evidence",
    )
    parser.add_argument(
        "--chain_path",
        type=str,
        default=None,
        help="Override blockchain chain.jsonl path",
    )
    parser.add_argument(
        "--enable_anchor",
        action="store_true",
        help="Anchor trace_data commitment to an external evidence chain",
    )
    parser.add_argument(
        "--anchor_mode",
        type=str,
        default="mock",
        choices=["mock", "evm"],
        help="external anchoring backend",
    )
    parser.add_argument("--anchor_path", type=str, default=None, help="mock anchor jsonl path")
    parser.add_argument("--anchor_rpc", type=str, default=None, help="EVM RPC URL")
    parser.add_argument("--anchor_contract", type=str, default=None, help="EVM contract address")
    parser.add_argument("--anchor_abi", type=str, default=None, help="EVM ABI JSON path")
    parser.add_argument(
        "--anchor_receipts_dir",
        type=str,
        default=None,
        help="directory for saved EVM anchor transaction receipts",
    )
    parser.add_argument(
        "--anchor_every_n_rounds",
        type=int,
        default=1,
        help="anchor client distribution commitments every N rounds",
    )

    args = parser.parse_args()

    args_file = os.path.join(args.save_dir, "args.txt")
    if args.num_clients is None:
        if os.path.exists(args_file):
            with open(args_file, "r") as f:
                args_dict = json.load(f)
            args.num_clients = args_dict.get("num_clients", 10)
        else:
            args.num_clients = 10

    from watermark.fingerprint_diffusion import (
        generate_fingerprints,
        generate_extracting_matrices,
        get_diffusion_embed_layers_length,
    )
    from utils.models import get_model
    from utils.utils import load_args
    import torch

    print("Generating fingerprints...")
    local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)

    print("Loading model to determine weight size...")
    model_args = load_args() if os.path.exists("args.txt") else args
    model = get_model(model_args)
    weight_size = get_diffusion_embed_layers_length(model, args.embed_layer_names)

    print("Generating extracting matrices...")
    extracting_matrices = generate_extracting_matrices(
        weight_size, args.lfp_length, args.num_clients
    )

    anchor_client = None
    if args.enable_blockchain:
        from blockchain.anchor_factory import create_anchor_client_from_args

        anchor_client = create_anchor_client_from_args(args)

    save_trace_data(
        args.save_dir,
        local_fingerprints,
        extracting_matrices,
        args.embed_layer_names,
        args.num_clients,
        args.lfp_length,
        enable_blockchain=args.enable_blockchain,
        run_id=args.run_id,
        chain_path=args.chain_path,
        anchor_client=anchor_client,
        anchor_every_n_rounds=args.anchor_every_n_rounds,
    )

    print("\nDone! You can now use simulate_leak.py and identify_owner.py.")


if __name__ == "__main__":
    main()
