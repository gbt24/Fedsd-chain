# -*- coding: UTF-8 -*-
import argparse
import os
import shutil

import numpy as np
import torch

from blockchain.evidence import EvidenceLogger
from eval_logo_watermark import load_simple_unet
from logo_eval_utils import infer_block_out_channels
from recovery_utils import (
    apply_fingerprint_to_client_model,
    find_latest_checkpoint,
    get_embed_layer_weight_size,
    load_json,
    trace_metadata_matches_args,
)
from save_trace_data import save_trace_data
from utils.simple_diffusion import SimpleDiffusion


def save_final_samples(model, args, device, save_path):
    from torchvision.utils import save_image

    model.eval()
    model.to(device)
    diffusion = SimpleDiffusion(
        num_timesteps=args.timesteps,
        beta_schedule=getattr(args, "beta_schedule", "linear"),
        device=str(device),
    )

    normal_labels = torch.randint(0, args.num_classes, (4,), device=device)
    normal_images = diffusion.sample(
        model,
        batch_size=4,
        class_labels=normal_labels,
        num_inference_steps=getattr(args, "num_inference_steps", 1000),
        seed=args.seed,
        device=str(device),
    )

    trigger_class = getattr(args, "trigger_class", args.num_classes)
    trigger_labels = torch.full((4,), trigger_class, dtype=torch.long, device=device)
    trigger_images = diffusion.sample(
        model,
        batch_size=4,
        class_labels=trigger_labels,
        num_inference_steps=getattr(args, "num_inference_steps", 1000),
        seed=args.seed if args.seed is None else args.seed + 1000,
        device=str(device),
    )

    save_image(torch.cat([normal_images, trigger_images], dim=0), save_path, nrow=4, normalize=True)
    model.cpu()


def copy_trace_data_from_source(source_trace_dir, target_trace_dir):
    os.makedirs(target_trace_dir, exist_ok=True)
    for file_name in ["fingerprints.npy", "extract_matrices.npy", "metadata.json"]:
        shutil.copy2(
            os.path.join(source_trace_dir, file_name),
            os.path.join(target_trace_dir, file_name),
        )


def main():
    parser = argparse.ArgumentParser(description="Recover incomplete stage2 result artifacts")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--trace_source_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=-1)
    args = parser.parse_args()

    result_dir = args.result_dir
    result_args = load_json(os.path.join(result_dir, "args.txt"))
    checkpoint_name = find_latest_checkpoint(os.listdir(result_dir))
    if checkpoint_name is None:
        raise FileNotFoundError("No checkpoint_epoch_*.pth found in result_dir")

    checkpoint_path = os.path.join(result_dir, checkpoint_name)
    model_dir = result_dir
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    model, run_args = load_simple_unet(model_dir, checkpoint_path, device)

    final_model_path = os.path.join(result_dir, "model_final.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    torch.save(state_dict, final_model_path)

    final_samples_path = os.path.join(result_dir, "final_samples.png")
    save_final_samples(model, run_args, device, final_samples_path)

    weight_size = get_embed_layer_weight_size(model, result_args["embed_layer_names"])
    source_metadata = load_json(os.path.join(args.trace_source_dir, "metadata.json"))
    if not trace_metadata_matches_args(source_metadata, result_args, weight_size):
        raise ValueError("trace_source_dir metadata does not match target run")

    target_trace_dir = os.path.join(result_dir, "trace_data")
    copy_trace_data_from_source(args.trace_source_dir, target_trace_dir)
    local_fingerprints = np.load(os.path.join(target_trace_dir, "fingerprints.npy"))
    extracting_matrices = np.load(os.path.join(target_trace_dir, "extract_matrices.npy"))
    local_fingerprints = [local_fingerprints[i] for i in range(len(local_fingerprints))]
    extracting_matrices = [
        extracting_matrices[i] for i in range(len(extracting_matrices))
    ]

    blockchain_dir = os.path.join(result_dir, "blockchain")
    client_commitments_dir = os.path.join(result_dir, "client_commitments")
    os.makedirs(blockchain_dir, exist_ok=True)
    os.makedirs(client_commitments_dir, exist_ok=True)

    evidence_logger = EvidenceLogger(
        save_dir=result_dir,
        run_id=result_args.get("run_id") or os.path.basename(result_dir),
        chain_path=result_args.get("chain_path"),
    )

    client_records = []
    for client_idx in range(result_args["num_clients"]):
        client_model, _, _ = apply_fingerprint_to_client_model(
            model,
            client_idx,
            local_fingerprints,
            extracting_matrices,
            result_args["embed_layer_names"],
            result_args["lambda2"],
            result_args["fingerprint_max_iters"],
        )
        client_records.append(
            evidence_logger.make_client_commitment(
                round_id=result_args["epochs"],
                client_id=client_idx,
                client_model_state=client_model.state_dict(),
                fingerprint=local_fingerprints[client_idx],
                extract_matrix=extracting_matrices[client_idx],
            )
        )

    evidence_logger.commit_client_distribution(
        round_id=result_args["epochs"], client_records=client_records
    )
    evidence_logger.commit_final_model(final_model_path)
    save_trace_data(
        result_dir,
        local_fingerprints,
        extracting_matrices,
        result_args["embed_layer_names"],
        result_args["num_clients"],
        result_args["lfp_length"],
        enable_blockchain=True,
        run_id=result_args.get("run_id") or os.path.basename(result_dir),
        chain_path=result_args.get("chain_path"),
    )

    print(f"Recovered final model: {final_model_path}")
    print(f"Recovered final samples: {final_samples_path}")
    print(f"Recovered trace data: {target_trace_dir}")
    print(f"Recovered blockchain: {blockchain_dir}")
    print(f"Recovered client commitments: {client_commitments_dir}")


if __name__ == "__main__":
    main()
