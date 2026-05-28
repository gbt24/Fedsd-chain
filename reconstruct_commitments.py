# -*- coding: UTF-8 -*-
import copy, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from blockchain.evidence import EvidenceLogger, build_run_id
from save_trace_data import load_trace_data
from utils.models import get_model
from watermark.fingerprint_diffusion import (
    calculate_local_grad,
    extracting_fingerprints,
    get_diffusion_embed_layers,
)

RESULT_DIR = "result/ablation_no_wm_25c"
CHECKPOINT = os.path.join(RESULT_DIR, "model_final.pth")
ARGS_FILE = os.path.join(RESULT_DIR, "args.txt")
TRACE_DIR = os.path.join(RESULT_DIR, "trace_data")
CHAIN_PATH = os.path.join(RESULT_DIR, "blockchain", "chain.jsonl")
COMMIT_DIR = os.path.join(RESULT_DIR, "client_commitments")

# load args & model
with open(ARGS_FILE) as f:
    args_dict = json.load(f)
class TrainArgs: pass
train_args = TrainArgs()
for k, v in args_dict.items():
    setattr(train_args, k, v)
train_args.device = "cpu"

model = get_model(train_args)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

# load trace data
local_fingerprints, extracting_matrices, metadata = load_trace_data(TRACE_DIR)
embed_layer_names = metadata["embed_layer_names"]
num_clients = metadata["num_clients"]
epsilon = 0.5
lambda_factor = 0.01
max_iters = 10

# init evidence logger (loads existing chain)
os.makedirs(COMMIT_DIR, exist_ok=True)
run_id = build_run_id(args=train_args)
evidence_logger = EvidenceLogger(
    save_dir=RESULT_DIR,
    run_id=run_id,
    chain_path=CHAIN_PATH,
)

# embed fingerprint + compute commitment for each client
round_id = 150
client_records = []
for client_idx in range(num_clients):
    client_model = copy.deepcopy(model)
    embed_layers = get_diffusion_embed_layers(client_model, embed_layer_names)

    fss, extract_idx = extracting_fingerprints(
        embed_layers, local_fingerprints, extracting_matrices, epsilon=epsilon
    )
    count = 0
    while (extract_idx != client_idx or fss < 0.85) and count <= max_iters:
        grad = calculate_local_grad(
            embed_layers,
            local_fingerprints[client_idx],
            extracting_matrices[client_idx],
        )
        grad = torch.mul(grad, -lambda_factor)
        wc = 0
        for layer in embed_layers:
            wl = layer.weight.numel()
            layer.weight = torch.nn.Parameter(
                layer.weight + grad[wc:wc+wl].view_as(layer.weight)
            )
            wc += wl
        count += 1
        fss, extract_idx = extracting_fingerprints(
            embed_layers, local_fingerprints, extracting_matrices, epsilon=epsilon
        )

    record = evidence_logger.make_client_commitment(
        round_id=round_id,
        client_id=client_idx,
        client_model_state=client_model.state_dict(),
        fingerprint=local_fingerprints[client_idx],
        extract_matrix=extracting_matrices[client_idx],
    )
    client_records.append(record)
    print(f"Client {client_idx}: FSS={fss:.4f}, extracted={extract_idx}, iters={count}")

# commit distribution
commitment = evidence_logger.commit_client_distribution(
    round_id=round_id, client_records=client_records
)
print(f"\nCommitted: round={round_id}, merkle_root={commitment['merkle_root']}")
print(f"Chain blocks: {len(evidence_logger.blocks)}")
print("Done.")
