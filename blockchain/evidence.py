# -*- coding: UTF-8 -*-
import json
import os

from blockchain.hash_utils import (
    sha256_array,
    sha256_file,
    sha256_json,
    sha256_model_state_dict,
)
from blockchain.local_chain import LocalBlockchain
from blockchain.merkle import get_merkle_root, merkle_leaf_hash


def build_run_id(args=None, model=None, dataset=None, seed=None):
    if args is not None:
        model = getattr(args, "model", model)
        dataset = getattr(args, "dataset", dataset)
        seed = getattr(args, "seed", seed)
        if getattr(args, "run_id", None):
            return args.run_id
    model = model or "run"
    dataset = dataset or "dataset"
    seed = 0 if seed is None else seed
    return f"{model}_{dataset}_seed{seed}"


class EvidenceLogger:
    def __init__(
        self,
        save_dir,
        run_id,
        chain_path=None,
        anchor_client=None,
        anchor_every_n_rounds=1,
    ):
        self.save_dir = save_dir
        self.run_id = run_id
        self.blockchain_dir = os.path.join(save_dir, "blockchain")
        self.chain_path = chain_path or os.path.join(self.blockchain_dir, "chain.jsonl")
        self.client_commitments_dir = os.path.join(save_dir, "client_commitments")
        self.chain = LocalBlockchain(self.chain_path)
        self.anchor_client = anchor_client
        self.anchor_every_n_rounds = max(1, int(anchor_every_n_rounds or 1))
        os.makedirs(self.blockchain_dir, exist_ok=True)
        os.makedirs(self.client_commitments_dir, exist_ok=True)

    def _anchor_block(self, block, payload_root_hash):
        if self.anchor_client is None:
            return None
        return self.anchor_client.anchor_block(block, payload_root_hash)

    def commit_trace_data(self, trace_dir):
        fingerprints_path = os.path.join(trace_dir, "fingerprints.npy")
        extract_matrices_path = os.path.join(trace_dir, "extract_matrices.npy")
        metadata_path = os.path.join(trace_dir, "metadata.json")

        commitment = {
            "event_type": "trace_data_commit",
            "run_id": self.run_id,
            "trace_dir": trace_dir,
            "fingerprints_hash": sha256_file(fingerprints_path),
            "extract_matrices_hash": sha256_file(extract_matrices_path),
            "metadata_hash": sha256_file(metadata_path),
        }
        commitment["trace_data_root_hash"] = sha256_json(
            {
                "fingerprints_hash": commitment["fingerprints_hash"],
                "extract_matrices_hash": commitment["extract_matrices_hash"],
                "metadata_hash": commitment["metadata_hash"],
            }
        )
        block = self.chain.add_block(
            "trace_data_commit",
            self.run_id,
            {
                "trace_dir": trace_dir,
                "trace_data_root_hash": commitment["trace_data_root_hash"],
                "fingerprints_hash": commitment["fingerprints_hash"],
                "extract_matrices_hash": commitment["extract_matrices_hash"],
                "metadata_hash": commitment["metadata_hash"],
            },
        )
        commitment["timestamp_utc"] = block["timestamp_utc"]
        commitment["block_hash"] = block["block_hash"]
        anchor_receipt = self._anchor_block(block, commitment["trace_data_root_hash"])
        if anchor_receipt is not None:
            commitment["anchor_receipt"] = anchor_receipt

        commitment_path = os.path.join(trace_dir, "trace_commitment.json")
        with open(commitment_path, "w") as f:
            json.dump(commitment, f, indent=2)
        return commitment

    def make_client_commitment(
        self, round_id, client_id, client_model_state, fingerprint, extract_matrix
    ):
        record = {
            "client_id": client_id,
            "round": round_id,
            "client_model_hash": sha256_model_state_dict(client_model_state),
            "fingerprint_commitment": sha256_array(fingerprint),
            "extract_matrix_commitment": sha256_array(extract_matrix),
        }
        record["leaf_hash"] = merkle_leaf_hash(record)
        return record

    def commit_client_distribution(self, round_id, client_records):
        leaf_hashes = [record["leaf_hash"] for record in client_records]
        merkle_root = get_merkle_root(leaf_hashes)
        commitment = {
            "event_type": "client_distribution_commit",
            "run_id": self.run_id,
            "round": round_id,
            "clients": client_records,
            "merkle_root": merkle_root,
        }
        block = self.chain.add_block(
            "client_distribution_commit",
            self.run_id,
            {
                "round": round_id,
                "client_merkle_root": merkle_root,
                "num_clients": len(client_records),
            },
        )
        commitment["timestamp_utc"] = block["timestamp_utc"]
        commitment["block_hash"] = block["block_hash"]
        if int(round_id) % self.anchor_every_n_rounds == 0:
            anchor_receipt = self._anchor_block(block, merkle_root)
            if anchor_receipt is not None:
                commitment["anchor_receipt"] = anchor_receipt

        commitment_path = os.path.join(
            self.client_commitments_dir, f"client_commitments_round_{round_id}.json"
        )
        with open(commitment_path, "w") as f:
            json.dump(commitment, f, indent=2)
        return commitment

    def commit_final_model(self, final_model_path):
        block = self.chain.add_block(
            "final_model_commit",
            self.run_id,
            {
                "model_path": final_model_path,
                "model_hash": sha256_file(final_model_path),
            },
        )
        commitment = {
            "event_type": "final_model_commit",
            "run_id": self.run_id,
            "model_path": final_model_path,
            "model_hash": block["payload"]["model_hash"],
            "timestamp_utc": block["timestamp_utc"],
            "block_hash": block["block_hash"],
        }
        anchor_receipt = self._anchor_block(block, commitment["model_hash"])
        if anchor_receipt is not None:
            commitment["anchor_receipt"] = anchor_receipt
        return commitment
