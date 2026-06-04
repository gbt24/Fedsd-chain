# -*- coding: UTF-8 -*-
import json
import os

from blockchain.hash_utils import sha256_file, sha256_json
from blockchain.local_chain import LocalBlockchain
from blockchain.merkle import get_merkle_proof, merkle_leaf_hash, verify_merkle_proof


def verify_chain_integrity(chain_path):
    if not chain_path or not os.path.exists(chain_path):
        return {"verified": False, "reason": f"chain file not found: {chain_path}"}
    return LocalBlockchain(chain_path).verify_chain()


def _payload_root_for_anchor(block):
    payload = block.get("payload", {})
    event_type = block.get("event_type")
    if event_type == "trace_data_commit":
        return payload.get("trace_data_root_hash")
    if event_type == "client_distribution_commit":
        return payload.get("client_merkle_root")
    if event_type == "final_model_commit":
        return payload.get("model_hash")
    return None


def verify_anchor_commitments(chain_path, anchor_client):
    if anchor_client is None:
        return {
            "verified": False,
            "reason": "anchor client not provided",
            "num_verified_anchors": 0,
            "anchors": [],
        }
    if not chain_path or not os.path.exists(chain_path):
        return {
            "verified": False,
            "reason": f"chain file not found: {chain_path}",
            "num_verified_anchors": 0,
            "anchors": [],
        }

    blockchain = LocalBlockchain(chain_path)
    anchors = []
    for block in blockchain._load_blocks():
        payload_root_hash = _payload_root_for_anchor(block)
        if payload_root_hash is None:
            continue
        result = anchor_client.verify_anchor(block, payload_root_hash)
        anchors.append(result)
        if not result.get("verified"):
            return {
                "verified": False,
                "reason": result.get("reason", "anchor verification failed"),
                "num_verified_anchors": len([item for item in anchors if item.get("verified")]),
                "anchors": anchors,
            }

    return {
        "verified": True,
        "reason": f"verified {len(anchors)} anchors",
        "num_verified_anchors": len(anchors),
        "anchors": anchors,
    }


def verify_trace_data_commitment(trace_dir, chain_path, run_id=None):
    commitment_path = os.path.join(trace_dir, "trace_commitment.json")
    if not os.path.exists(commitment_path):
        return {"verified": False, "reason": "trace_commitment.json not found"}

    with open(commitment_path, "r") as f:
        commitment = json.load(f)

    current_hashes = {
        "fingerprints_hash": sha256_file(os.path.join(trace_dir, "fingerprints.npy")),
        "extract_matrices_hash": sha256_file(
            os.path.join(trace_dir, "extract_matrices.npy")
        ),
        "metadata_hash": sha256_file(os.path.join(trace_dir, "metadata.json")),
    }
    current_root = sha256_json(current_hashes)

    for key, value in current_hashes.items():
        if commitment.get(key) != value:
            return {"verified": False, "reason": f"{key} mismatch"}
    if commitment.get("trace_data_root_hash") != current_root:
        return {"verified": False, "reason": "trace_data_root_hash mismatch"}

    blockchain = LocalBlockchain(chain_path)
    blocks = blockchain.find_blocks("trace_data_commit", run_id or commitment.get("run_id"))
    if not blocks:
        return {"verified": False, "reason": "trace_data_commit block not found"}

    matched_block = blocks[-1]
    if matched_block["payload"].get("trace_data_root_hash") != current_root:
        return {"verified": False, "reason": "chain trace_data_root_hash mismatch"}

    return {
        "verified": True,
        "reason": "trace_data hashes match chain commitment",
        "matched_block_hash": matched_block["block_hash"],
        "timestamp_utc": matched_block["timestamp_utc"],
    }


def verify_client_commitment(commitments_dir, chain_path, client_id, round_id=None):
    if not os.path.isdir(commitments_dir):
        return {"verified": False, "reason": f"commitments dir not found: {commitments_dir}"}

    commitment_files = sorted(
        [
            os.path.join(commitments_dir, name)
            for name in os.listdir(commitments_dir)
            if name.startswith("client_commitments_round_") and name.endswith(".json")
        ]
    )
    if not commitment_files:
        return {"verified": False, "reason": "no client commitment files found"}

    if round_id is not None:
        expected_name = f"client_commitments_round_{round_id}.json"
        commitment_files = [path for path in commitment_files if path.endswith(expected_name)]
        if not commitment_files:
            return {"verified": False, "reason": f"round {round_id} commitment not found"}

    blockchain = LocalBlockchain(chain_path)

    for commitment_path in reversed(commitment_files):
        with open(commitment_path, "r") as f:
            commitment = json.load(f)

        clients = commitment.get("clients", [])
        leaf_hashes = [client["leaf_hash"] for client in clients]
        for index, client_record in enumerate(clients):
            if client_record.get("client_id") != client_id:
                continue

            recomputed_leaf = merkle_leaf_hash(
                {
                    "client_id": client_record["client_id"],
                    "round": client_record["round"],
                    "client_model_hash": client_record["client_model_hash"],
                    "fingerprint_commitment": client_record["fingerprint_commitment"],
                    "extract_matrix_commitment": client_record["extract_matrix_commitment"],
                }
            )
            if recomputed_leaf != client_record.get("leaf_hash"):
                return {"verified": False, "reason": "client leaf hash mismatch"}

            proof = get_merkle_proof(leaf_hashes, index)
            root = commitment.get("merkle_root")
            if not verify_merkle_proof(client_record["leaf_hash"], proof, root):
                return {"verified": False, "reason": "merkle proof verification failed"}

            blocks = blockchain.find_blocks(
                "client_distribution_commit", commitment.get("run_id")
            )
            for block in reversed(blocks):
                if block["payload"].get("round") != commitment.get("round"):
                    continue
                if block["payload"].get("client_merkle_root") != root:
                    continue
                return {
                    "verified": True,
                    "reason": "client commitment matches committed merkle root",
                    "matched_block_hash": block["block_hash"],
                    "timestamp_utc": block["timestamp_utc"],
                    "matched_round": commitment.get("round"),
                    "matched_merkle_root": root,
                }

            return {"verified": False, "reason": "matching chain block not found"}

    return {"verified": False, "reason": f"client {client_id} not found in commitments"}


def generate_evidence_report(
    leaked_model_path,
    trace_dir,
    chain_path,
    commitments_dir,
    best_match_idx,
    confidence,
    all_scores,
    threshold=0.85,
    run_id=None,
    anchor_client=None,
): 
    chain_result = verify_chain_integrity(chain_path)
    trace_result = verify_trace_data_commitment(trace_dir, chain_path, run_id=run_id)
    client_result = verify_client_commitment(commitments_dir, chain_path, best_match_idx)
    anchor_result = verify_anchor_commitments(chain_path, anchor_client)
    top_scores = sorted(enumerate(all_scores), key=lambda item: item[1], reverse=True)[:5]

    report = {
        "leaked_model": leaked_model_path,
        "leaked_model_hash": sha256_file(leaked_model_path),
        "best_match_idx": int(best_match_idx),
        "confidence": float(confidence),
        "threshold": threshold,
        "trace_data_verified": trace_result["verified"],
        "chain_integrity_verified": chain_result["verified"],
        "client_commitment_verified": client_result["verified"],
        "anchor_verified": anchor_result["verified"],
        "num_verified_anchors": anchor_result.get("num_verified_anchors", 0),
        "anchors": anchor_result.get("anchors", []),
        "matched_round": client_result.get("matched_round"),
        "matched_merkle_root": client_result.get("matched_merkle_root"),
        "top_5": [[int(index), float(score)] for index, score in top_scores],
        "trace_data_reason": trace_result.get("reason"),
        "chain_reason": chain_result.get("reason"),
        "client_commitment_reason": client_result.get("reason"),
        "anchor_reason": anchor_result.get("reason"),
    }
    return report
