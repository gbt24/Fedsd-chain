# -*- coding: UTF-8 -*-
import json
import os
from datetime import datetime, timezone

from blockchain.hash_utils import sha256_json


GENESIS_PREVIOUS_HASH = "0" * 64


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _compute_block_hash(block):
    payload = dict(block)
    payload.pop("block_hash", None)
    return sha256_json(payload)


class LocalBlockchain:
    def __init__(self, chain_path):
        self.chain_path = chain_path
        self.chain_dir = os.path.dirname(chain_path)
        self.latest_block_path = os.path.join(self.chain_dir, "latest_block.json")
        if self.chain_dir:
            os.makedirs(self.chain_dir, exist_ok=True)

    def _load_blocks(self):
        if not os.path.exists(self.chain_path):
            return []
        with open(self.chain_path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_latest_block(self):
        blocks = self._load_blocks()
        return blocks[-1] if blocks else None

    def add_block(self, event_type, run_id, payload):
        blocks = self._load_blocks()
        latest_block = blocks[-1] if blocks else None
        block = {
            "schema_version": "1.0",
            "block_index": len(blocks),
            "timestamp_utc": _utc_now(),
            "event_type": event_type,
            "run_id": run_id,
            "payload": payload,
            "previous_block_hash": latest_block["block_hash"] if latest_block else GENESIS_PREVIOUS_HASH,
        }
        block["block_hash"] = _compute_block_hash(block)

        with open(self.chain_path, "a") as f:
            f.write(json.dumps(block, ensure_ascii=True, sort_keys=True) + "\n")

        with open(self.latest_block_path, "w") as f:
            json.dump(block, f, indent=2)

        return block

    def verify_chain(self):
        blocks = self._load_blocks()
        previous_hash = GENESIS_PREVIOUS_HASH

        for index, block in enumerate(blocks):
            if block.get("block_index") != index:
                return {
                    "verified": False,
                    "reason": f"block_index mismatch at index {index}",
                }
            if block.get("previous_block_hash") != previous_hash:
                return {
                    "verified": False,
                    "reason": f"previous_block_hash mismatch at index {index}",
                }
            if block.get("block_hash") != _compute_block_hash(block):
                return {
                    "verified": False,
                    "reason": f"block_hash mismatch at index {index}",
                }
            previous_hash = block["block_hash"]

        return {
            "verified": True,
            "reason": f"verified {len(blocks)} blocks",
            "num_blocks": len(blocks),
            "latest_block_hash": previous_hash if blocks else None,
        }

    def find_blocks(self, event_type=None, run_id=None):
        blocks = self._load_blocks()
        results = []
        for block in blocks:
            if event_type is not None and block.get("event_type") != event_type:
                continue
            if run_id is not None and block.get("run_id") != run_id:
                continue
            results.append(block)
        return results
