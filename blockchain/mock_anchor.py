# -*- coding: UTF-8 -*-
import json
import os

from blockchain.anchor_client import AnchorClient, build_anchor_record


class MockAnchorClient(AnchorClient):
    def __init__(self, anchor_path):
        self.anchor_path = anchor_path
        anchor_dir = os.path.dirname(anchor_path)
        if anchor_dir:
            os.makedirs(anchor_dir, exist_ok=True)

    def _load_records(self):
        if not os.path.exists(self.anchor_path):
            return []
        with open(self.anchor_path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]

    def anchor_block(self, block, payload_root_hash):
        record = build_anchor_record(block, payload_root_hash)
        records = self._load_records()
        for existing in records:
            if existing["anchor_id"] == record["anchor_id"]:
                if existing["record"] != record:
                    raise ValueError("anchor_id already exists with different content")
                return dict(existing, anchored=True)

        receipt = {
            "anchor_mode": "mock",
            "anchored": True,
            "anchor_id": record["anchor_id"],
            "record": record,
        }
        with open(self.anchor_path, "a") as f:
            f.write(json.dumps(receipt, ensure_ascii=True, sort_keys=True) + "\n")
        return receipt

    def verify_anchor(self, block, payload_root_hash):
        record = build_anchor_record(block, payload_root_hash)
        for existing in self._load_records():
            if existing["anchor_id"] != record["anchor_id"]:
                continue
            if existing.get("record") != record:
                return {
                    "verified": False,
                    "reason": "anchor record mismatch",
                    "anchor_id": record["anchor_id"],
                }
            return {
                "verified": True,
                "reason": "mock anchor record matches",
                "anchor_id": record["anchor_id"],
                "anchor_mode": "mock",
            }
        return {
            "verified": False,
            "reason": "anchor record not found",
            "anchor_id": record["anchor_id"],
            "anchor_mode": "mock",
        }
