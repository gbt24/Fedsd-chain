# -*- coding: UTF-8 -*-
import hashlib
import json


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_anchor_record(block, payload_root_hash):
    run_id_hash = _sha256_text(str(block["run_id"]))
    event_type_hash = _sha256_text(str(block["event_type"]))
    record = {
        "schema_version": "1.0",
        "run_id_hash": run_id_hash,
        "event_type_hash": event_type_hash,
        "local_block_index": int(block["block_index"]),
        "local_block_hash": block["block_hash"],
        "payload_root_hash": payload_root_hash,
        "previous_local_block_hash": block["previous_block_hash"],
    }
    record["anchor_id"] = _sha256_text(
        _canonical_json(
            {
                "run_id_hash": run_id_hash,
                "event_type_hash": event_type_hash,
                "local_block_index": record["local_block_index"],
                "local_block_hash": record["local_block_hash"],
            }
        )
    )
    return record


class AnchorClient:
    def anchor_block(self, block, payload_root_hash):
        raise NotImplementedError

    def verify_anchor(self, block, payload_root_hash):
        raise NotImplementedError
