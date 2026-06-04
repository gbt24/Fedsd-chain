# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest


class AnchorClientTest(unittest.TestCase):
    def _sample_block(self):
        return {
            "schema_version": "1.0",
            "block_index": 2,
            "timestamp_utc": "2026-06-03T00:00:00Z",
            "event_type": "trace_data_commit",
            "run_id": "SimpleUNet_cifar10_seed1",
            "payload": {"trace_data_root_hash": "a" * 64},
            "previous_block_hash": "b" * 64,
            "block_hash": "c" * 64,
        }

    def test_blockchain_local_chain_import_does_not_require_numpy(self):
        from blockchain.local_chain import GENESIS_PREVIOUS_HASH

        self.assertEqual(len(GENESIS_PREVIOUS_HASH), 64)

    def test_build_anchor_record_hashes_run_and_event_metadata(self):
        from blockchain.anchor_client import build_anchor_record

        record = build_anchor_record(self._sample_block(), "d" * 64)

        self.assertNotIn("SimpleUNet_cifar10_seed1", record.values())
        self.assertNotIn("trace_data_commit", record.values())
        self.assertEqual(record["local_block_index"], 2)
        self.assertEqual(record["local_block_hash"], "c" * 64)
        self.assertEqual(record["payload_root_hash"], "d" * 64)
        self.assertEqual(record["previous_local_block_hash"], "b" * 64)
        self.assertEqual(len(record["anchor_id"]), 64)
        self.assertEqual(len(record["run_id_hash"]), 64)
        self.assertEqual(len(record["event_type_hash"]), 64)

    def test_build_anchor_record_is_deterministic(self):
        from blockchain.anchor_client import build_anchor_record

        first = build_anchor_record(self._sample_block(), "d" * 64)
        second = build_anchor_record(self._sample_block(), "d" * 64)

        self.assertEqual(first, second)

    def test_mock_anchor_verifies_matching_record(self):
        from blockchain.mock_anchor import MockAnchorClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockAnchorClient(os.path.join(tmpdir, "anchors.jsonl"))
            block = self._sample_block()

            receipt = client.anchor_block(block, "d" * 64)
            verification = client.verify_anchor(block, "d" * 64)

        self.assertTrue(receipt["anchored"])
        self.assertTrue(verification["verified"])
        self.assertEqual(receipt["anchor_id"], verification["anchor_id"])

    def test_mock_anchor_rejects_duplicate_anchor_with_different_payload(self):
        from blockchain.mock_anchor import MockAnchorClient

        with tempfile.TemporaryDirectory() as tmpdir:
            client = MockAnchorClient(os.path.join(tmpdir, "anchors.jsonl"))
            block = self._sample_block()
            client.anchor_block(block, "d" * 64)

            with self.assertRaises(ValueError):
                changed = dict(block)
                changed["previous_block_hash"] = "e" * 64
                client.anchor_block(changed, "d" * 64)


if __name__ == "__main__":
    unittest.main()
