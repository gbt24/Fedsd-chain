# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest

from blockchain.evidence import EvidenceLogger


class RecordingAnchorClient:
    def __init__(self):
        self.calls = []

    def anchor_block(self, block, payload_root_hash):
        self.calls.append((block, payload_root_hash))
        return {"anchored": True, "anchor_id": block["block_hash"]}


class EvidenceAnchorIntegrationTest(unittest.TestCase):
    def _write_trace_files(self, trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
        for name, content in (
            ("fingerprints.npy", b"fingerprints"),
            ("extract_matrices.npy", b"extract"),
            ("metadata.json", b"{}"),
        ):
            with open(os.path.join(trace_dir, name), "wb") as f:
                f.write(content)

    def test_trace_data_commit_anchors_trace_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = os.path.join(tmpdir, "trace_data")
            self._write_trace_files(trace_dir)
            anchor = RecordingAnchorClient()
            logger = EvidenceLogger(tmpdir, "run-1", anchor_client=anchor)

            commitment = logger.commit_trace_data(trace_dir)

        self.assertEqual(len(anchor.calls), 1)
        block, payload_root_hash = anchor.calls[0]
        self.assertEqual(block["event_type"], "trace_data_commit")
        self.assertEqual(payload_root_hash, commitment["trace_data_root_hash"])

    def test_client_distribution_anchor_respects_round_interval(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = RecordingAnchorClient()
            logger = EvidenceLogger(
                tmpdir, "run-1", anchor_client=anchor, anchor_every_n_rounds=2
            )
            client_records = [
                {"client_id": 1, "round": 1, "leaf_hash": "a" * 64},
                {"client_id": 2, "round": 1, "leaf_hash": "b" * 64},
            ]

            logger.commit_client_distribution(1, client_records)
            round_two = logger.commit_client_distribution(2, client_records)

        self.assertEqual(len(anchor.calls), 1)
        block, payload_root_hash = anchor.calls[0]
        self.assertEqual(block["event_type"], "client_distribution_commit")
        self.assertEqual(payload_root_hash, round_two["merkle_root"])

    def test_final_model_commit_anchors_model_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model_final.pth")
            with open(model_path, "wb") as f:
                f.write(b"model")
            anchor = RecordingAnchorClient()
            logger = EvidenceLogger(tmpdir, "run-1", anchor_client=anchor)

            commitment = logger.commit_final_model(model_path)

        self.assertEqual(len(anchor.calls), 1)
        block, payload_root_hash = anchor.calls[0]
        self.assertEqual(block["event_type"], "final_model_commit")
        self.assertEqual(payload_root_hash, commitment["model_hash"])

    def test_logger_without_anchor_preserves_existing_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model_final.pth")
            with open(model_path, "wb") as f:
                f.write(b"model")
            logger = EvidenceLogger(tmpdir, "run-1")

            commitment = logger.commit_final_model(model_path)

        self.assertIn("model_hash", commitment)
        self.assertIn("block_hash", commitment)


if __name__ == "__main__":
    unittest.main()
