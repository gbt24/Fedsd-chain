# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest

from blockchain.evidence import EvidenceLogger
from blockchain.merkle import merkle_leaf_hash
from blockchain.mock_anchor import MockAnchorClient


class AnchorVerifyTest(unittest.TestCase):
    def _write_trace_files(self, trace_dir):
        os.makedirs(trace_dir, exist_ok=True)
        for name, content in (
            ("fingerprints.npy", b"fingerprints"),
            ("extract_matrices.npy", b"extract"),
            ("metadata.json", b"{}"),
        ):
            with open(os.path.join(trace_dir, name), "wb") as f:
                f.write(content)

    def _client_records(self):
        records = []
        for client_id, client_hash in ((0, "c" * 64), (1, "d" * 64)):
            record = {
                "client_id": client_id,
                "round": 1,
                "client_model_hash": client_hash,
                "fingerprint_commitment": str(client_id) * 64,
                "extract_matrix_commitment": str(client_id + 2) * 64,
            }
            record["leaf_hash"] = merkle_leaf_hash(record)
            records.append(record)
        return records

    def test_verify_evidence_module_import_does_not_require_numpy(self):
        from blockchain.verify_evidence import verify_anchor_commitments

        self.assertTrue(callable(verify_anchor_commitments))

    def test_verify_anchor_commitments_accepts_matching_anchors(self):
        from blockchain.verify_evidence import verify_anchor_commitments

        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MockAnchorClient(os.path.join(tmpdir, "anchors.jsonl"))
            logger = EvidenceLogger(tmpdir, "run-1", anchor_client=anchor)
            trace_dir = os.path.join(tmpdir, "trace_data")
            self._write_trace_files(trace_dir)
            logger.commit_trace_data(trace_dir)
            logger.commit_client_distribution(
                1,
                self._client_records(),
            )
            model_path = os.path.join(tmpdir, "model_final.pth")
            with open(model_path, "wb") as f:
                f.write(b"model")
            logger.commit_final_model(model_path)

            result = verify_anchor_commitments(
                os.path.join(tmpdir, "blockchain", "chain.jsonl"), anchor
            )

        self.assertTrue(result["verified"])
        self.assertEqual(result["num_verified_anchors"], 3)

    def test_verify_anchor_commitments_rejects_missing_anchor(self):
        from blockchain.verify_evidence import verify_anchor_commitments

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EvidenceLogger(tmpdir, "run-1")
            model_path = os.path.join(tmpdir, "model_final.pth")
            with open(model_path, "wb") as f:
                f.write(b"model")
            logger.commit_final_model(model_path)
            anchor = MockAnchorClient(os.path.join(tmpdir, "anchors.jsonl"))

            result = verify_anchor_commitments(
                os.path.join(tmpdir, "blockchain", "chain.jsonl"), anchor
            )

        self.assertFalse(result["verified"])
        self.assertIn("anchor record not found", result["reason"])

    def test_generate_evidence_report_includes_anchor_status(self):
        from blockchain.verify_evidence import generate_evidence_report

        with tempfile.TemporaryDirectory() as tmpdir:
            anchor = MockAnchorClient(os.path.join(tmpdir, "anchors.jsonl"))
            logger = EvidenceLogger(tmpdir, "run-1", anchor_client=anchor)
            trace_dir = os.path.join(tmpdir, "trace_data")
            self._write_trace_files(trace_dir)
            logger.commit_trace_data(trace_dir)
            logger.commit_client_distribution(
                1,
                self._client_records(),
            )
            leaked_model_path = os.path.join(tmpdir, "leaked.pth")
            with open(leaked_model_path, "wb") as f:
                f.write(b"leaked")

            report = generate_evidence_report(
                leaked_model_path=leaked_model_path,
                trace_dir=trace_dir,
                chain_path=os.path.join(tmpdir, "blockchain", "chain.jsonl"),
                commitments_dir=os.path.join(tmpdir, "client_commitments"),
                best_match_idx=1,
                confidence=0.9,
                all_scores=[0.1, 0.9],
                run_id="run-1",
                anchor_client=anchor,
            )

        self.assertTrue(report["anchor_verified"])
        self.assertEqual(report["num_verified_anchors"], 2)
        self.assertIn("anchors", report)


if __name__ == "__main__":
    unittest.main()
