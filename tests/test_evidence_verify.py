# -*- coding: UTF-8 -*-
import json
import os
import tempfile
import unittest

import numpy as np

from blockchain.evidence import EvidenceLogger
from blockchain.verify_evidence import (
    generate_evidence_report,
    verify_client_commitment,
    verify_trace_data_commitment,
)
from save_trace_data import save_trace_data


class EvidenceVerificationTest(unittest.TestCase):
    def test_trace_commitment_verification_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fingerprints = [np.array([1, -1, 1]), np.array([-1, 1, -1])]
            matrices = [np.eye(3), np.eye(3)]
            chain_path = os.path.join(tmpdir, "blockchain", "chain.jsonl")

            save_trace_data(
                save_dir=tmpdir,
                local_fingerprints=fingerprints,
                extracting_matrices=matrices,
                embed_layer_names="mid_block.attention.proj",
                num_clients=2,
                lfp_length=3,
                enable_blockchain=True,
                run_id="run-1",
                chain_path=chain_path,
            )

            result = verify_trace_data_commitment(
                trace_dir=os.path.join(tmpdir, "trace_data"),
                chain_path=chain_path,
                run_id="run-1",
            )

            self.assertTrue(result["verified"])

    def test_client_commitment_verification_succeeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EvidenceLogger(save_dir=tmpdir, run_id="run-1")
            client_records = []
            for client_id in range(2):
                state_dict = {"weight": [float(client_id), 1.0]}
                record = logger.make_client_commitment(
                    round_id=10,
                    client_id=client_id,
                    client_model_state=state_dict,
                    fingerprint=np.array([1, -1, 1]),
                    extract_matrix=np.eye(3),
                )
                client_records.append(record)

            logger.commit_client_distribution(round_id=10, client_records=client_records)
            verification = verify_client_commitment(
                commitments_dir=os.path.join(tmpdir, "client_commitments"),
                chain_path=os.path.join(tmpdir, "blockchain", "chain.jsonl"),
                client_id=1,
            )

            self.assertTrue(verification["verified"])

    def test_evidence_report_includes_verification_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leaked_model_path = os.path.join(tmpdir, "leaked_model.pth")
            with open(leaked_model_path, "wb") as f:
                f.write(b"leaked-model")

            fingerprints = [np.array([1, -1, 1]), np.array([-1, 1, -1])]
            matrices = [np.eye(3), np.eye(3)]
            chain_path = os.path.join(tmpdir, "blockchain", "chain.jsonl")
            save_trace_data(
                save_dir=tmpdir,
                local_fingerprints=fingerprints,
                extracting_matrices=matrices,
                embed_layer_names="mid_block.attention.proj",
                num_clients=2,
                lfp_length=3,
                enable_blockchain=True,
                run_id="run-1",
                chain_path=chain_path,
            )

            logger = EvidenceLogger(save_dir=tmpdir, run_id="run-1", chain_path=chain_path)
            records = [
                logger.make_client_commitment(
                    round_id=10,
                    client_id=0,
                    client_model_state={"weight": [0.0]},
                    fingerprint=np.array([1, -1, 1]),
                    extract_matrix=np.eye(3),
                ),
                logger.make_client_commitment(
                    round_id=10,
                    client_id=1,
                    client_model_state={"weight": [1.0]},
                    fingerprint=np.array([-1, 1, -1]),
                    extract_matrix=np.eye(3),
                ),
            ]
            logger.commit_client_distribution(round_id=10, client_records=records)

            report = generate_evidence_report(
                leaked_model_path=leaked_model_path,
                trace_dir=os.path.join(tmpdir, "trace_data"),
                chain_path=chain_path,
                commitments_dir=os.path.join(tmpdir, "client_commitments"),
                best_match_idx=1,
                confidence=0.91,
                all_scores=[0.2, 0.91],
                threshold=0.85,
                run_id="run-1",
            )

            self.assertTrue(report["trace_data_verified"])
            self.assertTrue(report["chain_integrity_verified"])
            self.assertTrue(report["client_commitment_verified"])
            self.assertEqual(report["best_match_idx"], 1)
            self.assertEqual(report["top_5"][0][0], 1)

    def test_evidence_report_is_json_serializable_with_numpy_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            leaked_model_path = os.path.join(tmpdir, "leaked_model.pth")
            with open(leaked_model_path, "wb") as f:
                f.write(b"leaked-model")

            fingerprints = [np.array([1, -1, 1]), np.array([-1, 1, -1])]
            matrices = [np.eye(3), np.eye(3)]
            chain_path = os.path.join(tmpdir, "blockchain", "chain.jsonl")
            save_trace_data(
                save_dir=tmpdir,
                local_fingerprints=fingerprints,
                extracting_matrices=matrices,
                embed_layer_names="mid_block.attention.proj",
                num_clients=2,
                lfp_length=3,
                enable_blockchain=True,
                run_id="run-1",
                chain_path=chain_path,
            )

            logger = EvidenceLogger(save_dir=tmpdir, run_id="run-1", chain_path=chain_path)
            records = [
                logger.make_client_commitment(
                    round_id=10,
                    client_id=0,
                    client_model_state={"weight": [0.0]},
                    fingerprint=np.array([1, -1, 1]),
                    extract_matrix=np.eye(3),
                ),
                logger.make_client_commitment(
                    round_id=10,
                    client_id=1,
                    client_model_state={"weight": [1.0]},
                    fingerprint=np.array([-1, 1, -1]),
                    extract_matrix=np.eye(3),
                ),
            ]
            logger.commit_client_distribution(round_id=10, client_records=records)

            report = generate_evidence_report(
                leaked_model_path=leaked_model_path,
                trace_dir=os.path.join(tmpdir, "trace_data"),
                chain_path=chain_path,
                commitments_dir=os.path.join(tmpdir, "client_commitments"),
                best_match_idx=np.int64(1),
                confidence=np.float32(0.91),
                all_scores=np.array([0.2, 0.91], dtype=np.float32),
                threshold=0.85,
                run_id="run-1",
            )

            json.dumps(report)


if __name__ == "__main__":
    unittest.main()
