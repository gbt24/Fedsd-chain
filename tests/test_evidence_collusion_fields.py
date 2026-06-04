# -*- coding: UTF-8 -*-
import os
import tempfile
import unittest

import numpy as np

from blockchain.evidence import EvidenceLogger
from blockchain.verify_evidence import generate_evidence_report
from save_trace_data import save_trace_data


class EvidenceCollusionFieldsTest(unittest.TestCase):
    def test_report_includes_fingerprint_analysis_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "blockchain", "chain.jsonl")
            save_trace_data(
                save_dir=tmpdir,
                local_fingerprints=[np.array([1, -1, 1]), np.array([-1, 1, -1])],
                extracting_matrices=[np.eye(3), np.eye(3)],
                embed_layer_names="layer",
                num_clients=2,
                lfp_length=3,
                enable_blockchain=True,
                run_id="run-1",
                chain_path=chain_path,
            )

            logger = EvidenceLogger(tmpdir, "run-1", chain_path=chain_path)
            client_records = [
                logger.make_client_commitment(
                    round_id=1,
                    client_id=0,
                    client_model_state={"weight": [1.0, 0.0]},
                    fingerprint=np.array([1, -1, 1]),
                    extract_matrix=np.eye(3),
                )
            ]
            logger.commit_client_distribution(round_id=1, client_records=client_records)

            fingerprint_analysis = {
                "best_match_idx": 0,
                "confidence": 0.62,
                "threshold": 0.85,
                "suspicious_threshold": 0.65,
                "score_gap": 0.04,
                "top_k": [[0, 0.62], [1, 0.58]],
                "suspicious_clients": [],
                "possible_collusion": True,
                "attribution_status": "low_confidence",
                "collusion_reason": "best score below attribution threshold",
            }
            leaked_model_path = os.path.join(tmpdir, "leaked.pth")
            with open(leaked_model_path, "wb") as f:
                f.write(b"leaked model")

            report = generate_evidence_report(
                leaked_model_path=leaked_model_path,
                trace_dir=os.path.join(tmpdir, "trace_data"),
                chain_path=chain_path,
                commitments_dir=os.path.join(tmpdir, "client_commitments"),
                best_match_idx=0,
                confidence=0.62,
                all_scores=[0.62, 0.58],
                threshold=0.85,
                run_id="run-1",
                fingerprint_analysis=fingerprint_analysis,
            )

            self.assertEqual(report["fingerprint_analysis"], fingerprint_analysis)
            self.assertTrue(report["possible_collusion"])
            self.assertEqual(report["attribution_status"], "low_confidence")
            self.assertEqual(
                report["collusion_reason"],
                "best score below attribution threshold",
            )


if __name__ == "__main__":
    unittest.main()
