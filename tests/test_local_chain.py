# -*- coding: UTF-8 -*-
import json
import os
import tempfile
import unittest

from blockchain.local_chain import LocalBlockchain


class LocalBlockchainTest(unittest.TestCase):
    def test_chain_append_and_verify(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "chain.jsonl")
            blockchain = LocalBlockchain(chain_path)

            first = blockchain.add_block("trace_data_commit", "run-1", {"x": 1})
            second = blockchain.add_block("final_model_commit", "run-1", {"y": 2})

            self.assertEqual(first["block_index"], 0)
            self.assertEqual(second["block_index"], 1)
            verification = blockchain.verify_chain()
            self.assertTrue(verification["verified"])

    def test_chain_tamper_detected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chain_path = os.path.join(tmpdir, "chain.jsonl")
            blockchain = LocalBlockchain(chain_path)
            blockchain.add_block("trace_data_commit", "run-1", {"x": 1})

            with open(chain_path, "r") as f:
                blocks = [json.loads(line) for line in f if line.strip()]

            blocks[0]["payload"]["x"] = 99

            with open(chain_path, "w") as f:
                for block in blocks:
                    f.write(json.dumps(block) + "\n")

            verification = blockchain.verify_chain()
            self.assertFalse(verification["verified"])


if __name__ == "__main__":
    unittest.main()
