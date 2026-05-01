# -*- coding: UTF-8 -*-
import unittest

from blockchain.merkle import (
    get_merkle_proof,
    get_merkle_root,
    merkle_leaf_hash,
    verify_merkle_proof,
)


class MerkleTest(unittest.TestCase):
    def test_merkle_proof_valid(self):
        records = [
            {"client_id": 0, "client_model_hash": "a"},
            {"client_id": 1, "client_model_hash": "b"},
            {"client_id": 2, "client_model_hash": "c"},
        ]
        leaf_hashes = [merkle_leaf_hash(record) for record in records]

        root = get_merkle_root(leaf_hashes)
        proof = get_merkle_proof(leaf_hashes, 1)

        self.assertTrue(verify_merkle_proof(leaf_hashes[1], proof, root))

    def test_merkle_proof_invalid_when_leaf_changes(self):
        records = [
            {"client_id": 0, "client_model_hash": "a"},
            {"client_id": 1, "client_model_hash": "b"},
        ]
        leaf_hashes = [merkle_leaf_hash(record) for record in records]
        root = get_merkle_root(leaf_hashes)
        proof = get_merkle_proof(leaf_hashes, 0)
        tampered_leaf = merkle_leaf_hash({"client_id": 0, "client_model_hash": "x"})

        self.assertFalse(verify_merkle_proof(tampered_leaf, proof, root))


if __name__ == "__main__":
    unittest.main()
