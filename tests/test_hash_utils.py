# -*- coding: UTF-8 -*-
import unittest

from blockchain.hash_utils import sha256_json, sha256_model_state_dict


class HashUtilsTest(unittest.TestCase):
    def test_same_model_state_dict_has_same_hash(self):
        state_dict = {
            "b": [[1.0, 2.0]],
            "a": [3.0],
        }

        first_hash = sha256_model_state_dict(state_dict)
        second_hash = sha256_model_state_dict(state_dict)

        self.assertEqual(first_hash, second_hash)

    def test_modified_model_state_dict_changes_hash(self):
        state_dict = {"weight": [1.0, 2.0, 3.0]}
        changed_state_dict = {"weight": [1.0, 2.0, 4.0]}

        original_hash = sha256_model_state_dict(state_dict)
        changed_hash = sha256_model_state_dict(changed_state_dict)

        self.assertNotEqual(original_hash, changed_hash)

    def test_json_hash_is_key_order_stable(self):
        left = {"b": 2, "a": 1, "nested": {"z": 3, "y": 4}}
        right = {"nested": {"y": 4, "z": 3}, "a": 1, "b": 2}

        self.assertEqual(sha256_json(left), sha256_json(right))


if __name__ == "__main__":
    unittest.main()
