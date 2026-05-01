# -*- coding: UTF-8 -*-
import unittest

from recovery_utils import (
    find_latest_checkpoint,
    trace_metadata_matches_args,
)


class RecoveryUtilsTest(unittest.TestCase):
    def test_find_latest_checkpoint_picks_highest_epoch(self):
        checkpoint = find_latest_checkpoint(
            [
                "checkpoint_epoch_110.pth",
                "checkpoint_epoch_150.pth",
                "checkpoint_epoch_140.pth",
            ]
        )

        self.assertEqual(checkpoint, "checkpoint_epoch_150.pth")

    def test_trace_metadata_matches_args_checks_core_dimensions(self):
        metadata = {
            "num_clients": 25,
            "lfp_length": 128,
            "embed_layer_names": "mid_block.attention.proj",
            "extract_matrices_shape": [25, 128, 65536],
        }
        args = {
            "num_clients": 25,
            "lfp_length": 128,
            "embed_layer_names": "mid_block.attention.proj",
        }

        self.assertTrue(trace_metadata_matches_args(metadata, args, 65536))
        self.assertFalse(trace_metadata_matches_args(metadata, args, 123))


if __name__ == "__main__":
    unittest.main()
