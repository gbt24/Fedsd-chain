# -*- coding: UTF-8 -*-
import unittest
from unittest.mock import patch

from eval_fid import load_fid_real_dataset


class EvalFidDatasetTest(unittest.TestCase):
    def test_load_fid_real_dataset_uses_eval_split_and_limit(self):
        class Args:
            dataset = "celeba64"
            image_size = 64
            max_test_samples = 5000
            seed = 3

        train_dataset = object()
        test_dataset = object()
        with patch("eval_fid.get_full_dataset", return_value=(train_dataset, test_dataset)) as mocked:
            result = load_fid_real_dataset(Args())

        self.assertIs(result, test_dataset)
        mocked.assert_called_once_with(
            "celeba64",
            img_size=(64, 64),
            max_train_samples=None,
            max_test_samples=5000,
            seed=3,
        )


if __name__ == "__main__":
    unittest.main()
