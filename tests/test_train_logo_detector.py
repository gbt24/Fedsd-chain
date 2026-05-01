# -*- coding: UTF-8 -*-
import unittest

from train_logo_detector import build_eval_transform, build_positive_train_transform


class TrainLogoDetectorTransformTest(unittest.TestCase):
    def test_positive_train_transform_includes_augmentation(self):
        transform = build_positive_train_transform(32)
        transform_names = [type(step).__name__ for step in transform.transforms]

        self.assertIn("RandomAffine", transform_names)
        self.assertIn("ColorJitter", transform_names)
        self.assertEqual(transform_names[-1], "ToTensor")

    def test_eval_transform_stays_deterministic(self):
        transform = build_eval_transform(32)
        transform_names = [type(step).__name__ for step in transform.transforms]

        self.assertEqual(transform_names, ["Resize", "ToTensor"])


if __name__ == "__main__":
    unittest.main()
