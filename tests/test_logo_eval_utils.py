# -*- coding: UTF-8 -*-
import unittest

from logo_eval_utils import (
    build_logo_eval_report,
    build_normal_class_ids,
    compute_logo_metrics,
)


class LogoEvalUtilsTest(unittest.TestCase):
    def test_compute_logo_metrics_counts_trigger_and_normal_rates(self):
        metrics = compute_logo_metrics(
            trigger_scores=[0.9, 0.6, 0.4, 0.8],
            normal_scores=[0.1, 0.3, 0.7, 0.2],
            threshold=0.5,
        )

        self.assertEqual(metrics["trigger_total"], 4)
        self.assertEqual(metrics["trigger_detected"], 3)
        self.assertAlmostEqual(metrics["trigger_success_rate"], 0.75)
        self.assertEqual(metrics["normal_total"], 4)
        self.assertEqual(metrics["normal_false_positives"], 1)
        self.assertAlmostEqual(metrics["normal_false_positive_rate"], 0.25)

    def test_build_logo_eval_report_preserves_core_fields(self):
        report = build_logo_eval_report(
            checkpoint="./result/model_final.pth",
            detector_checkpoint="./result/logo_detector/model_best.pth",
            trigger_scores=[0.9, 0.8],
            normal_scores=[0.2, 0.1],
            threshold=0.5,
            num_inference_steps=100,
            trigger_class=10,
        )

        self.assertEqual(report["checkpoint"], "./result/model_final.pth")
        self.assertEqual(
            report["detector_checkpoint"], "./result/logo_detector/model_best.pth"
        )
        self.assertEqual(report["num_inference_steps"], 100)
        self.assertEqual(report["trigger_class"], 10)
        self.assertAlmostEqual(report["trigger_success_rate"], 1.0)
        self.assertAlmostEqual(report["normal_false_positive_rate"], 0.0)

    def test_build_normal_class_ids_excludes_trigger_class(self):
        labels = build_normal_class_ids(num_classes=4, trigger_class=3, num_samples=7)

        self.assertEqual(labels, [0, 1, 2, 0, 1, 2, 0])


if __name__ == "__main__":
    unittest.main()
