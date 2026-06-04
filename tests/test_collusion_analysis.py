# -*- coding: UTF-8 -*-
import unittest

from identify_owner import analyze_fingerprint_scores


class CollusionAnalysisTest(unittest.TestCase):
    def test_high_confidence_single_owner_is_not_flagged(self):
        analysis = analyze_fingerprint_scores(
            [0.12, 0.94, 0.20, 0.18],
            threshold=0.85,
            top_k=3,
            suspicious_threshold=0.65,
            gap_margin=0.05,
        )

        self.assertEqual(analysis["attribution_status"], "single_owner")
        self.assertFalse(analysis["possible_collusion"])
        self.assertEqual(analysis["best_match_idx"], 1)
        self.assertAlmostEqual(analysis["confidence"], 0.94)
        self.assertAlmostEqual(analysis["score_gap"], 0.74)
        self.assertEqual(analysis["top_k"], [[1, 0.94], [2, 0.20], [3, 0.18]])
        self.assertEqual(analysis["suspicious_clients"], [1])
        self.assertEqual(analysis["collusion_reason"], "single high-confidence peak")

    def test_low_confidence_match_is_rejected_as_possible_collusion_or_unattributed(self):
        analysis = analyze_fingerprint_scores(
            [0.62, 0.58, 0.11, 0.10],
            threshold=0.85,
            top_k=3,
            suspicious_threshold=0.65,
            gap_margin=0.05,
        )

        self.assertEqual(analysis["attribution_status"], "low_confidence")
        self.assertTrue(analysis["possible_collusion"])
        self.assertEqual(analysis["best_match_idx"], 0)
        self.assertAlmostEqual(analysis["confidence"], 0.62)
        self.assertAlmostEqual(analysis["score_gap"], 0.04)
        self.assertEqual(analysis["suspicious_clients"], [])
        self.assertEqual(
            analysis["collusion_reason"],
            "best score below attribution threshold",
        )

    def test_multiple_high_scores_are_flagged_as_multi_peak(self):
        analysis = analyze_fingerprint_scores(
            [0.89, 0.87, 0.21, 0.15],
            threshold=0.85,
            top_k=4,
            suspicious_threshold=0.65,
            gap_margin=0.05,
        )

        self.assertEqual(analysis["attribution_status"], "multi_peak")
        self.assertTrue(analysis["possible_collusion"])
        self.assertEqual(analysis["best_match_idx"], 0)
        self.assertAlmostEqual(analysis["confidence"], 0.89)
        self.assertAlmostEqual(analysis["score_gap"], 0.02)
        self.assertEqual(analysis["suspicious_clients"], [0, 1])
        self.assertEqual(
            analysis["collusion_reason"],
            "multiple high-scoring clients within gap margin",
        )

    def test_empty_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_fingerprint_scores([])


if __name__ == "__main__":
    unittest.main()
