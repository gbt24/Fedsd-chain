# -*- coding: UTF-8 -*-
import unittest

from summarize_journal_results import mean_std, format_mean_std, summarize_group


class JournalSummaryTest(unittest.TestCase):
    def test_mean_std_uses_sample_standard_deviation(self):
        mean, std = mean_std([1.0, 2.0, 3.0])

        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(std, 1.0)

    def test_format_mean_std_formats_latex_pm(self):
        self.assertEqual(format_mean_std(12.345, 0.678, digits=2), "12.35 $\\pm$ 0.68")

    def test_summarize_group_ignores_missing_none_values(self):
        runs = [
            {"fid": 10.0, "trace_accuracy": 1.0},
            {"fid": 12.0, "trace_accuracy": None},
            {"fid": 14.0, "trace_accuracy": 0.9},
        ]

        summary = summarize_group(runs, ["fid", "trace_accuracy"])

        self.assertAlmostEqual(summary["fid"]["mean"], 12.0)
        self.assertAlmostEqual(summary["fid"]["std"], 2.0)
        self.assertAlmostEqual(summary["trace_accuracy"]["mean"], 0.95)


if __name__ == "__main__":
    unittest.main()
