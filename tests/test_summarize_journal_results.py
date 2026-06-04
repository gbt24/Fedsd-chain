# -*- coding: UTF-8 -*-
import json
import os
import tempfile
import unittest

from summarize_journal_results import (
    extract_run_metrics,
    mean_std,
    format_mean_std,
    summarize_group,
)


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

    def test_extract_run_metrics_reads_fid_total_from_eval_fid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_dir = os.path.join(tmpdir, "eval")
            os.makedirs(eval_dir)
            with open(os.path.join(eval_dir, "fid.json"), "w") as f:
                json.dump({"fid_total": 21.38}, f)

            metrics = extract_run_metrics(tmpdir)

        self.assertEqual(metrics["fid"], 21.38)


if __name__ == "__main__":
    unittest.main()
