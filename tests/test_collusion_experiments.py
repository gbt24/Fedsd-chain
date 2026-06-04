# -*- coding: UTF-8 -*-
import unittest

import torch
from torch import nn

from collusion_experiments import (
    average_state_dicts,
    build_collusion_groups,
    build_experiment_report,
    parse_float_list,
    parse_int_list,
    save_experiment_outputs,
    score_model_case,
    summarize_score_cases,
    threshold_sensitivity,
    write_sensitivity_csv,
)


class CollusionExperimentsTest(unittest.TestCase):
    def test_score_model_case_returns_scores_and_colluder_metadata(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 1, bias=False)

        model = TinyModel()
        with torch.no_grad():
            model.proj.weight.copy_(torch.tensor([[0.5, 0.0]]))

        case = score_model_case(
            name="single_0",
            model=model,
            colluders=[0],
            local_fingerprints=[torch.tensor([1.0, -1.0]).numpy(), torch.tensor([-1.0, 1.0]).numpy()],
            extracting_matrices=[torch.eye(2).numpy(), torch.eye(2).numpy()],
            embed_layer_names="proj",
        )

        self.assertEqual(case["name"], "single_0")
        self.assertEqual(case["colluders"], [0])
        self.assertEqual(len(case["scores"]), 2)
        self.assertEqual(case["best_match_idx"], 0)

    def test_average_state_dicts_equal_weights(self):
        averaged = average_state_dicts(
            [
                {"weight": torch.tensor([1.0, 3.0]), "step": torch.tensor(1)},
                {"weight": torch.tensor([3.0, 5.0]), "step": torch.tensor(2)},
            ]
        )

        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.equal(averaged["step"], torch.tensor(1)))

    def test_average_state_dicts_weighted(self):
        averaged = average_state_dicts(
            [
                {"weight": torch.tensor([1.0, 3.0])},
                {"weight": torch.tensor([3.0, 5.0])},
            ],
            weights=[0.25, 0.75],
        )

        self.assertTrue(torch.equal(averaged["weight"], torch.tensor([2.5, 4.5])))

    def test_build_collusion_groups_limits_each_size(self):
        groups = build_collusion_groups(num_clients=5, group_sizes=[2, 3], max_groups_per_size=2)

        self.assertEqual(groups, [(0, 1), (0, 2), (0, 1, 2), (0, 1, 3)])

    def test_summarize_score_cases_reports_collusion_and_false_alarm_rates(self):
        summary = summarize_score_cases(
            [
                {"name": "single_0", "scores": [0.91, 0.20, 0.10], "colluders": [0]},
                {"name": "single_1", "scores": [0.20, 0.90, 0.10], "colluders": [1]},
                {"name": "collude_0_1", "scores": [0.76, 0.74, 0.10], "colluders": [0, 1]},
                {"name": "collude_1_2", "scores": [0.20, 0.51, 0.49], "colluders": [1, 2]},
            ],
            threshold=0.70,
            suspicious_threshold=0.60,
            gap_margin=0.05,
            top_k=2,
        )

        self.assertEqual(summary["num_cases"], 4)
        self.assertEqual(summary["num_single_cases"], 2)
        self.assertEqual(summary["num_collusion_cases"], 2)
        self.assertAlmostEqual(summary["single_owner_accuracy"], 1.0)
        self.assertAlmostEqual(summary["false_collusion_flag_rate"], 0.0)
        self.assertAlmostEqual(summary["collusion_flag_rate"], 1.0)
        self.assertAlmostEqual(summary["top_k_any_colluder_hit_rate"], 1.0)
        self.assertAlmostEqual(summary["top_k_all_colluders_hit_rate"], 1.0)

    def test_threshold_sensitivity_evaluates_parameter_grid(self):
        rows = threshold_sensitivity(
            cases=[
                {"name": "single_0", "scores": [0.91, 0.20], "colluders": [0]},
                {"name": "collude_0_1", "scores": [0.76, 0.74], "colluders": [0, 1]},
            ],
            thresholds=[0.70],
            suspicious_thresholds=[0.60, 0.80],
            gap_margins=[0.03, 0.05],
            top_k=2,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["threshold"], 0.70)
        self.assertIn("collusion_flag_rate", rows[0])
        self.assertIn("false_collusion_flag_rate", rows[0])

    def test_parse_list_helpers(self):
        self.assertEqual(parse_int_list("2,3,5"), [2, 3, 5])
        self.assertEqual(parse_float_list("0.03,0.05"), [0.03, 0.05])

    def test_build_experiment_report_contains_three_requested_experiments(self):
        report = build_experiment_report(
            single_cases=[
                {"name": "single_0", "scores": [0.91, 0.20], "colluders": [0]},
            ],
            collusion_cases=[
                {"name": "equal_m2_0_1", "scores": [0.76, 0.74], "colluders": [0, 1]},
            ],
            threshold=0.70,
            suspicious_threshold=0.60,
            gap_margin=0.05,
            top_k=2,
            sensitivity_thresholds=[0.70],
            sensitivity_suspicious_thresholds=[0.60],
            sensitivity_gap_margins=[0.05],
        )

        self.assertIn("single_false_alarm", report["experiments"])
        self.assertIn("equal_averaging_collusion", report["experiments"])
        self.assertIn("threshold_sensitivity", report["experiments"])
        self.assertEqual(report["experiments"]["single_false_alarm"]["num_single_cases"], 1)
        self.assertEqual(
            report["experiments"]["equal_averaging_collusion"]["num_collusion_cases"],
            1,
        )

    def test_write_sensitivity_csv(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sensitivity.csv")
            write_sensitivity_csv(
                [
                    {
                        "threshold": 0.70,
                        "suspicious_threshold": 0.60,
                        "gap_margin": 0.05,
                        "num_cases": 2,
                        "single_owner_accuracy": 1.0,
                        "false_collusion_flag_rate": 0.0,
                        "collusion_flag_rate": 1.0,
                        "top_k_any_colluder_hit_rate": 1.0,
                        "top_k_all_colluders_hit_rate": 1.0,
                    }
                ],
                path,
            )

            with open(path, "r") as f:
                content = f.read()
            self.assertIn("threshold,suspicious_threshold,gap_margin", content)
            self.assertIn("0.7,0.6,0.05", content)

    def test_save_experiment_outputs_writes_summary_and_sensitivity_csv(self):
        import json
        import os
        import tempfile

        report = {
            "experiments": {
                "threshold_sensitivity": [
                    {
                        "threshold": 0.70,
                        "suspicious_threshold": 0.60,
                        "gap_margin": 0.05,
                        "num_cases": 2,
                        "single_owner_accuracy": 1.0,
                        "false_collusion_flag_rate": 0.0,
                        "collusion_flag_rate": 1.0,
                        "top_k_any_colluder_hit_rate": 1.0,
                        "top_k_all_colluders_hit_rate": 1.0,
                    }
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = save_experiment_outputs(report, tmpdir)

            self.assertTrue(os.path.exists(paths["summary_json"]))
            self.assertTrue(os.path.exists(paths["sensitivity_csv"]))
            with open(paths["summary_json"], "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, report)

    def test_script_entrypoint_comes_after_helper_definitions(self):
        import inspect
        import collusion_experiments

        source = inspect.getsource(collusion_experiments)
        self.assertLess(source.index("def _mean"), source.index('if __name__ == "__main__"'))


if __name__ == "__main__":
    unittest.main()
