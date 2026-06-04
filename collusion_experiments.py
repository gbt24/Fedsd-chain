# -*- coding: UTF-8 -*-
"""Utilities for lightweight collusion-diagnostic experiments."""

import argparse
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from identify_owner import analyze_fingerprint_scores, identify_owner


def score_model_case(
    name,
    model,
    colluders,
    local_fingerprints,
    extracting_matrices,
    embed_layer_names,
    epsilon=0.5,
):
    best_match_idx, confidence, all_scores = identify_owner(
        model,
        local_fingerprints,
        extracting_matrices,
        embed_layer_names,
        epsilon=epsilon,
    )
    return {
        "name": name,
        "colluders": [int(idx) for idx in colluders],
        "best_match_idx": int(best_match_idx),
        "confidence": float(confidence),
        "scores": [float(score) for score in all_scores],
    }


def average_state_dicts(state_dicts, weights=None):
    if not state_dicts:
        raise ValueError("state_dicts must contain at least one state dict")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    if len(weights) != len(state_dicts):
        raise ValueError("weights length must match state_dicts length")

    total_weight = float(sum(weights))
    if total_weight == 0:
        raise ValueError("weights must not sum to zero")
    normalized_weights = [float(weight) / total_weight for weight in weights]

    averaged = {}
    for key in state_dicts[0]:
        first_value = state_dicts[0][key]
        if torch.is_tensor(first_value) and torch.is_floating_point(first_value):
            value = torch.zeros_like(first_value)
            for state_dict, weight in zip(state_dicts, normalized_weights):
                value = value + state_dict[key].to(value.device) * weight
            averaged[key] = value
        else:
            averaged[key] = first_value.clone() if torch.is_tensor(first_value) else first_value
    return averaged


def build_collusion_groups(num_clients, group_sizes, max_groups_per_size=None):
    groups = []
    for group_size in group_sizes:
        count = 0
        for group in itertools.combinations(range(num_clients), int(group_size)):
            groups.append(group)
            count += 1
            if max_groups_per_size is not None and count >= max_groups_per_size:
                break
    return groups


def _top_k_clients(analysis):
    return [client_idx for client_idx, _ in analysis["top_k"]]


def summarize_score_cases(
    cases,
    threshold=0.85,
    suspicious_threshold=None,
    gap_margin=0.05,
    top_k=5,
):
    results = []
    single_cases = []
    collusion_cases = []
    for case in cases:
        colluders = [int(idx) for idx in case["colluders"]]
        analysis = analyze_fingerprint_scores(
            case["scores"],
            threshold=threshold,
            top_k=top_k,
            suspicious_threshold=suspicious_threshold,
            gap_margin=gap_margin,
        )
        top_clients = _top_k_clients(analysis)
        result = {
            "name": case.get("name", "case"),
            "colluders": colluders,
            "analysis": analysis,
            "top_k_any_colluder_hit": any(idx in top_clients for idx in colluders),
            "top_k_all_colluders_hit": all(idx in top_clients for idx in colluders),
        }
        results.append(result)
        if len(colluders) == 1:
            single_cases.append(result)
        else:
            collusion_cases.append(result)

    single_owner_accuracy = _mean(
        result["analysis"]["best_match_idx"] == result["colluders"][0]
        for result in single_cases
    )
    false_collusion_flag_rate = _mean(
        result["analysis"]["possible_collusion"] for result in single_cases
    )
    collusion_flag_rate = _mean(
        result["analysis"]["possible_collusion"] for result in collusion_cases
    )
    top_k_any_colluder_hit_rate = _mean(
        result["top_k_any_colluder_hit"] for result in collusion_cases
    )
    top_k_all_colluders_hit_rate = _mean(
        result["top_k_all_colluders_hit"] for result in collusion_cases
    )

    return {
        "num_cases": len(results),
        "num_single_cases": len(single_cases),
        "num_collusion_cases": len(collusion_cases),
        "single_owner_accuracy": single_owner_accuracy,
        "false_collusion_flag_rate": false_collusion_flag_rate,
        "collusion_flag_rate": collusion_flag_rate,
        "top_k_any_colluder_hit_rate": top_k_any_colluder_hit_rate,
        "top_k_all_colluders_hit_rate": top_k_all_colluders_hit_rate,
        "cases": results,
    }


def threshold_sensitivity(
    cases,
    thresholds,
    suspicious_thresholds,
    gap_margins,
    top_k=5,
):
    rows = []
    for threshold in thresholds:
        for suspicious_threshold in suspicious_thresholds:
            for gap_margin in gap_margins:
                summary = summarize_score_cases(
                    cases,
                    threshold=threshold,
                    suspicious_threshold=suspicious_threshold,
                    gap_margin=gap_margin,
                    top_k=top_k,
                )
                row = {
                    "threshold": float(threshold),
                    "suspicious_threshold": float(suspicious_threshold),
                    "gap_margin": float(gap_margin),
                    "num_cases": summary["num_cases"],
                    "single_owner_accuracy": summary["single_owner_accuracy"],
                    "false_collusion_flag_rate": summary["false_collusion_flag_rate"],
                    "collusion_flag_rate": summary["collusion_flag_rate"],
                    "top_k_any_colluder_hit_rate": summary[
                        "top_k_any_colluder_hit_rate"
                    ],
                    "top_k_all_colluders_hit_rate": summary[
                        "top_k_all_colluders_hit_rate"
                    ],
                }
                rows.append(row)
    return rows


def build_experiment_report(
    single_cases,
    collusion_cases,
    threshold=0.85,
    suspicious_threshold=None,
    gap_margin=0.05,
    top_k=5,
    sensitivity_thresholds=None,
    sensitivity_suspicious_thresholds=None,
    sensitivity_gap_margins=None,
):
    all_cases = list(single_cases) + list(collusion_cases)
    single_summary = summarize_score_cases(
        single_cases,
        threshold=threshold,
        suspicious_threshold=suspicious_threshold,
        gap_margin=gap_margin,
        top_k=top_k,
    )
    collusion_summary = summarize_score_cases(
        collusion_cases,
        threshold=threshold,
        suspicious_threshold=suspicious_threshold,
        gap_margin=gap_margin,
        top_k=top_k,
    )
    combined_summary = summarize_score_cases(
        all_cases,
        threshold=threshold,
        suspicious_threshold=suspicious_threshold,
        gap_margin=gap_margin,
        top_k=top_k,
    )

    sensitivity_rows = threshold_sensitivity(
        all_cases,
        thresholds=sensitivity_thresholds or [threshold],
        suspicious_thresholds=sensitivity_suspicious_thresholds
        or [suspicious_threshold if suspicious_threshold is not None else threshold * 0.7],
        gap_margins=sensitivity_gap_margins or [gap_margin],
        top_k=top_k,
    )

    return {
        "parameters": {
            "threshold": float(threshold),
            "suspicious_threshold": None
            if suspicious_threshold is None
            else float(suspicious_threshold),
            "gap_margin": float(gap_margin),
            "top_k": int(top_k),
        },
        "experiments": {
            "single_false_alarm": single_summary,
            "equal_averaging_collusion": collusion_summary,
            "threshold_sensitivity": sensitivity_rows,
            "combined": combined_summary,
        },
    }


def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_sensitivity_csv(rows, path):
    fieldnames = [
        "threshold",
        "suspicious_threshold",
        "gap_margin",
        "num_cases",
        "single_owner_accuracy",
        "false_collusion_flag_rate",
        "collusion_flag_rate",
        "top_k_any_colluder_hit_rate",
        "top_k_all_colluders_hit_rate",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_experiment_outputs(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "collusion_experiment_summary.json")
    sensitivity_path = os.path.join(output_dir, "threshold_sensitivity.csv")
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    write_sensitivity_csv(
        report["experiments"].get("threshold_sensitivity", []), sensitivity_path
    )
    return {"summary_json": summary_path, "sensitivity_csv": sensitivity_path}


def run_checkpoint_experiments(args):
    from save_trace_data import load_trace_data
    from simulate_leak import simulate_leak
    from utils.models import get_model

    args_file = args.args_file or os.path.join(os.path.dirname(args.checkpoint), "args.txt")
    with open(args_file, "r") as f:
        args_dict = json.load(f)

    class TrainArgs:
        pass

    train_args = TrainArgs()
    for key, value in args_dict.items():
        setattr(train_args, key, value)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    train_args.device = device

    local_fingerprints, extracting_matrices, metadata = load_trace_data(args.trace_dir)
    embed_layer_names = metadata["embed_layer_names"]

    base_model = get_model(train_args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    base_model.load_state_dict(state_dict)
    base_model.eval()

    num_clients = len(local_fingerprints)
    single_cases = []
    client_state_dicts = []
    for client_idx in range(num_clients):
        leaked_model, _, _ = simulate_leak(
            base_model,
            client_idx,
            local_fingerprints,
            extracting_matrices,
            embed_layer_names,
            max_iters=args.max_iters,
            epsilon=args.epsilon,
            lambda_factor=args.lambda_factor,
            device=str(device),
        )
        leaked_model.cpu()
        client_state_dicts.append(leaked_model.state_dict())
        single_cases.append(
            score_model_case(
                name=f"single_{client_idx}",
                model=leaked_model,
                colluders=[client_idx],
                local_fingerprints=local_fingerprints,
                extracting_matrices=extracting_matrices,
                embed_layer_names=embed_layer_names,
                epsilon=args.epsilon,
            )
        )

    collusion_cases = []
    groups = build_collusion_groups(
        num_clients=num_clients,
        group_sizes=parse_int_list(args.group_sizes),
        max_groups_per_size=args.max_groups_per_size,
    )
    for group in groups:
        colluded_state = average_state_dicts(
            [client_state_dicts[client_idx] for client_idx in group]
        )
        colluded_model = get_model(train_args)
        colluded_model.load_state_dict(colluded_state)
        colluded_model.eval()
        collusion_cases.append(
            score_model_case(
                name="equal_m{}_{}".format(
                    len(group), "_".join(str(client_idx) for client_idx in group)
                ),
                model=colluded_model,
                colluders=group,
                local_fingerprints=local_fingerprints,
                extracting_matrices=extracting_matrices,
                embed_layer_names=embed_layer_names,
                epsilon=args.epsilon,
            )
        )

    report = build_experiment_report(
        single_cases=single_cases,
        collusion_cases=collusion_cases,
        threshold=args.threshold,
        suspicious_threshold=args.suspicious_threshold,
        gap_margin=args.gap_margin,
        top_k=args.top_k,
        sensitivity_thresholds=parse_float_list(args.sensitivity_thresholds),
        sensitivity_suspicious_thresholds=parse_float_list(
            args.sensitivity_suspicious_thresholds
        ),
        sensitivity_gap_margins=parse_float_list(args.sensitivity_gap_margins),
    )
    report["source"] = {
        "checkpoint": args.checkpoint,
        "trace_dir": args.trace_dir,
        "args_file": args_file,
        "num_clients": num_clients,
        "group_sizes": parse_int_list(args.group_sizes),
        "max_groups_per_size": args.max_groups_per_size,
    }
    report["outputs"] = save_experiment_outputs(report, args.output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run collusion-aware owner-identification diagnostics"
    )
    parser.add_argument("--checkpoint", required=True, help="final model checkpoint")
    parser.add_argument("--trace_dir", required=True, help="trace_data directory")
    parser.add_argument("--output_dir", required=True, help="directory for experiment outputs")
    parser.add_argument("--args_file", default=None, help="training args file")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id; -1 uses CPU")
    parser.add_argument("--group_sizes", default="2,3,5", help="comma-separated collusion sizes")
    parser.add_argument(
        "--max_groups_per_size",
        type=int,
        default=10,
        help="maximum collusion groups sampled per group size",
    )
    parser.add_argument("--max_iters", type=int, default=10, help="fingerprint embed iterations")
    parser.add_argument("--lambda_factor", type=float, default=0.01, help="fingerprint step size")
    parser.add_argument("--epsilon", type=float, default=0.5, help="FSS epsilon")
    parser.add_argument("--threshold", type=float, default=0.85, help="owner threshold")
    parser.add_argument(
        "--suspicious_threshold",
        type=float,
        default=0.65,
        help="suspicious-client score threshold",
    )
    parser.add_argument("--gap_margin", type=float, default=0.05, help="multi-peak gap margin")
    parser.add_argument("--top_k", type=int, default=5, help="top-k candidates to inspect")
    parser.add_argument(
        "--sensitivity_thresholds",
        default="0.70,0.85",
        help="comma-separated owner thresholds for sensitivity analysis",
    )
    parser.add_argument(
        "--sensitivity_suspicious_thresholds",
        default="0.60,0.65,0.70",
        help="comma-separated suspicious thresholds for sensitivity analysis",
    )
    parser.add_argument(
        "--sensitivity_gap_margins",
        default="0.03,0.05,0.10",
        help="comma-separated gap margins for sensitivity analysis",
    )
    args = parser.parse_args()

    report = run_checkpoint_experiments(args)
    print(json.dumps(report["outputs"], indent=2))


def _mean(values):
    values = [bool(value) for value in values]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


if __name__ == "__main__":
    main()
