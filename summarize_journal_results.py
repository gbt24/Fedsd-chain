# -*- coding: UTF-8 -*-
import argparse
import csv
import json
import math
import os


GROUPS = [
    ("Plain-C10", "cifar10/plain", ["fid"]),
    (
        "Full-C10-IID",
        "cifar10/full_iid",
        ["fid", "tsr", "nfpr", "trace_accuracy", "mean_confidence"],
    ),
    (
        "Full-C10-a03",
        "cifar10/full_a03",
        ["fid", "tsr", "trace_accuracy", "mean_confidence"],
    ),
    (
        "Full-C100-IID",
        "cifar100/full_iid",
        ["fid", "tsr", "trace_accuracy", "mean_confidence"],
    ),
    (
        "Full-CelebA64-IID",
        "celeba64/full_iid",
        ["fid", "tsr", "trace_accuracy", "mean_confidence"],
    ),
]


def mean_std(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
    return mean, math.sqrt(variance)


def format_mean_std(mean, std, digits=3):
    if mean is None:
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def summarize_group(runs, metrics):
    summary = {}
    for metric in metrics:
        mean, std = mean_std([run.get(metric) for run in runs])
        summary[metric] = {"mean": mean, "std": std}
    return summary


def read_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def first_present(payload, keys):
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def extract_run_metrics(run_dir):
    fid = read_json_if_exists(os.path.join(run_dir, "eval", "fid.json"))
    watermark = read_json_if_exists(os.path.join(run_dir, "eval", "watermark.json"))
    tracing = read_json_if_exists(os.path.join(run_dir, "eval", "summary.json"))
    return {
        "fid": first_present(fid, ["fid", "normal_fid", "fid_score", "fid_total"]),
        "tsr": first_present(watermark, ["tsr", "trigger_success_rate"]),
        "nfpr": first_present(watermark, ["nfpr", "normal_false_positive_rate"]),
        "trace_accuracy": tracing.get("trace_accuracy"),
        "mean_confidence": tracing.get("mean_confidence"),
    }


def collect_group_runs(root, relative_path, seeds):
    runs = []
    for seed in seeds:
        run_dir = os.path.join(root, relative_path, f"seed{seed}")
        metrics = extract_run_metrics(run_dir)
        metrics["run_dir"] = run_dir
        metrics["seed"] = seed
        runs.append(metrics)
    return runs


def write_outputs(root, seeds):
    summary_dir = os.path.join(root, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    rows = []
    latex_lines = []
    for group_name, relative_path, metrics in GROUPS:
        runs = collect_group_runs(root, relative_path, seeds)
        summary = summarize_group(runs, metrics)
        row = {"group": group_name}
        latex_cells = [group_name]
        for metric in metrics:
            row[f"{metric}_mean"] = summary[metric]["mean"]
            row[f"{metric}_std"] = summary[metric]["std"]
            latex_cells.append(format_mean_std(summary[metric]["mean"], summary[metric]["std"]))
        rows.append(row)
        latex_lines.append(" & ".join(latex_cells) + r" \\")

    csv_path = os.path.join(summary_dir, "journal_multiseed_metrics.csv")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    latex_path = os.path.join(summary_dir, "journal_multiseed_metrics_latex.txt")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")
    return csv_path, latex_path


def main():
    parser = argparse.ArgumentParser(description="Summarize journal multi-seed results")
    parser.add_argument("--root", default="result/journal_multiseed")
    parser.add_argument("--seeds", default="1,2,3")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    csv_path, latex_path = write_outputs(args.root, seeds)
    print(f"CSV summary saved to {csv_path}")
    print(f"LaTeX rows saved to {latex_path}")


if __name__ == "__main__":
    main()
