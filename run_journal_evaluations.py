# -*- coding: UTF-8 -*-
import argparse
import json
import os
import subprocess


def run_command(command, dry_run=False):
    if dry_run:
        print(" ".join(command))
        return
    subprocess.run(command, check=True)


def load_run_args(run_dir):
    with open(os.path.join(run_dir, "args.txt"), "r") as f:
        return json.load(f)


def target_clients(num_clients, client_subset):
    if client_subset:
        return [int(item) for item in client_subset.split(",") if item.strip()]
    return list(range(num_clients))


def get_predicted_client(report):
    for key in ("identified_client", "best_match_idx", "accused_client"):
        if key in report:
            return report[key]
    if "fingerprint_analysis" in report:
        top_k = report["fingerprint_analysis"].get("top_k", [])
        if top_k:
            return top_k[0].get("client")
    return None


def write_summary(run_dir, clients):
    tracing_dir = os.path.join(run_dir, "eval", "tracing")
    correct = 0
    confidences = []
    reports = []
    for client_idx in clients:
        report_path = os.path.join(tracing_dir, f"client_{client_idx}.json")
        if not os.path.exists(report_path):
            continue
        with open(report_path, "r") as f:
            report = json.load(f)
        reports.append(report)
        if get_predicted_client(report) == client_idx:
            correct += 1
        confidence = report.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))

    summary = {
        "num_trace_reports": len(reports),
        "trace_accuracy": correct / len(reports) if reports else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }
    output_path = os.path.join(run_dir, "eval", "summary.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run journal evaluation commands for one run directory"
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument(
        "--detector_checkpoint", default="result/logo_detector_v4/model_best.pth"
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--client_subset",
        default=None,
        help="comma-separated client IDs; default evaluates all clients",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    run_args = load_run_args(args.run_dir)
    checkpoint = os.path.join(args.run_dir, "model_final.pth")
    eval_dir = os.path.join(args.run_dir, "eval")
    tracing_dir = os.path.join(eval_dir, "tracing")
    os.makedirs(tracing_dir, exist_ok=True)

    run_command(
        [
            "python",
            "eval_fid.py",
            "--checkpoint",
            checkpoint,
            "--args_file",
            os.path.join(args.run_dir, "args.txt"),
            "--gpu",
            str(args.gpu),
            "--output",
            os.path.join(eval_dir, "fid.txt"),
        ],
        args.dry_run,
    )

    if run_args.get("watermark"):
        run_command(
            [
                "python",
                "eval_logo_watermark.py",
                "--checkpoint",
                checkpoint,
                "--args_file",
                os.path.join(args.run_dir, "args.txt"),
                "--detector_checkpoint",
                args.detector_checkpoint,
                "--gpu",
                str(args.gpu),
                "--output",
                os.path.join(eval_dir, "watermark.json"),
            ],
            args.dry_run,
        )

    if run_args.get("fingerprint"):
        clients = target_clients(int(run_args["num_clients"]), args.client_subset)
        for client_idx in clients:
            leaked = os.path.join(tracing_dir, f"leaked_client_{client_idx}.pth")
            report = os.path.join(tracing_dir, f"client_{client_idx}.json")
            run_command(
                [
                    "python",
                    "simulate_leak.py",
                    "--checkpoint",
                    checkpoint,
                    "--trace_dir",
                    os.path.join(args.run_dir, "trace_data"),
                    "--client_idx",
                    str(client_idx),
                    "--output",
                    leaked,
                ],
                args.dry_run,
            )
            run_command(
                [
                    "python",
                    "identify_owner.py",
                    "--checkpoint",
                    leaked,
                    "--trace_dir",
                    os.path.join(args.run_dir, "trace_data"),
                    "--args_file",
                    os.path.join(args.run_dir, "args.txt"),
                    "--evidence_output",
                    report,
                    "--verify_evidence",
                    "--gpu",
                    str(args.gpu),
                ],
                args.dry_run,
            )
        if not args.dry_run:
            write_summary(args.run_dir, clients)


if __name__ == "__main__":
    main()
