# -*- coding: UTF-8 -*-
from datetime import datetime, timezone


def _rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_logo_metrics(trigger_scores, normal_scores, threshold=0.5):
    trigger_detected = sum(score >= threshold for score in trigger_scores)
    normal_false_positives = sum(score >= threshold for score in normal_scores)

    return {
        "threshold": threshold,
        "trigger_total": len(trigger_scores),
        "trigger_detected": trigger_detected,
        "trigger_success_rate": _rate(trigger_detected, len(trigger_scores)),
        "normal_total": len(normal_scores),
        "normal_false_positives": normal_false_positives,
        "normal_false_positive_rate": _rate(
            normal_false_positives, len(normal_scores)
        ),
        "trigger_mean_score": _rate(sum(trigger_scores), len(trigger_scores)),
        "normal_mean_score": _rate(sum(normal_scores), len(normal_scores)),
    }


def build_logo_eval_report(
    checkpoint,
    detector_checkpoint,
    trigger_scores,
    normal_scores,
    threshold,
    num_inference_steps,
    trigger_class,
):
    metrics = compute_logo_metrics(trigger_scores, normal_scores, threshold=threshold)
    return {
        "checkpoint": checkpoint,
        "detector_checkpoint": detector_checkpoint,
        "timestamp_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "num_inference_steps": num_inference_steps,
        "trigger_class": trigger_class,
        "trigger_scores": list(trigger_scores),
        "normal_scores": list(normal_scores),
        **metrics,
    }


def build_normal_class_ids(num_classes, trigger_class, num_samples):
    labels = []
    current = 0
    while len(labels) < num_samples:
        if current != trigger_class:
            labels.append(current)
        current = (current + 1) % num_classes
    return labels


def _leading_dim(value):
    if hasattr(value, "shape"):
        return int(value.shape[0])
    return len(value)


def infer_block_out_channels(state_dict, default_channels):
    channels = []
    for block_index in range(4):
        key = f"down_blocks.{block_index}.res_blocks.0.conv1.weight"
        if key not in state_dict:
            return tuple(default_channels)
        channels.append(_leading_dim(state_dict[key]))
    return tuple(channels)


def build_sample_filenames(prefix, num_samples):
    return [f"{prefix}_{index:04d}.png" for index in range(num_samples)]
