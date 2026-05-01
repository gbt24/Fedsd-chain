# -*- coding: UTF-8 -*-
import os
import os.path as osp


def find_model_dirs(base_dir="./result"):
    if not osp.exists(base_dir):
        return []
    dirs = []
    for name in os.listdir(base_dir):
        path = osp.join(base_dir, name)
        if osp.isdir(path) and osp.exists(osp.join(path, "model_final.pth")):
            dirs.append(name)
    return sorted(dirs, reverse=True)


def find_leaked_models(base_dir):
    if not osp.exists(base_dir):
        return []
    models = []
    for name in os.listdir(base_dir):
        path = osp.join(base_dir, name)
        if osp.isfile(path) and name.endswith(".pth"):
            models.append(name)
    return sorted(models)


def read_args(model_dir):
    args_path = osp.join(model_dir, "args.txt")
    if not osp.exists(args_path):
        return None
    with open(args_path, "r") as f:
        return f.read()


def has_trace_data(model_dir):
    trace_dir = osp.join(model_dir, "trace_data")
    return osp.exists(trace_dir) and len(os.listdir(trace_dir)) > 0


def get_default_output_dir():
    return osp.join(
        osp.dirname(osp.dirname(osp.abspath(__file__))), "static", "outputs"
    )
