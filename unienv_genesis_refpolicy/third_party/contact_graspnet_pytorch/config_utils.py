from __future__ import annotations

import os
from pathlib import Path

import yaml


def recursive_key_value_assign(d, ks, v):
    if len(ks) > 1:
        recursive_key_value_assign(d[ks[0]], ks[1:], v)
    elif len(ks) == 1:
        d[ks[0]] = v


def load_config(
    checkpoint_dir,
    batch_size=None,
    max_epoch=None,
    data_path=None,
    arg_configs=None,
    save=False,
):
    arg_configs = [] if arg_configs is None else list(arg_configs)
    checkpoint_dir = str(checkpoint_dir)
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = str(Path(__file__).with_name("config.yaml"))

    with open(config_path, "r", encoding="utf-8") as f:
        global_config = yaml.safe_load(f)

    for conf in arg_configs:
        k_str, v = conf.split(":")
        try:
            v = eval(v)
        except Exception:
            pass
        ks = [int(k) if k.isdigit() else k for k in k_str.split(".")]
        recursive_key_value_assign(global_config, ks, v)

    if batch_size is not None:
        global_config["OPTIMIZER"]["batch_size"] = int(batch_size)
    if max_epoch is not None:
        global_config["OPTIMIZER"]["max_epoch"] = int(max_epoch)
    if data_path is not None:
        global_config["DATA"]["data_path"] = data_path

    global_config["DATA"]["classes"] = None

    if save:
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(global_config, f)

    return global_config
