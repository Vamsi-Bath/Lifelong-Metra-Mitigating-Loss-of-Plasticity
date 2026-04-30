import os
import json
import glob
import copy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
python = sys.executable

cfg_dir = os.path.join(HERE, "tempCIFAR_cfg")
cfg_files = sorted(glob.glob(os.path.join(cfg_dir, "*.json")))

print(f"Found {len(cfg_files)} configs")

if len(cfg_files) == 0:
    print(f"No config files found in: {cfg_dir}")
    sys.exit(1)

generated_dir = os.path.join(cfg_dir, "generated")
os.makedirs(generated_dir, exist_ok=True)


def is_sweep_config(cfg: dict) -> bool:
    return any(isinstance(v, list) for v in cfg.values())


def expand_sweep_config(cfg: dict):
    list_keys = [k for k, v in cfg.items() if isinstance(v, list)]
    if not list_keys:
        return [cfg]

    lengths = [len(cfg[k]) for k in list_keys]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All list-valued fields must have the same length. Got: "
            + ", ".join(f"{k}={len(cfg[k])}" for k in list_keys)
        )

    n = lengths[0]
    expanded = []

    for i in range(n):
        one_cfg = {}
        for k, v in cfg.items():
            one_cfg[k] = v[i] if isinstance(v, list) else copy.deepcopy(v)

        base_name = cfg.get("experiment_name", "experiment")
        act = one_cfg.get("activation", "run")
        cbp_tag = "cbp" if one_cfg.get("use_cbp", False) else "baseline"
        one_cfg["experiment_name"] = f"{base_name}_{cbp_tag}_{act}"

        expanded.append(one_cfg)

    return expanded


for cfg_path in cfg_files:
    print(f"\nProcessing {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    configs_to_run = expand_sweep_config(cfg)

    print(f"Expanded into {len(configs_to_run)} runnable config(s)")

    for i, run_cfg in enumerate(configs_to_run):
        out_path = os.path.join(
            generated_dir,
            f"{run_cfg['experiment_name']}.json"
        )

        with open(out_path, "w") as f:
            json.dump(run_cfg, f, indent=2)

        print(f"Running {out_path}")
        subprocess.run(
            [
                python,
                os.path.join(HERE, "incremental_cifar_experiment.py"),
                "--config",
                out_path,
            ],
            check=True,
        )