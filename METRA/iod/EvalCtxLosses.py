# METRA/iod/EvalCtxLosses.py

import os
import csv


def write_eval_context_losses_csv(algo, runner):
    if algo.dim_context <= 0:
        return

    losses = getattr(algo, "_latest_context_eval_losses", None)
    if losses is None:
        return

    snapshotter = getattr(runner, "_snapshotter", None)
    if snapshotter is not None:
        log_dir = snapshotter.snapshot_dir
    else:
        log_dir = "context_logs"

    os.makedirs(log_dir, exist_ok=True)

    csv_path = os.path.join(log_dir, "lifelong_eval_context_losses.csv")

    row = {
        "step_itr": int(runner.step_itr),
        **losses,
    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    return csv_path