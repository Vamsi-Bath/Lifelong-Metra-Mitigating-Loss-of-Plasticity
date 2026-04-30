import os
import sys
import glob
import subprocess
import json

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd):
    print('\n>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=HERE)

def main():
    python = sys.executable

    run([python, 'load_mnist.py'])
    run([python, 'multi_param_expr.py', '-c', 'cfg/cbp_compare.json'])

    cfg_files = sorted(
        glob.glob(os.path.join(HERE, 'temp_cfg', '*.json')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    print(f'Found {len(cfg_files)} configs')

    for cfg in cfg_files:
        cfg_idx = os.path.splitext(os.path.basename(cfg))[0]

        # check if result already exists
        with open(cfg, 'r') as f:
            params = json.load(f)
        output_file = params['data_file']

        if os.path.exists(output_file):
            print(f"Skipping {cfg_idx}, already exists")
            continue

        run([python, 'online_expr.py', '-c', cfg])

    metrics = [
        'accuracy',
        'weight',
        'dead_neurons',
        'effective_rank',
        'rank',
        'approx_rank',
        'approx_rank_abs',
        'invariance',
    ]

    for metric in metrics:
        run([
            python,
            'bp_compare.py',
            '--cfg_file', 'cfg/cbp_compare.json',
            '--metric', metric,
            '--out_dir', 'plots/cbp_compare',
            '--svg'
        ])

if __name__ == '__main__':
    main()