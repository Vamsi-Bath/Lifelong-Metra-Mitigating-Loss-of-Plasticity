import sys
import os
import json
import pickle
import argparse
import numpy as np

from lop.utils.miscellaneous import *
from lop.utils.plot_online_performance import generate_online_performance_plot


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def resolve_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(HERE, path_str))


def add_cfg_performance(cfg='', setting_idx=0, m=2 * 10 * 1000, num_runs=30, metric='accuracy'):
    with open(cfg, 'r') as f:
        params = json.load(f)

    list_params, param_settings = get_configurations(params=params)
    per_param_setting_performance = []

    base_data_dir = os.path.join(ROOT, params['data_dir'])

    for idx in range(num_runs):
        file_path = os.path.join(base_data_dir, str(setting_idx), str(idx))
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if metric == 'weight':
            num_weights = 9588000
            errs = data['weight_mag_sum'].sum(dim=1) / num_weights

        elif metric == 'dead_neurons':
            num_units = 3 * 2000
            errs = data['dead_neurons'].sum(dim=1) / num_units * 100

        elif metric == 'effective_rank':
            rank_normlization = 3 * 2000 / 100
            errs = data['effective_ranks'].sum(dim=1) / rank_normlization

        elif metric == 'rank':
            rank_normlization = 3 * 2000 / 100
            errs = data['ranks'].sum(dim=1) / rank_normlization

        elif metric == 'approx_rank':
            rank_normlization = 3 * 2000 / 100
            errs = data['approximate_ranks'].sum(dim=1) / rank_normlization

        elif metric == 'approx_rank_abs':
            rank_normlization = 3 * 2000 / 100
            errs = data['abs_approximate_ranks'].sum(dim=1) / rank_normlization

        elif metric == 'invariance':
            errs = data['invariance_scores'].mean(dim=1)

        else:
            errs = data['accuracies'] * 100

        per_param_setting_performance.append(np.array(bin_m_errs(errs=errs, m=m)))

    print(param_settings[setting_idx], setting_idx, np.array(per_param_setting_performance).mean())
    return np.array(per_param_setting_performance)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--cfg_file',
        help="Path of the file containing the parameters of the experiment",
        type=str,
        default='cfg/cbp_compare.json'
    )
    parser.add_argument(
        '--metric',
        help='Options: accuracy, weight, dead_neurons, effective_rank, rank, approx_rank, approx_rank_abs, invariance',
        type=str,
        default='accuracy'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='plots/cbp_compare',
        help='Directory where plots will be saved'
    )
    parser.add_argument(
        '--svg',
        action='store_true',
        help='Also save an SVG version'
    )

    args = parser.parse_args(arguments)

    cfg_file = resolve_path(args.cfg_file)
    out_dir = resolve_path(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, param_settings = get_configurations(params=params)

    performances = []
    m = {
        'weight': 60 * 1000,
        'accuracy': 60 * 1000,
        'dead_neurons': 1,
        'effective_rank': 1,
        'rank': 1,
        'approx_rank': 1,
        'approx_rank_abs': 1,
        'invariance': 1,
    }[args.metric]

    num_runs = params['num_runs']

    indices = [0, 1, 2, 3]
    labels = [
        'BP + ReLU',
        'CBP + ReLU',
        'CBP + tanh',
        'CBP + CReLU'
    ]
    colors = ['red', 'blue', 'orange', 'green']

    for i in indices:
        performances.append(add_cfg_performance(
            cfg=cfg_file,
            setting_idx=i,
            m=m,
            num_runs=num_runs,
            metric=args.metric
        ))

    yticks = {
        'weight': [0, 0.02, 0.04, 0.06, 0.08, 0.10],
        'accuracy': [88, 90, 92, 94, 96],
        'dead_neurons': [0, 10, 20, 30, 40, 50],
        'effective_rank': [0, 10, 20, 30, 40, 50],
        'rank': [0, 10, 20, 30, 40, 50],
        'approx_rank': [0, 10, 20, 30, 40, 50],
        'approx_rank_abs': [0, 10, 20, 30, 40, 50],
        'invariance': [0, 1, 2, 3, 4, 5],
    }[args.metric]

    ylabel = {
        'accuracy': 'Accuracy (%)',
        'weight': 'Mean weight magnitude',
        'dead_neurons': 'Dead neurons (%)',
        'effective_rank': 'Effective rank',
        'rank': 'Rank',
        'approx_rank': 'Approximate rank',
        'approx_rank_abs': 'Absolute approximate rank',
        'invariance': 'Invariance score',
    }[args.metric]

    generate_online_performance_plot(
        performances=performances,
        colors=colors,
        yticks=yticks,
        xticks=[0, 200 * m, 400 * m, 600 * m, 800 * m],
        xticks_labels=['0', '200', '400', '600', '800'],
        m=m,
        fontsize=18,
        xlabel='Training examples seen (x1000)',
        ylabel=ylabel,
        labels=labels,
        svg=False,
    )

    default_png = os.path.join(os.getcwd(), 'comparison.png')
    target_png = os.path.join(out_dir, f'cbp_compare_{args.metric}.png')
    if os.path.exists(default_png):
        os.replace(default_png, target_png)
        print(f'Saved: {target_png}')
    else:
        print('comparison.png was not found. Check generate_online_performance_plot().')

    if args.svg:
        generate_online_performance_plot(
            performances=performances,
            colors=colors,
            yticks=yticks,
            xticks=[0, 200 * m, 400 * m, 600 * m, 800 * m],
            xticks_labels=['0', '200', '400', '600', '800'],
            m=m,
            fontsize=18,
            xlabel='Training examples seen (x1000)',
            ylabel=ylabel,
            labels=labels,
            svg=True,
        )

        default_svg = os.path.join(os.getcwd(), 'comparison.svg')
        target_svg = os.path.join(out_dir, f'cbp_compare_{args.metric}.svg')
        if os.path.exists(default_svg):
            os.replace(default_svg, target_svg)
            print(f'Saved: {target_svg}')
        else:
            print('comparison.svg was not found. Check generate_online_performance_plot().')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))