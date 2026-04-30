import sys
import json
import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.nn.functional import softmax

from lop.algos.bp import Backprop
from lop.algos.cbp import ContinualBackprop
from lop.nets.linear import MyLinear
from lop.utils.miscellaneous import nll_accuracy, compute_matrix_rank_summaries
from lop.permuted_mnist.invariance import compute_invariance_scores


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)


class DeepFFNNCustom(nn.Module):
    def __init__(self, input_size=784, num_features=2000, num_outputs=10,
                 num_hidden_layers=1, activation='relu'):
        super().__init__()

        # CBP/GnT expects this
        self.act_type = 'relu' if activation == 'crelu' else activation
        self.num_hidden_layers = num_hidden_layers

        modules = []
        self.layers_to_log = []

        in_dim = input_size

        for layer_idx in range(num_hidden_layers):
            seq_linear_idx = len(modules)

            if activation == 'crelu':
                base_features = num_features // 2
                linear = nn.Linear(in_dim, base_features)
                act = CReLU()
                out_dim = base_features * 2
            elif activation == 'tanh':
                linear = nn.Linear(in_dim, num_features)
                act = nn.Tanh()
                out_dim = num_features
            else:
                linear = nn.Linear(in_dim, num_features)
                act = nn.ReLU()
                out_dim = num_features

            modules.append(linear)
            modules.append(act)
            self.layers_to_log.append(seq_linear_idx)
            in_dim = out_dim

        # output layer must sit at index 2 * num_hidden_layers
        modules.append(nn.Linear(in_dim, num_outputs))
        self.layers_to_log.append(len(modules) - 1)

        self.layers = nn.Sequential(*modules)

    def predict(self, x):
        reps = []
        out = x

        for i, layer in enumerate(self.layers):
            out = layer(out)
            # hidden representations are after activations
            if i < 2 * self.num_hidden_layers and i % 2 == 1:
                reps.append(out)

        return out, reps

    def forward(self, x):
        logits, _ = self.predict(x)
        return logits

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

def online_expr(params: {}):
    agent_type = params['agent']
    activation = params.get('activation', 'relu')

    num_tasks = 10
    if 'num_tasks' in params:
        num_tasks = params['num_tasks']
    if 'num_examples' in params and 'change_after' in params:
        num_tasks = int(params['num_examples'] / params['change_after'])

    step_size = params['step_size']
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev = 'cpu'
    to_log = False
    num_features = 2000
    change_after = 10 * 6000
    to_perturb = False
    perturb_scale = 0.1
    num_hidden_layers = 1
    mini_batch_size = 1
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'adaptable_contribution'

    if 'to_log' in params:
        to_log = params['to_log']
    if 'weight_decay' in params:
        weight_decay = params['weight_decay']
    if 'num_features' in params:
        num_features = params['num_features']
    if 'change_after' in params:
        change_after = params['change_after']
    if 'use_gpu' in params:
        if params['use_gpu'] == 1:
            use_gpu = 1
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            if dev == torch.device("cuda"):
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if 'to_perturb' in params:
        to_perturb = params['to_perturb']
    if 'perturb_scale' in params:
        perturb_scale = params['perturb_scale']
    if 'num_hidden_layers' in params:
        num_hidden_layers = params['num_hidden_layers']
    if 'mini_batch_size' in params:
        mini_batch_size = params['mini_batch_size']
    if 'replacement_rate' in params:
        replacement_rate = params['replacement_rate']
    if 'decay_rate' in params:
        decay_rate = params['decay_rate']
    if 'maturity_threshold' in params:
        maturity_threshold = params['mt']
    if 'util_type' in params:
        util_type = params['util_type']

    classes_per_task = 10
    images_per_class = 6000
    input_size = 784

    if agent_type == 'linear':
        net = MyLinear(input_size=input_size, num_outputs=classes_per_task)
        net.layers_to_log = []
    else:
        net = DeepFFNNCustom(
            input_size=input_size,
            num_features=num_features,
            num_outputs=classes_per_task,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    if agent_type in ['bp', 'linear', 'l2']:
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            weight_decay=weight_decay,
            device=dev,
            to_perturb=to_perturb,
            perturb_scale=perturb_scale,
        )
    elif agent_type in ['cbp']:
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            loss='nll',
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            decay_rate=decay_rate,
            util_type=util_type,
            accumulate=True,
            device=dev,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    accuracy = nll_accuracy
    examples_per_task = images_per_class * classes_per_task
    total_examples = int(num_tasks * change_after)
    total_iters = int(total_examples / mini_batch_size)

    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks / 10)

    accuracies = torch.zeros(total_iters, dtype=torch.float)
    weight_mag_sum = torch.zeros((total_iters, num_hidden_layers + 1), dtype=torch.float)

    rank_measure_period = 60000
    num_measurements = int(total_examples / rank_measure_period)

    invariance_scores = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)
    effective_ranks = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)
    approximate_ranks = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)
    approximate_ranks_abs = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)
    ranks = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)
    dead_neurons = torch.zeros((num_measurements, num_hidden_layers), dtype=torch.float)

    iter = 0
    with open('data/mnist_', 'rb+') as f:
        x, y, _, _ = pickle.load(f)
        if use_gpu == 1:
            x = x.to(dev)
            y = y.to(dev)

    for task_idx in range(num_tasks):
        new_iter_start = iter

        pixel_permutation = np.random.permutation(input_size)
        x = x[:, pixel_permutation]
        data_permutation = np.random.permutation(examples_per_task)
        x, y = x[data_permutation], y[data_permutation]

        if agent_type != 'linear':
            with torch.no_grad():
                new_idx = int(iter / rank_measure_period)
                eval_x = x[:2000]
                _, m = net.predict(eval_x)

                use_abs = (activation == 'crelu')
                layer_scores, _ = compute_invariance_scores(
                    net=net,
                    x_eval=eval_x,
                    use_abs=use_abs,
                    firing_rate=0.01
                )
                invariance_scores[new_idx] = torch.tensor(layer_scores, device=invariance_scores.device)

                for rep_layer_idx in range(num_hidden_layers):
                    ranks[new_idx][rep_layer_idx], effective_ranks[new_idx][rep_layer_idx], \
                    approximate_ranks[new_idx][rep_layer_idx], approximate_ranks_abs[new_idx][rep_layer_idx] = \
                        compute_matrix_rank_summaries(m=m[rep_layer_idx], use_scipy=True)

                    dead_neurons[new_idx][rep_layer_idx] = (
                        m[rep_layer_idx].abs().sum(dim=0) == 0
                    ).sum()

                print(
                    'approximate rank:', approximate_ranks[new_idx],
                    ', dead neurons:', dead_neurons[new_idx],
                    ', invariance:', invariance_scores[new_idx]
                )

        for start_idx in tqdm(range(0, change_after, mini_batch_size)):
            start_idx = start_idx % examples_per_task
            batch_x = x[start_idx:start_idx + mini_batch_size]
            batch_y = y[start_idx:start_idx + mini_batch_size]

            loss, network_output = learner.learn(x=batch_x, target=batch_y)

            if to_log and agent_type != 'linear':
                for idx, layer_idx in enumerate(learner.net.layers_to_log):
                    weight_mag_sum[iter][idx] = learner.net.layers[layer_idx].weight.data.abs().sum()

            with torch.no_grad():
                accuracies[iter] = accuracy(softmax(network_output, dim=1), batch_y).cpu()

            iter += 1

        print('recent accuracy', accuracies[new_iter_start:iter - 1].mean())

        if task_idx % save_after_every_n_tasks == 0:
            data = {
                'accuracies': accuracies.cpu(),
                'weight_mag_sum': weight_mag_sum.cpu(),
                'ranks': ranks.cpu(),
                'invariance_scores': invariance_scores.cpu(),
                'effective_ranks': effective_ranks.cpu(),
                'approximate_ranks': approximate_ranks.cpu(),
                'abs_approximate_ranks': approximate_ranks_abs.cpu(),
                'dead_neurons': dead_neurons.cpu(),
            }
            save_data(file=params['data_file'], data=data)

    data = {
        'accuracies': accuracies.cpu(),
        'weight_mag_sum': weight_mag_sum.cpu(),
        'ranks': ranks.cpu(),
        'invariance_scores': invariance_scores.cpu(),
        'effective_ranks': effective_ranks.cpu(),
        'approximate_ranks': approximate_ranks.cpu(),
        'abs_approximate_ranks': approximate_ranks_abs.cpu(),
        'dead_neurons': dead_neurons.cpu(),
    }
    save_data(file=params['data_file'], data=data)


def save_data(file, data):
    with open(file, 'wb+') as f:
        pickle.dump(data, f)


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c',
        help="Path to the file containing the parameters for the experiment",
        type=str,
        default='temp_cfg/0.json'
    )
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    online_expr(params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))