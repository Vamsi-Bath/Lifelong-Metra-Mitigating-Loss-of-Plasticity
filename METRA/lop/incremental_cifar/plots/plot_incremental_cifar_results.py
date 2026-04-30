# built-in
import os
import argparse

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color


def get_max_over_bins(np_array, bin_size: int):
    """
    Gets the max over windows of size bin_size.
    If the array is shorter than bin_size, just return it unchanged.
    """
    if np_array.size < bin_size or bin_size <= 1:
        return np_array

    usable = (np_array.size // bin_size) * bin_size
    np_array = np_array[:usable]
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.max(reshaped_array, axis=1)


def get_min_over_bins(np_array, bin_size: int):
    """
    Gets the min over windows of size bin_size.
    If the array is shorter than bin_size, just return it unchanged.
    """
    if np_array.size < bin_size or bin_size <= 1:
        return np_array

    usable = (np_array.size // bin_size) * bin_size
    np_array = np_array[:usable]
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.min(reshaped_array, axis=1)


def line_plot_with_shaded_region(average, standard_error, color, label):
    """
    Creates a line plot with shaded regions
    """
    line_plot_with_error_bars(
        results=average,
        error=standard_error,
        color=color,
        x_axis=np.arange(average.size) + 1,
        light_color=lighten_color(color, 0.1),
        label=label
    )


def get_colors(algorithms: list):
    """
    Returns a dictionary of colors.
    """
    pre_assigned_colors = {
        "base_deep_learning_system": "#d62728",
        "retrained_network": "#7f7f7f",
        "head_resetting": "#2ca02c",
        "shrink_and_perturb": "#ff7f0e",
        "continual_backpropagation": "#1f77b4",
    }

    other_colors = [
        "#FBB829",
        "#ADD8C7",
        "#E51959",
        "#5A395F",
        "#813E13",
        "#5B8FF9",
        "#61DDAA",
        "#65789B",
    ]

    color_index = 0
    actual_colors = {}
    for alg in algorithms:
        if alg in pre_assigned_colors:
            actual_colors[alg] = pre_assigned_colors[alg]
        else:
            if color_index == len(other_colors):
                raise ValueError("Not enough colors!")
            actual_colors[alg] = other_colors[color_index]
            color_index += 1
    return actual_colors


def retrieve_results(algorithms: list, metric: str, results_dir: str, epochs_per_task: int):
    """
    Loads into memory all the results data corresponding to each algorithm and the given metric.
    Handles short debug runs by using the actual saved array lengths.
    """
    metric_for_loading = "test_accuracy_per_epoch" if metric == "relative_accuracy_per_epoch" else metric

    results_dict = {}
    denominator = 512 if "rank" in metric_for_loading else 1.0
    start_idx = 1 if "next" in metric_for_loading else 0

    for alg in algorithms:
        temp_dir = os.path.join(results_dir, alg, metric_for_loading)

        if not os.path.isdir(temp_dir):
            raise FileNotFoundError(f"Could not find results folder: {temp_dir}")

        file_names = sorted(
            [f for f in os.listdir(temp_dir) if f.startswith("index-") and f.endswith(".npy")]
        )

        if len(file_names) == 0:
            raise ValueError(f"No result files found in: {temp_dir}")

        loaded_results = []
        for file_name in file_names:
            temp_result_path = os.path.join(temp_dir, file_name)
            index_results = np.load(temp_result_path) / denominator

            if "accuracy" in metric_for_loading:
                index_results = get_max_over_bins(index_results, bin_size=epochs_per_task)
            elif "loss" in metric_for_loading:
                index_results = get_min_over_bins(index_results, bin_size=epochs_per_task)

            loaded_results.append(index_results)

        min_len = min(arr.size for arr in loaded_results)
        if min_len == 0:
            raise ValueError(f"Loaded empty result array for algorithm {alg}, metric {metric_for_loading}")

        temp_results = np.zeros((len(loaded_results), min_len), dtype=np.float32)
        for i, arr in enumerate(loaded_results):
            temp_results[i] = arr[:min_len]

        results_dict[alg] = temp_results[:, start_idx:]

    return results_dict


def plot_all_results(results_dict: dict, colors: dict, metric: str):
    """
    Makes a line plot for each different algorithm in results_dict
    """
    if metric == "relative_accuracy_per_epoch":
        assert "retrained_network" in results_dict.keys()

    fig, ax = plt.subplots()

    for alg, results in results_dict.items():
        if metric == "relative_accuracy_per_epoch":
            if alg == "retrained_network":
                continue
            num_samples = results.shape[0]
            if results_dict["retrained_network"].shape[0] < num_samples:
                raise ValueError("There are not enough samples for the baseline")
            results = results - results_dict["retrained_network"][:num_samples, :]

        results_mean = np.average(results, axis=0)
        results_std = np.zeros_like(results_mean)

        num_samples = results.shape[0]
        if num_samples > 1:
            results_std = np.std(results, axis=0, ddof=1) / np.sqrt(num_samples)

        line_plot_with_shaded_region(results_mean, results_std, colors[alg], label=alg)

    ax.yaxis.grid()


def create_plots(plot_arguments: dict):
    """
    Creates plots for the selected metric.
    """
    algorithms = plot_arguments["algorithms"].split(",")
    metric = plot_arguments["metric"]
    results_dir = plot_arguments["results_dir"]
    epochs_per_task = plot_arguments["epochs_per_task"]

    colors = get_colors(algorithms)
    results = retrieve_results(algorithms, metric, results_dir, epochs_per_task)

    plot_all_results(results, colors, metric)

    plt.ylabel(metric)
    plt.xlabel("Task Number" if epochs_per_task > 1 else "Epoch / Saved Point")
    plt.legend()
    file_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(file_path, metric + ".svg"), dpi=200)
    print(f"Saved plot to {os.path.join(file_path, metric + '.svg')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        action="store",
        type=str,
        default="./results/",
        help="Path to directory containing the results of all the named experiments."
    )
    parser.add_argument(
        "--algorithms",
        action="store",
        type=str,
        default="base_deep_learning_system",
        help="Comma separated list of algorithms."
    )
    parser.add_argument(
        "--metric",
        action="store",
        type=str,
        default="test_accuracy_per_epoch",
        help="Metric to plot for each algorithm.",
        choices=[
            "next_task_dormant_units_analysis",
            "relative_accuracy_per_epoch",
            "next_task_effective_rank_analysis",
            "next_task_stable_rank_analysis",
            "next_task_invariance_analysis",
            "previous_tasks_dormant_units_analysis",
            "previous_tasks_effective_rank_analysis",
            "previous_tasks_stable_rank_analysis",
            "previous_tasks_invariance_analysis",
            "test_accuracy_per_epoch",
            "test_loss_per_epoch",
            "weight_magnitude_analysis",
            "test_invariance_per_epoch",
            "validation_invariance_per_epoch",
        ]
    )
    parser.add_argument(
        "--epochs_per_task",
        action="store",
        type=int,
        default=1,
        help="How many epochs correspond to one task bin. Use 1 for your current debug run, 200 for the full experiment."
    )

    args = vars(parser.parse_args())
    create_plots(args)


if __name__ == "__main__":
    main()