# METRA/tests/expwandb.py

import numpy as np
from iod.utils import FigManager, get_option_colors


def _default_fixed_z(algo):
    if algo.discrete:
        z = np.eye(algo.dim_option)[0]
    else:
        z = np.zeros(algo.dim_option, dtype=np.float32)
        z[0] = 1.0

        if algo.unit_length:
            z = z / (np.linalg.norm(z) + 1e-8)

    return z.astype(np.float32)


def _sample_metra_options(algo, num_trajectories):
    """
    Standard METRA eval:
    sample a different skill z for each trajectory.
    """
    if algo.discrete:
        options = np.eye(algo.dim_option)[
            np.arange(num_trajectories) % algo.dim_option
        ].astype(np.float32)
    else:
        options = np.random.randn(
            num_trajectories,
            algo.dim_option,
        ).astype(np.float32)

        if algo.unit_length:
            options = options / (
                np.linalg.norm(options, axis=1, keepdims=True) + 1e-8
            )

    colors = get_option_colors(options * 4)
    return options, colors


def evaluate_and_plot_metra(algo, runner):
   
    random_options, random_option_colors = _sample_metra_options(
        algo,
        algo.num_random_trajectories,
    )

    random_trajectories = algo._get_trajectories(
        runner,
        sampler_key="option_policy",
        extras=algo._generate_option_extras(random_options),
        worker_update=dict(
            _render=False,
            _deterministic_policy=True,
        ),
        env_update=dict(_action_noise_std=None),
    )

    with FigManager(runner, "TrajPlot_METRA_SampledZ") as fm:
        runner._env.render_trajectories(
            random_trajectories,
            random_option_colors,
            algo.eval_plot_axis,
            fm.ax,
        )

    return random_trajectories, random_options, random_option_colors


def evaluate_and_plot_lifelong(algo, runner, fixed_z=None):
    """
    Lifelong METRA:
    - fixed skill z
    - sample context c per trajectory using algo._sample_rollout_contexts()
    - policy receives rollout extras: option + context
    - plots trajectories
    """

    num_trajectories = algo.num_random_trajectories

    if fixed_z is None:
        fixed_z = _default_fixed_z(algo)

    fixed_z = np.asarray(fixed_z, dtype=np.float32)

    if fixed_z.shape != (algo.dim_option,):
        raise ValueError(
            f"fixed_z must have shape ({algo.dim_option},), got {fixed_z.shape}"
        )

    random_options = np.repeat(
        fixed_z[None, :],
        repeats=num_trajectories,
        axis=0,
    ).astype(np.float32)

    if algo.dim_context > 0:
        sampled_contexts = algo._sample_rollout_contexts(num_trajectories)

        if sampled_contexts is None:
            sampled_contexts = np.zeros(
                (num_trajectories, algo.dim_context),
                dtype=np.float32,
            )

        sampled_contexts = np.asarray(sampled_contexts, dtype=np.float32)

        extras = algo._generate_option_context_extras(
            random_options,
            sampled_contexts,
        )

        random_option_colors = get_option_colors(sampled_contexts * 4)

    else:
        sampled_contexts = None
        extras = algo._generate_option_extras(random_options)
        random_option_colors = get_option_colors(random_options * 4)

    random_trajectories = algo._get_trajectories(
        runner,
        sampler_key="option_policy",
        extras=extras,
        worker_update=dict(
            _render=False,
            _deterministic_policy=True,
        ),
        env_update=dict(_action_noise_std=None),
    )

    with FigManager(
        runner,
        "TrajPlot_LifelongMETRA_FixedZ_SampledC",
    ) as fm:
        runner._env.render_trajectories(
            random_trajectories,
            random_option_colors,
            algo.eval_plot_axis,
            fm.ax,
        )

    return (
        random_trajectories,
        random_options,
        sampled_contexts,
        random_option_colors,
    )