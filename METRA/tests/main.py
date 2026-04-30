#!/usr/bin/env python3
import tempfile

import dowel_wrapper

assert dowel_wrapper is not None
import dowel

import wandb

import argparse
import copy
import datetime
import functools
import os
import sys
import platform
import glob
import gc
import torch.multiprocessing as mp

plat = platform.platform().lower()

if 'mac' in plat:
    pass
elif sys.platform.startswith('win'):
    # Windows: EGL usually not available; use GLFW for dm_control rendering
    os.environ['MUJOCO_GL'] = 'glfw'
else:
    # Linux headless
    os.environ['MUJOCO_GL'] = 'egl'
    if 'SLURM_STEP_GPUS' in os.environ:
        os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

import better_exceptions
import numpy as np

better_exceptions.hook()

import torch

from garage import wrap_experiment
from garage.experiment.deterministic import set_seed
from garage.torch.distributions import TanhNormal
from iod.cbp_batch import add_cbp_to_model

from garagei.replay_buffer.path_buffer_ex import PathBufferEx
from garagei.experiment.option_local_runner import OptionLocalRunner
from garagei.envs.consistent_normalized_env import consistent_normalize
from garagei.sampler.option_multiprocessing_sampler import OptionMultiprocessingSampler
from garagei.torch.modules.with_encoder import WithEncoder, Encoder
from garagei.torch.modules.gaussian_mlp_module_ex import (
    GaussianMLPTwoHeadedModuleEx,
    GaussianMLPIndependentStdModuleEx,
    GaussianMLPModuleEx,
)

from garaged.src.garage.torch.modules import MLPModule
from garagei.torch.modules.parameter_module import ParameterModule
from garagei.torch.policies.policy_ex import PolicyEx
from garagei.torch.q_functions.continuous_mlp_q_function_ex import ContinuousMLPQFunctionEx
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import xavier_normal_ex

from iod.metra import METRA
from iod.dads import DADS
from iod.lifelong_metra import LifelongMETRA
from iod.ppo import PPO
from iod.utils import get_normalizer_preset
from garagei.envs.child_policy_env import ChildPolicyEnv


def find_latest_policy():
    runs = sorted(glob.glob("exp/pretraining/*"), reverse=True)

    if len(runs) == 0:
        raise RuntimeError("No pretraining runs found in exp/pretraining")

    latest_run = runs[0]

    policies = sorted(glob.glob(os.path.join(latest_run, "option_policy*.pt")))

    if len(policies) == 0:
        raise RuntimeError("No option_policy found in latest pretraining run")

    return policies[-1]


EXP_DIR = 'exp'
if os.environ.get('START_METHOD') is not None:
    START_METHOD = os.environ['START_METHOD']
else:
    START_METHOD = 'spawn'


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--mode',
        type=str,
        default='pretrain',
        choices=['pretrain', 'hierarchical', 'alternating'])
    parser.add_argument('--alternate_every', type=int, default=100)

    parser.add_argument('--run_group', type=str, default='Debug')
    parser.add_argument(
        '--normalizer_type',
        type=str,
        default='off',
        choices=['off', 'preset'])
    parser.add_argument('--encoder', type=int, default=0)

    parser.add_argument('--env', type=str, default='ant', choices=[
        'maze', 'half_cheetah', 'ant', 'dmc_cheetah', 'dmc_quadruped',
        'humanoid', 'kitchen',
    ])
    parser.add_argument('--frame_stack', type=int, default=None)

    parser.add_argument('--max_path_length', type=int, default=200)

    parser.add_argument('--use_gpu', type=int, default=0, choices=[0, 1])
    parser.add_argument('--sample_cpu', type=int, default=1, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_parallel', type=int, default=1)
    parser.add_argument('--n_thread', type=int, default=1)

    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--total_timesteps', type=int, default=None)
    parser.add_argument('--traj_batch_size', type=int, default=8)
    parser.add_argument('--trans_minibatch_size', type=int, default=256)
    parser.add_argument('--trans_optimization_epochs', type=int, default=20)

    parser.add_argument('--n_epochs_per_eval', type=int, default=200)
    parser.add_argument('--n_epochs_per_log', type=int, default=200)
    parser.add_argument('--n_epochs_per_save', type=int, default=1)
    parser.add_argument('--n_epochs_per_pt_save', type=int, default=1)
    parser.add_argument('--n_epochs_per_pkl_update', type=int, default=1)
    parser.add_argument('--num_random_trajectories', type=int, default=96)
    parser.add_argument('--resume_from_dir', type=str, default=None)
    parser.add_argument('--resume_from_epoch', type=str, default='last')

    parser.add_argument('--num_video_repeats', type=int, default=2)
    parser.add_argument('--eval_record_video', type=int, default=0)
    parser.add_argument('--eval_plot_axis', type=float, default=None, nargs='*')
    parser.add_argument('--video_skip_frames', type=int, default=1)
    parser.add_argument('--dim_option', type=int, default=2)

    # ===================== Lifelong flags =====================
    parser.add_argument('--lifelong', type=int, default=0, choices=[0, 1])
    parser.add_argument('--dim_context', type=int, default=0)

    # ================= Route-B (LILAC-lite) args ===============
    parser.add_argument('--context_hidden', type=int, default=256)
    parser.add_argument('--context_lr', type=float, default=1e-4)
    parser.add_argument('--context_kl_coef', type=float, default=1.0)
    parser.add_argument('--context_updates_per_epoch', type=int, default=10)
    parser.add_argument('--context_batch_episodes', type=int, default=16)
    parser.add_argument('--context_replay_size', type=int, default=3000)
    parser.add_argument('--task_switch_period', type=int, default=0)
    # ==========================================================

    # ===================== Hierarchical control =====================
    parser.add_argument('--cp_path', type=str, default=None)
    parser.add_argument('--cp_action_range', type=float, default=1.5)
    parser.add_argument('--cp_unit_length', type=int, default=0, choices=[0, 1])
    parser.add_argument('--cp_multi_step', type=int, default=1)
    parser.add_argument('--cp_num_truncate_obs', type=int, default=0)
    # ===============================================================

    # ===================== CBP flags =====================
    parser.add_argument('--cbp', type=int, default=0, choices=[0, 1])
    parser.add_argument('--cbp_rr', type=float, default=1e-4)
    parser.add_argument('--cbp_mt', type=int, default=1000)
    parser.add_argument(
        '--cbp_init',
        type=str,
        default='kaiming',
        choices=['default', 'xavier', 'lecun', 'kaiming'])
    parser.add_argument(
        '--cbp_util',
        type=str,
        default='contribution',
        choices=['contribution', 'weight', 'random'])
    parser.add_argument('--cbp_decay', type=float, default=0.0)
    parser.add_argument('--cbp_rr_policy', type=float, default=None)
    parser.add_argument('--cbp_rr_q', type=float, default=None)
    parser.add_argument('--cbp_rr_te', type=float, default=None)
    parser.add_argument('--cbp_mt_policy', type=int, default=None)
    parser.add_argument('--cbp_mt_q', type=int, default=None)
    parser.add_argument('--cbp_mt_te', type=int, default=None)
    # ==============================================================

    parser.add_argument('--common_lr', type=float, default=1e-4)
    parser.add_argument('--lr_op', type=float, default=None)
    parser.add_argument('--lr_te', type=float, default=None)

    parser.add_argument('--alpha', type=float, default=0.01)

    parser.add_argument(
        '--pretrain_algo',
        type=str,
        default='metra',
        choices=['metra', 'dads'])

    parser.add_argument(
        '--hierarchical_algo',
        type=str,
        default='ppo',
        choices=['ppo'])

    parser.add_argument('--sac_tau', type=float, default=5e-3)
    parser.add_argument('--sac_lr_q', type=float, default=None)
    parser.add_argument('--sac_lr_a', type=float, default=None)
    parser.add_argument('--sac_discount', type=float, default=0.99)
    parser.add_argument('--sac_scale_reward', type=float, default=1.)
    parser.add_argument('--sac_target_coef', type=float, default=1.)
    parser.add_argument('--sac_min_buffer_size', type=int, default=1000)
    parser.add_argument('--sac_max_buffer_size', type=int, default=3000)

    parser.add_argument('--spectral_normalization', type=int, default=0, choices=[0, 1])

    parser.add_argument('--model_master_dim', type=int, default=1024)
    parser.add_argument('--model_master_num_layers', type=int, default=2)
    parser.add_argument(
        '--model_master_nonlinearity',
        type=str,
        default=None,
        choices=['relu', 'tanh'])
    parser.add_argument('--sd_const_std', type=int, default=1)
    parser.add_argument('--sd_batch_norm', type=int, default=1, choices=[0, 1])

    parser.add_argument('--num_alt_samples', type=int, default=100)
    parser.add_argument('--split_group', type=int, default=65536)

    parser.add_argument('--discrete', type=int, default=0, choices=[0, 1])
    parser.add_argument('--inner', type=int, default=1, choices=[0, 1])
    parser.add_argument('--unit_length', type=int, default=1, choices=[0, 1])

    parser.add_argument('--dual_reg', type=int, default = 1, choices=[0, 1])
    parser.add_argument('--dual_lam', type=float, default=30)
    parser.add_argument('--dual_slack', type=float, default=1e-3)
    parser.add_argument(
        '--dual_dist',
        type=str,
        default='one',
        choices=['l2', 's2_from_s', 'one'])
    parser.add_argument('--dual_lr', type=float, default=None)

    return parser


def get_selected_algo(args, mode=None):
    local_mode = args.mode if mode is None else mode

    if local_mode == 'pretrain':
        return args.pretrain_algo
    elif local_mode == 'hierarchical':
        return args.hierarchical_algo
    elif local_mode == 'alternating':
        raise ValueError(
            'alternating is a meta-mode; use explicit phase mode '
            "('pretrain' or 'hierarchical') when selecting an algorithm"
        )
    else:
        raise ValueError(f'Unknown mode: {local_mode}')


def get_effective_cp_path(args, mode=None, cp_path=None):
    local_mode = args.mode if mode is None else mode
    local_cp_path = args.cp_path if cp_path is None else cp_path

    # If a directory is given, find latest option_policy*.pt inside it
    if local_cp_path is not None and os.path.isdir(local_cp_path):
        policies = sorted(
            glob.glob(os.path.join(local_cp_path, "option_policy*.pt"))
        )
        if len(policies) == 0:
            raise RuntimeError(f"No option_policy*.pt found in {local_cp_path}")
        local_cp_path = policies[-1]
        print("Using latest policy from directory:", local_cp_path)

    # Fallback: auto-find latest run
    if local_mode == "hierarchical" and local_cp_path is None:
        local_cp_path = find_latest_policy()
        print("Using latest pretrained policy:", local_cp_path)

    return local_cp_path


args = get_argparser().parse_args()
g_start_time = int(datetime.datetime.now().timestamp())


def unwrap_env(e):
    while hasattr(e, 'env'):
        e = e.env
    return e


def task_switch_fn(env, task_id):
    base = unwrap_env(env)

    scales = [0.5, 1.0, 1.5, 2.0]
    scale = scales[task_id % len(scales)]

    model = None
    if hasattr(base, "model"):
        model = base.model
    elif hasattr(base, "sim") and hasattr(base.sim, "model"):
        model = base.sim.model

    if model is None or not hasattr(model, "geom_friction"):
        return

    fr = model.geom_friction.copy()
    fr[:, 0] = fr[:, 0] * scale
    model.geom_friction[:] = fr


def task_params_fn(env):
    return np.array([0.0], dtype=np.float32)


def get_exp_name(mode=None):
    local_mode = args.mode if mode is None else mode

    if local_mode == 'alternating':
        selected_algo = f'{args.pretrain_algo}_to_{args.hierarchical_algo}'
    else:
        selected_algo = get_selected_algo(args, mode=local_mode)

    exp_name = ''
    exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name += '_' + args.env
    exp_name += '_' + selected_algo

    if args.lifelong and args.pretrain_algo == 'metra':
        exp_name += f'_lifelong_ctx{args.dim_context}'

    return exp_name, exp_name_prefix

def get_log_dir():
    exp_name, _ = get_exp_name()

    if hasattr(os, "pathconf"):
        try:
            assert len(exp_name) <= os.pathconf('/', 'PC_NAME_MAX')
        except (OSError, ValueError):
            pass

    group = os.path.join(args.run_group, "pretraining")
    if args.mode == "pretrain":
        group = "pretraining"
    elif args.mode == "hierarchical":
        group = "hierarchical_control"
    elif args.mode == "alternating":
        group = "alternating"

    log_dir = os.path.realpath(os.path.join(EXP_DIR, group, exp_name))
    assert not os.path.exists(log_dir), f'The following path already exists: {log_dir}'
    return log_dir


def get_gaussian_module_construction(args,
                                     *,
                                     hidden_sizes,
                                     const_std=False,
                                     hidden_nonlinearity=torch.relu,
                                     w_init=torch.nn.init.xavier_uniform_,
                                     init_std=1.0,
                                     min_std=1e-6,
                                     max_std=None,
                                     **kwargs):
    module_kwargs = dict()
    if const_std:
        module_cls = GaussianMLPModuleEx
        module_kwargs.update(dict(
            learn_std=False,
            init_std=init_std,
        ))
    else:
        module_cls = GaussianMLPIndependentStdModuleEx
        module_kwargs.update(dict(
            std_hidden_sizes=hidden_sizes,
            std_hidden_nonlinearity=hidden_nonlinearity,
            std_hidden_w_init=w_init,
            std_output_w_init=w_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
        ))

    module_kwargs.update(dict(
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=w_init,
        output_w_init=w_init,
        std_parameterization='exp',
        bias=True,
        spectral_normalization=args.spectral_normalization,
        **kwargs,
    ))
    return module_cls, module_kwargs


def make_env(args, max_path_length, mode=None, cp_path=None):
    local_mode = args.mode if mode is None else mode
    local_cp_path = get_effective_cp_path(args, mode=local_mode, cp_path=cp_path)

    if args.env == 'maze':
        from envs.maze_env import MazeEnv
        env = MazeEnv(
            max_path_length=max_path_length,
            action_range=0.2,
        )
    elif args.env == 'humanoid':
        from GymnasiumConv import HumanoidEnv
        env = HumanoidEnv(render_hw=100)

    elif args.env == 'half_cheetah':
        from envs.mujoco.half_cheetah_env import HalfCheetahEnv
        env = HalfCheetahEnv(render_hw=100)

    elif args.env == 'ant':
        from GymnasiumConv import AntEnv
        env = AntEnv(render_hw=100)

    elif args.env.startswith('dmc'):
        from envs.custom_dmc_tasks import dmc
        from envs.custom_dmc_tasks.pixel_wrappers import RenderWrapper
        assert args.encoder
        if args.env == 'dmc_cheetah':
            env = dmc.make(
                'cheetah_run_forward_color',
                obs_type='states',
                frame_stack=1,
                action_repeat=2,
                seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_quadruped':
            env = dmc.make(
                'quadruped_run_forward_color',
                obs_type='states',
                frame_stack=1,
                action_repeat=2,
                seed=args.seed)
            env = RenderWrapper(env)
        elif args.env == 'dmc_humanoid':
            env = dmc.make(
                'humanoid_run_color',
                obs_type='states',
                frame_stack=1,
                action_repeat=2,
                seed=args.seed)
            env = RenderWrapper(env)
        else:
            raise NotImplementedError
    elif args.env == 'kitchen':
        sys.path.append('lexa')
        from envs.lexa.mykitchen import MyKitchenEnv
        assert args.encoder
        env = MyKitchenEnv(log_per_goal=True)
    else:
        raise NotImplementedError

    if args.frame_stack is not None:
        from envs.custom_dmc_tasks.pixel_wrappers import FrameStackWrapper
        env = FrameStackWrapper(env, args.frame_stack)

    normalizer_type = args.normalizer_type
    normalizer_kwargs = {}

    if normalizer_type == 'off':
        env = consistent_normalize(env, normalize_obs=False, **normalizer_kwargs)
    elif normalizer_type == 'preset':
        normalizer_name = args.env
        normalizer_mean, normalizer_std = get_normalizer_preset(f'{normalizer_name}_preset')
        env = consistent_normalize(
            env,
            normalize_obs=True,
            mean=normalizer_mean,
            std=normalizer_std,
            **normalizer_kwargs)

    if local_mode == "hierarchical":
        if local_cp_path is None:
            local_cp_path = find_latest_policy()
            print("Using latest pretrained policy:", local_cp_path)

        cp_dict = torch.load(local_cp_path, map_location='cpu')
        env = ChildPolicyEnv(
            env=env,
            cp_dict=cp_dict,
            cp_action_range=args.cp_action_range,
            cp_unit_length=args.cp_unit_length,
            cp_multi_step=args.cp_multi_step,
            cp_num_truncate_obs=args.cp_num_truncate_obs,
        )

    return env


def _act_type_from_args(args):
    if args.model_master_nonlinearity == "tanh":
        return "tanh"
    return "relu"


def _cbp_kwargs(rr, mt):
    return dict(
        replacement_rate=rr,
        maturity_threshold=mt,
        init=args.cbp_init,
        act_type=_act_type_from_args(args),
        util_type=args.cbp_util,
        decay_rate=args.cbp_decay,
    )


def _finalize_lr(args, lr):
    if lr is None:
        lr = args.common_lr
    else:
        assert bool(lr), 'To specify a lr of 0, use a negative value'
    if lr < 0.0:
        dowel.logger.log(f'Setting lr to ZERO given {lr}')
        lr = 0.0
    return lr


def save_option_policy_for_hierarchical(algo, save_dir, cycle):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'option_policy_cycle_{cycle:06d}.pt')

    payload = {
        'discrete': algo.discrete,
        'dim_option': algo.dim_option,
        'policy': algo.option_policy,
        'dim_context': getattr(algo, 'dim_context', 0),
        'use_context_in_phi': getattr(algo, 'use_context_in_phi', False),
        'deterministic_rollout_context': getattr(algo, 'deterministic_rollout_context', True),
    }

    if getattr(algo, 'dim_context', 0) > 0:
        payload.update({
            'context_encoder': getattr(algo, 'context_encoder', None),
            'context_prior_net': getattr(algo, 'context_prior_net', None),
            'context_decoder': getattr(algo, 'context_decoder', None),
        })

    torch.save(payload, save_path)
    return save_path

def build_runner_and_algo(args, ctxt=None, mode=None, cp_path=None):

    local_mode = args.mode if mode is None else mode
    selected_algo = get_selected_algo(args, mode=local_mode)
    local_cp_path = get_effective_cp_path(args, mode=local_mode, cp_path=cp_path)
    runner = OptionLocalRunner(ctxt)

    max_path_length = args.max_path_length
    if local_cp_path is not None and local_mode == 'hierarchical':
        assert args.cp_multi_step > 0
        max_path_length = max(1, args.max_path_length // args.cp_multi_step)

    contextualized_make_env = functools.partial(
        make_env,
        args=args,
        max_path_length=max_path_length,
        mode=local_mode,
        cp_path=local_cp_path,
    )

    env = contextualized_make_env()
    example_ob = env.reset()

    if args.encoder:
        pixel_shape = None
        if hasattr(env, 'ob_info'):
            if env.ob_info['type'] in ['hybrid', 'pixel']:
                pixel_shape = env.ob_info['pixel_shape']
        if pixel_shape is None:
            pixel_shape = (64, 64, 3)
    else:
        pixel_shape = None

    device = torch.device('cuda' if args.use_gpu else 'cpu')
    master_dims = [args.model_master_dim] * args.model_master_num_layers

    if args.model_master_nonlinearity == 'relu':
        nonlinearity = torch.relu
    elif args.model_master_nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        nonlinearity = None

    obs_space = env.spec.observation_space
    action_space = env.spec.action_space

    obs_dim = getattr(obs_space, 'flat_dim', int(np.prod(obs_space.shape)))
    action_dim = getattr(action_space, 'flat_dim', int(np.prod(action_space.shape)))

    if args.encoder:
        def make_encoder(**kwargs):
            return Encoder(pixel_shape=pixel_shape, **kwargs)

        def with_encoder(module, encoder=None):
            if encoder is None:
                encoder = make_encoder()
            return WithEncoder(encoder=encoder, module=module)

        example_encoder = make_encoder()
        module_obs_dim = example_encoder(
            torch.as_tensor(example_ob).float().unsqueeze(0)).shape[-1]
    else:
        make_encoder = None
        with_encoder = None
        module_obs_dim = obs_dim

    if selected_algo == 'metra' and args.lifelong and args.dim_context > 0:
        from iod.context_models import ContextEncoder, ContextPrior, ContextDecoder

        context_encoder = ContextEncoder(
            obs_dim=module_obs_dim,
            act_dim=action_dim,
            z_dim=args.dim_context,
            hidden=args.context_hidden,
        )
        context_prior = ContextPrior(
            z_dim=args.dim_context,
            hidden=args.context_hidden,
        )
        context_decoder = ContextDecoder(
            obs_dim=module_obs_dim,
            act_dim=action_dim,
            z_dim=args.dim_context,
            hidden=args.context_hidden,
        )
    else:
        context_encoder, context_prior, context_decoder = None, None, None

    option_info = {
        'dim_option': args.dim_option,
    }

    policy_kwargs = dict(
        name='option_policy',
        option_info=option_info,
    )
    module_kwargs = dict(
        hidden_sizes=master_dims,
        layer_normalization=False,
    )
    if nonlinearity is not None:
        module_kwargs.update(hidden_nonlinearity=nonlinearity)

    module_cls = GaussianMLPTwoHeadedModuleEx
    module_kwargs.update(dict(
        max_std=np.exp(2.),
        normal_distribution_cls=TanhNormal,
        output_w_init=functools.partial(xavier_normal_ex, gain=1.),
        init_std=1.,
    ))

    context_dim = args.dim_context if args.lifelong else 0

    if selected_algo == 'ppo' and local_mode == 'hierarchical' and local_cp_path is not None:
        cp_meta = torch.load(local_cp_path, map_location='cpu')
        context_dim = cp_meta.get('dim_context', context_dim)

    if selected_algo == 'ppo':
        policy_q_input_dim = module_obs_dim
    else:
        policy_q_input_dim = module_obs_dim + args.dim_option + context_dim
        
    policy_module = module_cls(
        input_dim=policy_q_input_dim,
        output_dim=action_dim,
        **module_kwargs
    )
    if args.encoder:
        policy_module = with_encoder(policy_module)

    policy_kwargs['module'] = policy_module
    option_policy = PolicyEx(**policy_kwargs)

    traj_encoder = None
    if selected_algo in ['metra', 'dads']:
        traj_encoder_obs_dim = module_obs_dim + context_dim

        module_cls, module_kwargs = get_gaussian_module_construction(
            args,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            w_init=torch.nn.init.xavier_uniform_,
            input_dim=traj_encoder_obs_dim,
            output_dim=args.dim_option,
        )
        traj_encoder = module_cls(**module_kwargs)
        if args.encoder:
            if args.spectral_normalization:
                te_encoder = make_encoder(spectral_normalization=True)
            else:
                te_encoder = None
            traj_encoder = with_encoder(traj_encoder, encoder=te_encoder)

    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        w_init=torch.nn.init.xavier_uniform_,
        input_dim=obs_dim,
        output_dim=obs_dim,
        min_std=1e-6,
        max_std=1e6,
    )
    if selected_algo in ['metra', 'dads'] and args.dual_dist == 's2_from_s':
        dist_predictor = module_cls(**module_kwargs)
    else:
        dist_predictor = None

    dual_lam = ParameterModule(torch.Tensor([np.log(args.dual_lam)]))

    sd_dim_option = args.dim_option
    skill_dynamics_obs_dim = obs_dim
    skill_dynamics_input_dim = skill_dynamics_obs_dim + sd_dim_option
    module_cls, module_kwargs = get_gaussian_module_construction(
        args,
        const_std=args.sd_const_std,
        hidden_sizes=master_dims,
        hidden_nonlinearity=nonlinearity or torch.relu,
        input_dim=skill_dynamics_input_dim,
        output_dim=skill_dynamics_obs_dim,
        min_std=0.3,
        max_std=10.0,
    )
    if selected_algo == 'dads':
        skill_dynamics = module_cls(**module_kwargs)
    else:
        skill_dynamics = None

    replay_buffer = PathBufferEx(
        capacity_in_transitions=int(args.sac_max_buffer_size),
        pixel_shape=pixel_shape
    )

    qf1 = qf2 = log_alpha = None
    vf = None
    if selected_algo in ['metra', 'dads']:
        qf1 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf1 = with_encoder(qf1)

        qf2 = ContinuousMLPQFunctionEx(
            obs_dim=policy_q_input_dim,
            action_dim=action_dim,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
        )
        if args.encoder:
            qf2 = with_encoder(qf2)

        log_alpha = ParameterModule(torch.Tensor([np.log(args.alpha)]))

    elif selected_algo == 'ppo':
        vf = MLPModule(
            input_dim=policy_q_input_dim,
            output_dim=1,
            hidden_sizes=master_dims,
            hidden_nonlinearity=nonlinearity or torch.relu,
            layer_normalization=False,
        )

    if args.cbp:
        rr_pol = args.cbp_rr if args.cbp_rr_policy is None else args.cbp_rr_policy
        rr_q = args.cbp_rr if args.cbp_rr_q is None else args.cbp_rr_q
        rr_te = (args.cbp_rr * 0.1) if args.cbp_rr_te is None else args.cbp_rr_te

        mt_pol = args.cbp_mt if args.cbp_mt_policy is None else args.cbp_mt_policy
        mt_q = args.cbp_mt if args.cbp_mt_q is None else args.cbp_mt_q
        mt_te = (args.cbp_mt * 2) if args.cbp_mt_te is None else args.cbp_mt_te

        print("== Enabling CBP ==")
        print({
            "rr_pol": rr_pol,
            "rr_q": rr_q,
            "rr_te": rr_te,
            "mt_pol": mt_pol,
            "mt_q": mt_q,
            "mt_te": mt_te
        })

        n_pol = add_cbp_to_model(option_policy, **_cbp_kwargs(rr_pol, mt_pol))
        n_te = 0 if traj_encoder is None else add_cbp_to_model(
            traj_encoder, **_cbp_kwargs(rr_te, mt_te))

        n_ctx_enc = 0
        n_ctx_prior = 0

        if context_encoder is not None:
            n_ctx_enc = add_cbp_to_model(
                context_encoder,
                **_cbp_kwargs(rr_te, mt_te)
            )

        if context_prior is not None:
            n_ctx_prior = add_cbp_to_model(
                context_prior,
                **_cbp_kwargs(rr_te, mt_te)
            )

        n_q1 = n_q2 = n_vf = 0
        if qf1 is not None:
            n_q1 = add_cbp_to_model(qf1, **_cbp_kwargs(rr_q, mt_q))
            n_q2 = add_cbp_to_model(qf2, **_cbp_kwargs(rr_q, mt_q))

        if vf is not None:
            n_vf = add_cbp_to_model(vf, **_cbp_kwargs(rr_q, mt_q))

        print("CBP patched sequentials:", dict(
            policy=n_pol,
            traj_encoder=n_te,
            qf1=n_q1,
            qf2=n_q2,
            vf=n_vf,
            context_encoder=n_ctx_enc,
            context_prior=n_ctx_prior,

        ))

    optimizers = {
        'option_policy': torch.optim.Adam([
            {'params': option_policy.parameters(), 'lr': _finalize_lr(args, args.lr_op)},
        ]),
        'dual_lam': torch.optim.Adam([
            {'params': dual_lam.parameters(), 'lr': _finalize_lr(args, args.dual_lr)},
        ]),
    }

    if traj_encoder is not None:
        optimizers['traj_encoder'] = torch.optim.Adam([
            {'params': traj_encoder.parameters(), 'lr': _finalize_lr(args, args.lr_te)},
        ])

    if context_encoder is not None:
        context_params = (
            list(context_encoder.parameters()) +
            list(context_prior.parameters())
        )
        if context_decoder is not None:
            context_params += list(context_decoder.parameters())

        optimizers['context'] = torch.optim.Adam(
            context_params,
            lr=_finalize_lr(args, args.context_lr),
        )

    if skill_dynamics is not None:
        optimizers['skill_dynamics'] = torch.optim.Adam([
            {'params': skill_dynamics.parameters(), 'lr': _finalize_lr(args, args.lr_te)},
        ])

    if dist_predictor is not None:
        optimizers['dist_predictor'] = torch.optim.Adam([
            {'params': dist_predictor.parameters(), 'lr': _finalize_lr(args, args.lr_op)},
        ])

    if qf1 is not None:
        optimizers['qf'] = torch.optim.Adam([
            {'params': list(qf1.parameters()) + list(qf2.parameters()),
             'lr': _finalize_lr(args, args.sac_lr_q)},
        ])
        optimizers['log_alpha'] = torch.optim.Adam([
            {'params': log_alpha.parameters(), 'lr': _finalize_lr(args, args.sac_lr_a)},
        ])

    if vf is not None:
        optimizers['vf'] = torch.optim.Adam([
            {'params': vf.parameters(), 'lr': _finalize_lr(args, args.lr_op)},
        ])

    optimizer = OptimizerGroupWrapper(
        optimizers=optimizers,
        max_optimization_epochs=None,
    )

    algo_name = {
    'metra': 'LifelongMETRA' if args.lifelong and selected_algo == 'metra' else 'METRA',
    'dads': 'DADS',
    'ppo': 'PPO',
    }[selected_algo]

    algo_kwargs = dict(
        env_name=args.env,
        algo=selected_algo,
        env_spec=env.spec,
        option_policy=option_policy,
        traj_encoder=traj_encoder,
        skill_dynamics=skill_dynamics,
        dist_predictor=dist_predictor,
        dual_lam=dual_lam,
        optimizer=optimizer,
        alpha=args.alpha,
        max_path_length=args.max_path_length,
        total_timesteps=args.total_timesteps,
        n_epochs_per_eval=args.n_epochs_per_eval,
        n_epochs_per_log=args.n_epochs_per_log,
        n_epochs_per_tb=args.n_epochs_per_log,
        n_epochs_per_save=args.n_epochs_per_save,
        n_epochs_per_pt_save=args.n_epochs_per_pt_save,
        n_epochs_per_pkl_update=(
            args.n_epochs_per_eval
            if args.n_epochs_per_pkl_update is None
            else args.n_epochs_per_pkl_update
        ),
        dim_option=args.dim_option,
        num_random_trajectories=args.num_random_trajectories,
        num_video_repeats=args.num_video_repeats,
        eval_record_video=args.eval_record_video,
        video_skip_frames=args.video_skip_frames,
        eval_plot_axis=args.eval_plot_axis,
        name=algo_name,
        device=device,
        sample_cpu=args.sample_cpu,
        num_train_per_epoch=1,
        sd_batch_norm=args.sd_batch_norm,
        skill_dynamics_obs_dim=skill_dynamics_obs_dim,
        trans_minibatch_size=args.trans_minibatch_size,
        trans_optimization_epochs=args.trans_optimization_epochs,
        discount=args.sac_discount,
        discrete=args.discrete,
        unit_length=args.unit_length,
    )

    skill_common_args = dict(
        qf1=qf1,
        qf2=qf2,
        log_alpha=log_alpha,
        tau=args.sac_tau,
        scale_reward=args.sac_scale_reward,
        target_coef=args.sac_target_coef,
        replay_buffer=replay_buffer,
        min_buffer_size=args.sac_min_buffer_size,
        inner=args.inner,
        num_alt_samples=args.num_alt_samples,
        split_group=args.split_group,
        dual_reg=args.dual_reg,
        dual_slack=args.dual_slack,
        dual_dist=args.dual_dist,
        pixel_shape=pixel_shape,
    )

    if selected_algo == 'metra':
        if args.lifelong:
            algo = LifelongMETRA(
            **algo_kwargs,
            **skill_common_args,
            dim_context=args.dim_context,
            use_context_in_phi=True,
            task_switch_period=args.task_switch_period,
            context_encoder=context_encoder,
            context_prior_net=context_prior,
            context_decoder=context_decoder,
            context_kl_coef=args.context_kl_coef,
            recon_coef=1.0,
            context_updates_per_epoch=args.context_updates_per_epoch,
            context_batch_episodes=args.context_batch_episodes,
            context_replay_size=args.context_replay_size,
        )
        else:
            algo = METRA(
                **algo_kwargs,
                **skill_common_args,
            )
    elif selected_algo == 'dads':
        algo = DADS(
            **algo_kwargs,
            **skill_common_args,
        )
    elif selected_algo == 'ppo':
        algo = PPO(
            **algo_kwargs,
            vf=vf,
            gae_lambda=0.95,
            ppo_clip=0.2,
        )
    else:
        raise NotImplementedError

    if args.sample_cpu:
        algo.option_policy.cpu()
    else:
        algo.option_policy.to(device)

    runner.setup(
        algo=algo,
        env=env,
        make_env=contextualized_make_env,
        sampler_cls=OptionMultiprocessingSampler,
        sampler_args=dict(n_thread=args.n_thread),
        n_workers=args.n_parallel,
    )

    algo.option_policy.to(device)

    return {
        'runner': runner,
        'algo': algo,
        'env': env,
        'cp_path': local_cp_path,
        'device': device,
    }


def run_alternating(ctxt=None):
    block = args.alternate_every
    assert block > 0, '--alternate_every must be > 0'

    n_cycles = args.n_epochs // (2 * block)
    if n_cycles <= 0:
        raise ValueError(
            f'n_epochs={args.n_epochs} too small for alternating with '
            f'alternate_every={block}. Need at least {2 * block}.'
        )

    dowel.logger.log(
        f'Alternating mode enabled: {n_cycles} cycles, '
        f'{block} pretrain epochs + {block} hierarchical epochs per cycle.'
    )

    run_dir = ctxt.snapshot_dir
    pretrain_snapshot_dir = os.path.join(run_dir, 'pretrain_snapshots')
    hier_snapshot_dir = os.path.join(run_dir, 'hierarchical_snapshots')
    alt_cp_dir = os.path.join(run_dir, 'alternating_cp_exports')

    os.makedirs(pretrain_snapshot_dir, exist_ok=True)
    os.makedirs(hier_snapshot_dir, exist_ok=True)
    os.makedirs(alt_cp_dir, exist_ok=True)

    pretrain_resume_dir = None

    for cycle in range(n_cycles):
        dowel.logger.log(
            f'=== Alternating cycle {cycle + 1}/{n_cycles}: PRETRAIN ===')

        pretrain_bundle = build_runner_and_algo(
            args=args,
            ctxt=ctxt,
            mode='pretrain',
            cp_path=None,
        )
        pretrain_bundle['runner']._snapshotter._snapshot_dir = pretrain_snapshot_dir

        try:
            if pretrain_resume_dir is not None:
                train_args = pretrain_bundle['runner'].restore(
                    from_dir=pretrain_resume_dir,
                    make_env=pretrain_bundle['runner']._make_env,
                    from_epoch='last',
                )
                target_epoch = train_args.start_epoch + block
                pretrain_bundle['runner'].resume_train(
                    n_epochs=target_epoch,
                    batch_size=args.traj_batch_size,
                )
            else:
                pretrain_bundle['runner'].train(
                    n_epochs=block,
                    batch_size=args.traj_batch_size,
                )

            pretrain_bundle['runner'].save(cycle + 1, pkl_update=True)

            cp_path = save_option_policy_for_hierarchical(
                pretrain_bundle['algo'],
                alt_cp_dir,
                cycle,
            )
            dowel.logger.log(f'Saved child policy checkpoint: {cp_path}')

            pretrain_resume_dir = pretrain_snapshot_dir

        finally:
            pretrain_bundle['runner']._shutdown_worker()
            del pretrain_bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dowel.logger.log(
            f'=== Alternating cycle {cycle + 1}/{n_cycles}: HIERARCHICAL ===')

        hierarchical_bundle = build_runner_and_algo(
            args=args,
            ctxt=ctxt,
            mode='hierarchical',
            cp_path=cp_path,
        )
        cur_hier_dir = os.path.join(hier_snapshot_dir, f'cycle_{cycle:06d}')
        os.makedirs(cur_hier_dir, exist_ok=True)
        hierarchical_bundle['runner']._snapshotter._snapshot_dir = cur_hier_dir

        try:
            hierarchical_bundle['runner'].train(
                n_epochs=block,
                batch_size=args.traj_batch_size,
            )
        finally:
            hierarchical_bundle['runner']._shutdown_worker()
            del hierarchical_bundle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

@wrap_experiment(log_dir=get_log_dir(), name=get_exp_name()[0])
def run(ctxt=None):
    if 'WANDB_API_KEY' in os.environ:
        wandb_output_dir = tempfile.mkdtemp()
        wandb.init(
            project='metra',
            entity='',
            group=args.run_group,
            name=get_exp_name()[0],
            config=vars(args),
            dir=wandb_output_dir)

    dowel.logger.log('ARGS: ' + str(args))

    if args.n_thread is not None:
        torch.set_num_threads(args.n_thread)

    set_seed(args.seed)

    if args.mode == 'alternating':
        run_alternating(ctxt)
        return

    if args.resume_from_dir is not None:
        local_mode = args.mode
        local_cp_path = args.cp_path
        effective_cp_path = get_effective_cp_path(args, mode=local_mode, cp_path=local_cp_path)

        max_path_length = args.max_path_length
        if effective_cp_path is not None and local_mode == 'hierarchical':
            max_path_length *= args.cp_multi_step

        contextualized_make_env = functools.partial(
            make_env,
            args=args,
            max_path_length=max_path_length,
            mode=local_mode,
            cp_path=local_cp_path,
        )

        runner = OptionLocalRunner(ctxt)
        train_args = runner.restore(
        from_dir=args.resume_from_dir,
        make_env=contextualized_make_env,
        from_epoch=args.resume_from_epoch,
        )
        train_args.n_epochs = args.n_epochs

        runner.resume_train(
        n_epochs=args.n_epochs,
        batch_size=train_args.batch_size,
        store_paths=train_args.store_paths,
        pause_for_plot=train_args.pause_for_plot,
    )
        return

    bundle = build_runner_and_algo(
        args=args,
        ctxt=ctxt,
        mode=args.mode,
        cp_path=args.cp_path,
    )

    bundle['runner'].train(
        n_epochs=args.n_epochs,
        batch_size=args.traj_batch_size,
    )

if __name__ == '__main__':
    mp.set_start_method(START_METHOD)
    run()