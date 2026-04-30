from collections import defaultdict, deque
from unittest import runner

import numpy as np
import torch

import global_context
import dowel_wrapper
from dowel import Histogram
from garage import TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos.rl_algorithm import RLAlgorithm
from garagei import log_performance_ex
from garagei.torch.optimizers.optimizer_group_wrapper import OptimizerGroupWrapper
from garagei.torch.utils import compute_total_norm
from iod.utils import MeasureAndAccTime


class IOD(RLAlgorithm):
    def __init__(
            self,
            *,
            env_name,
            algo,
            env_spec,
            option_policy,
            traj_encoder,
            skill_dynamics,
            dist_predictor,
            dual_lam,
            optimizer,
            alpha,
            max_path_length,
            n_epochs_per_eval,
            n_epochs_per_log,
            n_epochs_per_tb,
            n_epochs_per_save,
            n_epochs_per_pt_save,
            n_epochs_per_pkl_update,
            dim_option,
            num_random_trajectories,
            num_video_repeats,
            eval_record_video,
            video_skip_frames,
            eval_plot_axis,
            name='IOD',
            total_timesteps = None,
            device=torch.device('cpu'),
            sample_cpu=True,
            num_train_per_epoch=1,
            discount=0.99,
            sd_batch_norm=False,
            skill_dynamics_obs_dim=None,
            trans_minibatch_size=None,
            trans_optimization_epochs=None,
            discrete=False,
            unit_length=False,
    ):
        self.env_name = env_name
        self.algo = algo

        self.discount = discount
        self.max_path_length = max_path_length

        self.device = device
        self.sample_cpu = sample_cpu
        self.option_policy = option_policy.to(self.device)
        self.traj_encoder = traj_encoder.to(self.device) if traj_encoder is not None else None
        self.dual_lam = dual_lam.to(self.device)
        self.param_modules = {
            'traj_encoder': self.traj_encoder,
            'option_policy': self.option_policy,
            'dual_lam': self.dual_lam,
        }
        if skill_dynamics is not None:
            self.skill_dynamics = skill_dynamics.to(self.device)
            self.param_modules['skill_dynamics'] = self.skill_dynamics
        if dist_predictor is not None:
            self.dist_predictor = dist_predictor.to(self.device)
            self.param_modules['dist_predictor'] = self.dist_predictor

        self.alpha = alpha
        self.name = name

        self.dim_option = dim_option
        self.total_timesteps = total_timesteps

        self._num_train_per_epoch = num_train_per_epoch
        self._env_spec = env_spec

        self.n_epochs_per_eval = n_epochs_per_eval
        self.n_epochs_per_log = n_epochs_per_log
        self.n_epochs_per_tb = n_epochs_per_tb
        self.n_epochs_per_save = n_epochs_per_save
        self.n_epochs_per_pt_save = n_epochs_per_pt_save
        self.n_epochs_per_pkl_update = n_epochs_per_pkl_update
        self.num_random_trajectories = num_random_trajectories
        self.num_video_repeats = num_video_repeats
        self.eval_record_video = eval_record_video
        self.video_skip_frames = video_skip_frames
        self.eval_plot_axis = eval_plot_axis

        assert isinstance(optimizer, OptimizerGroupWrapper)
        self._optimizer = optimizer

        self._sd_batch_norm = sd_batch_norm
        self._skill_dynamics_obs_dim = skill_dynamics_obs_dim

        if self._sd_batch_norm:
            self._sd_input_batch_norm = torch.nn.BatchNorm1d(
                self._skill_dynamics_obs_dim, momentum=0.01
            ).to(self.device)
            self._sd_target_batch_norm = torch.nn.BatchNorm1d(
                self._skill_dynamics_obs_dim, momentum=0.01, affine=False
            ).to(self.device)
            self._sd_input_batch_norm.eval()
            self._sd_target_batch_norm.eval()

        self._trans_minibatch_size = trans_minibatch_size
        self._trans_optimization_epochs = trans_optimization_epochs

        self.discrete = discrete
        self.unit_length = unit_length

        if self.traj_encoder is not None:
            self.traj_encoder.eval()

    @property
    def policy(self):
        raise NotImplementedError()

    def all_parameters(self):
        for m in self.param_modules.values():
            if m is None:
                continue
            for p in m.parameters():
                yield p

    def _flatten_env_info_values(self, paths, key):
        values = []
        for path in paths:
            env_infos = path['env_infos']
            if key in env_infos:
                values.extend(np.asarray(env_infos[key]).reshape(-1).tolist())
        return values

    def _record_stats_if_any(self, out_dict, values, prefix):
        if len(values) == 0:
            return
        arr = np.asarray(values, dtype=np.float32)
        out_dict[f'{prefix}Mean'] = float(np.mean(arr))
        out_dict[f'{prefix}Std'] = float(np.std(arr))
        out_dict[f'{prefix}Max'] = float(np.max(arr))
        out_dict[f'{prefix}Min'] = float(np.min(arr))

    def _collect_hierarchical_metrics(self, paths):
        hier_metrics = {}

        metric_specs = [
            ('cp_action_norm', 'HierCpActionNorm'),
            ('low_level_reward_sum', 'HierLowLevelRewardSum'),
            ('low_level_reward_mean', 'HierLowLevelReward'),
            ('low_level_steps', 'HierLowLevelSteps'),
            ('low_level_action_norm_mean', 'HierLowLevelActionNorm'),
            ('context_norm', 'HierContextNorm'),
            ('posterior_context_norm', 'HierPosteriorContextNorm'),
            ('context_drift', 'HierContextDrift'),
        ]

        for env_key, metric_prefix in metric_specs:
            values = self._flatten_env_info_values(paths, env_key)
            self._record_stats_if_any(hier_metrics, values, metric_prefix)

        return hier_metrics

    def train_once(self, itr, paths, runner, extra_scalar_metrics=None):
            if extra_scalar_metrics is None:
                extra_scalar_metrics = {}

            logging_enabled = ((runner.step_itr + 1) % self.n_epochs_per_log == 0)

            data = self.process_samples(paths)

            hier_metrics = self._collect_hierarchical_metrics(paths)

            extra_scalar_metrics = dict(extra_scalar_metrics)
            extra_scalar_metrics.update(hier_metrics)

            time_computing_metrics = [0.0]
            time_training = [0.0]

            with MeasureAndAccTime(time_training):
                tensors = self._train_once_inner(data)

            performance = log_performance_ex(
                itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, paths),
                discount=self.discount,
            )
            discounted_returns = performance['discounted_returns']
            undiscounted_returns = performance['undiscounted_returns']

            if logging_enabled:
                prefix_tabular = global_context.get_metric_prefix()
                with dowel_wrapper.get_tabular().prefix(prefix_tabular + self.name + '/'), \
                        dowel_wrapper.get_tabular('plot').prefix(prefix_tabular + self.name + '/'):

                    def _record_scalar(key, val):
                        dowel_wrapper.get_tabular().record(key, val)

                    def _record_histogram(key, val):
                        dowel_wrapper.get_tabular('plot').record(key, Histogram(val))

                    for k, v in tensors.items():
                        if v.numel() == 1:
                            _record_scalar(k, v.item())
                        else:
                            _record_scalar(
                                k,
                                np.array2string(v.detach().cpu().numpy(), suppress_small=True)
                            )

                    with torch.no_grad():
                        total_norm = compute_total_norm(self.all_parameters())
                        _record_scalar('TotalGradNormAll', total_norm.item())

                        for key, module in self.param_modules.items():
                            if module is None:
                                continue
                            module_norm = compute_total_norm(module.parameters())
                            _record_scalar(
                                f'TotalGradNorm{key.replace("_", " ").title().replace(" ", "")}',
                                module_norm.item()
                            )

                    for k, v in extra_scalar_metrics.items():
                        _record_scalar(k, v)

                    _record_scalar('TimeComputingMetrics', time_computing_metrics[0])
                    _record_scalar('TimeTraining', time_training[0])

                    path_lengths = [len(path['actions']) for path in paths]
                    steps_this_epoch = sum(path_lengths)

                    if not hasattr(self, "total_env_steps"):
                        self.total_env_steps = 0

                    self.total_env_steps += steps_this_epoch
                    runner._stats.total_env_steps = self.total_env_steps
                    _record_scalar('StepsThisEpoch', steps_this_epoch)
                    _record_scalar('TotalEnvSteps', self.total_env_steps)
                    _record_scalar('PathLengthMean', float(np.mean(path_lengths)))
                    _record_scalar('PathLengthMax', float(np.max(path_lengths)))
                    _record_scalar('PathLengthMin', float(np.min(path_lengths)))

                    _record_histogram('ExternalDiscountedReturns', np.asarray(discounted_returns))
                    _record_histogram('ExternalUndiscountedReturns', np.asarray(undiscounted_returns))

            return float(np.mean(undiscounted_returns))
    
    def train(self, runner):
        last_return = None

        with global_context.GlobalContext({'phase': 'train', 'policy': 'sampling'}):
            for _ in runner.step_epochs(
                    full_tb_epochs=0,
                    log_period=self.n_epochs_per_log,
                    tb_period=self.n_epochs_per_tb,
                    pt_save_period=self.n_epochs_per_pt_save,
                    pkl_update_period=self.n_epochs_per_pkl_update,
                    new_save_period=self.n_epochs_per_save,
            ):
                for p in self.policy.values():
                    p.eval()
                if self.traj_encoder is not None:
                    self.traj_encoder.eval()

                if self.n_epochs_per_eval != 0 and runner.step_itr % self.n_epochs_per_eval == 0:
                    self._evaluate_policy(runner)

                for p in self.policy.values():
                    p.train()
                if self.traj_encoder is not None:
                    self.traj_encoder.train()

                for _ in range(self._num_train_per_epoch):
                    time_sampling = [0.0]
                    with MeasureAndAccTime(time_sampling):
                        runner.step_path = self._get_train_trajectories(runner)

                    if self.total_timesteps is not None and runner._stats.total_env_steps >= self.total_timesteps:
                        return last_return

                    last_return = self.train_once(
                        runner.step_itr,
                        runner.step_path,
                        runner,
                        extra_scalar_metrics={
                            'TimeSampling': time_sampling[0],
                        },
                    )

                runner.step_itr += 1

        return last_return

    def _get_trajectories(self,
                          runner,
                          sampler_key,
                          batch_size=None,
                          extras=None,
                          update_stats=False,
                          worker_update=None,
                          env_update=None):
        if batch_size is None:
            batch_size = len(extras)
        policy_sampler_key = sampler_key[6:] if sampler_key.startswith('local_') else sampler_key
        time_get_trajectories = [0.0]
        with MeasureAndAccTime(time_get_trajectories):
            trajectories, infos = runner.obtain_exact_trajectories(
                runner.step_itr,
                sampler_key=sampler_key,
                batch_size=batch_size,
                agent_update=self._get_policy_param_values(policy_sampler_key),
                env_update=env_update,
                worker_update=worker_update,
                extras=extras,
                update_stats=update_stats,
            )
        print(f'_get_trajectories({sampler_key}) {time_get_trajectories[0]}s')

        for traj in trajectories:
            for key in ['ori_obs', 'next_ori_obs', 'coordinates', 'next_coordinates']:
                if key not in traj['env_infos']:
                    continue

        return trajectories

    def _get_train_trajectories(self, runner):
        default_kwargs = dict(
            runner=runner,
            update_stats=True,
            worker_update=dict(
                _render=False,
                _deterministic_policy=False,
            ),
            env_update=dict(_action_noise_std=None),
        )
        kwargs = dict(default_kwargs, **self._get_train_trajectories_kwargs(runner))

        paths = self._get_trajectories(**kwargs)
        return paths

    def process_samples(self, paths):
        data = defaultdict(list)
        for path in paths:
            data['obs'].append(path['observations'])
            data['next_obs'].append(path['next_observations'])
            data['actions'].append(path['actions'])
            data['rewards'].append(path['rewards'])
            data['dones'].append(path['dones'])
            data['returns'].append(tensor_utils.discount_cumsum(path['rewards'], self.discount))
            data['ori_obs'].append(path['env_infos']['ori_obs'])
            data['next_ori_obs'].append(path['env_infos']['next_ori_obs'])

            if 'pre_tanh_value' in path['agent_infos']:
                data['pre_tanh_values'].append(path['agent_infos']['pre_tanh_value'])
            if 'log_prob' in path['agent_infos']:
                data['log_probs'].append(path['agent_infos']['log_prob'])

            if 'option' in path['agent_infos']:
                data['options'].append(path['agent_infos']['option'])
                data['next_options'].append(
                    np.concatenate(
                        [path['agent_infos']['option'][1:], path['agent_infos']['option'][-1:]],
                        axis=0
                    )
                )

            # ===================== [LIFELONG ADDITION] =====================
            # If OptionWorker extras include "context", it will be recorded in agent_infos.
            # We pass it through so METRA/LifelongMETRA can store it in replay and condition SAC on it.
            if 'context' in path['agent_infos']:
                data['context'].append(path['agent_infos']['context'])
                data['next_context'].append(
                    np.concatenate(
                        [path['agent_infos']['context'][1:], path['agent_infos']['context'][-1:]],
                        axis=0
                    )
                )
            # ===============================================================
            if 'cp_action_norm' in path['env_infos']:
                data['cp_action_norm'].append(path['env_infos']['cp_action_norm'])

            if 'low_level_reward_sum' in path['env_infos']:
                data['low_level_reward_sum'].append(path['env_infos']['low_level_reward_sum'])

            if 'low_level_steps' in path['env_infos']:
                data['low_level_steps'].append(path['env_infos']['low_level_steps'])

            if 'context_norm' in path['env_infos']:
                data['context_norm'].append(path['env_infos']['context_norm'])

            if 'context_drift' in path['env_infos']:
                data['context_drift'].append(path['env_infos']['context_drift'])

            if 'low_level_reward_mean' in path['env_infos']:
                data['low_level_reward_mean'].append(path['env_infos']['low_level_reward_mean'])

            if 'low_level_action_norm_mean' in path['env_infos']:
                data['low_level_action_norm_mean'].append(path['env_infos']['low_level_action_norm_mean'])

            if 'posterior_context_norm' in path['env_infos']:
                data['posterior_context_norm'].append(path['env_infos']['posterior_context_norm'])
        return data

    def _get_policy_param_values(self, key):
        param_dict = self.policy[key].get_param_values()
        for k in param_dict.keys():
            if self.sample_cpu:
                param_dict[k] = param_dict[k].detach().cpu()
            else:
                param_dict[k] = param_dict[k].detach()
        return param_dict

    def _generate_option_extras(self, options):
        return [{'option': option} for option in options]

    def _gradient_descent(self, loss, optimizer_keys):
        self._optimizer.zero_grad(keys=optimizer_keys)
        loss.backward()
        self._optimizer.step(keys=optimizer_keys)

    def _get_mini_tensors(self, epoch_data):
        num_transitions = len(epoch_data['actions'])
        idxs = np.random.choice(num_transitions, self._trans_minibatch_size)

        data = {}
        for key, value in epoch_data.items():
            data[key] = value[idxs]

        return data

    def _log_eval_metrics(self, runner):
        runner.eval_log_diagnostics()
        runner.plot_log_diagnostics()