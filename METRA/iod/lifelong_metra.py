# iod/lifelong_metra.py
import collections
import copy
import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD

from iod.context_models import gaussian_kl, reparameterize
from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians


class LifelongMETRA(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,
            task_switch_period,

            pixel_shape=None,

            dim_context=0,
            use_context_in_phi=False,

            context_encoder=None,       
            context_prior_net=None,      
            context_kl_coef=1.0,
            context_updates_per_epoch=10,
            context_batch_episodes=16,
            context_replay_size=5000,

            **kwargs,
    ):
        super().__init__(**kwargs)

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

        # Lifelong settings
        self.dim_context = int(dim_context)
        self.use_context_in_phi = bool(use_context_in_phi)
        self.task_switch_period = int(task_switch_period)

        # Route B modules
        self.context_encoder = context_encoder.to(self.device) if context_encoder is not None else None
        self.context_prior_net = context_prior_net.to(self.device) if context_prior_net is not None else None
        if self.context_encoder is not None:
            self.param_modules.update(context_encoder=self.context_encoder)
        if self.context_prior_net is not None:
            self.param_modules.update(context_prior_net=self.context_prior_net)

        self.context_kl_coef = float(context_kl_coef)
        self.context_updates_per_epoch = int(context_updates_per_epoch)
        self.context_batch_episodes = int(context_batch_episodes)

        # Episodic replay (ordered)
        self.episode_replay = collections.deque(maxlen=int(context_replay_size))

        # current context vector used for NEW rollouts
        self._current_context = None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option, context=None):
        if self.dim_context <= 0:
            return get_torch_concat_obs(obs, option)
        assert context is not None, "context must be provided when dim_context > 0"
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(obs.shape[0], -1)
        return torch.cat([obs, option, context], dim=-1)

    def _generate_option_context_extras(self, options, context):
        if self.dim_context <= 0:
            return self._generate_option_extras(options)
        if context.ndim == 1:
            context = np.repeat(context[None, :], repeats=len(options), axis=0)
        return [{'option': opt, 'context': ctx} for opt, ctx in zip(options, context)]
    
    def _fix_ctx(self, ctx, batch_size):
        
        if not torch.is_tensor(ctx):
            ctx = torch.as_tensor(ctx, dtype=torch.float32, device=self.device)

        if ctx.dim() == 3 and ctx.shape[1] == 1:
            ctx = ctx[:, 0, :]

        if ctx.dim() == 1:
            ctx = ctx.unsqueeze(0).expand(batch_size, -1)

        return ctx
    
    # ---------- Episodic buffer storage ----------
    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        # store raw episodes (needed for Route B)
        if self.dim_context > 0:
            self._store_episodes(paths)
        return super().train_once(itr, paths, runner, extra_scalar_metrics)

    def _store_episodes(self, paths):
        for p in paths:
            ep = {
                'obs': p['observations'],
                'actions': p['actions'],
                'rewards': p['rewards'].reshape(-1, 1),
                'next_obs': p['next_observations'],
            }
            self.episode_replay.append(ep)

    def _store_episodes2(self, paths):
        for p in paths:
            task_id = None
            if 'task_id' in p.get('env_infos', {}):
                ti = p['env_infos']['task_id']
                # task_id might be per-step array/list; take the first element
                if isinstance(ti, np.ndarray):
                    task_id = int(ti[0])
                elif isinstance(ti, list):
                    task_id = int(ti[0])
                else:
                    task_id = int(ti)

            ep = {
                'obs': p['observations'],
                'actions': p['actions'],
                'rewards': p['rewards'].reshape(-1, 1),
                'next_obs': p['next_observations'],
                'task_id': task_id,
            }
            self.episode_replay.append(ep)

    # ---------- Train trajectory sampling ----------
    def _get_train_trajectories_kwargs(self, runner):
        # Sample skills as in METRA

        if self.dim_context > 0 and self.task_switch_period > 0:
            if runner.step_itr > 0 and (runner.step_itr % self.task_switch_period == 0):
                if hasattr(runner._env, "switch_task"):
                    runner._env.switch_task()
        if self.discrete:
            options = np.eye(self.dim_option)[
                np.random.randint(0, self.dim_option, runner._train_args.batch_size)
            ]
        else:
            options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                options /= np.linalg.norm(options, axis=-1, keepdims=True)

        if self.dim_context > 0:
            if self._current_context is None:
                self._current_context = np.zeros(self.dim_context, dtype=np.float32)
            extras = self._generate_option_context_extras(options, context=self._current_context)
        else:
            extras = self._generate_option_extras(options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    # ---------- Replay buffer ----------
    def _update_replay_buffer(self, data):
        if self.replay_buffer is None:
            return
        for i in range(len(data['actions'])):
            path = {}
            for key in data.keys():
                cur_list = data[key][i]
                if cur_list.ndim == 1:
                    cur_list = cur_list[..., np.newaxis]  # FIXED
                path[key] = cur_list

            # ensure context exists if enabled (fallback)
            if self.dim_context > 0 and 'context' not in path:
                T = path['actions'].shape[0]
                ctx = np.repeat(self._current_context[None, :], repeats=T, axis=0)
                path['context'] = ctx
                path['next_context'] = ctx

            self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key and 'context' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    # ---------- Route B: context model training ----------
    def _sample_episode_pairs(self, batch_size):
        n = len(self.episode_replay)
        if n < 2:
            return None
        idx = np.random.randint(1, n, size=batch_size)  # 1..n-1
        prev_eps = [self.episode_replay[i - 1] for i in idx]
        cur_eps = [self.episode_replay[i] for i in idx]
        return prev_eps, cur_eps
    
    def _sample_episode_pairs2(self, batch_size):
        n = len(self.episode_replay)
        if n < 2:
            return None

        prev_eps, cur_eps = [], []
        tries = 0
        max_tries = batch_size * 20

        while len(cur_eps) < batch_size and tries < max_tries:
            tries += 1
            i = np.random.randint(1, n)
            ep_i = self.episode_replay[i]
            tid_i = ep_i.get('task_id', None)
            if tid_i is None:
                continue

            # find nearest previous episode with different task_id
            j = i - 1
            while j >= 0:
                ep_j = self.episode_replay[j]
                tid_j = ep_j.get('task_id', None)
                if tid_j is not None and tid_j != tid_i:
                    prev_eps.append(ep_j)
                    cur_eps.append(ep_i)
                    break
                j -= 1

        if len(cur_eps) == 0:
            return None
        return prev_eps, cur_eps

    def _episodes_to_tensors(self, episodes):
        B = len(episodes)
        lengths = [len(ep['actions']) for ep in episodes]
        Tm = max(lengths)

        def pad(arr, Tm):
            T = arr.shape[0]
            if T == Tm:
                return arr
            pad_len = Tm - T
            return np.concatenate([arr, np.repeat(arr[-1:], pad_len, axis=0)], axis=0)

        obs = np.stack([pad(ep['obs'], Tm) for ep in episodes], axis=0)
        act = np.stack([pad(ep['actions'], Tm) for ep in episodes], axis=0)
        rew = np.stack([pad(ep['rewards'], Tm) for ep in episodes], axis=0)
        nxt = np.stack([pad(ep['next_obs'], Tm) for ep in episodes], axis=0)

        obs = torch.from_numpy(obs).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device)
        rew = torch.from_numpy(rew).float().to(self.device)
        nxt = torch.from_numpy(nxt).float().to(self.device)
        return obs, act, rew, nxt

    def _update_context_models(self, tensors):
        if self.dim_context <= 0:
            return
        if self.context_encoder is None or self.context_prior_net is None:
            return
        if len(self.episode_replay) < 2:
            return

        for _ in range(self.context_updates_per_epoch):
            batch = self._sample_episode_pairs(self.context_batch_episodes)
            if batch is None:
                return
            prev_eps, cur_eps = batch

            s0, a0, r0, sp0 = self._episodes_to_tensors(prev_eps)
            s1, a1, r1, sp1 = self._episodes_to_tensors(cur_eps)

            mu_prev, logstd_prev = self.context_encoder(s0, a0, r0, sp0)
            mu_q, logstd_q = self.context_encoder(s1, a1, r1, sp1)

            mu_p, logstd_p = self.context_prior_net(mu_prev.detach())

            kl = gaussian_kl(mu_q, logstd_q, mu_p, logstd_p)
            loss = self.context_kl_coef * kl

            tensors['LossContextKL'] = kl.detach()

            self._gradient_descent(loss, optimizer_keys=['context'])

    def _set_current_context_from_latest(self):
        if self.dim_context <= 0:
            return
        if self.context_encoder is None:
            return
        if len(self.episode_replay) < 1:
            return
        ep = self.episode_replay[-1]
        s, a, r, sp = self._episodes_to_tensors([ep])
        mu, logstd = self.context_encoder(s, a, r, sp)
        z = reparameterize(mu, logstd)  # shape (1, dim_context)
        self._current_context = z.detach().cpu().numpy().squeeze(0).astype(np.float32)

    # ---------- Main per-epoch training ----------
    def _train_once_inner(self, path_data):
        # update transition replay
        self._update_replay_buffer(path_data)
        epoch_data = self._flatten_data(path_data)

        # normal METRA training
        tensors = self._train_components(epoch_data)

        # Route B context learning + update context for next epoch rollouts
        if self.dim_context > 0:
            self._update_context_models(tensors)
            self._set_current_context_from_latest()

        return tensors

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_te(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    # ---------- Rewards / TE ----------
    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        # if conditioning phi on context, use context and next_context correctly
        if self.dim_context > 0 and self.use_context_in_phi:
            ctx = self._fix_ctx(v['context'], obs.shape[0])
            next_ctx = self._fix_ctx(v.get('next_context', v['context']), obs.shape[0])
            obs_for_phi = torch.cat([obs, ctx], dim=-1)
            next_obs_for_phi = torch.cat([next_obs, next_ctx], dim=-1)
        else:
            obs_for_phi = obs
            next_obs_for_phi = next_obs

        if self.inner:
            cur_z = self.traj_encoder(obs_for_phi).mean
            next_z = self.traj_encoder(next_obs_for_phi).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (
                    self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                rewards = (target_z * v['options']).sum(dim=1)

            v.update({'cur_z': cur_z, 'next_z': next_z})
        else:
            target_dists = self.traj_encoder(next_obs_for_phi)
            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })
        v['rewards'] = rewards

    
    
    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v['rewards']

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({'LossDp': loss_dp})

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({'cst_penalty': cst_penalty})
            tensors.update({'DualCstPenalty': cst_penalty.mean()})
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()
        tensors.update({'TeObjMean': te_obj.mean(), 'LossTe': loss_te})

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()
        tensors.update({'DualLam': dual_lam, 'LossDualLam': loss_dual_lam})

    # ---------- SAC losses ----------
    def _update_loss_qf(self, tensors, v):
        if self.dim_context > 0:
            ctx = self._fix_ctx(v['context'], v['obs'].shape[0])
            next_ctx = self._fix_ctx(v.get('next_context', v['context']), v['obs'].shape[0])
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options'],
                ctx
            )
            next_processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['next_obs']),
                v['next_options'],
                next_ctx
            )
        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options']
            )
            next_processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['next_obs']),
                v['next_options']
            )

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v['rewards'] * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        if self.dim_context > 0:
            ctx = self._fix_ctx(v['context'], v['obs'].shape[0])
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options'],
                ctx
            )
        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options']
            )

        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(self, tensors, v)

    # ---------- Eval ----------
    def _evaluate_policy(self, runner):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (
                    i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)

        if self.dim_context > 0:
            ctx = self._current_context
            if ctx is None:
                ctx = np.zeros(self.dim_context, dtype=np.float32)
            extras = self._generate_option_context_extras(random_options, ctx)
        else:
            extras = self._generate_option_extras(random_options)

        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=extras,
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)

            if self.dim_context > 0:
                ctx = self._current_context
                if ctx is None:
                    ctx = np.zeros(self.dim_context, dtype=np.float32)
                video_extras = self._generate_option_context_extras(video_options, ctx)
            else:
                video_extras = self._generate_option_extras(video_options)

            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=video_extras,
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)