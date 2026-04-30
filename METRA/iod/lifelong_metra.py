# iod/lifelong_metra.py
import collections
import copy
import numpy as np
import torch
import torch.nn.functional as F

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
from iod.expwandb import evaluate_and_plot_lifelong  
from iod.context_models import gaussian_kl, reparameterize
from iod.EvalCtxLosses import write_eval_context_losses_csv
from iod.utils import (
    get_torch_concat_obs,
    FigManager,
    get_option_colors,
    record_video,
    draw_2d_gaussians,
)


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
            context_jq_coef=1.0,
            use_context_in_phi=True,

            context_encoder=None,
            context_prior_net=None,
            context_decoder=None,
            context_kl_coef=1.0,
            recon_coef=1.0,
            context_updates_per_epoch=10,
            context_batch_episodes=16,
            context_replay_size=5000,
            deterministic_rollout_context=True,
            deterministic_context_decoder=False,

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
        self._target_entropy = (
            -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef
        )
        self.context_jq_coef = float(context_jq_coef)

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

        self.dim_context = int(dim_context)
        self.use_context_in_phi = bool(use_context_in_phi)
        self.task_switch_period = int(task_switch_period)

        self.context_encoder = (
            context_encoder.to(self.device) if context_encoder is not None else None
        )
        self.context_prior_net = (
            context_prior_net.to(self.device) if context_prior_net is not None else None
        )
        self.context_decoder = (
            context_decoder.to(self.device) if context_decoder is not None else None
        )

        if self.context_encoder is not None:
            self.param_modules.update(context_encoder=self.context_encoder)
        if self.context_prior_net is not None:
            self.param_modules.update(context_prior_net=self.context_prior_net)
        if self.context_decoder is not None:
            self.param_modules.update(context_decoder=self.context_decoder)

        self.context_kl_coef = float(context_kl_coef)
        self.recon_coef = float(recon_coef)
        self.context_updates_per_epoch = int(context_updates_per_epoch)
        self.context_batch_episodes = int(context_batch_episodes)
        self.deterministic_rollout_context = bool(deterministic_rollout_context)
        self.deterministic_context_decoder = bool(deterministic_context_decoder)

        self.episode_replay = collections.deque(maxlen=int(context_replay_size))
        self._current_context = None

        self._latest_posterior_context = None

    def _set_requires_grad(self, module, flag):
        for p in module.parameters():
            p.requires_grad_(flag)

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

    def _expand_context_over_time(self, ctx, T):
        # ctx: [B, C] -> [B, T, C]
        if ctx.dim() != 2:
            raise ValueError(f"Expected context shape [B, C], got {tuple(ctx.shape)}")
        return ctx.unsqueeze(1).expand(-1, T, -1)

    def _sample_rollout_contexts(self, batch_size):
        """
        Sample one context per trajectory/episode.
        For episodic task changes, each rollout episode gets its own context.
        """
        if self.dim_context <= 0:
            return None

        if self._latest_posterior_context is None or self.context_prior_net is None:
            if self._latest_posterior_context is None:
                contexts = np.zeros((batch_size, self.dim_context), dtype=np.float32)
            else:
                contexts = np.repeat(
                    self._latest_posterior_context[None, :],
                    repeats=batch_size,
                    axis=0,
                ).astype(np.float32)
            return contexts

        z_prev = torch.as_tensor(
            self._latest_posterior_context[None, :],
            dtype=torch.float32,
            device=self.device,
        )

        contexts = []
        for _ in range(batch_size):
            mu_p, logstd_p = self.context_prior_net(z_prev)
            if self.deterministic_rollout_context:
                c = mu_p
            else:
                c = reparameterize(mu_p, logstd_p)
            contexts.append(c.detach().cpu().numpy().squeeze(0))

        contexts = np.stack(contexts, axis=0).astype(np.float32)
        return contexts

    # ---------- Episodic buffer storage ----------
    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        if self.dim_context > 0:
            self._store_episodes(paths)
        return super().train_once(itr, paths, runner, extra_scalar_metrics)

    def _store_episodes(self, paths):
        for p in paths:
            if 'option' in p.get('agent_infos', {}):
                options = p['agent_infos']['option']
            elif 'options' in p:
                options = p['options']
            else:
                raise KeyError(
                    "Could not find per-step options in path. "
                    "Expected path['agent_infos']['option'] or path['options']."
                )

            ep = {
                'obs': p['observations'],
                'actions': p['actions'],
                'options': options,
                'rewards': p['rewards'].reshape(-1, 1),
                'next_obs': p['next_observations'],
                'dones': p['dones'].reshape(-1, 1),
            }
            self.episode_replay.append(ep)

    def _compute_context_jq(self, s1, a1, o1, r1, sp1, d1, mask, c_sample):
        B, T, _ = s1.shape

        obs = s1.reshape(B * T, -1)
        actions = a1.reshape(B * T, -1)
        options = o1.reshape(B * T, -1)
        rewards = r1.reshape(B * T, -1).squeeze(-1)
        next_obs = sp1.reshape(B * T, -1)
        dones = d1.reshape(B * T, -1).squeeze(-1)
        flat_mask = mask.reshape(B * T, -1).squeeze(-1)

        ctx = c_sample.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        next_ctx = ctx

        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(obs),
            options,
            ctx,
        )
        next_processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(next_obs),
            options,
            next_ctx,
        )

        with torch.no_grad():
            alpha = self.log_alpha.param.exp()

            next_action_dists, *_ = self.option_policy(next_processed_cat_obs)
            if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
                pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
                new_next_action_log_probs = next_action_dists.log_prob(
                    new_next_actions, pre_tanh_value=pre_tanh
                )
            else:
                new_next_actions = next_action_dists.rsample()
                new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

            target_q_values = torch.min(
                self.target_qf1(next_processed_cat_obs, new_next_actions).flatten(),
                self.target_qf2(next_processed_cat_obs, new_next_actions).flatten(),
            )
            target_q_values = target_q_values - alpha * new_next_action_log_probs
            target_q_values = target_q_values * self.discount

            q_target = rewards + target_q_values * (1.0 - dones)

        q1_pred = self.qf1(processed_cat_obs, actions).flatten()
        q2_pred = self.qf2(processed_cat_obs, actions).flatten()

        err1 = 0.5 * (q1_pred - q_target) ** 2
        err2 = 0.5 * (q2_pred - q_target) ** 2

        denom = flat_mask.sum().clamp_min(1.0)
        loss_qf1_ctx = (err1 * flat_mask).sum() / denom
        loss_qf2_ctx = (err2 * flat_mask).sum() / denom

        return loss_qf1_ctx + loss_qf2_ctx

    def _get_train_trajectories_kwargs(self, runner):
    
        if self.discrete:
            options = np.eye(self.dim_option)[
                np.random.randint(0, self.dim_option, runner._train_args.batch_size)
            ]
        else:
            options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                options /= np.linalg.norm(options, axis=-1, keepdims=True)

        if self.dim_context > 0:
            contexts = self._sample_rollout_contexts(runner._train_args.batch_size)

            # keep a representative context around for eval/video
            if contexts is not None and len(contexts) > 0:
                self._current_context = contexts[0].copy()
            elif self._current_context is None:
                self._current_context = np.zeros(self.dim_context, dtype=np.float32)

            extras = self._generate_option_context_extras(options, contexts)
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
            for key, value in data.items():
                cur_list = value[i]
                if cur_list.ndim == 1:
                    cur_list = cur_list[..., np.newaxis]
                path[key] = cur_list

            if self.dim_context > 0 and 'context' not in path:
                raise KeyError(
                    "Missing context in collected path while dim_context > 0. "
                    "The sampler/worker must preserve per-trajectory context from extras."
                )

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
            epoch_data[key] = torch.tensor(
                np.concatenate(value, axis=0),
                dtype=torch.float32,
                device=self.device,
            )
        return epoch_data

    # ---------- Context model training ----------
    def _sample_episode_pairs(self, batch_size):
        n = len(self.episode_replay)
        if n < 2:
            return None
        idx = np.random.randint(1, n, size=batch_size)
        prev_eps = [self.episode_replay[i - 1] for i in idx]
        cur_eps = [self.episode_replay[i] for i in idx]
        return prev_eps, cur_eps

    def _episodes_to_tensors(self, episodes):
        lengths = [len(ep['actions']) for ep in episodes]
        Tm = max(lengths)

        def pad(arr, Tm_):
            T = arr.shape[0]
            if T == Tm_:
                return arr
            pad_len = Tm_ - T
            return np.concatenate(
                [arr, np.repeat(arr[-1:], pad_len, axis=0)],
                axis=0,
            )

        obs = np.stack([pad(ep['obs'], Tm) for ep in episodes], axis=0)
        act = np.stack([pad(ep['actions'], Tm) for ep in episodes], axis=0)
        opt = np.stack([pad(ep['options'], Tm) for ep in episodes], axis=0)
        rew = np.stack([pad(ep['rewards'], Tm) for ep in episodes], axis=0)
        nxt = np.stack([pad(ep['next_obs'], Tm) for ep in episodes], axis=0)
        dns = np.stack([pad(ep['dones'], Tm) for ep in episodes], axis=0)

        mask = np.zeros((len(episodes), Tm, 1), dtype=np.float32)
        for i, L in enumerate(lengths):
            mask[i, :L, 0] = 1.0

        obs = torch.from_numpy(obs).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device)
        opt = torch.from_numpy(opt).float().to(self.device)
        rew = torch.from_numpy(rew).float().to(self.device)
        nxt = torch.from_numpy(nxt).float().to(self.device)
        dns = torch.from_numpy(dns).float().to(self.device)
        mask = torch.from_numpy(mask).float().to(self.device)

        return obs, act, opt, rew, nxt, dns, mask
    
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

            s0, a0, o0, r0, sp0, d0, m0 = self._episodes_to_tensors(prev_eps)
            s1, a1, o1, r1, sp1, d1, m1 = self._episodes_to_tensors(cur_eps)

            mu_prev, logstd_prev = self.context_encoder(s0, a0, r0, sp0)
            mu_q, logstd_q = self.context_encoder(s1, a1, r1, sp1)
            mu_p, logstd_p = self.context_prior_net(mu_prev.detach())

            kl = gaussian_kl(mu_q, logstd_q, mu_p, logstd_p)
            tensors['LossContextKL'] = kl.detach()

            if self.deterministic_context_decoder:
                c_sample = mu_q
            else:
                c_sample = reparameterize(mu_q, logstd_q)

            loss = self.context_kl_coef * kl

            if self.context_decoder is not None:
                T = s1.shape[1]
                c_seq = self._expand_context_over_time(c_sample, T)
                pred_next_obs, pred_reward = self.context_decoder(s1, a1, c_seq)

                # masked reconstruction loss
                recon_next_per = ((pred_next_obs - sp1) ** 2).mean(dim=-1, keepdim=True)
                recon_rew_per = ((pred_reward - r1) ** 2).mean(dim=-1, keepdim=True)

                denom = m1.sum().clamp_min(1.0)
                recon_next = (recon_next_per * m1).sum() / denom
                recon_reward = (recon_rew_per * m1).sum() / denom
                recon_loss = recon_next + recon_reward

                tensors['LossContextRecon'] = recon_loss.detach()
                tensors['LossContextReconNextObs'] = recon_next.detach()
                tensors['LossContextReconReward'] = recon_reward.detach()

                loss = loss + self.recon_coef * recon_loss

            # JQ term with mask + dones
            self._set_requires_grad(self.qf1, False)
            self._set_requires_grad(self.qf2, False)
            try:
                jq_loss = self._compute_context_jq(
                    s1, a1, o1, r1, sp1, d1, m1, c_sample
                )
            finally:
                self._set_requires_grad(self.qf1, True)
                self._set_requires_grad(self.qf2, True)

            tensors['LossContextJQ'] = jq_loss.detach()
            loss = loss + self.context_jq_coef * jq_loss

            tensors['LossContext'] = loss.detach()

            self._latest_context_eval_losses = {
                "kl_loss": float(tensors["LossContextKL"].detach().cpu().mean().item()),
                "reconstruction_loss": float(
                    tensors.get("LossContextRecon", torch.tensor(float("nan"), device=self.device)).detach().cpu().mean().item()
                ),
                "reconstruction_next_obs_loss": float(
                    tensors.get("LossContextReconNextObs", torch.tensor(float("nan"), device=self.device)).detach().cpu().mean().item()
                ),
                "reconstruction_reward_loss": float(
                    tensors.get("LossContextReconReward", torch.tensor(float("nan"), device=self.device)).detach().cpu().mean().item()
                ),
                "jq_loss": float(tensors["LossContextJQ"].detach().cpu().mean().item()),
                "total_context_loss": float(tensors["LossContext"].detach().cpu().mean().item()),
            }

            self._gradient_descent(loss, optimizer_keys=['context'])

    def _set_current_context_from_prior(self):
        """
        Keeps a representative predicted context for eval/video.
        Training rollouts themselves use per-trajectory contexts from _sample_rollout_contexts().
        """
        if self.dim_context <= 0:
            return

        if self.context_prior_net is None:
            if self._latest_posterior_context is None:
                self._current_context = np.zeros(self.dim_context, dtype=np.float32)
            else:
                self._current_context = self._latest_posterior_context.copy()
            return

        if self._latest_posterior_context is None:
            self._current_context = np.zeros(self.dim_context, dtype=np.float32)
            return

        z_prev = torch.as_tensor(
            self._latest_posterior_context[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        mu_p, logstd_p = self.context_prior_net(z_prev)

        if self.deterministic_rollout_context:
            z_pred = mu_p
        else:
            z_pred = reparameterize(mu_p, logstd_p)

        self._current_context = (
            z_pred.detach().cpu().numpy().squeeze(0).astype(np.float32)
        )

    def _set_latest_posterior_context(self):
        if self.dim_context <= 0 or self.context_encoder is None:
            return
        if len(self.episode_replay) < 1:
            return

        ep = self.episode_replay[-1]
        s, a, o, r, sp, d, m = self._episodes_to_tensors([ep])
        mu, logstd = self.context_encoder(s, a, r, sp)

        self._latest_posterior_context = (
            mu.detach().cpu().numpy().squeeze(0).astype(np.float32)
        )

    # ---------- Main per-epoch training ----------
    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)
        epoch_data = self._flatten_data(path_data)

        tensors = self._train_components(epoch_data)

        if self.dim_context > 0:
            self._update_context_models(tensors)
            self._set_latest_posterior_context()
            self._set_current_context_from_prior()

        return tensors

    def _train_components(self, epoch_data):
        if (
            self.replay_buffer is not None
            and self.replay_buffer.n_transitions_stored < self.min_buffer_size
        ):
            return {}

        tensors = {}
        for _ in range(self._trans_optimization_epochs):
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

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

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
                masks = (
                    (v['options'] - v['options'].mean(dim=1, keepdim=True))
                    * self.dim_option
                    / (self.dim_option - 1 if self.dim_option != 1 else 1)
                )
                rewards = (target_z * masks).sum(dim=1)
            else:
                rewards = (target_z * v['options']).sum(dim=1)

            v.update({'cur_z': cur_z, 'next_z': next_z})
        else:
            target_dists = self.traj_encoder(next_obs_for_phi)
            if self.discrete:
                logits = target_dists.mean
                rewards = -F.cross_entropy(
                    logits,
                    v['options'].argmax(dim=1),
                    reduction='none',
                )
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
                geo_mean = torch.exp(
                    torch.log(scaling_factor).mean(dim=1, keepdim=True)
                )
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(
                    torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor,
                    dim=1,
                )

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
                ctx,
            )
            next_processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['next_obs']),
                v['next_options'],
                next_ctx,
            )
        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options'],
            )
            next_processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['next_obs']),
                v['next_options'],
            )

        sac_utils.update_loss_qf(
            self,
            tensors,
            v,
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
                ctx,
            )
        else:
            processed_cat_obs = self._get_concat_obs(
                self.option_policy.process_observations(v['obs']),
                v['options'],
            )

        sac_utils.update_loss_sacp(
            self,
            tensors,
            v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(self, tensors, v)

    # ---------- Eval ----------
    def _evaluate_policy(self, runner):
        random_trajectories, random_options, sampled_contexts, random_option_colors = (
            evaluate_and_plot_lifelong(self, runner)
        )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([
            torch.from_numpy(ob[-1]).float().to(self.device) for ob in data['obs']
        ])

        if self.dim_context > 0 and self.use_context_in_phi:
            ctx_t = torch.as_tensor(
                sampled_contexts,
                dtype=torch.float32,
                device=self.device,
            )
            last_obs_for_phi = torch.cat([last_obs, ctx_t], dim=-1)
        else:
            last_obs_for_phi = last_obs

        option_dists = self.traj_encoder(last_obs_for_phi)

        option_means = option_dists.mean.detach().cpu().numpy()

        if self.inner:
            option_stddevs = torch.ones_like(
                option_dists.stddev.detach().cpu()
            ).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()

        option_samples = option_dists.mean.detach().cpu().numpy()
        option_colors = random_option_colors

        with FigManager(runner, 'PhiPlot') as fm:
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
            video_options = random_options[:9]

            if sampled_contexts is not None:
                video_contexts = sampled_contexts[:9]
            else:
                video_contexts = None

            if self.dim_context > 0:
                video_extras = self._generate_option_context_extras(
                    video_options,
                    video_contexts,
                )
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

            record_video(
                runner,
                'Video_Lifelong_FixedZ_SampledC',
                video_trajectories,
                skip_frames=self.video_skip_frames,
            )

        eval_option_metrics.update(
            runner._env.calc_eval_metrics(
                random_trajectories,
                is_option_trajectories=True,
            )
        )

        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(
                    self._env_spec,
                    random_trajectories,
                ),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        write_eval_context_losses_csv(self, runner)
        self._log_eval_metrics(runner)