from collections import defaultdict

import akro
import gym
import numpy as np
import torch

from garage.envs import EnvSpec
from iod.context_models import reparameterize
from iod.utils import get_torch_concat_obs


class ChildPolicyEnv(gym.Wrapper):
    """
    Hierarchical wrapper that:
      - exposes a high-level action space over child-policy options
      - executes the frozen child policy for cp_multi_step low-level env steps
      - tracks a latent lifelong context online using:
            context_prior_net: p(c_t | c_{t-1})
            context_encoder:   q(c_t | trajectory segment / episode)
      - returns augmented high-level observations [obs, context]

    Notes
    -----
    This implementation updates the posterior context at EPISODE END.
    That is the simplest stable version and matches the "between-episode
    non-stationarity" assumption best.

    If you later want faster adaptation, you can move the posterior update
    to every high-level step instead.
    """

    def __init__(
            self,
            env,
            cp_dict,
            cp_action_range,
            cp_unit_length,
            cp_multi_step,
            cp_num_truncate_obs,
            cp_omit_obs_idxs=None,
    ):
        super().__init__(env)

        # ---------------- Child policy ----------------
        self.child_policy = cp_dict['policy']
        self.child_policy.eval()

        self.cp_dim_action = cp_dict['dim_option']
        self.cp_dim_context = cp_dict.get('dim_context', 0)
        self.cp_discrete = cp_dict['discrete']

        self.cp_action_range = cp_action_range
        self.cp_unit_length = cp_unit_length
        self.cp_multi_step = cp_multi_step
        self.cp_num_truncate_obs = cp_num_truncate_obs
        self.cp_omit_obs_idxs = cp_omit_obs_idxs

        # ---------------- Lifelong context modules ----------------
        self.context_encoder = cp_dict.get('context_encoder', None)
        self.context_prior_net = cp_dict.get('context_prior_net', None)
        self.context_decoder = cp_dict.get('context_decoder', None)  # not used online here
        self.deterministic_rollout_context = cp_dict.get(
            'deterministic_rollout_context', True
        )

        if self.context_encoder is not None:
            self.context_encoder.eval()
        if self.context_prior_net is not None:
            self.context_prior_net.eval()
        if self.context_decoder is not None:
            self.context_decoder.eval()

        try:
            self.device = next(self.child_policy.parameters()).device
        except Exception:
            self.device = torch.device('cpu')

        # ---------------- High-level spaces ----------------
        self._base_observation_space = self.env.observation_space

        if self.cp_discrete:
            self.action_space = akro.Discrete(n=self.cp_dim_action)
        else:
            self.action_space = akro.Box(low=-1., high=1., shape=(self.cp_dim_action,))

        # High-level policy observes [obs, context] if context exists.
        if self.cp_dim_context > 0:
            base_low = np.asarray(self._base_observation_space.low, dtype=np.float32).reshape(-1)
            base_high = np.asarray(self._base_observation_space.high, dtype=np.float32).reshape(-1)

            ctx_low = -np.inf * np.ones(self.cp_dim_context, dtype=np.float32)
            ctx_high = np.inf * np.ones(self.cp_dim_context, dtype=np.float32)

            aug_low = np.concatenate([base_low, ctx_low], axis=0)
            aug_high = np.concatenate([base_high, ctx_high], axis=0)

            self.observation_space = akro.Box(
                low=aug_low,
                high=aug_high,
                shape=aug_low.shape,
                dtype=np.float32,
            )
        else:
            self.observation_space = self._base_observation_space

        # ---------------- Runtime state ----------------
        self.last_obs = None          # raw env obs
        self.first_obs = None         # raw env obs at episode start

        self.current_context = None               # rollout context used NOW
        self.latest_posterior_context = None      # posterior from previous episode

        # Episode buffers for posterior update
        self._ep_obs = []
        self._ep_actions = []
        self._ep_rewards = []
        self._ep_next_obs = []
        self._ep_dones = []

    @property
    def spec(self):
        return EnvSpec(
            action_space=self.action_space,
            observation_space=self.observation_space,
        )

    # =========================================================
    # Context helpers
    # =========================================================

    def set_context(self, context):
        if context is None:
            self.current_context = None
        else:
            context = np.asarray(context, dtype=np.float32).reshape(-1)
            if self.cp_dim_context > 0 and context.shape[0] != self.cp_dim_context:
                raise ValueError(
                    f"Expected context dim {self.cp_dim_context}, got {context.shape[0]}"
                )
            self.current_context = context

    def _get_context(self):
        if self.cp_dim_context <= 0:
            return None
        if self.current_context is None:
            return np.zeros(self.cp_dim_context, dtype=np.float32)
        return self.current_context.astype(np.float32, copy=False)

    def _augment_obs(self, obs):
        if self.cp_dim_context <= 0:
            return obs
        ctx = self._get_context()
        return np.concatenate(
            [np.asarray(obs, dtype=np.float32).reshape(-1), ctx],
            axis=0,
        ).astype(np.float32, copy=False)

    def _predict_context_from_prior(self):
        """
        Predict rollout context for the NEW episode using the previous episode's
        posterior context.

        If no posterior exists yet, fall back to zeros.
        """
        if self.cp_dim_context <= 0:
            return None

        if self.latest_posterior_context is None or self.context_prior_net is None:
            ctx = np.zeros(self.cp_dim_context, dtype=np.float32)
        else:
            z_prev = torch.as_tensor(
                self.latest_posterior_context[None, :],
                dtype=torch.float32,
                device=self.device,
            )
            with torch.no_grad():
                mu_p, logstd_p = self.context_prior_net(z_prev)
                if self.deterministic_rollout_context:
                    z_pred = mu_p
                else:
                    z_pred = reparameterize(mu_p, logstd_p)

            ctx = z_pred.detach().cpu().numpy().squeeze(0).astype(np.float32)

        self.current_context = ctx
        return ctx

    def _clear_episode_buffers(self):
        self._ep_obs = []
        self._ep_actions = []
        self._ep_rewards = []
        self._ep_next_obs = []
        self._ep_dones = []

    def _append_low_level_transition(self, obs, action, reward, next_obs, done):
        self._ep_obs.append(np.asarray(obs, dtype=np.float32))
        self._ep_actions.append(np.asarray(action, dtype=np.float32))
        self._ep_rewards.append(float(reward))
        self._ep_next_obs.append(np.asarray(next_obs, dtype=np.float32))
        self._ep_dones.append(float(done))

    def _update_posterior_from_episode(self):
        """
        Infer posterior context q(c | episode) using the low-level trajectory
        collected in the just-finished episode.

        This mirrors the pretraining logic:
            mu, logstd = context_encoder(s, a, r, s')
            latest_posterior_context <- mu
        """
        if self.cp_dim_context <= 0:
            return

        if self.context_encoder is None:
            return

        if len(self._ep_actions) == 0:
            return

        s = torch.as_tensor(
            np.stack(self._ep_obs, axis=0),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, T, obs_dim]

        a = torch.as_tensor(
            np.stack(self._ep_actions, axis=0),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, T, act_dim]

        r = torch.as_tensor(
            np.asarray(self._ep_rewards, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1, 1)  # [1, T, 1]

        sp = torch.as_tensor(
            np.stack(self._ep_next_obs, axis=0),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, T, obs_dim]

        with torch.no_grad():
            mu, logstd = self.context_encoder(s, a, r, sp)

        self.latest_posterior_context = (
            mu.detach().cpu().numpy().squeeze(0).astype(np.float32)
        )

    # =========================================================
    # Gym API
    # =========================================================

    def reset(self, **kwargs):
        # Predict rollout context for the new episode
        if self.cp_dim_context > 0:
            self._predict_context_from_prior()

        ret = self.env.reset(**kwargs)

        # Raw obs used internally by child policy
        self.last_obs = ret
        self.first_obs = ret

        # Clear episode trajectory buffers
        self._clear_episode_buffers()

        # High-level policy sees [obs, context]
        return self._augment_obs(ret)

    def step(self, cp_action, debug: bool = False, **kwargs):
        """
        cp_action is the HIGH-LEVEL action:
          - continuous option vector, or
          - discrete option id / one-hot-equivalent control input

        This wrapper rolls out the child policy for cp_multi_step low-level env steps.
        """
        cp_action = np.asarray(cp_action, dtype=np.float32).copy()
        cp_action_norm = np.linalg.norm(cp_action)

        if not self.cp_discrete:
            if self.cp_unit_length:
                cp_action = cp_action / max(cp_action_norm, 1e-8)
            else:
                cp_action = cp_action * self.cp_action_range

        sum_rewards = 0.
        acc_infos = defaultdict(list)

        low_level_obs = []
        low_level_actions = []
        low_level_rewards = []
        low_level_next_obs = []
        low_level_dones = []

        done_final = False
        next_obs = None

        for _ in range(self.cp_multi_step):
            raw_obs = self.last_obs
            cp_obs = torch.as_tensor(raw_obs, dtype=torch.float32)

            if self.cp_num_truncate_obs > 0:
                cp_obs = cp_obs[:-self.cp_num_truncate_obs]
            if self.cp_omit_obs_idxs is not None:
                cp_obs = cp_obs.clone()
                cp_obs[self.cp_omit_obs_idxs] = 0

            cp_action_t = torch.as_tensor(cp_action, dtype=torch.float32)

            if self.cp_dim_context > 0:
                if self.current_context is None:
                    raise ValueError(
                        "ChildPolicyEnv: child policy expects context, "
                        "but current_context is not set."
                    )
                cp_context_t = torch.as_tensor(self.current_context, dtype=torch.float32)
                cp_input = torch.cat([cp_obs, cp_action_t, cp_context_t], dim=0).float()
            else:
                cp_input = get_torch_concat_obs(cp_obs, cp_action_t, dim=0).float()

            with torch.no_grad():
                if hasattr(self.child_policy._module, 'forward_mode'):
                    child_action = self.child_policy.get_mode_actions(
                        cp_input.unsqueeze(dim=0)
                    )[0]
                else:
                    action_dist = self.child_policy(cp_input.unsqueeze(dim=0))[0]
                    child_action = action_dist.mean.detach().cpu().numpy()

            child_action = child_action[0]

            # Rescale from [-1, 1] to env bounds
            lb = np.asarray(self.env.action_space.low, dtype=np.float32)
            ub = np.asarray(self.env.action_space.high, dtype=np.float32)
            env_action = lb + (child_action + 1) * (0.5 * (ub - lb))
            env_action = np.clip(env_action, lb, ub)

            next_obs, reward, done, info = self.env.step(env_action, **kwargs)

            # Store low-level transition for online context inference
            self._append_low_level_transition(
                obs=raw_obs,
                action=env_action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            low_level_obs.append(np.asarray(raw_obs, dtype=np.float32))
            low_level_actions.append(np.asarray(env_action, dtype=np.float32))
            low_level_rewards.append(float(reward))
            low_level_next_obs.append(np.asarray(next_obs, dtype=np.float32))
            low_level_dones.append(float(done))

            self.last_obs = next_obs

            sum_rewards += reward
            for k, v in info.items():
                acc_infos[k].append(v)

            if info.get('done_internal', False):
                done_final = True

            if done:
                done_final = True
                break

        # If the episode ended, update posterior context from the collected episode
        if done_final and self.cp_dim_context > 0:
            self._update_posterior_from_episode()

        infos = {}
        for k, v in acc_infos.items():
            if debug:
                if k in ['coordinates', 'next_coordinates', 'ori', 'next_ori']:
                    infos[k] = np.concatenate(v).reshape(-1, v[0].shape[-1])
                elif k in ['ori_obs', 'next_ori_obs']:
                    infos[k] = v[-1]
                else:
                    if isinstance(v[0], np.ndarray):
                        infos[k] = np.array(v)
                    elif isinstance(v[0], tuple):
                        infos[k] = np.array([list(l) for l in v])
                    else:
                        infos[k] = sum(v)
            else:
                infos[k] = v[-1]

        infos['cp_action_norm'] = cp_action_norm
        infos['cp_action'] = cp_action.astype(np.float32, copy=False)

        # Summaries over the low-level rollout segment
        if len(low_level_rewards) > 0:
            ll_rewards = np.asarray(low_level_rewards, dtype=np.float32)
            infos['low_level_reward_sum'] = float(ll_rewards.sum())
            infos['low_level_reward_mean'] = float(ll_rewards.mean())
            infos['low_level_steps'] = int(len(ll_rewards))
        else:
            infos['low_level_reward_sum'] = 0.0
            infos['low_level_reward_mean'] = 0.0
            infos['low_level_steps'] = 0

        if len(low_level_actions) > 0:
            ll_actions = np.stack(low_level_actions, axis=0).astype(np.float32)
            ll_action_norms = np.linalg.norm(ll_actions, axis=-1)
            infos['low_level_action_norm_mean'] = float(ll_action_norms.mean())
            infos['low_level_action_norm_max'] = float(ll_action_norms.max())
        else:
            infos['low_level_action_norm_mean'] = 0.0
            infos['low_level_action_norm_max'] = 0.0

        if self.cp_dim_context > 0:
            ctx = self._get_context().astype(np.float32, copy=False)
            infos['context'] = ctx.copy()
            infos['context_norm'] = float(np.linalg.norm(ctx))

            if self.latest_posterior_context is None:
                infos['latest_posterior_context'] = np.zeros(
                    self.cp_dim_context, dtype=np.float32
                )
                infos['posterior_context_norm'] = 0.0
                infos['context_drift'] = 0.0
            else:
                post = self.latest_posterior_context.astype(np.float32, copy=False)
                infos['latest_posterior_context'] = post.copy()
                infos['posterior_context_norm'] = float(np.linalg.norm(post))
                infos['context_drift'] = float(np.linalg.norm(post - ctx))

        if self.cp_dim_context > 0:
            infos['context'] = self._get_context().copy()

            if self.latest_posterior_context is None:
                infos['latest_posterior_context'] = np.zeros(
                    self.cp_dim_context, dtype=np.float32
                )
            else:
                infos['latest_posterior_context'] = (
                    self.latest_posterior_context.copy().astype(np.float32)
                )

        # Expose low-level rollout segment for debugging / future step-wise updates
        infos['low_level_obs'] = np.stack(low_level_obs, axis=0) if len(low_level_obs) > 0 else np.zeros((0,), dtype=np.float32)
        infos['low_level_actions'] = np.stack(low_level_actions, axis=0) if len(low_level_actions) > 0 else np.zeros((0,), dtype=np.float32)
        infos['low_level_rewards'] = np.asarray(low_level_rewards, dtype=np.float32)
        infos['low_level_next_obs'] = np.stack(low_level_next_obs, axis=0) if len(low_level_next_obs) > 0 else np.zeros((0,), dtype=np.float32)
        infos['low_level_dones'] = np.asarray(low_level_dones, dtype=np.float32)

        # High-level controller sees [obs, context]
        return self._augment_obs(next_obs), sum_rewards, done_final, infos