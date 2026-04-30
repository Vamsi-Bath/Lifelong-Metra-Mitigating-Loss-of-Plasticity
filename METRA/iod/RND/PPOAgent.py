import os
import numpy as np
import torch as T
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.optim.lr_scheduler import StepLR

from . import PPO_Models
from .RNDAgent import RNDModule
from . import Memory as Memory


class Agent:
    """
    Your Agent API, modified to load and run the other PPO checkpoints:
      - {model_name}_actor.pth
      - {model_name}_critic.pth
      - {model_name}_rms.pth
    """
    def __init__(
        self,
        variant,
        observation_space,
        RND,
        device,
        action_space,
        n_actions,
        input_dims,
        Reward_Scaler,
        NormaliseAdvantage,
        chkpt_dir,
        Network,
        gamma,
        alpha,
        gae_lambda,
        policy_clip,
        batch_size,
        n_epochs,
        penaltyCoefficient,
        d_target,
        c1,
        c2,
        kl_div,
        action_type,
        model_name=None,
    ):
        self.variant = variant
        self.action_space = action_space
        self.input_dims = input_dims
        self.alpha = alpha
        self.Reward_Scaler = Reward_Scaler
        self.NormaliseAdvantage = NormaliseAdvantage
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.Network = Network
        self.gae_lambda = gae_lambda
        self.RND = RND
        self.policy_clip = policy_clip
        self.chkpt_dir = chkpt_dir
        self.beta = penaltyCoefficient
        self.d_target = d_target
        self.kldiv = kl_div
        self.c1 = c1
        self.c2 = c2
        self.device = device
        self.action_type = action_type
        self.model_name = model_name

        max_action = 1.0
        if hasattr(action_space, "high"):
            try:
                max_action = float(action_space.high[0])
            except Exception:
                max_action = 1.0

        if self.action_type == 'discrete':
            if self.Network == 'CNN':
                self.actor = PPO_Models.ActorPixelNetwork(
                    n_actions, input_dims, alpha, device, chkpt_dir, model_name=model_name
                )
                self.critic = PPO_Models.CriticPixelNetwork(
                    input_dims, alpha, device, chkpt_dir, model_name=model_name
                )
            elif self.Network == 'Dense':
                self.actor = PPO_Models.ActorVectorNetwork(
                    n_actions, input_dims, alpha, device, chkpt_dir, model_name=model_name
                )
                self.critic = PPO_Models.CriticVectorNetwork(
                    input_dims, alpha, device, chkpt_dir, model_name=model_name
                )
            else:
                raise ValueError("Network must be 'Dense' or 'CNN' for discrete actions.")

        elif self.action_type == 'continuous':
            self.actor = PPO_Models.ActorContinuousNetwork(
                n_actions, input_dims, alpha, device, chkpt_dir, model_name=model_name, max_action=max_action
            )
            self.critic = PPO_Models.CriticVectorNetwork(
                input_dims, alpha, device, chkpt_dir, model_name=model_name
            )
        else:
            raise ValueError("action_type must be either 'discrete' or 'continuous'.")

        self.obs_rms = PPO_Models.RunningMeanStd(shape=(input_dims[0],), device=device)
        self.reward_rms = PPO_Models.RunningMeanStd(shape=(1,), device=device)
        self.normalize_rewards = True

        self.rms_file = None
        if model_name is not None:
            self.rms_file = os.path.join(chkpt_dir, f"{model_name}_rms.pth")

        self.memory = Memory.Memory(batch_size)
        self.actor_optim_file = self.actor.checkpoint_file + "_optim"
        self.critic_optim_file = self.critic.checkpoint_file + "_optim"

        self.actor_scheduler = StepLR(self.actor.optimiser, step_size=1000, gamma=0.9)
        self.critic_scheduler = StepLR(self.critic.optimiser, step_size=1000, gamma=0.9)

        if self.RND:
            self.rnd = RNDModule(obs_shape=input_dims, lr=alpha, device=device)

    def _normalize_obs(self, states):
        return T.clamp(
            (states - self.obs_rms.mean) / T.sqrt(self.obs_rms.var + 1e-8),
            -10.0,
            10.0,
        )

    def _get_log_prob(self, dist, actions):
        log_prob = dist.log_prob(actions)
        if self.action_type == 'continuous':
            log_prob = log_prob.sum(dim=-1)
        return log_prob

    def store_transition(self, state, action, probs, vals, extrinsic_reward, intrinsic_reward, done, truncated):
        self.memory.store_memory(state, action, probs, vals, extrinsic_reward, intrinsic_reward, done, truncated)

    def save_models(self):
        print("Saving Models")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

        if self.rms_file is not None:
            T.save({
                "mean": self.obs_rms.mean.detach().cpu(),
                "var": self.obs_rms.var.detach().cpu(),
                "count": self.obs_rms.count,
                "reward_mean": self.reward_rms.mean.detach().cpu(),
                "reward_var": self.reward_rms.var.detach().cpu(),
                "reward_count": self.reward_rms.count,
            }, self.rms_file)

        T.save(self.actor.optimiser.state_dict(), self.actor_optim_file)
        T.save(self.critic.optimiser.state_dict(), self.critic_optim_file)

    def load_models(self):
        print("Loading Models")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

        if self.rms_file is not None and os.path.exists(self.rms_file):
            rms_data = T.load(self.rms_file, map_location=self.device)
            self.obs_rms.mean = rms_data["mean"].to(self.device)
            self.obs_rms.var = rms_data["var"].to(self.device)
            self.obs_rms.count = rms_data["count"]

            if "reward_mean" in rms_data:
                self.reward_rms.mean = rms_data["reward_mean"].to(self.device)
                self.reward_rms.var = rms_data["reward_var"].to(self.device)
                self.reward_rms.count = rms_data["reward_count"]

        if os.path.exists(self.actor_optim_file):
            self.actor.optimiser.load_state_dict(
                T.load(self.actor_optim_file, map_location=self.device)
            )
        if os.path.exists(self.critic_optim_file):
            self.critic.optimiser.load_state_dict(
                T.load(self.critic_optim_file, map_location=self.device)
            )

    def kl_penalty_loss(self, states, actions, batch_gaes, old_probs):
        norm_states = self._normalize_obs(states)
        dist = self.actor(norm_states)
        new_probs_log = self._get_log_prob(dist, actions)
        policy_ratio = (new_probs_log - old_probs).exp()

        if self.action_type == 'discrete':
            old_dist = Categorical(logits=old_probs)
            new_dist = Categorical(logits=new_probs_log)
            self.kl_div = kl_divergence(old_dist, new_dist).mean()
        else:
            self.kl_div = T.tensor(0.0, device=self.device)

        kl_penalty_loss = -((policy_ratio * batch_gaes).mean() - self.beta * self.kl_div)
        return kl_penalty_loss, dist

    def compute_policyRatio(self, states, actions, old_probs):
        norm_states = self._normalize_obs(states)
        dist = self.actor(norm_states)
        new_probs_log = self._get_log_prob(dist, actions)
        return (new_probs_log - old_probs).exp()

    def choose_action(self, observation, deterministic=False):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        norm_state = self._normalize_obs(state)

        dist = self.actor(norm_state)
        value = self.critic(norm_state)

        if deterministic and self.action_type == "continuous":
            action = dist.mean
        else:
            action = dist.sample()

        log_prob = self._get_log_prob(dist, action)
        probs = T.squeeze(log_prob).item()

        if self.action_type == 'continuous':
            entropy = dist.entropy().sum(dim=-1).mean().item()
        else:
            entropy = dist.entropy().mean().item()

        action = T.squeeze(action).cpu().detach().numpy()
        value = T.squeeze(value).item()
        return action, probs, value, entropy

    def adjust_kl_penalty(self, kl_div):
        if kl_div < self.d_target / 1.5:
            self.beta /= 2
        elif kl_div > self.d_target * 1.5:
            self.beta *= 2

    def clippedLoss(self, states, actions, batch_gaes, old_probs):
        norm_states = self._normalize_obs(states)
        dist = self.actor(norm_states)
        new_log_probs = self._get_log_prob(dist, actions)
        policy_ratio = (new_log_probs - old_probs).exp()
        clipped_ratios = T.clamp(policy_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
        clip_advantages = clipped_ratios * batch_gaes
        actor_loss = -T.min(policy_ratio * batch_gaes, clip_advantages).mean()
        return actor_loss, dist

    def noClip(self, states, actions, batch_gaes, old_probs):
        norm_states = self._normalize_obs(states)
        dist = self.actor(norm_states)
        new_log_probs = self._get_log_prob(dist, actions)
        policy_ratio = (new_log_probs - old_probs).exp()
        actor_loss = -(policy_ratio * batch_gaes).mean()
        return actor_loss, dist

    def compute_loss(self, actor_loss, critic_loss, dist):
        if self.action_type == 'continuous':
            entropy = dist.entropy().sum(dim=-1).mean()
        else:
            entropy = dist.entropy().mean()
        entropy_bonus = self.c2 * entropy
        scaled_critic_loss = self.c1 * critic_loss
        return actor_loss + scaled_critic_loss - entropy_bonus

    def compute_gaes(self, reward_arr, values, dones_arr, truncated_arr, gamma, gae_lambda):
        advantage = np.zeros(len(reward_arr), dtype=np.float32)
        for t in range(len(reward_arr) - 1):
            discount = 1.0
            a_t = 0.0
            for k in range(t, len(reward_arr) - 1):
                end_condition = (1 - int(dones_arr[k])) * (1 - int(truncated_arr[k]))
                a_t += discount * (reward_arr[k] + gamma * values[k + 1] * end_condition - values[k])
                discount *= gamma * gae_lambda
            advantage[t] = a_t
        return advantage

    def learn(self):
        epoch_actor_losses = []
        epoch_critic_losses = []
        total_loss = T.tensor(0.0, device=self.device)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            ext_reward_arr, int_reward_arr, dones_arr, truncated_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            ext_reward_arr = np.array(ext_reward_arr, dtype=np.float32) / self.Reward_Scaler
            int_reward_arr = np.array(int_reward_arr, dtype=np.float32)

            advantages_int = 0
            advantages_ext = self.compute_gaes(
                reward_arr=ext_reward_arr,
                values=values,
                dones_arr=dones_arr,
                truncated_arr=truncated_arr,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

            if self.RND:
                advantages_int = self.compute_gaes(
                    reward_arr=int_reward_arr,
                    values=values,
                    dones_arr=dones_arr,
                    truncated_arr=truncated_arr,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda
                )

            combined_advantages = advantages_ext + advantages_int
            combined_advantages = T.tensor(combined_advantages, dtype=T.float32).to(self.actor.device)

            if self.NormaliseAdvantage:
                combined_advantages = (
                    (combined_advantages - combined_advantages.mean()) /
                    (combined_advantages.std() + 1e-8)
                )

            values = T.tensor(values, dtype=T.float32).to(self.actor.device)

            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0

            states_tensor = T.tensor(state_arr, dtype=T.float32).to(self.actor.device)
            self.obs_rms.update(states_tensor)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)

                if self.action_type == 'discrete':
                    actions = T.tensor(action_arr[batch], dtype=T.long).to(self.actor.device)
                else:
                    actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)

                batch_advantages = combined_advantages[batch]

                if self.variant == 'PPO Penalty':
                    actor_loss, dist = self.kl_penalty_loss(states, actions, batch_advantages, old_probs)
                elif self.variant == 'PPO Clip':
                    actor_loss, dist = self.clippedLoss(states, actions, batch_advantages, old_probs)
                elif self.variant == 'PPO No Clip':
                    actor_loss, dist = self.noClip(states, actions, batch_advantages, old_probs)
                else:
                    raise ValueError("Unknown PPO variant")

                norm_states = self._normalize_obs(states)
                critic_value = self.critic(norm_states).squeeze()
                returns = batch_advantages + values[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                total_loss = self.compute_loss(actor_loss, critic_loss, dist)

                self.actor.optimiser.zero_grad()
                self.critic.optimiser.zero_grad()
                total_loss.backward()
                self.actor.optimiser.step()
                self.critic.optimiser.step()

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()

            epoch_actor_losses.append(epoch_actor_loss / len(batches))
            epoch_critic_losses.append(epoch_critic_loss / len(batches))

            if self.RND:
                states_for_rnd = T.tensor(state_arr, dtype=T.float32).view(len(state_arr), -1)
                _ = self.rnd.train_rnd(states_for_rnd)

            if self.variant == 'PPO Penalty':
                self.adjust_kl_penalty(self.kldiv)

            self.actor_scheduler.step()
            self.critic_scheduler.step()

        self.memory.clear_memory()
        return total_loss, epoch_actor_losses, epoch_critic_losses