import gymnasium as gymnasium
import gym
import numpy as np


def convert_box_space(space):
    return gym.spaces.Box(
        low=np.array(space.low, dtype=space.dtype),
        high=np.array(space.high, dtype=space.dtype),
        shape=space.shape,
        dtype=space.dtype,
    )


class GymnasiumToOldGymAPI:
    def __init__(self, env):
        self.env = env
        self.action_space = convert_box_space(env.action_space)
        self.observation_space = convert_box_space(env.observation_space)
        self.metadata = getattr(env, "metadata", {})
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.spec = getattr(env, "spec", None)

        self._last_obs = None
        self._last_xy = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        self._last_obs = obs.copy()
        self._last_xy = self._get_xy()
        return obs

    def _get_xy(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
            qpos = np.asarray(unwrapped.data.qpos).ravel()
            if qpos.shape[0] >= 2:
                return np.array([qpos[0], qpos[1]], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        obs_before = self._last_obs.copy() if self._last_obs is not None else None
        xy_before = self._last_xy.copy() if self._last_xy is not None else self._get_xy()

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)

        obs = np.asarray(obs, dtype=np.float32)
        xy_after = self._get_xy()

        info = dict(info)
        info["ori_obs"] = obs_before.copy() if obs_before is not None else obs.copy()
        info["next_ori_obs"] = obs.copy()
        info["coordinates"] = xy_before.copy()
        info["next_coordinates"] = xy_after.copy()
        info["terminated"] = terminated
        info["truncated"] = truncated

        self._last_obs = obs.copy()
        self._last_xy = xy_after.copy()

        return obs, float(reward), done, info

    def render(self, mode=None):
        return self.env.render()

    def close(self):
        return self.env.close()

    def render_trajectories(self, trajectories, colors, plot_axis, ax):
        for traj, color in zip(trajectories, colors):
            env_infos = traj.get("env_infos", {})
            if "coordinates" in env_infos:
                coords = np.asarray(env_infos["coordinates"])
                if coords.ndim == 2 and coords.shape[1] >= 2:
                    ax.plot(coords[:, 0], coords[:, 1], color=color[:3], alpha=0.8)

        if plot_axis is not None:
            ax.axis(plot_axis)
        ax.set_title("Trajectories")

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        final_coords = []
        returns = []
        lengths = []

        for traj in trajectories:
            rewards = np.asarray(traj["rewards"], dtype=np.float32)
            returns.append(float(rewards.sum()))
            lengths.append(len(rewards))

            env_infos = traj.get("env_infos", {})
            if "next_coordinates" in env_infos:
                next_coords = np.asarray(env_infos["next_coordinates"])
                if len(next_coords) > 0:
                    final_coords.append(next_coords[-1])

        metrics = {
            "EvalMeanReturn": float(np.mean(returns)) if returns else 0.0,
            "EvalStdReturn": float(np.std(returns)) if returns else 0.0,
            "EvalMeanLength": float(np.mean(lengths)) if lengths else 0.0,
        }

        if final_coords:
            final_coords = np.asarray(final_coords, dtype=np.float32)
            metrics["EvalFinalX"] = float(np.mean(final_coords[:, 0]))
            metrics["EvalFinalY"] = float(np.mean(final_coords[:, 1]))

        return metrics

    def __getattr__(self, name):
        return getattr(self.env, name)


class AntEnv(GymnasiumToOldGymAPI):
    def __init__(self, render_hw=100):
        env = gymnasium.make("Ant-v5")
        super().__init__(env)
        self.render_hw = render_hw


class HumanoidEnv(GymnasiumToOldGymAPI):
    def __init__(self, render_hw=100):
        env = gymnasium.make("Humanoid-v5")
        super().__init__(env)
        self.render_hw = render_hw