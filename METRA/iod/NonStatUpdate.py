import numpy as np
import gymnasium as gym
import tqdm
import matplotlib.pyplot as plt
from mujoco import mj_name2id, mjtObj


def discretise_state_space(xy_positions, grid_size=0.25):
    
    if not isinstance(xy_positions, np.ndarray):
        xy_positions = np.array(xy_positions)
    
    assert xy_positions.shape[1] == 2

    min_xy = xy_positions.min(axis=0)
    max_xy = xy_positions.max(axis=0)

    grid_indices = ((xy_positions - min_xy) / grid_size).astype(int)
    unique_bins = np.unique(grid_indices, axis=0)
    grid_counts = ((max_xy - min_xy) / grid_size).astype(int) + 1
    total_possible_bins = int(grid_counts[0] * grid_counts[1])

    coverage = len(unique_bins) / total_possible_bins

    return coverage, unique_bins, total_possible_bins, grid_indices


def measure_state_coverage(env_name, skill_policy, latent_dim, num_skills=10, episodes_per_skill=100, bin_size=0.5):
    skill_vectors = [np.random.randn(latent_dim).astype(np.float32) for _ in range(num_skills)]

    def discrete_Obs(x, y, bin_size=0.25):
        x_bin = int(x // bin_size)
        y_bin = int(y // bin_size)
        return (x_bin, y_bin)

    env = gym.make(env_name, exclude_current_positions_from_observation=False)
    coverage_ratios = []

    for idx, z in enumerate(tqdm(skill_vectors, desc="Measuring state coverage")):
        coverage_bins = set()
        x_vals, y_vals = [], []

        for _ in range(episodes_per_skill):
            obs, info = env.reset()
            x_pos, y_pos = info.get("x_position", 0.0), info.get("y_position", 0.0)
            x_vals.append(x_pos)
            y_vals.append(y_pos)
            coverage_bins.add(discrete_Obs(x_pos, y_pos, bin_size))

            for _ in range(1000):  # rollout length
                state_z = np.concatenate([obs, z], axis=-1)
                action = skill_policy.choose_action(state_z)
                obs, reward, done, truncated, info = env.step(action)

                x_pos, y_pos = info.get("x_position", 0.0), info.get("y_position", 0.0)
                x_vals.append(x_pos)
                y_vals.append(y_pos)
                coverage_bins.add(discrete_Obs(x_pos, y_pos, bin_size))

                if done or truncated:
                    break

        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        x_bins = int(np.ceil((x_max - x_min) / bin_size))
        y_bins = int(np.ceil((y_max - y_min) / bin_size))
        total_bins = x_bins * y_bins

        coverage_ratio = len(coverage_bins) / total_bins
        coverage_ratios.append(coverage_ratio * 100)  # percentage

        print(f"[Skill {idx}] Unique bins: {len(coverage_bins)}, Total bins: {total_bins}, Coverage: {coverage_ratio*100:.2f}%")

    env.close()

    plt.figure(figsize=(10, 5))
    plt.bar(range(num_skills), coverage_ratios)
    plt.xticks(range(num_skills), [f"z{i}" for i in range(num_skills)])
    plt.ylabel("State Coverage (%)")
    plt.xlabel("Sampled Skill Vector")
    plt.title("State Space Coverage")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return coverage_ratios

def evaluate_hierarchical_policy(env, high_level_policy, low_level_policy, 
                                 target_pos, threshold=0.05, num_episodes=50, max_steps=200):
    success_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        success = False

        high_level_policy.reset()
        low_level_policy.reset()

        while not done and step < max_steps:
            high_goal = high_level_policy.select_goal(obs)
            action = low_level_policy.select_action(obs, high_goal)
            obs, reward, done, info = env.step(action)
            step += 1

            block_pos = env.get_block_position()
            dist = np.linalg.norm(np.array(block_pos) - np.array(target_pos))
            if dist < threshold:
                success = True
                break 

        if success:
            success_count += 1

    print(f"Evaluation Success Rate: {success_count}/{num_episodes} = {success_count / num_episodes * 100:.2f}%")
    return success_count, num_episodes



class HardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        target_names=["torso"],
        base_val=12.0,
        delta=6.0,
        interval=50000,
        change_target="mass"
    ):
        super().__init__(env)
        self.env = env
        self.model = env.unwrapped.model
        self.interval = interval
        self.base_val = base_val
        self.delta = delta
        self.change_target = change_target.lower()
        self.target_names = target_names
        self.t = 0

        if self.change_target == "mass":
            self.indices = self._get_body_indices()
        elif self.change_target == "friction":
            self.indices = self._get_geom_indices()
        else:
            raise ValueError("change_target must be 'mass' or 'friction'")

    def _get_body_indices(self):
        indices = []
        for name in self.target_names:
            try:
                idx = self.model.body_name2id(name)
            except AttributeError:
                if mj_name2id:
                    idx = mj_name2id(self.model, mjtObj.mjOBJ_BODY, name)
                else:
                    raise RuntimeError("Could not resolve body name.")
            indices.append(idx)
        return indices

    def _get_geom_indices(self):
        indices = []
        for name in self.target_names:
            try:
                idx = self.model.geom_name2id(name)
            except AttributeError:
                if mj_name2id:
                    idx = mj_name2id(self.model, mjtObj.mjOBJ_GEOM, name)
                else:
                    raise RuntimeError("Could not resolve geom name.")
            indices.append(idx)
        return indices

    def _apply_update(self):
        phase = (self.t // self.interval) % 2
        val = self.base_val if phase == 0 else self.base_val - self.delta

        for idx in self.indices:
            if self.change_target == "mass":
                self.model.body_mass[idx] = val
            elif self.change_target == "friction":
                self.model.geom_friction[idx][0] = val

    def step(self, action):
        self.t += 1
        if self.t % self.interval == 0:
            self._apply_update()
        return self.env.step(action)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._apply_update()
        return obs


class SoftWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        target_names=['torso'],   
        delta=2.0,
        omega=1.0,
        update_interval=200,
        change_target="mass"  
        
    ):
        super().__init__(env)
        self.env = env
        self.model = env.unwrapped.model
        self.update_interval = update_interval
        self.delta = delta
        self.omega = omega
        self.target_names = target_names
        self.change_target = change_target.lower()
        self.t = 0

        if self.change_target == "mass":
            self.indices = self._get_body_indices()
            self.original_values = [self.model.body_mass[i] for i in self.indices]
        elif self.change_target == "friction":
            self.indices = self._get_geom_indices()
            self.original_values = [self.model.geom_friction[i][0] for i in self.indices]
        else:
            raise ValueError("change_target must be 'mass' or 'friction'")

    def _get_body_indices(self):
        indices = []
        for name in self.target_names:
            try:
                idx = self.model.body_name2id(name)
            except AttributeError:
                if mj_name2id:
                    idx = mj_name2id(self.model, mjtObj.mjOBJ_BODY, name)
                else:
                    raise RuntimeError("Cannot resolve body name.")
            indices.append(idx)
        return indices

    def _get_geom_indices(self):
        indices = []
        for name in self.target_names:
            try:
                idx = self.model.geom_name2id(name)
            except AttributeError:
                if mj_name2id:
                    idx = mj_name2id(self.model, mjtObj.mjOBJ_GEOM, name)
                else:
                    raise RuntimeError("Cannot resolve geom name.")
            indices.append(idx)
        return indices

    def _update(self):
        value = self.delta * np.sin(self.omega * self.t / self.update_interval)
        for i, idx in enumerate(self.indices):
            base = self.original_values[i]
            if self.change_target == "mass":
                self.model.body_mass[idx] = base + value
            elif self.change_target == "friction":
                self.model.geom_friction[idx][0] = base + value

    def step(self, action):
        self.t += 1
        if self.t % self.update_interval == 0:
            self._update()
        return self.env.step(action)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._update()
        return obs


def apply_nonstationary_wrapper(env, params):
    if not params.get("nonstat", False):
        return env
    nonstat_type = params.get("nonstat_type", "soft").lower()

    if nonstat_type == "soft":
        return SoftWrapper(
            env,
            target_names=params["var_bodies"],
            delta=params["delta_var"],
            omega=params["omegaVar"],
            update_interval=params["interval"],
            change_target=params["var_target"]
        )

    elif nonstat_type == "hard":
        return HardWrapper(
            env,
            target_names=params["var_bodies"],
            base_val=params["base_var"],
            delta=params["delta_var"],
            interval=params["interval"],
            change_target=params["var_target"]
        )
    else:
        raise ValueError(f"Invalid nonstat_type '{nonstat_type}'. Choose 'soft' or 'hard'.")