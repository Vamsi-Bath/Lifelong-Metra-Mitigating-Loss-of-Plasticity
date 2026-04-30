# import akro
# import gym
# import numpy as np


# class MazeEnv(gym.Env):
#     def __init__(self, max_path_length, action_range=0.2, render_size=256):
#         self.max_path_length = max_path_length
#         self.observation_space = akro.Box(low=-np.inf, high=np.inf, shape=(2,))
#         self.action_space = akro.Box(low=-action_range, high=action_range, shape=(2,))
#         self.render_size = render_size

#     def reset(self):
#         self._cur_step = 0
#         self._state = np.zeros(2, dtype=np.float32)
#         self._trajectory = [self._state.copy()]
#         return self._state

#     def step(self, action):
#         action = np.asarray(action, dtype=np.float32)
#         obsbefore = self._state.copy()
#         self._cur_step += 1
#         self._state = self._state + action
#         obsafter = self._state.copy()
#         self._trajectory.append(obsafter.copy())

#         done = self._cur_step >= self.max_path_length
#         reward = obsafter[0] - obsbefore[0]
#         return self._state, reward, done, {
#             'coordinates': obsbefore,
#             'next_coordinates': obsafter,
#             'ori_obs': obsbefore,
#             'next_ori_obs': obsafter,
#         }

#     def plot_trajectories(self, trajectories, colors, plot_axis, ax):
#         rmin, rmax = None, None
#         for trajectory, color in zip(trajectories, colors):
#             trajectory = np.array(trajectory)
#             ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=0.7)

#             if rmin is None or rmin > np.min(trajectory[:, :2]):
#                 rmin = np.min(trajectory[:, :2])
#             if rmax is None or rmax < np.max(trajectory[:, :2]):
#                 rmax = np.max(trajectory[:, :2])

#         if plot_axis == 'nowalls':
#             rcenter = (rmax + rmin) / 2.0
#             rmax = rcenter + (rmax - rcenter) * 1.2
#             rmin = rcenter + (rmin - rcenter) * 1.2
#             plot_axis = [rmin, rmax, rmin, rmax]

#         if plot_axis is not None:
#             ax.axis(plot_axis)
#         else:
#             ax.axis('scaled')

#     def render_trajectories(self, trajectories, colors, plot_axis, ax):
#         coordinates_trajectories = self._get_coordinates_trajectories(trajectories)
#         self.plot_trajectories(coordinates_trajectories, colors, plot_axis, ax)

#     def _get_coordinates_trajectories(self, trajectories):
#         coordinates_trajectories = []
#         for trajectory in trajectories:
#             if trajectory['env_infos']['coordinates'].ndim == 2:
#                 coordinates_trajectories.append(np.concatenate([
#                     trajectory['env_infos']['coordinates'],
#                     [trajectory['env_infos']['next_coordinates'][-1]]
#                 ]))
#             elif trajectory['env_infos']['coordinates'].ndim > 2:
#                 coordinates_trajectories.append(np.concatenate([
#                     trajectory['env_infos']['coordinates'].reshape(-1, 2),
#                     trajectory['env_infos']['next_coordinates'].reshape(-1, 2)[-1:]
#                 ]))

#         return coordinates_trajectories

#     def calc_eval_metrics(self, trajectories, is_option_trajectories):
#         return {}

#     def _world_to_pixel(self, points, radius=5.0):
#         points = np.asarray(points, dtype=np.float32)
#         scale = (self.render_size - 1) / (2.0 * radius)
#         shifted = (points + radius) * scale
#         shifted[:, 1] = (self.render_size - 1) - shifted[:, 1]
#         return np.clip(np.round(shifted).astype(np.int32), 0, self.render_size - 1)

#     def _draw_disk(self, img, center, color, radius=3):
#         cx, cy = int(center[0]), int(center[1])
#         h, w = img.shape[:2]
#         x0, x1 = max(0, cx - radius), min(w - 1, cx + radius)
#         y0, y1 = max(0, cy - radius), min(h - 1, cy + radius)

#         for y in range(y0, y1 + 1):
#             for x in range(x0, x1 + 1):
#                 if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
#                     img[y, x] = color

#     def _draw_line(self, img, p0, p1, color, thickness=1):
#         x0, y0 = int(p0[0]), int(p0[1])
#         x1, y1 = int(p1[0]), int(p1[1])

#         dx = abs(x1 - x0)
#         dy = abs(y1 - y0)
#         x, y = x0, y0
#         sx = 1 if x0 < x1 else -1
#         sy = 1 if y0 < y1 else -1

#         if dx > dy:
#             err = dx / 2.0
#             while x != x1:
#                 self._draw_disk(img, (x, y), color, radius=thickness)
#                 err -= dy
#                 if err < 0:
#                     y += sy
#                     err += dx
#                 x += sx
#         else:
#             err = dy / 2.0
#             while y != y1:
#                 self._draw_disk(img, (x, y), color, radius=thickness)
#                 err -= dx
#                 if err < 0:
#                     x += sx
#                     err += dy
#                 y += sy

#         self._draw_disk(img, (x1, y1), color, radius=thickness)

#     def render(self, mode='human'):
#         if mode != 'rgb_array':
#             return None

#         img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255

#         traj = np.asarray(self._trajectory, dtype=np.float32)
#         pix = self._world_to_pixel(traj)

#         # Draw path in blue
#         for i in range(len(pix) - 1):
#             self._draw_line(img, pix[i], pix[i + 1], color=np.array([50, 100, 220], dtype=np.uint8), thickness=1)

#         # Start point in green
#         self._draw_disk(img, pix[0], color=np.array([40, 180, 60], dtype=np.uint8), radius=4)

#         # Current point in red
#         self._draw_disk(img, pix[-1], color=np.array([220, 50, 50], dtype=np.uint8), radius=4)

#         return img




# maze_env.py

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, max_path_length: int = 200, action_range: float = 0.2, render_size: int = 256):
        super().__init__()
        self.max_path_length = max_path_length
        self.render_size = render_size

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-action_range,
            high=action_range,
            shape=(2,),
            dtype=np.float32,
        )

        self._cur_step = 0
        self._state = np.zeros(2, dtype=np.float32)
        self._trajectory = [self._state.copy()]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._cur_step = 0
        self._state = np.zeros(2, dtype=np.float32)
        self._trajectory = [self._state.copy()]

        obs = self._state.copy()
        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obsbefore = self._state.copy()
        self._cur_step += 1
        self._state = self._state + action
        obsafter = self._state.copy()
        self._trajectory.append(obsafter.copy())

        # Reward: move in +x direction
        reward = float(obsafter[0] - obsbefore[0])

        # No real terminal state in this simple env
        terminated = False
        truncated = self._cur_step >= self.max_path_length

        info = {
            "coordinates": obsbefore.copy(),
            "next_coordinates": obsafter.copy(),
            "ori_obs": obsbefore.copy(),
            "next_ori_obs": obsafter.copy(),
        }

        return obsafter, reward, terminated, truncated, info

    def _world_to_pixel(self, points, radius=5.0):
        points = np.asarray(points, dtype=np.float32)
        scale = (self.render_size - 1) / (2.0 * radius)
        shifted = (points + radius) * scale
        shifted[:, 1] = (self.render_size - 1) - shifted[:, 1]
        return np.clip(np.round(shifted).astype(np.int32), 0, self.render_size - 1)

    def _draw_disk(self, img, center, color, radius=3):
        cx, cy = int(center[0]), int(center[1])
        h, w = img.shape[:2]
        x0, x1 = max(0, cx - radius), min(w - 1, cx + radius)
        y0, y1 = max(0, cy - radius), min(h - 1, cy + radius)

        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    img[y, x] = color

    def _draw_line(self, img, p0, p1, color, thickness=1):
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                self._draw_disk(img, (x, y), color, radius=thickness)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                self._draw_disk(img, (x, y), color, radius=thickness)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        self._draw_disk(img, (x1, y1), color, radius=thickness)

    def render(self):
        img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 255

        traj = np.asarray(self._trajectory, dtype=np.float32)
        pix = self._world_to_pixel(traj)

        for i in range(len(pix) - 1):
            self._draw_line(
                img,
                pix[i],
                pix[i + 1],
                color=np.array([50, 100, 220], dtype=np.uint8),
                thickness=1,
            )

        self._draw_disk(img, pix[0], color=np.array([40, 180, 60], dtype=np.uint8), radius=4)
        self._draw_disk(img, pix[-1], color=np.array([220, 50, 50], dtype=np.uint8), radius=4)

        return img

    def close(self):
        pass