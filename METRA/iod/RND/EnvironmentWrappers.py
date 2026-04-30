#EnvironmentWrappers.py
import numpy as np
import gymnasium as gym
from collections import deque
from gymnasium import spaces
import cv2
import csv
import ast
from scipy.stats import truncnorm
import pandas as pd


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        # Initialize the wrapper and set the maximum number of no-op actions
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max  # Maximum number of no-op actions
        self.override_num_noops = None  # Allows overriding the number of no-ops
        self.noop_action = 0  # The action corresponding to 'no operation'
        # Ensure that the first action is 'NOOP' (no operation)
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    # Reset the environment and perform a random number of no-op actions
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Determine the number of no-op actions to perform
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        # Perform the no-op actions
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:  # If the environment is done, reset it
                obs, info = self.env.reset(**kwargs)
        return obs, info

    # Forward the step method to the wrapped environment
    def step(self, ac):
        return self.env.step(ac)


# Wrapper that automatically performs the FIRE action at the start of an episode for environments where it is required
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        # Initialize the wrapper and ensure the environment supports the FIRE action
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    # Reset the environment and perform the FIRE action twice
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(1)  # Perform the FIRE action
        if done:  # If the environment is done, reset it
            obs, info = self.env.reset(**kwargs)
        obs, _, done, _, _ = self.env.step(2)  # Perform the FIRE action again
        if done:  # If the environment is done, reset it again
            obs, info = self.env.reset(**kwargs)
        return obs, info

    # Forward the step method to the wrapped environment
    def step(self, ac):
        return self.env.step(ac)


# Wrapper that makes the environment end the episode when the agent loses a life, but not the entire game
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        # Initialize the wrapper and track the number of lives
        gym.Wrapper.__init__(self, env)
        self.lives = 0  # Tracks the number of lives left
        self.was_real_done = True  # Indicates whether the episode was really done

    # Step the environment, and check if the agent lost a life
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:  # Agent lost a life but the game is not over
            done = True  # End the episode
        self.lives = lives  # Update the number of lives
        return obs, reward, done, terminated, info

    # Reset the environment if the episode was really done; otherwise, take a no-op action
    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()  # Update the number of lives
        return obs, info

# Wrapper that skips frames in the environment and returns the maximum frame over the skipped frames
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        # Initialize the wrapper and set the number of frames to skip
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)  # Buffer for the last two frames
        self._skip = skip  # Number of frames to skip

    # Forward the reset method to the wrapped environment
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    # Step the environment, skipping frames and returning the max frame and total reward
    def step(self, action):
        total_reward = 0.0
        done = None
        terminated = None
        for i in range(self._skip):
            obs, reward, done, terminated, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs  # Store the second-to-last frame
            if i == self._skip - 1: self._obs_buffer[1] = obs  # Store the last frame
            total_reward += reward  # Accumulate the reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)  # Take the maximum over the buffered frames
        return max_frame, total_reward, done, terminated, info


# Wrapper that clips the val to be in the range [-1, 0, 1]
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        # Initialize the wrapper
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        # Initialize the wrapper and set the target frame dimensions
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84  # Target width
        self.height = 80  # Target height
        # Update the observation space to reflect the grayscale and resized frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    # Convert the observation to grayscale and resize it
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)  # Resize the frame
        return frame[:, :, None]  # Add a channel dimension


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        # Initialize the wrapper and set up the frame stack
        gym.Wrapper.__init__(self, env)
        self.k = k  # Number of frames to stack
        self.frames = deque([], maxlen=k)  # Deque to store the last k frames
        shp = env.observation_space.shape
        # Update the observation space to reflect the stacked frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0] * k, shp[1], shp[2]), dtype=np.uint8)

    # Reset the environment and fill the frame stack with the initial observation
    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.k):  # Fill the deque with the initial observation
            self.frames.append(ob)
        return self._get_ob(), info

    # Step the environment and update the frame stack
    def step(self, action):
        ob, reward, done, terminated, info = self.env.step(action)
        self.frames.append(ob)  # Add the new observation to the frame stack
        return self._get_ob(), reward, done, terminated, info

    # Return the stacked frames as a single observation
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        # Initialize the wrapper
        gym.ObservationWrapper.__init__(self, env)

    # Scale the observation to [0, 1]
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        # Store the frames
        self._frames = frames


    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype) 
        return out


    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return self._frames[i]
    
def LogReturnAndEntropy(entropy ,val, csv_file_path, rnd):
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        first_row = next(reader)
        range_value = first_row.get('entrList')
        if range_value != None:
            loggedlist = [x*(1e3) for x in ast.literal_eval(range_value)]
            clip, rng, l, u = map(float, loggedlist)
            lowerEntropyBonus, upperEntropyBonus = (l - clip) / rng, (u - clip) / rng
            val[0] = val[0]*entropy + truncnorm.rvs(lowerEntropyBonus, upperEntropyBonus, loc=clip, scale=rng)
            if not rnd and loggedlist[0] < 0:
                val[0] = np.ceil(val[0])
        

class PyTorchFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(PyTorchFrame, self).__init__(env)
        shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape[-1], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.rollaxis(observation, 2)
    
def flatten_observation(obs_dict):
    return np.concatenate([
        np.ravel(v) for v in obs_dict.values()
        if isinstance(v, (np.ndarray, list, tuple))
    ]).astype(np.float32)
