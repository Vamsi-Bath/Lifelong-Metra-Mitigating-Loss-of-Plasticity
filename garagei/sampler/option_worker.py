import functools
import numpy as np
from garage.experiment import deterministic

from garage.sampler import DefaultWorker

from iod.utils import get_np_concat_obs


class OptionWorker(DefaultWorker):
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number,
            sampler_key,
    ):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._sampler_key = sampler_key
        self._max_path_length_override = None
        self._cur_extras = None
        self._cur_extra_idx = None
        self._cur_extra_keys = set()
        self._render = False
        self._deterministic_policy = None

    def update_env(self, env_update):
        """Update/construct the environment for this worker (Windows-safe).

        Supports:
        - dict: attribute updates on current env
        - callable: env factory -> constructs env
        - env instance: directly set env
        """
        if env_update is None:
            return

        if isinstance(env_update, dict):
            for k, v in env_update.items():
                setattr(self.env, k, v)
            return

        if callable(env_update):
            self.env = env_update()
            return

        self.env = env_update

    def worker_init(self):
        """Initialize a worker."""
        if self._seed is not None:
            deterministic.set_seed(self._seed + self._worker_number * 10000)

    def update_worker(self, worker_update):
        if worker_update is not None:
            if isinstance(worker_update, dict):
                for k, v in worker_update.items():
                    setattr(self, k, v)
                    if k == '_cur_extras':
                        if v is None:
                            self._cur_extra_keys = set()
                        else:
                            if len(self._cur_extras) > 0:
                                self._cur_extra_keys = set(self._cur_extras[0].keys())
                            else:
                                self._cur_extra_keys = None

            else:
                raise TypeError('Unknown worker update type.')

    def get_attrs(self, keys):
        attr_dict = {}
        for key in keys:
            attr_dict[key] = functools.reduce(getattr, [self] + key.split('.'))
        return attr_dict

    def start_rollout(self):
        """Begin a new rollout."""
        if 'goal' in self._cur_extra_keys:
            goal = self._cur_extras[self._cur_extra_idx]['goal']
            reset_kwargs = {'goal': goal}
        else:
            reset_kwargs = {}

        env = self.env
        while hasattr(env, 'env'):
            env = getattr(env, 'env')

        self._path_length = 0
        self._prev_obs = self.env.reset(**reset_kwargs)
        self._prev_extra = None

        self.agent.reset()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination of due to reaching `max_path_length`.

        """
        cur_max_path_length = self._max_path_length if self._max_path_length_override is None else self._max_path_length_override
        if self._path_length < cur_max_path_length:
            # ===================== [LIFELONG ADDITION] =====================
            # Support passing BOTH 'option' and 'context' extras.
            # - If only 'option' exists: agent_input = concat(obs, option)
            # - If only 'context' exists: agent_input = concat(obs, context)
            # - If both exist: agent_input = concat(obs, option, context)
            has_option = ('option' in self._cur_extra_keys)
            has_context = ('context' in self._cur_extra_keys)

            if not has_option and not has_context:
                agent_input = self._prev_obs
            else:
                # Fetch option (could be fixed per rollout or time-varying list)
                if has_option:
                    opt_val = self._cur_extras[self._cur_extra_idx]['option']
                    if isinstance(opt_val, list):
                        cur_option = opt_val[self._path_length]
                        if cur_option is None:
                            cur_option = self._prev_extra
                            opt_val[self._path_length] = cur_option
                    else:
                        cur_option = opt_val
                else:
                    cur_option = None

                # Fetch context (could be fixed per rollout or time-varying list)
                if has_context:
                    ctx_val = self._cur_extras[self._cur_extra_idx]['context']
                    if isinstance(ctx_val, list):
                        cur_context = ctx_val[self._path_length]
                    else:
                        cur_context = ctx_val
                else:
                    cur_context = None

                # Build agent_input by concatenating available extras
                if cur_option is not None and cur_context is not None:
                    agent_input = get_np_concat_obs(self._prev_obs, cur_option)
                    agent_input = get_np_concat_obs(agent_input, cur_context)
                elif cur_option is not None:
                    agent_input = get_np_concat_obs(self._prev_obs, cur_option)
                else:
                    agent_input = get_np_concat_obs(self._prev_obs, cur_context)

                # Keep old behavior of caching the "prev extra" for time-varying option lists
                self._prev_extra = cur_option
            # ===============================================================

            if self._deterministic_policy is not None:
                self.agent._force_use_mode_actions = self._deterministic_policy

            a, agent_info = self.agent.get_action(agent_input)

            next_o, r, d, env_info = self.env.step(a)

            # If rendering is requested, capture frame and store it
            if self._render:
                frame = None
                try:
                    frame = self.env.render(mode='rgb_array')
                except TypeError:
                    try:
                        frame = self.env.render()
                    except Exception:
                        frame = None
                except Exception:
                    frame = None

                if frame is not None:
                    env_info['render'] = frame

            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)

            for k, v in agent_info.items():
                self._agent_infos[k].append(v)

            # IMPORTANT: this already records ALL extras (including 'context') into agent_infos.
            for k in self._cur_extra_keys:
                if isinstance(self._cur_extras[self._cur_extra_idx][k], list):
                    self._agent_infos[k].append(self._cur_extras[self._cur_extra_idx][k][self._path_length])
                else:
                    self._agent_infos[k].append(self._cur_extras[self._cur_extra_idx][k])

            for k, v in env_info.items():
                self._env_infos[k].append(v)

            self._path_length += 1
            self._terminals.append(d)
            if not d:
                self._prev_obs = next_o
                return False

        self._terminals[-1] = True
        self._lengths.append(self._path_length)
        self._last_observations.append(self._prev_obs)
        return True

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        if self._cur_extras is not None:
            self._cur_extra_idx += 1
        self.start_rollout()
        while not self.step_rollout():
            pass
        return self.collect_rollout()
    
