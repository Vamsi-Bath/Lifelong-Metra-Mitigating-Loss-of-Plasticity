# envs/task_id_wrapper.py
import numpy as np

class TaskIDWrapper:
    """
    Wrap an env so that:
      - env_info always contains 'task_id'
      - optional: env_info contains 'task_params' (e.g. friction snapshot)
      - exposes switch_task() to change task/dynamics and increment task_id
    """
    def __init__(self, env, task_switch_fn=None, task_params_fn=None):
        """
        task_switch_fn(env, task_id) -> None
            Should mutate env dynamics (e.g. friction) for the new task_id.

        task_params_fn(env) -> np.ndarray or dict
            Optional: returns current task parameters for logging/debugging
            (e.g. friction vector). Included in env_info as 'task_params'.
        """
        self.env = env
        self.task_id = 0
        self.task_switch_fn = task_switch_fn
        self.task_params_fn = task_params_fn

    def switch_task(self, task_id=None):
        """Increment task_id (or set it) and apply task_switch_fn."""
        if task_id is None:
            self.task_id += 1
        else:
            self.task_id = int(task_id)

        if self.task_switch_fn is not None:
            self.task_switch_fn(self.env, self.task_id)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action, **kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        if info is None:
            info = {}

        info['task_id'] = int(self.task_id)

        if self.task_params_fn is not None:
            params = self.task_params_fn(self.env)
            # Make it numpy-friendly for logging/storage
            if isinstance(params, dict):
                info['task_params'] = {k: np.asarray(v) for k, v in params.items()}
            else:
                info['task_params'] = np.asarray(params)

        return obs, rew, done, info

    def __getattr__(self, name):
        # passthrough to underlying env
        return getattr(self.env, name)