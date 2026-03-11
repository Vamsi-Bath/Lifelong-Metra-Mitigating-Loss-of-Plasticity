"""Original parallel sampler pool backend."""
# pylint: skip-file

import signal

import cloudpickle
from dowel import logger
import numpy as np

from garage.experiment import deterministic
from garage.sampler.stateful_pool import SharedGlobal
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import rollout


def _worker_init(g, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    g.worker_id = id


def initialize(max_cpus=None):
    # Windows doesn't have pthread_sigmask
    if hasattr(signal, "pthread_sigmask"):
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        except Exception:
            pass

    try:
        # existing initialize code continues here...
        # (whatever Garage currently does)
        ...
    finally:
        if hasattr(signal, "pthread_sigmask"):
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT])
            except Exception:
                pass


def _get_scoped_g(g, scope):
    if scope is None:
        return g
    if not hasattr(g, 'scopes'):
        g.scopes = dict()
    if scope not in g.scopes:
        g.scopes[scope] = SharedGlobal()
        g.scopes[scope].worker_id = g.worker_id
    return g.scopes[scope]


def _worker_populate_task(g, env, policy, scope=None):
    g = _get_scoped_g(g, scope)
    g.env = cloudpickle.loads(env)
    g.policy = cloudpickle.loads(policy)


def _worker_terminate_task(g, scope=None):
    g = _get_scoped_g(g, scope)
    if getattr(g, 'env', None):
        g.env.close()
        g.env = None
    if getattr(g, 'policy', None):
        g.policy.terminate()
        g.policy = None


def populate_task(env, policy, scope=None):
    """Set each worker's env and policy."""
    logger.log('Populating workers...')
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_populate_task,
            [(cloudpickle.dumps(env), cloudpickle.dumps(policy), scope)] *
            singleton_pool.n_parallel)
    else:
        # avoid unnecessary copying
        g = _get_scoped_g(singleton_pool.G, scope)
        g.env = env
        g.policy = policy
    logger.log('Populated')


def terminate_task(scope=None):
    """Close each worker's env and terminate each policy."""
    singleton_pool.run_each(_worker_terminate_task,
                            [(scope, )] * singleton_pool.n_parallel)


def close():
    """Close the worker pool."""
    singleton_pool.close()


def _worker_set_seed(_, seed):
    logger.log('Setting seed to %d' % seed)
    deterministic.set_seed(seed)


def set_seed(seed):
    """Set the seed in each worker."""
    singleton_pool.run_each(_worker_set_seed,
                            [(seed + i, )
                             for i in range(singleton_pool.n_parallel)])


def _worker_set_policy_params(g, params, scope=None):
    g = _get_scoped_g(g, scope)
    g.policy.set_param_values(params)


def _worker_collect_one_path(g, max_path_length, scope=None):
    g = _get_scoped_g(g, scope)
    path = rollout(g.env, g.policy, max_path_length=max_path_length)
    return path, len(path['rewards'])


def sample_paths(policy_params,
                 max_samples,
                 max_path_length=np.inf,
                 scope=None):
    """Sample paths from each worker.

    :param policy_params: parameters for the policy. This will be updated on
     each worker process
    :param max_samples: desired maximum number of samples to be collected. The
     actual number of collected samples might be greater since all trajectories
     will be rolled out either until termination or until max_path_length is
     reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(_worker_set_policy_params,
                            [(policy_params, scope)] *
                            singleton_pool.n_parallel)
    return singleton_pool.run_collect(_worker_collect_one_path,
                                      threshold=max_samples,
                                      args=(max_path_length, scope),
                                      show_prog_bar=True)
