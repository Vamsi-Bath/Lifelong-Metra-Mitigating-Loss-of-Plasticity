#Memory.py
import numpy as np

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.extrinsic_rewards = []
        self.intrinsic_rewards = []
        self.dones = []
        self.truncateds = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals,
                     extrinsic_reward, intrinsic_reward,
                     done, truncated):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.extrinsic_rewards.append(extrinsic_reward)
        self.intrinsic_rewards.append(intrinsic_reward)
        self.dones.append(done)
        self.truncateds.append(truncated)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.extrinsic_rewards),
            np.array(self.intrinsic_rewards),
            np.array(self.dones),
            np.array(self.truncateds),
            batches
        )

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.extrinsic_rewards = []
        self.intrinsic_rewards = []
        self.dones = []
        self.truncateds = []
